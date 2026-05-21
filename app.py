"""3GPP Agent - Dual-Skill Chainlit Frontend.

Two skills:
  - TDoc: search meeting documents (3gpp_tdocs ChromaDB)
  - Spec: query technical specifications (3gpp_specs ChromaDB)

Interaction flow:
  1. Welcome → user selects a skill (mandatory)
  2. on_action → initialize skill-specific retriever + settings
  3. on_message → route to appropriate retriever → display results
  4. Switch skill available at any time
"""

import json
import os
import sys
import uuid

# Ensure project root is on sys.path
try:
    _ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _ROOT = os.getcwd()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Load .env before other imports
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

# Fix protobuf compatibility for chromadb
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import chainlit as cl
from chainlit.input_widget import Select

from agent.router import SkillRouter, Skill
from memory import MemoryManager
from utils.llm_utils import llm_available

# ── Skill-specific constants ───────────────────────────────────────────────────

WORKING_GROUPS = ["", "R1", "R2", "R3", "R4", "SA1", "SA2", "SA3", "SA5", "CT1", "CT3", "CT4"]
TOP_K_OPTIONS = ["5", "10", "20", "30"]

# Load TDoc company list
def _load_companies():
    companies = set()
    try:
        with open("data/tdocs/processed/metadata.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for co in rec.get("companies", []):
                    if co != "Unknown":
                        companies.add(co)
    except FileNotFoundError:
        pass
    return sorted(companies)

COMPANY_LIST = _load_companies()
COMPANY_OPTIONS = [""] + COMPANY_LIST

# Spec list
SPEC_OPTIONS = [""] + [
    "38.211", "38.212", "38.213", "38.214",
    "38.300", "38.321", "38.322", "38.331",
]

# ── LLM prompts ──────────────────────────────────────────────────────────────

TDOC_ANALYSIS_PROMPT = """你是一名资深 3GPP 标准化专家。请基于以下提供的 3GPP Tdoc 分块内容，完成以下任务：

1. **按提交公司分组**，总结每家公司在该技术方向的核心观点和提出的技术方案
2. **指出不同公司方案的主要分歧点**
3. 所有总结必须标注信息来源（Tdoc 编号），禁止编造内容

请使用中文回答，输出格式为 Markdown。

提供的 Tdoc 分块内容：
{context}
"""

SPEC_ANALYSIS_PROMPT = """你是一名资深 3GPP 协议专家。请基于以下提供的 3GPP 规范分块内容，完成以下任务：

1. **解释技术概念**：清晰说明该规范章节涉及的技术机制和过程
2. **总结关键内容**：提取该章节的核心定义、流程步骤和重要参数
3. **指出跨章节引用**：标注对其他规范章节的引用（如 TS 38.213 Section 9）
4. 所有结论必须标注来源（Spec 编号和章节号），禁止编造内容

请使用中文回答，输出格式为 Markdown。

提供的规范分块内容：
{context}
"""


# ── LLM streaming ────────────────────────────────────────────────────────────

async def _stream_analysis(context: str, query: str, skill: Skill):
    from llm.config import create_llm_provider

    prompt_template = TDOC_ANALYSIS_PROMPT if skill == Skill.TDOC else SPEC_ANALYSIS_PROMPT
    prompt = prompt_template.format(context=context[:12000])

    # Build memory-injected system prompt
    system_parts = ["你是一名资深 3GPP 通信标准化专家。请用中文回答。"]
    mem: MemoryManager = cl.user_session.get("memory")
    if mem:
        memory_ctx = mem.build_context(skill.value)
        if memory_ctx:
            system_parts.append(memory_ctx)

    provider = create_llm_provider()

    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    async for token in provider.stream_chat(
        messages=[{"role": "user", "content": prompt}],
        system="\n\n".join(system_parts),
    ):
        full_response += token
        await msg.stream_token(token)

    await msg.update()

    if mem:
        mem.record_analysis(full_response, skill.value)

    return full_response


# ── Result formatters ────────────────────────────────────────────────────────

def _format_tdoc_table(results: list[dict], query: str) -> str:
    if not results:
        return "未找到相关文档。"
    lines = [
        f"## TDoc 检索结果：\"{query}\"",
        f"共找到 **{len(results)}** 条相关结果\n",
        "| # | TDoc | 公司 | 标题 | 相关度 |",
        "|---|------|------|------|--------|",
    ]
    seen = set()
    idx = 0
    for r in results:
        tdoc = r.get("tdoc_number", "")
        if tdoc in seen or not tdoc:
            continue
        seen.add(tdoc)
        idx += 1
        companies = r.get("companies", "")
        if isinstance(companies, list):
            companies = ", ".join(companies)
        title = (r.get("title") or r.get("heading") or "")[:60]
        score = r.get("rerank_score") or r.get("dense_score", 0)
        lines.append(f"| {idx} | **{tdoc}** | {companies} | {title} | {score:.3f} |")
    return "\n".join(lines)


def _format_spec_table(results: list[dict], query: str) -> str:
    if not results:
        return "未找到相关规范内容。"
    lines = [
        f"## Spec 检索结果：\"{query}\"",
        f"共找到 **{len(results)}** 条相关结果\n",
        "| # | 规范章节 | 标题 | 相关度 |",
        "|---|---------|------|--------|",
    ]
    seen = set()
    idx = 0
    for r in results:
        spec = r.get("spec_number", "")
        sec = r.get("section_path", "")
        if not spec or not sec:
            continue
        key = f"{spec}:{sec}"
        if key in seen:
            continue
        seen.add(key)
        idx += 1
        title = (r.get("section_title") or r.get("heading") or "")[:60]
        score = r.get("rerank_score") or r.get("dense_score", 0)
        lines.append(f"| {idx} | **TS {spec} §{sec}** | {title} | {score:.3f} |")
    return "\n".join(lines)


def _format_tdoc_details(results: list[dict]) -> str:
    lines = ["\n### 详细内容\n"]
    seen = set()
    count = 0
    for r in results:
        tdoc = r.get("tdoc_number", "")
        if tdoc in seen or not tdoc:
            continue
        seen.add(tdoc)
        if count >= 5:
            break
        count += 1
        chunk_text = r.get("chunk_text", "")[:500]
        parent = r.get("parent_text", "")[:300]
        companies = r.get("companies", "")
        if isinstance(companies, list):
            companies = ", ".join(companies)
        title = r.get("title") or r.get("heading", "")
        lines.append(f"#### {tdoc} — {title[:80]}")
        lines.append(f"- **公司**: {companies}")
        lines.append(f"- **摘要**: {chunk_text}...")
        if parent:
            lines.append(f"- **上下文**: {parent}...")
        lines.append("")
    return "\n".join(lines)


def _format_spec_details(results: list[dict]) -> str:
    lines = ["\n### 详细内容\n"]
    seen = set()
    count = 0
    for r in results:
        spec = r.get("spec_number", "")
        sec = r.get("section_path", "")
        if not spec or not sec:
            continue
        key = f"{spec}:{sec}"
        if key in seen:
            continue
        seen.add(key)
        if count >= 5:
            break
        count += 1
        chunk_text = r.get("chunk_text", "")[:500]
        parent = r.get("parent_text", "")[:300]
        heading = r.get("heading", "")
        lines.append(f"#### TS {spec} §{sec} — {heading[:80]}")
        lines.append(f"- **摘要**: {chunk_text}...")
        if parent:
            lines.append(f"- **章节上下文**: {parent[:300]}...")
        lines.append("")
    return "\n".join(lines)


def _build_context(results: list[dict], skill: Skill) -> str:
    seen = set()
    parts = []
    for r in results[:15]:
        if skill == Skill.TDOC:
            tdoc = r.get("tdoc_number", "")
            if not tdoc or tdoc in seen:
                continue
            seen.add(tdoc)
            companies = r.get("companies", "")
            if isinstance(companies, list):
                companies = ", ".join(companies)
            title = r.get("title") or r.get("heading", "")
            text = r.get("chunk_text", "")[:800]
            parts.append(f"[{tdoc}] 公司: {companies}\n标题: {title}\n内容: {text}")
        else:
            spec = r.get("spec_number", "")
            sec = r.get("section_path", "")
            if not spec or not sec:
                continue
            key = f"{spec}:{sec}"
            if key in seen:
                continue
            seen.add(key)
            heading = r.get("heading", "")
            text = r.get("chunk_text", "")[:800]
            parts.append(f"[TS {spec} §{sec}] {heading}\n{text}")
    return "\n\n---\n\n".join(parts)


# ── Chainlit event handlers ──────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Show welcome screen with skill selection."""
    cl.user_session.set("skill", None)
    cl.user_session.set("router", SkillRouter())

    # Initialize memory (non-fatal)
    session_id = str(uuid.uuid4())
    try:
        mem = MemoryManager(session_id=session_id, user_id="default")
        cl.user_session.set("memory", mem)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Memory init failed: {e}")

    welcome = """# 3GPP 双 Skill 智能检索助手

欢迎使用 3GPP 检索分析系统！请先选择要使用的 **Skill**：
"""
    actions = [
        cl.Action(name="select_tdoc", payload={"skill": "tdoc"}, label="TDoc 会议文档检索"),
        cl.Action(name="select_spec", payload={"skill": "spec"}, label="3GPP 规范查询"),
    ]
    await cl.Message(content=welcome, actions=actions).send()


@cl.action_callback("select_tdoc")
async def on_select_tdoc(action: cl.Action):
    """Initialize TDoc skill."""
    skill = Skill.TDOC
    cl.user_session.set("skill", skill)

    mem: MemoryManager = cl.user_session.get("memory")
    prefs = mem.preferences if mem else None

    await cl.ChatSettings([
        Select(id="working_group", label="工作组", values=WORKING_GROUPS,
               initial_value=prefs.working_groups[0] if prefs and prefs.working_groups else ""),
        Select(id="company", label="提交公司", values=COMPANY_OPTIONS,
               initial_value=prefs.companies[0] if prefs and prefs.companies else ""),
        Select(id="top_k", label="结果数量", values=TOP_K_OPTIONS,
               initial_value=str(prefs.top_k) if prefs else "20"),
    ]).send()

    examples = """## TDoc 会议文档检索（当前 Skill: **TDoc**）

当前数据库：**TSGR2_134** 会议 947 个 Tdoc 文档

**使用方式：**
- 输入技术关键词进行检索
- 点击右上角 ⚙️ 设置工作组、公司过滤条件
- 检索后可点击 **深度分析** 获取 AI 结构化报告

**示例查询：**
- `beam management FR2`
- `NTN satellite NR`
- `QoS 6G framework`
- `Sidelink V2X mode`

---
*如需切换，请点击底部的 [切换 Skill] 按钮*
"""
    await cl.Message(content=examples).send()


@cl.action_callback("select_spec")
async def on_select_spec(action: cl.Action):
    """Initialize Spec skill."""
    skill = Skill.SPEC
    cl.user_session.set("skill", skill)

    mem: MemoryManager = cl.user_session.get("memory")
    prefs = mem.preferences if mem else None

    await cl.ChatSettings([
        Select(id="spec_filter", label="规范筛选", values=SPEC_OPTIONS, initial_value=""),
        Select(id="top_k", label="结果数量", values=TOP_K_OPTIONS,
               initial_value=str(prefs.top_k) if prefs else "20"),
    ]).send()

    examples = """## 3GPP 规范查询（当前 Skill: **Spec**）

当前数据库：Rel-19 **8 个** NR 规范（38.211-214, 38.300, 38.321-323, 38.331）

**使用方式：**
- 输入技术名词查询协议原文（如 `random access procedure`）
- 可直接引用章节号（如 `38.321 5.1`、`TS 38.321 Section 9`）
- 检索后可点击 **深度分析** 获取协议内容整理

**示例查询：**
- `random access procedure` → 定位到 TS 38.321 §5.1
- `PDCCH monitoring` → 控制信道监听相关章节
- `38.321 5.1` → 直接精确定位 MAC 层随机接入
- `carrier aggregation` → 跨规范检索载波聚合

---
*如需切换，请点击底部的 [切换 Skill] 按钮*
"""
    await cl.Message(content=examples).send()


@cl.action_callback("switch_skill")
async def on_switch_skill(action: cl.Action):
    """Switch back to skill selection."""
    cl.user_session.set("skill", None)
    welcome = """# 3GPP 双 Skill 智能检索助手

请选择要使用的 **Skill**：
"""
    actions = [
        cl.Action(name="select_tdoc", payload={"skill": "tdoc"}, label="TDoc 会议文档检索"),
        cl.Action(name="select_spec", payload={"skill": "spec"}, label="3GPP 规范查询"),
    ]
    await cl.Message(content=welcome, actions=actions).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("settings", settings)
    try:
        mem: MemoryManager = cl.user_session.get("memory")
        if mem:
            skill = cl.user_session.get("skill")
            mem.save_preferences_from_settings(settings, skill.value if skill else "tdoc")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to save preferences: {e}")


@cl.on_message
async def on_message(message: cl.Message):
    """Route message to the appropriate retriever based on active skill."""
    skill = cl.user_session.get("skill")

    # ── Skill not selected: show selection prompt ──────────────────────────
    if skill is None:
        actions = [
            cl.Action(name="select_tdoc", payload={"skill": "tdoc"}, label="TDoc 会议文档检索"),
            cl.Action(name="select_spec", payload={"skill": "spec"}, label="3GPP 规范查询"),
        ]
        await cl.Message(
            content="**请先选择 Skill，再输入查询！**\n点击上方按钮选择检索模式。",
            actions=actions,
        ).send()
        return

    query = message.content.strip()
    if not query:
        return

    # ── Handle analysis command ────────────────────────────────────────────
    if query.lower() in ["分析", "analyze", "深度分析"]:
        await _handle_analysis()
        return

    # ── Get settings ────────────────────────────────────────────────────────
    settings = cl.user_session.get("settings", {})
    top_k = int(settings.get("top_k", "20"))

    # ── Get or init router ────────────────────────────────────────────────
    router: SkillRouter = cl.user_session.get("router")
    if router is None:
        router = SkillRouter()
        cl.user_session.set("router", router)

    # ── Route retrieval ───────────────────────────────────────────────────
    async with cl.Step(name="检索", type="run") as step:
        step.output = f"[{skill.value.upper()}] 查询: `{query}`"

        if skill == Skill.TDOC:
            working_group = settings.get("working_group", "") or None
            company = settings.get("company", "") or None
            companies_filter = [company] if company else None

            results = router.retrieve(
                skill=Skill.TDOC,
                query=query,
                top_k=top_k,
                working_group=working_group,
                companies=companies_filter,
                use_reranker=False,
                use_glossary=True,
            )
        else:
            spec_filter = settings.get("spec_filter", "") or None
            results = router.retrieve(
                skill=Skill.SPEC,
                query=query,
                top_k=top_k,
                spec_filter=spec_filter,
                use_reranker=True,
                use_glossary=True,
            )

        # Extract stage log
        stage_log = None
        if results and isinstance(results[-1], dict) and "_stage_log" in results[-1]:
            stage_log = results.pop().get("_stage_log", [])

        if stage_log:
            step.output = "\n".join(f"  • {e}" for e in stage_log)
        else:
            step.output = f"返回 {len(results)} 条结果"

    # ── Cache for analysis ─────────────────────────────────────────────────
    cl.user_session.set("results_cache", results)
    cl.user_session.set("last_query", query)

    # ── Record in memory ──────────────────────────────────────────────────
    mem: MemoryManager = cl.user_session.get("memory")
    if mem:
        mem.record_query(
            query=query, skill=skill.value, results=results,
            top_k=top_k,
            filters={
                "working_group": settings.get("working_group", ""),
                "company": settings.get("company", ""),
                "spec_filter": settings.get("spec_filter", ""),
            },
        )

    # ── Format results ────────────────────────────────────────────────────
    if skill == Skill.TDOC:
        table_md = _format_tdoc_table(results, query)
        details_md = _format_tdoc_details(results)
    else:
        table_md = _format_spec_table(results, query)
        details_md = _format_spec_details(results)

    # ── Action buttons ───────────────────────────────────────────────────
    actions = []
    if results and llm_available():
        actions.append(cl.Action(name="analyze", payload={"skill": skill.value}, label="深度分析"))
    actions.append(cl.Action(name="bookmark", payload={}, label="收藏结果"))
    actions.append(cl.Action(name="switch_skill", payload={}, label="切换 Skill"))

    await cl.Message(content=table_md + details_md, actions=actions).send()

    # ── LLM hint ─────────────────────────────────────────────────────────
    if not llm_available():
        await cl.Message(
            content="> 提示：Ollama 未运行且未配置 Claude API Key，深度分析功能不可用。"
        ).send()


@cl.action_callback("analyze")
async def on_analyze(action: cl.Action):
    await _handle_analysis()


@cl.action_callback("bookmark")
async def on_bookmark(action: cl.Action):
    mem: MemoryManager = cl.user_session.get("memory")
    results = cl.user_session.get("results_cache", [])
    skill = cl.user_session.get("skill", Skill.TDOC)

    if not mem or not results:
        await cl.Message(content="没有可收藏的结果。").send()
        return

    saved = []
    for r in results[:5]:
        if skill == Skill.TDOC:
            doc_id = r.get("tdoc_number", "")
            title = r.get("title") or r.get("heading", "")
        else:
            doc_id = f"TS {r.get('spec_number', '')}:{r.get('section_path', '')}"
            title = r.get("section_title") or r.get("heading", "")

        if doc_id and mem.add_bookmark(doc_id, skill.value, title=title):
            saved.append(doc_id)

    if saved:
        await cl.Message(content=f"已收藏 **{len(saved)}** 条结果：{', '.join(saved)}").send()
    else:
        await cl.Message(content="这些结果已经在收藏夹中了。").send()


async def _handle_analysis():
    results = cl.user_session.get("results_cache", [])
    query = cl.user_session.get("last_query", "")
    skill = cl.user_session.get("skill", Skill.TDOC)

    if not results:
        await cl.Message(content="请先进行检索查询。").send()
        return

    if not llm_available():
        await cl.Message(content="无可用的 LLM 服务。请启动 Ollama 或配置 ANTHROPIC_API_KEY。").send()
        return

    context = _build_context(results, skill)

    async with cl.Step(name="AI 分析", type="llm") as step:
        step.output = f"基于 {len(results)} 条结果进行 {'TDoc' if skill == Skill.TDOC else 'Spec'} 分析..."

    await cl.Message(content=f"### AI 深度分析\n").send()

    try:
        await _stream_analysis(context, query, skill)
    except Exception as e:
        await cl.Message(content=f"LLM 分析出错: {e}").send()
