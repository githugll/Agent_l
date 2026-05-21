# 3GPP Agent - 双 Skill 智能检索分析系统

基于 RAG 技术的 3GPP 垂直领域检索分析 Agent，支持两个独立 Skill：

| Skill | 功能 | 数据源 |
|-------|------|--------|
| **TDoc 会议文档检索** | 检索 RAN2 会议提案，按公司/工作组过滤，LLM 分析公司立场 | TSGR2_134 会议 947 个 Tdoc |
| **3GPP 规范查询** | 按技术名词查询协议原文，精确定位到 Spec 章节并整理结论 | TS 38.211-215/321-323/300/331 共 8 个 NR 规范 |

用户在输入查询前必须选择至少一个 Skill，系统根据选择路由到对应的检索管道。

---

## 系统架构

```
                            ┌─────────────────────────────────────────────────────────┐
                            │                    Chainlit Chat UI                      │
                            │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
                            │  │ Skill 选择   │→│ 查询输入      │→│ 结果展示      │   │
                            │  │ (TDoc/Spec)  │  │ + 过滤条件    │  │ + LLM 分析   │   │
                            │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
                            └─────────┼─────────────────┼─────────────────┼───────────┘
                                      │                 │                 │
                                      ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SkillRouter (agent/router.py)                           │
│                    根据 skill 选择惰性初始化并路由到对应 Retriever                     │
└──────────────┬──────────────────────────────────┬───────────────────────────────────┘
               │                                  │
        skill="tdoc"                       skill="spec"
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐         ┌──────────────────────────┐
│   HybridRetriever        │         │   SpecRetriever          │
│   (retriever/)           │         │   (retriever/)           │
│                          │         │                          │
│  ┌───────┐ ┌──────┐     │         │  ┌───────┐ ┌──────┐     │
│  │Dense  │ │ BM25 │     │         │  │Dense  │ │ BM25 │     │
│  │Search │ │      │     │         │  │Search │ │      │     │
│  └───┬───┘ └──┬───┘     │         │  └───┬───┘ └──┬───┘     │
│      └───┬────┘         │         │      └───┬────┘         │
│          ▼              │         │          ▼              │
│  ┌──────────────┐       │         │  ┌──────────────┐       │
│  │  BGE Reranker│       │         │  │  BGE Reranker│       │
│  └──────┬───────┘       │         │  └──────┬───────┘       │
│         │               │         │         │               │
│  术语扩展/消歧           │         │  术语扩展 +              │
│  元数据过滤              │         │  章节引用检测 +           │
│  (公司/工作组/会议)       │         │  Spec/Section 定向检索   │
└─────────┬───────────────┘         └─────────┬───────────────┘
          │                                   │
          ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────────┐
│  data/tdocs/        │           │  data/specs/            │
│  chroma_db/         │           │  chroma_db/             │
│  collection:        │           │  collection:            │
│    3gpp_tdocs       │           │    3gpp_specs           │
│  22,504 chunks      │           │  8 specs, ~5,000 chunks │
│  (4,677 parent +    │           │  (section-level         │
│   17,827 child)     │           │   parent-child)         │
└─────────────────────┘           └─────────────────────────┘
```

---

## 数据管道

### TDoc 管道（现有）

```
3GPP FTP 会议目录 → tdoc_crawler → parser(.docx) → metadata → chunker → indexer → data/tdocs/chroma_db
```

### Spec 管道（新增）

```
3GPP FTP Specs 目录 → spec_crawler(ZIP) → spec_parser(PDF) → spec_chunker(章节层级) → spec_indexer → data/specs/chroma_db
```

```
                     ┌─────────────────────────────────────────────────────────────┐
                     │                 build_spec_db.py (一键构建)                  │
                     │                                                             │
                     │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐│
3GPP FTP             │  │spec_     │   │spec_     │   │spec_     │   │spec_     ││
(Specs/2025-12/      │  │crawler   │──▶│parser    │──▶│chunker   │──▶│indexer   ││
 Rel-19/38_series)   │  │(ZIP→PDF) │   │(PDF→TXT) │   │(分片)     │   │(ChromaDB)││
                     │  └──────────┘   └──────────┘   └──────────┘   └──────────┘│
                     └─────────────────────────────────────────────────────────────┘
```

**Spec 检索增强机制：**
```
用户查询: "random access procedure"
    │
    ├─ 术语扩展: 3gpp_glossary.json → _spec_sections 映射
    │  "random access" → {"spec": "38.321", "section": "5.1"}
    │  扩展后查询: "random access procedure TS 38.321 Section 5.1"
    │
    ├─ 章节引用检测: 正则匹配 "38.321 5.1" → Chroma metadata filter
    │  where={"spec_number": "38.321", "section_path": {"$regex": "^5\\.1"}}
    │
    └─ 定向检索: dense + BM25 → rerank → 返回 38.321 §5.1 相关 chunks
       结果格式: "TS 38.321 Section 5.1 - Random Access procedure"
```

---

## 项目结构

```
3gpp-agent/
├── app.py                           # Chainlit 前端（双 Skill UI + 路由）
├── build_spec_db.py                 # Spec 数据管道一键构建脚本
├── crawl.py                         # TDoc 交互式爬虫脚本
├── requirements.txt
│
├── agent/                           # Agent 路由层
│   ├── __init__.py
│   └── router.py                        # SkillRouter：Skill 选择 + 检索路由
│
├── crawler/                         # 爬取模块（tdocs/specs 并列）
│   ├── __init__.py
│   ├── tdocs/                           # TDoc 爬虫
│   │   ├── __init__.py
│   │   ├── config.py                      # TDoc 爬虫配置
│   │   ├── tdoc_crawler.py                # 同步爬虫
│   │   └── async_crawler.py               # 异步爬虫
│   └── specs/                           # Spec 爬虫
│       ├── __init__.py
│       ├── config.py                      # Spec 爬虫配置
│       └── spec_crawler.py                # 3GPP FTP → ZIP → PDF
│
├── processor/                       # 数据处理模块（tdocs/specs 并列）
│   ├── __init__.py
│   ├── tdocs/                           # TDoc 处理器
│   │   ├── __init__.py
│   │   ├── parser.py                     # .docx/.doc 文本提取
│   │   ├── metadata.py                   # 元数据（70+ 公司识别）
│   │   ├── chunker.py                    # 层级分片（parent-child）
│   │   └── indexer.py                    # TDoc ChromaDB 索引构建
│   └── specs/                           # Spec 处理器
│       ├── __init__.py
│       ├── spec_parser.py                 # PDF 文本提取（pdfplumber）
│       ├── spec_chunker.py               # 章节层级分片（6 级标题 + 附录）
│       └── spec_indexer.py               # Spec ChromaDB 索引构建
│
├── retriever/                       # 检索模块（tdocs/specs 并列）
│   ├── __init__.py
│   ├── tdocs/                           # TDoc 检索器
│   │   ├── __init__.py
│   │   └── hybrid_retriever.py             # dense + BM25 + reranker
│   └── specs/                           # Spec 检索器
│       ├── __init__.py
│       └── spec_retriever.py              # 章节引用检测 + 定向检索
│
├── llm/                             # LLM 提供者（tdocs/specs 共用）
│   ├── base.py
│   ├── claude_provider.py
│   ├── ollama_provider.py
│   └── config.py
│
├── utils/                           # 工具模块
│   ├── __init__.py
│   └── llm_utils.py                     # Ollama 检测、模型选择
│
└── data/                            # 数据目录（tdocs/specs 并列）
    ├── 3gpp_glossary.json             # 术语词典（含 _spec_sections 章节映射）
    ├── tdocs/                          # TDoc 数据
    │   ├── raw/                            # 原始文档
    │   │   ├── manifest.json
    │   │   └── TSGR2_134/
    │   ├── processed/                    # 处理后数据
    │   │   ├── texts/
    │   │   ├── metadata.jsonl
    │   │   └── chunks/
    │   └── chroma_db/                   # 向量数据库（collection: 3gpp_tdocs）
    └── specs/                          # Spec 数据
        ├── raw/                            # 下载的 ZIP + PDF + manifest
        ├── processed/                    # 解析后数据
        │   ├── texts/                      # 文本（38321.txt 等）
        │   └── chunks/                    # 分片 JSON
        └── chroma_db/                   # 向量数据库（collection: 3gpp_specs）
```

---

## 快速开始

### 1. 环境准备

```bash
cd ~/3gpp-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# TDoc .doc 文件支持（macOS）
brew install antiword
```

### 2. 构建 TDoc 数据库

```bash
# 爬取文档
python crawl.py

# 构建管线
python -c "
from processor.tdocs.parser import TdocParser
from crawler.tdocs.config import RAW_DIR, PROCESSED_DIR
parser = TdocParser(RAW_DIR, PROCESSED_DIR)
parser.process_all(f'{RAW_DIR}/manifest.json')
"
python -c "
from processor.tdocs.metadata import process_all_metadata
from crawler.tdocs.config import RAW_DIR, PROCESSED_DIR
process_all_metadata(f'{RAW_DIR}/manifest.json', f'{PROCESSED_DIR}/texts', f'{PROCESSED_DIR}/metadata.jsonl')
"
python -c "
from processor.tdocs.chunker import process_all_chunks
from crawler.tdocs.config import PROCESSED_DIR
process_all_chunks(f'{PROCESSED_DIR}/metadata.jsonl', f'{PROCESSED_DIR}/texts', f'{PROCESSED_DIR}/chunks')
"
python processor/tdocs/indexer.py
```

### 3. 构建 Spec 数据库

```bash
# 完整流程（下载 + 解析 + 分片 + 索引）
python build_spec_db.py

# 跳过下载，重新处理已有文件
python build_spec_db.py --skip-crawl

# 仅处理部分 spec（测试用）
python build_spec_db.py --spec-list 38.321,38.331
```

**目标 Spec 列表：**

| Spec | 标题 | 内容范围 |
|------|------|---------|
| TS 38.211 | NR; Physical channels and modulation | 物理信道与调制 |
| TS 38.212 | NR; Multiplexing and channel coding | 复用与信道编码 |
| TS 38.213 | NR; Physical layer procedures for control | 控制信道物理层过程 |
| TS 38.214 | NR; Physical layer procedures for data | 数据信道物理层过程 |
| TS 38.300 | NR; Overall description | NR 整体描述 |
| TS 38.321 | NR; MAC layer protocol | MAC 层协议 |
| TS 38.322 | NR; RLC layer protocol | RLC 层协议 |
| TS 38.331 | NR; RRC protocol | RRC 层协议 |

数据来源：`https://www.3gpp.org/ftp/Specs/2025-12/Rel-19/38_series/`

### 4. 启动

```bash
# 启动 Agent
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python chainlit run app.py --port 8000

# 如需 AI 深度分析，启动 Ollama
ollama pull qwen3:8b
ollama serve
```

浏览器打开 http://localhost:8000，选择 Skill 后即可查询。

---

## 使用流程

```
1. 打开页面 → 显示欢迎页和两个 Skill 按钮
       │
       ├── [TDoc 会议文档检索] ──→ 输入技术关键词检索会议提案
       │                            支持按工作组/公司过滤
       │                            点击「深度分析」获取 AI 结构化报告
       │
       └── [3GPP 规范查询] ────→ 输入技术名词查询协议原文
                                    精确定位到 Spec 章节
                                    支持直接引用章节号（如 "38.321 5.1"）
                                    点击「深度分析」获取协议内容整理

2. 查询后可点击 [切换技能] 切换到另一个 Skill
```

**Spec 查询示例：**
- `random access procedure` → 定位到 TS 38.321 Section 5.1
- `38.321 5.1` → 直接检索 MAC 层随机接入流程
- `PDCCH monitoring` → 检索控制信道监听相关章节
- `carrier aggregation` → 跨 spec 检索载波聚合相关内容
- `HARQ-ACK codebook` → 检索 HARQ 反馈相关章节

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 前端 | Chainlit 2.x | 聊天式交互 + Skill 选择 |
| Embedding | all-MiniLM-L6-v2 (384 维) | CPU 优化的句向量模型 |
| 向量数据库 | ChromaDB | 双库隔离（TDoc + Spec） |
| 稀疏检索 | rank-bm25 | BM25 关键词检索 |
| 重排序 | BAAI/bge-reranker-v2-m3 | 交叉编码器精排 |
| LLM | Qwen3-8B (Ollama) / Claude API | 结构化分析 |
| TDoc 解析 | python-docx, textract/antiword | .docx/.doc 文件 |
| Spec 解析 | pdfplumber, PyPDF2 | PDF 文件（含表格/多栏） |
| 术语词典 | 3gpp_glossary.json | 60+ 术语 + 章节映射 |

---

## 当前数据

### TDoc 数据

| 指标 | 数值 |
|------|------|
| 会议 | TSGR2_134（RAN2 第 134 次会议） |
| Tdoc 总数 | 947 |
| 文本提取成功 | 940（99.3%） |
| 公司识别率 | 77.8%（737/947） |
| 层级分块 | 22,504（4,677 parent + 17,827 child） |
| 术语词典 | 60+ 核心术语 |
| 向量模型 | all-MiniLM-L6-v2（384 维） |

### Spec 数据

| 指标 | 数值 |
|------|------|
| 规范版本 | Rel-19 (2025-12) |
| 规范数量 | 8 个（38.211-214, 38.300, 38.321-323, 38.331） |
| 数据来源 | 3GPP FTP Specs 目录 |
| 分片策略 | 章节层级（parent-child，6 级标题 + 附录） |
| 元数据 | spec_number, section_path, section_title, section_level |
| 章节映射 | _spec_sections 支持技术名词→章节定向检索 |

---

## 注意事项

1. **双库隔离**：TDoc 和 Spec 使用独立的 ChromaDB 实例，互不影响，可独立重建
2. **Skill 强制选择**：用户必须先选择 Skill 才能输入查询，避免检索范围混淆
3. **速率限制**：爬虫默认 0.5 秒间隔，建议不超过 10 个并发
4. **protobuf 兼容性**：运行时需设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
5. **Spec 重建**：`build_spec_db.py --skip-crawl` 可跳过下载重新处理，`--spec-list` 支持部分构建
6. **存储需求**：TDoc 约 300MB-2GB/会议；Spec 约 200-500MB（8 个规范）

## License

本项目仅供学习研究使用，3GPP 文档版权归 3GPP 所有。
