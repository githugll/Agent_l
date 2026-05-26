# 3GPP Agent — 双 Skill 智能检索分析系统

基于 RAG 技术的 3GPP 垂直领域检索分析助手，支持两个独立 Skill：

| Skill | 功能 | 数据源 |
|-------|------|--------|
| **TDoc 会议文档检索** | 检索 RAN2 会议提案，按公司/工作组过滤，LLM 分析各公司立场 | TSGR2_134 会议 ~940 个 TDoc |
| **3GPP 规范查询** | 查询协议原文，检索后直接由 LLM 整理，可选查看标准原文溯源 | Rel-19 NR 规范 8 个（38.211–214, 38.300, 38.321–323, 38.331） |

---

## 系统架构

```
                    Chainlit Chat UI
                    ┌──────────────────────────────────┐
                    │  Skill 选择（TDoc / Spec）        │
                    │  ↓                              │
                    │  设置面板（工作组/公司/模型/结果数）  │
                    │  ↓                              │
                    │  输入查询词                       │
                    │  ↓                              │
                    │  检索 → LLM 分析 → 结果展示       │
                    └──────────────────────────────────┘
                              ↓
                    ┌────────────────────────────────┐
                    │      SkillRouter (agent/)        │
                    └────────┬─────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              │              ▼
     HybridRetriever    ┌──────────┐   SpecRetriever
     (TDoc 检索)       │ LLM 配置  │   (Spec 检索)
     dense + BM25      │ (Ollama/ │   dense + BM25
     + reranker        │  Claude)  │   + 章节引用检测
              │        └──────────┘              │
              ▼                                  ▼
           ChromaDB                           ChromaDB
           (tdocs)                            (specs)
```

---

## 项目结构

```
3gpp-agent/
├── app.py                         # Chainlit 前端：Skill 路由 + UI 交互
├── requirements.txt               # Python 依赖
├── .env                           # 环境变量（API Key 等）
├── .chainlit/
│   └── config.toml                # Chainlit UI 配置
│
├── agent/                         # Agent 层
│   └── router.py                 # SkillRouter：根据 skill 分发到对应 Retriever
│
├── llm/                          # LLM 提供者
│   ├── base.py                   # LLMProvider 抽象接口
│   ├── config.py                 # LLMConfig + create_llm_provider() 工厂
│   ├── claude_provider.py        # Claude / Anthropic API 提供者
│   └── ollama_provider.py        # Ollama 本地模型提供者
│
├── memory/                        # 记忆模块（会话 + 长期）
│   ├── models.py                 # ConversationTurn / Bookmark 等数据模型
│   ├── db.py                     # SQLite 数据库管理
│   ├── short_term.py             # 短期会话记忆
│   ├── long_term.py              # 长期用户记忆（偏好/收藏夹）
│   └── context_builder.py        # System prompt 上下文构建
│
├── retriever/                     # 检索层
│   ├── tdocs/
│   │   └── hybrid_retriever.py   # TDoc 混合检索：dense + BM25 + rerank
│   └── specs/
│       └── spec_retriever.py     # Spec 混合检索：+ 章节引用检测
│
├── crawler/                       # 数据采集
│   ├── tdocs/
│   │   ├── config.py             # TDoc 爬虫配置
│   │   ├── tdoc_crawler.py      # TDoc 主爬虫
│   │   └── async_crawler.py      # 异步并发爬取
│   └── specs/
│       ├── config.py             # Spec 下载配置
│       └── spec_crawler.py       # 规范 PDF 下载
│
├── processor/                     # 数据处理
│   ├── tdocs/
│   │   ├── parser.py             # TDoc 文本解析
│   │   ├── chunker.py            # TDoc 分块策略
│   │   ├── indexer.py            # ChromaDB 索引写入
│   │   └── metadata.py           # 元数据提取
│   └── specs/
│       ├── spec_parser.py        # PDF 解析
│       ├── spec_chunker.py       # Spec 分块（按章节）
│       └── spec_indexer.py       # ChromaDB 索引写入
│
├── utils/
│   └── llm_utils.py              # LLM 工具：可用性检测 / 中文翻译 / 模型列表
│
└── data/
    ├── 3gpp_glossary.json         # 术语词典（含中文/英文映射 + Spec 章节映射）
    ├── tdocs/
    │   ├── raw/                  # 原始 .docx 文件
    │   ├── processed/            # 处理后文本/chunks
    │   └── chroma_db/            # ChromaDB（collection: 3gpp_tdocs）
    ├── specs/
    │   ├── raw/                  # 下载的 PDF
    │   ├── processed/            # 解析后文本/chunks
    │   └── chroma_db/            # ChromaDB（collection: 3gpp_specs）
    └── memory/                   # SQLite 记忆数据库
```

---

## 快速开始

### 1. 安装依赖

```bash
cd 3gpp-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# .env
ANTHROPIC_API_KEY=sk-xxx
ANTHROPIC_BASE_URL=https://your-proxy-url.com  # 可选
HF_HUB_OFFLINE=1  # 使用本地缓存的 HuggingFace 模型
```

### 3. 启动

```bash
chainlit run app.py --port 8111
```

浏览器打开 http://localhost:8111。

---

## 功能说明

### Skill 选择
- 欢迎页有两个 Skill 按钮：**TDoc 会议文档检索** / **3GPP 规范查询**
- 选择后可在**输入框上方**设置：工作组、公司（仅 TDoc）、规范（仅 Spec）、LLM 模型、结果数量

### TDoc 检索
- 输入技术关键词（如 `beam management FR2`）
- 支持按工作组（R1/R2/…）和公司过滤
- 快捷命令：`top:50 <关键词>` 直接指定结果数量
- 检索结果展示表格 + 详情，点击**深度分析**由 LLM 按公司分组整理立场

### Spec 规范查询
- 输入技术名词（如 `random access procedure`）
- 支持精确章节引用（如 `38.321 5.1`）
- **检索后直接进入 LLM 分析**，输出协议整理结果
- 分析完成后可点击**标准原文溯源**查看召回的标准原文

### 通用
- **收藏结果**：将当前结果存入收藏夹
- **切换 Skill**：返回 Skill 选择页

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 前端 | Chainlit 2.x | 聊天式交互 |
| Embedding | all-MiniLM-L6-v2（384 维） | CPU 优化，向量检索 |
| 向量数据库 | ChromaDB | 双库隔离（TDoc / Spec） |
| 稀疏检索 | rank-bm25 | BM25 关键词检索 |
| 重排序 | BAAI/bge-reranker-v2-m3 | 交叉编码器精排 |
| LLM | Claude API / Ollama | 分析与翻译 |
| 术语词典 | data/3gpp_glossary.json | 60+ 术语 + Spec 章节映射 |

---

## 当前数据

| 数据集 | 规模 |
|--------|------|
| TDoc（TSGR2_134） | ~940 篇提案，22,504 chunks |
| Spec（Rel-19） | 8 个规范，~5,000 chunks |

---

## License

本项目仅供学习研究使用，3GPP 文档版权归 3GPP 所有。
