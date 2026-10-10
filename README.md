# Commerce SearchRec Agent

一个面向算法岗位作品集的统一电商智能系统，覆盖搜索、推荐、广告、LLM、RAG、Agent、评测和服务化。

项目没有把多个热门词简单拼接在一起。搜索和推荐共享商品表示与候选层；广告只在相关候选上增加
CTR/CVR/价值目标；LLM 负责结构化意图、Query 改写和有依据的解释；Agent 只编排可评测的搜推工具。

## 已完成的链路

- 搜索：BM25 + BGE/Qdrant 稠密召回 + 多信号融合。
- 推荐：ItemCF 本地基线 + Gorse 服务，失败时自动降级。
- 广告：Sponsored 候选、CTR × CVR × value 信号和有机相关性约束。
- LLM：本地规则实现用于联调；服务器通过 OpenAI 兼容协议接国产 VLM 或本地 vLLM。
- 多模态：Chinese-CLIP 文搜图/图搜图，Qwen-VL 提取视觉属性。
- RAG：推荐理由只能使用商品元数据和排序信号。
- Agent：搜索、个性化推荐、商品比较三个受限工具，返回完整 trace。
- 评测：NDCG@10、Recall@10、空结果率，以及 API 和 Agent 回归测试。
- 工程：FastAPI、Streamlit、Docker Compose、数据校验和本地/服务器配置。

架构图见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，JD 技能映射见
[docs/JD_COVERAGE.md](docs/JD_COVERAGE.md)，低资源组合与实验矩阵见
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)。

## 为什么先在本地运行

`local` 配置不下载大模型，不需要 GPU。它用于提前发现数据格式、接口、排序融合、Agent 路由和评测错误。
仓库自带的 10 个手工商品（`data/sample`）仅用于 smoke test，不能作为简历里的模型效果。默认 Demo
还会优先加载 `data/demo`：当前由 50,000 条 Amazon Reviews 2023 商品元数据组成，覆盖 8 个品类，
全部带官方图片链接和采集时的 USD 价格。

`server` 配置将相同接口替换为：

| 模块 | 本地实现 | 服务器实现 |
|---|---|---|
| 意图理解与解释 | 可复现规则 | 国产 OpenAI 兼容 VLM / 本地 vLLM |
| 稠密表示 | 384 维哈希向量 | BAAI/bge-m3 + Qdrant |
| 重排 | 透明加权排序 | bge-reranker-v2-m3 |
| 推荐 | ItemCF | Gorse，ItemCF 自动回退 |
| 图文检索 | 关闭 | Chinese-CLIP |
| 广告 | CTR/CVR 先验 | ESMM 多任务模型 |

服务器运行步骤见 [docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)。

## 本地快速开始

Windows PowerShell：

```powershell
cd commerce_searchrec_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest -q
python -m commerce_ai.cli smoke --query "预算 500 的降噪耳机"
python -m commerce_ai.cli evaluate
```

分别启动 API 和 Demo：

```powershell
uvicorn commerce_ai.api:app --port 8008
streamlit run app.py --server.port 8503
```

打开：

- API 文档：http://127.0.0.1:8008/docs
- Demo：http://127.0.0.1:8503

也可以直接运行：

```powershell
docker compose up --build
```

## API

```bash
curl -X POST http://127.0.0.1:8008/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"预算 500 的降噪耳机","user_id":"u001","top_k":5}'

curl -X POST http://127.0.0.1:8008/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","top_k":5}'

curl -X POST http://127.0.0.1:8008/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我找预算 500 的降噪耳机","user_id":"u001"}'
```

图片搜索复用 `/v1/search`，在请求 JSON 中增加不超过 8 MB 的 `image_data` data URL。
Streamlit 界面可以直接上传 JPG、PNG 或 WebP。

## 正式数据契约

服务器数据目录至少包含：

- `products.jsonl`：必需字段为 `item_id, title, category, description`；`price` 可以为 `null`；可选字段为
  `brand, image_url, attributes, source, source_id, source_price, source_currency, sponsored, ctr_prior, cvr_prior`
- `interactions.jsonl`：`user_id, item_id, event, timestamp`
- `queries.jsonl`：`query, user_id, relevant_item_ids`

上传数据前运行：

```bash
python scripts/validate_data.py /path/to/data
```

### 重建公开演示目录

演示目录使用 McAuley Lab 的 Amazon Reviews 2023 商品元数据。导入器从 8 个来源品类流式读取，
只保留同时具有标题、类目标签、属性标签、价格、描述和官方图片 URL 的商品。当前目录包含 50,000 件商品，价格保留
数据集采集时的 USD 币种：

```powershell
$env:PYTHONPATH='src'
python scripts/import_amazon_reviews_2023.py --source ucsd --output data/demo --max-products 1000
python scripts/validate_data.py data/demo --require-complete-products
```

导入器不会为缺失价格或图片的记录补造字段，而是直接过滤。来源、抽样参数、币种、完整商品数和
各字段缺失数记录在 `data/demo/manifest.json` 与 `data/demo/ATTRIBUTION.md`。演示目录当前只建立
商品目录和查询集，`interactions.jsonl` 为空；个性化推荐数据应在推荐阶段单独导入并按时间切分。

建议使用 Amazon Shopping Queries 构造搜索相关性集，Amazon Reviews 构造用户行为集，使用 ASIN 作为统一
`item_id`。必须按时间切分交互数据，并按 Query 切分搜索数据，避免信息泄漏。

## 服务器实验顺序

1. 10 万交互：验证 BGE 建索引、指标和显存峰值。
2. 对比 BM25、BGE、混合召回和 BGE reranker，记录 Recall、NDCG、MRR 和 P95。
3. 接入 Gorse，按时间切分交互数据，评估个性化推荐和冷启动。
4. 加入 LLM Query 改写，做开启/关闭消融。
5. 加入商品图片后评估文搜图、图搜图和图文融合。
6. 评估 Grounded Explanation 的无依据陈述率。
7. 最后启用 Agent，评估工具选择、参数完全匹配率、任务成功率和无效调用率。

## 下载本地模型并传到服务器

下面的命令会把官方 Qwen3-4B 快照下载到 `D:\models\Qwen3-4B`，支持断点续传，并生成
`TRANSFER_MANIFEST.json` 和 `verify_transfer.py`：

```powershell
python scripts/download_hf_model.py
```

使用 WinSCP 将整个 `Qwen3-4B` 目录上传到服务器，例如 `/data/models/Qwen3-4B`。上传完成后在
服务器目录中执行：

```bash
python verify_transfer.py .
```

只有输出 `OK` 才表示所有文件大小和 SHA-256 均与本地一致。下载其他官方模型时可以传入
`--repo-id` 和 `--output`，已完成下载的目录可用 `--verify-only` 重新生成清单。

不要同时训练所有模型。每一阶段产生独立、可复现的指标表后再进入下一阶段。

## 亮点

- 同一个 Item 表示服务搜索、推荐和广告，减少重复系统。
- LLM 输出严格的意图 JSON，无法绕开传统召回排序指标。
- 广告提升受有机相关性约束，能讨论收益与体验的权衡。
- Agent 是可回放状态机，每一步都能评测，而不是只有聊天截图。
- 本地和服务器共享 API 合约，昂贵算力只用于已通过验收的模型替换。

## 当前验证

```text
15 tests passed
sample NDCG@10: 1.0000
sample Recall@10: 1.0000
sample MRR@10: 1.0000
sample empty-result rate: 0.0000
```

这些 sample 指标只验证评测代码和端到端链路，不能代表正式模型质量。测试与上面的指标使用
`data/sample` 的 10 条中文样例；`data/demo` 是 5 万条英文商品目录，用于演示规模与链路，不用于模型质量结论。

