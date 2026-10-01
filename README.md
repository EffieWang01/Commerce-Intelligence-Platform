# Commerce SearchRec Agent

一个面向算法岗位作品集的统一电商智能系统，覆盖搜索、推荐、广告、LLM、RAG、Agent、评测和服务化。

项目没有把多个热门词简单拼接在一起。搜索和推荐共享商品表示与候选层；广告只在相关候选上增加
CTR/CVR/价值目标；LLM 负责结构化意图、Query 改写和有依据的解释；Agent 只编排可评测的搜推工具。

## 已完成的链路

- 搜索：BM25 + 稠密召回接口 + 多信号排序。
- 推荐：热门基线 + ItemCF，保留双塔/OpenP5/Gorse 替换接口。
- 广告：Sponsored 候选、CTR × CVR × value 信号和有机相关性约束。
- LLM：本地规则实现用于联调；服务器配置切换 Qwen3-4B。
- RAG：推荐理由只能使用商品元数据和排序信号。
- Agent：搜索、个性化推荐、商品比较三个受限工具，返回完整 trace。
- 评测：NDCG@10、Recall@10、空结果率，以及 API 和 Agent 回归测试。
- 工程：FastAPI、Streamlit、Docker Compose、数据校验和本地/服务器配置。

架构图见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，JD 技能映射见
[docs/JD_COVERAGE.md](docs/JD_COVERAGE.md)。

## 为什么先在本地运行

`local` 配置不下载大模型，不需要 GPU。它用于提前发现数据格式、接口、排序融合、Agent 路由和评测错误。
仓库自带的 10 个商品仅用于 smoke test，不能作为简历里的模型效果。

`server` 配置将相同接口替换为：

| 模块 | 本地实现 | 服务器实现 |
|---|---|---|
| 意图理解与解释 | 可复现规则 | Qwen3-4B 4-bit |
| 稠密表示 | 384 维哈希向量 | BAAI/bge-m3 |
| 重排 | 透明加权排序 | bge-reranker-v2-m3 / LambdaRank |
| 推荐 | ItemCF | 双塔 / OpenP5 / Gorse |
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

## 正式数据契约

服务器数据目录至少包含：

- `products.jsonl`：`item_id, title, category, description, price, sponsored, ctr_prior, cvr_prior`
- `interactions.jsonl`：`user_id, item_id, event, timestamp`
- `queries.jsonl`：`query, user_id, relevant_item_ids`

上传数据前运行：

```bash
python scripts/validate_data.py /path/to/data
```

建议使用 Amazon Shopping Queries 构造搜索相关性集，Amazon Reviews 构造用户行为集，使用 ASIN 作为统一
`item_id`。必须按时间切分交互数据，并按 Query 切分搜索数据，避免信息泄漏。

## 服务器实验顺序

1. 10 万交互：验证 BGE 建索引、指标和显存峰值。
2. 100 万交互：训练双塔与 LambdaRank，记录 Recall、NDCG 和 P95。
3. 加入 LLM Query 改写，做开启/关闭消融。
4. 加入 Grounded Explanation，评估引用正确率。
5. 加入 ESMM 广告分支，报告 AUC、LogLoss 和相关性守门指标。
6. 最后启用 Agent，评估工具选择、参数完全匹配率、任务成功率和无效调用率。

不要同时训练所有模型。每一阶段产生独立、可复现的指标表后再进入下一阶段。

## 亮点

- 同一个 Item 表示服务搜索、推荐和广告，减少重复系统。
- LLM 输出严格的意图 JSON，无法绕开传统召回排序指标。
- 广告提升受有机相关性约束，能讨论收益与体验的权衡。
- Agent 是可回放状态机，每一步都能评测，而不是只有聊天截图。
- 本地和服务器共享 API 合约，昂贵算力只用于已通过验收的模型替换。

## 当前验证

```text
4 tests passed
sample NDCG@10: 0.9799
sample Recall@10: 1.0000
sample empty-result rate: 0.0000
```

这些 sample 指标只验证评测代码和端到端链路，不能代表正式模型质量。

