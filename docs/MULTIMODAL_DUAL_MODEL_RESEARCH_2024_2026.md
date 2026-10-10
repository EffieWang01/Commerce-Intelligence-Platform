# 多模态 Query Understanding 与排序模型调研（2024-2026）

## 1. 结论先行

在单卡 RTX 4090（24 GB）约束下，建议训练两个**独立任务模型/适配器**，但不从零预训练基础模型：

| 模块 | 推荐底座 | 总参数量 | 建议训练方式 | 预计可训练参数 | 主要职责 |
|---|---:|---:|---|---:|---|
| Query Understanding | Qwen3-VL-4B-Instruct | 4B 级 | 4-bit QLoRA，先冻结视觉塔 | 约 20M-50M，需按 target modules 实测 | 文本/图片查询意图、类目、属性槽位、约束、改写，输出结构化 JSON |
| Multimodal Ranker | Qwen3-VL-2B-Instruct；保守替代为 Qwen2-VL-2B-Instruct | 2B 级 | 4-bit QLoRA，LoRA r=8，冻结视觉塔 | 约 0.5%，即约 10M | 对 top 20-25 商品做图文联合 pointwise/listwise 重排 |

这不是“各训练一个完整 4B/2B 模型”，而是冻结绝大部分底座参数，分别训练小型 LoRA 权重。两个模型应顺序训练；推理时可将 4-bit 权重同时装入 24 GB 显存，但最终并发量和视觉 token 数仍需基准测试。

当前已下载的 `Qwen3-4B` 是**纯文本模型**，不能直接接收商品图片。它可以作为文本版 QU 基线，但多模态正式方案需要另下 `Qwen3-VL-4B-Instruct` 或 `Qwen3-VL-2B-Instruct`。

## 2. 调研范围与问题

时间窗口取 2024-2026。优先检索 SIGIR、KDD、WWW、RecSys、ACL/EMNLP、CVPR、ICLR 等会议论文；2026 年截至调研日的材料多数仍是技术报告，因此与同行评审论文分开看待。

研究问题：

1. 近三年 Query Understanding、检索与排序模型常见参数量是多少？
2. 哪些工作真正联合使用文本和图像，而不是只在标题上写“multimodal”？
3. 在单卡 4090 上，哪些规模能够微调、推理并接入当前搜推 Demo？

参数口径：优先采用论文明确报告的总参数量；论文只给底座名称时按官方型号标称；未披露时标为“未披露”，不猜测。LoRA 参数与底座总参数分开报告。

## 3. 论文与技术报告对照

### 3.1 Query Understanding 与电商领域适配

| 工作 | 年份/来源 | 模态 | 模型规模 | 关键证据 | 对本项目的意义 |
|---|---|---|---:|---|---|
| [Exploring Query Understanding for Amazon Product Search](https://arxiv.org/abs/2408.02215) | IEEE BigData 2024 | 文本 | multilingual DistilBERT，约 135M | 生产系统把商品意图识别、Query Parsing 和排序特征联动；强调延迟与内存约束 | 证明生产 QU 不必用 7B，结构化多任务比自由生成更重要；但其本身不是多模态 |
| [BSharedRAG](https://aclanthology.org/2024.findings-emnlp.62.pdf) | Findings of EMNLP 2024 | 文本 | Baichuan2-7B | 共享领域底座，检索与生成使用两个 LoRA；论文也显示完全共享目标会发生负迁移 | 支持“同族底座、任务适配器分离”的设计，而不是把理解与排序硬塞进一个 loss |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | 2025 技术报告/官方模型 | 图像+文本+视频 | 2B、4B、8B 等 | 官方提供 2B/4B/8B Dense 版本，支持多图输入、OCR、商品识别和视觉 token 预算控制 | 4B 适合承担结构化 QU；2B 可作为低成本替代 |
| Pailitao-MMSearch | 2026 技术报告 | 商品图像+文本 | 论文/报告版本需逐版核对 | 面向原生电商多模态搜索，方向高度匹配，但尚不应替代已同行评审证据 | 可用于后续数据构造和任务设计参考，不作为当前参数预算的唯一依据 |

multilingual DistilBERT 的“约 135M”来自官方配置（6 层、hidden size 768、词表 119,547）的参数量计算；[官方配置](https://huggingface.co/distilbert/distilbert-base-multilingual-cased/raw/main/config.json)没有直接打印总参数数值，因此这里保留近似号。

### 3.2 多模态检索与排序

| 工作 | 年份/来源 | 模态/任务 | 参数量 | 训练方式与资源信号 | 判断 |
|---|---|---|---:|---|---|
| [Setwise Ranking](https://hongleizhuang.github.io/files/shengyao_sigir_2024.pdf) | SIGIR 2024 | 文本重排 | 780M、3B、11B | Flan-T5 zero-shot，对 BM25 top 100 重排；setwise 减少调用次数 | 表明排序效果与计算量强相关，0.8B-3B 已是实用区间；非多模态基线 |
| [PRIME](https://cseweb.ucsd.edu/~jmcauley/pdfs/kdd25d.pdf) | KDD 2025 | 商品图片+标题+价格+品牌+类目，多模态推荐排序 | Qwen2-VL-2B | 4-bit QLoRA，r=8、alpha=16、dropout=0.05；冻结图像编码器，训练约 0.5%（约 10M）；top 25 一次前向打分 | 与本项目最接近，是选择 2B ranker 的首要证据 |
| [LamRA](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_LamRA_Large_Multimodal_Model_as_Your_Advanced_Retrieval_Assistant_CVPR_2025_paper.html) | CVPR 2025 | 通用多模态检索与 pointwise/listwise rerank | Qwen2-VL-7B 级 | 冻结视觉编码器、给语言模型加 LoRA；原论文训练使用多张 A100 | 方法值得复用，但 7B 原配方不适合单卡 4090 直接复现 |
| [GME](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Bridging_Modalities_Improving_Universal_Multimodal_Retrieval_by_Multimodal_Large_Language_CVPR_2025_paper.pdf) | CVPR 2025 | 文本、图片、图文组合统一检索 | 2.2B、8.2B | Qwen2-VL 2B/7B 初始化，BF16 + LoRA r=8；每个 query 配 1 正例和 8 个 hard negatives | 2.2B 已有很强表现，适合作为多模态召回/特征编码器；它不是最终交叉排序器 |
| [VLM2Vec](https://proceedings.iclr.cc/paper_files/paper/2025/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf) | ICLR 2025 | 通用多模态 embedding | 约 4B-8B 级底座 | 通过对比学习与 LoRA 把生成式 VLM 变为 embedding 模型 | 证明 VLM 可兼做召回，但对当前 4090 项目，优先级低于先把 2B ranker 做好 |
| [VLM Is a Strong Reranker / RagVL](https://aclanthology.org/2025.findings-emnlp.432/) | Findings of EMNLP 2025 | 图像 RAG 重排 | 约 7B-13B 级实验模型 | 指令微调 VLM 过滤 top-k 图像，并对生成器做噪声注入 | 是多模态重排的正面证据，但规模偏大、场景不是商品搜索 |
| [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) | 2025 技术报告 | 文本 embedding/rerank | 0.6B、4B、8B | 同时提供 embedding 与 reranker | 0.6B 是低延迟文本重排强基线，但不能单独满足图片排序需求 |

## 4. 参数量规律

从上述证据可以得到三个比“越大越好”更有用的规律：

1. **Query Understanding 的生产基线仍可小到 0.1B 级。** Amazon 的真实系统使用 DistilBERT 级模型完成意图和解析，说明 QU 的价值主要来自任务定义、标签体系和与排序特征的联动。
2. **多模态能力将实用下限推到约 2B。** GME 的 2.2B、PRIME 的 2B 都表明，小型 VLM 已能处理图片与结构化商品属性。视觉塔、投影层和语言主干共同构成这部分参数。
3. **排序目标比单纯扩大参数更重要。** PRIME 用 2B 和约 10M 可训练参数取得有竞争力结果；LamRA/GME 也通过 LoRA、hard negatives、pointwise/listwise 目标获得能力，而不是全参重训。

因此本项目的合理区间是：

| 层次 | QU | Ranker | 适用情况 |
|---|---:|---:|---|
| 最小可运行 | 0.1B-0.6B 文本编码器 + 冻结视觉编码器 | 0.3B-0.6B cross-encoder + 图像向量 | 极低延迟，但图文交互能力较弱 |
| 推荐 | 4B VLM + QLoRA | 2B VLM + QLoRA | 兼顾作品完整度、多模态与 4090 可训练性 |
| 研究上界 | 7B-13B VLM | 7B-13B VLM | 多卡训练或 API 教师模型，不适合作为本机主方案 |

## 5. 推荐的双模型设计

### 5.1 模型 A：Multimodal Query Understanding

底座：`Qwen3-VL-4B-Instruct`。

输入：文本 query，可选参考图片，以及可选会话上下文。

输出固定 JSON Schema：

```json
{
  "intent": "product_search",
  "category": "running_shoes",
  "attributes": {"color": "black", "gender": "women"},
  "hard_constraints": {"price_max": 500},
  "soft_preferences": ["lightweight", "cushioning"],
  "normalized_query": "女士黑色轻量缓震跑鞋 500元以内",
  "visual_evidence": ["black", "low-top", "mesh upper"],
  "confidence": 0.91
}
```

训练目标采用多任务 SFT：意图分类、类目分类、属性抽取、查询改写、图片属性描述共享一套结构化输出。第一版冻结 vision tower，只给语言主干和跨模态 projector/必要线性层加 LoRA；当图片属性错误成为主要瓶颈时，再解冻 projector 或视觉塔最后 1-2 层。

数据优先级：真实 query-product 曝光/点击/购买日志 > 人工标注 query schema > 商品标题与属性构造的合成 query > 教师模型生成并经规则/人工抽检的数据。必须保留无图样本，防止模型只靠图片猜类目。

### 5.2 模型 B：Multimodal Ranker

底座：`Qwen3-VL-2B-Instruct`；若工具链兼容性优先，则先复现论文已验证的 `Qwen2-VL-2B-Instruct`。

输入：QU 结构化结果 + 用户上下文 + top 20-25 候选；每个候选含商品图、标题、品牌、类目、价格和召回分数。

输出：每个候选的相关性/偏好分数，或一次前向输出候选索引的 logits。第一版采用 PRIME 式 verbalizer，避免逐商品生成长文本；训练混合三类损失：

- pointwise：是否满足硬约束与 query 相关性；
- pairwise：购买/点击商品优于曝光未点击商品；
- listwise：优化 NDCG，并学习候选间相对次序。

训练配置从已验证的 PRIME 配方开始：4-bit QLoRA、r=8、alpha=16、dropout=0.05、学习率 `1e-4`、1 epoch 起步、冻结图像编码器。负例至少包含同类目近邻、同图不同规格、标题相似但属性冲突三类 hard negatives。

## 6. 单卡 4090 可行性

以下是工程预算，不是论文实测值，落地时要用当前 CUDA/Transformers 版本做峰值显存基准：

| 作业 | 预计设置 | 可行性判断 |
|---|---|---|
| QU 4B QLoRA | 4-bit、batch 1、梯度累积、图像 336-448 px、文本 1K-2K token、gradient checkpointing | 可行，建议把视觉 token 限在 256-512/图 |
| Ranker 2B QLoRA | 4-bit、batch 1、top 20 起步、商品图缩放/视觉 token 压缩、FlashAttention | 可行；多图长上下文是主要显存风险 |
| 同时训练两模型 | 不建议 | 顺序训练并保存各自 adapter |
| 同时推理两模型 | 4-bit 加载，限制并发和上下文 | 权重本身可容纳；KV cache、图片数量和并发需实测 |
| 7B 多模态全参训练 | 不可行 | 4090 不适合作为主路线 |

降低风险的顺序：先减候选数到 20，再降低每张图视觉 token，再做梯度累积，最后才降低底座规模。不要一开始把 top 100 商品图片全部塞给 VLM。

## 7. 实施路线与验收

### Phase 0：数据与纯文本基线

- 建立 QU JSON 标签规范和 Ranker 样本协议。
- 用现有 `Qwen3-4B` 做文本 QU LoRA 基线；用 Qwen3-Reranker-0.6B 或现有排序器做文本重排基线。
- 指标：QU intent accuracy、slot micro/macro F1、JSON valid rate；排序 Recall@K、NDCG@10、MRR、P95 latency。

### Phase 1：多模态 QU

- 下载 Qwen3-VL-4B-Instruct，加入“文字+图搜同款/相似款/属性约束”样本。
- 与 text-only baseline 做消融：text、image、text+image。
- 验收：图像存在时属性 F1/类目准确率显著提升，无图集不能明显退化。

### Phase 2：多模态 Ranker

- 复现 PRIME 式 2B QLoRA，候选数从 20 开始。
- 采用 hard negatives、点击/购买加权和位置偏差校正。
- 验收：相同召回候选下，相对当前规则/BM25/向量混排的 NDCG@10 提升；硬约束违规率下降；记录吞吐和 P95。

### Phase 3：蒸馏与部署优化

- 用 4B QU 或国产 API 教师生成难例解释与伪标签，但线上 ranker 仍保持 2B。
- 缓存商品图像特征；对热门 query 缓存 QU 结构化结果。
- 视延迟结果决定是否把 2B ranker 蒸馏到 0.6B cross-encoder + 冻结图像特征的版本。

最终验收必须同时报告效果、资源和消融：`text-only vs multimodal`、`random negatives vs hard negatives`、`pointwise vs pointwise+listwise`、模型总参数、可训练参数、峰值显存、P50/P95 延迟。这样既能证明功能，也能回答 JD 中关于模型训练、多模态、检索排序、评估与工程部署的问题。

## 8. 风险与尚未定论

- 2026 年电商多模态报告更新很快，部分尚未同行评审；它们适合提供方向，不宜单独决定架构。
- 通用多模态 benchmark 的提升不等于电商搜索转化提升，必须用本项目 query-product 数据验证。
- 点击日志存在位置偏差、曝光偏差和流行度偏差；未经校正直接训练会让 ranker 学会复现旧排序。
- 4B QU 并非必然优于 2B。若结构化任务上 2B 与 4B 差距很小，应优先部署 2B，把显存留给并发和 ranker。
- 多模态输入最昂贵的部分通常是视觉 token 与候选图片数量，而不只是模型权重大小。

## 9. 用公开赛题固定项目目标

最推荐的主线是 **TREC Product Search 2025 + Amazon ESCI + SQID**，而不是自行发明任务与评价指标。

### 9.1 Query Understanding：TREC Query Reformulation

[TREC Product Search 2025](https://trec-product-search.github.io/search.html) 的搜索任务正是面向任务型电商查询的 Query Reformulation。输入类似“布置生日派对”或“做千层面”，模型需要补全隐含商品需求，输出一条或最多四条更适合检索的改写查询。

可直接采用赛题目标：

- 输入：原始文字 query，可扩展可选参考图片；
- 输出：结构化 query understanding JSON 和 reformulated query；
- 官方训练信号：人工改写、query-product relevance judgment；
- 指标：Task Completion nDCG、Essential Product Recall@K、Product Coverage、MAP 和 diversity；
- 模型：`Qwen3-VL-2B-Instruct + qu_adapter`。

TREC 2024 数据还提供 30,734 条查询、1,661,907 个商品、392,119 条训练 qrels、169,952 条验证 qrels，以及 20,888 组带正负商品的训练 triples，可直接复用官方切分。

### 9.2 排序：Amazon ESCI/SQID

[Amazon ESCI Challenge](https://www.amazon.science/blog/amazon-product-query-competition-draws-more-than-9-200-submissions) 为每个真实购物 query 提供最多 40 个候选商品和人工相关性标签：Exact、Substitute、Complement、Irrelevant。三个官方任务分别是 query-product ranking、ESCI 四分类和 substitute identification。

[SQID](https://arxiv.org/abs/2405.15190) 在 ESCI 基础上为约 190,000 个商品补充图片，因此能够把原本的文本排序题转换成真正的多模态排序题，而不改变 query、product ID、ESCI 标签和官方评价逻辑。

可直接采用赛题目标：

- 输入：QU 结果，以及最多 40 个候选的标题、描述、属性和图片；
- 主任务：按 `E > S > C > I` 重排；
- 辅助任务：ESCI 四分类和 Substitute 二分类；
- 主指标：nDCG@10；辅助指标：四分类 Accuracy/Macro-F1、Recall@K、MRR；
- 模型：`Qwen3-VL-2B-Instruct + ranker_adapter`。

### 9.3 最终项目题目

> 面向任务型电商查询的轻量级多模态理解与商品重排系统：基于 TREC Query Reformulation 和 Amazon ESCI/SQID 的单卡训练实践。

端到端流程固定为：

```text
文字 query / 参考图片
        ↓
Qwen3-VL-2B + QU Adapter
        ↓
结构化意图、隐含需求、改写 query
        ↓
BM25 + 多模态向量召回 top 40
        ↓
Qwen3-VL-2B + Ranker Adapter
        ↓
ESCI 相关性分数与最终 top 10
```

### 9.4 备选赛题

| 赛题/数据 | 优点 | 不作为主线的原因 |
|---|---|---|
| [CAMEO](https://cameo-task.com/) | 2025-2026、新、原生“参考图+文字修改”组合检索；nDCG@5/10 | 仅约 4,086 个商品、主要是越南语美妆，适合做多模态扩展实验 |
| [SIGIR eCom 2021 Coveo](https://github.com/coveooss/SIGIR-ecom-data-challenge) | 36M 会话事件、80 万搜索事件，包含图文向量与点击负反馈 | query 和商品内容被预编码/匿名化，不利于训练可解释 QU 与原生 VLM |
| Shopee Price Match | 图片+标题、适合度量学习和商品去重 | 没有自然语言搜索 query，也没有相关性分级 |
| KDD Cup 2020 Taobao Multimodal Recall | 真实 query、图像特征、约 30 候选排序 | 数据较旧且多为预提取特征，任务覆盖不如 ESCI+SQID 完整 |

因此，主实验使用 TREC/ESCI/SQID；CAMEO 用作“图片查询+文字修改”的跨数据集泛化测试。这样 Query Understanding 和 Ranker 都有公开任务、官方标签、固定指标和可对比基线。

