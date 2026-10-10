# 轻量级多模态电商搜推系统项目背景与要求

## 一、项目背景

传统电商搜索系统通常将用户输入的关键词直接交给 BM25、向量检索或混合检索模块，再对召回商品进行排序。这种流程面对“黑色女鞋”“500 元以内的机械键盘”等明确查询时通常能够正常工作，但面对“布置生日派对需要什么”“想找和这张图类似、但更适合夏天穿的衣服”这类任务型或多模态查询时，效果往往明显下降。

问题主要来自两个方面。第一，用户表达的是目标，而不是商品名称，搜索系统需要推断隐含的类目、属性和约束。第二，商品图片中包含颜色、版型、纹理和外观等难以完全写进标题的信息，仅依赖文本容易产生误匹配。因此，一个更完整的电商搜索系统应当同时具备 Query Understanding 和多模态排序能力。

本项目计划训练两个任务模型。第一个模型负责理解用户输入，将自然语言和可选参考图片转换为结构化搜索意图与改写查询。第二个模型负责联合分析查询、商品文本、商品图片和结构化属性，对召回候选进行精排。两个任务都基于公开赛题定义，不自行创造难以验证的目标。

## 二、为什么使用公开赛题

自己设计搜索任务时，最困难的通常不是搭建模型，而是确定标签、构造负样本、选择指标以及判断结果究竟是否有意义。公开赛题已经给出了任务定义、数据切分、评价指标和强基线，可以使项目结果具有可复现性和可比较性。

本项目选择 TREC Product Search、Amazon ESCI Challenge 和 SQID 作为主要数据来源。三者分别覆盖查询改写、商品相关性排序和商品图片，可以组成一条相对完整的多模态电商搜索链路。

## 三、Query Understanding 任务

Query Understanding 部分参考 [TREC Product Search 2025 Query Reformulation](https://trec-product-search.github.io/search.html)。该赛题关注任务型电商查询。例如，用户输入“布置生日派对”，查询中没有直接出现气球、蜡烛、餐具和装饰等商品词，但这些商品是完成任务所必需的。模型需要识别用户的真实目标，并把隐含需求改写成搜索系统能够处理的查询。

TREC 提供任务型查询、人工改写结果和 query-product 相关性标注。官方评价不只检查生成文字是否相似，而是把改写后的查询真正用于商品检索，再通过 Task Completion nDCG、Essential Product Recall@K、Product Coverage、MAP 和多样性等指标评价检索结果。这种评价方式比 BLEU 或 ROUGE 更适合搜索系统，因为它直接衡量改写是否帮助用户找到商品。

在本项目中，Query Understanding 模型的输入包括原始文字查询、可选参考图片和可选会话上下文。模型输出固定格式的 JSON，其中包括查询意图、目标类目、属性槽位、硬约束、软偏好、视觉证据和改写查询。例如：

```json
{
  "intent": "product_search",
  "category": "running_shoes",
  "attributes": {
    "gender": "women",
    "color": "black"
  },
  "hard_constraints": {
    "price_max": 500
  },
  "soft_preferences": [
    "lightweight",
    "cushioning"
  ],
  "normalized_query": "女士黑色轻量缓震跑鞋 500元以内",
  "visual_evidence": [
    "black",
    "low-top",
    "mesh upper"
  ]
}
```

这样的输出既可以供 BM25 使用，也可以作为向量召回和排序模型的结构化特征。模型训练目标包括 JSON 格式正确率、意图分类、类目识别、属性抽取、查询改写和图像属性识别。最终仍以改写后的检索效果作为核心评价，而不是只看生成文本是否流畅。

## 四、商品排序任务

排序部分采用 [Amazon ESCI Challenge](https://www.amazon.science/blog/amazon-product-query-competition-draws-more-than-9-200-submissions) 的任务定义。ESCI 为真实购物查询提供候选商品，并将 query-product 关系标注为四种类型：

- Exact：商品满足查询中的主要要求；
- Substitute：不是完全匹配，但可以作为合理替代；
- Complement：商品适合搭配使用，但不是用户直接寻找的目标；
- Irrelevant：与查询无关。

相比简单的相关和不相关二分类，ESCI 更接近真实电商业务。例如，查询“iPhone 充电器”时，兼容充电器可能是 Exact 或 Substitute，手机壳属于 Complement，普通台灯则是 Irrelevant。这种分级标签能够同时训练相关性、替代品识别和互补商品理解。

ESCI 原始数据主要包含商品标题、描述、品牌、颜色和 bullet points。为了加入图片，本项目采用 [SQID](https://arxiv.org/abs/2405.15190)。SQID 在 ESCI 的基础上为约十九万个商品补充了图片和视觉表示，同时保留原有 query、商品标识和相关性标签，因此可以在不改变评价任务的情况下进行多模态排序。

排序模型接收 Query Understanding 输出和最多四十个候选商品。每个商品包含图片、标题、品牌、类目、价格、结构化属性以及第一阶段召回分数。模型需要输出每个候选的 ESCI 分类概率与最终相关性分数，并按照 `Exact > Substitute > Complement > Irrelevant` 生成最终排序。

主评价指标采用 nDCG@10，同时报告 Recall@K、MRR、四分类 Accuracy 和 Macro-F1。nDCG@10 用于评价排序位置，Macro-F1 用于防止模型只学习数量最多的标签，硬约束违规率则用于检查价格、类目和属性条件是否真正得到满足。

## 五、模型规模与训练方式

项目受到单卡 RTX 4090、24 GB 显存的约束，因此不考虑从零预训练模型，也不采用两个独立的 7B 模型。推荐方案是让两个任务共享 `Qwen3-VL-2B-Instruct` 基础权重，但分别训练两套 LoRA Adapter：

```text
Qwen3-VL-2B-Instruct
├── qu_adapter
└── ranker_adapter
```

`qu_adapter` 只负责查询理解和结构化改写，`ranker_adapter` 只负责多模态商品排序。二者在逻辑上是两个任务模型，但工程上只需要存储一份基础权重。这样既能减少磁盘和显存占用，也能避免两个训练目标相互干扰。

训练采用 4-bit QLoRA。第一阶段冻结视觉编码器，LoRA rank 从 8 开始，只训练语言主干和必要的跨模态线性层。Query Understanding Adapter 预计训练约八百万到两千万参数；Ranker Adapter 可以参考 KDD 2025 PRIME 的配置，将可训练参数控制在总参数的约 0.5%，即约一千万参数。

当前已经下载的 `Qwen3-4B` 是纯文本模型，不能直接处理商品图片。它可以作为 text-only Query Understanding 基线，但正式的多模态实验应使用 `Qwen3-VL-2B-Instruct`。

## 六、训练数据与样本构造

Query Understanding 训练数据主要来自 TREC 人工改写数据。每条样本包含原始查询、目标改写、相关商品和相关性标签。在此基础上，可以从 Exact 商品的类目和属性中抽取监督信号，生成结构化 JSON 标签。为了避免信息泄漏，训练、验证和测试应按照官方 query 切分，不能让同一查询的候选商品跨越不同集合。

排序训练数据来自 ESCI/SQID。Exact 商品作为强正例，Substitute 作为弱正例或次优商品，Complement 与 Irrelevant 根据训练目标作为不同强度的负例。训练时不能只随机采样无关商品，还需要构造以下 hard negatives：

- 标题非常相似但颜色、尺寸或规格冲突的商品；
- 图片相似但商品类目不同的商品；
- 属于同一类目但违反价格或属性硬约束的商品；
- 原搜索系统排位较高，但人工标签为 Irrelevant 的商品。

排序目标由 pointwise、pairwise 和 listwise 三部分组成。Pointwise 学习单个 query-product 的 ESCI 标签；pairwise 学习 Exact 应排在 Substitute、Complement 和 Irrelevant 前面；listwise 直接学习候选集合中的相对顺序。第一版可以先实现 pointwise 与 pairwise，确认数据管道和指标正常后再加入 listwise loss。

## 七、完整系统流程

系统运行时，用户输入文本，也可以附带一张参考商品图片。Query Understanding 模型首先输出结构化意图和改写查询。系统随后通过 BM25 和多模态向量检索召回候选商品，并合并为 top 40 候选。最后，Ranker Adapter 联合读取查询、商品属性和图片，输出 ESCI 相关性分数并返回 top 10。

```text
文字查询 / 参考图片
        ↓
Qwen3-VL-2B + QU Adapter
        ↓
结构化意图 + Reformulated Query
        ↓
BM25 召回 + 多模态向量召回
        ↓
候选合并与去重，保留 Top 40
        ↓
Qwen3-VL-2B + Ranker Adapter
        ↓
ESCI 分数 + 最终 Top 10
```

为了控制显存和延迟，不能把整个商品库交给 VLM。图片和商品文本应当离线建立向量索引，VLM Ranker 只处理第一阶段召回的少量候选。训练两个 Adapter 时采用顺序训练，而不是同时训练。

## 八、实验设计

实验至少包含四组对照。第一组比较原始 query 与 reformulated query，验证 Query Understanding 是否提高召回效果。第二组比较纯文本排序和图文排序，验证商品图片是否提供额外价值。第三组比较随机负例和 hard negatives，验证困难样本构造的重要性。第四组比较规则排序、文本 cross-encoder 和 2B 多模态 Ranker，观察模型规模与效果、延迟之间的关系。

最终报告不能只给 nDCG。还应记录模型总参数、实际可训练参数、训练峰值显存、单轮训练时间、推理吞吐、P50/P95 延迟和不同候选数量下的性能变化。通过这些指标可以证明该方案不仅有效，而且能够在低资源环境中部署。

## 九、项目边界与扩展

项目第一阶段以英文 TREC/ESCI/SQID 数据为主，因为这些数据具有公开标签和标准评价流程。中文能力可以在基础系统稳定后，通过中文商品数据、翻译增强和少量人工标注进行迁移。中文测试集应独立建立，不能把机器翻译后的训练样本同时用于测试。

[CAMEO](https://cameo-task.com/) 可以作为额外泛化测试。它要求系统根据参考商品图片和文字修改描述检索目标商品，与“找和这张图类似，但换成黑色或更适合夏天”的交互方式高度一致。不过其商品数量较少，并且主要集中在越南语美妆领域，因此更适合作为扩展实验，而不是主训练集。

## 十、最终项目定义

本项目可以正式命名为：

> 面向任务型电商查询的轻量级多模态理解与商品重排系统：基于 TREC Query Reformulation 和 Amazon ESCI/SQID 的单卡训练实践。

项目最终交付包含两个 LoRA Adapter、一套 BM25 与多模态向量召回服务、一套 ESCI 精排服务、标准化离线评估脚本和可交互 Demo。其核心价值不在于简单调用大模型，而在于利用公开赛题完成数据构造、查询理解、多模态召回、Learning to Rank、消融实验和低资源部署的完整闭环。
