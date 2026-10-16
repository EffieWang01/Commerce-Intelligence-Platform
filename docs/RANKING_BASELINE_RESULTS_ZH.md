# 第一版排序模型训练结果

本次在 `server44-moxu` 完成 LightGBM LambdaMART 训练，并通过真实 50,000 件商品目录的 API 加载验证。
训练使用 4 个 CPU 线程；两张 RTX 3090 未参与这次训练。

## 数据与划分

现有完整商品目录与 Amazon ESCI 的交集只有 405 条判断，其中仅 6 个 query 在交集内包含多个相关性等级。
因此本次训练使用独立的官方 ESCI US 文本子集，原有 50,000 件商品目录保留，未混入缺少图片的训练商品。

| 划分 | Query 数 | Query-product 标注数 |
| --- | ---: | ---: |
| Train | 2,898 | 54,336 |
| Validation | 482 | 9,213 |
| Test | 486 | 9,191 |

该子集涉及 68,683 件商品。商品必须有原始标题，以及原始描述或 bullet points。官方 test query 保留为测试集，
validation 从 train 按 query 哈希划分，规范化后各划分之间没有重复 query。官方原始文件经过 Git LFS SHA-256 校验。

## 指标

验证集早停选择 63 棵树，随机种子为 42。使用 ESCI gain：I=0、C=0.01、S=0.1、E=1。
MRR@10 将 E 视为精确相关结果。

| 划分 / 指标 | 原规则基线 | LambdaMART |
| --- | ---: | ---: |
| Validation NDCG@10 | 0.7042 | 0.7262 |
| Validation MRR@10 | 0.7280 | 0.7475 |
| Test NDCG@10 | 0.7013 | 0.7237 |
| Test MRR@10 | 0.7205 | 0.7472 |

这是已标注候选集合内的排序评估。规则基线复用现有排序器，其 BM25/hash 统计在每个已标注候选组内建立。
这些数字不代表完整 ESCI 榜单指标，也不代表当前 50,000 商品目录的端到端召回与排序质量。尚未做显著性检验。

本次模型使用词项覆盖、标题/描述/品牌匹配、长度等文本特征。ESCI 原始商品表没有价格、类目和图片；
相关特征在该训练集内没有贡献，未提取图像向量。本次结果属于文本排序基线，不是多模态模型结果。

## 工程验证

服务器全套测试：19 项通过。原 50,000 件商品的标题、类目、属性、价格、描述及图片 URL 缺失数均为 0。
图片完整性检查指 URL 字段存在，不代表已经下载或逐一验证全部远程图片可访问。

模型在真实商品目录上的 API 验证通过：耳机、键盘、跑鞋、指甲油四类查询均返回不同结果；
无匹配查询返回空集；health 显示实际使用 `lightgbm-lambdarank`。这项检查验证链路，不等于相关性质量评估。
首轮验证使用临时进程。后续本机接入通过 `scripts/run_local.ps1` 启动，模型存在时默认加载 LambdaMART，
页面侧栏显示实际排序器；仍可通过 `-Ranker Rules` 启动规则基线。

## 产物与复现

服务器所有文件集中在 `/home/moxu/commerce-searchrec`，当前占用约 2.2GB。
Python、虚拟环境、包缓存、模型缓存、数据、日志均在该目录内。未安装后台训练服务或定时任务。

- `artifacts/lambdamart/model.txt`：约 112KB 的模型。
- `artifacts/lambdamart/model.report.json`：完整指标、特征重要性、数据与模型校验和。
- `artifacts/lambdamart/api-smoke.json`：真实目录 API 验证记录。
- `artifacts/server-requirements.txt`：服务器实际依赖版本。
- `logs/lambdamart-train.log`：训练日志。

模型、评测报告和 API 验证结果已同步到本地项目的同名 `artifacts/lambdamart` 目录。
环境初始化与训练命令见 `docs/SERVER44_TRAINING.md`。模型通过 `COMMERCE_RANKING_MODEL_PATH` 配置加载；
未配置路径时仍使用规则基线。

## 本机接入验证（2026-10-16）

修复 Windows 中文路径下 LightGBM 原生文件接口加载失败的问题，改由 Python 读取模型文本后加载。
Agent 商品对比按目录内真实 item_id 解析，并去除重复 ID。

全套 20 项回归测试通过。另通过运行中的真实 HTTP API 验证五类查询、分数降序、价格和类目约束、
无匹配空结果，以及 Agent 搜索与真实商品 ID 对比。运行记录写入 `artifacts/runtime/e2e-report.json`。
页面展示查询解析、候选数量、实际排序器、结果分数和可展开的执行轨迹。图片服务未启用时上传入口禁用。

数据来源：[Amazon Science ESCI](https://github.com/amazon-science/esci-data)。
