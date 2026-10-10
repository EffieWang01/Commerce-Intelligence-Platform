# EcomMMMU 查询规则基线评测

## 目的

验证当前低成本查询规则能否用于真实电商数据，并区分两类结果：

1. 查询理解能力：能否稳定提取价格、颜色、材质、属性和排除词等显式约束。
2. 下游相关性代理效果：只使用 query 和商品标题，能否预测 EcomMMMU 的 A/B/C/D 四级相关性。

EcomMMMU 没有提供 category、brand、attribute 等查询槽位的标准答案，因此本实验不能报告“查询理解准确率”。`query_constraint_coverage` 只表示规则在多少查询上抽取到了显式约束。

## 数据与设置

- 来源：[NingLab/EcomMMMU](https://huggingface.co/datasets/NingLab/EcomMMMU)
- 任务：`multiclass_product_classification`
- 切分：官方行字段中的 `test`
- 样本数：300
- 官方标签：A 完全相关、B 功能替代、C 互补商品、D 不相关
- 输入：query 与 product title
- 未使用：商品图片、LLM、训练参数

抽取的 300 行全部带有图片引用，其中 35.7% 标记为需要视觉上下文。这部分样本仍按文本规则处理，用来明确量化当前系统尚未利用的多模态信息。

复现命令：

```powershell
$env:PYTHONPATH='src'
python scripts\evaluate_ecommmmu_rules.py --sample-size 300 --split test --start-offset 390000
```

## 结果

| 指标 | 结果 |
| --- | ---: |
| 显式约束提取覆盖率 | 24.3% |
| Query 与标题至少一个有效词命中 | 88.3% |
| 四分类 Accuracy | 49.0% |
| 四分类 Macro-F1 | 35.0% |
| 多数类基线 Accuracy | 63.7% |
| 多数类基线 Macro-F1 | 19.5% |

标签分布为 A=191、B=76、C=8、D=25。规则预测分布为 A=100、B=125、C=19、D=56。

## 结论

规则适合继续作为查询理解的第一层：成本低、确定性强，能够处理明确的价格、颜色、属性和排除条件。它还不适合独立承担相关性排序。虽然 Macro-F1 高于只猜多数类的基线，说明规则能识别部分 B/D 样本，但总体 Accuracy 更低，且 8 个 C 类样本一个都未正确识别。

主要误差来自同义表达、功能等价、商品与配件关系，以及 query 和标题词面不同但语义相同的情况。这些是本次完整测评实际暴露的瓶颈，而不是预先假设出来的优化方向。

本次是文本规则测试，不是多模态测试。数据行保留了图片存在性与 `is_vc` 字段，但规则没有读取图片内容；后续多模态模型应在同一批固定样本上与该基线对比。
