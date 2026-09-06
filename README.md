# 福建用电结构转型与解释型预测

面向国创项目的可审阅研究仓库。项目以福建省日度分行业、分用电类别数据为对象，研究三个彼此区分的问题：

1. 用电数据能否在统一统计口径下被可靠处理和复现？
2. 制造业内部是否存在可量化的结构变化与前置信号？
3. 结构信息能否在严格时间顺序回测中提升预测，或至少提供可解释的监测指标？

本仓库保留代码、衍生字典、结果表、图表、研究过程主文档与 LaTeX 论文源；原始电力数据不公开，以避免擅自再分发。

## 论文

- [NCILG English PDF](papers/NCILG_EN.pdf)
- [NCILG 中文 PDF](papers/NCILG_CN.pdf)

两份论文 PDF 已从独立的 `papers` 仓库归档至此，评阅者可在同一项目仓库中查看论文、代码和证据。

## 给评阅者的阅读路线

| 想了解什么 | 从这里开始 |
| --- | --- |
| 项目问题、方法、图表与当前结论 | [`writing/研究过程主文档.md`](writing/研究过程主文档.md) |
| 研究范围与数据口径 | [`docs/Overview.md`](docs/Overview.md) |
| 当前阶段和下一步 | [`docs/Status.md`](docs/Status.md) |
| 关键研究决定与证据边界 | [`docs/Decision Log.md`](docs/Decision Log.md) |
| 数据字段与派生表清单 | `data/dictionary/` |
| 可执行分析脚本 | `experiments/scripts/` |
| 已记录的图表和数值输出 | `experiments/figures/`、`experiments/outputs/` |
| 阶段性论文源与 PDF | `writing/LaTeX/` |

## 已完成的证据链

- 口径审计和闭合验证：确认全社会总量及制造业细分结构可在项目主数据中闭合。
- 总量与制造业结构分解：保留占比、增长贡献和季节性结果。
- 结构转型信号：用分组占比、结构差值/比值与行业结构特征描述变化，避免把它写成产业升级已经完成。
- 前置信号筛选：保留滞后相关、稳健性和预测增益检验；其解释严格限于预测领先，**不等同于因果关系**。
- 预测基线与长假分段改进：保留单步、递推和直接多步评估，以及节假日误差分解。

## 快速环境

```powershell
python -m pip install -r requirements.txt
```

主要脚本仅使用 `numpy`、`pandas`、`matplotlib` 与 `scipy`。原始数据到位后，请按顺序运行：

```powershell
python experiments/scripts/fujian_structure_stage1.py
python experiments/scripts/fujian_structure_stage2.py
python experiments/scripts/fujian_structure_stage3.py
python experiments/scripts/fujian_structure_stage4_robust.py
python experiments/scripts/fujian_structure_stage5_features.py
```

这些脚本会从原始输入建立处理表、输出表和图表。现有 `experiments/outputs/` 与 `experiments/figures/` 是已记录的研究证据，便于在未取得原始数据时审阅工作过程。

## 结论边界

本项目目前支持“结构信号”和“预测贡献”的审慎结论，不支持从观察性时间序列直接得出强产业因果结论。天气变量的历史实况实验也只能作为诊断性回测，不能替代真实部署时的天气预报或情景输入。详见主文档和决策记录。

## 数据可用性

原始 CSV 与 Excel 含有受限的大体积电力统计数据，未提交到 Git。仓库提供字段字典、数据处理脚本、衍生结果和图表，使处理路径和主要结论仍可审计。请见 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)。
