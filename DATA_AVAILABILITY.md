# Data availability

## Included

- 原始字段、统计口径和派生表的字典：`data/dictionary/`。
- 数据清理、结构分析、预测与稳健性检验脚本：`experiments/scripts/`。
- 紧凑的结果 CSV/JSON/Markdown、可视化图表、研究过程与论文源。

## Not included

下列原始输入没有上传：

- `分行业&分用电类别(1).csv`（约 146 MB）；
- `福建全省用电量统计20230101至20260109.xlsx`；
- 根据原始输入派生、但仍包含可逐行还原电力记录的大型处理表。

这些文件受到数据来源、再分发权限和体积限制的共同约束。完整复现需要获得相同口径的原始数据；到位后置于 `data/raw/`，并根据 `experiments/scripts/fujian_structure_stage1.py` 中的输入约定调整路径或配置。

不含原始数据时，读者仍可审阅全部脚本、变量字典、冻结结果、图表与研究结论边界。
