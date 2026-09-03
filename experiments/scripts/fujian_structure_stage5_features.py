from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
STAGE1_WIDE = PROJECT / "Data" / "processed" / "structure_stage1" / "province_usage_daily_wide.csv"
GROUP_MAP = PROJECT / "Data" / "dictionary" / "structure_stage1" / "manufacturing_industry_group_map.csv"
STAGE3_INDUSTRY_CONTRIB = PROJECT / "Experiments" / "outputs" / "structure_stage3" / "industry_contribution.csv"
ROBUST_BEST = PROJECT / "Experiments" / "outputs" / "structure_stage4_robust" / "robust_best_by_industry.csv"
OUT = PROJECT / "Experiments" / "outputs" / "structure_stage5_features"
FIG = PROJECT / "Experiments" / "figures" / "structure_stage5_features"

MANUFACTURING_TOTAL = "（二） 制造业"
HEAVY_GROUP = "传统高耗能与材料行业"
ADVANCED_GROUPS = ["装备制造", "先进制造与技术相关行业"]
GROUP_ORDER = [
    "传统高耗能与材料行业",
    "消费与轻工制造",
    "装备制造",
    "先进制造与技术相关行业",
    "其他制造",
]
GROUP_COLORS = {
    "传统高耗能与材料行业": "#D55E00",
    "消费与轻工制造": "#009E73",
    "装备制造": "#0072B2",
    "先进制造与技术相关行业": "#CC79A7",
    "其他制造": "#666666",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def pct_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def num_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.2f}"


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(STAGE1_WIDE, encoding="utf-8-sig", parse_dates=["date"]).set_index("date").sort_index()
    group_map = pd.read_csv(GROUP_MAP, encoding="utf-8-sig")
    group_map = group_map[group_map["present_in_data"] == True].copy()
    missing = [name for name in group_map["industry_name"] if name not in wide.columns]
    if missing:
        raise RuntimeError(f"宽表缺少制造业行业：{missing}")
    if MANUFACTURING_TOTAL not in wide.columns:
        raise RuntimeError(f"宽表缺少制造业总量列：{MANUFACTURING_TOTAL}")
    return wide, group_map


def build_monthly_matrices(wide: pd.DataFrame, group_map: pd.DataFrame) -> dict[str, pd.DataFrame]:
    industry_names = group_map["industry_name"].tolist()
    monthly_industry = wide[industry_names].resample("MS").sum()
    monthly_industry = monthly_industry.loc[monthly_industry.index <= pd.Timestamp("2025-12-01")].copy()
    monthly_total = wide[MANUFACTURING_TOTAL].resample("MS").sum().loc[monthly_industry.index]
    monthly_share = monthly_industry.div(monthly_total, axis=0)
    monthly_yoy = monthly_industry.pct_change(12).replace([np.inf, -np.inf], np.nan)

    monthly_group = pd.DataFrame(index=monthly_industry.index)
    for group in GROUP_ORDER:
        names = group_map.loc[group_map["manufacturing_group"] == group, "industry_name"].tolist()
        monthly_group[group] = monthly_industry[names].sum(axis=1)
    monthly_group.insert(0, "制造业", monthly_total)
    monthly_group_share = monthly_group[GROUP_ORDER].div(monthly_group["制造业"], axis=0)
    monthly_group_yoy = monthly_group[GROUP_ORDER + ["制造业"]].pct_change(12).replace([np.inf, -np.inf], np.nan)

    return {
        "monthly_industry": monthly_industry,
        "monthly_industry_share": monthly_share,
        "monthly_industry_yoy": monthly_yoy,
        "monthly_group": monthly_group,
        "monthly_group_share": monthly_group_share,
        "monthly_group_yoy": monthly_group_yoy,
    }


def entropy(row: pd.Series) -> float:
    values = row.dropna().astype(float)
    values = values[values > 0]
    if len(values) == 0:
        return np.nan
    return float(-(values * np.log(values)).sum())


def classify_month(row: pd.Series) -> str:
    total_delta = row["制造业同比变化量"]
    abs_total = row["制造业同比变化量绝对值"]
    if pd.isna(total_delta) or pd.isna(abs_total):
        return "同比信息不足"
    if abs_total <= row["低变化阈值"]:
        return "低变化波动型"
    if total_delta < 0:
        if row["结构转型差值T同比变化"] > 0:
            return "收缩中的结构优化型"
        return "制造业回落型"

    heavy_pos = row["传统高耗能与材料行业正向贡献占比"]
    equip_adv_pos = row["装备与先进制造正向贡献占比"]
    light_pos = row["消费与轻工制造正向贡献占比"]
    t_change = row["结构转型差值T同比变化"]

    if heavy_pos >= 0.45 and equip_adv_pos < 0.35:
        return "传统高耗能扩张主导型"
    if equip_adv_pos >= 0.35 and t_change > 0:
        return "装备与先进制造增强型"
    if light_pos >= 0.35:
        return "消费轻工扩张主导型"
    if heavy_pos >= 0.25 and equip_adv_pos >= 0.25:
        return "多组共同扩张型"
    return "其他扩张型"


def build_structure_features(matrices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    share = matrices["monthly_industry_share"]
    group = matrices["monthly_group"]
    group_share = matrices["monthly_group_share"]

    features = pd.DataFrame(index=share.index)
    features["制造业用电量"] = group["制造业"]
    features["制造业同比增长率"] = group["制造业"].pct_change(12)
    features["制造业同比变化量"] = group["制造业"] - group["制造业"].shift(12)
    features["制造业同比变化量绝对值"] = features["制造业同比变化量"].abs()
    features["低变化阈值"] = features["制造业同比变化量绝对值"].dropna().quantile(0.2)

    features["HHI集中度"] = (share**2).sum(axis=1)
    features["有效行业数"] = 1 / features["HHI集中度"].replace(0, np.nan)
    features["结构熵"] = share.apply(entropy, axis=1)
    features["结构熵标准化"] = features["结构熵"] / np.log(share.shape[1])
    features["前5行业占比"] = share.apply(lambda row: row.sort_values(ascending=False).head(5).sum(), axis=1)
    features["最大行业占比"] = share.max(axis=1)

    share_lag12 = share.shift(12)
    diff = share - share_lag12
    features["行业占比L1距离_同比"] = diff.abs().sum(axis=1)
    features["行业占比L2距离_同比"] = np.sqrt((diff**2).sum(axis=1))
    features["行业占比最大变化_同比"] = diff.abs().max(axis=1)

    for group_name in GROUP_ORDER:
        features[f"{group_name}占比"] = group_share[group_name]
        features[f"{group_name}同比增长率"] = group[group_name].pct_change(12)
        features[f"{group_name}同比变化量"] = group[group_name] - group[group_name].shift(12)

    features["装备与先进制造合计占比"] = group_share[ADVANCED_GROUPS].sum(axis=1)
    features["结构转型差值T"] = features["装备与先进制造合计占比"] - features[f"{HEAVY_GROUP}占比"]
    features["结构转型比值R"] = group[ADVANCED_GROUPS].sum(axis=1) / group[HEAVY_GROUP].replace(0, np.nan)
    features["结构转型差值T同比变化"] = features["结构转型差值T"] - features["结构转型差值T"].shift(12)
    features["结构转型比值R同比变化"] = features["结构转型比值R"] - features["结构转型比值R"].shift(12)

    group_delta_cols = [f"{group_name}同比变化量" for group_name in GROUP_ORDER]
    pos_delta = features[group_delta_cols].clip(lower=0)
    pos_sum = pos_delta.sum(axis=1).replace(0, np.nan)
    pos_contrib = pd.DataFrame(index=features.index)
    for group_name in GROUP_ORDER:
        col = f"{group_name}正向贡献占比"
        pos_contrib[col] = pos_delta[f"{group_name}同比变化量"] / pos_sum
        features[col] = pos_contrib[col]
    features["装备与先进制造正向贡献占比"] = features["装备制造正向贡献占比"].fillna(0) + features["先进制造与技术相关行业正向贡献占比"].fillna(0)
    features["结构状态"] = features.apply(classify_month, axis=1)
    features.loc[features.index < features.index.min() + pd.DateOffset(months=12), "结构状态"] = "同比信息不足"

    return features, pos_contrib


def read_optional_contribution() -> pd.DataFrame | None:
    if not STAGE3_INDUSTRY_CONTRIB.exists():
        return None
    df = pd.read_csv(STAGE3_INDUSTRY_CONTRIB, encoding="utf-8-sig")
    if "行业" not in df.columns and "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "行业"})
    return df


def read_optional_robust() -> pd.DataFrame | None:
    if not ROBUST_BEST.exists():
        return None
    return pd.read_csv(ROBUST_BEST, encoding="utf-8-sig")


def build_industry_profile(
    matrices: dict[str, pd.DataFrame],
    group_map: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    monthly = matrices["monthly_industry"]
    share = matrices["monthly_industry_share"]
    yoy = matrices["monthly_industry_yoy"]
    manufacturing_yoy = features["制造业同比增长率"]
    group_lookup = group_map.set_index("industry_name")["manufacturing_group"].to_dict()

    rows = []
    for industry in monthly.columns:
        pair = pd.concat([yoy[industry], manufacturing_yoy], axis=1).dropna()
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= 12 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0 else np.nan
        rows.append(
            {
                "行业": industry,
                "制造业分组": group_lookup[industry],
                "平均占比": share[industry].mean(),
                "2025平均占比": share.loc[share.index.year == 2025, industry].mean(),
                "最大月度占比": share[industry].max(),
                "同比增长率均值": yoy[industry].mean(),
                "同比增长率标准差": yoy[industry].std(),
                "与制造业同比相关": corr,
            }
        )
    profile = pd.DataFrame(rows)

    contribution = read_optional_contribution()
    if contribution is not None and "行业" in contribution.columns:
        keep = ["行业", "增长贡献率", "2022占制造业比重", "2025占制造业比重"]
        profile = profile.merge(contribution[[col for col in keep if col in contribution.columns]], on="行业", how="left")

    robust = read_optional_robust()
    if robust is not None and "行业" in robust.columns:
        robust_keep = robust[["目标变量", "行业", "证据等级", "检验滞后阶数", "q_value", "mae_gain"]].copy()
        robust_keep = robust_keep.sort_values(["行业", "证据等级", "mae_gain"], ascending=[True, True, False])
        robust_text = robust_keep.groupby("行业").apply(
            lambda df: "；".join(
                f"{row['目标变量']}:{row['证据等级']},lag={int(row['检验滞后阶数'])},q={row['q_value']:.4f},gain={row['mae_gain']:.2%}"
                for _, row in df.head(3).iterrows()
            )
        )
        profile = profile.merge(robust_text.rename("前置信号摘要").reset_index(), on="行业", how="left")

    share_q75 = profile["2025平均占比"].quantile(0.75)
    contrib_q75 = profile["增长贡献率"].quantile(0.75) if "增长贡献率" in profile.columns else np.nan
    vol_q75 = profile["同比增长率标准差"].quantile(0.75)

    def data_role(row: pd.Series) -> str:
        roles = []
        if row["2025平均占比"] >= share_q75:
            roles.append("存量主导")
        if "增长贡献率" in row and pd.notna(row["增长贡献率"]) and row["增长贡献率"] >= contrib_q75:
            roles.append("增长拉动")
        if row["同比增长率标准差"] >= vol_q75:
            roles.append("高波动")
        if not roles:
            roles.append("一般结构项")
        return "、".join(roles)

    profile["数据行为标签"] = profile.apply(data_role, axis=1)
    profile = profile.sort_values(["2025平均占比", "增长贡献率" if "增长贡献率" in profile.columns else "平均占比"], ascending=False)
    return profile


def build_correlation_network(matrices: dict[str, pd.DataFrame], group_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    yoy = matrices["monthly_industry_yoy"].dropna(how="all")
    corr = yoy.corr()
    group_lookup = group_map.set_index("industry_name")["manufacturing_group"].to_dict()
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "行业A": left,
                    "行业B": right,
                    "分组A": group_lookup[left],
                    "分组B": group_lookup[right],
                    "同期同比相关系数": value,
                    "相关绝对值": abs(value),
                    "是否跨分组": group_lookup[left] != group_lookup[right],
                }
            )
    edges = pd.DataFrame(rows).sort_values("相关绝对值", ascending=False)
    return corr, edges


def save_outputs(
    matrices: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    pos_contrib: pd.DataFrame,
    profile: pd.DataFrame,
    corr: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    for name, df in matrices.items():
        df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")
    features.to_csv(OUT / "monthly_structure_features.csv", encoding="utf-8-sig")
    pos_contrib.to_csv(OUT / "monthly_group_positive_contribution_share.csv", encoding="utf-8-sig")
    profile.to_csv(OUT / "industry_structure_profile.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT / "industry_yoy_correlation_matrix.csv", encoding="utf-8-sig")
    edges.to_csv(OUT / "industry_yoy_correlation_edges.csv", index=False, encoding="utf-8-sig")
    edges.head(40).to_csv(OUT / "industry_yoy_correlation_edges_top40.csv", index=False, encoding="utf-8-sig")

    display_profile = profile.head(12).copy()
    for col in ["平均占比", "2025平均占比", "最大月度占比", "增长贡献率", "2022占制造业比重", "2025占制造业比重"]:
        if col in display_profile.columns:
            display_profile[col] = display_profile[col].map(pct_fmt)
    for col in ["同比增长率均值", "同比增长率标准差", "与制造业同比相关"]:
        if col in display_profile.columns:
            display_profile[col] = display_profile[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    write_markdown_table(display_profile, OUT / "top12_industry_structure_profile.md")

    state_counts = (
        features.loc[features["结构状态"] != "同比信息不足", "结构状态"]
        .value_counts()
        .rename_axis("结构状态")
        .reset_index(name="月份数")
    )
    state_counts["占可判定月份比例"] = state_counts["月份数"] / state_counts["月份数"].sum()
    state_display = state_counts.copy()
    state_display["占可判定月份比例"] = state_display["占可判定月份比例"].map(pct_fmt)
    write_markdown_table(state_display, OUT / "structure_state_counts.md")

    dictionary = """# 第五阶段结构特征字典

## 原始矩阵

- `monthly_industry.csv`：月度行业用电量矩阵，行是月份，列是 31 个制造业细分行业。
- `monthly_industry_share.csv`：行业占制造业总用电的月度占比矩阵。
- `monthly_industry_yoy.csv`：行业月度同比增长率矩阵。

## 规模结构变量

- `HHI集中度`：行业占比平方和，越大说明制造业用电越集中在少数行业。
- `有效行业数`：`1 / HHI`，可理解为等效主导行业数量。
- `结构熵标准化`：行业占比分散程度，越接近 1 越分散。
- `前5行业占比`：前 5 个行业占制造业总用电比例。

## 变化结构变量

- `制造业同比增长率`：制造业总用电相对去年同月的增长率。
- `行业占比L1距离_同比`：本月行业占比向量与去年同月占比向量的绝对差之和。
- `行业占比L2距离_同比`：本月行业占比向量与去年同月占比向量的欧氏距离。
- `行业占比最大变化_同比`：所有行业占比变化中绝对值最大的变化。

## 转型结构变量

- `结构转型差值T`：装备制造与先进制造合计占比减去传统高耗能与材料行业占比。
- `结构转型比值R`：装备制造与先进制造合计用电量除以传统高耗能与材料行业用电量。
- `结构转型差值T同比变化`、`结构转型比值R同比变化`：相对去年同月的变化。

## 状态结构变量

- `结构状态`：基于制造业同比变化、各分组正向贡献占比和 T 的同比变化给月份打标签。

## 协同结构变量

- `industry_yoy_correlation_matrix.csv`：31 个行业同比增长率同期相关矩阵。
- `industry_yoy_correlation_edges_top40.csv`：同期相关绝对值最高的行业对。

## 使用边界

- 这些变量来自原始行业用电矩阵，不是外部编造指标。
- 规模、变化、转型和状态变量可以进入预测模型。
- 协同相关和前置信号只能写成关联或时序领先证据，不能直接写成强因果。
"""
    (OUT / "structure_feature_dictionary.md").write_text(dictionary, encoding="utf-8")


def plot_feature_system() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.axis("off")
    boxes = [
        (0.05, 0.72, "原始矩阵\n时间 × 行业用电量\nE(t,i)", "#F2F2F2"),
        (0.34, 0.82, "规模结构\n占比、HHI、熵、前5占比", "#DCEBFA"),
        (0.34, 0.60, "变化结构\n同比、增长贡献、占比距离", "#E5F2E6"),
        (0.62, 0.82, "转型结构\nT、R、T/R同比变化", "#FBE5D6"),
        (0.62, 0.60, "协同结构\n相关网络、前置信号", "#EEE2F8"),
        (0.62, 0.38, "状态结构\n扩张主导、转型增强、回落", "#FFF2CC"),
        (0.34, 0.18, "预测与解释\n进入模型或作为解释证据", "#EDEDED"),
    ]
    for x, y, text, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="#444444", linewidth=1.2),
        )
    arrows = [
        ((0.16, 0.72), (0.28, 0.82)),
        ((0.16, 0.72), (0.28, 0.60)),
        ((0.46, 0.82), (0.55, 0.82)),
        ((0.46, 0.60), (0.55, 0.60)),
        ((0.74, 0.60), (0.74, 0.46)),
        ((0.62, 0.38), (0.46, 0.22)),
        ((0.62, 0.82), (0.46, 0.22)),
        ((0.62, 0.60), (0.46, 0.22)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#444444"))
    ax.set_title("行业结构信息构建框架", fontsize=15, pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "stage5_structure_feature_system.png", bbox_inches="tight")
    plt.close(fig)


def plot_hhi_entropy(features: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5.4))
    ax1.plot(features.index, features["HHI集中度"], label="HHI集中度", color="#0072B2", linewidth=2)
    ax1.set_ylabel("HHI集中度")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(features.index, features["结构熵标准化"], label="结构熵标准化", color="#D55E00", linestyle="--", linewidth=2)
    ax2.set_ylabel("结构熵标准化")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax1.set_title("制造业行业集中度与分散度")
    ax1.set_xlabel("月份")
    fig.tight_layout()
    fig.savefig(FIG / "stage5_hhi_entropy.png", bbox_inches="tight")
    plt.close(fig)


def plot_structure_distance(features: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(features.index, features["行业占比L1距离_同比"], label="行业占比 L1 距离", color="#0072B2", linewidth=2)
    ax.plot(features.index, features["行业占比L2距离_同比"], label="行业占比 L2 距离", color="#D55E00", linewidth=2)
    ax.set_title("行业占比结构相对去年同月的变化距离")
    ax.set_xlabel("月份")
    ax.set_ylabel("结构距离")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stage5_structure_distance.png", bbox_inches="tight")
    plt.close(fig)


def plot_state_timeline(features: pd.DataFrame) -> None:
    valid = features.loc[features["结构状态"] != "同比信息不足"].copy()
    states = valid["结构状态"].drop_duplicates().tolist()
    color_map = {
        state: color
        for state, color in zip(
            states,
            ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999"],
        )
    }
    fig, ax = plt.subplots(figsize=(12, 3.2))
    for idx, (date, row) in enumerate(valid.iterrows()):
        ax.bar(date, 1, width=24, color=color_map[row["结构状态"]], align="center")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("制造业月度结构状态标签")
    ax.set_xlabel("月份")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[state]) for state in states]
    ax.legend(handles, states, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stage5_structure_state_timeline.png", bbox_inches="tight")
    plt.close(fig)


def plot_group_positive_contribution(pos_contrib: pd.DataFrame) -> None:
    data = pos_contrib[[f"{group}正向贡献占比" for group in GROUP_ORDER]].copy()
    data.columns = GROUP_ORDER
    data = data.dropna(how="all")
    fig, ax = plt.subplots(figsize=(12, 5.4))
    im = ax.imshow(data.T.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels(GROUP_ORDER)
    tick_idx = np.linspace(0, len(data.index) - 1, min(8, len(data.index))).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([data.index[i].strftime("%Y-%m") for i in tick_idx], rotation=35, ha="right")
    ax.set_title("各分组对制造业同比正向增量的贡献占比")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="正向贡献占比")
    fig.tight_layout()
    fig.savefig(FIG / "stage5_group_positive_contribution_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("31类制造业行业同比增长率同期相关矩阵")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels([c.split(".")[0] for c in corr.columns], rotation=90, fontsize=8)
    ax.set_yticklabels([c.split(".")[0] for c in corr.index], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="相关系数")
    fig.tight_layout()
    fig.savefig(FIG / "stage5_industry_yoy_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_industry_profile(profile: pd.DataFrame) -> None:
    df = profile.copy()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for group, sub in df.groupby("制造业分组"):
        ax.scatter(
            sub["2025平均占比"],
            sub["增长贡献率"],
            s=np.clip(sub["同比增长率标准差"].fillna(0) * 400, 30, 450),
            color=GROUP_COLORS.get(group, "#666666"),
            alpha=0.75,
            edgecolor="white",
            linewidth=0.6,
            label=group,
        )
    label_df = pd.concat([df.nlargest(5, "2025平均占比"), df.nlargest(5, "增长贡献率")]).drop_duplicates("行业")
    for _, row in label_df.iterrows():
        ax.text(row["2025平均占比"], row["增长贡献率"], row["行业"], fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("行业结构画像：存量占比、增长贡献与波动")
    ax.set_xlabel("2025 年平均占制造业比重")
    ax.set_ylabel("2022-2025 增长贡献率")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stage5_industry_structure_profile_scatter.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(features: pd.DataFrame, profile: pd.DataFrame, edges: pd.DataFrame) -> None:
    valid_states = features.loc[features["结构状态"] != "同比信息不足", "结构状态"]
    state_counts = valid_states.value_counts()
    latest = features.iloc[-1]
    top_profile = profile.head(5)
    summary = {
        "date_range": {
            "start": str(features.index.min().date()),
            "end": str(features.index.max().date()),
            "months": int(len(features)),
        },
        "latest_month": {
            "month": str(features.index.max().date()),
            "HHI集中度": float(latest["HHI集中度"]),
            "有效行业数": float(latest["有效行业数"]),
            "前5行业占比": float(latest["前5行业占比"]),
            "结构转型差值T": float(latest["结构转型差值T"]),
            "结构转型比值R": float(latest["结构转型比值R"]),
            "结构状态": str(latest["结构状态"]),
        },
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "top5_2025_share_industries": [
            {
                "行业": str(row["行业"]),
                "制造业分组": str(row["制造业分组"]),
                "2025平均占比": float(row["2025平均占比"]),
                "增长贡献率": float(row["增长贡献率"]) if "增长贡献率" in row and pd.notna(row["增长贡献率"]) else None,
                "数据行为标签": str(row["数据行为标签"]),
            }
            for _, row in top_profile.iterrows()
        ],
        "top10_corr_edges_abs_mean": float(edges.head(10)["相关绝对值"].mean()),
        "cross_group_edges_top40": int(edges.head(40)["是否跨分组"].sum()),
    }
    (OUT / "stage5_structure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ensure_dirs()
    setup_plot_style()
    wide, group_map = read_inputs()
    matrices = build_monthly_matrices(wide, group_map)
    features, pos_contrib = build_structure_features(matrices)
    profile = build_industry_profile(matrices, group_map, features)
    corr, edges = build_correlation_network(matrices, group_map)

    save_outputs(matrices, features, pos_contrib, profile, corr, edges)
    plot_feature_system()
    plot_hhi_entropy(features)
    plot_structure_distance(features)
    plot_state_timeline(features)
    plot_group_positive_contribution(pos_contrib)
    plot_corr_heatmap(corr)
    plot_industry_profile(profile)
    write_summary(features, profile, edges)


if __name__ == "__main__":
    main()
