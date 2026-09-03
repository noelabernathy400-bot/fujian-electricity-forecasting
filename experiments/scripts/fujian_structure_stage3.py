from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
STAGE1_WIDE = PROJECT / "Data" / "processed" / "structure_stage1" / "province_usage_daily_wide.csv"
GROUP_MAP = PROJECT / "Data" / "dictionary" / "structure_stage1" / "manufacturing_industry_group_map.csv"
OUT = PROJECT / "Experiments" / "outputs" / "structure_stage3"
FIG = PROJECT / "Experiments" / "figures" / "structure_stage3"

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
    present = group_map[group_map["present_in_data"] == True].copy()
    missing = [name for name in present["industry_name"] if name not in wide.columns]
    if missing:
        raise RuntimeError(f"分组表标记存在但宽表缺失这些行业：{missing}")
    if MANUFACTURING_TOTAL not in wide.columns:
        raise RuntimeError(f"宽表缺少制造业总量列：{MANUFACTURING_TOTAL}")
    return wide, present


def build_tables(wide: pd.DataFrame, group_map: pd.DataFrame) -> dict[str, pd.DataFrame]:
    industry_names = group_map["industry_name"].tolist()
    daily_industry = wide[industry_names].copy()
    daily_group = pd.DataFrame(index=wide.index)
    for group in GROUP_ORDER:
        names = group_map.loc[group_map["manufacturing_group"] == group, "industry_name"].tolist()
        daily_group[group] = daily_industry[names].sum(axis=1, min_count=1)
    daily_group.insert(0, "制造业", wide[MANUFACTURING_TOTAL])

    daily_share = daily_group[GROUP_ORDER].div(daily_group["制造业"], axis=0)
    daily_group["装备与先进制造合计"] = daily_group[ADVANCED_GROUPS].sum(axis=1)
    daily_share["装备与先进制造合计"] = daily_group["装备与先进制造合计"] / daily_group["制造业"]
    daily_share["结构转型差值T"] = daily_share["装备与先进制造合计"] - daily_share[HEAVY_GROUP]
    daily_share["结构转型比值R"] = daily_group["装备与先进制造合计"] / daily_group[HEAVY_GROUP].replace(0, np.nan)

    monthly_group = daily_group.resample("MS").sum()
    monthly_group = monthly_group.loc[monthly_group.index <= pd.Timestamp("2025-12-01")].copy()
    monthly_share = monthly_group[GROUP_ORDER].div(monthly_group["制造业"], axis=0)
    monthly_group["装备与先进制造合计"] = monthly_group[ADVANCED_GROUPS].sum(axis=1)
    monthly_share["装备与先进制造合计"] = monthly_group["装备与先进制造合计"] / monthly_group["制造业"]
    monthly_share["结构转型差值T"] = monthly_share["装备与先进制造合计"] - monthly_share[HEAVY_GROUP]
    monthly_share["结构转型比值R"] = monthly_group["装备与先进制造合计"] / monthly_group[HEAVY_GROUP].replace(0, np.nan)

    annual_group = daily_group.resample("YS").sum()
    annual_group = annual_group.loc[annual_group.index.year <= 2025].copy()
    annual_share = annual_group[GROUP_ORDER].div(annual_group["制造业"], axis=0)
    annual_group["装备与先进制造合计"] = annual_group[ADVANCED_GROUPS].sum(axis=1)
    annual_share["装备与先进制造合计"] = annual_group["装备与先进制造合计"] / annual_group["制造业"]
    annual_share["结构转型差值T"] = annual_share["装备与先进制造合计"] - annual_share[HEAVY_GROUP]
    annual_share["结构转型比值R"] = annual_group["装备与先进制造合计"] / annual_group[HEAVY_GROUP].replace(0, np.nan)

    annual_delta = annual_group.diff()
    annual_growth_contribution = annual_delta[GROUP_ORDER].div(annual_delta["制造业"], axis=0).dropna(how="all")

    start_year = 2022
    end_year = 2025
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-01-01")
    long_delta = annual_group.loc[end] - annual_group.loc[start]
    long_group_contribution = pd.DataFrame(
        {
            "制造业分组": GROUP_ORDER,
            "2022用电量": [annual_group.loc[start, group] for group in GROUP_ORDER],
            "2025用电量": [annual_group.loc[end, group] for group in GROUP_ORDER],
            "变化量": [long_delta[group] for group in GROUP_ORDER],
            "增长贡献率": [long_delta[group] / long_delta["制造业"] for group in GROUP_ORDER],
            "2022占制造业比重": [annual_share.loc[start, group] for group in GROUP_ORDER],
            "2025占制造业比重": [annual_share.loc[end, group] for group in GROUP_ORDER],
        }
    )

    annual_industry = daily_industry.resample("YS").sum()
    annual_industry = annual_industry.loc[annual_industry.index.year <= 2025].copy()
    industry_delta = annual_industry.loc[end] - annual_industry.loc[start]
    industry_contribution = pd.DataFrame(
        {
            "行业": industry_delta.index,
            "制造业分组": [group_map.set_index("industry_name").loc[name, "manufacturing_group"] for name in industry_delta.index],
            "2022用电量": annual_industry.loc[start].values,
            "2025用电量": annual_industry.loc[end].values,
            "变化量": industry_delta.values,
            "增长贡献率": (industry_delta / long_delta["制造业"]).values,
            "2022占制造业比重": (annual_industry.loc[start] / annual_group.loc[start, "制造业"]).values,
            "2025占制造业比重": (annual_industry.loc[end] / annual_group.loc[end, "制造业"]).values,
        }
    ).sort_values("增长贡献率", ascending=False)

    monthly_yoy_group_growth = monthly_group[GROUP_ORDER].pct_change(12)
    monthly_yoy_group_growth = monthly_yoy_group_growth.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    return {
        "daily_group": daily_group,
        "daily_share": daily_share,
        "monthly_group": monthly_group,
        "monthly_share": monthly_share,
        "annual_group": annual_group,
        "annual_share": annual_share,
        "annual_growth_contribution": annual_growth_contribution,
        "long_group_contribution": long_group_contribution,
        "industry_contribution": industry_contribution,
        "monthly_yoy_group_growth": monthly_yoy_group_growth,
    }


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")

    annual_share_display = tables["annual_share"].copy()
    annual_share_display.index = annual_share_display.index.year
    annual_share_display = annual_share_display.map(pct_fmt)
    write_markdown_table(annual_share_display.reset_index().rename(columns={"date": "年份", "index": "年份"}), OUT / "annual_group_share_table.md")

    group_contrib = tables["long_group_contribution"].copy()
    for col in ["2022用电量", "2025用电量", "变化量"]:
        group_contrib[col] = group_contrib[col].map(num_fmt)
    for col in ["增长贡献率", "2022占制造业比重", "2025占制造业比重"]:
        group_contrib[col] = group_contrib[col].map(pct_fmt)
    write_markdown_table(group_contrib, OUT / "long_group_contribution_table.md")

    industry_top = tables["industry_contribution"].head(10).copy()
    for col in ["2022用电量", "2025用电量", "变化量"]:
        industry_top[col] = industry_top[col].map(num_fmt)
    for col in ["增长贡献率", "2022占制造业比重", "2025占制造业比重"]:
        industry_top[col] = industry_top[col].map(pct_fmt)
    write_markdown_table(industry_top, OUT / "top10_industry_contribution_table.md")


def plot_annual_group_share(annual_share: pd.DataFrame) -> None:
    df = annual_share[GROUP_ORDER].copy()
    df.index = df.index.year
    ax = df.plot(kind="bar", stacked=True, figsize=(11, 5.8), width=0.72, colormap="Set3")
    ax.set_title("制造业五类分组年度占比")
    ax.set_xlabel("年份")
    ax.set_ylabel("占制造业用电比例")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIG / "stage3_annual_group_share.png", bbox_inches="tight")
    plt.close(fig)


def plot_monthly_group_share(monthly_share: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    colors = ["#D55E00", "#009E73", "#0072B2", "#CC79A7", "#999999"]
    for group, color in zip(GROUP_ORDER, colors):
        ax.plot(monthly_share.index, monthly_share[group], label=group, linewidth=1.7, color=color)
    ax.set_title("制造业分组月度占比变化")
    ax.set_xlabel("月份")
    ax.set_ylabel("占制造业用电比例")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stage3_monthly_group_share.png", bbox_inches="tight")
    plt.close(fig)


def plot_transition_indicators(monthly_share: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.plot(monthly_share.index, monthly_share["结构转型差值T"], color="#0072B2", linewidth=2.0, label="结构转型差值 T")
    ax1.axhline(0, color="#333333", linewidth=1, alpha=0.75)
    ax1.set_ylabel("T：装备与先进制造占比 - 传统高耗能占比")
    ax1.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(monthly_share.index, monthly_share["结构转型比值R"], color="#D55E00", linewidth=1.8, linestyle="--", label="结构转型比值 R")
    ax2.set_ylabel("R：装备与先进制造用电 / 传统高耗能用电")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax1.set_title("制造业结构转型指标月度变化")
    ax1.set_xlabel("月份")
    fig.tight_layout()
    fig.savefig(FIG / "stage3_transition_indicators.png", bbox_inches="tight")
    plt.close(fig)


def plot_long_group_contribution(contrib: pd.DataFrame) -> None:
    df = contrib.set_index("制造业分组")
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    colors = ["#D55E00", "#009E73", "#0072B2", "#CC79A7", "#999999"]
    ax.bar(df.index, df["增长贡献率"], color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("2022-2025 年制造业新增用电来源分解")
    ax.set_xlabel("制造业分组")
    ax.set_ylabel("对制造业新增用电贡献率")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(df["增长贡献率"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.015 if value >= 0 else -0.015
        ax.text(i, value + offset, f"{value:.1%}", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "stage3_2022_2025_group_contribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_industry_contribution(industry_contrib: pd.DataFrame) -> None:
    df = industry_contrib.head(10).sort_values("增长贡献率")
    fig, ax = plt.subplots(figsize=(11, 6.4))
    colors = df["制造业分组"].map({
        "传统高耗能与材料行业": "#D55E00",
        "消费与轻工制造": "#009E73",
        "装备制造": "#0072B2",
        "先进制造与技术相关行业": "#CC79A7",
        "其他制造": "#999999",
    })
    ax.barh(df["行业"], df["增长贡献率"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("2022-2025 年制造业新增用电贡献前十行业")
    ax.set_xlabel("对制造业新增用电贡献率")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "stage3_top10_industry_contribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_industry_share_change_scatter(industry_contrib: pd.DataFrame) -> None:
    df = industry_contrib.copy()
    df["占比变化"] = df["2025占制造业比重"] - df["2022占制造业比重"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    group_colors = {
        "传统高耗能与材料行业": "#D55E00",
        "消费与轻工制造": "#009E73",
        "装备制造": "#0072B2",
        "先进制造与技术相关行业": "#CC79A7",
        "其他制造": "#999999",
    }
    for group, sub in df.groupby("制造业分组"):
        ax.scatter(
            sub["2022占制造业比重"],
            sub["占比变化"],
            s=np.clip(np.abs(sub["增长贡献率"]) * 800, 28, 420),
            alpha=0.75,
            label=group,
            color=group_colors.get(group, "#666666"),
            edgecolor="white",
            linewidth=0.6,
        )
    label_df = pd.concat([df.nlargest(5, "增长贡献率"), df.nsmallest(3, "增长贡献率")]).drop_duplicates("行业")
    for _, row in label_df.iterrows():
        ax.text(row["2022占制造业比重"], row["占比变化"], row["行业"], fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("制造业行业初始占比与占比变化")
    ax.set_xlabel("2022 年占制造业比重")
    ax.set_ylabel("2025 占比 - 2022 占比")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stage3_industry_share_change_scatter.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(tables: dict[str, pd.DataFrame]) -> None:
    annual_share = tables["annual_share"].copy()
    annual_share.index = annual_share.index.year
    long_group = tables["long_group_contribution"].sort_values("增长贡献率", ascending=False).copy()
    industry = tables["industry_contribution"].sort_values("增长贡献率", ascending=False).copy()
    transition_2022 = annual_share.loc[2022, "结构转型差值T"]
    transition_2025 = annual_share.loc[2025, "结构转型差值T"]
    ratio_2022 = annual_share.loc[2022, "结构转型比值R"]
    ratio_2025 = annual_share.loc[2025, "结构转型比值R"]
    summary = {
        "annual_group_share_percent": {
            str(year): {col: round(float(value * 100), 4) for col, value in row.items() if col in GROUP_ORDER}
            for year, row in annual_share.iterrows()
        },
        "transition_indicator": {
            "T_2022": float(transition_2022),
            "T_2025": float(transition_2025),
            "T_change_2022_to_2025": float(transition_2025 - transition_2022),
            "R_2022": float(ratio_2022),
            "R_2025": float(ratio_2025),
            "R_change_2022_to_2025": float(ratio_2025 - ratio_2022),
        },
        "largest_group_growth_contributor_2022_2025": {
            "制造业分组": str(long_group.iloc[0]["制造业分组"]),
            "增长贡献率": float(long_group.iloc[0]["增长贡献率"]),
        },
        "top5_industry_growth_contributors_2022_2025": [
            {
                "行业": str(row["行业"]),
                "制造业分组": str(row["制造业分组"]),
                "增长贡献率": float(row["增长贡献率"]),
                "占比变化": float(row["2025占制造业比重"] - row["2022占制造业比重"]),
            }
            for _, row in industry.head(5).iterrows()
        ],
    }
    (OUT / "stage3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ensure_dirs()
    setup_plot_style()
    wide, group_map = read_inputs()
    tables = build_tables(wide, group_map)
    save_tables(tables)
    plot_annual_group_share(tables["annual_share"])
    plot_monthly_group_share(tables["monthly_share"])
    plot_transition_indicators(tables["monthly_share"])
    plot_long_group_contribution(tables["long_group_contribution"])
    plot_top_industry_contribution(tables["industry_contribution"])
    plot_industry_share_change_scatter(tables["industry_contribution"])
    write_summary(tables)


if __name__ == "__main__":
    main()
