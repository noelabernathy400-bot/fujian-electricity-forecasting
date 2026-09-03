from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
STAGE1_DATA = PROJECT / "Data" / "processed" / "structure_stage1" / "province_usage_daily_wide.csv"
OUT = PROJECT / "Experiments" / "outputs" / "structure_stage2"
FIG = PROJECT / "Experiments" / "figures" / "structure_stage2"

TOTAL = "全社会用电总计"
STRUCTURE = ["第一产业", "第二产业", "第三产业", "B、城乡居民生活用电合计"]
STRUCTURE_CLEAN = {
    "第一产业": "第一产业",
    "第二产业": "第二产业",
    "第三产业": "第三产业",
    "B、城乡居民生活用电合计": "居民生活",
}
MANUFACTURING = "（二） 制造业"


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
    return f"{x * 100:.2f}%"


def read_stage1_wide() -> pd.DataFrame:
    if not STAGE1_DATA.exists():
        raise FileNotFoundError(f"缺少第一阶段清洗数据：{STAGE1_DATA}")
    wide = pd.read_csv(STAGE1_DATA, encoding="utf-8-sig", parse_dates=["date"])
    wide = wide.set_index("date").sort_index()
    needed = [TOTAL, *STRUCTURE, MANUFACTURING]
    missing = [col for col in needed if col not in wide.columns]
    if missing:
        raise RuntimeError(f"第二阶段需要的列缺失：{missing}")
    return wide


def build_stage2_tables(wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily = wide[[TOTAL, *STRUCTURE, MANUFACTURING]].rename(columns=STRUCTURE_CLEAN).copy()
    daily = daily.rename(columns={MANUFACTURING: "制造业"})

    monthly_total = daily.resample("MS").sum()
    annual_total = daily.resample("YS").sum()
    annual_total = annual_total.loc[annual_total.index.year <= 2025].copy()

    share_cols = ["第一产业", "第二产业", "第三产业", "居民生活", "制造业"]
    monthly_share = monthly_total[share_cols].div(monthly_total[TOTAL], axis=0)
    annual_share = annual_total[share_cols].div(annual_total[TOTAL], axis=0)

    annual_delta = annual_total.diff()
    contribution_cols = ["第一产业", "第二产业", "第三产业", "居民生活"]
    annual_growth_contribution = annual_delta[contribution_cols].div(annual_delta[TOTAL], axis=0)
    annual_growth_contribution = annual_growth_contribution.dropna(how="all")

    # 同比月度增长贡献：用本月相对去年同月的变化解释总量同比变化。
    monthly_delta_yoy = monthly_total.diff(12)
    monthly_yoy_contribution = monthly_delta_yoy[contribution_cols].div(monthly_delta_yoy[TOTAL], axis=0)
    monthly_yoy_contribution = monthly_yoy_contribution.dropna(how="all")

    # 2022 到 2025 的长期变化贡献，用于回答“研究期内新增用电主要来自哪里”。
    start_year = 2022
    end_year = 2025
    long_delta = annual_total.loc[pd.Timestamp(f"{end_year}-01-01")] - annual_total.loc[pd.Timestamp(f"{start_year}-01-01")]
    long_growth_contribution = pd.DataFrame(
        {
            "结构项": contribution_cols,
            "2022用电量": [annual_total.loc[pd.Timestamp(f"{start_year}-01-01"), col] for col in contribution_cols],
            "2025用电量": [annual_total.loc[pd.Timestamp(f"{end_year}-01-01"), col] for col in contribution_cols],
            "变化量": [long_delta[col] for col in contribution_cols],
            "增长贡献率": [long_delta[col] / long_delta[TOTAL] for col in contribution_cols],
            "2022占比": [annual_share.loc[pd.Timestamp(f"{start_year}-01-01"), col] for col in contribution_cols],
            "2025占比": [annual_share.loc[pd.Timestamp(f"{end_year}-01-01"), col] for col in contribution_cols],
        }
    )

    return {
        "daily_core": daily,
        "monthly_total": monthly_total,
        "annual_total": annual_total,
        "monthly_share": monthly_share,
        "annual_share": annual_share,
        "annual_growth_contribution": annual_growth_contribution,
        "monthly_yoy_contribution": monthly_yoy_contribution,
        "long_growth_contribution": long_growth_contribution,
    }


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")

    annual_share_display = tables["annual_share"].copy()
    annual_share_display.index = annual_share_display.index.year
    annual_share_display = annual_share_display.map(pct_fmt)
    write_markdown_table(annual_share_display.reset_index().rename(columns={"date": "年份", "index": "年份"}), OUT / "annual_share_table.md")

    contrib_display = tables["long_growth_contribution"].copy()
    for col in ["2022用电量", "2025用电量", "变化量"]:
        contrib_display[col] = contrib_display[col].map(lambda x: f"{x:,.2f}")
    for col in ["增长贡献率", "2022占比", "2025占比"]:
        contrib_display[col] = contrib_display[col].map(pct_fmt)
    write_markdown_table(contrib_display, OUT / "long_growth_contribution_table.md")


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_process_flow() -> None:
    steps = [
        ("1 数据口径审计", "确定 CSV 是主数据\nExcel 只作辅助核对"),
        ("2 闭合验证", "总量=四大结构项\n制造业=31类细分行业"),
        ("3 总量结构分解", "占比、趋势、增长贡献"),
        ("4 制造业结构预验证", "分组占比、转型差值\n转型比值、贡献象限"),
        ("5 前置信号识别", "滞后相关、Granger\n滚动稳定性、预测增益"),
        ("6 解释型预测", "比较基线模型和结构增强模型\n输出误差与解释"),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(steps))
    colors = ["#2E86AB", "#4B8F8C", "#77A649", "#C49A3A", "#B56576", "#6D597A"]
    for i, ((title, body), x, color) in enumerate(zip(steps, xs, colors)):
        ax.text(
            x,
            0.58,
            f"{title}\n{body}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.08", facecolor=color, edgecolor="none"),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.07, 0.58),
                xytext=(x + 0.07, 0.58),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#555555"),
            )
    ax.text(0.5, 0.9, "本项目研究流程：先验证数据结构，再进入解释型预测", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.16, "读图方式：每一步的输出都会成为下一步的输入；若某一步证据不足，后续结论必须降级。", ha="center", fontsize=10, color="#333333")
    fig.savefig(FIG / "stage2_research_process_flow.png", bbox_inches="tight")
    plt.close(fig)


def plot_annual_share(annual_share: pd.DataFrame) -> None:
    four = annual_share[["第一产业", "第二产业", "第三产业", "居民生活"]].copy()
    four.index = four.index.year
    ax = four.plot(kind="bar", stacked=True, figsize=(10, 5.8), width=0.72, colormap="Set2")
    ax.plot(range(len(annual_share)), annual_share["制造业"].values, color="#222222", marker="o", linewidth=2.0, label="制造业占全社会")
    ax.set_title("福建省全社会用电年度结构占比")
    ax.set_xlabel("年份")
    ax.set_ylabel("占全社会用电总量比例")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIG / "stage2_annual_structure_share.png", bbox_inches="tight")
    plt.close(fig)


def plot_monthly_share(monthly_share: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    colors = {
        "第一产业": "#4C78A8",
        "第二产业": "#F58518",
        "第三产业": "#54A24B",
        "居民生活": "#B279A2",
        "制造业": "#222222",
    }
    for col in ["第一产业", "第二产业", "第三产业", "居民生活", "制造业"]:
        style = "--" if col == "制造业" else "-"
        width = 2.0 if col == "制造业" else 1.6
        ax.plot(monthly_share.index, monthly_share[col], label=col, color=colors[col], linewidth=width, linestyle=style)
    ax.set_title("福建省用电结构月度占比变化")
    ax.set_xlabel("月份")
    ax.set_ylabel("占全社会用电总量比例")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stage2_monthly_structure_share.png", bbox_inches="tight")
    plt.close(fig)


def plot_growth_contribution(contrib: pd.DataFrame) -> None:
    df = contrib[["第一产业", "第二产业", "第三产业", "居民生活"]].copy()
    df.index = df.index.year
    ax = df.plot(kind="bar", figsize=(10, 5.5), width=0.72, colormap="Set2")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("各结构项对年度新增用电的贡献率")
    ax.set_xlabel("相对上一年")
    ax.set_ylabel("贡献率")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIG / "stage2_annual_growth_contribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_long_contribution(contrib: pd.DataFrame) -> None:
    df = contrib.set_index("结构项")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    ax.bar(df.index, df["增长贡献率"], color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("2022-2025 年新增用电来源分解")
    ax.set_xlabel("结构项")
    ax.set_ylabel("对总新增用电的贡献率")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(df["增长贡献率"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.015 if value >= 0 else -0.015
        ax.text(i, value + offset, f"{value:.1%}", ha="center", va=va, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "stage2_2022_2025_growth_contribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_monthly_yoy_heatmap(contrib: pd.DataFrame) -> None:
    df = contrib[["第一产业", "第二产业", "第三产业", "居民生活"]].copy()
    df = df.loc[df.index <= pd.Timestamp("2025-12-01")]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    matrix = df.T.values
    vmax = np.nanpercentile(np.abs(matrix), 95)
    vmax = max(vmax, 0.5)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("月度同比新增用电贡献率热力图")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    tick_locs = np.arange(0, len(df.index), 6)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([df.index[i].strftime("%Y-%m") for i in tick_locs], rotation=45, ha="right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    cbar.set_label("贡献率")
    fig.tight_layout()
    fig.savefig(FIG / "stage2_monthly_yoy_contribution_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(tables: dict[str, pd.DataFrame]) -> None:
    annual_share = tables["annual_share"].copy()
    annual_share.index = annual_share.index.year
    long_contrib = tables["long_growth_contribution"].copy()
    top_long = long_contrib.sort_values("增长贡献率", ascending=False).iloc[0]
    summary = {
        "date_start": str(tables["daily_core"].index.min().date()),
        "date_end": str(tables["daily_core"].index.max().date()),
        "annual_share_percent": {
            str(year): {col: round(float(value * 100), 4) for col, value in row.items()}
            for year, row in annual_share.iterrows()
        },
        "largest_2022_2025_growth_contributor": {
            "结构项": str(top_long["结构项"]),
            "增长贡献率": float(top_long["增长贡献率"]),
        },
    }
    (OUT / "stage2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ensure_dirs()
    setup_plot_style()
    wide = read_stage1_wide()
    tables = build_stage2_tables(wide)
    save_tables(tables)
    plot_process_flow()
    plot_annual_share(tables["annual_share"])
    plot_monthly_share(tables["monthly_share"])
    plot_growth_contribution(tables["annual_growth_contribution"])
    plot_long_contribution(tables["long_growth_contribution"])
    plot_monthly_yoy_heatmap(tables["monthly_yoy_contribution"])
    write_summary(tables)


if __name__ == "__main__":
    main()
