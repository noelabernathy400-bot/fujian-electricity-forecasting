from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fujian_pipeline import CITY_COLS, FIG, NOTES, OUT, PROCESSED, WRITING, pct, write_md


DATE = "2026-05-07"
FULL_YEARS = [2023, 2024, 2025]
KEY_VARS = ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电", "非普工业", "电子", "纺织业", "非金属矿物制品业", "黑色金属冶炼"]
MONTH_LABELS = [f"{m}月" for m in range(1, 13)]


def read_daily() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    df["year"] = df["日期"].dt.year
    df["month"] = df["日期"].dt.month
    return df[df["year"].isin(FULL_YEARS)].reset_index(drop=True)


def city_growth_contribution(df: pd.DataFrame) -> pd.DataFrame:
    annual = df.groupby("year")[CITY_COLS + ["合计"]].sum(numeric_only=True).reset_index()
    rows = []
    for year in [2024, 2025]:
        cur = annual[annual["year"] == year].iloc[0]
        prev = annual[annual["year"] == year - 1].iloc[0]
        total_delta = float(cur["合计"] - prev["合计"])
        for city in CITY_COLS:
            delta = float(cur[city] - prev[city])
            rows.append(
                {
                    "year": year,
                    "city": city,
                    "annual_total": float(cur[city]),
                    "delta": delta,
                    "growth_rate": delta / float(prev[city]),
                    "contribution_to_total_growth": delta / total_delta if total_delta else math.nan,
                }
            )
    out = pd.DataFrame(rows).sort_values(["year", "contribution_to_total_growth"], ascending=[True, False])
    out.to_csv(OUT / "structure_city_growth_contribution.csv", index=False, encoding="utf-8-sig")
    return out


def key_variable_yoy(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEY_VARS if c in df.columns]
    annual = df.groupby("year")[cols].sum(numeric_only=True).reset_index()
    rows = []
    for col in cols:
        prev_val = None
        for _, row in annual.iterrows():
            val = float(row[col])
            yoy = (val / prev_val - 1) if prev_val and prev_val != 0 else math.nan
            rows.append({"year": int(row["year"]), "variable": col, "annual_total": val, "yoy": yoy})
            prev_val = val
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "structure_key_variable_yoy.csv", index=False, encoding="utf-8-sig")
    return out


def seasonality_index(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEY_VARS if c in df.columns]
    rows = []
    for col in cols:
        overall_daily_mean = float(df[col].mean())
        for month in range(1, 13):
            g = df[df["month"] == month]
            month_daily_mean = float(g[col].mean())
            rows.append(
                {
                    "variable": col,
                    "month": month,
                    "month_label": f"{month}月",
                    "daily_mean": month_daily_mean,
                    "seasonality_index": month_daily_mean / overall_daily_mean if overall_daily_mean else math.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "structure_monthly_seasonality_index.csv", index=False, encoding="utf-8-sig")
    return out


def volatility_profile(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEY_VARS + CITY_COLS if c in df.columns]
    rows = []
    for col in cols:
        s = df[col].astype(float)
        logdiff = np.log1p(s).diff().dropna()
        rows.append(
            {
                "variable": col,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "cv": float(s.std(ddof=1) / s.mean()) if s.mean() else math.nan,
                "logdiff_sd": float(logdiff.std(ddof=1)),
                "mean_abs_logdiff": float(logdiff.abs().mean()),
                "p95_abs_logdiff": float(logdiff.abs().quantile(0.95)),
                "peak_to_mean": float(s.max() / s.mean()) if s.mean() else math.nan,
                "min_to_mean": float(s.min() / s.mean()) if s.mean() else math.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("cv", ascending=False)
    out.to_csv(OUT / "structure_volatility_profile.csv", index=False, encoding="utf-8-sig")
    return out


def svg_diverging_bar(path: Path, labels: list[str], values: list[float], title: str, suffix: str = "%") -> None:
    width = 980
    height = 80 + 44 * max(len(labels), 1)
    left, right, top = 180, 70, 55
    plot_w = width - left - right
    max_abs = max([abs(v) for v in values] + [1])
    zero_x = left + plot_w / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{zero_x}" y1="{top-8}" x2="{zero_x}" y2="{height-25}" stroke="#475569" stroke-width="1"/>',
    ]
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = top + i * 44
        w = abs(val) / max_abs * (plot_w / 2)
        x = zero_x if val >= 0 else zero_x - w
        color = "#2563eb" if val >= 0 else "#dc2626"
        parts.append(f'<text x="{left-10}" y="{y+18}" text-anchor="end" font-size="13" font-family="Arial">{html.escape(lab)}</text>')
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="24" fill="{color}" opacity="0.85"/>')
        text_x = x + w + 6 if val >= 0 else x - 6
        anchor = "start" if val >= 0 else "end"
        parts.append(f'<text x="{text_x}" y="{y+17}" text-anchor="{anchor}" font-size="12" font-family="Arial">{val:.1f}{suffix}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_bar(path: Path, labels: list[str], values: list[float], title: str, suffix: str = "%") -> None:
    width = 980
    height = 80 + 42 * max(len(labels), 1)
    left, right, top = 190, 70, 55
    plot_w = width - left - right
    vmax = max(values) if values else 1
    vmax = vmax if vmax > 0 else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{html.escape(title)}</text>',
    ]
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = top + i * 42
        w = val / vmax * plot_w
        parts.append(f'<text x="{left-10}" y="{y+18}" text-anchor="end" font-size="13" font-family="Arial">{html.escape(lab)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w}" height="24" fill="#0f766e" opacity="0.85"/>')
        parts.append(f'<text x="{left+w+6}" y="{y+17}" font-size="12" font-family="Arial">{val:.2f}{suffix}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_heatmap(path: Path, season: pd.DataFrame) -> None:
    variables = ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电", "非普工业", "电子"]
    variables = [v for v in variables if v in set(season["variable"])]
    width = 1060
    cell_w, cell_h = 62, 34
    left, top = 150, 65
    height = top + cell_h * len(variables) + 55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">重点变量月度季节性指数</text>',
    ]
    for j, label in enumerate(MONTH_LABELS):
        x = left + j * cell_w
        parts.append(f'<text x="{x+cell_w/2}" y="{top-14}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')
    for i, var in enumerate(variables):
        y = top + i * cell_h
        parts.append(f'<text x="{left-10}" y="{y+22}" text-anchor="end" font-size="13" font-family="Arial">{html.escape(var)}</text>')
        g = season[season["variable"] == var].set_index("month")
        for month in range(1, 13):
            val = float(g.loc[month, "seasonality_index"])
            centered = max(-0.25, min(0.25, val - 1.0))
            if centered >= 0:
                intensity = int(220 - centered / 0.25 * 120)
                color = f"rgb({intensity},{intensity},255)"
            else:
                intensity = int(230 + centered / 0.25 * 130)
                color = f"rgb(255,{intensity},{intensity})"
            x = left + (month - 1) * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{color}" stroke="white"/>')
            parts.append(f'<text x="{x+cell_w/2}" y="{y+21}" text-anchor="middle" font-size="10" font-family="Arial">{val:.2f}</text>')
    parts.append('<text x="150" y="{0}" font-size="12" font-family="Arial">指数大于 1 表示该月日均用电高于全年平均；小于 1 表示低于全年平均。</text>'.format(height - 18))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def top_city_table(city_growth: pd.DataFrame, year: int) -> list[str]:
    g = city_growth[city_growth["year"] == year].sort_values("contribution_to_total_growth", ascending=False)
    lines = ["| 地市 | 增量 | 对全省增长贡献 | 本地增速 |", "| --- | ---: | ---: | ---: |"]
    for _, r in g.iterrows():
        lines.append(f"| {r['city']} | {r['delta']:.0f} | {pct(r['contribution_to_total_growth'])} | {pct(r['growth_rate'])} |")
    return lines


def top_season_table(season: pd.DataFrame) -> list[str]:
    lines = ["| 变量 | 季节性高点 | 高点指数 | 季节性低点 | 低点指数 |", "| --- | --- | ---: | --- | ---: |"]
    for var in ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电", "电子", "纺织业"]:
        g = season[season["variable"] == var]
        if g.empty:
            continue
        hi = g.loc[g["seasonality_index"].idxmax()]
        lo = g.loc[g["seasonality_index"].idxmin()]
        lines.append(f"| {var} | {int(hi['month'])}月 | {hi['seasonality_index']:.2f} | {int(lo['month'])}月 | {lo['seasonality_index']:.2f} |")
    return lines


def top_volatility_table(vol: pd.DataFrame) -> list[str]:
    g = vol[vol["variable"].isin(KEY_VARS)].sort_values("cv", ascending=False)
    lines = ["| 变量 | CV | 日度 log 差分标准差 | 95% 绝对 log 差分 | 峰值/均值 |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, r in g.iterrows():
        lines.append(f"| {r['variable']} | {r['cv']:.3f} | {r['logdiff_sd']:.3f} | {r['p95_abs_logdiff']:.3f} | {r['peak_to_mean']:.2f} |")
    return lines


def write_note(city_growth: pd.DataFrame, key_yoy: pd.DataFrame, season: pd.DataFrame, vol: pd.DataFrame) -> None:
    y2025 = city_growth[city_growth["year"] == 2025].sort_values("contribution_to_total_growth", ascending=False)
    svg_diverging_bar(
        FIG / "structure_city_growth_contribution_2025.svg",
        y2025["city"].tolist(),
        (y2025["contribution_to_total_growth"] * 100).tolist(),
        "2025 年地市对全省用电增长贡献",
    )
    key_vol = vol[vol["variable"].isin(KEY_VARS)].sort_values("cv", ascending=False)
    svg_bar(FIG / "structure_key_variable_volatility_cv.svg", key_vol["variable"].tolist(), (key_vol["cv"] * 100).tolist(), "重点变量日度波动强度 CV")
    svg_heatmap(FIG / "structure_monthly_seasonality_heatmap.svg", season)

    y2024_lines = top_city_table(city_growth, 2024)
    y2025_lines = top_city_table(city_growth, 2025)
    season_lines = top_season_table(season)
    vol_lines = top_volatility_table(vol)

    key_2025 = key_yoy[key_yoy["year"] == 2025].sort_values("yoy", ascending=False)
    key_lines = ["| 变量 | 2025 年总量 | 2025 同比 |", "| --- | ---: | ---: |"]
    for _, r in key_2025.iterrows():
        key_lines.append(f"| {r['variable']} | {r['annual_total']:.0f} | {pct(r['yoy'])} |")

    md = f"""
# 行业结构动态：季节性、波动性与增长贡献

## 1. 为什么补这一轮

前面的结构分析回答了“谁占比大”，但研究还需要回答三个更细的问题：

1. 哪些变量有明显季节性？
2. 哪些变量日度波动更强、更难预测？
3. 全省用电增长主要来自哪些地市？

这三件事分别服务于不同部分：季节性帮助设计预测特征，波动性帮助解释预测难度，增长贡献帮助把结构分析写得更像实证研究而不是静态占比表。

## 2. 季节性指数

季节性指数定义为：

$$
SI_m=\\frac{{\\text{{第 }}m\\text{{ 月的平均日用电量}}}}{{\\text{{全年平均日用电量}}}}
$$

如果 $SI_m>1$，说明这个月日均用电高于全年平均；如果 $SI_m<1$，说明这个月低于全年平均。

{chr(10).join(season_lines)}

图表输出：`Experiments/figures/structure_monthly_seasonality_heatmap.svg`。

## 3. 日度波动性

这里使用变异系数：

$$
CV=\\frac{{s}}{{\\bar y}}
$$

以及日度对数差分：

$$
r_t=\\log(1+y_t)-\\log(1+y_{{t-1}})
$$

CV 越大，表示该变量相对均值的波动越强；$r_t$ 的标准差越大，说明日度变化越剧烈。

{chr(10).join(vol_lines)}

图表输出：`Experiments/figures/structure_key_variable_volatility_cv.svg`。

## 4. 地市增长贡献

9 个地市用电量可以严格加总到全省合计，因此地市增长贡献可以写成：

$$
Contribution_{{i,t}}=\\frac{{Y_{{i,t}}-Y_{{i,t-1}}}}{{Y_{{total,t}}-Y_{{total,t-1}}}}
$$

这个指标回答：全省用电增量中，有多大比例来自某个地市。

### 2024 年相对 2023 年

{chr(10).join(y2024_lines)}

### 2025 年相对 2024 年

{chr(10).join(y2025_lines)}

图表输出：`Experiments/figures/structure_city_growth_contribution_2025.svg`。

## 5. 重点变量同比

下面表格不是严格加总分解，因为 `工业`、`制造业`、`大工业电量` 等变量之间存在包含关系。它只能用于观察每个重点变量自身的年度变化，不能把这些变量的增量相加。

{chr(10).join(key_lines)}

## 6. 怎么解释

地市增长贡献是严格可加的结构证据；重点变量同比是口径相关的描述性证据。论文中应优先使用地市贡献做“增长来自哪里”的分解，用重点变量同比解释“哪些产业或用电类别自身增长更快”。

如果某个变量季节性强、波动性也强，预测模型应优先给它加入天气、节假日、长假分段或专门的季节项。如果某个变量占比大但波动低，它对总量水平重要，但不一定是误差的主要来源。

## 7. 输出

- `Experiments/outputs/structure_city_growth_contribution.csv`
- `Experiments/outputs/structure_key_variable_yoy.csv`
- `Experiments/outputs/structure_monthly_seasonality_index.csv`
- `Experiments/outputs/structure_volatility_profile.csv`
- `Experiments/figures/structure_city_growth_contribution_2025.svg`
- `Experiments/figures/structure_key_variable_volatility_cv.svg`
- `Experiments/figures/structure_monthly_seasonality_heatmap.svg`
"""
    write_md(NOTES / f"行业结构动态分析-季节性波动性增长贡献-{DATE}.md", md)
    report = f"""
# 福建实证行业结构动态补充报告

本轮补充行业结构动态分析：月度季节性指数、日度波动性和地市增长贡献。

入口：

- `Notes/行业结构动态分析-季节性波动性增长贡献-{DATE}.md`
- `Experiments/outputs/structure_city_growth_contribution.csv`
- `Experiments/outputs/structure_volatility_profile.csv`
- `Experiments/figures/structure_monthly_seasonality_heatmap.svg`
"""
    write_md(WRITING / f"福建实证行业结构动态补充报告-{DATE}.md", report)


def run_all() -> None:
    df = read_daily()
    cg = city_growth_contribution(df)
    ky = key_variable_yoy(df)
    season = seasonality_index(df)
    vol = volatility_profile(df)
    write_note(cg, ky, season, vol)


if __name__ == "__main__":
    run_all()
