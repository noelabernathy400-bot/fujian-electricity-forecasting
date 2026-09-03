from __future__ import annotations

import pandas as pd

from fujian_calendar_iteration import ensure_calendar
from fujian_direct_multistep import add_target_date_features
from fujian_direct_multistep_panel import CONTEXT_POOL, prepare_base
from fujian_holiday_segment_model import add_holiday_segment_columns
from fujian_pipeline import CITY_COLS, FIG, NOTES, OUT, WRITING, fit_ridge, make_lag_features, metrics, pct, write_md
from fujian_second_iteration import CALENDAR_COLS


DATE = "2026-05-07"
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-01-09")
HORIZONS = list(range(1, 31))
TARGETS = ["居民生活", "宁德", "莆田"]


def prepare_segment_base() -> pd.DataFrame:
    df = prepare_base()
    cal = add_holiday_segment_columns(ensure_calendar())
    segment_cols = [c for c in cal.columns if c.startswith("seg_")] + [
        "holiday_day_index",
        "days_to_holiday_start",
        "days_from_holiday_end",
        "is_pre_holiday3",
        "is_post_holiday3",
    ]
    return df.merge(cal[["日期"] + segment_cols], on="日期", how="left")


def context_for_target(df: pd.DataFrame, target: str) -> list[str]:
    if target in CITY_COLS:
        base = ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电"]
        return [c for c in base + CITY_COLS if c != target and c in df.columns]
    return [c for c in CONTEXT_POOL if c != target and c in df.columns]


def target_segment_frame(df: pd.DataFrame, target: str, horizon: int, with_segments: bool) -> tuple[pd.DataFrame, list[str]]:
    feat = make_lag_features(df, target)
    feat["y_future"] = df[target].shift(-horizon)
    own_cols = [
        f"{target}_lag1",
        f"{target}_lag2",
        f"{target}_lag3",
        f"{target}_lag7",
        f"{target}_lag14",
        f"{target}_lag28",
        f"{target}_lag365",
        f"{target}_roll7",
        f"{target}_roll28",
        *CALENDAR_COLS,
    ]
    own_cols = [c for c in own_cols if c in feat.columns]
    feat, target_cols = add_target_date_features(feat, df, horizon)

    extra_cols: list[str] = []
    for var in context_for_target(df, target):
        for lag in [1, 7, 14, 28]:
            col = f"{var}_lag{lag}"
            feat[col] = df[var].shift(lag)
            extra_cols.append(col)

    segment_cols = [c for c in df.columns if c.startswith("seg_")] + [
        "holiday_day_index",
        "days_to_holiday_start",
        "days_from_holiday_end",
        "is_pre_holiday3",
        "is_post_holiday3",
    ]
    lookup = df[["日期"] + segment_cols].copy().rename(columns={"日期": "target_date"})
    lookup = lookup.rename(columns={c: f"target_{c}" for c in segment_cols})
    feat = feat.merge(lookup, on="target_date", how="left")

    cols = own_cols + extra_cols + target_cols
    if with_segments:
        cols += [f"target_{c}" for c in segment_cols]
    return feat.dropna(subset=["y_future"] + cols).copy(), cols


def run_segment_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = prepare_segment_base()
    pred_rows = []
    metric_rows = []
    for target in [t for t in TARGETS if t in df.columns]:
        group = "city" if target in CITY_COLS else "key_variable"
        for horizon in HORIZONS:
            for model_name, with_segments in [
                ("direct_context_weather_holiday", False),
                ("direct_context_weather_holiday_segment", True),
            ]:
                feat, cols = target_segment_frame(df, target, horizon, with_segments)
                target_date = feat["日期"] + pd.to_timedelta(horizon, unit="D")
                train = feat[target_date < TEST_START]
                test = feat[(target_date >= TEST_START) & (target_date <= TEST_END)]
                test_target_date = test["日期"] + pd.to_timedelta(horizon, unit="D")
                if len(train) < 100 or len(test) == 0:
                    continue
                model = fit_ridge(train, "y_future", cols)
                pred = model.predict(test[cols])
                metric_rows.append({"group": group, "target": target, "horizon": horizon, "model": model_name, **metrics(test["y_future"].to_numpy(float), pred), "n": len(test)})
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "group": group,
                            "target": target,
                            "origin_date": test["日期"].dt.strftime("%Y-%m-%d"),
                            "target_date": test_target_date.dt.strftime("%Y-%m-%d"),
                            "horizon": horizon,
                            "model": model_name,
                            "actual": test["y_future"].to_numpy(float),
                            "prediction": pred,
                        }
                    )
                )
    preds = pd.concat(pred_rows, ignore_index=True)
    metrics_by_h = pd.DataFrame(metric_rows).sort_values(["group", "target", "model", "horizon"])
    overall_rows = []
    for (group, target, model), g in preds.groupby(["group", "target", "model"]):
        overall_rows.append({"group": group, "target": target, "model": model, **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)), "n": len(g)})
    overall = pd.DataFrame(overall_rows).sort_values(["group", "target", "rmse"])
    preds.to_csv(OUT / "forecast_holiday_segment_panel_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_by_h.to_csv(OUT / "forecast_holiday_segment_panel_by_horizon.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT / "forecast_holiday_segment_panel_overall.csv", index=False, encoding="utf-8-sig")
    comparison = build_comparison(overall)
    comparison.to_csv(OUT / "forecast_holiday_segment_panel_comparison.csv", index=False, encoding="utf-8-sig")
    write_mape_comparison_svg(comparison)
    return preds, overall, comparison


def build_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    base = overall[overall["model"] == "direct_context_weather_holiday"].set_index("target")
    seg = overall[overall["model"] == "direct_context_weather_holiday_segment"].set_index("target")
    rows = []
    for target in [t for t in TARGETS if t in base.index and t in seg.index]:
        rows.append(
            {
                "target": target,
                "group": seg.loc[target, "group"],
                "baseline_mape": base.loc[target, "mape"],
                "segment_mape": seg.loc[target, "mape"],
                "mape_change_pp": (seg.loc[target, "mape"] - base.loc[target, "mape"]) * 100,
                "baseline_rmse": base.loc[target, "rmse"],
                "segment_rmse": seg.loc[target, "rmse"],
                "rmse_change_pct": (seg.loc[target, "rmse"] / base.loc[target, "rmse"] - 1) * 100,
                "n": int(seg.loc[target, "n"]),
            }
        )
    return pd.DataFrame(rows).sort_values("segment_mape")


def write_mape_comparison_svg(comparison: pd.DataFrame) -> None:
    labels = comparison["target"].tolist()
    base = (comparison["baseline_mape"] * 100).tolist()
    seg = (comparison["segment_mape"] * 100).tolist()
    width = 980
    height = 120 + 88 * max(len(labels), 1)
    left, right, top = 150, 70, 58
    plot_w = width - left - right
    max_v = max(base + seg) if labels else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">高误差对象长假分段前后 MAPE</text>',
        f'<rect x="{left}" y="48" width="18" height="12" fill="#94a3b8"/><text x="{left+26}" y="59" font-size="12" font-family="Arial">原综合模型</text>',
        f'<rect x="{left+140}" y="48" width="18" height="12" fill="#2563eb"/><text x="{left+166}" y="59" font-size="12" font-family="Arial">长假分段模型</text>',
    ]
    for i, label in enumerate(labels):
        y = top + 22 + i * 88
        bw = base[i] / max_v * plot_w
        sw = seg[i] / max_v * plot_w
        parts.append(f'<text x="{left-14}" y="{y+28}" text-anchor="end" font-size="15" font-family="Arial">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bw}" height="24" fill="#94a3b8" opacity="0.85"/>')
        parts.append(f'<text x="{left+bw+6}" y="{y+17}" font-size="12" font-family="Arial">{base[i]:.2f}%</text>')
        parts.append(f'<rect x="{left}" y="{y+32}" width="{sw}" height="24" fill="#2563eb" opacity="0.88"/>')
        parts.append(f'<text x="{left+sw+6}" y="{y+49}" font-size="12" font-family="Arial">{seg[i]:.2f}%</text>')
    parts.append("</svg>")
    (FIG / "forecast_holiday_segment_panel_mape.svg").write_text("\n".join(parts), encoding="utf-8")


def decompose_by_holiday(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = add_holiday_segment_columns(ensure_calendar())
    cal = cal[["日期", "holiday_name", "is_public_holiday", "is_rest_day", "holiday_segment"]].rename(columns={"日期": "target_date"})
    merged = preds.copy()
    merged["target_date"] = pd.to_datetime(merged["target_date"])
    merged = merged.merge(cal, on="target_date", how="left")
    rows = []
    for (target, model, is_holiday), g in merged.groupby(["target", "model", "is_public_holiday"]):
        rows.append({"target": target, "model": model, "is_public_holiday": int(is_holiday), **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)), "n": len(g)})
    by_public = pd.DataFrame(rows).sort_values(["target", "is_public_holiday", "model"])
    by_public.to_csv(OUT / "forecast_holiday_segment_panel_error_by_public_holiday.csv", index=False, encoding="utf-8-sig")

    rows = []
    holiday_rows = merged[merged["holiday_name"].fillna("") != ""]
    for (target, model, name), g in holiday_rows.groupby(["target", "model", "holiday_name"]):
        rows.append({"target": target, "model": model, "holiday_name": name, **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)), "n": len(g)})
    by_name = pd.DataFrame(rows).sort_values(["target", "holiday_name", "model"])
    by_name.to_csv(OUT / "forecast_holiday_segment_panel_error_by_name.csv", index=False, encoding="utf-8-sig")
    return by_public, by_name


def write_note(comparison: pd.DataFrame, by_public: pd.DataFrame) -> None:
    comp_lines = ["| 对象 | 原综合模型 MAPE | 长假分段 MAPE | 变化百分点 | 原 RMSE | 分段 RMSE |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in comparison.iterrows():
        comp_lines.append(
            f"| {r['target']} | {pct(r['baseline_mape'])} | {pct(r['segment_mape'])} | {r['mape_change_pp']:.2f} | {r['baseline_rmse']:.2f} | {r['segment_rmse']:.2f} |"
        )
    public_lines = ["| 对象 | 模型 | 是否公共假日 | MAPE | RMSE | 样本数 |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for _, r in by_public.iterrows():
        label = "是" if int(r["is_public_holiday"]) == 1 else "否"
        public_lines.append(f"| {r['target']} | {r['model']} | {label} | {pct(r['mape'])} | {r['rmse']:.2f} | {int(r['n'])} |")

    md = f"""
# 高误差对象长假分段模型扩展

## 1. 这一步在解决什么问题

上一轮只对全省 `合计` 做了长假分段，结果很好：整体 MAPE 从 3.40% 降到 3.06%，公共假日 MAPE 从 5.22% 降到 2.62%。但是一个预测方法不能只在总量上好看，还要看它能不能帮助更难的对象。

本轮选择三个高误差对象：

- `居民生活`：上一轮 1-30 日直接多步最优 MAPE 约 7.25%，比工业、制造业明显更难。
- `莆田`：上一轮地市直接多步最优 MAPE 约 7.22%，是地市中较难的一类。
- `宁德`：上一轮地市直接多步最优 MAPE 约 6.09%，同样需要误差诊断。

## 2. 模型对比

两类模型完全使用同一个测试期，即 2025-10-01 到 2026-01-09 的目标日期。

- `direct_context_weather_holiday`：自身滞后、上下文变量滞后、目标日天气、目标日节假日。
- `direct_context_weather_holiday_segment`：在上一模型基础上加入目标日的节前、节中、节后位置变量。

这里的上下文变量不是未来信息。对行业变量，它来自其他用电类别的历史滞后；对地市变量，它来自全省、重点行业和其他地市的历史滞后。

## 3. 整体结果

{chr(10).join(comp_lines)}

`变化百分点` 的计算方式是：

$$
\\Delta\\mathrm{{MAPE}} = 100\\times(\\mathrm{{MAPE}}_{{\\mathrm{{segment}}}}-\\mathrm{{MAPE}}_{{\\mathrm{{baseline}}}})
$$

所以负数表示长假分段模型降低了误差，正数表示误差反而上升。

## 4. 公共假日分解

{chr(10).join(public_lines)}

## 5. 怎么读这个结果

如果一个对象的公共假日误差下降，但非公共假日变化不大，说明它的问题主要在假期位置效应。换句话说，模型不是整体变复杂后“平均凑出来”更好，而是在我们怀疑的薄弱场景上起作用。

如果整体 MAPE 下降但公共假日没有明显改善，要进一步检查是否是节前或节后恢复期带来的改善。如果整体 MAPE 不降反升，说明总量模型中有效的长假分段规则不能直接迁移到这个对象，后续需要分对象建模。

## 6. 当前边界

这一步仍然是预测模型改进，不是因果识别。我们能说的是“长假位置特征对预测有帮助或没有帮助”，不能说“长假导致了某个行业或地市用电变化的因果效应”。要把它升级成因果结论，需要明确反事实：如果同样的日期没有长假安排，用电量会怎样，并且要有可识别的对照或外生变化。

## 7. 输出

- `Experiments/outputs/forecast_holiday_segment_panel_comparison.csv`
- `Experiments/outputs/forecast_holiday_segment_panel_overall.csv`
- `Experiments/outputs/forecast_holiday_segment_panel_error_by_public_holiday.csv`
- `Experiments/outputs/forecast_holiday_segment_panel_error_by_name.csv`
- `Experiments/figures/forecast_holiday_segment_panel_mape.svg`
"""
    write_md(NOTES / f"高误差对象长假分段模型扩展-{DATE}.md", md)
    report = f"""
# 福建实证高误差对象长假分段模型扩展报告

本轮将长假分段模型从全省 `合计` 扩展到 `居民生活`、`莆田`、`宁德` 三个高误差对象。

结果入口：

- `Notes/高误差对象长假分段模型扩展-{DATE}.md`
- `Experiments/outputs/forecast_holiday_segment_panel_comparison.csv`
- `Experiments/figures/forecast_holiday_segment_panel_mape.svg`
"""
    write_md(WRITING / f"福建实证高误差对象长假分段模型扩展报告-{DATE}.md", report)


def run_all() -> None:
    preds, _, comparison = run_segment_panel()
    by_public, _ = decompose_by_holiday(preds)
    write_note(comparison, by_public)


if __name__ == "__main__":
    run_all()
