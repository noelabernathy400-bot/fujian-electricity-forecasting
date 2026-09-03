from __future__ import annotations

import numpy as np
import pandas as pd

from fujian_calendar_iteration import ensure_calendar
from fujian_direct_multistep import add_target_date_features
from fujian_pipeline import CITY_COLS, FIG, NOTES, OUT, PROCESSED, WRITING, fit_ridge, make_lag_features, metrics, pct, write_md
from fujian_second_iteration import CALENDAR_COLS, svg_bar_percent
from fujian_weather_iteration import ensure_weather


DATE = "2026-05-07"
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-01-09")
HORIZONS = list(range(1, 31))


KEY_TARGETS = ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电", "非普工业"]
CONTEXT_POOL = ["合计", "工业", "制造业", "大工业电量", "非普工业", "居民生活", "商业用电"]


def prepare_base() -> pd.DataFrame:
    province = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    weather = ensure_weather()
    calendar = ensure_calendar()
    return province.merge(weather, on="日期", how="left").merge(calendar, on="日期", how="left").sort_values("日期").reset_index(drop=True)


def target_feature_frame(df: pd.DataFrame, target: str, horizon: int, context: list[str]) -> tuple[pd.DataFrame, list[str]]:
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
    extra_cols = []
    for var in context:
        if var == target or var not in df.columns:
            continue
        for lag in [1, 7, 14, 28]:
            col = f"{var}_lag{lag}"
            feat[col] = df[var].shift(lag)
            extra_cols.append(col)
    cols_own = own_cols + target_cols
    cols_context = own_cols + extra_cols + target_cols
    return feat, cols_own, cols_context


def evaluate_targets(targets: list[str], group_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_base()
    pred_rows = []
    metric_rows = []
    for target in [t for t in targets if t in df.columns]:
        if target in CITY_COLS:
            context = ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电"] + [c for c in CITY_COLS if c != target and c in df.columns]
        else:
            context = [c for c in CONTEXT_POOL if c != target and c in df.columns]
        for horizon in HORIZONS:
            feat, cols_own, cols_context = target_feature_frame(df, target, horizon, context)
            for model_name, cols in [
                ("direct_own_weather_holiday", cols_own),
                ("direct_context_weather_holiday", cols_context),
            ]:
                target_date = feat["日期"] + pd.to_timedelta(horizon, unit="D")
                sample = feat.dropna(subset=["y_future"] + cols).copy()
                sample_target_date = sample["日期"] + pd.to_timedelta(horizon, unit="D")
                train = sample[sample_target_date < TEST_START]
                test = sample[(sample_target_date >= TEST_START) & (sample_target_date <= TEST_END)]
                test_target_date = test["日期"] + pd.to_timedelta(horizon, unit="D")
                if len(train) < 100 or len(test) == 0:
                    continue
                model = fit_ridge(train, "y_future", cols)
                pred = model.predict(test[cols])
                metric_rows.append({"group": group_name, "target": target, "horizon": horizon, "model": model_name, **metrics(test["y_future"].to_numpy(float), pred), "n": len(test)})
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "group": group_name,
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
    metrics_by_h = pd.DataFrame(metric_rows).sort_values(["group", "target", "model", "horizon"])
    preds = pd.concat(pred_rows, ignore_index=True)
    overall_rows = []
    for (group, target, model), g in preds.groupby(["group", "target", "model"]):
        overall_rows.append({"group": group, "target": target, "model": model, **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)), "n": len(g)})
    overall = pd.DataFrame(overall_rows).sort_values(["group", "target", "rmse"])
    return metrics_by_h, overall


def run_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    province = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"])
    city_targets = [c for c in CITY_COLS if c in province.columns]
    m1, o1 = evaluate_targets([t for t in KEY_TARGETS if t in province.columns], "key_variable")
    m2, o2 = evaluate_targets(city_targets, "city")
    metrics_by_h = pd.concat([m1, m2], ignore_index=True)
    overall = pd.concat([o1, o2], ignore_index=True).sort_values(["group", "target", "rmse"])
    metrics_by_h.to_csv(OUT / "forecast_direct_multistep_panel_by_horizon.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT / "forecast_direct_multistep_panel_overall.csv", index=False, encoding="utf-8-sig")
    best = overall.groupby(["group", "target"], as_index=False).first().sort_values(["group", "mape"])
    best.to_csv(OUT / "forecast_direct_multistep_panel_best.csv", index=False, encoding="utf-8-sig")
    city_best = best[best["group"] == "city"].sort_values("mape")
    if not city_best.empty:
        svg_bar_percent(FIG / "forecast_direct_multistep_city_best_mape.svg", city_best["target"].tolist(), (city_best["mape"] * 100).tolist(), "地市直接多步预测最优 MAPE")
    key_best = best[best["group"] == "key_variable"].sort_values("mape")
    if not key_best.empty:
        svg_bar_percent(FIG / "forecast_direct_multistep_key_best_mape.svg", key_best["target"].tolist(), (key_best["mape"] * 100).tolist(), "重点变量直接多步预测最优 MAPE")
    return metrics_by_h, overall


def write_note(overall: pd.DataFrame) -> None:
    best = overall.groupby(["group", "target"], as_index=False).first().sort_values(["group", "mape"])
    key = best[best["group"] == "key_variable"]
    city = best[best["group"] == "city"]
    key_lines = ["| 目标 | 最优模型 | MAPE | RMSE |", "| --- | --- | ---: | ---: |"]
    for _, r in key.iterrows():
        key_lines.append(f"| {r['target']} | {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} |")
    city_lines = ["| 地市 | 最优模型 | MAPE | RMSE |", "| --- | --- | ---: | ---: |"]
    for _, r in city.iterrows():
        city_lines.append(f"| {r['target']} | {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} |")
    md = f"""
# 重点行业与地市直接多步预测

## 1. 为什么补这一轮

全省合计预测只能说明总量层面可预测，不足以支撑行业结构和地市差异研究。真正有用的结果应当回答：哪些行业更容易预测、哪些地市更难预测、加入上下文滞后变量是否有帮助。

本轮对重点变量和 9 个地市做 1-30 日直接多步预测。每个目标比较两类模型：

- `direct_own_weather_holiday`：目标自身滞后 + 目标日天气 + 目标日节假日。
- `direct_context_weather_holiday`：在前者基础上加入全省、行业或其他地市的历史滞后。

## 2. 重点变量结果

{chr(10).join(key_lines)}

## 3. 地市结果

{chr(10).join(city_lines)}

## 4. 怎么解释

如果 `direct_context_weather_holiday` 优于自身模型，说明其他行业或地市的历史变化对该目标有预测帮助；但这仍然是预测贡献，不是强因果。如果某个地市 MAPE 明显高，后续要检查长假、产业结构、天气敏感性和异常波动。

## 5. 输出

- `Experiments/outputs/forecast_direct_multistep_panel_by_horizon.csv`
- `Experiments/outputs/forecast_direct_multistep_panel_overall.csv`
- `Experiments/outputs/forecast_direct_multistep_panel_best.csv`
- `Experiments/figures/forecast_direct_multistep_city_best_mape.svg`
- `Experiments/figures/forecast_direct_multistep_key_best_mape.svg`
"""
    write_md(NOTES / f"重点行业与地市直接多步预测-{DATE}.md", md)
    report = f"""
# 福建实证重点行业与地市直接多步预测补充报告

本轮将 1-30 日直接多步预测扩展到重点行业和地市。入口：

- `Notes/重点行业与地市直接多步预测-{DATE}.md`
- `Experiments/outputs/forecast_direct_multistep_panel_best.csv`
"""
    write_md(WRITING / f"福建实证重点行业与地市直接多步预测补充报告-{DATE}.md", report)


def run_all() -> None:
    _, overall = run_panel()
    write_note(overall)


if __name__ == "__main__":
    run_all()
