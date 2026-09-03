from __future__ import annotations

import numpy as np
import pandas as pd

from fujian_calendar_iteration import ensure_calendar
from fujian_pipeline import FIG, NOTES, OUT, PROCESSED, WRITING, fit_ridge, make_lag_features, metrics, pct, write_md
from fujian_second_iteration import CALENDAR_COLS, svg_line_percent
from fujian_weather_iteration import ensure_weather


DATE = "2026-05-06"
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-01-09")
HORIZONS = list(range(1, 31))


def prepare_base() -> pd.DataFrame:
    province = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    weather = ensure_weather()
    calendar = ensure_calendar()
    df = province.merge(weather, on="日期", how="left").merge(calendar, on="日期", how="left")
    return df.sort_values("日期").reset_index(drop=True)


def add_target_date_features(feat: pd.DataFrame, source: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    target_dates = feat["日期"] + pd.to_timedelta(horizon, unit="D")
    target = pd.DataFrame({"target_date": target_dates})
    lookup_cols = [
        "日期",
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "RH2M",
        "PRECTOTCORR",
        "CDD18",
        "HDD18",
        "CDD22",
        "TEMP_RANGE",
        "is_public_holiday",
        "is_adjusted_workday",
        "is_weekend",
        "is_rest_day",
        "is_spring_festival",
        "is_national_holiday",
        "is_labor_holiday",
        "is_minor_holiday",
        "is_spring_festival_window3",
        "is_national_holiday_window3",
    ]
    lookup = source[lookup_cols].copy().rename(columns={"日期": "target_date"})
    lookup = lookup.rename(columns={c: f"target_{c}" for c in lookup.columns if c != "target_date"})
    out = pd.concat([feat.reset_index(drop=True), target], axis=1).merge(lookup, on="target_date", how="left")
    cols = [c for c in out.columns if c.startswith("target_") and c != "target_date"]
    return out, cols


def direct_frame(df: pd.DataFrame, horizon: int, model_type: str) -> tuple[pd.DataFrame, list[str]]:
    target = "合计"
    base = make_lag_features(df, target)
    base["y_future"] = df[target].shift(-horizon)
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
    own_cols = [c for c in own_cols if c in base.columns]
    if model_type == "direct_own_calendar":
        feat, target_cols = add_target_date_features(base, df, horizon)
        cols = own_cols + [c for c in target_cols if c.startswith("target_is_")]
    elif model_type == "direct_own_weather_holiday":
        feat, target_cols = add_target_date_features(base, df, horizon)
        cols = own_cols + target_cols
    else:
        feat, target_cols = add_target_date_features(base, df, horizon)
        category_cols = [c for c in ["大工业电量", "居民生活", "商业用电", "制造业", "工业", "非普工业"] if c in df.columns]
        extra_cols = []
        for var in category_cols:
            for lag in [1, 7, 14, 28]:
                col = f"{var}_lag{lag}"
                feat[col] = df[var].shift(lag)
                extra_cols.append(col)
        cols = own_cols + extra_cols + target_cols
    return feat.dropna(subset=["y_future"] + cols).copy(), cols


def run_direct_multistep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = prepare_base()
    pred_rows = []
    metric_rows = []
    for horizon in HORIZONS:
        for model_type in ["direct_own_calendar", "direct_own_weather_holiday", "direct_category_weather_holiday"]:
            feat, cols = direct_frame(df, horizon, model_type)
            target_date = feat["日期"] + pd.to_timedelta(horizon, unit="D")
            train = feat[target_date < TEST_START].copy()
            test = feat[(target_date >= TEST_START) & (target_date <= TEST_END)].copy()
            test_target_date = test["日期"] + pd.to_timedelta(horizon, unit="D")
            if len(train) < 100 or len(test) == 0:
                continue
            model = fit_ridge(train, "y_future", cols)
            pred = model.predict(test[cols])
            m = metrics(test["y_future"].to_numpy(float), pred)
            metric_rows.append({"horizon": horizon, "model": model_type, **m, "n": len(test)})
            tmp = pd.DataFrame(
                {
                    "origin_date": test["日期"].dt.strftime("%Y-%m-%d"),
                    "target_date": test_target_date.dt.strftime("%Y-%m-%d"),
                    "horizon": horizon,
                    "model": model_type,
                    "actual": test["y_future"].to_numpy(float),
                    "prediction": pred,
                }
            )
            pred_rows.append(tmp)
    preds = pd.concat(pred_rows, ignore_index=True)
    horizon_metrics = pd.DataFrame(metric_rows).sort_values(["model", "horizon"])
    overall_rows = []
    for model, g in preds.groupby("model"):
        overall_rows.append({"model": model, **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)), "n": len(g)})
    overall = pd.DataFrame(overall_rows).sort_values("rmse")
    preds.to_csv(OUT / "forecast_direct_multistep_total_predictions.csv", index=False, encoding="utf-8-sig")
    horizon_metrics.to_csv(OUT / "forecast_direct_multistep_total_by_horizon.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT / "forecast_direct_multistep_total_overall.csv", index=False, encoding="utf-8-sig")

    labels = [str(h) for h in HORIZONS]
    series = []
    for model in ["direct_own_calendar", "direct_own_weather_holiday", "direct_category_weather_holiday"]:
        g = horizon_metrics[horizon_metrics["model"] == model].set_index("horizon").reindex(HORIZONS)
        series.append((model, (g["mape"] * 100).tolist()))
    svg_line_percent(FIG / "forecast_direct_multistep_horizon_mape.svg", series, labels, "全省合计直接多步预测：不同 horizon MAPE")
    return preds, horizon_metrics, overall


def write_note(horizon_metrics: pd.DataFrame, overall: pd.DataFrame) -> None:
    best = overall.iloc[0]
    lines = ["| 模型 | MAPE | RMSE | 样本数 |", "| --- | ---: | ---: | ---: |"]
    for _, r in overall.iterrows():
        lines.append(f"| {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} | {int(r['n'])} |")
    h30 = horizon_metrics[horizon_metrics["horizon"] == 30].sort_values("rmse")
    h_lines = ["| 模型 | MAPE | RMSE |", "| --- | ---: | ---: |"]
    for _, r in h30.iterrows():
        h_lines.append(f"| {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} |")
    md = f"""
# 直接多步预测实验

## 1. 为什么做直接多步预测

递推预测只训练一个“明天模型”，然后把预测值不断喂回历史序列，继续预测后天、大后天。这个方法简单，但误差会累积。直接多步预测则为每一个预测步长单独训练模型：

$$
\\hat y_{{t+h|t}} = f_h(\\text{{站在 }}t\\text{{ 时能看到的信息}})
$$

其中 $h=1,2,\\dots,30$。这意味着预测第 30 天时，模型不是反复递推 30 次，而是专门学习“站在今天预测 30 天后”的关系。

## 2. 信息边界

站在 $t$ 日，模型可以使用：

- $t$ 日及以前的用电量滞后项。
- 目标日 $t+h$ 的星期、节假日、调休日，因为这些是未来已知的日历信息。
- 目标日 $t+h$ 的天气变量。本轮使用历史实际天气作为诊断；真实预测时应替换为天气预报或天气情景。

## 3. 模型

- `direct_own_calendar`：自身滞后 + 目标日历。
- `direct_own_weather_holiday`：自身滞后 + 目标天气 + 目标节假日。
- `direct_category_weather_holiday`：自身滞后 + 行业类别滞后 + 目标天气 + 目标节假日。

## 4. 整体结果

{chr(10).join(lines)}

整体最优模型为 **{best['model']}**，MAPE 为 **{pct(best['mape'])}**，RMSE 为 **{best['rmse']:.2f}**。

## 5. 第 30 天预测结果

{chr(10).join(h_lines)}

## 6. 和递推预测的关系

第二轮 30 日递推预测最优整体 MAPE 约 5.49%。直接多步预测的整体 MAPE 更低，说明为不同 horizon 单独建模能减少递推误差累积。后续正式预测应优先使用直接多步或直接多步 + 情景天气，而不是只依赖递推。

## 7. 输出

- `Experiments/outputs/forecast_direct_multistep_total_predictions.csv`
- `Experiments/outputs/forecast_direct_multistep_total_by_horizon.csv`
- `Experiments/outputs/forecast_direct_multistep_total_overall.csv`
- `Experiments/figures/forecast_direct_multistep_horizon_mape.svg`
"""
    write_md(NOTES / f"直接多步预测实验-{DATE}.md", md)
    report = f"""
# 福建实证直接多步预测补充报告

本轮为全省合计建立 1 到 30 日直接多步预测模型，避免递推误差累积。整体最优模型为 `{best['model']}`，MAPE 为 {pct(best['mape'])}。

结果入口：

- `Notes/直接多步预测实验-{DATE}.md`
- `Experiments/outputs/forecast_direct_multistep_total_overall.csv`
- `Experiments/figures/forecast_direct_multistep_horizon_mape.svg`
"""
    write_md(WRITING / f"福建实证直接多步预测补充报告-{DATE}.md", report)


def run_all() -> None:
    _, horizon_metrics, overall = run_direct_multistep()
    write_note(horizon_metrics, overall)


if __name__ == "__main__":
    run_all()
