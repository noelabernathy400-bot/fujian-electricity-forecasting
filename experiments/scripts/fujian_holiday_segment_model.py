from __future__ import annotations

import numpy as np
import pandas as pd

from fujian_calendar_iteration import HOLIDAY_RANGES, ensure_calendar
from fujian_direct_multistep import add_target_date_features, prepare_base
from fujian_pipeline import FIG, NOTES, OUT, PROCESSED, WRITING, fit_ridge, make_lag_features, metrics, pct, write_md
from fujian_second_iteration import CALENDAR_COLS, svg_bar_percent


DATE = "2026-05-07"
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-01-09")
HORIZONS = list(range(1, 31))


def add_holiday_segment_columns(cal: pd.DataFrame) -> pd.DataFrame:
    out = cal.copy()
    out["holiday_segment"] = "normal"
    out["holiday_day_index"] = 0
    out["days_to_holiday_start"] = 99
    out["days_from_holiday_end"] = 99
    out["is_pre_holiday3"] = 0
    out["is_post_holiday3"] = 0
    for start, end, name in HOLIDAY_RANGES:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        mask_mid = (out["日期"] >= s) & (out["日期"] <= e)
        out.loc[mask_mid, "holiday_segment"] = "mid_" + name
        out.loc[mask_mid, "holiday_day_index"] = (out.loc[mask_mid, "日期"] - s).dt.days + 1
        pre_mask = (out["日期"] >= s - pd.Timedelta(days=3)) & (out["日期"] < s)
        post_mask = (out["日期"] > e) & (out["日期"] <= e + pd.Timedelta(days=3))
        out.loc[pre_mask, "holiday_segment"] = "pre_" + name
        out.loc[post_mask, "holiday_segment"] = "post_" + name
        out.loc[pre_mask, "is_pre_holiday3"] = 1
        out.loc[post_mask, "is_post_holiday3"] = 1
        out.loc[pre_mask, "days_to_holiday_start"] = (s - out.loc[pre_mask, "日期"]).dt.days
        out.loc[post_mask, "days_from_holiday_end"] = (out.loc[post_mask, "日期"] - e).dt.days
    out["holiday_day_index"] = out["holiday_day_index"].clip(upper=10)
    seg_dummies = pd.get_dummies(out["holiday_segment"], prefix="seg", dtype=int)
    keep = [c for c in seg_dummies.columns if c != "seg_normal"]
    out = pd.concat([out, seg_dummies[keep]], axis=1)
    return out


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


def direct_segment_frame(df: pd.DataFrame, horizon: int, with_segments: bool) -> tuple[pd.DataFrame, list[str]]:
    target = "合计"
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
    category_cols = [c for c in ["大工业电量", "居民生活", "商业用电", "制造业", "工业", "非普工业"] if c in df.columns]
    extra_cols = []
    for var in category_cols:
        for lag in [1, 7, 14, 28]:
            col = f"{var}_lag{lag}"
            feat[col] = df[var].shift(lag)
            extra_cols.append(col)
    cols = own_cols + extra_cols + target_cols
    if with_segments:
        cols += [f"target_{c}" for c in segment_cols]
    return feat.dropna(subset=["y_future"] + cols).copy(), cols


def run_segment_model() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_segment_base()
    pred_rows = []
    metric_rows = []
    for horizon in HORIZONS:
        for model_name, with_segments in [
            ("direct_category_weather_holiday", False),
            ("direct_category_weather_holiday_segment", True),
        ]:
            feat, cols = direct_segment_frame(df, horizon, with_segments)
            target_date = feat["日期"] + pd.to_timedelta(horizon, unit="D")
            train = feat[target_date < TEST_START]
            test = feat[(target_date >= TEST_START) & (target_date <= TEST_END)]
            test_target_date = test["日期"] + pd.to_timedelta(horizon, unit="D")
            if len(train) < 100 or len(test) == 0:
                continue
            model = fit_ridge(train, "y_future", cols)
            pred = model.predict(test[cols])
            metric_rows.append({"horizon": horizon, "model": model_name, **metrics(test["y_future"].to_numpy(float), pred), "n": len(test)})
            pred_rows.append(
                pd.DataFrame(
                    {
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
    metrics_by_h = pd.DataFrame(metric_rows).sort_values(["model", "horizon"])
    overall_rows = []
    for model, g in preds.groupby("model"):
        overall_rows.append({"model": model, **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)), "n": len(g)})
    overall = pd.DataFrame(overall_rows).sort_values("rmse")
    preds.to_csv(OUT / "forecast_holiday_segment_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_by_h.to_csv(OUT / "forecast_holiday_segment_by_horizon.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT / "forecast_holiday_segment_overall.csv", index=False, encoding="utf-8-sig")
    return preds, overall


def decompose_segment(preds: pd.DataFrame) -> pd.DataFrame:
    cal = add_holiday_segment_columns(ensure_calendar())
    cols = ["日期", "holiday_name", "is_public_holiday", "is_rest_day", "holiday_segment"]
    cal = cal[cols].rename(columns={"日期": "target_date"})
    preds = preds.copy()
    preds["target_date"] = pd.to_datetime(preds["target_date"])
    merged = preds.merge(cal, on="target_date", how="left")
    rows = []
    for (model, is_holiday), g in merged.groupby(["model", "is_public_holiday"]):
        rows.append({"model": model, "is_public_holiday": is_holiday, **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)), "n": len(g)})
    by_h = pd.DataFrame(rows).sort_values(["is_public_holiday", "model"])
    by_h.to_csv(OUT / "forecast_holiday_segment_error_by_public_holiday.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (model, name), g in merged[merged["holiday_name"].fillna("") != ""].groupby(["model", "holiday_name"]):
        rows.append({"model": model, "holiday_name": name, **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)), "n": len(g)})
    by_name = pd.DataFrame(rows).sort_values(["holiday_name", "model"])
    by_name.to_csv(OUT / "forecast_holiday_segment_error_by_name.csv", index=False, encoding="utf-8-sig")
    return by_h


def write_note(overall: pd.DataFrame, by_h: pd.DataFrame) -> None:
    overall_lines = ["| 模型 | MAPE | RMSE | 样本数 |", "| --- | ---: | ---: | ---: |"]
    for _, r in overall.iterrows():
        overall_lines.append(f"| {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} | {int(r['n'])} |")
    h_lines = ["| 模型 | 是否公共假日 | MAPE | RMSE | 样本数 |", "| --- | --- | ---: | ---: | ---: |"]
    for _, r in by_h.iterrows():
        label = "是" if int(r["is_public_holiday"]) == 1 else "否"
        h_lines.append(f"| {r['model']} | {label} | {pct(r['mape'])} | {r['rmse']:.2f} | {int(r['n'])} |")
    md = f"""
# 高误差长假分段模型

## 1. 为什么做分段

上一轮误差分解显示，公共假日 MAPE 约 5.22%，国庆中秋 MAPE 约 5.99%，明显高于普通日期。单纯用“是否节假日”还不够，因为长假内部也有结构：节前备产、节中停产/消费、节后恢复。

## 2. 新增特征

本轮为目标日期构造：

- `is_pre_holiday3`：是否节前 3 天。
- `is_post_holiday3`：是否节后 3 天。
- `holiday_day_index`：节中第几天。
- `days_to_holiday_start`：距离假期开始还有几天。
- `days_from_holiday_end`：距离假期结束已经几天。
- `seg_*`：节前、节中、节后按假日名称展开的哑变量。

## 3. 整体结果

{chr(10).join(overall_lines)}

## 4. 公共假日分解

{chr(10).join(h_lines)}

## 5. 解释

如果分段模型整体或公共假日误差下降，说明长假位置特征有效。如果下降不明显，说明长假冲击还需要更细的行业、地区、天气或人工规则解释。

## 6. 输出

- `Experiments/outputs/forecast_holiday_segment_overall.csv`
- `Experiments/outputs/forecast_holiday_segment_by_horizon.csv`
- `Experiments/outputs/forecast_holiday_segment_error_by_public_holiday.csv`
- `Experiments/outputs/forecast_holiday_segment_error_by_name.csv`
"""
    write_md(NOTES / f"高误差长假分段模型-{DATE}.md", md)
    report = f"""
# 福建实证高误差长假分段模型报告

本轮对全省合计直接多步预测加入节前/节中/节后分段变量。入口：

- `Notes/高误差长假分段模型-{DATE}.md`
- `Experiments/outputs/forecast_holiday_segment_overall.csv`
- `Experiments/outputs/forecast_holiday_segment_error_by_public_holiday.csv`
"""
    write_md(WRITING / f"福建实证高误差长假分段模型报告-{DATE}.md", report)


def run_all() -> None:
    preds, overall = run_segment_model()
    by_h = decompose_segment(preds)
    write_note(overall, by_h)


if __name__ == "__main__":
    run_all()
