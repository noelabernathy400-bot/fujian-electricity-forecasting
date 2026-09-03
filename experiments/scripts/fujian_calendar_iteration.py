from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from fujian_pipeline import FIG, NOTES, OUT, PROCESSED, WRITING, fit_ridge, metrics, pct, write_md
from fujian_second_iteration import feature_frame, svg_bar_percent
from fujian_weather_iteration import ensure_weather


DATE = "2026-05-06"
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-01-09")
CALENDAR_PATH = PROCESSED / "calendar_cn_holidays_2023_2026.csv"


HOLIDAY_RANGES = [
    ("2022-12-31", "2023-01-02", "元旦"),
    ("2023-01-21", "2023-01-27", "春节"),
    ("2023-04-05", "2023-04-05", "清明"),
    ("2023-04-29", "2023-05-03", "劳动节"),
    ("2023-06-22", "2023-06-24", "端午"),
    ("2023-09-29", "2023-10-06", "中秋国庆"),
    ("2023-12-30", "2024-01-01", "元旦"),
    ("2024-02-10", "2024-02-17", "春节"),
    ("2024-04-04", "2024-04-06", "清明"),
    ("2024-05-01", "2024-05-05", "劳动节"),
    ("2024-06-08", "2024-06-10", "端午"),
    ("2024-09-15", "2024-09-17", "中秋"),
    ("2024-10-01", "2024-10-07", "国庆"),
    ("2025-01-01", "2025-01-01", "元旦"),
    ("2025-01-28", "2025-02-04", "春节"),
    ("2025-04-04", "2025-04-06", "清明"),
    ("2025-05-01", "2025-05-05", "劳动节"),
    ("2025-05-31", "2025-06-02", "端午"),
    ("2025-10-01", "2025-10-08", "国庆中秋"),
    ("2026-01-01", "2026-01-03", "元旦"),
]

ADJUSTED_WORKDAYS = {
    "2023-01-28",
    "2023-01-29",
    "2023-04-23",
    "2023-05-06",
    "2023-06-25",
    "2023-10-07",
    "2023-10-08",
    "2024-02-04",
    "2024-02-18",
    "2024-04-07",
    "2024-04-28",
    "2024-05-11",
    "2024-09-14",
    "2024-09-29",
    "2024-10-12",
    "2025-01-26",
    "2025-02-08",
    "2025-04-27",
    "2025-09-28",
    "2025-10-11",
    "2026-01-04",
}

MAJOR_HOLIDAYS = ["春节", "国庆", "中秋国庆", "国庆中秋"]


def daterange(start: str, end: str) -> list[date]:
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    days = []
    cur = s
    while cur <= e:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def build_calendar() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", "2026-01-09", freq="D")
    holiday_map: dict[date, str] = {}
    for start, end, name in HOLIDAY_RANGES:
        for d in daterange(start, end):
            holiday_map[d] = name
    rows = []
    for ts in dates:
        d = ts.date()
        name = holiday_map.get(d, "")
        is_holiday = int(name != "")
        is_adjusted = int(d.isoformat() in ADJUSTED_WORKDAYS)
        is_weekend = int(ts.dayofweek >= 5)
        is_rest_day = int((is_weekend or is_holiday) and not is_adjusted)
        rows.append(
            {
                "日期": ts,
                "holiday_name": name,
                "is_public_holiday": is_holiday,
                "is_adjusted_workday": is_adjusted,
                "is_weekend": is_weekend,
                "is_rest_day": is_rest_day,
                "is_spring_festival": int(name == "春节"),
                "is_national_holiday": int(name in ["国庆", "中秋国庆", "国庆中秋"]),
                "is_labor_holiday": int(name == "劳动节"),
                "is_minor_holiday": int(name in ["元旦", "清明", "端午", "中秋"]),
            }
        )
    cal = pd.DataFrame(rows)
    for col in ["is_spring_festival", "is_national_holiday"]:
        idx = np.where(cal[col].to_numpy(int) == 1)[0]
        around = np.zeros(len(cal), dtype=int)
        for i in idx:
            lo = max(0, i - 3)
            hi = min(len(cal), i + 4)
            around[lo:hi] = 1
        cal[f"{col}_window3"] = around
    cal.to_csv(CALENDAR_PATH, index=False, encoding="utf-8-sig")
    return cal


def ensure_calendar() -> pd.DataFrame:
    if CALENDAR_PATH.exists():
        return pd.read_csv(CALENDAR_PATH, parse_dates=["日期"])
    return build_calendar()


def calendar_weather_forecast() -> pd.DataFrame:
    cal = ensure_calendar()
    weather = ensure_weather()
    province = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    df = province.merge(weather, on="日期", how="left").merge(cal, on="日期", how="left")
    weather_cols = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "CDD18", "HDD18", "CDD22", "TEMP_RANGE"]
    holiday_cols = [
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
    category_cols = [c for c in ["大工业电量", "居民生活", "商业用电", "制造业", "工业", "非普工业"] if c in df.columns]
    targets = [t for t in ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电"] if t in df.columns]
    rows = []
    pred_rows = []
    for target in targets:
        specs = []
        base_feat, base_cols = feature_frame(df[["日期", target]].copy(), target)
        specs.append(("ridge_own_lags_calendar", base_feat, base_cols))
        w_cols_input = ["日期", target] + weather_cols
        w_feat, w_cols = feature_frame(df[w_cols_input].copy(), target)
        w_feat = w_feat.merge(df[["日期"] + weather_cols], on="日期", how="left")
        w_cols = w_cols + weather_cols
        specs.append(("ridge_own_weather", w_feat.dropna(subset=[target] + w_cols), w_cols))
        h_cols_input = ["日期", target] + holiday_cols
        h_feat, h_cols = feature_frame(df[h_cols_input].copy(), target)
        h_feat = h_feat.merge(df[["日期"] + holiday_cols], on="日期", how="left")
        h_cols = h_cols + holiday_cols
        specs.append(("ridge_own_holiday", h_feat.dropna(subset=[target] + h_cols), h_cols))
        wh_cols_input = ["日期", target] + weather_cols + holiday_cols
        wh_feat, wh_cols = feature_frame(df[wh_cols_input].copy(), target)
        wh_feat = wh_feat.merge(df[["日期"] + weather_cols + holiday_cols], on="日期", how="left")
        wh_cols = wh_cols + weather_cols + holiday_cols
        specs.append(("ridge_own_weather_holiday", wh_feat.dropna(subset=[target] + wh_cols), wh_cols))
        if target == "合计":
            cwh_input = ["日期", target] + category_cols + weather_cols + holiday_cols
            cwh_feat, cwh_cols = feature_frame(df[cwh_input].copy(), target, category_cols)
            cwh_feat = cwh_feat.merge(df[["日期"] + weather_cols + holiday_cols], on="日期", how="left")
            cwh_cols = cwh_cols + weather_cols + holiday_cols
            specs.append(("ridge_category_weather_holiday", cwh_feat.dropna(subset=[target] + cwh_cols), cwh_cols))
        for model_name, feat, cols in specs:
            train = feat[feat["日期"] < TEST_START]
            test = feat[(feat["日期"] >= TEST_START) & (feat["日期"] <= TEST_END)]
            if len(train) < 100 or len(test) == 0:
                continue
            model = fit_ridge(train, target, cols)
            pred = model.predict(test[cols])
            rows.append({"target": target, "model": model_name, **metrics(test[target].to_numpy(float), pred), "n": len(test)})
            tmp = test[["日期", target]].copy().rename(columns={target: "actual"})
            tmp["target"] = target
            tmp["model"] = model_name
            tmp["prediction"] = pred
            pred_rows.append(tmp)
    res = pd.DataFrame(rows).sort_values(["target", "rmse"])
    preds = pd.concat(pred_rows, ignore_index=True)
    res.to_csv(OUT / "forecast_calendar_weather_metrics_daily.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT / "forecast_calendar_weather_predictions_daily.csv", index=False, encoding="utf-8-sig")
    gains = []
    for target, g in res.groupby("target"):
        base = g[g["model"] == "ridge_own_lags_calendar"]
        best = g.sort_values("rmse").head(1)
        if not base.empty and not best.empty:
            gains.append({"target": target, "best_model": best["model"].iloc[0], "best_rmse_gain_pct": (1 - float(best["rmse"].iloc[0]) / float(base["rmse"].iloc[0])) * 100})
    gain_df = pd.DataFrame(gains).sort_values("best_rmse_gain_pct", ascending=False)
    gain_df.to_csv(OUT / "forecast_calendar_weather_best_gain.csv", index=False, encoding="utf-8-sig")
    if not gain_df.empty:
        plot = gain_df.copy()
        plot["plot_value"] = plot["best_rmse_gain_pct"].clip(lower=0)
        svg_bar_percent(
            FIG / "forecast_calendar_weather_best_gain.svg",
            plot["target"].tolist(),
            plot["plot_value"].tolist(),
            "加入天气/节假日后最优 RMSE 改善幅度（负值按 0 显示）",
        )
    return res


def write_calendar_note(res: pd.DataFrame) -> None:
    lines = ["| 目标 | 模型 | MAPE | RMSE |", "| --- | --- | ---: | ---: |"]
    for _, r in res.iterrows():
        lines.append(f"| {r['target']} | {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} |")
    total_best = res[res["target"] == "合计"].sort_values("rmse").iloc[0]
    md = f"""
# 节假日与调休日变量实验

## 1. 为什么要加入节假日

用电量不仅受星期和季节影响，还会被春节、国庆、调休工作日等制度性日历冲击改变。春节期间工业和商业活动通常下降，居民生活负荷可能呈现不同模式；国庆和中秋长假也会改变生产和消费节奏。

## 2. 数据来源

本轮根据国务院办公厅关于 2023-2026 年部分节假日安排的通知手工编码节假日和调休日。主要核验来源：

- 2023 年：宿州市人民政府转载国务院办公厅通知，列明 2023 年各节假日安排。
- 2024 年：中国政府网政策库《国务院办公厅关于2024年部分节假日安排的通知》。
- 2025 年：青山区人民政府转载中国政府网通知，列明 2025 年各节假日安排。
- 2026 年：七一网转载新华社《国务院办公厅关于2026年部分节假日安排的通知》。

生成表：`Data/processed/calendar_cn_holidays_2023_2026.csv`

## 3. 特征

本轮构造：

- `is_public_holiday`：是否官方放假日。
- `is_adjusted_workday`：是否调休上班日。
- `is_weekend`：是否周末。
- `is_rest_day`：是否实际休息日。
- `is_spring_festival`：是否春节假期。
- `is_national_holiday`：是否国庆或国庆中秋合并假期。
- `is_labor_holiday`：是否劳动节。
- `is_minor_holiday`：是否元旦、清明、端午、中秋。
- `is_spring_festival_window3`：春节前后 3 日窗口。
- `is_national_holiday_window3`：国庆前后 3 日窗口。

## 4. 结果

{chr(10).join(lines)}

全省 `合计` 在节假日/天气综合实验中最优模型为 **{total_best['model']}**，MAPE 为 **{pct(total_best['mape'])}**，RMSE 为 **{total_best['rmse']:.2f}**。

## 5. 解释边界

节假日和调休日是未来已知变量，适合进入真实预测。天气变量在历史回测中使用的是实际天气，真实未来预测时需要替换为天气预报或天气情景。

## 6. 下一步

1. 做直接多步预测，避免 30 日递推误差累积。
2. 把节假日和天气都加入多步预测。
3. 对春节、国庆做单独误差分解，看模型是否仍在长假期间失效。
"""
    write_md(NOTES / f"节假日与调休日变量实验-{DATE}.md", md)
    report = f"""
# 福建实证节假日与调休日变量补充报告

本轮将中国 2023-2026 年官方节假日和调休日编码为日历特征，并与天气变量一起进入预测模型。核心输出：

- `Notes/节假日与调休日变量实验-{DATE}.md`
- `Data/processed/calendar_cn_holidays_2023_2026.csv`
- `Experiments/outputs/forecast_calendar_weather_metrics_daily.csv`
- `Experiments/figures/forecast_calendar_weather_best_gain.svg`

全省合计最优模型为 `{total_best['model']}`，MAPE 为 {pct(total_best['mape'])}。
"""
    write_md(WRITING / f"福建实证节假日变量补充报告-{DATE}.md", report)


def run_all() -> None:
    build_calendar()
    res = calendar_weather_forecast()
    write_calendar_note(res)


if __name__ == "__main__":
    run_all()
