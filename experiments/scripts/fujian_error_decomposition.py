from __future__ import annotations

import pandas as pd

from fujian_calendar_iteration import ensure_calendar
from fujian_pipeline import NOTES, OUT, WRITING, metrics, pct, write_md


DATE = "2026-05-06"


def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error"] = out["actual"] - out["prediction"]
    out["abs_error"] = out["error"].abs()
    out["ape"] = out["abs_error"] / out["actual"].abs()
    return out


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec.update(metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float)))
        rec["n"] = len(g)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols)


def decompose() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal = ensure_calendar()[["日期", "holiday_name", "is_public_holiday", "is_adjusted_workday", "is_rest_day", "is_national_holiday"]].copy()
    cal["日期"] = pd.to_datetime(cal["日期"])

    direct = pd.read_csv(OUT / "forecast_direct_multistep_total_predictions.csv")
    direct = direct[direct["model"] == "direct_category_weather_holiday"].copy()
    direct["target_date"] = pd.to_datetime(direct["target_date"])
    direct = direct.merge(cal.rename(columns={"日期": "target_date"}), on="target_date", how="left")
    direct = add_errors(direct)
    by_holiday = summarize_group(direct, ["is_public_holiday"])
    by_rest = summarize_group(direct, ["is_rest_day"])
    by_name = summarize_group(direct[direct["holiday_name"].fillna("") != ""], ["holiday_name"])
    by_horizon = summarize_group(direct, ["horizon", "is_public_holiday"])
    by_holiday.to_csv(OUT / "forecast_error_direct_by_public_holiday.csv", index=False, encoding="utf-8-sig")
    by_rest.to_csv(OUT / "forecast_error_direct_by_rest_day.csv", index=False, encoding="utf-8-sig")
    by_name.to_csv(OUT / "forecast_error_direct_by_holiday_name.csv", index=False, encoding="utf-8-sig")
    by_horizon.to_csv(OUT / "forecast_error_direct_by_horizon_holiday.csv", index=False, encoding="utf-8-sig")
    return by_holiday, by_rest, by_name


def write_note(by_holiday: pd.DataFrame, by_rest: pd.DataFrame, by_name: pd.DataFrame) -> None:
    holiday_lines = ["| 是否公共假日 | MAPE | RMSE | 样本数 |", "| --- | ---: | ---: | ---: |"]
    for _, r in by_holiday.iterrows():
        label = "是" if int(r["is_public_holiday"]) == 1 else "否"
        holiday_lines.append(f"| {label} | {pct(r['mape'])} | {r['rmse']:.2f} | {int(r['n'])} |")
    rest_lines = ["| 是否实际休息日 | MAPE | RMSE | 样本数 |", "| --- | ---: | ---: | ---: |"]
    for _, r in by_rest.iterrows():
        label = "是" if int(r["is_rest_day"]) == 1 else "否"
        rest_lines.append(f"| {label} | {pct(r['mape'])} | {r['rmse']:.2f} | {int(r['n'])} |")
    name_lines = ["| 假日 | MAPE | RMSE | 样本数 |", "| --- | ---: | ---: | ---: |"]
    for _, r in by_name.sort_values("mape", ascending=False).iterrows():
        name_lines.append(f"| {r['holiday_name']} | {pct(r['mape'])} | {r['rmse']:.2f} | {int(r['n'])} |")
    md = f"""
# 长假与休息日误差分解

## 1. 为什么做这个分解

整体 MAPE 低不代表所有日期都好预测。春节、国庆、中秋等长假会改变工业生产、商业活动和居民生活节奏，模型可能在这些日期系统性失效。因此需要把预测误差按节假日拆开看。

本轮使用 `direct_category_weather_holiday` 的 1-30 日直接多步预测结果。

## 2. 公共假日 vs 非公共假日

{chr(10).join(holiday_lines)}

## 3. 实际休息日 vs 工作日

{chr(10).join(rest_lines)}

## 4. 按假日名称

{chr(10).join(name_lines)}

## 5. 解释

如果公共假日或实际休息日的误差显著高于普通日期，说明即使加入节假日变量，模型仍没有完全捕捉长假冲击。后续可以做：

1. 春节、国庆单独模型。
2. 节前/节中/节后分段变量。
3. 分行业长假误差分解。
4. 对长假期间降低工业变量滞后的权重，增加假日窗口变量。

## 6. 输出

- `Experiments/outputs/forecast_error_direct_by_public_holiday.csv`
- `Experiments/outputs/forecast_error_direct_by_rest_day.csv`
- `Experiments/outputs/forecast_error_direct_by_holiday_name.csv`
- `Experiments/outputs/forecast_error_direct_by_horizon_holiday.csv`
"""
    write_md(NOTES / f"长假与休息日误差分解-{DATE}.md", md)
    report = f"""
# 福建实证长假误差分解补充报告

本轮对 `direct_category_weather_holiday` 的 1-30 日直接多步预测结果按公共假日、实际休息日和假日名称分解误差。入口：

- `Notes/长假与休息日误差分解-{DATE}.md`
- `Experiments/outputs/forecast_error_direct_by_public_holiday.csv`
- `Experiments/outputs/forecast_error_direct_by_holiday_name.csv`
"""
    write_md(WRITING / f"福建实证长假误差分解补充报告-{DATE}.md", report)


def run_all() -> None:
    by_holiday, by_rest, by_name = decompose()
    write_note(by_holiday, by_rest, by_name)


if __name__ == "__main__":
    run_all()
