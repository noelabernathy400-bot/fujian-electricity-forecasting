from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from fujian_pipeline import (
    CITY_COLS,
    FIG,
    NOTES,
    OUT,
    PROCESSED,
    WRITING,
    fit_ridge,
    make_lag_features,
    metrics,
    pct,
    write_md,
)
from fujian_second_iteration import CALENDAR_COLS, feature_frame, svg_bar_percent


DATE = "2026-05-06"
START = "20230101"
END = "20260109"
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-01-09")
WEATHER_RAW = PROCESSED / "weather_nasa_power_daily_city.csv"
WEATHER_PROVINCE = PROCESSED / "weather_nasa_power_daily_province_weighted.csv"


CITY_COORDS = {
    "福州": (26.0745, 119.2965),
    "厦门": (24.4798, 118.0894),
    "宁德": (26.6657, 119.5482),
    "莆田": (25.4541, 119.0078),
    "泉州": (24.8741, 118.6757),
    "漳州": (24.5130, 117.6471),
    "龙岩": (25.0751, 117.0175),
    "三明": (26.2634, 117.6387),
    "南平": (26.6418, 118.1777),
}

PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR"]


def nasa_power_url(lat: float, lon: float) -> str:
    query = {
        "parameters": ",".join(PARAMS),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": START,
        "end": END,
        "format": "JSON",
        "time-standard": "LST",
    }
    return "https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(query)


def fetch_city_weather(city: str, lat: float, lon: float) -> tuple[pd.DataFrame, dict[str, object]]:
    url = nasa_power_url(lat, lon)
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    params = payload["properties"]["parameter"]
    rows = []
    for date_key in sorted(params["T2M"].keys()):
        row = {"city": city, "date": pd.to_datetime(date_key, format="%Y%m%d")}
        for p in PARAMS:
            val = params[p].get(date_key)
            row[p] = np.nan if val in [-999, -999.0, None] else float(val)
        rows.append(row)
    meta = {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "url": url,
        "rows": len(rows),
        "source": "NASA POWER Daily Point API",
    }
    return pd.DataFrame(rows), meta


def fetch_weather() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    logs = []
    for city, (lat, lon) in CITY_COORDS.items():
        df, meta = fetch_city_weather(city, lat, lon)
        all_rows.append(df)
        logs.append(meta)
        time.sleep(0.2)
    weather = pd.concat(all_rows, ignore_index=True)
    weather.to_csv(WEATHER_RAW, index=False, encoding="utf-8-sig")
    pd.DataFrame(logs).to_csv(OUT / "weather_nasa_power_request_log.csv", index=False, encoding="utf-8-sig")

    province = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"])
    city_cols = [c for c in CITY_COLS if c in province.columns and c in CITY_COORDS]
    weights = province[province["日期"] < "2026-01-01"][city_cols].sum()
    weights = weights / weights.sum()
    weight_df = weights.reset_index()
    weight_df.columns = ["city", "weight_2023_2025_power_share"]
    weight_df.to_csv(OUT / "weather_city_weights_from_power_share.csv", index=False, encoding="utf-8-sig")

    merged = weather.merge(weight_df, on="city", how="left")
    rows = []
    for date, g in merged.groupby("date"):
        row = {"日期": date}
        for p in PARAMS:
            row[p] = float((g[p] * g["weight_2023_2025_power_share"]).sum())
        rows.append(row)
    prov = pd.DataFrame(rows).sort_values("日期")
    prov["CDD18"] = (prov["T2M"] - 18).clip(lower=0)
    prov["HDD18"] = (18 - prov["T2M"]).clip(lower=0)
    prov["CDD22"] = (prov["T2M"] - 22).clip(lower=0)
    prov["TEMP_RANGE"] = prov["T2M_MAX"] - prov["T2M_MIN"]
    prov.to_csv(WEATHER_PROVINCE, index=False, encoding="utf-8-sig")
    return weather, prov


def ensure_weather() -> pd.DataFrame:
    if WEATHER_PROVINCE.exists():
        return pd.read_csv(WEATHER_PROVINCE, parse_dates=["日期"])
    _, prov = fetch_weather()
    return prov


def weather_feature_frame(df: pd.DataFrame, target: str, weather_cols: list[str], category_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    category_cols = category_cols or []
    feat, cols = feature_frame(df[["日期", target] + category_cols].copy(), target, category_cols)
    weather = df[["日期"] + weather_cols].copy()
    feat = feat.merge(weather, on="日期", how="left")
    cols = cols + weather_cols
    return feat.dropna(subset=[target] + cols).copy(), cols


def weather_forecast_experiment() -> pd.DataFrame:
    weather = ensure_weather()
    province = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    df = province.merge(weather, on="日期", how="left")
    weather_cols = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "CDD18", "HDD18", "CDD22", "TEMP_RANGE"]
    category_cols = [c for c in ["大工业电量", "居民生活", "商业用电", "制造业", "工业", "非普工业"] if c in df.columns]
    targets = [t for t in ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电"] if t in df.columns]
    rows = []
    pred_rows = []
    for target in targets:
        specs = []
        base_feat, base_cols = feature_frame(df[["日期", target]].copy(), target)
        specs.append(("ridge_own_lags_calendar", base_feat, base_cols))
        w_feat, w_cols = weather_feature_frame(df[["日期", target] + weather_cols].copy(), target, weather_cols)
        specs.append(("ridge_own_weather", w_feat, w_cols))
        if target == "合计":
            cw_feat, cw_cols = weather_feature_frame(df[["日期", target] + category_cols + weather_cols].copy(), target, weather_cols, category_cols)
            specs.append(("ridge_category_weather", cw_feat, cw_cols))
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
    res.to_csv(OUT / "forecast_weather_metrics_daily.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT / "forecast_weather_predictions_daily.csv", index=False, encoding="utf-8-sig")

    gains = []
    for target, g in res.groupby("target"):
        base = g[g["model"] == "ridge_own_lags_calendar"]
        weather_only = g[g["model"] == "ridge_own_weather"]
        if not base.empty and not weather_only.empty:
            gains.append(
                {
                    "target": target,
                    "weather_rmse_gain_pct": (1 - float(weather_only["rmse"].iloc[0]) / float(base["rmse"].iloc[0])) * 100,
                }
            )
    gain_df = pd.DataFrame(gains).sort_values("weather_rmse_gain_pct", ascending=False)
    gain_df.to_csv(OUT / "forecast_weather_gain.csv", index=False, encoding="utf-8-sig")
    if not gain_df.empty:
        plot = gain_df.copy()
        plot["plot_value"] = plot["weather_rmse_gain_pct"].clip(lower=0)
        svg_bar_percent(
            FIG / "forecast_weather_gain_positive.svg",
            plot["target"].tolist(),
            plot["plot_value"].tolist(),
            "加入天气变量后 RMSE 改善幅度（负值按 0 显示）",
        )
    return res


def write_weather_note(res: pd.DataFrame) -> None:
    lines = ["| 目标 | 模型 | MAPE | RMSE |", "| --- | --- | ---: | ---: |"]
    for _, r in res.iterrows():
        lines.append(f"| {r['target']} | {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} |")
    best_total = res[res["target"] == "合计"].sort_values("rmse").iloc[0]
    md = f"""
# 天气变量加入实验

## 1. 为什么要加入天气

电力负荷预测中，天气通常是最重要的外部变量之一。居民生活和商业用电会受气温影响；工业用电也可能受高温、降雨和生产节奏影响。如果不加入天气，模型只能从历史用电量中间接学习季节性，无法解释同一季节内的冷暖波动。

## 2. 数据来源

本轮使用 NASA POWER Daily Point API，按福建 9 个地市中心点近似抓取日度气象变量：

- `T2M`：2 米日平均气温。
- `T2M_MAX`：2 米日最高气温。
- `T2M_MIN`：2 米日最低气温。
- `RH2M`：2 米相对湿度。
- `PRECTOTCORR`：校正降水。

原始城市天气表：`Data/processed/weather_nasa_power_daily_city.csv`

全省加权天气表：`Data/processed/weather_nasa_power_daily_province_weighted.csv`

接口请求记录：`Experiments/outputs/weather_nasa_power_request_log.csv`

## 3. 全省加权方法

各地市天气按 2023-2025 年地市用电总量份额加权。这样做的直觉是：用电量占比越高的地市，对全省负荷的天气敏感性贡献越大。

如果某地市权重为 $w_i$，天气变量为 $T_{{i,t}}$，则全省加权天气为：

$$
T_t^{{province}}=\\sum_i w_i T_{{i,t}}
$$

## 4. 构造的天气特征

除原始气象变量外，还构造：

$$
CDD18_t=\\max(T_t-18,0)
$$

$$
HDD18_t=\\max(18-T_t,0)
$$

$$
CDD22_t=\\max(T_t-22,0)
$$

`CDD` 表示制冷度日，温度高于阈值越多，制冷需求可能越强；`HDD` 表示采暖度日，温度低于阈值越多，采暖需求可能越强。

## 5. 结果

{chr(10).join(lines)}

全省 `合计` 在天气实验中最优模型为 **{best_total['model']}**，MAPE 为 **{pct(best_total['mape'])}**，RMSE 为 **{best_total['rmse']:.2f}**。

## 6. 解释边界

本轮使用的是历史实际天气，因此可以视为“天气信息是否有解释和预测价值”的诊断实验。真实未来预测时，必须把这些实际天气替换为天气预报或天气场景。否则会产生未来信息泄漏。

## 7. 下一步

1. 加入节假日和调休日变量。
2. 对天气变量做非线性项和分段项。
3. 把天气加入 30 日递推预测时，需要使用未来天气预报或天气场景。
"""
    write_md(NOTES / f"天气变量加入实验-{DATE}.md", md)
    report = f"""
# 福建实证天气变量补充报告

本轮使用 NASA POWER 日度气象数据补充福建 9 个地市天气，并按地市用电份额加权为全省天气变量。实验结果见：

- `Notes/天气变量加入实验-{DATE}.md`
- `Experiments/outputs/forecast_weather_metrics_daily.csv`
- `Experiments/outputs/forecast_weather_gain.csv`
- `Experiments/figures/forecast_weather_gain_positive.svg`

关键边界：本轮使用历史实际天气，主要用于判断天气变量价值；正式预测未来时应替换为天气预报或天气场景。
"""
    write_md(WRITING / f"福建实证天气变量补充报告-{DATE}.md", report)


def run_all() -> None:
    res = weather_forecast_experiment()
    write_weather_note(res)


if __name__ == "__main__":
    run_all()
