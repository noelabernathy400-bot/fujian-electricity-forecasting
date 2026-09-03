from __future__ import annotations

import numpy as np
import pandas as pd

from fujian_calendar_iteration import ensure_calendar
from fujian_pipeline import KEY_VARIABLES, NOTES, OUT, PROCESSED, WRITING, f_sf, fdr_bh, ols_rss, write_md
from fujian_weather_iteration import ensure_weather


DATE = "2026-05-07"
CONTROL_WEATHER = ["T2M", "RH2M", "PRECTOTCORR", "CDD18", "HDD18"]
CONTROL_CALENDAR = ["is_public_holiday", "is_adjusted_workday", "is_rest_day", "is_spring_festival", "is_national_holiday"]


def build_weekly_varx_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    weather = ensure_weather()
    calendar = ensure_calendar()
    merged = df.merge(weather, on="日期", how="left").merge(calendar, on="日期", how="left")
    merged = merged[(merged["日期"] >= "2023-01-01") & (merged["日期"] < "2026-01-01")].copy()
    variables = [v for v in KEY_VARIABLES if v in merged.columns]
    weekly_y = merged.set_index("日期")[variables].resample("W-SUN").sum()
    y_trans = np.log1p(weekly_y).diff()
    weekly_weather = merged.set_index("日期")[CONTROL_WEATHER].resample("W-SUN").mean()
    weekly_calendar = merged.set_index("日期")[CONTROL_CALENDAR].resample("W-SUN").sum()
    x = pd.concat([y_trans, weekly_weather, weekly_calendar], axis=1).dropna()
    x.to_csv(PROCESSED / "varx_weekly_logdiff_weather_calendar.csv", encoding="utf-8-sig")
    return x


def varx_edge(data: pd.DataFrame, cause: str, effect: str, controls: list[str], lag: int = 2) -> dict[str, float]:
    y = data[effect].to_numpy(float)
    x = data[cause].to_numpy(float)
    ctrl = {c: data[c].to_numpy(float) for c in controls}
    rows = []
    for t in range(lag, len(data)):
        y_lags = [y[t - i] for i in range(1, lag + 1)]
        x_lags = [x[t - i] for i in range(1, lag + 1)]
        c_lags = []
        for c in controls:
            c_lags.extend([ctrl[c][t - i] for i in range(1, lag + 1)])
        vals = [y[t]] + y_lags + x_lags + c_lags
        if np.all(np.isfinite(vals)):
            rows.append((y[t], y_lags, x_lags, c_lags))
    if len(rows) < max(40, lag * (len(controls) + 3) * 3):
        return {"f_stat": np.nan, "p_value": 1.0, "n": len(rows), "df2": np.nan}
    yy = np.array([r[0] for r in rows], dtype=float)
    ylags = np.array([r[1] for r in rows], dtype=float)
    xlags = np.array([r[2] for r in rows], dtype=float)
    clags = np.array([r[3] for r in rows], dtype=float)
    Xr = np.column_stack([np.ones(len(yy)), ylags, clags])
    Xu = np.column_stack([np.ones(len(yy)), ylags, clags, xlags])
    rss_r, _ = ols_rss(yy, Xr)
    rss_u, ku = ols_rss(yy, Xu)
    df1 = lag
    df2 = len(yy) - ku
    if rss_u <= 0 or df2 <= 0:
        return {"f_stat": np.nan, "p_value": 1.0, "n": len(rows), "df2": df2}
    f = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return {"f_stat": f, "p_value": f_sf(f, df1, df2), "n": len(rows), "df2": df2}


def run_varx_screen() -> pd.DataFrame:
    data = build_weekly_varx_data()
    variables = [v for v in KEY_VARIABLES if v in data.columns]
    stable = pd.read_csv(OUT / "granger_rolling_window_stability.csv")
    candidates = stable.sort_values(["pass_count", "min_q"], ascending=[False, True]).head(80)
    rows = []
    for _, edge in candidates.iterrows():
        cause = edge["cause"]
        effect = edge["effect"]
        if cause not in variables or effect not in variables:
            continue
        variable_controls = [v for v in ["合计", "工业", "制造业", "大工业电量", "非普工业", "居民生活", "商业用电"] if v in variables and v not in [cause, effect]]
        controls = variable_controls + [c for c in CONTROL_WEATHER + CONTROL_CALENDAR if c in data.columns]
        res = varx_edge(data, cause, effect, controls, lag=2)
        rows.append(
            {
                "cause": cause,
                "effect": effect,
                "lag_weeks": 2,
                "controls": ";".join(controls),
                "ordinary_stability_score": edge["stability_score"],
                "ordinary_min_q": edge["min_q"],
                **res,
            }
        )
    out = pd.DataFrame(rows)
    out["q_value_fdr"] = fdr_bh(out["p_value"])
    out["pass_fdr_05"] = out["q_value_fdr"] < 0.05
    out = out.sort_values(["q_value_fdr", "p_value"])
    out.to_csv(OUT / "varx_weather_calendar_edge_screen.csv", index=False, encoding="utf-8-sig")
    return out


def write_note(out: pd.DataFrame) -> None:
    passed = out[out["pass_fdr_05"]]
    lines = ["| 起点 | 终点 | p值 | FDR q值 | 是否通过 |", "| --- | --- | ---: | ---: | --- |"]
    for _, r in out.head(20).iterrows():
        lines.append(f"| {r['cause']} | {r['effect']} | {r['p_value']:.4g} | {r['q_value_fdr']:.4g} | {bool(r['pass_fdr_05'])} |")
    md = f"""
# VARX 天气节假日控制复检

## 1. 为什么做 VARX 复检

条件 Granger 已经控制部分共同用电变量，但还没有把天气和节假日纳入传导边筛选。VARX 的思想是在自回归系统中加入外生变量 $X$，例如天气、节假日、调休日，从而降低共同外部冲击造成的假边。

## 2. 本轮做法

- 频率：周度。
- 因变量：重点行业和类别用电量的 `log1p` 一阶差分。
- 候选边：普通 Granger 滚动窗口稳定性前 80 条。
- 滞后：2 周。
- 控制变量：共同用电变量滞后 + 周度天气均值 + 周度节假日/调休日天数。

受限模型不含 `cause` 滞后项，非受限模型加入 `cause` 滞后项。若非受限模型显著改善预测，则说明在控制天气和日历后，`cause` 对 `effect` 仍有额外预测贡献。

## 3. 结果

- 复检候选边：{len(out)}
- 通过 FDR 0.05 的边：{len(passed)}

{chr(10).join(lines)}

完整结果：`Experiments/outputs/varx_weather_calendar_edge_screen.csv`

## 4. 解释边界

这一步比普通 Granger 和只控制共同用电变量的条件 Granger 更保守，但仍是预测贡献证据。由于没有政策事件、价格、订单、产值等变量，仍不能写成强因果。
"""
    write_md(NOTES / f"VARX天气节假日控制复检-{DATE}.md", md)
    report = f"""
# 福建实证 VARX 天气节假日控制复检报告

本轮对普通 Granger 稳定边做 VARX 风格复检，加入天气和节假日控制。通过 FDR 0.05 的边数为 {len(passed)}。

入口：

- `Notes/VARX天气节假日控制复检-{DATE}.md`
- `Experiments/outputs/varx_weather_calendar_edge_screen.csv`
"""
    write_md(WRITING / f"福建实证VARX天气节假日控制复检报告-{DATE}.md", report)


def run_all() -> None:
    out = run_varx_screen()
    write_note(out)


if __name__ == "__main__":
    run_all()
