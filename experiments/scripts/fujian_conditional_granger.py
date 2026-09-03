from __future__ import annotations

import numpy as np
import pandas as pd

from fujian_pipeline import KEY_VARIABLES, NOTES, OUT, PROCESSED, WRITING, f_sf, fdr_bh, ols_rss, write_md


DATE = "2026-05-06"
CONTROL_POOL = ["合计", "工业", "制造业", "大工业电量", "非普工业", "居民生活", "商业用电"]


def conditional_granger_pair(data: pd.DataFrame, cause: str, effect: str, controls: list[str], lag: int) -> dict[str, float]:
    rows = []
    y = data[effect].to_numpy(float)
    x = data[cause].to_numpy(float)
    ctrl_arrays = {c: data[c].to_numpy(float) for c in controls}
    for t in range(lag, len(data)):
        y_lags = [y[t - i] for i in range(1, lag + 1)]
        x_lags = [x[t - i] for i in range(1, lag + 1)]
        c_lags = []
        for c in controls:
            c_lags.extend([ctrl_arrays[c][t - i] for i in range(1, lag + 1)])
        vals = [y[t]] + y_lags + x_lags + c_lags
        if np.all(np.isfinite(vals)):
            rows.append((y[t], y_lags, x_lags, c_lags))
    if len(rows) < max(30, lag * (len(controls) + 3) * 4):
        return {"f_stat": np.nan, "p_value": 1.0, "n": len(rows), "df2": np.nan}
    yy = np.array([r[0] for r in rows], dtype=float)
    ylags = np.array([r[1] for r in rows], dtype=float)
    xlags = np.array([r[2] for r in rows], dtype=float)
    clags = np.array([r[3] for r in rows], dtype=float) if controls else np.empty((len(rows), 0))
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


def run_conditional() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    df = df[(df["日期"] >= "2023-01-01") & (df["日期"] < "2026-01-01")]
    variables = [v for v in KEY_VARIABLES if v in df.columns]
    weekly = df.set_index("日期")[variables].resample("W-SUN").sum()
    trans = np.log1p(weekly).diff().dropna()
    stable = pd.read_csv(OUT / "granger_rolling_window_stability.csv").head(40)
    rows = []
    for _, edge in stable.iterrows():
        cause = edge["cause"]
        effect = edge["effect"]
        if cause not in trans.columns or effect not in trans.columns:
            continue
        base_lag = int(round(float(edge["median_lag"])))
        base_lag = max(1, min(4, base_lag))
        controls = [c for c in CONTROL_POOL if c in trans.columns and c not in [cause, effect]]
        res = conditional_granger_pair(trans, cause, effect, controls, base_lag)
        rows.append(
            {
                "cause": cause,
                "effect": effect,
                "lag_weeks": base_lag,
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
    out.to_csv(OUT / "conditional_granger_stable_edges.csv", index=False, encoding="utf-8-sig")
    return out


def write_note(out: pd.DataFrame) -> None:
    lines = ["| 起点 | 终点 | 滞后 | p值 | FDR q值 | 是否通过 | 控制变量 |", "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for _, r in out.head(20).iterrows():
        lines.append(
            f"| {r['cause']} | {r['effect']} | {int(r['lag_weeks'])} | {r['p_value']:.4g} | {r['q_value_fdr']:.4g} | {bool(r['pass_fdr_05'])} | {r['controls']} |"
        )
    passed = out[out["pass_fdr_05"]]
    md = f"""
# 条件 Granger 稳定边复检

## 1. 为什么做条件 Granger

普通 Granger 检验只比较 `cause` 的历史是否改善 `effect` 的预测，但可能受到共同冲击影响。例如工业景气同时影响多个行业，两个行业之间就可能出现看似显著的时序领先关系。

条件 Granger 的思路是：在模型里加入一组共同控制变量的滞后项，再检验 `cause` 的滞后项是否还有额外预测贡献。

## 2. 模型比较

受限模型：

$$
y_t = f(y_{{t-1:t-p}}, Z_{{t-1:t-p}})+\\varepsilon_t
$$

非受限模型：

$$
y_t = f(y_{{t-1:t-p}}, Z_{{t-1:t-p}}, x_{{t-1:t-p}})+\\varepsilon_t
$$

其中 $Z$ 是共同控制变量集合，本轮使用 `合计`、`工业`、`制造业`、`大工业电量`、`非普工业`、`居民生活`、`商业用电` 中除 cause/effect 外的可用变量。

## 3. 结果

- 复检边数：{len(out)}
- 通过 FDR 0.05 的条件边数：{len(passed)}

{chr(10).join(lines)}

完整结果：`Experiments/outputs/conditional_granger_stable_edges.csv`

## 4. 解释边界

条件 Granger 比普通 Granger 更保守，但仍不是强因果识别。它只说明在控制若干共同变量的历史之后，某个变量的历史仍对目标有额外预测贡献。若要提高到更强因果证据，还需要天气、节假日、政策事件、价格、宏观变量、产业链约束或准实验设计。
"""
    write_md(NOTES / f"条件Granger稳定边复检-{DATE}.md", md)
    report = f"""
# 福建实证条件 Granger 补充报告

本轮对滚动窗口稳定边做条件 Granger 复检，加入共同控制变量滞后项。通过 FDR 0.05 的边数为 {len(passed)}。

入口：

- `Notes/条件Granger稳定边复检-{DATE}.md`
- `Experiments/outputs/conditional_granger_stable_edges.csv`
"""
    write_md(WRITING / f"福建实证条件Granger补充报告-{DATE}.md", report)


def run_all() -> None:
    out = run_conditional()
    write_note(out)


if __name__ == "__main__":
    run_all()
