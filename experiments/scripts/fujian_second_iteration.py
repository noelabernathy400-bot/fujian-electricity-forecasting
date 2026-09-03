from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from fujian_pipeline import (
    CITY_COLS,
    FIG,
    KEY_VARIABLES,
    NOTES,
    OUT,
    PROCESSED,
    WRITING,
    add_time_features,
    fdr_bh,
    fit_ridge,
    granger_pair,
    make_lag_features,
    metrics,
    pct,
    svg_bar,
    svg_line,
    write_md,
)


DATE = "2026-05-06"
TEST_START = pd.Timestamp("2025-10-01")
TEST_END = pd.Timestamp("2026-01-09")
FULL_END = pd.Timestamp("2026-01-01")
CALENDAR_COLS = [
    "doy_sin",
    "doy_cos",
    "dow_0",
    "dow_1",
    "dow_2",
    "dow_3",
    "dow_4",
    "dow_5",
    "dow_6",
]


def read_province() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"])
    return df.sort_values("日期").reset_index(drop=True)


def svg_line_percent(path: Path, series: list[tuple[str, list[float]]], labels: list[str], title: str) -> None:
    width, height = 1100, 420
    margin = dict(left=70, right=30, top=55, bottom=70)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    vals = [v for _, ys in series for v in ys if pd.notna(v)]
    ymin, ymax = min(vals), max(vals)
    if ymax == ymin:
        ymax = ymin + 1

    def xy(i: int, y: float, n: int) -> tuple[float, float]:
        x = margin["left"] + (i / max(n - 1, 1)) * plot_w
        yy = margin["top"] + (ymax - y) / (ymax - ymin) * plot_h
        return x, yy

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{title}</text>',
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for k in range(5):
        y = margin["top"] + k * plot_h / 4
        val = ymax - k * (ymax - ymin) / 4
        parts.append(f'<line x1="{margin["left"]}" x2="{margin["left"]+plot_w}" y1="{y}" y2="{y}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="62" y="{y+4}" text-anchor="end" font-size="11" font-family="Arial">{val:.1f}%</text>')
    n = len(labels)
    for i in np.linspace(0, n - 1, min(8, n), dtype=int):
        x, _ = xy(int(i), ymin, n)
        parts.append(f'<text x="{x}" y="{height-35}" text-anchor="middle" font-size="11" font-family="Arial">{labels[int(i)]}</text>')
    for si, (name, ys) in enumerate(series):
        pts = []
        for i, y in enumerate(ys):
            if pd.notna(y):
                x, yy = xy(i, y, n)
                pts.append(f"{x:.1f},{yy:.1f}")
        color = colors[si % len(colors)]
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        lx = margin["left"] + si * 205
        parts.append(f'<line x1="{lx}" x2="{lx+24}" y1="{height-16}" y2="{height-16}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+30}" y="{height-12}" font-size="12" font-family="Arial">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_bar_percent(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 980, 520
    margin = dict(left=210, right=70, top=55, bottom=45)
    plot_w = width - margin["left"] - margin["right"]
    bar_h = min(30, (height - margin["top"] - margin["bottom"]) / max(len(labels), 1) * 0.72)
    gap = ((height - margin["top"] - margin["bottom"]) - bar_h * len(labels)) / max(len(labels) - 1, 1)
    vmax = max(values) if values else 1
    if vmax <= 0:
        vmax = 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{title}</text>',
    ]
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = margin["top"] + i * (bar_h + gap)
        w = max(val, 0) / vmax * plot_w
        parts.append(f'<text x="{margin["left"]-10}" y="{y+bar_h*0.68}" text-anchor="end" font-size="12" font-family="Arial">{lab}</text>')
        parts.append(f'<rect x="{margin["left"]}" y="{y}" width="{w}" height="{bar_h}" fill="#2563eb" opacity="0.82"/>')
        parts.append(f'<text x="{margin["left"]+w+6}" y="{y+bar_h*0.68}" font-size="11" font-family="Arial">{val:.2f}%</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def own_feature_cols(target: str) -> list[str]:
    return [
        c
        for c in [
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
    ]


def feature_frame(df: pd.DataFrame, target: str, exog: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    feat = make_lag_features(df, target)
    cols = own_feature_cols(target)
    exog = exog or []
    for var in exog:
        if var == target or var not in df.columns:
            continue
        for lag in [1, 7, 14, 28]:
            col = f"{var}_lag{lag}"
            feat[col] = df[var].shift(lag)
            cols.append(col)
    cols = [c for c in cols if c in feat.columns]
    return feat.dropna(subset=[target] + cols).copy(), cols


def mase_scale(train: pd.DataFrame, target: str, season: int = 7) -> float:
    diff = train[target] - train[target].shift(season)
    scale = float(diff.abs().dropna().mean())
    return scale if np.isfinite(scale) and scale > 0 else float("nan")


def row_for_next_date(hist: pd.DataFrame, target: str, next_date: pd.Timestamp) -> pd.DataFrame:
    row = add_time_features(pd.DataFrame({"日期": [next_date], target: [np.nan]}))
    for lag in [1, 2, 3, 7, 14, 28, 365]:
        row[f"{target}_lag{lag}"] = hist[target].iloc[-lag] if len(hist) >= lag else np.nan
    row[f"{target}_roll7"] = hist[target].iloc[-7:].mean()
    row[f"{target}_roll28"] = hist[target].iloc[-28:].mean()
    return row


def recursive_rolling_origin_total() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = read_province()
    target = "合计"
    cutoffs = [
        pd.Timestamp("2025-04-01"),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-08-01"),
        pd.Timestamp("2025-10-01"),
        pd.Timestamp("2025-11-01"),
        pd.Timestamp("2025-12-01"),
    ]
    rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        train = df[df["日期"] < cutoff].copy()
        feat, cols = feature_frame(train, target)
        model = fit_ridge(feat, target, cols)
        for model_name in ["naive_yesterday", "seasonal_week", "rolling7", "rolling28", "ridge_recursive"]:
            hist = train[["日期", target]].copy().reset_index(drop=True)
            for horizon in range(1, 31):
                next_date = hist["日期"].max() + pd.Timedelta(days=1)
                if next_date > TEST_END:
                    break
                if model_name == "naive_yesterday":
                    pred = float(hist[target].iloc[-1])
                elif model_name == "seasonal_week":
                    pred = float(hist[target].iloc[-7])
                elif model_name == "rolling7":
                    pred = float(hist[target].iloc[-7:].mean())
                elif model_name == "rolling28":
                    pred = float(hist[target].iloc[-28:].mean())
                else:
                    row = row_for_next_date(hist, target, next_date)
                    pred = float(model.predict(row[cols])[0])
                actual = df.loc[df["日期"] == next_date, target]
                if not actual.empty:
                    rows.append(
                        {
                            "cutoff": cutoff.date().isoformat(),
                            "date": next_date.date().isoformat(),
                            "horizon": horizon,
                            "model": model_name,
                            "actual": float(actual.iloc[0]),
                            "prediction": pred,
                        }
                    )
                hist = pd.concat(
                    [hist, pd.DataFrame({"日期": [next_date], target: [pred]})],
                    ignore_index=True,
                )
    preds = pd.DataFrame(rows)
    preds["error"] = preds["actual"] - preds["prediction"]
    preds["abs_error"] = preds["error"].abs()
    preds["ape"] = preds["abs_error"] / preds["actual"].abs()
    scale = mase_scale(df[df["日期"] < cutoffs[-1]], target)
    metric_rows = []
    for model_name, g in preds.groupby("model"):
        m = metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float))
        m["mase"] = float(g["abs_error"].mean() / scale)
        metric_rows.append({"model": model_name, **m, "n": len(g)})
    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse")
    horizon = (
        preds.groupby(["model", "horizon"], as_index=False)
        .agg(mae=("abs_error", "mean"), mape=("ape", "mean"), n=("actual", "size"))
        .sort_values(["model", "horizon"])
    )
    preds.to_csv(OUT / "forecast_rolling_origin_total_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUT / "forecast_rolling_origin_total_metrics.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(OUT / "forecast_rolling_origin_total_by_horizon.csv", index=False, encoding="utf-8-sig")

    labels = [str(i) for i in range(1, 31)]
    series = []
    for model_name in ["naive_yesterday", "seasonal_week", "rolling7", "ridge_recursive"]:
        g = horizon[horizon["model"] == model_name].set_index("horizon").reindex(range(1, 31))
        series.append((model_name, (g["mape"] * 100).tolist()))
    svg_line_percent(FIG / "forecast_rolling_origin_horizon_mape.svg", series, labels, "全省合计滚动起点回测：不同预测步长 MAPE")
    return preds, metrics_df, horizon


def city_forecast() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_province()
    rows = []
    pred_rows = []
    for city in [c for c in CITY_COLS if c in df.columns]:
        feat, cols = feature_frame(df[["日期", city]].copy(), city)
        train = feat[feat["日期"] < TEST_START]
        test = feat[(feat["日期"] >= TEST_START) & (feat["日期"] <= TEST_END)]
        if len(train) < 100 or len(test) == 0:
            continue
        model = fit_ridge(train, city, cols)
        pred_map = {
            "naive_yesterday": test[f"{city}_lag1"].to_numpy(float),
            "seasonal_week": test[f"{city}_lag7"].to_numpy(float),
            "rolling7": test[f"{city}_roll7"].to_numpy(float),
            "ridge_own_lags_calendar": model.predict(test[cols]),
        }
        for model_name, pred in pred_map.items():
            m = metrics(test[city].to_numpy(float), pred)
            rows.append({"city": city, "model": model_name, **m, "n": len(test)})
            tmp = test[["日期", city]].copy()
            tmp["city"] = city
            tmp["model"] = model_name
            tmp["prediction"] = pred
            pred_rows.append(tmp.rename(columns={city: "actual"}))
    metrics_df = pd.DataFrame(rows).sort_values(["city", "rmse"])
    preds = pd.concat(pred_rows, ignore_index=True)
    best = metrics_df.groupby("city", as_index=False).first().sort_values("mape")
    metrics_df.to_csv(OUT / "forecast_city_metrics_daily.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT / "forecast_city_best_metrics.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT / "forecast_city_predictions_daily.csv", index=False, encoding="utf-8-sig")
    best_plot = best.sort_values("mape", ascending=True)
    svg_bar_percent(
        FIG / "forecast_city_best_mape.svg",
        best_plot["city"].tolist(),
        (best_plot["mape"] * 100).tolist(),
        "2025Q4 地市预测最优 MAPE（百分比）",
    )
    return metrics_df, best


def choose_granger_causes(target: str, max_causes: int = 3) -> list[str]:
    path = OUT / "granger_weekly_logdiff_edges.csv"
    if not path.exists():
        return []
    g = pd.read_csv(path)
    if "pass_fdr_05" in g.columns:
        g = g[g["pass_fdr_05"].astype(str).str.lower().isin(["true", "1"])]
    g = g[(g["effect"] == target) & (g["cause"] != target)]
    return g.sort_values("q_value_fdr")["cause"].head(max_causes).tolist()


def causal_feature_forecast() -> pd.DataFrame:
    df = read_province()
    target_candidates = ["合计", "工业", "制造业", "大工业电量", "纺织业", "电子", "非金属矿物制品业", "商业用电", "居民生活"]
    rows = []
    for target in [t for t in target_candidates if t in df.columns]:
        causes = choose_granger_causes(target)
        if not causes:
            continue
        base_feat, base_cols = feature_frame(df[["日期", target]].copy(), target)
        use_cols = ["日期", target] + [c for c in causes if c in df.columns]
        exog_feat, exog_cols = feature_frame(df[use_cols].copy(), target, causes)
        for model_name, feat, cols in [
            ("ridge_own_lags_calendar", base_feat, base_cols),
            ("ridge_with_granger_causes", exog_feat, exog_cols),
        ]:
            train = feat[feat["日期"] < TEST_START]
            test = feat[(feat["日期"] >= TEST_START) & (feat["日期"] <= TEST_END)]
            if len(train) < 100 or len(test) == 0:
                continue
            model = fit_ridge(train, target, cols)
            pred = model.predict(test[cols])
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "causes": ";".join(causes),
                    **metrics(test[target].to_numpy(float), pred),
                    "n": len(test),
                }
            )
    res = pd.DataFrame(rows).sort_values(["target", "rmse"])
    res.to_csv(OUT / "forecast_causal_feature_metrics.csv", index=False, encoding="utf-8-sig")

    if not res.empty:
        gains = []
        for target, g in res.groupby("target"):
            base = g[g["model"] == "ridge_own_lags_calendar"]
            exog = g[g["model"] == "ridge_with_granger_causes"]
            if not base.empty and not exog.empty:
                gains.append(
                    {
                        "target": target,
                        "rmse_gain_pct": (1 - float(exog["rmse"].iloc[0]) / float(base["rmse"].iloc[0])) * 100,
                    }
                )
        gain_df = pd.DataFrame(gains).sort_values("rmse_gain_pct", ascending=False)
        gain_df.to_csv(OUT / "forecast_causal_feature_gain.csv", index=False, encoding="utf-8-sig")
        if not gain_df.empty:
            plot = gain_df.copy()
            plot["plot_value"] = plot["rmse_gain_pct"].clip(lower=0)
            svg_bar_percent(
                FIG / "forecast_causal_feature_gain_positive.svg",
                plot["target"].tolist(),
                plot["plot_value"].tolist(),
                "Granger 候选变量加入后 RMSE 改善幅度（负值按 0 显示）",
            )
    return res


def granger_window_edges(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    sub = df[(df["日期"] >= start) & (df["日期"] <= end)].copy()
    variables = [v for v in KEY_VARIABLES if v in sub.columns]
    weekly = sub.set_index("日期")[variables].resample("W-SUN").sum()
    trans = np.log1p(weekly).diff().dropna()
    rows = []
    for effect in variables:
        for cause in variables:
            if cause == effect:
                continue
            best = None
            for lag in [1, 2, 3, 4]:
                rec = {"cause": cause, "effect": effect, "lag_weeks": lag, **granger_pair(trans, cause, effect, lag)}
                if best is None or rec["p_value"] < best["p_value"]:
                    best = rec
            rows.append(best)
    res = pd.DataFrame(rows)
    res["q_value_fdr"] = fdr_bh(res["p_value"])
    res["pass_fdr_05"] = res["q_value_fdr"] < 0.05
    res["window_start"] = start
    res["window_end"] = end
    return res


def granger_robustness() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = read_province()
    df = df[(df["日期"] >= "2023-01-01") & (df["日期"] < FULL_END)].copy()
    windows = [
        ("2023-01-01", "2024-12-31"),
        ("2023-07-01", "2025-06-30"),
        ("2024-01-01", "2025-12-31"),
        ("2023-01-01", "2025-12-31"),
    ]
    edges = pd.concat([granger_window_edges(df, s, e) for s, e in windows], ignore_index=True)
    edges.to_csv(OUT / "granger_rolling_window_edges.csv", index=False, encoding="utf-8-sig")
    stable = (
        edges.groupby(["cause", "effect"], as_index=False)
        .agg(
            windows_tested=("q_value_fdr", "size"),
            pass_count=("pass_fdr_05", "sum"),
            min_q=("q_value_fdr", "min"),
            median_q=("q_value_fdr", "median"),
            median_lag=("lag_weeks", "median"),
        )
        .sort_values(["pass_count", "min_q"], ascending=[False, True])
    )
    stable["stability_score"] = stable["pass_count"] / stable["windows_tested"]
    stable.to_csv(OUT / "granger_rolling_window_stability.csv", index=False, encoding="utf-8-sig")

    full = edges[(edges["window_start"] == "2023-01-01") & (edges["window_end"] == "2025-12-31")].copy()
    passed = full[full["pass_fdr_05"]].copy()
    passed["weight"] = -np.log10(passed["q_value_fdr"].clip(lower=1e-300))
    out_score = passed.groupby("cause", as_index=False).agg(out_degree=("effect", "nunique"), out_strength=("weight", "sum"))
    in_score = passed.groupby("effect", as_index=False).agg(in_degree=("cause", "nunique"), in_strength=("weight", "sum"))
    node = pd.merge(out_score.rename(columns={"cause": "node"}), in_score.rename(columns={"effect": "node"}), on="node", how="outer").fillna(0)
    node["net_out_strength"] = node["out_strength"] - node["in_strength"]
    node = node.sort_values(["out_strength", "out_degree"], ascending=False)
    node.to_csv(OUT / "granger_network_node_scores.csv", index=False, encoding="utf-8-sig")
    top_node = node.head(12).sort_values("out_strength", ascending=True)
    if not top_node.empty:
        svg_bar(
            FIG / "granger_network_node_out_strength.svg",
            top_node["node"].tolist(),
            top_node["out_strength"].tolist(),
            "Granger 网络节点外向预测强度",
        )
    return edges, stable, node


def empirical_interval_total() -> pd.DataFrame:
    df = read_province()
    target = "合计"
    causes = [c for c in ["大工业电量", "居民生活", "商业用电", "制造业", "工业", "非普工业"] if c in df.columns]
    feat, cols = feature_frame(df[["日期", target] + causes].copy(), target, causes)
    train = feat[feat["日期"] < TEST_START]
    test = feat[(feat["日期"] >= TEST_START) & (feat["日期"] <= TEST_END)].copy()
    model = fit_ridge(train, target, cols)
    train_pred = model.predict(train[cols])
    resid = train[target].to_numpy(float) - train_pred
    q05, q50, q95 = np.quantile(resid, [0.05, 0.5, 0.95])
    pred = model.predict(test[cols])
    out = test[["日期", target]].copy().rename(columns={target: "actual"})
    out["prediction"] = pred
    out["lower_90"] = pred + q05
    out["median_bias_adjusted"] = pred + q50
    out["upper_90"] = pred + q95
    out["covered_90"] = (out["actual"] >= out["lower_90"]) & (out["actual"] <= out["upper_90"])
    out.to_csv(OUT / "forecast_total_empirical_interval_90.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "target": target,
                "model": "ridge_with_selected_category_lags",
                "nominal_coverage": 0.9,
                "empirical_test_coverage": float(out["covered_90"].mean()),
                "q05_residual": float(q05),
                "q50_residual": float(q50),
                "q95_residual": float(q95),
                **metrics(out["actual"].to_numpy(float), out["prediction"].to_numpy(float)),
                "n": len(out),
            }
        ]
    ).to_csv(OUT / "forecast_total_empirical_interval_summary.csv", index=False, encoding="utf-8-sig")
    return out


def write_second_round_notes(
    rolling_metrics: pd.DataFrame,
    city_best: pd.DataFrame,
    causal_metrics: pd.DataFrame,
    stable: pd.DataFrame,
    node: pd.DataFrame,
) -> None:
    best_roll = rolling_metrics.sort_values("rmse").iloc[0]
    city_lines = ["| 地市 | 最优模型 | MAPE | RMSE |", "| --- | --- | ---: | ---: |"]
    for _, r in city_best.sort_values("mape").iterrows():
        city_lines.append(f"| {r['city']} | {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} |")

    causal_lines = ["| 目标 | 自身模型 RMSE | 加入 Granger 候选 RMSE | 改善 | 候选原因变量 |", "| --- | ---: | ---: | ---: | --- |"]
    if not causal_metrics.empty:
        for target, g in causal_metrics.groupby("target"):
            base = g[g["model"] == "ridge_own_lags_calendar"]
            exog = g[g["model"] == "ridge_with_granger_causes"]
            if not base.empty and not exog.empty:
                gain = 1 - float(exog["rmse"].iloc[0]) / float(base["rmse"].iloc[0])
                causal_lines.append(
                    f"| {target} | {float(base['rmse'].iloc[0]):.2f} | {float(exog['rmse'].iloc[0]):.2f} | {pct(gain)} | {exog['causes'].iloc[0]} |"
                )

    stable_lines = ["| 起点 | 终点 | 通过窗口数 | 稳定度 | 最小 q值 | 中位滞后 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for _, r in stable.head(15).iterrows():
        stable_lines.append(
            f"| {r['cause']} | {r['effect']} | {int(r['pass_count'])}/{int(r['windows_tested'])} | {pct(r['stability_score'])} | {r['min_q']:.4g} | {r['median_lag']:.1f} |"
        )

    node_lines = ["| 节点 | 外向边数 | 内向边数 | 外向强度 | 净外向强度 |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, r in node.head(12).iterrows():
        node_lines.append(
            f"| {r['node']} | {int(r['out_degree'])} | {int(r['in_degree'])} | {r['out_strength']:.2f} | {r['net_out_strength']:.2f} |"
        )

    md = f"""
# 预测方法精讲与第二轮滚动回测

## 1. 为什么第一轮之后还要做滚动起点回测

第一轮用 2025-10-01 到 2026-01-09 作为一次固定测试集。这个做法能快速判断模型有没有用，但还不够强，因为一次测试集可能刚好处在某个特殊季节。第二轮改用多个预测起点：2025-04-01、2025-06-01、2025-08-01、2025-10-01、2025-11-01、2025-12-01。每个起点只使用起点以前的数据训练，然后向后递推 30 天。

这样做的含义是：我们模拟“真实站在某一天做未来 30 天预测”的情境。对第 10 天预测时，不能偷偷使用第 1 到第 9 天的真实值，只能使用模型前面已经预测出来的值继续滚动。

## 2. 递推预测是什么

设目标序列为 $y_t$，比如全省日度合计用电量。单步模型学的是：

$$
y_t = f(y_{{t-1}}, y_{{t-2}}, y_{{t-7}}, \\text{{日历特征}}) + \\varepsilon_t
$$

如果只预测明天，右边的滞后项都是真实历史。可是要预测未来第 7 天时，$y_{{t+6}}$ 还没有真实值，所以要把前面预测出来的值放回历史序列。这叫递推预测。它更严格，也更接近真实使用场景。

## 3. 为什么仍然保留朴素模型

朴素模型不是“低级模型”，而是科研里必须保留的基线。原因是：如果复杂模型连“用昨天预测今天”或“用上周同日预测今天”都打不过，那么复杂模型没有解释价值。这里保留四类基线：

- `naive_yesterday`：明天等于今天。
- `seasonal_week`：今天等于上周同一天。
- `rolling7`：今天等于过去 7 天平均。
- `rolling28`：今天等于过去 28 天平均。

## 4. 滚动起点结果

全省合计用电量滚动 30 日回测中，整体 RMSE 最低的模型是 **{best_roll['model']}**，MAPE 为 **{pct(best_roll['mape'])}**，MASE 为 **{best_roll['mase']:.3f}**。

MASE 的意思是“模型平均误差相当于季节朴素模型误差的多少倍”。如果 MASE 小于 1，说明模型平均上优于一个合理的季节性朴素预测。

完整结果：

- `Experiments/outputs/forecast_rolling_origin_total_metrics.csv`
- `Experiments/outputs/forecast_rolling_origin_total_by_horizon.csv`
- `Experiments/figures/forecast_rolling_origin_horizon_mape.svg`

## 5. 地市预测结果

{chr(10).join(city_lines)}

地市结果的作用不是只看谁更好预测，而是识别“哪些地区用电更稳定、哪些地区波动更难预测”。如果某地市 MAPE 明显较高，后续要检查产业结构、节假日、天气或异常工业活动。

## 6. Granger 候选变量能否提升预测

{chr(10).join(causal_lines)}

这里要非常谨慎：如果加入某个行业的滞后项让预测变好，只能说它有“前置预测信息”。它不是结构因果效应。真正的因果解释还需要共同冲击控制、产业知识、可能的政策事件或外生冲击。

## 7. 经验预测区间

第二轮还生成了全省合计 90% 经验预测区间：

- `Experiments/outputs/forecast_total_empirical_interval_90.csv`
- `Experiments/outputs/forecast_total_empirical_interval_summary.csv`

这个区间来自训练期残差的 5% 和 95% 分位数。它比只给一个点预测更适合论文和答辩，因为它承认未来有不确定性。
"""
    write_md(NOTES / f"预测方法精讲与第二轮滚动回测-{DATE}.md", md)

    md2 = f"""
# 因果特征预测与传导稳健性

## 1. 本轮想解决什么问题

第一轮 Granger 初筛找到了很多候选边，但单次窗口检验不够稳。第二轮做两件事：

1. 把 Granger 候选原因变量作为预测特征，看它们是否真的改善目标变量预测。
2. 把样本切成多个时间窗口，检查候选边是否在不同窗口中反复出现。

## 2. 为什么要做窗口稳健性

如果一条边只在某个时间段显著，可能是偶然异常、共同冲击或春节错位造成的。更可信的边应当在多个时间窗口中都出现。这里的稳定度定义为：

$$
\\text{{稳定度}} = \\frac{{\\text{{通过 FDR 的窗口数}}}}{{\\text{{被检验的窗口数}}}}
$$

## 3. 稳定边前 15 条

{chr(10).join(stable_lines)}

完整结果：

- `Experiments/outputs/granger_rolling_window_edges.csv`
- `Experiments/outputs/granger_rolling_window_stability.csv`

## 4. 网络节点

把通过 FDR 的边看成一个有向网络。某个变量外向边多，说明它的历史变化对多个目标有前置预测信息；内向边多，说明它更容易被其他变量预测。

{chr(10).join(node_lines)}

完整节点表：`Experiments/outputs/granger_network_node_scores.csv`

## 5. 结论等级

本轮仍然是 Level 1 到 Level 2 之间的证据：比单次 Granger 更稳，但还没有加入天气、价格、政策和宏观经济控制。因此写作时应使用“时序传导候选”“预测领先关系”“可能的产业联动线索”，不要写成“已经证明上下游因果”。
"""
    write_md(NOTES / f"因果特征预测与传导稳健性-{DATE}.md", md2)

    report = f"""
# 福建用电量预测与产业因果推断：第二轮实证研究报告

## 1. 本轮新增内容

第二轮在第一轮基础上补强四个方面：

- 全省合计用电量做多起点 30 日递推回测。
- 所有地市做日度预测基线。
- 将 Granger 候选原因变量加入预测模型，检查其预测贡献。
- 对 Granger 边做滚动窗口稳健性和节点网络分析。

## 2. 预测主结论

全省合计滚动起点回测中，最优模型是 **{best_roll['model']}**，MAPE 为 **{pct(best_roll['mape'])}**，RMSE 为 **{best_roll['rmse']:.2f}**，MASE 为 **{best_roll['mase']:.3f}**。

这说明福建全省总用电短期预测已经可以形成可复现的第一版实证结果，但未来要继续加入天气和节假日，才能达到更像正式电力负荷预测论文的完整度。

## 3. 地市预测

{chr(10).join(city_lines)}

## 4. 因果预测与传导网络

{chr(10).join(causal_lines)}

滚动窗口稳定边前 15 条：

{chr(10).join(stable_lines)}

## 5. 当前可写成论文/答辩的表述

可以说：

> 基于福建省 2023-2026 年日度用电数据，本文建立了时间顺序回测的短期用电预测基线，并进一步用周度对数差分 Granger 网络筛选行业间时序传导候选关系。结果显示，短期预测在全省合计和主要行业上达到较低 MAPE；若干行业变量在多个窗口中对其他行业具有稳定的前置预测信息。

不能说：

> 某行业已经被证明导致某行业变化。

原因是当前缺少天气、价格、政策、订单、产值等控制变量，Granger 结果仍可能来自共同冲击或共同季节性。

## 6. 文件入口

- `Notes/预测方法精讲与第二轮滚动回测-{DATE}.md`
- `Notes/因果特征预测与传导稳健性-{DATE}.md`
- `Experiments/scripts/fujian_second_iteration.py`
"""
    write_md(WRITING / f"福建实证第二轮研究报告-{DATE}.md", report)


def run_all() -> None:
    rolling_preds, rolling_metrics, _ = recursive_rolling_origin_total()
    _, city_best = city_forecast()
    causal_metrics = causal_feature_forecast()
    _, stable, node = granger_robustness()
    empirical_interval_total()
    write_second_round_notes(rolling_metrics, city_best, causal_metrics, stable, node)


if __name__ == "__main__":
    run_all()
