from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


PROJECT = Path(__file__).resolve().parents[2]
STAGE1_WIDE = PROJECT / "Data" / "processed" / "structure_stage1" / "province_usage_daily_wide.csv"
GROUP_MAP = PROJECT / "Data" / "dictionary" / "structure_stage1" / "manufacturing_industry_group_map.csv"
OUT = PROJECT / "Experiments" / "outputs" / "structure_stage4_robust"
FIG = PROJECT / "Experiments" / "figures" / "structure_stage4_robust"

MANUFACTURING_TOTAL = "（二） 制造业"
HEAVY_GROUP = "传统高耗能与材料行业"
ADVANCED_GROUPS = ["装备制造", "先进制造与技术相关行业"]
GROUP_ORDER = [
    "传统高耗能与材料行业",
    "消费与轻工制造",
    "装备制造",
    "先进制造与技术相关行业",
    "其他制造",
]
MAX_GRANGER_LAG = 3
ROLLING_WINDOW = 18
TARGET_SLUG = {
    "制造业同比增长率": "manufacturing_yoy_growth",
    "结构转型差值T同比变化": "transition_T_yoy_change",
    "结构转型比值R同比变化": "transition_R_yoy_change",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def pct_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def float_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.4f}"


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(STAGE1_WIDE, encoding="utf-8-sig", parse_dates=["date"]).set_index("date").sort_index()
    group_map = pd.read_csv(GROUP_MAP, encoding="utf-8-sig")
    group_map = group_map[group_map["present_in_data"] == True].copy()
    missing = [name for name in group_map["industry_name"] if name not in wide.columns]
    if missing:
        raise RuntimeError(f"宽表缺少制造业行业：{missing}")
    return wide, group_map


def build_monthly_features(wide: pd.DataFrame, group_map: pd.DataFrame) -> dict[str, pd.DataFrame]:
    industry_names = group_map["industry_name"].tolist()
    monthly_industry = wide[industry_names].resample("MS").sum()
    monthly_industry = monthly_industry.loc[monthly_industry.index <= pd.Timestamp("2025-12-01")].copy()
    monthly_industry_yoy = monthly_industry.pct_change(12).replace([np.inf, -np.inf], np.nan)

    monthly_group = pd.DataFrame(index=monthly_industry.index)
    for group in GROUP_ORDER:
        names = group_map.loc[group_map["manufacturing_group"] == group, "industry_name"].tolist()
        monthly_group[group] = monthly_industry[names].sum(axis=1)
    monthly_group.insert(0, "制造业", wide[[MANUFACTURING_TOTAL]].resample("MS").sum().loc[monthly_industry.index, MANUFACTURING_TOTAL])
    monthly_group["装备与先进制造合计"] = monthly_group[ADVANCED_GROUPS].sum(axis=1)
    monthly_share = monthly_group[GROUP_ORDER].div(monthly_group["制造业"], axis=0)
    monthly_share["装备与先进制造合计"] = monthly_group["装备与先进制造合计"] / monthly_group["制造业"]
    monthly_share["结构转型差值T"] = monthly_share["装备与先进制造合计"] - monthly_share[HEAVY_GROUP]
    monthly_share["结构转型比值R"] = monthly_group["装备与先进制造合计"] / monthly_group[HEAVY_GROUP].replace(0, np.nan)

    targets = pd.DataFrame(index=monthly_industry.index)
    targets["制造业同比增长率"] = monthly_group["制造业"].pct_change(12)
    targets["结构转型差值T同比变化"] = monthly_share["结构转型差值T"] - monthly_share["结构转型差值T"].shift(12)
    targets["结构转型比值R同比变化"] = monthly_share["结构转型比值R"] - monthly_share["结构转型比值R"].shift(12)
    targets = targets.replace([np.inf, -np.inf], np.nan)
    return {
        "monthly_industry_yoy": monthly_industry_yoy,
        "targets": targets,
    }


def add_lags(df: pd.DataFrame, col: str, p: int, prefix: str) -> pd.DataFrame:
    out = df.copy()
    for lag in range(1, p + 1):
        out[f"{prefix}_lag{lag}"] = out[col].shift(lag)
    return out


def rss_ols(X: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    return float(residual @ residual)


def granger_style_test(y: pd.Series, x: pd.Series, p: int) -> dict[str, float | int]:
    df = pd.DataFrame({"y": y, "x": x})
    df = add_lags(df, "y", p, "y")
    df = add_lags(df, "x", p, "x")
    lag_y_cols = [f"y_lag{lag}" for lag in range(1, p + 1)]
    lag_x_cols = [f"x_lag{lag}" for lag in range(1, p + 1)]
    df = df[["y", *lag_y_cols, *lag_x_cols]].dropna()
    n = len(df)
    k_unrestricted = 1 + 2 * p
    df_num = p
    df_den = n - k_unrestricted
    if df_den <= 0 or n < 12:
        return {"n": n, "F": np.nan, "p_value": np.nan, "df_num": df_num, "df_den": df_den}
    rss_r = rss_ols(df[lag_y_cols].values, df["y"].values)
    rss_u = rss_ols(df[[*lag_y_cols, *lag_x_cols]].values, df["y"].values)
    if rss_u <= 0 or rss_r < rss_u:
        return {"n": n, "F": np.nan, "p_value": np.nan, "df_num": df_num, "df_den": df_den}
    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    p_value = float(f_dist.sf(f_stat, df_num, df_den))
    return {"n": n, "F": float(f_stat), "p_value": p_value, "df_num": df_num, "df_den": df_den}


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float)
    valid = p.dropna().sort_values()
    q = pd.Series(np.nan, index=p.index, dtype=float)
    m = len(valid)
    if m == 0:
        return q
    ranked = valid.reset_index()
    ranked["rank"] = np.arange(1, m + 1)
    ranked["raw_q"] = ranked[p.name] * m / ranked["rank"]
    # Monotone adjustment from largest to smallest p.
    adjusted = ranked["raw_q"].iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    q.loc[ranked["index"]] = adjusted.values
    return q


def best_lag_corr(y: pd.Series, x: pd.Series, max_lag: int = 6) -> dict[str, float | int]:
    best = {"abs_corr": -np.inf, "corr": np.nan, "lag": np.nan, "n": 0}
    for lag in range(1, max_lag + 1):
        pair = pd.concat([x.shift(lag), y], axis=1).dropna()
        if len(pair) < 12 or pair.iloc[:, 0].std(ddof=0) == 0 or pair.iloc[:, 1].std(ddof=0) == 0:
            continue
        corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if abs(corr) > best["abs_corr"]:
            best = {"abs_corr": abs(corr), "corr": corr, "lag": lag, "n": len(pair)}
    return best


def rolling_corr_stability(y: pd.Series, x: pd.Series, lag: int) -> dict[str, float | int]:
    pair = pd.concat([x.shift(lag), y], axis=1).dropna()
    corrs = []
    for start in range(0, len(pair) - ROLLING_WINDOW + 1):
        window = pair.iloc[start : start + ROLLING_WINDOW]
        if window.iloc[:, 0].std(ddof=0) == 0 or window.iloc[:, 1].std(ddof=0) == 0:
            continue
        corrs.append(float(window.iloc[:, 0].corr(window.iloc[:, 1])))
    if not corrs:
        return {"rolling_windows": 0, "rolling_median_abs_corr": np.nan, "rolling_strong_share": np.nan, "rolling_sign_consistency": np.nan}
    arr = np.array(corrs)
    pos = np.mean(arr > 0)
    neg = np.mean(arr < 0)
    return {
        "rolling_windows": len(arr),
        "rolling_median_abs_corr": float(np.median(np.abs(arr))),
        "rolling_strong_share": float(np.mean(np.abs(arr) >= 0.3)),
        "rolling_sign_consistency": float(max(pos, neg)),
    }


def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> float:
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.r_[1.0, X_test]
    beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    return float(X_test @ beta)


def expanding_predictive_gain(y: pd.Series, x: pd.Series, p: int) -> dict[str, float | int]:
    df = pd.DataFrame({"y": y, "x": x})
    df = add_lags(df, "y", p, "y")
    df = add_lags(df, "x", p, "x")
    lag_y_cols = [f"y_lag{lag}" for lag in range(1, p + 1)]
    lag_x_cols = [f"x_lag{lag}" for lag in range(1, p + 1)]
    df = df[["y", *lag_y_cols, *lag_x_cols]].dropna()
    min_train = max(12, 3 * p + 4)
    if len(df) <= min_train + 4:
        return {"backtest_n": 0, "mae_base": np.nan, "mae_full": np.nan, "mae_gain": np.nan, "point_improvement_share": np.nan}
    truth, pred_base, pred_full = [], [], []
    for test_pos in range(min_train, len(df)):
        train = df.iloc[:test_pos]
        test = df.iloc[test_pos]
        pred_base.append(fit_predict(train[lag_y_cols].values, train["y"].values, test[lag_y_cols].values))
        pred_full.append(fit_predict(train[[*lag_y_cols, *lag_x_cols]].values, train["y"].values, test[[*lag_y_cols, *lag_x_cols]].values))
        truth.append(float(test["y"]))
    truth_arr = np.array(truth)
    base_arr = np.array(pred_base)
    full_arr = np.array(pred_full)
    err_base = np.abs(truth_arr - base_arr)
    err_full = np.abs(truth_arr - full_arr)
    mae_base = float(np.mean(err_base))
    mae_full = float(np.mean(err_full))
    gain = (mae_base - mae_full) / mae_base if mae_base != 0 else np.nan
    return {
        "backtest_n": len(truth),
        "mae_base": mae_base,
        "mae_full": mae_full,
        "mae_gain": float(gain),
        "point_improvement_share": float(np.mean(err_full < err_base)),
    }


def evidence_label(row: pd.Series) -> str:
    if (
        row["q_value"] <= 0.1
        and row["mae_gain"] >= 0.05
        and row["point_improvement_share"] >= 0.55
        and row["rolling_strong_share"] >= 0.5
    ):
        return "较稳健前置信号候选"
    if row["mae_gain"] >= 0.05 and row["point_improvement_share"] >= 0.5:
        return "有预测增益候选"
    if row["best_abs_corr"] >= 0.45 or row["p_value"] <= 0.1:
        return "仅相关或检验候选"
    return "证据较弱"


def build_robust_table(features: dict[str, pd.DataFrame], group_map: pd.DataFrame) -> pd.DataFrame:
    x_data = features["monthly_industry_yoy"]
    targets = features["targets"]
    group_lookup = group_map.set_index("industry_name")["manufacturing_group"].to_dict()
    rows = []
    for target_name in targets.columns:
        y = targets[target_name]
        for industry in x_data.columns:
            x = x_data[industry]
            corr = best_lag_corr(y, x, max_lag=6)
            for p in range(1, MAX_GRANGER_LAG + 1):
                test = granger_style_test(y, x, p)
                pred = expanding_predictive_gain(y, x, p)
                roll = rolling_corr_stability(y, x, int(corr["lag"])) if pd.notna(corr["lag"]) else {}
                rows.append(
                    {
                        "目标变量": target_name,
                        "行业": industry,
                        "制造业分组": group_lookup[industry],
                        "检验滞后阶数": p,
                        "best_corr_lag": corr["lag"],
                        "best_corr": corr["corr"],
                        "best_abs_corr": corr["abs_corr"],
                        "corr_n": corr["n"],
                        "F": test["F"],
                        "p_value": test["p_value"],
                        "test_n": test["n"],
                        "df_num": test["df_num"],
                        "df_den": test["df_den"],
                        **pred,
                        **roll,
                    }
                )
    table = pd.DataFrame(rows)
    table["q_value"] = np.nan
    for target_name, idx in table.groupby("目标变量").groups.items():
        sub = table.loc[idx, "p_value"].rename("p_value")
        table.loc[idx, "q_value"] = benjamini_hochberg(sub)
    table["证据等级"] = table.apply(evidence_label, axis=1)
    table["综合排序分"] = (
        table["mae_gain"].fillna(-1)
        + 0.15 * table["point_improvement_share"].fillna(0)
        + 0.10 * table["rolling_strong_share"].fillna(0)
        + 0.05 * (1 - table["q_value"].fillna(1))
    )
    return table.sort_values(["目标变量", "证据等级", "综合排序分"], ascending=[True, True, False])


def select_best_by_industry(table: pd.DataFrame) -> pd.DataFrame:
    # 每个目标-行业只保留综合排序分最高的一个滞后阶数，避免同一行业重复刷屏。
    idx = table.groupby(["目标变量", "行业"])["综合排序分"].idxmax()
    return table.loc[idx].sort_values(["目标变量", "证据等级", "综合排序分"], ascending=[True, True, False])


def save_outputs(table: pd.DataFrame, best: pd.DataFrame) -> None:
    table.to_csv(OUT / "robust_all_tests.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT / "robust_best_by_industry.csv", index=False, encoding="utf-8-sig")
    for target_name, sub in best.groupby("目标变量"):
        slug = TARGET_SLUG[target_name]
        top = sub.sort_values("综合排序分", ascending=False).head(12).copy()
        display_cols = [
            "目标变量",
            "行业",
            "制造业分组",
            "检验滞后阶数",
            "best_corr",
            "p_value",
            "q_value",
            "mae_gain",
            "point_improvement_share",
            "rolling_strong_share",
            "证据等级",
        ]
        display = top[display_cols].copy()
        for col in ["best_corr", "p_value", "q_value"]:
            display[col] = display[col].map(float_fmt)
        for col in ["mae_gain", "point_improvement_share", "rolling_strong_share"]:
            display[col] = display[col].map(pct_fmt)
        write_markdown_table(display, OUT / f"robust_top_{slug}.md")


def plot_evidence_counts(best: pd.DataFrame) -> None:
    order = ["较稳健前置信号候选", "有预测增益候选", "仅相关或检验候选", "证据较弱"]
    counts = best.groupby(["目标变量", "证据等级"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    ax = counts.plot(kind="bar", stacked=True, figsize=(11, 5.8), colormap="Set2")
    ax.set_title("前置信号候选证据等级数量")
    ax.set_xlabel("目标变量")
    ax.set_ylabel("行业数量")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIG / "stage4_robust_evidence_counts.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_gain(best: pd.DataFrame, target_name: str) -> None:
    sub = best[best["目标变量"] == target_name].sort_values("综合排序分", ascending=False).head(10)
    if sub.empty:
        return
    sub = sub.sort_values("mae_gain")
    colors = sub["证据等级"].map({
        "较稳健前置信号候选": "#0072B2",
        "有预测增益候选": "#009E73",
        "仅相关或检验候选": "#E69F00",
        "证据较弱": "#999999",
    })
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.barh(sub["行业"], sub["mae_gain"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title(f"{target_name}：加强版候选行业 MAE 改善率")
    ax.set_xlabel("MAE 改善率")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / f"stage4_robust_top_gain_{TARGET_SLUG[target_name]}.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(best: pd.DataFrame) -> None:
    summary = {
        "method_note": "加强版使用嵌套滞后回归 F 检验、BH-FDR、多滞后预测增益和滚动相关稳定性；仍然不是结构因果识别。",
        "max_granger_lag": MAX_GRANGER_LAG,
        "rolling_window_months": ROLLING_WINDOW,
        "evidence_counts": {},
        "top_candidates": {},
    }
    for target_name, sub in best.groupby("目标变量"):
        counts = sub["证据等级"].value_counts().to_dict()
        summary["evidence_counts"][target_name] = {str(k): int(v) for k, v in counts.items()}
        top = sub.sort_values("综合排序分", ascending=False).head(6)
        summary["top_candidates"][target_name] = [
            {
                "行业": str(row["行业"]),
                "制造业分组": str(row["制造业分组"]),
                "证据等级": str(row["证据等级"]),
                "检验滞后阶数": int(row["检验滞后阶数"]),
                "best_corr": float(row["best_corr"]) if pd.notna(row["best_corr"]) else None,
                "p_value": float(row["p_value"]) if pd.notna(row["p_value"]) else None,
                "q_value": float(row["q_value"]) if pd.notna(row["q_value"]) else None,
                "mae_gain": float(row["mae_gain"]) if pd.notna(row["mae_gain"]) else None,
                "point_improvement_share": float(row["point_improvement_share"]) if pd.notna(row["point_improvement_share"]) else None,
                "rolling_strong_share": float(row["rolling_strong_share"]) if pd.notna(row["rolling_strong_share"]) else None,
            }
            for _, row in top.iterrows()
        ]
    (OUT / "stage4_robust_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ensure_dirs()
    setup_plot_style()
    wide, group_map = read_inputs()
    features = build_monthly_features(wide, group_map)
    all_tests = build_robust_table(features, group_map)
    best = select_best_by_industry(all_tests)
    save_outputs(all_tests, best)
    plot_evidence_counts(best)
    for target_name in features["targets"].columns:
        plot_top_gain(best, target_name)
    write_summary(best)


if __name__ == "__main__":
    main()
