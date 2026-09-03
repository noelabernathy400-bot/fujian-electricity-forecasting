from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
STAGE1_WIDE = PROJECT / "Data" / "processed" / "structure_stage1" / "province_usage_daily_wide.csv"
GROUP_MAP = PROJECT / "Data" / "dictionary" / "structure_stage1" / "manufacturing_industry_group_map.csv"
OUT = PROJECT / "Experiments" / "outputs" / "structure_stage4"
FIG = PROJECT / "Experiments" / "figures" / "structure_stage4"

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
LAGS = list(range(1, 7))
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
    daily_industry = wide[industry_names].copy()
    monthly_industry = daily_industry.resample("MS").sum()
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
        "monthly_industry": monthly_industry,
        "monthly_industry_yoy": monthly_industry_yoy,
        "monthly_group": monthly_group,
        "monthly_share": monthly_share,
        "targets": targets,
    }


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < 12:
        return np.nan, n
    if pair.iloc[:, 0].std(ddof=0) == 0 or pair.iloc[:, 1].std(ddof=0) == 0:
        return np.nan, n
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), n


def build_lag_correlation_table(features: dict[str, pd.DataFrame], group_map: pd.DataFrame) -> pd.DataFrame:
    x_data = features["monthly_industry_yoy"]
    targets = features["targets"]
    group_lookup = group_map.set_index("industry_name")["manufacturing_group"].to_dict()
    records = []
    for target_name in targets.columns:
        y = targets[target_name]
        for industry in x_data.columns:
            best = {"abs_corr": -np.inf}
            for lag in LAGS:
                corr, n = safe_corr(x_data[industry].shift(lag), y)
                if pd.notna(corr) and abs(corr) > best["abs_corr"]:
                    best = {
                        "abs_corr": abs(corr),
                        "相关系数": corr,
                        "滞后月数": lag,
                        "有效样本数": n,
                    }
            if best["abs_corr"] != -np.inf:
                records.append(
                    {
                        "目标变量": target_name,
                        "行业": industry,
                        "制造业分组": group_lookup[industry],
                        "最佳滞后月数": best["滞后月数"],
                        "相关系数": best["相关系数"],
                        "相关系数绝对值": best["abs_corr"],
                        "有效样本数": best["有效样本数"],
                    }
                )
    return pd.DataFrame(records).sort_values(["目标变量", "相关系数绝对值"], ascending=[True, False])


def ols_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> float:
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.r_[1.0, X_test]
    beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    return float(X_test @ beta)


def expanding_window_predictive_gain(features: dict[str, pd.DataFrame], lag_table: pd.DataFrame, target_name: str) -> pd.DataFrame:
    # 用最强滞后相关的前 12 个行业做非常初步的滚动预测增益检查。
    candidates = lag_table[lag_table["目标变量"] == target_name].head(12).copy()
    x_data = features["monthly_industry_yoy"]
    y = features["targets"][target_name]
    rows = []
    for _, cand in candidates.iterrows():
        industry = cand["行业"]
        lag = int(cand["最佳滞后月数"])
        df = pd.DataFrame(
            {
                "y": y,
                "y_lag1": y.shift(1),
                "x_lag": x_data[industry].shift(lag),
            }
        ).dropna()
        if len(df) < 18:
            continue
        preds_base = []
        preds_x = []
        truth = []
        # 训练窗口从至少 12 个样本开始，随后逐月扩展。
        for test_pos in range(12, len(df)):
            train = df.iloc[:test_pos]
            test = df.iloc[test_pos]
            preds_base.append(ols_fit_predict(train[["y_lag1"]].values, train["y"].values, test[["y_lag1"]].values))
            preds_x.append(ols_fit_predict(train[["y_lag1", "x_lag"]].values, train["y"].values, test[["y_lag1", "x_lag"]].values))
            truth.append(float(test["y"]))
        truth_arr = np.array(truth)
        base_arr = np.array(preds_base)
        x_arr = np.array(preds_x)
        mae_base = float(np.mean(np.abs(truth_arr - base_arr)))
        mae_x = float(np.mean(np.abs(truth_arr - x_arr)))
        gain = (mae_base - mae_x) / mae_base if mae_base != 0 else np.nan
        rows.append(
            {
                "目标变量": target_name,
                "行业": industry,
                "制造业分组": cand["制造业分组"],
                "滞后月数": lag,
                "回测样本数": len(truth),
                "基线MAE": mae_base,
                "加入行业后MAE": mae_x,
                "MAE改善率": gain,
                "滞后相关系数": cand["相关系数"],
            }
        )
    return pd.DataFrame(rows).sort_values("MAE改善率", ascending=False)


def save_tables(lag_table: pd.DataFrame, gain_tables: dict[str, pd.DataFrame]) -> None:
    lag_table.to_csv(OUT / "lag_correlation_all_targets.csv", index=False, encoding="utf-8-sig")
    for target, table in gain_tables.items():
        safe_name = target.replace("/", "_")
        table.to_csv(OUT / f"predictive_gain_{safe_name}.csv", index=False, encoding="utf-8-sig")

    for target in lag_table["目标变量"].unique():
        display = lag_table[lag_table["目标变量"] == target].head(10).copy()
        display["相关系数"] = display["相关系数"].map(lambda x: f"{x:.3f}")
        display["相关系数绝对值"] = display["相关系数绝对值"].map(lambda x: f"{x:.3f}")
        write_markdown_table(display, OUT / f"top_lag_correlation_{TARGET_SLUG.get(target, target)}.md")

    for target, table in gain_tables.items():
        if table.empty:
            continue
        display = table.head(10).copy()
        for col in ["基线MAE", "加入行业后MAE"]:
            display[col] = display[col].map(lambda x: f"{x:.4f}")
        display["MAE改善率"] = display["MAE改善率"].map(pct_fmt)
        display["滞后相关系数"] = display["滞后相关系数"].map(lambda x: f"{x:.3f}")
        write_markdown_table(display, OUT / f"top_predictive_gain_{TARGET_SLUG.get(target, target)}.md")


def plot_lag_correlation_heatmap(lag_table: pd.DataFrame, target_name: str) -> None:
    top = lag_table[lag_table["目标变量"] == target_name].head(12).copy()
    if top.empty:
        return
    industries = top["行业"].tolist()
    matrix = []
    for industry in industries:
        vals = []
        subset = lag_table[(lag_table["目标变量"] == target_name) & (lag_table["行业"] == industry)]
        # 重新从最佳表无法取所有滞后；此处画最佳相关条形图更诚实。
        vals.append(float(subset.iloc[0]["相关系数"]))
        matrix.append(vals)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = ["#D55E00" if v < 0 else "#0072B2" for v in top["相关系数"]]
    ax.barh(top["行业"][::-1], top["相关系数"][::-1], color=colors[::-1])
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title(f"{target_name}：最佳滞后相关前十二行业")
    ax.set_xlabel("相关系数")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / f"stage4_lag_corr_{TARGET_SLUG.get(target_name, target_name)}.png", bbox_inches="tight")
    plt.close(fig)


def plot_predictive_gain(table: pd.DataFrame, target_name: str) -> None:
    if table.empty:
        return
    top = table.head(10).sort_values("MAE改善率")
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    colors = ["#0072B2" if v >= 0 else "#D55E00" for v in top["MAE改善率"]]
    ax.barh(top["行业"], top["MAE改善率"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title(f"{target_name}：加入单个行业滞后变量的初步预测增益")
    ax.set_xlabel("MAE 改善率")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / f"stage4_predictive_gain_{TARGET_SLUG.get(target_name, target_name)}.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(lag_table: pd.DataFrame, gain_tables: dict[str, pd.DataFrame]) -> None:
    summary = {
        "method_note": "月度同比增长率滞后相关 + 扩展窗口单变量滞后回归初筛；不是 Granger 检验，不是因果识别。",
        "lags_months": LAGS,
        "top_lag_correlation": {},
        "top_predictive_gain": {},
    }
    for target in lag_table["目标变量"].unique():
        top_corr = lag_table[lag_table["目标变量"] == target].head(5)
        summary["top_lag_correlation"][target] = [
            {
                "行业": str(row["行业"]),
                "制造业分组": str(row["制造业分组"]),
                "最佳滞后月数": int(row["最佳滞后月数"]),
                "相关系数": float(row["相关系数"]),
                "有效样本数": int(row["有效样本数"]),
            }
            for _, row in top_corr.iterrows()
        ]
        table = gain_tables.get(target, pd.DataFrame())
        if not table.empty:
            summary["top_predictive_gain"][target] = [
                {
                    "行业": str(row["行业"]),
                    "制造业分组": str(row["制造业分组"]),
                    "滞后月数": int(row["滞后月数"]),
                    "回测样本数": int(row["回测样本数"]),
                    "MAE改善率": float(row["MAE改善率"]),
                }
                for _, row in table.head(5).iterrows()
            ]
    (OUT / "stage4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ensure_dirs()
    setup_plot_style()
    wide, group_map = read_inputs()
    features = build_monthly_features(wide, group_map)
    lag_table = build_lag_correlation_table(features, group_map)
    gain_tables = {
        target: expanding_window_predictive_gain(features, lag_table, target)
        for target in features["targets"].columns
    }
    save_tables(lag_table, gain_tables)
    for target in features["targets"].columns:
        plot_lag_correlation_heatmap(lag_table, target)
        plot_predictive_gain(gain_tables[target], target)
    write_summary(lag_table, gain_tables)


if __name__ == "__main__":
    main()
