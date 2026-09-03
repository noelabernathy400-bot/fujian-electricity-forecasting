from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "Sources" / "原始数据"
PROCESSED = PROJECT / "Data" / "processed"
DICT = PROJECT / "Data" / "dictionary"
OUT = PROJECT / "Experiments" / "outputs"
FIG = PROJECT / "Experiments" / "figures"
NOTES = PROJECT / "Notes"
WRITING = PROJECT / "Writing" / "阶段报告"

CSV_PATH = RAW / "分行业&分用电类别(1).csv"
XLSX_PATH = RAW / "福建全省用电量统计20230101至20260109.xlsx"

META_COLS = ["类型", "汇总方式", "单位代码", "单位名称", "行号", "行业代码", "行业名称"]
CITY_COLS = ["福州", "厦门", "莆田", "泉州", "漳州", "龙岩", "三明", "南平", "宁德"]
BASE_USAGE_COLS = ["大工业电量", "非普工业", "居民生活", "非居照明", "商业用电", "农业", "抽水蓄能"]
KEY_VARIABLES = [
    "合计",
    "大工业电量",
    "非普工业",
    "居民生活",
    "商业用电",
    "农业",
    "制造业",
    "工业",
    "交通仓储和邮政业",
    "信息传输",
    "批发和零售业",
    "住宿和餐饮业",
    "黑色金属冶炼",
    "非金属矿物制品业",
    "化学制品制造业",
    "纺织业",
    "电子",
    "石化",
]


def ensure_dirs() -> None:
    for d in [PROCESSED, DICT, OUT, FIG, NOTES, WRITING]:
        d.mkdir(parents=True, exist_ok=True)


def parse_cn_date_col(col: str) -> pd.Timestamp | None:
    m = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", str(col))
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return pd.Timestamp(year=y, month=mo, day=d)


def date_cols_from_csv_header() -> tuple[list[str], dict[str, pd.Timestamp]]:
    header = pd.read_csv(CSV_PATH, nrows=0, encoding="utf-8").columns.tolist()
    mapping = {}
    for col in header:
        dt = parse_cn_date_col(col)
        if dt is not None:
            mapping[col] = dt
    return list(mapping.keys()), mapping


def clean_xlsx_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df["日期"] = pd.to_datetime(df["日期"].astype(str), format="%Y%m%d", errors="coerce")
    for col in df.columns:
        if col not in ["序号", "日期"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def audit() -> None:
    ensure_dirs()
    inventory = [
        {
            "file": CSV_PATH.name,
            "path": str(CSV_PATH),
            "size_mb": round(CSV_PATH.stat().st_size / 1024 / 1024, 2),
            "role": "地市/行业/用电类别日度宽表",
        },
        {
            "file": XLSX_PATH.name,
            "path": str(XLSX_PATH),
            "size_mb": round(XLSX_PATH.stat().st_size / 1024 / 1024, 2),
            "role": "全省和地市日度用电统计工作簿",
        },
    ]
    pd.DataFrame(inventory).to_csv(DICT / "raw_data_inventory.csv", index=False, encoding="utf-8-sig")

    date_cols, date_map = date_cols_from_csv_header()
    meta = pd.read_csv(CSV_PATH, usecols=META_COLS, dtype=str, encoding="utf-8")
    csv_summary = pd.DataFrame(
        [
            {"item": "rows", "value": len(meta)},
            {"item": "columns", "value": len(pd.read_csv(CSV_PATH, nrows=0, encoding="utf-8").columns)},
            {"item": "date_columns", "value": len(date_cols)},
            {"item": "start_date", "value": min(date_map.values()).strftime("%Y-%m-%d")},
            {"item": "end_date", "value": max(date_map.values()).strftime("%Y-%m-%d")},
            {"item": "unique_types", "value": meta["类型"].nunique(dropna=True)},
            {"item": "unique_summary_modes", "value": meta["汇总方式"].nunique(dropna=True)},
            {"item": "unique_units", "value": meta["单位名称"].nunique(dropna=True)},
            {"item": "unique_industry_codes", "value": meta["行业代码"].nunique(dropna=True)},
            {"item": "unique_industry_names", "value": meta["行业名称"].nunique(dropna=True)},
        ]
    )
    csv_summary.to_csv(DICT / "csv_basic_summary.csv", index=False, encoding="utf-8-sig")

    for col in META_COLS:
        vc = meta[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, "count"]
        vc.to_csv(DICT / f"csv_meta_{col}_counts.csv", index=False, encoding="utf-8-sig")

    xl = pd.ExcelFile(XLSX_PATH)
    sheet_rows = []
    col_rows = []
    for sheet in xl.sheet_names:
        df = clean_xlsx_sheet(pd.read_excel(XLSX_PATH, sheet_name=sheet))
        date_min = df["日期"].min()
        date_max = df["日期"].max()
        sheet_rows.append(
            {
                "sheet": sheet,
                "rows": len(df),
                "columns": len(df.columns),
                "start_date": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else "",
                "end_date": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else "",
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_dates": int(df["日期"].duplicated().sum()),
            }
        )
        for col in df.columns:
            if col in ["序号", "日期"]:
                group = "索引"
            elif sheet == "全省" and col in CITY_COLS:
                group = "地市总量"
            elif col in BASE_USAGE_COLS:
                group = "用电类别"
            elif col in CITY_COLS:
                group = "地市或区县"
            else:
                group = "行业或细分区域"
            col_rows.append({"sheet": sheet, "column": col, "group": group, "non_null": int(df[col].notna().sum())})
    pd.DataFrame(sheet_rows).to_csv(DICT / "xlsx_sheet_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(col_rows).to_csv(DICT / "xlsx_column_dictionary.csv", index=False, encoding="utf-8-sig")

    md = f"""
# 数据审计报告：福建用电量数据

## 1. 原始数据

| 文件 | 大小 | 作用 |
| --- | ---: | --- |
| `{CSV_PATH.name}` | {inventory[0]["size_mb"]} MB | 地市/行业/用电类别日度宽表 |
| `{XLSX_PATH.name}` | {inventory[1]["size_mb"]} MB | 全省和地市日度用电统计工作簿 |

## 2. CSV 宽表初步判断

- 行数：{len(meta)}
- 日期列数：{len(date_cols)}
- 日期范围：{min(date_map.values()).strftime("%Y-%m-%d")} 至 {max(date_map.values()).strftime("%Y-%m-%d")}
- 元信息列：`{", ".join(META_COLS)}`
- 唯一地市/单位数：{meta["单位名称"].nunique(dropna=True)}
- 唯一行业名称数：{meta["行业名称"].nunique(dropna=True)}

这个 CSV 是典型宽表：每行代表一个地区/行业/类别组合，后面每一天是一列。建模前需要转成长表或聚合成周度、月度面板。

## 3. Excel 工作簿初步判断

工作表：{", ".join(xl.sheet_names)}

Excel 更适合作为第一轮预测和 EDA 的主数据源，因为它已经按“日期为行、变量为列”组织。

## 4. 当前建模策略

1. 第一轮预测使用 Excel 的 `全省` 表。
2. 地市预测使用 Excel 的各地市工作表和 `全省` 表中的地市列。
3. 行业结构使用 Excel 的行业列和 CSV 的地市/行业口径。
4. 时序传导先用周度聚合，避免日度强星期效应直接制造假边。

## 5. 风险

- 日期范围到 2026-01 上旬，2026 年不能作为完整年度解释。
- CSV 中行业分类需要进一步确认，不能直接等同国民经济行业大类。
- 所有因果传导结论第一轮只按 Level 1：预测贡献或时序领先证据处理。
"""
    write_md(NOTES / "数据审计报告-2026-05-06.md", md)


def prepare() -> None:
    ensure_dirs()
    xl = pd.ExcelFile(XLSX_PATH)
    all_long = []
    all_sheet_summaries = []
    for sheet in xl.sheet_names:
        df = clean_xlsx_sheet(pd.read_excel(XLSX_PATH, sheet_name=sheet))
        df = df.sort_values("日期")
        df.to_csv(PROCESSED / f"xlsx_{sheet}_daily_wide.csv", index=False, encoding="utf-8-sig")
        value_cols = [c for c in df.columns if c not in ["序号", "日期"]]
        long = df.melt(id_vars=["日期"], value_vars=value_cols, var_name="variable", value_name="value")
        long.insert(0, "area", sheet)
        all_long.append(long)
        all_sheet_summaries.append({"sheet": sheet, "rows": len(df), "variables": len(value_cols)})
    pd.concat(all_long, ignore_index=True).to_csv(PROCESSED / "xlsx_all_sheets_daily_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_sheet_summaries).to_csv(OUT / "prepared_xlsx_sheet_counts.csv", index=False, encoding="utf-8-sig")

    province = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"])
    city_cols = [c for c in CITY_COLS if c in province.columns]
    province[["日期", "合计"] + city_cols].to_csv(PROCESSED / "province_city_total_daily_wide.csv", index=False, encoding="utf-8-sig")
    province.melt(id_vars=["日期"], value_vars=city_cols, var_name="city", value_name="value").to_csv(
        PROCESSED / "province_city_total_daily_long.csv", index=False, encoding="utf-8-sig"
    )
    category_cols = [c for c in province.columns if c not in ["序号", "日期"] + city_cols]
    province.melt(id_vars=["日期"], value_vars=category_cols, var_name="variable", value_name="value").to_csv(
        PROCESSED / "province_category_daily_long.csv", index=False, encoding="utf-8-sig"
    )

    date_cols, date_map = date_cols_from_csv_header()
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    month_groups: dict[str, list[str]] = {}
    for col in date_cols:
        month = date_map[col].strftime("%Y-%m")
        month_groups.setdefault(month, []).append(col)
    monthly = df[META_COLS].copy()
    for month, cols in sorted(month_groups.items()):
        monthly[month] = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    monthly.to_csv(PROCESSED / "csv_monthly_by_row_wide.csv", index=False, encoding="utf-8-sig")
    monthly_long = monthly.melt(id_vars=META_COLS, var_name="month", value_name="value")
    monthly_long.to_csv(PROCESSED / "csv_monthly_by_city_industry_long.csv", index=False, encoding="utf-8-sig")

    md = f"""
# 数据处理说明：福建用电量数据

## 已生成表

| 表 | 说明 |
| --- | --- |
| `xlsx_全省_daily_wide.csv` | 全省工作表清洗后的日度宽表 |
| `xlsx_all_sheets_daily_long.csv` | Excel 所有工作表转成长表 |
| `province_city_total_daily_wide.csv` | 全省表中的地市总量宽表 |
| `province_city_total_daily_long.csv` | 地市总量长表 |
| `province_category_daily_long.csv` | 全省用电类别和行业变量长表 |
| `csv_monthly_by_row_wide.csv` | CSV 宽表按月聚合后的宽表 |
| `csv_monthly_by_city_industry_long.csv` | CSV 月度地市/行业长表 |

## 为什么这样处理

原 CSV 每一天是一列，直接建模不方便；先按月聚合可以保留地市、行业、用电类别信息，同时避免生成过大的日度长表。Excel 已经是日度行格式，所以直接作为预测和 EDA 的主数据源。

## 当前主分析表

第一轮预测使用：

- `Data/processed/xlsx_全省_daily_wide.csv`
- `Data/processed/province_city_total_daily_wide.csv`

第一轮结构分析使用：

- `Data/processed/province_category_daily_long.csv`
- `Data/processed/csv_monthly_by_city_industry_long.csv`
"""
    write_md(NOTES / "数据处理说明-2026-05-06.md", md)


def svg_line(path: Path, series: list[tuple[str, list[float]]], labels: list[str], title: str, width=1100, height=420) -> None:
    margin = dict(left=70, right=30, top=55, bottom=70)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    vals = [v for _, ys in series for v in ys if pd.notna(v)]
    ymin, ymax = min(vals), max(vals)
    if ymax == ymin:
        ymax = ymin + 1
    def xy(i, y, n):
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
        parts.append(f'<text x="62" y="{y+4}" text-anchor="end" font-size="11" font-family="Arial">{val/1e8:.1f}亿</text>')
    n = len(labels)
    tick_idx = np.linspace(0, n - 1, min(8, n), dtype=int)
    for i in tick_idx:
        x, _ = xy(i, ymin, n)
        parts.append(f'<text x="{x}" y="{height-35}" text-anchor="middle" font-size="11" font-family="Arial">{labels[i]}</text>')
    for si, (name, ys) in enumerate(series):
        pts = []
        for i, y in enumerate(ys):
            if pd.notna(y):
                x, yy = xy(i, y, n)
                pts.append(f"{x:.1f},{yy:.1f}")
        color = colors[si % len(colors)]
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        lx = margin["left"] + si * 180
        parts.append(f'<line x1="{lx}" x2="{lx+24}" y1="{height-16}" y2="{height-16}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+30}" y="{height-12}" font-size="12" font-family="Arial">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_bar(path: Path, labels: list[str], values: list[float], title: str, width=980, height=520) -> None:
    margin = dict(left=210, right=40, top=55, bottom=45)
    plot_w = width - margin["left"] - margin["right"]
    bar_h = min(30, (height - margin["top"] - margin["bottom"]) / max(len(labels), 1) * 0.72)
    gap = ((height - margin["top"] - margin["bottom"]) - bar_h * len(labels)) / max(len(labels) - 1, 1)
    vmax = max(values) if values else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{title}</text>',
    ]
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = margin["top"] + i * (bar_h + gap)
        w = val / vmax * plot_w
        parts.append(f'<text x="{margin["left"]-10}" y="{y+bar_h*0.68}" text-anchor="end" font-size="12" font-family="Arial">{lab}</text>')
        parts.append(f'<rect x="{margin["left"]}" y="{y}" width="{w}" height="{bar_h}" fill="#2563eb" opacity="0.82"/>')
        parts.append(f'<text x="{margin["left"]+w+6}" y="{y+bar_h*0.68}" font-size="11" font-family="Arial">{val/1e8:.1f}亿</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def eda() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"])
    df = df.sort_values("日期")
    full = df[df["日期"] < "2026-01-01"].copy()
    monthly = full.set_index("日期")["合计"].resample("MS").sum().reset_index()
    svg_line(FIG / "eda_全省日度合计.svg", [("全省合计", df["合计"].tolist())], df["日期"].dt.strftime("%Y-%m-%d").tolist(), "福建全省日度用电量")
    svg_line(FIG / "eda_全省月度合计.svg", [("月度合计", monthly["合计"].tolist())], monthly["日期"].dt.strftime("%Y-%m").tolist(), "福建全省月度用电量")

    city = pd.read_csv(PROCESSED / "province_city_total_daily_long.csv", parse_dates=["日期"])
    city_full = city[city["日期"] < "2026-01-01"]
    city_totals = city_full.groupby("city", as_index=False)["value"].sum().sort_values("value", ascending=True)
    svg_bar(FIG / "eda_地市总量对比.svg", city_totals["city"].tolist(), city_totals["value"].tolist(), "2023-2025 福建各地市用电量对比")

    cat = pd.read_csv(PROCESSED / "province_category_daily_long.csv", parse_dates=["日期"])
    cat_full = cat[(cat["日期"] < "2026-01-01") & (cat["variable"] != "合计")]
    cat_totals = cat_full.groupby("variable", as_index=False)["value"].sum().sort_values("value", ascending=False).head(18)
    cat_totals_asc = cat_totals.sort_values("value", ascending=True)
    svg_bar(FIG / "eda_重点变量总量对比.svg", cat_totals_asc["variable"].tolist(), cat_totals_asc["value"].tolist(), "2023-2025 福建重点用电变量总量")

    dow = full.assign(dow=full["日期"].dt.dayofweek).groupby("dow", as_index=False)["合计"].mean()
    dow["name"] = dow["dow"].map({0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"})
    svg_bar(FIG / "eda_星期效应.svg", dow["name"].tolist(), dow["合计"].tolist(), "福建全省合计用电的星期效应")

    annual = full.assign(year=full["日期"].dt.year).groupby("year", as_index=False)["合计"].sum()
    annual["yoy"] = annual["合计"].pct_change()
    annual.to_csv(OUT / "eda_annual_total.csv", index=False, encoding="utf-8-sig")
    city_totals.to_csv(OUT / "eda_city_totals_2023_2025.csv", index=False, encoding="utf-8-sig")
    cat_totals.to_csv(OUT / "eda_top_variables_2023_2025.csv", index=False, encoding="utf-8-sig")
    dow.to_csv(OUT / "eda_dayofweek_pattern.csv", index=False, encoding="utf-8-sig")

    top_city = city_totals.sort_values("value", ascending=False).iloc[0]
    top_var = cat_totals.iloc[0]
    md = f"""
# EDA 研究过程：福建用电量

## 1. 分析对象

第一轮 EDA 使用 `xlsx_全省_daily_wide.csv` 和 `province_city_total_daily_long.csv`。为避免 2026 年只有 1 月上旬导致年度解释偏误，年度和结构统计主要使用 2023-2025 完整年份。

## 2. 全省时间序列

已生成图：`Experiments/figures/eda_全省日度合计.svg` 和 `eda_全省月度合计.svg`。

这些图用于观察趋势、春节低谷、夏季高峰和异常波动。日度数据有明显星期效应，因此预测模型不能随机切分训练集和测试集。

## 3. 地市结构

2023-2025 年用电量最高的地市是 **{top_city["city"]}**，合计约 **{top_city["value"]/1e8:.2f} 亿**。完整表见 `eda_city_totals_2023_2025.csv`。

## 4. 变量结构

除地市列外，2023-2025 年总量最高的变量是 **{top_var["variable"]}**，合计约 **{top_var["value"]/1e8:.2f} 亿**。完整表见 `eda_top_variables_2023_2025.csv`。

## 5. 星期效应

已生成 `eda_星期效应.svg`。星期效应说明：如果直接用日度数据做 Granger 或回归，必须加入日历特征，或者聚合到周度/月度。

## 6. 对后续模型的影响

1. 预测部分采用时间顺序划分，不能随机切分。
2. 基线至少要包含昨日、上周同日、去年同日。
3. 行业传导部分优先使用周度或月度数据，降低星期效应造成的伪关系。
4. 所有 2026 年结论仅作为最新样本延伸，不作为完整年度比较。
"""
    write_md(NOTES / "EDA研究过程-2026-05-06.md", md)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    p = y_pred[mask]
    err = y - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9)))
    smape = float(np.mean(2 * np.abs(err) / np.maximum(np.abs(y) + np.abs(p), 1e-9)))
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "n": int(len(y))}


@dataclass
class RidgeModel:
    beta: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    cols: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X[self.cols].to_numpy(dtype=float)
        Xs = (Xv - self.mean) / self.std
        Xd = np.column_stack([np.ones(len(Xs)), Xs])
        return Xd @ self.beta


def fit_ridge(train: pd.DataFrame, target: str, feature_cols: list[str], alpha: float = 10.0) -> RidgeModel:
    X = train[feature_cols].to_numpy(dtype=float)
    y = train[target].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    pen = np.eye(Xd.shape[1]) * alpha
    pen[0, 0] = 0
    beta = np.linalg.pinv(Xd.T @ Xd + pen) @ Xd.T @ y
    return RidgeModel(beta=beta, mean=mean, std=std, cols=feature_cols)


def add_time_features(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    date = out["日期"]
    out["dow"] = date.dt.dayofweek
    out["month"] = date.dt.month
    out["doy_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 366)
    out["doy_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 366)
    for d in range(7):
        out[f"dow_{d}"] = (out["dow"] == d).astype(int)
    return out


def make_lag_features(df: pd.DataFrame, target: str, external_cols: list[str] | None = None) -> pd.DataFrame:
    out = add_time_features(df[["日期", target] + ([c for c in external_cols if c in df.columns] if external_cols else [])].copy())
    lags = [1, 2, 3, 7, 14, 28, 365]
    for lag in lags:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    out[f"{target}_roll7"] = out[target].shift(1).rolling(7).mean()
    out[f"{target}_roll28"] = out[target].shift(1).rolling(28).mean()
    if external_cols:
        for col in external_cols:
            if col not in df.columns or col == target:
                continue
            for lag in [1, 7, 28]:
                out[f"{col}_lag{lag}"] = df[col].shift(lag)
    return out


def forecast() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    df = df[df["日期"] < "2026-01-10"].copy()
    test_start = pd.Timestamp("2025-10-01")
    targets = [t for t in ["合计", "大工业电量", "居民生活", "商业用电", "制造业", "工业"] if t in df.columns]
    external = [c for c in KEY_VARIABLES if c in df.columns and c != "合计"]
    metric_rows = []
    pred_frames = []
    for target in targets:
        feat = make_lag_features(df, target, external_cols=external if target == "合计" else None)
        feat["naive_yesterday"] = feat[target].shift(1)
        feat["seasonal_week"] = feat[target].shift(7)
        feat["seasonal_year"] = feat[target].shift(365)
        feat["rolling7"] = feat[target].shift(1).rolling(7).mean()
        feat["rolling28"] = feat[target].shift(1).rolling(28).mean()
        model_df = feat.dropna().copy()
        train = model_df[model_df["日期"] < test_start].copy()
        test = model_df[model_df["日期"] >= test_start].copy()
        for model in ["naive_yesterday", "seasonal_week", "seasonal_year", "rolling7", "rolling28"]:
            m = metrics(test[target].to_numpy(float), test[model].to_numpy(float))
            metric_rows.append({"target": target, "model": model, **m})
            pred_frames.append(test[["日期", target]].assign(target_name=target, model=model, prediction=test[model].to_numpy()))
        own_cols = [c for c in model_df.columns if c.startswith(f"{target}_lag") or c.startswith(f"{target}_roll")] + [
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
        own = fit_ridge(train, target, own_cols)
        own_pred = own.predict(test)
        m = metrics(test[target].to_numpy(float), own_pred)
        metric_rows.append({"target": target, "model": "ridge_own_lags_calendar", **m})
        pred_frames.append(test[["日期", target]].assign(target_name=target, model="ridge_own_lags_calendar", prediction=own_pred))
        if target == "合计":
            ext_cols = own_cols + [c for c in model_df.columns if any(c.startswith(f"{v}_lag") for v in external)]
            enh = fit_ridge(train, target, ext_cols)
            enh_pred = enh.predict(test)
            m = metrics(test[target].to_numpy(float), enh_pred)
            metric_rows.append({"target": target, "model": "ridge_with_category_lags", **m})
            pred_frames.append(test[["日期", target]].assign(target_name=target, model="ridge_with_category_lags", prediction=enh_pred))

    metrics_df = pd.DataFrame(metric_rows).sort_values(["target", "rmse"])
    preds = pd.concat(pred_frames, ignore_index=True)
    metrics_df.to_csv(OUT / "forecast_metrics_daily.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT / "forecast_predictions_daily.csv", index=False, encoding="utf-8-sig")

    total_preds = preds[preds["target_name"] == "合计"].copy()
    best_total_model = metrics_df[metrics_df["target"] == "合计"].sort_values("rmse").iloc[0]["model"]
    plot = total_preds[total_preds["model"].isin(["seasonal_week", "seasonal_year", "ridge_own_lags_calendar", "ridge_with_category_lags"])]
    actual = df[df["日期"] >= test_start][["日期", "合计"]].dropna()
    labels = actual["日期"].dt.strftime("%Y-%m-%d").tolist()
    series = [("实际", actual["合计"].tolist())]
    for model in plot["model"].unique():
        p = plot[plot["model"] == model].set_index("日期").reindex(actual["日期"])["prediction"].tolist()
        series.append((model, p))
    svg_line(FIG / "forecast_全省合计回测.svg", series, labels, "福建全省合计用电量：2025Q4 回测")

    own_feat = make_lag_features(df, "合计")
    own_feat = own_feat.dropna().copy()
    own_cols = [c for c in own_feat.columns if c.startswith("合计_lag") or c.startswith("合计_roll")] + [
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
    own_model = fit_ridge(own_feat, "合计", own_cols)
    hist = df[["日期", "合计"]].copy()
    future_rows = []
    for step in range(1, 31):
        next_date = hist["日期"].max() + pd.Timedelta(days=1)
        temp = hist.copy()
        temp = make_lag_features(temp, "合计")
        row = add_time_features(pd.DataFrame({"日期": [next_date], "合计": [np.nan]}))
        for lag in [1, 2, 3, 7, 14, 28, 365]:
            row[f"合计_lag{lag}"] = hist["合计"].iloc[-lag] if len(hist) >= lag else np.nan
        row["合计_roll7"] = hist["合计"].iloc[-7:].mean()
        row["合计_roll28"] = hist["合计"].iloc[-28:].mean()
        pred = float(own_model.predict(row[["日期", "合计"] + own_cols])[0])
        future_rows.append({"日期": next_date, "prediction": pred, "model": "ridge_own_recursive_demo"})
        hist = pd.concat([hist, pd.DataFrame({"日期": [next_date], "合计": [pred]})], ignore_index=True)
    future = pd.DataFrame(future_rows)
    future.to_csv(OUT / "forecast_next30_total.csv", index=False, encoding="utf-8-sig")

    top = metrics_df.groupby("target", as_index=False).first()[["target", "model", "rmse", "mape", "smape"]]
    lines = ["| 目标 | 最优模型 | RMSE | MAPE | sMAPE |", "| --- | --- | ---: | ---: | ---: |"]
    for _, r in top.iterrows():
        lines.append(f"| {r['target']} | {r['model']} | {r['rmse']:.2f} | {pct(r['mape'])} | {pct(r['smape'])} |")
    total_base = metrics_df[(metrics_df["target"] == "合计") & (metrics_df["model"] == "ridge_own_lags_calendar")].iloc[0]
    total_enh = metrics_df[(metrics_df["target"] == "合计") & (metrics_df["model"] == "ridge_with_category_lags")]
    enh_text = ""
    if not total_enh.empty:
        enh = total_enh.iloc[0]
        gain = 1 - enh["rmse"] / total_base["rmse"]
        enh_text = f"加入行业/类别滞后特征后，全省合计 RMSE 相对自身历史模型变化为 **{gain*100:.2f}%**。这只能解释为预测贡献，不是强因果。"
    md = f"""
# 预测基线研究过程：福建日度用电量

## 1. 预测问题

第一轮预测回答：在只使用历史用电量和日历信息的情况下，福建全省总用电量以及重点类别日度用电量能预测到什么水平？进一步地，加入其他行业或用电类别的滞后信息后，全省合计预测是否改善？

## 2. 回测设定

- 数据：`Data/processed/xlsx_全省_daily_wide.csv`
- 训练期：2023-01-01 至 2025-09-30
- 测试期：2025-10-01 至 2026-01-09
- 预测方式：单步滚动回测。测试集中每一天只使用当日以前的历史滞后特征。

## 3. 模型

1. `naive_yesterday`：用昨日值预测今日。
2. `seasonal_week`：用上周同日预测今日。
3. `seasonal_year`：用去年同日预测今日。
4. `rolling7` 和 `rolling28`：用过去 7 日或 28 日均值预测。
5. `ridge_own_lags_calendar`：使用自身滞后、滚动均值和日历特征的岭回归。
6. `ridge_with_category_lags`：在全省合计预测中加入重点行业/类别滞后特征。

## 4. 最优结果摘要

{chr(10).join(lines)}

## 5. 行业/类别滞后特征是否有帮助

{enh_text}

## 6. 已生成结果

- 指标表：`Experiments/outputs/forecast_metrics_daily.csv`
- 回测预测值：`Experiments/outputs/forecast_predictions_daily.csv`
- 未来 30 日示范预测：`Experiments/outputs/forecast_next30_total.csv`
- 回测图：`Experiments/figures/forecast_全省合计回测.svg`

## 7. 结论边界

预测准确说明模型捕捉到时间规律和滞后信息，但不能直接证明行业之间存在结构因果。后续需要结合 Granger、VAR、行业结构解释和稳健性检查，把结论限制在合适证据等级内。
"""
    write_md(NOTES / "预测基线研究过程-2026-05-06.md", md)


def structure() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    full = df[df["日期"] < "2026-01-01"].copy()
    full["year"] = full["日期"].dt.year
    city_cols = [c for c in CITY_COLS if c in full.columns]
    vars_all = [c for c in full.columns if c not in ["序号", "日期", "year"]]
    annual = full.groupby("year")[vars_all].sum().T.reset_index().rename(columns={"index": "variable"})
    for y in [2023, 2024, 2025]:
        if y in annual.columns and "合计" in annual["variable"].values:
            total = float(annual.loc[annual["variable"] == "合计", y].iloc[0])
            annual[f"share_{y}"] = annual[y] / total
    annual["total_2023_2025"] = annual[[c for c in [2023, 2024, 2025] if c in annual.columns]].sum(axis=1)
    annual.to_csv(OUT / "structure_annual_variable_totals.csv", index=False, encoding="utf-8-sig")

    city_annual = full.groupby("year")[city_cols].sum().reset_index().melt(id_vars="year", var_name="city", value_name="value")
    city_annual.to_csv(OUT / "structure_city_annual_totals.csv", index=False, encoding="utf-8-sig")
    city_total = city_annual.groupby("city", as_index=False)["value"].sum()
    total_city_sum = city_total["value"].sum()
    city_total["share"] = city_total["value"] / total_city_sum
    city_total.to_csv(OUT / "structure_city_share_2023_2025.csv", index=False, encoding="utf-8-sig")

    non_city_vars = [v for v in vars_all if v not in city_cols and v != "合计"]
    var_total = full[non_city_vars].sum().sort_values(ascending=False).reset_index()
    var_total.columns = ["variable", "value"]
    var_total["share_of_total"] = var_total["value"] / full["合计"].sum()
    var_total.to_csv(OUT / "structure_variable_share_2023_2025.csv", index=False, encoding="utf-8-sig")

    top_var = var_total.iloc[0]
    top_city = city_total.sort_values("value", ascending=False).iloc[0]
    top_vars_asc = var_total.head(15).sort_values("value", ascending=True)
    svg_bar(FIG / "structure_用电变量结构.svg", top_vars_asc["variable"].tolist(), top_vars_asc["value"].tolist(), "2023-2025 福建用电结构：重点变量")
    city_asc = city_total.sort_values("value", ascending=True)
    svg_bar(FIG / "structure_地市份额.svg", city_asc["city"].tolist(), city_asc["value"].tolist(), "2023-2025 福建地市用电结构")

    md = f"""
# 行业结构研究过程：福建用电量

## 1. 目标

结构分析回答：福建用电量主要由哪些地市、用电类别和行业变量构成？哪些变量在 2023-2025 年占比高、波动大，适合作为后续预测和传导分析重点？

## 2. 数据处理

使用 `Data/processed/xlsx_全省_daily_wide.csv`，只对 2023-2025 完整年份做年度结构统计，避免 2026 年不完整样本影响判断。

## 3. 地市结构

2023-2025 年用电量最高的地市是 **{top_city['city']}**，占地市总量约 **{pct(top_city['share'])}**。完整表见 `structure_city_share_2023_2025.csv`。

## 4. 行业和用电类别结构

2023-2025 年除地市列外，总量最高的变量是 **{top_var['variable']}**，占全省合计约 **{pct(top_var['share_of_total'])}**。完整表见 `structure_variable_share_2023_2025.csv`。

## 5. 已生成图表

- `Experiments/figures/structure_用电变量结构.svg`
- `Experiments/figures/structure_地市份额.svg`

## 6. 对后续研究的作用

1. 预测模型优先覆盖全省合计、工业、大工业、居民生活、商业用电和制造业。
2. 传导分析优先选择结构占比高且波动明显的变量。
3. 地市层面可以先做福州、泉州、厦门、宁德等重点地市，再扩展到全省网络。

## 7. 边界

这些结构结论是描述性证据，不能直接解释为上下游因果。行业结构只告诉我们哪些变量重要，传导方向仍需要时序模型和产业知识共同判断。
"""
    write_md(NOTES / "行业结构研究过程-2026-05-06.md", md)


def betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-12
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def reg_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def f_sf(f: float, d1: int, d2: int) -> float:
    if not np.isfinite(f) or f < 0:
        return 1.0
    x = d1 * f / (d1 * f + d2)
    return max(0.0, min(1.0, 1.0 - reg_incomplete_beta(d1 / 2.0, d2 / 2.0, x)))


def ols_rss(y: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    beta = np.linalg.pinv(X) @ y
    resid = y - X @ beta
    return float(resid @ resid), X.shape[1]


def granger_pair(data: pd.DataFrame, cause: str, effect: str, lag: int) -> dict[str, float]:
    rows = []
    y = data[effect].to_numpy(float)
    x = data[cause].to_numpy(float)
    for t in range(lag, len(data)):
        row_y_lags = [y[t - i] for i in range(1, lag + 1)]
        row_x_lags = [x[t - i] for i in range(1, lag + 1)]
        if np.all(np.isfinite([y[t]] + row_y_lags + row_x_lags)):
            rows.append((y[t], row_y_lags, row_x_lags))
    if len(rows) < lag * 8:
        return {"f_stat": np.nan, "p_value": 1.0, "n": len(rows)}
    yy = np.array([r[0] for r in rows], dtype=float)
    Ylags = np.array([r[1] for r in rows], dtype=float)
    Xlags = np.array([r[2] for r in rows], dtype=float)
    Xr = np.column_stack([np.ones(len(yy)), Ylags])
    Xu = np.column_stack([np.ones(len(yy)), Ylags, Xlags])
    rss_r, _ = ols_rss(yy, Xr)
    rss_u, ku = ols_rss(yy, Xu)
    df1 = lag
    df2 = len(yy) - ku
    if rss_u <= 0 or df2 <= 0:
        return {"f_stat": np.nan, "p_value": 1.0, "n": len(rows)}
    f = ((rss_r - rss_u) / df1) / (rss_u / df2)
    p = f_sf(f, df1, df2)
    return {"f_stat": f, "p_value": p, "n": len(rows)}


def fdr_bh(pvals: pd.Series) -> pd.Series:
    p = pvals.to_numpy(float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        i = n - rank + 1
        val = min(prev, p[idx] * n / i)
        q[idx] = val
        prev = val
    return pd.Series(q, index=pvals.index)


def granger() -> None:
    ensure_dirs()
    df = pd.read_csv(PROCESSED / "xlsx_全省_daily_wide.csv", parse_dates=["日期"]).sort_values("日期")
    df = df[(df["日期"] >= "2023-01-01") & (df["日期"] < "2026-01-01")]
    variables = [v for v in KEY_VARIABLES if v in df.columns]
    weekly = df.set_index("日期")[variables].resample("W-SUN").sum()
    trans = np.log1p(weekly).diff().dropna()
    rows = []
    for effect in variables:
        for cause in variables:
            if cause == effect:
                continue
            best = None
            for lag in [1, 2, 3, 4]:
                res = granger_pair(trans, cause, effect, lag)
                rec = {"cause": cause, "effect": effect, "lag_weeks": lag, **res}
                if best is None or rec["p_value"] < best["p_value"]:
                    best = rec
            rows.append(best)
    resdf = pd.DataFrame(rows)
    resdf["q_value_fdr"] = fdr_bh(resdf["p_value"])
    resdf["pass_fdr_05"] = resdf["q_value_fdr"] < 0.05
    resdf = resdf.sort_values(["q_value_fdr", "p_value", "effect", "cause"])
    resdf.to_csv(OUT / "granger_weekly_logdiff_edges.csv", index=False, encoding="utf-8-sig")
    top = resdf.head(40).copy()
    top.to_csv(OUT / "granger_weekly_top40_edges.csv", index=False, encoding="utf-8-sig")

    pass_edges = resdf[resdf["pass_fdr_05"]]
    lines = ["| 起点 | 终点 | 滞后周数 | F统计量 | p值 | FDR q值 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for _, r in top.head(15).iterrows():
        lines.append(f"| {r['cause']} | {r['effect']} | {int(r['lag_weeks'])} | {r['f_stat']:.2f} | {r['p_value']:.4g} | {r['q_value_fdr']:.4g} |")
    md = f"""
# 时序传导研究过程：福建周度 Granger 初筛

## 1. 为什么用周度而不是日度

日度用电量有明显星期效应。如果直接用日度序列两两检验，模型可能把共同的星期节奏误判成行业传导。因此第一轮先把日度数据聚合为周度总量，再对 `log1p` 后的一阶差分做 Granger 初筛。

## 2. 变量集合

本轮变量来自全省表的重点类别和行业变量，包括：{", ".join(variables)}

## 3. 检验方式

对每一组 `cause -> effect`，测试 1 到 4 周滞后，保留 p 值最小的滞后阶数。p 值基于受限模型和非受限模型的 F 检验；随后对所有边做 Benjamini-Hochberg FDR 校正。

原假设是：

$$
H_0: cause 的滞后项整体没有改善 effect 的预测
$$

拒绝原假设只能说明“时序领先或预测贡献”，不能直接说明结构因果。

## 4. 初筛结果

- 总候选边数：{len(resdf)}
- 通过 FDR 0.05 的边数：{len(pass_edges)}

前 15 条边：

{chr(10).join(lines)}

完整结果见：

- `Experiments/outputs/granger_weekly_logdiff_edges.csv`
- `Experiments/outputs/granger_weekly_top40_edges.csv`

## 5. 解释边界

本轮是 Level 1 证据。它没有控制天气、节假日、价格、政策事件和共同经济周期。后续应对通过 FDR 的边做：滞后阶数替换、训练窗口替换、加入日历/天气控制、与产业结构解释对照。
"""
    write_md(NOTES / "时序传导研究过程-2026-05-06.md", md)


def summary_report() -> None:
    ensure_dirs()
    metric_path = OUT / "forecast_metrics_daily.csv"
    struct_path = OUT / "structure_variable_share_2023_2025.csv"
    granger_path = OUT / "granger_weekly_logdiff_edges.csv"
    metric_summary = ""
    if metric_path.exists():
        metrics_df = pd.read_csv(metric_path)
        best = metrics_df.sort_values("rmse").groupby("target", as_index=False).first()
        lines = ["| 目标 | 最优模型 | MAPE | RMSE |", "| --- | --- | ---: | ---: |"]
        for _, r in best.iterrows():
            lines.append(f"| {r['target']} | {r['model']} | {pct(r['mape'])} | {r['rmse']:.2f} |")
        metric_summary = "\n".join(lines)
    struct_summary = ""
    if struct_path.exists():
        s = pd.read_csv(struct_path).head(10)
        lines = ["| 变量 | 2023-2025 总量 | 占全省合计 |", "| --- | ---: | ---: |"]
        for _, r in s.iterrows():
            lines.append(f"| {r['variable']} | {r['value']/1e8:.2f} 亿 | {pct(r['share_of_total'])} |")
        struct_summary = "\n".join(lines)
    granger_summary = ""
    if granger_path.exists():
        g = pd.read_csv(granger_path).head(10)
        lines = ["| 起点 | 终点 | 滞后周数 | q值 |", "| --- | --- | ---: | ---: |"]
        for _, r in g.iterrows():
            lines.append(f"| {r['cause']} | {r['effect']} | {int(r['lag_weeks'])} | {r['q_value_fdr']:.4g} |")
        granger_summary = "\n".join(lines)

    md = f"""
# 福建用电量预测与产业因果推断：第一轮实证研究报告

## 1. 研究对象

本轮研究对象是福建省日度用电量数据。它继承新疆项目的方法框架，但研究结论只对福建数据口径成立，不能直接写成新疆结论。

## 2. 数据

- CSV 宽表：`Sources/原始数据/分行业&分用电类别(1).csv`
- Excel 工作簿：`Sources/原始数据/福建全省用电量统计20230101至20260109.xlsx`

第一轮建模以 Excel 全省表为主，CSV 用于补充地市/行业/月度结构。

## 3. 数据处理

已完成：

- 原始数据审计；
- Excel 工作表清洗为日度宽表和长表；
- CSV 宽表按月聚合；
- 变量字典和数据清单。

详见：

- `Notes/数据审计报告-2026-05-06.md`
- `Notes/数据处理说明-2026-05-06.md`

## 4. EDA

已生成全省日度、月度、地市结构、变量结构和星期效应图。EDA 的关键结论是：日度用电具有明显星期效应和季节性，因此预测必须按时间顺序回测，传导分析不宜直接用原始日度序列两两检验。

## 5. 预测结果

{metric_summary}

回测图：`Experiments/figures/forecast_全省合计回测.svg`

## 6. 行业结构

{struct_summary}

结构图：

- `Experiments/figures/structure_用电变量结构.svg`
- `Experiments/figures/structure_地市份额.svg`

## 7. 时序传导初筛

本轮使用周度 `log1p` 一阶差分做 Granger 初筛，并进行 FDR 校正。前 10 条候选边如下：

{granger_summary}

完整边表：`Experiments/outputs/granger_weekly_logdiff_edges.csv`

## 8. 当前结论等级

- 预测结果：可作为模型性能证据。
- 行业结构：描述性结构证据。
- Granger 边：Level 1，时序领先或预测贡献证据。

目前不能写成“某行业导致某行业用电变化”。更稳妥写法是：“某变量的历史变化对目标变量具有前置预测信息，提示可能存在传导关系，仍需进一步控制天气、节假日、政策和共同冲击。”

## 9. 下一步

1. 对重点地市分别建立预测基线。
2. 对通过 FDR 的边做滚动窗口稳健性检查。
3. 加入节假日和天气数据，检查预测和传导边是否稳定。
4. 如果能建立政策事件时间线，再做准实验或 CausalImpact。
"""
    write_md(WRITING / "福建实证第一轮研究报告-2026-05-06.md", md)
    write_md(NOTES / "研究过程总览-2026-05-06.md", md)


def run_all() -> None:
    audit()
    prepare()
    eda()
    forecast()
    structure()
    granger()
    summary_report()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["audit", "prepare", "eda", "forecast", "structure", "granger", "summary", "all"])
    args = parser.parse_args()
    if args.step == "audit":
        audit()
    elif args.step == "prepare":
        prepare()
    elif args.step == "eda":
        eda()
    elif args.step == "forecast":
        forecast()
    elif args.step == "structure":
        structure()
    elif args.step == "granger":
        granger()
    elif args.step == "summary":
        summary_report()
    else:
        run_all()


if __name__ == "__main__":
    main()
