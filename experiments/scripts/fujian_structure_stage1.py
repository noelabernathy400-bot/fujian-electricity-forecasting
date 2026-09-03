from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "Sources" / "原始数据"
PROCESSED = PROJECT / "Data" / "processed" / "structure_stage1"
DICTIONARY = PROJECT / "Data" / "dictionary" / "structure_stage1"
OUT = PROJECT / "Experiments" / "outputs" / "structure_stage1"
TABLES = PROJECT / "Writing" / "LaTeX" / "tables"

CSV_PATH = RAW / "分行业&分用电类别(1).csv"

META_COLS = ["类型", "汇总方式", "单位代码", "单位名称", "行号", "行业代码", "行业名称"]

TOTAL_NAME = "全社会用电总计"
STRUCTURE_NAMES = {
    "第一产业": "第一产业",
    "第二产业": "第二产业",
    "第三产业": "第三产业",
    "B、城乡居民生活用电合计": "城乡居民生活用电合计",
}
MANUFACTURING_TOTAL = "（二） 制造业"

MANUFACTURING_GROUPS = {
    "1.农副食品加工业": "消费与轻工制造",
    "2.食品制造业": "消费与轻工制造",
    "3.酒、饮料及精制茶制造业": "消费与轻工制造",
    "4.烟草制品业": "消费与轻工制造",
    "5.纺织业": "消费与轻工制造",
    "6.纺织服装、服饰业": "消费与轻工制造",
    "7.皮革、毛皮、羽毛及其制品和制鞋业": "消费与轻工制造",
    "8.木材加工和木、竹、藤、棕、草制品业": "消费与轻工制造",
    "9.家具制造业": "消费与轻工制造",
    "10.造纸和纸制品业": "消费与轻工制造",
    "11.印刷和记录媒介复制业": "消费与轻工制造",
    "12.文教、工美、体育和娱乐用品制造业": "消费与轻工制造",
    "13.石油、煤炭及其他燃料加工业": "传统高耗能与材料行业",
    "14.化学原料和化学制品制造业": "传统高耗能与材料行业",
    "15.医药制造业": "先进制造与技术相关行业",
    "16.化学纤维制造业": "传统高耗能与材料行业",
    "17.橡胶和塑料制品业": "传统高耗能与材料行业",
    "18.非金属矿物制品业": "传统高耗能与材料行业",
    "19.黑色金属冶炼和压延加工业": "传统高耗能与材料行业",
    "20.有色金属冶炼和压延加工业": "传统高耗能与材料行业",
    "21.金属制品业": "装备制造",
    "22.通用设备制造业": "装备制造",
    "23.专用设备制造业": "装备制造",
    "24.汽车制造业": "装备制造",
    "25.铁路、船舶、航空航天和其他运输设备制造业": "装备制造",
    "26.电气机械和器材制造业": "装备制造",
    "27.计算机、通信和其他电子设备制造业": "先进制造与技术相关行业",
    "28.仪器仪表制造业": "先进制造与技术相关行业",
    "29.其他制造业": "其他制造",
    "30.废弃资源综合利用业": "其他制造",
    "31.金属制品、机械和设备修理业": "其他制造",
}


def ensure_dirs() -> None:
    for path in [PROCESSED, DICTIONARY, OUT, TABLES]:
        path.mkdir(parents=True, exist_ok=True)


def parse_cn_date_col(col: str) -> pd.Timestamp | None:
    match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", str(col))
    if not match:
        return None
    year, month, day = map(int, match.groups())
    return pd.Timestamp(year=year, month=month, day=day)


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100 * x:.4f}\\%"


def fmt_num(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.2f}"


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_latex_tabular(df: pd.DataFrame, path: Path, escape: bool = True) -> None:
    cols = list(df.columns)
    align = "l" * len(cols)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    header = " & ".join(latex_escape(col) if escape else str(col) for col in cols) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        vals = [latex_escape(row[col]) if escape else str(row[col]) for col in cols]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    header = pd.read_csv(CSV_PATH, nrows=0, encoding="utf-8").columns.tolist()
    date_cols = [col for col in header if parse_cn_date_col(col) is not None]
    date_map = {col: parse_cn_date_col(col) for col in date_cols}

    raw = pd.read_csv(CSV_PATH, encoding="utf-8", dtype=str)
    raw["行业名称_清洗"] = raw["行业名称"].astype(str).str.strip()

    province = raw[
        (raw["类型"].astype(str).str.strip() == "用电")
        & (raw["汇总方式"].astype(str).str.strip() == "按地市汇总")
        & (raw["单位名称"].astype(str).str.strip() == "全省")
    ].copy()

    if province.empty:
        raise RuntimeError("未找到 类型=用电、汇总方式=按地市汇总、单位名称=全省 的记录。")

    keep = META_COLS + ["行业名称_清洗"] + date_cols
    province = province[keep].copy()
    for col in date_cols:
        province[col] = pd.to_numeric(province[col], errors="coerce")

    long = province.melt(
        id_vars=META_COLS + ["行业名称_清洗"],
        value_vars=date_cols,
        var_name="date_raw",
        value_name="electricity",
    )
    long["date"] = long["date_raw"].map(date_map)
    long = long.drop(columns=["date_raw"])
    long = long.sort_values(["date", "行业名称_清洗"]).reset_index(drop=True)
    long.to_csv(PROCESSED / "province_usage_daily_long.csv", index=False, encoding="utf-8-sig")

    wide = long.pivot_table(index="date", columns="行业名称_清洗", values="electricity", aggfunc="first").sort_index()
    wide.to_csv(PROCESSED / "province_usage_daily_wide.csv", encoding="utf-8-sig")

    dictionary = province[META_COLS + ["行业名称_清洗"]].drop_duplicates().sort_values("行业名称_清洗")
    dictionary.to_csv(DICTIONARY / "province_usage_row_dictionary.csv", index=False, encoding="utf-8-sig")

    missing_structure = [name for name in [TOTAL_NAME, *STRUCTURE_NAMES.keys()] if name not in wide.columns]
    if missing_structure:
        raise RuntimeError(f"总量闭合需要的列缺失：{missing_structure}")

    total_parts = wide[list(STRUCTURE_NAMES.keys())].sum(axis=1)
    total_closure = pd.DataFrame(
        {
            "date": wide.index,
            "total": wide[TOTAL_NAME],
            "parts_sum": total_parts,
            "closure_error": wide[TOTAL_NAME] - total_parts,
        }
    )
    total_closure["abs_error"] = total_closure["closure_error"].abs()
    total_closure["relative_abs_error"] = total_closure["abs_error"] / total_closure["total"].replace(0, np.nan)
    total_closure.to_csv(OUT / "total_closure_daily.csv", index=False, encoding="utf-8-sig")

    manufacturing_names = list(MANUFACTURING_GROUPS.keys())
    present_manu = [name for name in manufacturing_names if name in wide.columns]
    missing_manu = [name for name in manufacturing_names if name not in wide.columns]
    if MANUFACTURING_TOTAL not in wide.columns:
        raise RuntimeError(f"制造业总量列缺失：{MANUFACTURING_TOTAL}")

    manu_parts = wide[present_manu].sum(axis=1)
    manu_closure = pd.DataFrame(
        {
            "date": wide.index,
            "manufacturing_total": wide[MANUFACTURING_TOTAL],
            "industry_sum": manu_parts,
            "closure_error": wide[MANUFACTURING_TOTAL] - manu_parts,
        }
    )
    manu_closure["abs_error"] = manu_closure["closure_error"].abs()
    manu_closure["relative_abs_error"] = manu_closure["abs_error"] / manu_closure["manufacturing_total"].replace(0, np.nan)
    manu_closure.to_csv(OUT / "manufacturing_closure_daily.csv", index=False, encoding="utf-8-sig")

    group_map = pd.DataFrame(
        [{"industry_name": name, "manufacturing_group": group, "present_in_data": name in wide.columns} for name, group in MANUFACTURING_GROUPS.items()]
    )
    group_map.to_csv(DICTIONARY / "manufacturing_industry_group_map.csv", index=False, encoding="utf-8-sig")

    group_daily = pd.DataFrame(index=wide.index)
    for group in sorted(set(MANUFACTURING_GROUPS.values())):
        names = [name for name, g in MANUFACTURING_GROUPS.items() if g == group and name in wide.columns]
        group_daily[group] = wide[names].sum(axis=1, min_count=1)
    group_daily.insert(0, "制造业", wide[MANUFACTURING_TOTAL])
    group_daily.to_csv(PROCESSED / "manufacturing_group_daily_wide.csv", encoding="utf-8-sig")

    group_share = group_daily.drop(columns=["制造业"]).div(group_daily["制造业"], axis=0)
    group_share.to_csv(PROCESSED / "manufacturing_group_share_daily_wide.csv", encoding="utf-8-sig")

    annual = pd.DataFrame(index=sorted(wide.index.year.unique()))
    annual.index.name = "year"
    annual["全社会用电总计"] = wide[TOTAL_NAME].groupby(wide.index.year).sum()
    for raw_name, clean_name in STRUCTURE_NAMES.items():
        annual[clean_name] = wide[raw_name].groupby(wide.index.year).sum()
    annual["制造业"] = wide[MANUFACTURING_TOTAL].groupby(wide.index.year).sum()
    annual = annual.loc[annual.index <= 2025].copy()
    annual.to_csv(OUT / "annual_structure_totals.csv", encoding="utf-8-sig")

    annual_share = pd.DataFrame(index=annual.index)
    for col in ["第一产业", "第二产业", "第三产业", "城乡居民生活用电合计", "制造业"]:
        annual_share[col] = annual[col] / annual["全社会用电总计"]
    annual_share.to_csv(OUT / "annual_structure_shares.csv", encoding="utf-8-sig")

    closure_summary = pd.DataFrame(
        [
            {
                "check": "全社会总量闭合",
                "days": len(total_closure),
                "max_abs_error": total_closure["abs_error"].max(),
                "mean_abs_error": total_closure["abs_error"].mean(),
                "max_relative_abs_error": total_closure["relative_abs_error"].max(),
                "mean_relative_abs_error": total_closure["relative_abs_error"].mean(),
            },
            {
                "check": "制造业31类闭合",
                "days": len(manu_closure),
                "max_abs_error": manu_closure["abs_error"].max(),
                "mean_abs_error": manu_closure["abs_error"].mean(),
                "max_relative_abs_error": manu_closure["relative_abs_error"].max(),
                "mean_relative_abs_error": manu_closure["relative_abs_error"].mean(),
            },
        ]
    )
    closure_summary.to_csv(OUT / "closure_summary.csv", index=False, encoding="utf-8-sig")

    annual_share_latex = annual_share.reset_index().copy()
    for col in annual_share_latex.columns:
        if col != "year":
            annual_share_latex[col] = annual_share_latex[col].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "")
    write_latex_tabular(annual_share_latex, TABLES / "stage1_annual_structure_shares.tex", escape=True)

    closure_latex = closure_summary.copy()
    closure_latex["max_abs_error"] = closure_latex["max_abs_error"].map(fmt_num)
    closure_latex["mean_abs_error"] = closure_latex["mean_abs_error"].map(fmt_num)
    closure_latex["max_relative_abs_error"] = closure_latex["max_relative_abs_error"].map(lambda x: f"{x * 100:.6f}%")
    closure_latex["mean_relative_abs_error"] = closure_latex["mean_relative_abs_error"].map(lambda x: f"{x * 100:.6f}%")
    write_latex_tabular(closure_latex, TABLES / "stage1_closure_summary.tex", escape=True)

    summary = {
        "source_csv": str(CSV_PATH),
        "date_start": str(wide.index.min().date()),
        "date_end": str(wide.index.max().date()),
        "province_rows": int(len(province)),
        "date_columns": int(len(date_cols)),
        "long_rows": int(len(long)),
        "industry_names": int(dictionary["行业名称_清洗"].nunique()),
        "manufacturing_expected_industries": int(len(manufacturing_names)),
        "manufacturing_present_industries": int(len(present_manu)),
        "manufacturing_missing_industries": missing_manu,
        "total_closure_mean_relative_abs_error": float(total_closure["relative_abs_error"].mean()),
        "manufacturing_closure_mean_relative_abs_error": float(manu_closure["relative_abs_error"].mean()),
    }
    (OUT / "stage1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
