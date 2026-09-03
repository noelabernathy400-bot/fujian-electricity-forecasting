from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Experiments" / "outputs"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(30, True)
F_LABEL = font(18)
F_SMALL = font(15)
F_TINY = font(13)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt: ImageFont.ImageFont, fill=(0, 0, 0)) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(img: Image.Image, name: str) -> None:
    img.save(FIG / name, dpi=(150, 150))


def draw_barh(name: str, title: str, labels: list[str], values: list[float], color=(37, 99, 235), suffix="%", width=1300, row_h=58) -> None:
    height = 110 + row_h * len(labels)
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    text_center(d, (width / 2, 38), title, F_TITLE)
    left, right, top = 250, 105, 84
    plot_w = width - left - right
    vmax = max(values) if values else 1
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = top + i * row_h
        d.text((left - 18 - d.textlength(lab, font=F_LABEL), y + 10), lab, font=F_LABEL, fill=(0, 0, 0))
        w = max(0, val) / vmax * plot_w
        d.rectangle((left, y + 4, left + w, y + 34), fill=color)
        d.text((left + w + 8, y + 8), f"{val:.2f}{suffix}", font=F_SMALL, fill=(0, 0, 0))
    save(img, name)


def draw_grouped_barh() -> None:
    rows = read_csv("forecast_holiday_segment_panel_comparison.csv")
    rows = sorted(rows, key=lambda r: float(r["segment_mape"]))
    labels = [r["target"] for r in rows]
    base = [float(r["baseline_mape"]) * 100 for r in rows]
    seg = [float(r["segment_mape"]) * 100 for r in rows]
    width, row_h = 1300, 92
    height = 125 + row_h * len(labels)
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    text_center(d, (width / 2, 38), "高误差对象长假分段前后 MAPE", F_TITLE)
    left, top, plot_w = 210, 85, 970
    d.rectangle((left, 58, left + 18, 72), fill=(148, 163, 184))
    d.text((left + 28, 54), "原综合模型", font=F_SMALL, fill=(0, 0, 0))
    d.rectangle((left + 150, 58, left + 168, 72), fill=(37, 99, 235))
    d.text((left + 178, 54), "长假分段模型", font=F_SMALL, fill=(0, 0, 0))
    vmax = max(base + seg)
    for i, lab in enumerate(labels):
        y = top + i * row_h
        d.text((left - 20 - d.textlength(lab, font=F_LABEL), y + 22), lab, font=F_LABEL, fill=(0, 0, 0))
        bw = base[i] / vmax * plot_w
        sw = seg[i] / vmax * plot_w
        d.rectangle((left, y + 3, left + bw, y + 33), fill=(169, 181, 198))
        d.text((left + bw + 8, y + 8), f"{base[i]:.2f}%", font=F_SMALL, fill=(0, 0, 0))
        d.rectangle((left, y + 43, left + sw, y + 73), fill=(37, 99, 235))
        d.text((left + sw + 8, y + 48), f"{seg[i]:.2f}%", font=F_SMALL, fill=(0, 0, 0))
    save(img, "fig02_holiday_segment_panel.png")


def draw_line_horizon() -> None:
    rows = read_csv("forecast_direct_multistep_total_by_horizon.csv")
    models = [
        ("direct_own_calendar", (37, 99, 235), "自身+日历"),
        ("direct_own_weather_holiday", (220, 38, 38), "自身+天气节假日"),
        ("direct_category_weather_holiday", (22, 163, 74), "类别+天气节假日"),
    ]
    series = {}
    for key, _, _ in models:
        g = [r for r in rows if r["model"] == key]
        series[key] = [(int(r["horizon"]), float(r["mape"]) * 100) for r in sorted(g, key=lambda r: int(r["horizon"]))]
    width, height = 1400, 650
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    text_center(d, (width / 2, 38), "全省合计直接多步预测：不同 horizon MAPE", F_TITLE)
    left, right, top, bottom = 95, 55, 80, 90
    pw, ph = width - left - right, height - top - bottom
    all_vals = [v for pts in series.values() for _, v in pts]
    ymin, ymax = min(all_vals) - 0.3, max(all_vals) + 0.4
    d.rectangle((left, top, left + pw, top + ph), outline=(203, 213, 225), width=2)
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = top + ph * frac
        d.line((left, y, left + pw, y), fill=(226, 232, 240), width=1)
        val = ymax - frac * (ymax - ymin)
        d.text((20, y - 10), f"{val:.1f}%", font=F_SMALL, fill=(0, 0, 0))
    def xy(h, val):
        x = left + (h - 1) / 29 * pw
        y = top + (ymax - val) / (ymax - ymin) * ph
        return x, y
    for key, color, label in models:
        pts = [xy(h, v) for h, v in series[key]]
        d.line(pts, fill=color, width=4)
        for x, y in pts[::5]:
            d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    lx = left + 20
    for i, (_, color, label) in enumerate(models):
        y = height - 55 + i * 22
        d.line((lx, y, lx + 40, y), fill=color, width=4)
        d.text((lx + 50, y - 10), label, font=F_SMALL, fill=(0, 0, 0))
    d.text((width / 2 - 70, height - 28), "预测步长 horizon", font=F_SMALL, fill=(0, 0, 0))
    save(img, "fig01_direct_multistep_horizon.png")


def draw_city_growth() -> None:
    rows = [r for r in read_csv("structure_city_growth_contribution.csv") if r["year"] == "2025"]
    rows = sorted(rows, key=lambda r: float(r["contribution_to_total_growth"]), reverse=True)
    labels = [r["city"] for r in rows]
    values = [float(r["contribution_to_total_growth"]) * 100 for r in rows]
    draw_barh("fig03_city_growth_2025.png", "2025 年地市对全省用电增长贡献", labels, values, color=(37, 99, 235))


def draw_volatility() -> None:
    keep = {"合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电", "非普工业", "电子", "纺织业", "非金属矿物制品业", "黑色金属冶炼"}
    rows = [r for r in read_csv("structure_volatility_profile.csv") if r["variable"] in keep]
    rows = sorted(rows, key=lambda r: float(r["cv"]), reverse=True)
    labels = [r["variable"] for r in rows]
    values = [float(r["cv"]) * 100 for r in rows]
    draw_barh("fig05_volatility_cv.png", "重点变量日度波动强度 CV", labels, values, color=(15, 118, 110), width=1300, row_h=52)


def draw_granger() -> None:
    rows = sorted(read_csv("granger_network_node_scores.csv"), key=lambda r: float(r["out_strength"]), reverse=True)[:12]
    labels = [r["node"] for r in rows]
    values = [float(r["out_strength"]) for r in rows]
    draw_barh("fig06_granger_out_strength.png", "Granger 网络节点输出强度", labels, values, color=(124, 58, 237), suffix="", width=1300, row_h=50)


def draw_heatmap() -> None:
    rows = read_csv("structure_monthly_seasonality_index.csv")
    variables = ["合计", "工业", "制造业", "大工业电量", "居民生活", "商业用电", "非普工业", "电子"]
    data = {(r["variable"], int(r["month"])): float(r["seasonality_index"]) for r in rows}
    width, height = 1320, 520
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    text_center(d, (width / 2, 38), "重点变量月度季节性指数", F_TITLE)
    left, top, cw, ch = 180, 95, 86, 42
    for m in range(1, 13):
        text_center(d, (left + (m - 1) * cw + cw / 2, top - 22), f"{m}月", F_SMALL)
    for i, var in enumerate(variables):
        y = top + i * ch
        d.text((left - 16 - d.textlength(var, font=F_LABEL), y + 9), var, font=F_LABEL, fill=(0, 0, 0))
        for m in range(1, 13):
            val = data[(var, m)]
            centered = max(-0.3, min(0.3, val - 1.0))
            if centered >= 0:
                t = centered / 0.3
                color = (int(225 - 130 * t), int(235 - 135 * t), 255)
            else:
                t = -centered / 0.3
                color = (255, int(235 - 125 * t), int(225 - 125 * t))
            x = left + (m - 1) * cw
            d.rectangle((x, y, x + cw - 3, y + ch - 3), fill=color, outline=(255, 255, 255))
            text_center(d, (x + cw / 2, y + ch / 2), f"{val:.2f}", F_TINY)
    d.text((left, height - 45), "指数大于 1 表示该月日均用电高于全年平均；小于 1 表示低于全年平均。", font=F_SMALL, fill=(0, 0, 0))
    save(img, "fig04_seasonality_heatmap.png")


def main() -> None:
    draw_line_horizon()
    draw_grouped_barh()
    draw_city_growth()
    draw_heatmap()
    draw_volatility()
    draw_granger()


if __name__ == "__main__":
    main()
