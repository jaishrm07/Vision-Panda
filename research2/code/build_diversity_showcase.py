"""Build a 1920x1080 'dataset diversity showcase' image for the deck.

Image-forward layout: minimal text (just column headers and one-word axis
labels), maximum image size. The slide HTML already carries the title and
explanation, so the image itself doesn't repeat them.

Picks the best camera per axis:
  - Color: EEF (wrist) camera — cube fills the frame
  - Spatial / Camera / Lighting: external camera

Output: results/slide_charts/dataset_diversity.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREVIEWS = ROOT / "results" / "previews_128px_v1"
OUT = ROOT / "results" / "slide_charts" / "dataset_diversity.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

CANVAS_W, CANVAS_H = 1920, 1080
BG = (250, 248, 243)
INK = (21, 24, 29)
INK2 = (42, 47, 58)
MUTED = (107, 111, 122)
LINE = (231, 227, 216)
LINE_DARK = (200, 196, 184)
ACCENT = (21, 107, 117)
NARROW_FRAME = (190, 186, 175)
DIVERSE_FRAME = ACCENT

# (axis_label, narrow_cfg, diverse_cfg, camera)
ROWS = [
    ("Color",                "color_red_only",   "color_multi",       "eef"),
    ("Spatial",              "spatial_narrow",   "spatial_wide",      "external"),
    ("Camera",               "camera_fixed",     "camera_multi_pose", "external"),
    ("Lighting",             "lighting_fixed",   "lighting_diverse",  "external"),
]

# ---------- Layout ----------
PAD_X = 18
PAD_TOP = 12
HEADER_H = 36
LEFT_LABEL_W = 150
GROUP_GAP = 22

usable_h = CANVAS_H - PAD_TOP - HEADER_H - PAD_TOP
ROW_H = usable_h // 4

usable_w = CANVAS_W - 2 * PAD_X - LEFT_LABEL_W - GROUP_GAP
COL_W = usable_w // 4

IMG_SIDE = min(COL_W - 8, ROW_H - 6)


def font(size, bold=False):
    candidates_bold = [
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    candidates_reg = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in (candidates_bold if bold else candidates_reg):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def crop_workspace(img: Image.Image, camera: str) -> Image.Image:
    """Crop preview to focus on the workspace (drop empty space at the top)."""
    w, h = img.size
    if camera == "external":
        top = int(h * 0.18)
        return img.crop((0, top, w, h))
    else:
        side = int(min(w, h) * 0.92)
        x = (w - side) // 2
        y = (h - side) // 2
        return img.crop((x, y, x + side, y + side))


def load_cell(cfg: str, sample: int, camera: str) -> Image.Image:
    path = PREVIEWS / f"preview__{cfg}__seed000__sample{sample:03d}__{camera}.png"
    img = Image.open(path).convert("RGB")
    img = crop_workspace(img, camera)
    img = img.resize((IMG_SIDE, IMG_SIDE), Image.Resampling.LANCZOS)
    return img


def main():
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)

    f_header = font(20, bold=True)
    f_axis = font(28, bold=True)

    # ========= Column header bar =========
    narrow_band_x = PAD_X + LEFT_LABEL_W
    narrow_band_w = 2 * COL_W
    diverse_band_x = narrow_band_x + narrow_band_w + GROUP_GAP
    diverse_band_w = 2 * COL_W

    header_y = PAD_TOP

    draw.rectangle([narrow_band_x, header_y, narrow_band_x + narrow_band_w, header_y + HEADER_H - 6],
                   fill=(241, 238, 230))
    draw.rectangle([diverse_band_x, header_y, diverse_band_x + diverse_band_w, header_y + HEADER_H - 6],
                   fill=(229, 241, 240))

    n_text = "NARROW TRAIN"
    d_text = "DIVERSE TRAIN"
    nw = draw.textbbox((0, 0), n_text, font=f_header)[2]
    dw = draw.textbbox((0, 0), d_text, font=f_header)[2]
    draw.text((narrow_band_x + (narrow_band_w - nw) // 2, header_y + 4), n_text,
              font=f_header, fill=INK2)
    draw.text((diverse_band_x + (diverse_band_w - dw) // 2, header_y + 4), d_text,
              font=f_header, fill=ACCENT)

    # ========= Rows =========
    grid_top = PAD_TOP + HEADER_H

    for ri, (axis_name, narrow_cfg, diverse_cfg, camera) in enumerate(ROWS):
        y0 = grid_top + ri * ROW_H

        if ri % 2 == 1:
            draw.rectangle([PAD_X, y0, CANVAS_W - PAD_X, y0 + ROW_H - 4],
                           fill=(245, 242, 234))

        # Axis label (centered vertically in row, aligned left)
        ax_bbox = draw.textbbox((0, 0), axis_name, font=f_axis)
        ax_h = ax_bbox[3] - ax_bbox[1]
        ax_y = y0 + (ROW_H - ax_h) // 2 - 4
        draw.text((PAD_X + 8, ax_y), axis_name, font=f_axis, fill=INK)

        # Image cells
        groups = [
            (narrow_cfg,  0, 0, NARROW_FRAME),
            (narrow_cfg,  1, 1, NARROW_FRAME),
            (diverse_cfg, 0, 2, DIVERSE_FRAME),
            (diverse_cfg, 1, 3, DIVERSE_FRAME),
        ]
        for cfg, sample_idx, col_idx, frame_color in groups:
            if col_idx < 2:
                cx = narrow_band_x + col_idx * COL_W
            else:
                cx = diverse_band_x + (col_idx - 2) * COL_W
            x = cx + (COL_W - IMG_SIDE) // 2
            y = y0 + (ROW_H - IMG_SIDE) // 2 - 2

            draw.rectangle([x - 3, y - 3, x + IMG_SIDE + 3, y + IMG_SIDE + 3],
                           outline=frame_color, width=3)
            try:
                img = load_cell(cfg, sample_idx, camera)
                canvas.paste(img, (x, y))
            except FileNotFoundError:
                draw.rectangle([x, y, x + IMG_SIDE, y + IMG_SIDE], fill=(245, 245, 245))

        # Vertical separator between Narrow and Diverse groups
        sep_x = narrow_band_x + narrow_band_w + GROUP_GAP // 2
        draw.line([(sep_x, y0 + 6), (sep_x, y0 + ROW_H - 10)], fill=LINE_DARK, width=1)

    canvas.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)  ·  image cells: {IMG_SIDE}×{IMG_SIDE}")


if __name__ == "__main__":
    main()
