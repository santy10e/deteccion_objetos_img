"""Visualización de detecciones: colores por clase, PNG anotado e imagen de vista previa."""

from __future__ import annotations

import base64
import io
import math
from functools import lru_cache
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

from core.detection import Detection

# Colores saturados y distinguibles entre sí; cada caja lleva además un contorno
# oscuro para mantener el contraste sobre fondos claros y oscuros.
PALETTE: tuple[str, ...] = (
    "#635BFF",  # índigo (acento)
    "#0EA5E9",  # azul cielo
    "#16A36A",  # verde
    "#F59E0B",  # ámbar
    "#E11D48",  # rosa intenso
    "#14B8A6",  # turquesa
    "#C084FC",  # violeta claro
    "#F97316",  # naranja
    "#84CC16",  # lima
    "#EC4899",  # rosa
)
INK = "#17191C"
WHITE = "#FFFFFF"


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@lru_cache(maxsize=64)
def text_color_on(bg: str) -> str:
    """Blanco o tinta, el que ofrezca más contraste (WCAG) sobre `bg`."""
    return WHITE if _contrast(bg, WHITE) >= _contrast(bg, INK) else INK


def class_colors(detections: tuple[Detection, ...]) -> dict[str, str]:
    """Asigna un color por clase en orden de aparición (estable para un mismo resultado)."""
    colors: dict[str, str] = {}
    for det in detections:
        if det.label not in colors:
            colors[det.label] = PALETTE[len(colors) % len(PALETTE)]
    return colors


def format_pct(score: float, decimals: int = 1) -> str:
    """0.9988 -> '99,8 %'. Trunca en lugar de redondear para no mostrar nunca un 100 % falso."""
    factor = 10**decimals
    value = math.floor(score * 100 * factor + 1e-9) / factor
    return f"{value:.{decimals}f}".replace(".", ",") + " %"


def tag_text(det: Detection) -> str:
    return f"#{det.id} {det.label_es} · {format_pct(det.score, 0)}"


class TagPlacement(NamedTuple):
    position: str  # above | inside | below | inside-bottom
    right: bool  # alineada al borde derecho de la caja
    x: float
    y: float


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_tags(
    detections: tuple[Detection, ...], sizes: list[tuple[float, float]], width: float, height: float
) -> dict[int, TagPlacement]:
    """Coloca cada etiqueta evitando solapes con las ya colocadas (prioridad: mayor confianza).

    `sizes` son (ancho, alto) de cada etiqueta en píxeles de la imagen original.
    """
    placed: list[tuple[float, float, float, float]] = []
    result: dict[int, TagPlacement] = {}
    for det, (tw, th) in zip(detections, sizes):
        b = det.box
        right = b.xmin + tw > width
        x = max(0.0, b.xmax - tw) if right else b.xmin
        candidates = [
            ("above", b.ymin - th),
            ("inside", b.ymin),
            ("below", b.ymax),
            ("inside-bottom", b.ymax - th),
        ]
        valid = [(name, y) for name, y in candidates if y >= 0 and y + th <= height] or [("inside", b.ymin)]
        name, y = next(
            ((n, y) for n, y in valid if not any(_overlaps((x, y, x + tw, y + th), p) for p in placed)),
            valid[0],
        )
        placed.append((x, y, x + tw, y + th))
        result[det.id] = TagPlacement(name, right, x, y)
    return result


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    return (*(int(hex_color[i : i + 2], 16) for i in (1, 3, 5)), alpha)


def render_annotated_png(image: Image.Image, detections: tuple[Detection, ...]) -> bytes:
    """Dibuja las detecciones sobre la imagen a resolución original (para descargar)."""
    base = image.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    short_side = min(w, h)
    line = max(2, round(short_side / 280))
    font = ImageFont.load_default(size=max(12, round(short_side / 42)))
    pad = max(3, line * 2)
    colors = class_colors(detections)

    # Las cajas grandes primero para que las pequeñas queden visibles encima.
    ordered = sorted(detections, key=lambda d: d.box.width * d.box.height, reverse=True)
    for det in ordered:
        color = colors[det.label]
        x0, y0, x1, y1 = det.box.xmin, det.box.ymin, det.box.xmax, det.box.ymax
        draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), outline=(0, 0, 0, 110), width=line + 2)
        draw.rectangle((x0, y0, x1, y1), outline=_hex_to_rgba(color), width=line)

    texts = {det.id: tag_text(det) for det in detections}
    bboxes = {i: draw.textbbox((0, 0), t, font=font) for i, t in texts.items()}
    sizes = [(bboxes[d.id][2] - bboxes[d.id][0] + pad * 2, bboxes[d.id][3] - bboxes[d.id][1] + pad * 2) for d in detections]
    placements = place_tags(detections, sizes, w, h)
    for det, (bw, bh) in zip(detections, sizes):
        color = colors[det.label]
        left, top = bboxes[det.id][0], bboxes[det.id][1]
        tx, ty = placements[det.id].x, placements[det.id].y
        draw.rounded_rectangle((tx, ty, tx + bw, ty + bh), radius=pad, fill=_hex_to_rgba(color))
        draw.text((tx + pad - left, ty + pad - top), texts[det.id], font=font, fill=_hex_to_rgba(text_color_on(color)))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def display_data_uri(image: Image.Image, max_side: int = 1600) -> str:
    """Copia reducida en JPEG para el visor web (las cajas se posicionan en %)."""
    preview = image.convert("RGB")
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
