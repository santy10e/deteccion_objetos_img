"""Componentes visuales reutilizables (HTML + Streamlit). Sin lógica de inferencia."""

from __future__ import annotations

from collections import Counter
from html import escape

import streamlit as st

from core.config import APP_NAME, APP_TAGLINE, APP_VERSION, Settings
from core.detection import AnalysisResult, LoadedImage, human_size
from core.hf_client import ModelStatus
from core.pipeline import active_detection_model
from core.render import class_colors, format_pct, place_tags, text_color_on
from ui.icons import icon

FRIENDLY_NAMES = {
    "SenseTime/deformable-detr": "Deformable DETR",
    "facebook/detr-resnet-50": "DETR ResNet-50",
    "Helsinki-NLP/opus-mt-en-es": "Helsinki-NLP opus-mt-en-es",
}


def friendly(model_id: str) -> str:
    return FRIENDLY_NAMES.get(model_id, model_id.split("/")[-1])


def pct(score: float, decimals: int = 1) -> str:
    """Porcentaje con formato español y truncado: 0.9988 -> '99,8 %'."""
    return format_pct(score, decimals)


def _html(markup: str) -> None:
    st.html(markup)


# ---------------------------------------------------------------- status

STATE_TEXT = {
    "live": "Operativo",
    "local": "Local",
    "fallback": "Modo alternativo",
    "unavailable": "No disponible",
    "unknown": "Sin verificar",
    "error": "Sin token",
}


def detection_state(settings: Settings, statuses: dict[str, ModelStatus]) -> str:
    primary = statuses.get(settings.detection_model, "unknown")
    if primary != "unavailable":
        return primary
    fallback = settings.detection_fallback_model
    if fallback and statuses.get(fallback) != "unavailable":
        return "fallback"
    return "unavailable"


def state_badge(state: str, text: str | None = None) -> str:
    label = escape(text or STATE_TEXT[state])
    return f'<span class="od-state is-{state}"><span class="od-dot is-{state}"></span>{label}</span>'


# ---------------------------------------------------------------- sidebar

def sidebar(settings: Settings, statuses: dict[str, ModelStatus], pages: list) -> None:
    det_state = detection_state(settings, statuses)
    det_model, _ = active_detection_model(settings, statuses)
    tr_state = statuses.get(settings.translation_model, "unknown")
    with st.sidebar:
        _html(
            f'<div class="od-brand"><div class="od-brand__mark">{icon("scan-eye", 20)}</div>'
            f'<div><div class="od-brand__name">{APP_NAME}</div>'
            f'<div class="od-brand__sub">Computer Vision Studio</div></div></div>'
            '<div class="od-side-label" id="nav-label">Navegación</div>'
        )
        for page in pages:
            st.page_link(page, width="stretch")
        _html(
            '<div class="od-side-label" style="margin-top:20px">Modelos</div>'
            '<div class="od-side-models">'
            f'<div class="od-side-model"><div class="od-side-model__role">Detección {state_badge(det_state)}</div>'
            f'<div class="od-side-model__name">{escape(det_model)}</div></div>'
            f'<div class="od-side-model"><div class="od-side-model__role">Traducción {state_badge(tr_state)}</div>'
            f'<div class="od-side-model__name">{escape(settings.translation_model)}</div></div>'
            '</div>'
            f'<div class="od-side-foot"><span>Versión {APP_VERSION}</span>'
            f'<span>Streamlit {st.__version__} · {escape(settings.platform_label)}</span></div>'
        )


# ---------------------------------------------------------------- header

def header(settings: Settings, statuses: dict[str, ModelStatus], title: str = APP_NAME, subtitle: str = APP_TAGLINE) -> None:
    det_state = detection_state(settings, statuses)
    tr_state = statuses.get(settings.translation_model, "unknown")
    pills = [
        f'<span class="od-pill"><span class="od-pill__label">Detección</span>{state_badge(det_state)}</span>',
        f'<span class="od-pill"><span class="od-pill__label">Traducción</span>{state_badge(tr_state)}</span>',
    ]
    if not settings.is_local and not settings.has_token:
        pills.append(f'<span class="od-pill"><span class="od-pill__label">Token HF</span>{state_badge("error", "No configurado")}</span>')
    _html(
        '<header class="od-header">'
        f'<div><h1>{escape(title)}</h1><p class="od-header__sub">{escape(subtitle)}</p>'
        '<div class="od-header__tech">'
        f'<span class="od-chip">{icon("cpu", 13)}{escape(friendly(settings.detection_model))}</span>'
        f'<span class="od-chip">{icon("languages", 13)}Helsinki-NLP · EN → ES</span>'
        f'<span class="od-chip">{icon("server", 13)}{escape(settings.platform_label)}</span>'
        '</div></div>'
        f'<div class="od-header__status" aria-label="Estado de los modelos">{"".join(pills)}</div>'
        '</header>'
    )


# ---------------------------------------------------------------- cards

def card_head(step: str, title: str, subtitle: str, *, done: bool = False, badge: str | None = None) -> None:
    step_html = icon("tick", 14) if done else escape(step)
    badge_html = f'<span class="od-badge">{badge}</span>' if badge else ""
    _html(
        f'<div class="od-card-head"><span class="od-step{" is-done" if done else ""}" aria-hidden="true">{step_html}</span>'
        f'<div class="od-card-head__text"><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div>{badge_html}</div>'
    )


def notice(kind: str, title: str, body: str, *, detail: str | None = None, body_is_html: bool = False) -> None:
    icon_name = {"error": "error", "warning": "alert", "info": "info", "success": "check"}[kind]
    detail_html = (
        f"<details><summary>Detalle técnico</summary><code>{escape(detail)}</code></details>" if detail else ""
    )
    role = "alert" if kind == "error" else "status"
    _html(
        f'<div class="od-notice is-{kind} od-appear" role="{role}">{icon(icon_name, 18)}'
        f'<div><div class="od-notice__title">{escape(title)}</div>'
        f'<div class="od-notice__body">{body if body_is_html else escape(body)}</div>{detail_html}</div></div>'
    )


def flow_steps(current: int) -> str:
    labels = ["Cargar imagen", "Analizar", "Revisar resultados"]
    items = []
    for i, label in enumerate(labels, start=1):
        cls = "is-done" if i < current else "is-current" if i == current else ""
        mark = "✓" if i < current else str(i)
        sep = '<li class="od-flow__sep" aria-hidden="true">›</li>' if i < len(labels) else ""
        aria = ' aria-current="step"' if i == current else ""
        items.append(f'<li class="{cls}"{aria}><b aria-hidden="true">{mark}</b>{label}</li>{sep}')
    return f'<ol class="od-flow" aria-label="Progreso">{"".join(items)}</ol>'


def empty_results(state: str, model_name: str = "") -> None:
    if state == "ready":
        _html(
            '<div class="od-empty is-ready od-appear" role="status">'
            f'<div class="od-empty__icon">{icon("scan-eye", 24)}</div>'
            '<h3>Imagen lista para analizar</h3>'
            f'<p>Pulsa <strong>Analizar imagen</strong> para detectar los objetos con {escape(model_name)} '
            'y traducir sus etiquetas al español.</p>'
            f'{flow_steps(2)}</div>'
        )
    else:
        _html(
            '<div class="od-empty">'
            f'<div class="od-empty__icon">{icon("image", 24)}</div>'
            '<h3>Los resultados aparecerán aquí</h3>'
            '<p>Carga una imagen para identificar los objetos que contiene, con su etiqueta en español '
            'y el nivel de confianza del modelo.</p>'
            f'{flow_steps(1)}</div>'
        )


# ---------------------------------------------------------------- input

def image_preview(data_uri: str, name: str, *, processing: bool = False) -> None:
    cls = "od-preview od-processing" if processing else "od-preview od-appear"
    _html(f'<div class="{cls}"><img src="{data_uri}" alt="Vista previa de la imagen cargada: {escape(name)}"></div>')


def file_meta(img: LoadedImage) -> None:
    def item(ico: str, key: str, value: str, extra: str = "") -> str:
        return (
            f'<div class="od-meta__item {extra}"><div class="od-meta__k">{icon(ico, 12)}{key}</div>'
            f'<div class="od-meta__v" title="{escape(value)}">{escape(value)}</div></div>'
        )

    _html(
        '<div class="od-meta">'
        + item("file", "Archivo", img.name, "is-name")
        + item("ruler", "Dimensiones", f"{img.width} × {img.height} px")
        + item("drive", "Tamaño", human_size(img.size_bytes))
        + item("image", "Formato", img.format)
        + "</div>"
    )


# ---------------------------------------------------------------- results

# El visor es responsive: estimamos el tamaño de cada etiqueta para un ancho de
# referencia (~ancho real de la columna en escritorio) y colocamos en consecuencia.
_REF_WIDTH_PX = 720
_TAG_HEIGHT_PX = 21


def _tag_sizes(result: AnalysisResult, w: int, compact: bool) -> list[tuple[float, float]]:
    scale = w / _REF_WIDTH_PX  # píxeles de imagen por píxel de pantalla
    sizes = []
    for det in result.detections:
        text = f"#{det.id}" if compact else f"#{det.id} {det.label_es} · {pct(det.score, 0)}"
        sizes.append(((len(text) * 6.9 + 16) * scale, _TAG_HEIGHT_PX * scale))
    return sizes


def _box_style(det, w: int, h: int) -> str:
    b = det.box
    return (
        f"left:{b.xmin / w * 100:.3f}%;top:{b.ymin / h * 100:.3f}%;"
        f"width:{b.width / w * 100:.3f}%;height:{b.height / h * 100:.3f}%"
    )


def detection_viewer(result: AnalysisResult, data_uri: str, w: int, h: int) -> None:
    colors = class_colors(result.detections)
    compact = result.total > 12
    # Las cajas pequeñas se pintan al final para que queden encima y sean seleccionables.
    ordered = sorted(result.detections, key=lambda d: d.box.width * d.box.height, reverse=True)
    placements = place_tags(result.detections, _tag_sizes(result, w, compact), w, h)
    boxes = []
    for det in ordered:
        style = _box_style(det, w, h)
        tag = placements[det.id]
        extra = f"pos-{tag.position}" + (" is-right" if tag.right else "")
        color = colors[det.label]
        boxes.append(
            f'<div class="od-box {extra}" id="det-box-{det.id}" '
            f'style="{style};--c:{color};--t:{text_color_on(color)}" title="#{det.id} {escape(det.label_es)} · {pct(det.score)}">'
            f'<span class="od-tag">#{det.id}<span class="od-tag__full"> {escape(det.label_es)} · {pct(det.score, 0)}</span></span></div>'
        )
    summary = ", ".join(f"{d.label_es} ({pct(d.score)})" for d in result.detections[:15])
    more = f" y {result.total - 15} más" if result.total > 15 else ""
    alt = f"Imagen analizada con {result.total} objeto(s) detectado(s): {summary}{more}."
    _html(
        '<figure class="od-viewer od-appear">'
        f'<div class="od-stage{" is-compact" if compact else ""}">'
        f'<div class="od-stage__inner" style="aspect-ratio:{w}/{h};width:min(100%, calc(72vh * {w} / {h}))">'
        f'<img src="{data_uri}" alt="{escape(alt)}">'
        f'<div aria-hidden="true">{"".join(boxes)}</div>'
        '</div></div></figure>'
    )


def highlight_css(result: AnalysisResult) -> None:
    """Vincula cada fila del listado con su caja en la imagen (hover y foco de teclado)."""
    if not result.detections:
        return
    rules = [
        "body:has(.od-det-row:hover, .od-det-row:focus-visible) .od-box{opacity:.25}",
    ]
    for det in result.detections:
        i = det.id
        rules.append(
            f"body:has(#det-row-{i}:hover, #det-row-{i}:focus-visible) #det-box-{i}"
            "{opacity:1;z-index:20;border-width:3px;box-shadow:0 0 0 1px rgba(0,0,0,.5),0 0 0 9999px rgba(10,11,13,.4)}"
            f"body:has(#det-row-{i}:hover, #det-row-{i}:focus-visible) #det-box-{i} .od-tag__full{{display:inline}}"
            f"body:has(#det-box-{i}:hover) #det-row-{i}"
            "{border-color:var(--c);background:#FBFBFD;box-shadow:0 0 0 1px var(--c)}"
        )
    _html(f"<style>{''.join(rules)}</style>")


def detection_summary(result: AnalysisResult) -> None:
    colors = class_colors(result.detections)
    counts = Counter(d.label for d in result.detections)
    es_names = {d.label: d.label_es for d in result.detections}
    mean = result.mean_score
    classes = "".join(
        f'<span class="od-class" style="--c:{colors[label]}"><i aria-hidden="true"></i>'
        f'{escape(es_names[label])} <b>×{n}</b></span>'
        for label, n in counts.most_common()
    )
    _html(
        '<div class="od-appear">'
        '<div class="od-stats">'
        f'<div class="od-stat"><div class="od-stat__k">{icon("scan-eye", 15)}Objetos detectados</div>'
        f'<div class="od-stat__v">{result.total}</div><div class="od-stat__d">Detecciones devueltas por el modelo</div></div>'
        f'<div class="od-stat"><div class="od-stat__k">{icon("layers", 15)}Clases distintas</div>'
        f'<div class="od-stat__v">{result.class_count}</div><div class="od-stat__d">Categorías diferentes en la imagen</div></div>'
        f'<div class="od-stat"><div class="od-stat__k">{icon("gauge", 15)}Confianza media</div>'
        f'<div class="od-stat__v">{pct(mean) if mean is not None else "—"}</div>'
        '<div class="od-stat__d">Promedio de las puntuaciones del modelo</div></div>'
        '</div>'
        f'<div class="od-classes" aria-label="Clases detectadas">{classes}</div>'
        '</div>'
    )


def detection_list(result: AnalysisResult) -> None:
    colors = class_colors(result.detections)
    rows = []
    for idx, det in enumerate(result.detections):
        color = colors[det.label]
        b = det.box
        aria = f"Objeto {det.id}: {det.label_es}, en inglés {det.label}, confianza {pct(det.score)}"
        rows.append(
            f'<li class="od-det-row" id="det-row-{det.id}" tabindex="0" aria-label="{escape(aria)}" '
            f'style="--c:{color};--t:{text_color_on(color)};--i:{min(idx, 20)}">'
            f'<span class="od-det-id" aria-hidden="true">#{det.id}</span>'
            '<div style="min-width:0" aria-hidden="true">'
            f'<div class="od-det-es">{escape(det.label_es)}</div>'
            f'<div class="od-det-en"><span class="od-lang">EN</span>{escape(det.label)}</div>'
            f'<div class="od-det-box">x {b.xmin:.0f} · y {b.ymin:.0f} · {b.width:.0f}×{b.height:.0f} px</div></div>'
            '<div class="od-conf" aria-hidden="true">'
            f'<div class="od-conf__v">{pct(det.score)}</div>'
            f'<div class="od-conf__bar"><span style="width:{det.score * 100:.1f}%"></span></div></div>'
            '</li>'
        )
    _html(f'<ol class="od-det-list" aria-label="Objetos detectados, ordenados por confianza">{"".join(rows)}</ol>')


def section_title(title: str, subtitle: str = "") -> None:
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    _html(f'<div class="od-section-title"><h2>{escape(title)}</h2>{sub}</div>')


# ---------------------------------------------------------------- tech info

def tech_info(settings: Settings, model_used: str | None = None) -> None:
    det_model = model_used or settings.detection_model

    def item(ico: str, key: str, value: str) -> str:
        return f'<div class="od-tech__item"><div class="od-tech__k">{icon(ico, 13)}{key}</div><div class="od-tech__v">{value}</div></div>'

    def link(model_id: str) -> str:
        return (
            f'<a href="https://huggingface.co/{escape(model_id)}" target="_blank" rel="noopener noreferrer">'
            f'{escape(friendly(model_id))}</a> <code>{escape(model_id)}</code>'
        )

    _html(
        '<div class="od-tech">'
        + item("cpu", "Modelo de detección", link(det_model))
        + item("languages", "Modelo de traducción", link(settings.translation_model))
        + item("server", "Ejecución", escape(settings.platform_label))
        + item("tag", "Idioma de las etiquetas", "Inglés (original)")
        + item("languages", "Idioma de salida", "Español")
        + "</div>"
    )
