"""Página principal: cargar imagen -> analizar -> revisar resultados.

Estados (`st.session_state.phase`):
    idle     sin imagen
    invalid  archivo rechazado por la validación
    ready    imagen válida, pendiente de análisis
    running  inferencia en curso (el botón queda deshabilitado)
    done     análisis completado (con o sin detecciones)
    error    fallo de Hugging Face; se puede reintentar
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import streamlit as st

from core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, Settings
from core.detection import AnalysisResult, ImageValidationError, LoadedImage, load_image
from core.hf_client import HFAuthError, HFError, HFQuotaError, ModelStatus
from core.pipeline import active_detection_model, run_analysis
from core.render import display_data_uri, render_annotated_png
from ui import components as ui
from ui.icons import icon

S = st.session_state

TOKEN_HELP_HTML = (
    "Añade tu token de Hugging Face en <code>.streamlit/secrets.toml</code> como "
    "<code>HF_TOKEN = \"hf_…\"</code> o en la variable de entorno <code>HF_TOKEN</code> y recarga la página."
)


def _init_state() -> None:
    defaults = {
        "file_key": None,
        "loaded": None,
        "invalid": None,
        "phase": "idle",
        "result": None,
        "error": None,
        "preview_uri": None,
        "display_uri": None,
        "png": None,
        "export_json": None,
    }
    for key, value in defaults.items():
        S.setdefault(key, value)


def _reset_results() -> None:
    S.result = S.error = S.png = S.export_json = None


def _sync_upload(uploaded) -> None:
    """Valida el archivo una sola vez por subida y reinicia los resultados al cambiar de imagen."""
    if uploaded is None:
        if S.file_key is not None:
            S.file_key = S.loaded = S.invalid = S.preview_uri = S.display_uri = None
            _reset_results()
        S.phase = "idle"
        return

    key = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
    if key == S.file_key:
        return
    S.file_key = key
    _reset_results()
    try:
        loaded = load_image(uploaded.name, uploaded.getvalue())
    except ImageValidationError as exc:
        S.loaded, S.invalid, S.phase = None, exc, "invalid"
        S.preview_uri = S.display_uri = None
        return
    S.loaded, S.invalid, S.phase = loaded, None, "ready"
    S.preview_uri = display_data_uri(loaded.image, max_side=1000)
    S.display_uri = display_data_uri(loaded.image, max_side=1600)


def _start_analysis() -> None:
    # Se ejecuta antes del rerun: el botón ya aparece deshabilitado mientras se infiere,
    # lo que evita ejecuciones duplicadas por doble clic.
    if S.phase in ("ready", "error") and S.loaded is not None:
        _reset_results()
        S.phase = "running"


# ------------------------------------------------------------------ input

def _input_card(settings: Settings) -> None:
    loaded: LoadedImage | None = S.loaded
    with st.container(key="card-input"):
        ui.card_head(
            "1",
            "Imagen de entrada",
            f"{', '.join(e.upper() for e in ALLOWED_EXTENSIONS)} · máximo {MAX_UPLOAD_MB} MB",
            done=loaded is not None,
        )
        with st.container(key="uploader"):
            uploaded = st.file_uploader(
                "Imagen a analizar",
                type=list(ALLOWED_EXTENSIONS),
                key="file",
                label_visibility="collapsed",
                max_upload_size=MAX_UPLOAD_MB,
                disabled=S.phase == "running",
            )
        _sync_upload(uploaded)  # no-op salvo en la primera ejecución de la sesión
        loaded = S.loaded

        if S.phase == "invalid" and S.invalid is not None:
            ui.notice("error", S.invalid.title, S.invalid.message)
        elif loaded is not None:
            ui.image_preview(S.preview_uri, loaded.name)
            ui.file_meta(loaded)

        phase = S.phase
        label, btn_icon = {
            "running": ("Analizando…", "spinner"),
            "done": ("Análisis completado", ":material/check_circle:"),
            "error": ("Reintentar análisis", ":material/refresh:"),
        }.get(phase, ("Analizar imagen", ":material/center_focus_strong:"))
        can_run = phase in ("ready", "error") and loaded is not None and settings.can_analyze
        st.button(
            label,
            key="analyze",
            type="primary",
            icon=btn_icon,
            width="stretch",
            disabled=not can_run,
            on_click=_start_analysis,
            shortcut="Ctrl+Enter",
            help="Envía la imagen al modelo de detección" if can_run else None,
        )

        if not settings.can_analyze:
            ui.notice("warning", "Falta el token de Hugging Face", TOKEN_HELP_HTML, body_is_html=True)
        elif phase in ("idle", "invalid"):
            _hint("info", "Carga una imagen para habilitar el análisis.")
        elif phase == "ready":
            _hint("info", "También puedes pulsar Ctrl + Enter para analizar.")
        elif phase == "done":
            _hint("info", "Para analizar otra imagen, sustitúyela en la zona de carga.")


def _hint(ico: str, text: str) -> None:
    st.html(f'<p class="od-hint">{icon(ico, 14)}<span>{text}</span></p>')


# ------------------------------------------------------------------ results

STEP_LABELS = {
    "prepare": "Preparando inferencia",
    "load": "Cargando modelo en memoria",
    "detect": "Detectando objetos",
    "fallback": "Modelo principal no disponible; usando modelo alternativo",
    "translate": "Traduciendo etiquetas al español",
    "done": "Procesamiento completado",
}


def _run(settings: Settings, statuses: dict[str, ModelStatus]) -> None:
    loaded: LoadedImage = S.loaded
    ui.card_head("2", "Resultado", "Procesando la imagen…")
    ui.image_preview(S.display_uri, loaded.name, processing=True)
    with st.status(STEP_LABELS["prepare"] + "…", expanded=True) as status:

        def on_step(step: str, detail: str) -> None:
            text = STEP_LABELS[step]
            if step == "done":
                status.update(label=f"{text} · {detail}", state="complete", expanded=False)
            else:
                status.update(label=f"{text}…")
            status.write(f"{text} — `{detail}`")

        try:
            result = run_analysis(loaded, settings, statuses, on_step)
        except HFError as exc:
            status.update(label=exc.title, state="error", expanded=False)
            S.error, S.phase = exc, "error"
        except Exception as exc:  # noqa: BLE001 - nunca dejar la UI en estado "running"
            status.update(label="Error inesperado", state="error", expanded=False)
            S.error = HFError("Ocurrió un error inesperado durante el análisis.", repr(exc))
            S.phase = "error"
        else:
            S.result, S.phase = result, "done"
            if result.detections:
                S.png = render_annotated_png(loaded.image, result.detections)
            S.export_json = _export_json(loaded, result)
    st.rerun()


def _export_json(loaded: LoadedImage, result: AnalysisResult) -> str:
    payload = {
        "file": loaded.name,
        "image": {"width": loaded.width, "height": loaded.height, "format": loaded.format},
        "models": {"detection": result.model_used, "translation": result.translation_model},
        "detections": [
            {
                "id": d.id,
                "label": d.label,
                "label_es": d.label_es,
                "score": round(d.score, 6),
                "box": {k: round(v, 2) for k, v in vars(d.box).items()},
            }
            for d in result.detections
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _error_view(settings: Settings) -> None:
    err: HFError = S.error
    ui.card_head("2", "Resultado", "No se pudo completar el análisis")
    if isinstance(err, HFAuthError):
        body = f"{err.user_message} {TOKEN_HELP_HTML}"
        ui.notice("error", err.title, body, detail=err.detail, body_is_html=True)
    elif isinstance(err, HFQuotaError):
        body = (
            f"{escape(err.user_message)} Puedes esperar a que se renueven los créditos mensuales, "
            'comprar créditos en <a href="https://huggingface.co/settings/billing" target="_blank" '
            'rel="noopener noreferrer">Billing</a> o suscribirte a '
            '<a href="https://huggingface.co/pro" target="_blank" rel="noopener noreferrer">PRO</a>.'
        )
        ui.notice("warning", err.title, body, detail=err.detail, body_is_html=True)
        return
    else:
        ui.notice("error", err.title, err.user_message, detail=err.detail)
    _hint("help", "Pulsa «Reintentar análisis» para volver a intentarlo. La imagen no se ha modificado.")


def _done_view(settings: Settings) -> None:
    result: AnalysisResult = S.result
    loaded: LoadedImage = S.loaded
    model_name = ui.friendly(result.model_used)
    noun = "objeto" if result.total == 1 else "objetos"
    ui.card_head(
        "2",
        "Resultado",
        f"{model_name} · etiquetas traducidas con Helsinki-NLP",
        done=True,
        badge=f"{result.total} {noun}",
    )
    if result.fallback_used:
        ui.notice(
            "info",
            "Se usó el modelo alternativo",
            f"{ui.friendly(settings.detection_model)} "
            + ("no se pudo cargar en este servidor" if settings.is_local else "no está disponible en la inferencia de Hugging Face")
            + f", así que la detección se hizo con {model_name} ({result.model_used}).",
        )
    if result.translation_failures:
        n = len(result.translation_failures)
        ui.notice(
            "warning",
            "Traducción incompleta",
            f"No se pudo traducir {n} etiqueta(s); se muestran en inglés: {', '.join(result.translation_failures)}.",
        )

    if not result.detections:
        ui.image_preview(S.display_uri, loaded.name)
        ui.notice(
            "info",
            "No se detectaron objetos",
            "El modelo no encontró objetos reconocibles en esta imagen. Prueba con una imagen más nítida "
            "o en la que los objetos ocupen más espacio.",
        )
        return

    w, h = loaded.width, loaded.height
    ui.detection_viewer(result, S.display_uri, w, h)
    _hint("info", "Pasa el cursor sobre una caja, o sobre un objeto del listado, para resaltarlo.")
    stem = Path(loaded.name).stem
    with st.container(horizontal=True, gap="small", key="downloads"):
        st.download_button(
            "Descargar imagen",
            data=S.png,
            file_name=f"{stem}_detecciones.png",
            mime="image/png",
            icon=":material/download:",
            on_click="ignore",
        )
        st.download_button(
            "Exportar JSON",
            data=S.export_json,
            file_name=f"{stem}_detecciones.json",
            mime="application/json",
            icon=":material/data_object:",
            on_click="ignore",
        )


def _results_card(settings: Settings, statuses: dict[str, ModelStatus]) -> None:
    with st.container(key="card-results"):
        phase = S.phase
        if phase == "running":
            _run(settings, statuses)
        elif phase == "error":
            _error_view(settings)
        elif phase == "done":
            _done_view(settings)
        else:
            ui.card_head("2", "Resultado", "Objetos detectados, etiquetas y confianza")
            model, _ = active_detection_model(settings, statuses)
            ui.empty_results("ready" if phase == "ready" else "waiting", ui.friendly(model))


# ------------------------------------------------------------------ page

def render(settings: Settings, statuses: dict[str, ModelStatus]) -> None:
    _init_state()
    # El valor del uploader ya está en session_state antes de dibujarlo: sincronizamos
    # primero para que cabecera, tarjeta de entrada y resultados reflejen el mismo estado.
    if "file" in S:
        _sync_upload(S["file"])
    ui.header(settings, statuses)

    with st.container(key="workspace"):
        left, right = st.columns([5, 7], gap="large")
        with left:
            _input_card(settings)
        with right:
            _results_card(settings, statuses)

    result: AnalysisResult | None = S.result
    if S.phase == "done" and result and result.detections:
        ui.section_title(
            "Resultados detallados",
            "Ordenados por confianza · pasa el cursor o usa Tab para localizar cada objeto en la imagen",
        )
        ui.detection_summary(result)
        ui.detection_list(result)
        ui.highlight_css(result)

    st.html('<div style="height:8px"></div>')
    with st.expander("Información técnica", icon=":material/info:"):
        ui.tech_info(settings, result.model_used if result else None)
