"""Página secundaria: arquitectura del pipeline, estado de los modelos y configuración."""

from __future__ import annotations

from html import escape

import streamlit as st

from core.config import Settings
from core.hf_client import ModelStatus
from core.pipeline import active_detection_model
from ui import components as ui
from ui.icons import icon


def _model_card(ico: str, role: str, model_id: str, state: str, description: str, rows: list[tuple[str, str]]) -> None:
    kv = "".join(f"<dt>{escape(k)}</dt><dd>{v}</dd>" for k, v in rows)
    st.html(
        '<article class="od-model-card od-appear">'
        f'<div class="od-model-card__head"><div class="od-model-card__icon">{icon(ico, 18)}</div>{ui.state_badge(state)}</div>'
        f'<div><div class="od-side-label" style="padding:0;margin:0">{escape(role)}</div>'
        f'<h3 style="margin-top:4px">{escape(ui.friendly(model_id))}</h3>'
        f'<p style="margin-top:4px">{escape(description)}</p></div>'
        f'<dl class="od-kv">{kv}</dl>'
        f'<a class="od-chip" style="width:max-content;text-decoration:none" href="https://huggingface.co/{escape(model_id)}" '
        f'target="_blank" rel="noopener noreferrer">{icon("external", 12)}Ver en Hugging Face</a>'
        '</article>'
    )


def render(settings: Settings, statuses: dict[str, ModelStatus]) -> None:
    ui.header(settings, statuses, title="Modelos", subtitle="Arquitectura del pipeline y estado de los servicios")

    ui.section_title("Pipeline de inferencia")
    det_model, fallback_used = active_detection_model(settings, statuses)
    steps = [
        ("Entrada", "Imagen JPG · PNG · WEBP"),
        ("Detección", ui.friendly(det_model)),
        ("Salida del modelo", "Cajas, etiquetas EN y confianza"),
        ("Traducción", "Helsinki-NLP EN → ES"),
        ("Resultado", "Etiquetas en español"),
    ]
    arrow = f'<div class="od-pipeline__arrow" aria-hidden="true">{icon("arrow-right", 16)}</div>'
    st.html(
        '<div class="od-pipeline" role="list">'
        + arrow.join(
            f'<div class="od-pipeline__step" role="listitem"><small>{escape(k)}</small><div>{escape(v)}</div></div>'
            for k, v in steps
        )
        + "</div>"
    )

    ui.section_title(
        "Modelos configurados",
        "Se ejecutan en este servidor con transformers"
        if settings.is_local
        else "Estado consultado en el Hub de Hugging Face (se actualiza cada 10 min)",
    )
    if fallback_used:
        ui.notice(
            "info",
            "Modelo de detección alternativo activo",
            f"{ui.friendly(settings.detection_model)} ({settings.detection_model}) no tiene un proveedor de "
            f"inferencia activo en Hugging Face. Las detecciones se realizan con {ui.friendly(det_model)} "
            f"({det_model}), de la misma familia DETR y con el mismo formato de salida. Puedes cambiar el "
            "modelo con DETECTION_MODEL y DETECTION_FALLBACK_MODEL.",
        )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _model_card(
            "cpu",
            "Detección de objetos",
            settings.detection_model,
            ui.detection_state(settings, statuses),
            "Transformer de detección con atención deformable: localiza objetos y devuelve su caja, "
            "etiqueta (en inglés) y puntuación de confianza.",
            [
                ("Modelo", f"<code>{escape(settings.detection_model)}</code>"),
                ("Alternativo", f"<code>{escape(settings.detection_fallback_model or '—')}</code>"),
                ("Tarea", "object-detection"),
                ("Salida", "box · label · score"),
            ],
        )
    with c2:
        _model_card(
            "languages",
            "Traducción de etiquetas",
            settings.translation_model,
            statuses.get(settings.translation_model, "unknown"),
            "Modelo MarianMT de Helsinki-NLP que traduce cada etiqueta distinta del inglés al español. "
            "Las traducciones se guardan en caché para no repetir peticiones.",
            [
                ("Modelo", f"<code>{escape(settings.translation_model)}</code>"),
                ("Tarea", "translation"),
                ("Idiomas", "Inglés → Español"),
                ("Si falla", "Se conserva la etiqueta en inglés"),
            ],
        )

    ui.section_title("Configuración")
    if settings.is_local:
        token_state, token_text = "local", "Inferencia local: no requiere token ni créditos"
    else:
        token_state = "live" if settings.has_token else "error"
        token_text = "API de Hugging Face · " + ("token configurado" if settings.has_token else "token no configurado")
    st.html(f'<p class="od-hint" style="margin-bottom:8px">{ui.state_badge(token_state, token_text)}</p>')
    st.html(
        '<p class="od-hint">Crea <code>.streamlit/secrets.toml</code> (está en .gitignore) '
        "o define variables de entorno con los mismos nombres:</p>"
    )
    st.code(
        '# INFERENCE_BACKEND = "auto"   # auto | local | api  (auto = local si torch está instalado)\n'
        'HF_TOKEN = "hf_..."             # solo necesario con el backend "api"\n'
        '# DETECTION_MODEL = "SenseTime/deformable-detr"\n'
        '# DETECTION_FALLBACK_MODEL = "facebook/detr-resnet-50"   # "none" para desactivarlo\n'
        '# TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"\n'
        '# HF_INFERENCE_BASE_URL = "https://router.huggingface.co/hf-inference/models"',
        language="toml",
    )

    ui.section_title("Información técnica")
    ui.tech_info(settings, det_model)
