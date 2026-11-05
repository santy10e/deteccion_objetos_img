"""Orquestación de la inferencia: detección -> traducción de etiquetas.

Emite eventos de progreso reales (`on_step`) para que la UI refleje en todo
momento lo que se está ejecutando, sin animaciones simuladas.
"""

from __future__ import annotations

from typing import Callable, Literal

import streamlit as st

from core import hf_client, local_inference
from core.config import DEFAULT_INFERENCE_BASE_URL, Settings
from core.detection import AnalysisResult, Detection, LoadedImage, parse_detections
from core.hf_client import HFModelUnavailableError, ModelStatus

Step = Literal["prepare", "load", "detect", "fallback", "translate", "done"]
StepCallback = Callable[[Step, str], None]


@st.cache_data(ttl=600, show_spinner=False)
def get_model_status(model: str, hub_api_url: str) -> ModelStatus:
    return hf_client.fetch_model_status(model, hub_api_url=hub_api_url)


def model_statuses(settings: Settings) -> dict[str, ModelStatus]:
    """Estado de cada modelo configurado según el Hub (cacheado 10 min)."""
    if settings.is_local:
        # Se ejecutan en este mismo servidor: no dependen de proveedores externos.
        return {settings.detection_model: "local", settings.translation_model: "local"}
    if settings.inference_base_url != DEFAULT_INFERENCE_BASE_URL:
        # Endpoint propio: el Hub no sabe si está activo.
        models = [settings.detection_model, settings.detection_fallback_model, settings.translation_model]
        return {m: "unknown" for m in models if m}
    statuses = {
        settings.detection_model: get_model_status(settings.detection_model, settings.hub_api_url),
        settings.translation_model: get_model_status(settings.translation_model, settings.hub_api_url),
    }
    fallback = settings.detection_fallback_model
    if fallback and statuses[settings.detection_model] != "live":
        statuses[fallback] = get_model_status(fallback, settings.hub_api_url)
    return statuses


def active_detection_model(settings: Settings, statuses: dict[str, ModelStatus]) -> tuple[str, bool]:
    """Modelo de detección que se usará y si es el alternativo."""
    fallback = settings.detection_fallback_model
    if statuses.get(settings.detection_model) == "unavailable" and fallback:
        return fallback, True
    return settings.detection_model, False


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=2000)
def _cached_translation(text: str, model: str, backend: str, base_url: str, _token: str | None) -> str:
    # Las excepciones no se cachean: un fallo puntual se reintenta en la siguiente ejecución.
    if backend == "local":
        return local_inference.translate_text(text, model=model)
    return hf_client.translate_text(text, model=model, token=_token, base_url=base_url)


def _detect(image: LoadedImage, model: str, settings: Settings, on_step: StepCallback) -> list:
    if settings.is_local:
        if not local_inference.is_loaded("object-detection", model):
            on_step("load", f"{model} · solo la primera vez")
        return local_inference.query_detection(image.image, model=model)
    return hf_client.query_detection(
        image.data, image.mime, model=model, token=settings.hf_token, base_url=settings.inference_base_url
    )


def run_analysis(
    image: LoadedImage,
    settings: Settings,
    statuses: dict[str, ModelStatus],
    on_step: StepCallback,
) -> AnalysisResult:
    token, base = settings.hf_token, settings.inference_base_url
    model, fallback_used = active_detection_model(settings, statuses)

    on_step("prepare", f"{image.format} · {image.width}×{image.height}px")
    on_step("detect", model)
    try:
        raw = _detect(image, model, settings, on_step)
    except HFModelUnavailableError:
        fallback = settings.detection_fallback_model
        if fallback_used or not fallback or fallback == model:
            raise
        model, fallback_used = fallback, True
        on_step("fallback", model)
        raw = _detect(image, model, settings, on_step)

    parsed = parse_detections(raw, image.width, image.height)

    # Traducimos cada etiqueta distinta una sola vez (la original lo hacía por detección).
    unique_labels = list(dict.fromkeys(label for label, _, _ in parsed))
    translations: dict[str, str] = {}
    failures: list[str] = []
    if unique_labels:
        if settings.is_local and not local_inference.is_loaded("translation", settings.translation_model):
            on_step("load", f"{settings.translation_model} · solo la primera vez")
        on_step("translate", f"{len(unique_labels)} etiqueta(s) únicas")
    for label in unique_labels:
        try:
            translations[label] = _cached_translation(label, settings.translation_model, settings.backend, base, token)
        except hf_client.HFError:
            # Igual que la versión original: si falla, se conserva la etiqueta en inglés.
            translations[label] = label
            failures.append(label)

    detections = tuple(
        Detection(id=i, label=label, label_es=translations[label], score=score, box=box)
        for i, (label, score, box) in enumerate(parsed, start=1)
    )
    on_step("done", f"{len(detections)} objeto(s)")
    return AnalysisResult(
        detections=detections,
        model_used=model,
        fallback_used=fallback_used,
        translation_model=settings.translation_model,
        translation_failures=tuple(failures),
    )
