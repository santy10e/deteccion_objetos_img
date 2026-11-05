"""Configuración central de la aplicación.

Los valores sensibles (token de Hugging Face) se leen de `st.secrets` o de
variables de entorno; nunca deben escribirse en el código fuente.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Literal

Backend = Literal["api", "local"]

APP_NAME = "Object Detection"
APP_TAGLINE = "AI-powered visual recognition"
APP_VERSION = "2.0.0"

# Formatos aceptados: extensión del archivo -> formato PIL esperado.
ALLOWED_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png", "webp")
ALLOWED_FORMATS: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
MAX_UPLOAD_MB = 10

# Hugging Face retiró `api-inference.huggingface.co`; el endpoint vigente de la
# inferencia serverless es el router. Se puede sobrescribir (p. ej. para un
# Inference Endpoint dedicado o para pruebas locales).
DEFAULT_INFERENCE_BASE_URL = "https://router.huggingface.co/hf-inference/models"
DEFAULT_HUB_API_URL = "https://huggingface.co/api/models"

DEFAULT_DETECTION_MODEL = "SenseTime/deformable-detr"
# Mismo tipo de tarea y formato de respuesta que Deformable DETR. Solo se usa
# si el modelo principal no está desplegado en la inferencia de Hugging Face.
DEFAULT_DETECTION_FALLBACK_MODEL = "facebook/detr-resnet-50"
DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"

REQUEST_TIMEOUT_S = 90
STATUS_TIMEOUT_S = 6


def _read_setting(name: str, default: str | None = None) -> str | None:
    """Lee un ajuste de `st.secrets` y, si no existe, de las variables de entorno."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            return str(value).strip() or default
    except Exception:  # noqa: BLE001 - sin secrets.toml, Streamlit lanza excepción
        pass
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    hf_token: str | None
    inference_base_url: str
    hub_api_url: str
    detection_model: str
    detection_fallback_model: str | None
    translation_model: str
    backend: Backend = "api"

    @property
    def has_token(self) -> bool:
        return bool(self.hf_token)

    @property
    def is_local(self) -> bool:
        return self.backend == "local"

    @property
    def can_analyze(self) -> bool:
        # La inferencia local no necesita token; la API sí.
        return self.is_local or self.has_token

    @property
    def platform_label(self) -> str:
        return "transformers · CPU" if self.is_local else "Hugging Face Inference"


def local_inference_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def _resolve_backend(value: str | None) -> Backend:
    """`auto` (por defecto): local si torch+transformers están instalados; si no, API."""
    value = (value or "auto").lower()
    if value in ("api", "local"):
        return value  # type: ignore[return-value]
    return "local" if local_inference_available() else "api"


def load_settings() -> Settings:
    token = _read_setting("HF_TOKEN") or _read_setting("HUGGINGFACEHUB_API_TOKEN")
    fallback = _read_setting("DETECTION_FALLBACK_MODEL", DEFAULT_DETECTION_FALLBACK_MODEL)
    return Settings(
        hf_token=token,
        inference_base_url=_read_setting("HF_INFERENCE_BASE_URL", DEFAULT_INFERENCE_BASE_URL).rstrip("/"),
        hub_api_url=_read_setting("HF_HUB_API_URL", DEFAULT_HUB_API_URL).rstrip("/"),
        detection_model=_read_setting("DETECTION_MODEL", DEFAULT_DETECTION_MODEL),
        # "none" desactiva el modelo alternativo.
        detection_fallback_model=None if (fallback or "").lower() == "none" else fallback,
        translation_model=_read_setting("TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL),
        backend=_resolve_backend(_read_setting("INFERENCE_BACKEND")),
    )
