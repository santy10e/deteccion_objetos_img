"""Cliente HTTP para la inferencia de Hugging Face.

No depende de Streamlit: lanza excepciones tipadas que la capa de UI traduce a
mensajes comprensibles para el usuario.
"""

from __future__ import annotations

from typing import Any, Literal

import requests

from core.config import REQUEST_TIMEOUT_S, STATUS_TIMEOUT_S

ModelStatus = Literal["live", "local", "unavailable", "unknown"]


class HFError(Exception):
    """Error base. `user_message` es apto para mostrarse en la interfaz."""

    title = "Error de Hugging Face"

    def __init__(self, user_message: str, detail: str | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class HFAuthError(HFError):
    title = "Token de Hugging Face no válido"


class HFModelUnavailableError(HFError):
    title = "Modelo no disponible"


class HFModelLoadingError(HFError):
    title = "El modelo se está iniciando"


class HFRateLimitError(HFError):
    title = "Límite de peticiones alcanzado"


class HFQuotaError(HFError):
    """402: la cuenta agotó los créditos de Inference Providers (no depende del modelo)."""

    title = "Créditos de Hugging Face agotados"


class HFConnectionError(HFError):
    title = "Sin conexión con Hugging Face"


class HFResponseError(HFError):
    title = "Respuesta inesperada"


def _error_text(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return ""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, list):
            return "; ".join(map(str, err))
        return str(err or "")
    return ""


def _post(url: str, token: str | None, **kwargs: Any) -> Any:
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Espera a que el modelo cargue en lugar de devolver 503 de inmediato.
    headers.setdefault("x-wait-for-model", "true")
    try:
        response = requests.post(url, headers=headers, timeout=REQUEST_TIMEOUT_S, **kwargs)
    except requests.Timeout as exc:
        raise HFConnectionError(
            "Hugging Face tardó demasiado en responder. Inténtalo de nuevo en unos segundos.",
            str(exc),
        ) from exc
    except requests.RequestException as exc:
        raise HFConnectionError(
            "No se pudo conectar con Hugging Face. Revisa tu conexión a internet.",
            str(exc),
        ) from exc

    status = response.status_code
    detail = _error_text(response) or response.text[:200]

    if status in (401, 403):
        raise HFAuthError(
            "Hugging Face rechazó el token de acceso (no válido, caducado o sin permisos de inferencia).",
            detail,
        )
    if status == 402:
        raise HFQuotaError(
            "Tu cuenta de Hugging Face ha consumido los créditos gratuitos de Inference Providers. "
            "Le pasa a cualquier modelo que uses con este token, así que cambiar de modelo no lo soluciona.",
            detail,
        )
    if status == 404 or (status == 400 and "not supported" in detail.lower()):
        raise HFModelUnavailableError(
            "El modelo no está desplegado en la inferencia serverless de Hugging Face.",
            detail,
        )
    if status == 429:
        raise HFRateLimitError(
            "Se alcanzó el límite de peticiones de tu cuenta de Hugging Face. Espera un momento.",
            detail,
        )
    if status == 503:
        raise HFModelLoadingError(
            "El modelo se está iniciando en los servidores de Hugging Face. Reintenta en unos segundos.",
            detail,
        )
    if status >= 400:
        raise HFResponseError(f"Hugging Face devolvió el error HTTP {status}.", detail)

    try:
        return response.json()
    except ValueError as exc:
        raise HFResponseError("La respuesta de Hugging Face no es JSON válido.", response.text[:200]) from exc


def query_detection(
    image_bytes: bytes, mime: str, *, model: str, token: str | None, base_url: str
) -> list[dict[str, Any]]:
    """Envía la imagen al modelo de detección y devuelve la lista cruda de detecciones."""
    payload = _post(
        f"{base_url}/{model}",
        token,
        data=image_bytes,
        headers={"Content-Type": mime},
    )
    if isinstance(payload, dict) and payload.get("error"):
        raise HFResponseError("El modelo de detección devolvió un error.", str(payload["error"]))
    if not isinstance(payload, list):
        raise HFResponseError("Formato de respuesta de detección no reconocido.", str(payload)[:200])
    return payload


def translate_text(text: str, *, model: str, token: str | None, base_url: str) -> str:
    """Traduce un texto (inglés -> español) con el modelo Helsinki-NLP."""
    payload = _post(f"{base_url}/{model}", token, json={"inputs": text})
    try:
        translated = payload[0]["translation_text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HFResponseError("Respuesta de traducción no reconocida.", str(payload)[:200]) from exc
    return str(translated).strip() or text


def fetch_model_status(model: str, *, hub_api_url: str) -> ModelStatus:
    """Consulta en el Hub si el modelo tiene un proveedor de inferencia activo.

    Es una petición pública de metadatos (no ejecuta el modelo ni consume cuota).
    """
    try:
        response = requests.get(
            f"{hub_api_url}/{model}",
            params={"expand[]": "inferenceProviderMapping"},
            timeout=STATUS_TIMEOUT_S,
        )
        if response.status_code == 404:
            return "unavailable"
        response.raise_for_status()
        mapping = response.json().get("inferenceProviderMapping") or {}
    except (requests.RequestException, ValueError, AttributeError):
        return "unknown"

    if isinstance(mapping, list):  # formato alternativo de la API
        mapping = {item.get("provider"): item for item in mapping if isinstance(item, dict)}
    provider = mapping.get("hf-inference")
    if isinstance(provider, dict) and provider.get("status") == "live":
        return "live"
    return "unavailable"
