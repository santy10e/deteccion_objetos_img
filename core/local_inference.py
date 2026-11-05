"""Inferencia local con `transformers` (CPU).

Pensada para ejecutar la app en un Space gratuito de Hugging Face o en
cualquier servidor: no consume créditos de Inference Providers ni requiere
token. Devuelve exactamente el mismo formato que la API, de modo que el resto
de la aplicación no distingue entre ambos backends.
"""

from __future__ import annotations

import threading
from typing import Any

from PIL import Image

from core.hf_client import HFModelUnavailableError, HFResponseError

_MODELS: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def is_loaded(task: str, model: str) -> bool:
    return (task, model) in _MODELS


def _load_detector(model: str) -> Any:
    from transformers import pipeline

    return pipeline("object-detection", model=model, device=-1)


def _load_translator(model: str) -> Any:
    # transformers 5 eliminó el pipeline "translation": usamos MarianMT directamente.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model).eval()
    return tokenizer, seq2seq


_LOADERS = {"object-detection": _load_detector, "translation": _load_translator}


def get_model(task: str, model: str) -> Any:
    """Carga el modelo una sola vez por proceso (ocupan cientos de MB)."""
    key = (task, model)
    if key in _MODELS:
        return _MODELS[key]
    with _LOCK:  # evita cargas duplicadas si dos usuarios analizan a la vez
        if key not in _MODELS:
            try:
                _MODELS[key] = _LOADERS[task](model)
            except ImportError as exc:
                raise HFModelUnavailableError(
                    "Falta una dependencia para ejecutar el modelo en local. "
                    "Instala requirements-local.txt.",
                    str(exc),
                ) from exc
            except (OSError, ValueError, KeyError) as exc:
                raise HFModelUnavailableError(
                    f"No se pudo descargar o cargar el modelo {model}.", str(exc)
                ) from exc
    return _MODELS[key]


def query_detection(image: Image.Image, *, model: str) -> list[dict[str, Any]]:
    detector = get_model("object-detection", model)
    try:
        return list(detector(image.convert("RGB")))
    except Exception as exc:  # noqa: BLE001 - cualquier fallo del modelo se muestra en la UI
        raise HFResponseError("El modelo de detección falló al procesar la imagen.", repr(exc)) from exc


def translate_text(text: str, *, model: str) -> str:
    tokenizer, seq2seq = get_model("translation", model)
    try:
        import torch

        with torch.inference_mode():
            batch = tokenizer([text], return_tensors="pt", padding=True)
            output = seq2seq.generate(**batch, max_new_tokens=64)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip() or text
    except Exception as exc:  # noqa: BLE001
        raise HFResponseError("El modelo de traducción falló.", repr(exc)) from exc
