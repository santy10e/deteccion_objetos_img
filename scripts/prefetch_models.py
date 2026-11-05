"""Descarga y carga una vez los modelos para dejarlos en la caché de Hugging Face.

Se usa al construir la imagen Docker; también sirve en local para no esperar
la descarga en el primer análisis:  python scripts/prefetch_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import DEFAULT_DETECTION_MODEL, DEFAULT_TRANSLATION_MODEL  # noqa: E402
from core.local_inference import get_model  # noqa: E402


def main() -> None:
    models = [
        ("object-detection", os.environ.get("DETECTION_MODEL", DEFAULT_DETECTION_MODEL)),
        ("translation", os.environ.get("TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL)),
    ]
    for task, model in models:
        print(f"-> {task}: {model}", flush=True)
        get_model(task, model)  # descarga pesos, configuración y backbone
    print("Modelos listos.")


if __name__ == "__main__":
    main()
