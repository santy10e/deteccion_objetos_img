"""Estructuras de datos y validación de imágenes y detecciones."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from core.config import ALLOWED_EXTENSIONS, ALLOWED_FORMATS, MAX_UPLOAD_MB


class ImageValidationError(Exception):
    """Error de validación con mensaje apto para el usuario."""

    def __init__(self, title: str, message: str):
        super().__init__(message)
        self.title = title
        self.message = message


@dataclass(frozen=True)
class LoadedImage:
    name: str
    data: bytes
    mime: str
    format: str
    width: int
    height: int
    signature: str
    image: Image.Image = field(repr=False, compare=False)

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass(frozen=True)
class Detection:
    id: int
    label: str
    label_es: str
    score: float
    box: Box


@dataclass(frozen=True)
class AnalysisResult:
    detections: tuple[Detection, ...]
    model_used: str
    fallback_used: bool
    translation_model: str
    translation_failures: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return len(self.detections)

    @property
    def class_count(self) -> int:
        return len({d.label for d in self.detections})

    @property
    def mean_score(self) -> float | None:
        if not self.detections:
            return None
        return sum(d.score for d in self.detections) / len(self.detections)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} MB"


def load_image(name: str, data: bytes) -> LoadedImage:
    """Valida el archivo antes de enviarlo al modelo."""
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    allowed = ", ".join(ext.upper() for ext in ALLOWED_EXTENSIONS)
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Formato no admitido", f"Usa una imagen {allowed}.")
    if not data:
        raise ImageValidationError("Archivo vacío", "El archivo seleccionado no contiene datos.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageValidationError(
            "Archivo demasiado grande",
            f"El tamaño máximo es {MAX_UPLOAD_MB} MB; este archivo ocupa {human_size(len(data))}.",
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # detecta archivos truncados o corruptos
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            "No se pudo leer la imagen",
            "El archivo está dañado o no es una imagen válida, aunque tenga la extensión correcta.",
        ) from exc

    fmt = (image.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(
            "Formato no admitido",
            f"El contenido real del archivo es {fmt or 'desconocido'}. Usa una imagen {allowed}.",
        )

    # El pipeline de Hugging Face aplica la orientación EXIF antes de inferir,
    # así que la usamos también para mostrar y dibujar las cajas.
    image = ImageOps.exif_transpose(image)
    return LoadedImage(
        name=name,
        data=data,
        mime=ALLOWED_FORMATS[fmt],
        format=fmt,
        width=image.width,
        height=image.height,
        signature=hashlib.sha1(data).hexdigest(),
        image=image,
    )


def parse_detections(
    raw: list[Any], width: int, height: int
) -> list[tuple[str, float, Box]]:
    """Filtra entradas mal formadas, recorta cajas al tamaño de la imagen y ordena por confianza."""
    parsed: list[tuple[str, float, Box]] = []
    for item in raw:
        if not (isinstance(item, dict) and {"box", "label", "score"} <= item.keys()):
            continue
        b = item["box"]
        try:
            box = Box(
                xmin=max(0.0, min(float(b["xmin"]), width)),
                ymin=max(0.0, min(float(b["ymin"]), height)),
                xmax=max(0.0, min(float(b["xmax"]), width)),
                ymax=max(0.0, min(float(b["ymax"]), height)),
            )
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if box.width <= 0 or box.height <= 0:
            continue
        parsed.append((str(item["label"]), score, box))
    parsed.sort(key=lambda entry: entry[1], reverse=True)
    return parsed
