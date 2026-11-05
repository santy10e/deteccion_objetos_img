"""Crea (o actualiza) un Space de Hugging Face con esta aplicación.

    pip install huggingface_hub
    hf auth login                      # token con permiso de escritura ("Write")
    python scripts/deploy_space.py tu-usuario/object-detection --dry-run   # revisar
    python scripts/deploy_space.py tu-usuario/object-detection

El Space usa el SDK Docker en CPU gratuita y ejecuta los modelos con
transformers, así que no necesita token ni créditos de inferencia.
Solo se suben los archivos listados abajo: nunca `.streamlit/secrets.toml`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "app.py",
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "requirements-local.txt",
    ".streamlit/config.toml",
    "scripts/prefetch_models.py",
]
PACKAGES = ["core", "ui"]
SPACE_README = ROOT / "space" / "README.md"  # README con la cabecera YAML que exige Spaces


def files_to_upload() -> list[str]:
    paths = list(FILES)
    for package in PACKAGES:
        paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / package).rglob("*.py"))]
    missing = [p for p in paths if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(f"Faltan archivos: {', '.join(missing)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space_id", help="usuario/nombre-del-space")
    parser.add_argument("--private", action="store_true", help="crear el Space como privado")
    parser.add_argument("--dry-run", action="store_true", help="mostrar qué se subiría, sin subir nada")
    args = parser.parse_args()

    paths = files_to_upload()
    print(f"Space: {args.space_id}  ({len(paths) + 1} archivos)")
    for path in [*paths, "README.md  <- space/README.md"]:
        print("  ", path)
    if args.dry_run:
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))  # sin HF_TOKEN usa la sesión de `hf auth login`
    api.create_repo(args.space_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(ROOT / p)) for p in paths]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(SPACE_README)))
    api.create_commit(
        repo_id=args.space_id,
        repo_type="space",
        operations=operations,
        commit_message="Deploy Object Detection",
    )
    print(f"\nListo. La primera construcción tarda varios minutos: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
