#!/usr/bin/env python3
"""Package the repo into a single zip for Databricks import.

Default: include code + configs + assets + docs + notebooks.
Optional: include the local sentence-transformer model.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile

DEFAULT_INCLUDE = [
    "src",
    "config",
    "assets",
    "docs",
    "notebooks",
    "scripts",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "Methedology",
]

SKIP_DIRS = {
    ".git",
    ".venv",
    ".cache",
    "__pycache__",
    "outputs",
    "Deliverables",
    "Performance_metrics",
    "Final_deliverable",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "build",
    "dist",
}

SKIP_SUFFIXES = {".log", ".tmp"}


def _should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def _add_file(zf: zipfile.ZipFile, file_path: Path, root: Path) -> None:
    rel = file_path.relative_to(root)
    zf.write(file_path, rel.as_posix())


def _add_dir(zf: zipfile.ZipFile, dir_path: Path, root: Path) -> None:
    for base, dirs, files in os.walk(dir_path):
        base_path = Path(base)
        # Prune skipped dirs
        dirs[:] = [d for d in dirs if not _should_skip(base_path / d)]
        for name in files:
            fpath = base_path / name
            if _should_skip(fpath):
                continue
            _add_file(zf, fpath, root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package repo into a zip bundle.")
    parser.add_argument("--root", default=None, help="Repo root (defaults to parent of scripts/)")
    parser.add_argument("--out", default=None, help="Output zip path")
    parser.add_argument("--include-model", action="store_true", help="Include Data/embedding_classifier_multi/encoder")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    out_path = Path(args.out).resolve() if args.out else root / "dist" / "wellbeing_bundle.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    include_paths = list(DEFAULT_INCLUDE)
    if args.include_model:
        include_paths.append("Data/embedding_classifier_multi/encoder")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in include_paths:
            p = (root / item).resolve()
            if not p.exists():
                continue
            if p.is_file():
                _add_file(zf, p, root)
            else:
                _add_dir(zf, p, root)

    print(f"Bundle created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
