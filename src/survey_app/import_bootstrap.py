import os
import sys
from pathlib import Path


def _find_project_root(start: Path, max_depth: int = 8) -> str:
    candidate = start if start.is_dir() else start.parent
    for _ in range(max_depth):
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
        if (candidate / "src" / "survey_app").exists():
            return str(candidate)
        if (candidate / "config" / "pipeline_settings.yaml").exists():
            return str(candidate)
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return None


def _infer_from_dbutils() -> str:
    try:
        import builtins
        dbutils = getattr(builtins, "dbutils", None)
        if not dbutils:
            return None
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        # notebookPath returns workspace path like /Users/<user>/...
        ws_path = Path("/Workspace" + nb_path)
        return _find_project_root(ws_path)
    except Exception:
        return None


def bootstrap(project_root: str = None) -> Path:
    """
    Add src/ to sys.path so the survey_app package imports cleanly in notebooks.

    Args:
        project_root: Absolute repo root. If None, tries env vars or notebook path.

    Returns:
        Path to the survey_app package directory.
    """
    if project_root is None:
        project_root = (
            os.environ.get("PROJECT_ROOT")
            or os.environ.get("PIPELINE_PROJECT_ROOT")
            or os.environ.get("DATABRICKS_PROJECT_ROOT")
        )

    if project_root is None:
        project_root = _infer_from_dbutils()

    if project_root is None:
        project_root = _find_project_root(Path.cwd())

    if project_root is None:
        raise ValueError(
            "project_root not set. Pass project_root or set PROJECT_ROOT env var."
        )

    src_dir = Path(project_root) / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"src directory not found: {src_dir}")

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    pkg_dir = src_dir / "survey_app"
    if not pkg_dir.exists():
        raise FileNotFoundError(f"Package directory not found: {pkg_dir}")

    return pkg_dir
