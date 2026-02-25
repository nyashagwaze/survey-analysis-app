import os
import shutil
from pathlib import Path


def _dbfs_to_local(dbfs_path: str) -> str:
    if dbfs_path.startswith("dbfs:/"):
        return "/dbfs/" + dbfs_path[len("dbfs:/"):].lstrip("/")
    return dbfs_path


def _local_to_dbfs(local_path: str) -> str:
    if local_path.startswith("/dbfs/"):
        return "dbfs:/" + local_path[len("/dbfs/"):].lstrip("/")
    return local_path


def copy_workspace_to_dbfs(workspace_path: str, dbfs_dir: str, overwrite: bool = True) -> str:
    """
    Copy a /Workspace file to a DBFS location (driver-local /dbfs).

    Returns a Spark-readable dbfs:/ URI.
    """
    if not workspace_path or not str(workspace_path).startswith("/Workspace/"):
        raise ValueError("workspace_path must start with /Workspace/ for this helper.")

    src = Path(workspace_path)
    if not src.exists():
        raise FileNotFoundError(f"Workspace file not found: {workspace_path}")

    local_dbfs_dir = _dbfs_to_local(dbfs_dir)
    os.makedirs(local_dbfs_dir, exist_ok=True)

    dest_local = Path(local_dbfs_dir) / src.name
    if dest_local.exists():
        if not overwrite:
            return _local_to_dbfs(str(dest_local))
        dest_local.unlink()

    shutil.copy2(str(src), str(dest_local))
    return _local_to_dbfs(str(dest_local))
