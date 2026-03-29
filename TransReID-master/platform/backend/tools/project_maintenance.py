"""Project cleanup helpers for local development/runtime maintenance.

Usage:
  python tools/project_maintenance.py --all
  python tools/project_maintenance.py --clear-db --clear-uploads
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime_paths import UPLOAD_DIR, ensure_runtime_dirs

DB_PATH = BACKEND_DIR / "ballshow.db"

DB_TABLES_TO_CLEAR = (
    "search_history",
    "video_tasks",
    "trajectory_tasks",
    "game_analysis_tasks",
    "game_analysis_exports",
)

FILES_TO_PURGE = (
    BACKEND_DIR / "_tmp_check_logic.py",
    BACKEND_DIR / "trajectory_ab_test.py",
    BACKEND_DIR / "trajectory_batch_eval.py",
    BACKEND_DIR / "trajectory_pseudo_label.py",
    BACKEND_DIR / "trajectory_pseudo_train.py",
)

DIRS_TO_PURGE = (
    BACKEND_DIR / "__pycache__",
    BACKEND_DIR / "routes" / "__pycache__",
    BACKEND_DIR / ".hf_cache",
)


def _safe_remove_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False


def _safe_remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        shutil.rmtree(path, ignore_errors=True)
        return True
    except Exception:
        return False


def purge_test_and_temp_files() -> dict:
    removed_files = sum(1 for p in FILES_TO_PURGE if _safe_remove_file(p))
    removed_dirs = sum(1 for p in DIRS_TO_PURGE if _safe_remove_tree(p))
    return {"removed_files": removed_files, "removed_dirs": removed_dirs}


def clear_uploads() -> dict:
    upload_path = Path(UPLOAD_DIR)
    removed_entries = 0
    if upload_path.exists():
        for child in upload_path.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                removed_entries += 1
            except Exception:
                continue
    ensure_runtime_dirs()
    return {"removed_entries": removed_entries}


def clear_runtime_db() -> dict:
    if not DB_PATH.exists():
        return {"db_exists": False}
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    for table in DB_TABLES_TO_CLEAR:
        cur.execute(f"DELETE FROM {table}")
    try:
        names = ",".join(f"'{t}'" for t in DB_TABLES_TO_CLEAR)
        cur.execute(f"DELETE FROM sqlite_sequence WHERE name IN ({names})")
    except Exception:
        pass
    conn.commit()
    summary = {"db_exists": True}
    for table in ("users",) + DB_TABLES_TO_CLEAR:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        summary[table] = int(count)
    conn.close()
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BallShow project maintenance utility")
    parser.add_argument("--purge-temp", action="store_true", help="Delete temp/test/cache files")
    parser.add_argument("--clear-uploads", action="store_true", help="Delete all generated upload artifacts")
    parser.add_argument("--clear-db", action="store_true", help="Clear runtime task/history tables")
    parser.add_argument("--all", action="store_true", help="Run all maintenance steps")
    args = parser.parse_args(argv)

    if args.all:
        args.purge_temp = True
        args.clear_uploads = True
        args.clear_db = True

    if not any((args.purge_temp, args.clear_uploads, args.clear_db)):
        parser.print_help()
        return 1

    if args.purge_temp:
        print("[maintenance] purge-temp:", purge_test_and_temp_files())
    if args.clear_uploads:
        print("[maintenance] clear-uploads:", clear_uploads())
    if args.clear_db:
        print("[maintenance] clear-db:", clear_runtime_db())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
