"""Database helpers for BallShow platform."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import bcrypt

DB_PATH = os.path.join(os.path.dirname(__file__), "ballshow.db")
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"
DEFAULT_ADMIN_NICKNAME = "管理员"


def _hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_default_admin(cursor: sqlite3.Cursor) -> None:
    row = cursor.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (DEFAULT_ADMIN_USERNAME,),
    ).fetchone()

    if row is None:
        cursor.execute(
            "INSERT INTO users (username, password_hash, nickname) VALUES (?, ?, ?)",
            (
                DEFAULT_ADMIN_USERNAME,
                _hash_password(DEFAULT_ADMIN_PASSWORD),
                DEFAULT_ADMIN_NICKNAME,
            ),
        )
        return

    stored_hash = row["password_hash"] or ""
    valid = False
    try:
        valid = bcrypt.checkpw(DEFAULT_ADMIN_PASSWORD.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        valid = False

    if not valid:
        cursor.execute(
            "UPDATE users SET password_hash = ?, nickname = ? WHERE username = ?",
            (
                _hash_password(DEFAULT_ADMIN_PASSWORD),
                DEFAULT_ADMIN_NICKNAME,
                DEFAULT_ADMIN_USERNAME,
            ),
        )


def init_db() -> None:
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            nickname TEXT DEFAULT '',
            avatar TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query_image TEXT NOT NULL,
            result_count INTEGER DEFAULT 0,
            top1_score REAL DEFAULT 0.0,
            search_type TEXT DEFAULT 'image',
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS video_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            video_filename TEXT NOT NULL,
            query_image TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            total_frames INTEGER DEFAULT 0,
            matched_segments INTEGER DEFAULT 0,
            result_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            finished_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trajectory_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            video_filename TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            total_frames INTEGER DEFAULT 0,
            detected_points INTEGER DEFAULT 0,
            result_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            finished_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS game_analysis_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            video_filename TEXT NOT NULL,
            query_image TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            options_json TEXT DEFAULT '{}',
            total_frames INTEGER DEFAULT 0,
            result_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            finished_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS game_analysis_exports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            format TEXT NOT NULL,
            filename TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (task_id) REFERENCES game_analysis_tasks(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    _ensure_default_admin(cursor)
    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str, nickname: str = "") -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, password_hash, nickname) VALUES (?, ?, ?)",
        (username, password_hash, nickname),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return int(user_id)


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_search_history(user_id: int, query_image: str, result_count: int, top1_score: float, search_type: str = "image") -> None:
    conn = get_db()
    conn.execute(
        "INSERT INTO search_history (user_id, query_image, result_count, top1_score, search_type) VALUES (?, ?, ?, ?, ?)",
        (user_id, query_image, result_count, top1_score, search_type),
    )
    conn.commit()
    conn.close()


def get_dashboard_stats() -> Dict[str, Any]:
    conn = get_db()
    stats: Dict[str, Any] = {}

    stats["total_users"] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    stats["total_searches"] = conn.execute("SELECT COUNT(*) FROM search_history").fetchone()[0]
    stats["total_video_tasks"] = conn.execute("SELECT COUNT(*) FROM video_tasks").fetchone()[0]
    stats["total_trajectory_tasks"] = conn.execute("SELECT COUNT(*) FROM trajectory_tasks").fetchone()[0]
    stats["total_game_analysis_tasks"] = conn.execute("SELECT COUNT(*) FROM game_analysis_tasks").fetchone()[0]
    stats["avg_top1_score"] = conn.execute("SELECT COALESCE(AVG(top1_score), 0) FROM search_history").fetchone()[0]

    rows = conn.execute(
        """
        SELECT DATE(created_at) AS day, COUNT(*) AS cnt
        FROM search_history
        WHERE created_at >= datetime('now', '-7 days', 'localtime')
        GROUP BY DATE(created_at)
        ORDER BY day
        """
    ).fetchall()
    stats["daily_searches"] = [{"date": r["day"], "count": r["cnt"]} for r in rows]

    rows = conn.execute("SELECT search_type, COUNT(*) AS cnt FROM search_history GROUP BY search_type").fetchall()
    stats["search_type_dist"] = [{"type": r["search_type"], "count": r["cnt"]} for r in rows]

    conn.close()
    return stats


def create_video_task(user_id: int, video_filename: str, query_image: str) -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO video_tasks (user_id, video_filename, query_image) VALUES (?, ?, ?)",
        (user_id, video_filename, query_image),
    )
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    return int(task_id)


def update_video_task(task_id: int, **kwargs: Any) -> None:
    if not kwargs:
        return
    conn = get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    conn.execute(f"UPDATE video_tasks SET {sets} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_video_task(task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM video_tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_video_history(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM video_tasks WHERE user_id = ? ORDER BY id DESC LIMIT 50",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_trajectory_task(user_id: int, video_filename: str) -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO trajectory_tasks (user_id, video_filename) VALUES (?, ?)",
        (user_id, video_filename),
    )
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    return int(task_id)


def update_trajectory_task(task_id: int, **kwargs: Any) -> None:
    if not kwargs:
        return
    conn = get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    conn.execute(f"UPDATE trajectory_tasks SET {sets} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_trajectory_task(task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM trajectory_tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_trajectory_history(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM trajectory_tasks WHERE user_id = ? ORDER BY id DESC LIMIT 50",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_trajectory_history_paginated(
    user_id: int,
    page: int,
    page_size: int,
    status: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    page = max(1, int(page or 1))
    page_size = max(1, min(100, int(page_size or 10)))
    offset = (page - 1) * page_size

    conn = get_db()
    if status:
        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM trajectory_tasks WHERE user_id = ? AND status = ?",
            (user_id, status),
        ).fetchone()
        rows = conn.execute(
            """
            SELECT *
            FROM trajectory_tasks
            WHERE user_id = ? AND status = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, status, page_size, offset),
        ).fetchall()
    else:
        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM trajectory_tasks WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        rows = conn.execute(
            """
            SELECT *
            FROM trajectory_tasks
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, page_size, offset),
        ).fetchall()

    conn.close()
    total = int(count_row["cnt"] if count_row else 0)
    return [dict(r) for r in rows], total


def get_user_trajectory_task_detail(user_id: int, task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM trajectory_tasks WHERE user_id = ? AND id = ?",
        (user_id, task_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def create_game_analysis_task(user_id: int, video_filename: str, query_image: str = "", options_json: str = "{}") -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO game_analysis_tasks (user_id, video_filename, query_image, options_json) VALUES (?, ?, ?, ?)",
        (user_id, video_filename, query_image, options_json),
    )
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    return int(task_id)


def update_game_analysis_task(task_id: int, **kwargs: Any) -> None:
    if not kwargs:
        return
    conn = get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    conn.execute(f"UPDATE game_analysis_tasks SET {sets} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_game_analysis_task(task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM game_analysis_tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_game_analysis_task_detail(user_id: int, task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM game_analysis_tasks WHERE user_id = ? AND id = ?",
        (user_id, task_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_game_analysis_history_paginated(
    user_id: int,
    page: int,
    page_size: int,
    status: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    page = max(1, int(page or 1))
    page_size = max(1, min(100, int(page_size or 10)))
    offset = (page - 1) * page_size

    conn = get_db()
    if status:
        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM game_analysis_tasks WHERE user_id = ? AND status = ?",
            (user_id, status),
        ).fetchone()
        rows = conn.execute(
            """
            SELECT *
            FROM game_analysis_tasks
            WHERE user_id = ? AND status = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, status, page_size, offset),
        ).fetchall()
    else:
        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM game_analysis_tasks WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        rows = conn.execute(
            """
            SELECT *
            FROM game_analysis_tasks
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, page_size, offset),
        ).fetchall()

    conn.close()
    total = int(count_row["cnt"] if count_row else 0)
    return [dict(r) for r in rows], total


def create_game_analysis_export(task_id: int, user_id: int, fmt: str, filename: str) -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO game_analysis_exports (task_id, user_id, format, filename) VALUES (?, ?, ?, ?)",
        (task_id, user_id, fmt, filename),
    )
    conn.commit()
    export_id = cursor.lastrowid
    conn.close()
    return int(export_id)


def get_task_exports(task_id: int, user_id: int) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        """
        SELECT *
        FROM game_analysis_exports
        WHERE task_id = ? AND user_id = ?
        ORDER BY id DESC
        """,
        (task_id, user_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
