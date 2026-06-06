"""SQLite database for query and forecast history."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).resolve().parent / "history.db"


class HistoryDatabase:
    """Manages persistent storage of queries and forecast results."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_query TEXT NOT NULL,
                    forecast_result TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()
        logger.info("Database initialized at %s", self.db_path)

    def save_query(
        self,
        user_query: str,
        forecast_result: dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Store a query and its forecast result."""
        ts = (timestamp or datetime.utcnow()).isoformat()
        payload = json.dumps(forecast_result, default=str)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO query_history (user_query, forecast_result, timestamp)
                VALUES (?, ?, ?)
                """,
                (user_query, payload, ts),
            )
            conn.commit()
            row_id = cursor.lastrowid

        logger.info("Saved query history id=%d", row_id)
        return row_id

    def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent query history entries."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, user_query, forecast_result, timestamp
                FROM query_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "user_query": row["user_query"],
                    "forecast_result": json.loads(row["forecast_result"]),
                    "timestamp": row["timestamp"],
                }
            )
        return results

    def get_by_id(self, record_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a single history record by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM query_history WHERE id = ?", (record_id,)
            ).fetchone()

        if not row:
            return None
        return {
            "id": row["id"],
            "user_query": row["user_query"],
            "forecast_result": json.loads(row["forecast_result"]),
            "timestamp": row["timestamp"],
        }

    def count(self) -> int:
        """Return total number of stored records."""
        with self._get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()
        return result[0] if result else 0
