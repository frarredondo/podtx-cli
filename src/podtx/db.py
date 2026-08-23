from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from podtx.models import Feed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS feeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed_id INTEGER NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
                guid TEXT NOT NULL,
                title TEXT NOT NULL,
                published_at TEXT,
                episode_num INTEGER,
                enclosure_url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                engine TEXT,
                model TEXT,
                output_paths_json TEXT,
                transcribed_at TEXT,
                UNIQUE(feed_id, guid)
            );
            """
        )
        self._conn.commit()

    def add_feed(self, url: str, slug: str, title: str) -> Feed:
        now = _utc_now()
        cur = self._conn.execute(
            "INSERT INTO feeds (url, slug, title, created_at) VALUES (?, ?, ?, ?)",
            (url, slug, title, now),
        )
        self._conn.commit()
        return Feed(id=int(cur.lastrowid), url=url, slug=slug, title=title, created_at=datetime.fromisoformat(now))

    def remove_feed(self, slug_or_url: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM feeds WHERE slug = ? OR url = ?",
            (slug_or_url, slug_or_url),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_feeds(self) -> list[Feed]:
        rows = self._conn.execute("SELECT * FROM feeds ORDER BY title COLLATE NOCASE").fetchall()
        return [self._row_to_feed(r) for r in rows]

    def get_feed(self, slug_or_url: str) -> Feed | None:
        row = self._conn.execute(
            "SELECT * FROM feeds WHERE slug = ? OR url = ?",
            (slug_or_url, slug_or_url),
        ).fetchone()
        return self._row_to_feed(row) if row else None

    def get_feed_by_id(self, feed_id: int) -> Feed | None:
        row = self._conn.execute("SELECT * FROM feeds WHERE id = ?", (feed_id,)).fetchone()
        return self._row_to_feed(row) if row else None

    def done_guids(self, feed_id: int) -> set[str]:
        rows = self._conn.execute(
            "SELECT guid FROM episodes WHERE feed_id = ? AND status = 'done'",
            (feed_id,),
        ).fetchall()
        return {r["guid"] for r in rows}

    def episode_count(self, feed_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE feed_id = ?",
            (feed_id,),
        ).fetchone()
        return int(row["c"]) if row else 0

    def done_count(self, feed_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE feed_id = ? AND status = 'done'",
            (feed_id,),
        ).fetchone()
        return int(row["c"]) if row else 0

    def list_episodes(self, feed_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM episodes
            WHERE feed_id = ?
            ORDER BY published_at DESC NULLS LAST, id DESC
            """,
            (feed_id,),
        ).fetchall()

    def upsert_episode(
        self,
        *,
        feed_id: int,
        guid: str,
        title: str,
        published_at: datetime | None,
        episode_num: int | None,
        enclosure_url: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO episodes (
                feed_id, guid, title, published_at, episode_num, enclosure_url, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(feed_id, guid) DO UPDATE SET
                title = excluded.title,
                published_at = excluded.published_at,
                episode_num = excluded.episode_num,
                enclosure_url = excluded.enclosure_url
            WHERE episodes.status != 'done'
            """,
            (
                feed_id,
                guid,
                title,
                published_at.isoformat() if published_at else None,
                episode_num,
                enclosure_url,
            ),
        )
        self._conn.commit()

    def mark_done(
        self,
        *,
        feed_id: int,
        guid: str,
        engine: str,
        model: str,
        output_paths: list[Path],
    ) -> None:
        self._conn.execute(
            """
            UPDATE episodes
            SET status = 'done',
                engine = ?,
                model = ?,
                output_paths_json = ?,
                transcribed_at = ?
            WHERE feed_id = ? AND guid = ?
            """,
            (
                engine,
                model,
                json.dumps([str(p) for p in output_paths]),
                _utc_now(),
                feed_id,
                guid,
            ),
        )
        self._conn.commit()

    def update_episode_paths(
        self,
        *,
        feed_id: int,
        guid: str,
        episode_num: int,
        output_paths: list[Path],
    ) -> bool:
        """Update episode number and output paths after a rename (no status change)."""
        cur = self._conn.execute(
            """
            UPDATE episodes
            SET episode_num = ?,
                output_paths_json = ?
            WHERE feed_id = ? AND guid = ?
            """,
            (
                episode_num,
                json.dumps([str(p) for p in output_paths]),
                feed_id,
                guid,
            ),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def mark_error(self, *, feed_id: int, guid: str, message: str) -> None:
        self._conn.execute(
            """
            UPDATE episodes
            SET status = 'error',
                output_paths_json = ?
            WHERE feed_id = ? AND guid = ?
            """,
            (json.dumps({"error": message}), feed_id, guid),
        )
        self._conn.commit()

    def is_done(self, feed_id: int, guid: str) -> bool:
        row = self._conn.execute(
            "SELECT status FROM episodes WHERE feed_id = ? AND guid = ?",
            (feed_id, guid),
        ).fetchone()
        return bool(row and row["status"] == "done")

    @staticmethod
    def _row_to_feed(row: sqlite3.Row) -> Feed:
        return Feed(
            id=row["id"],
            url=row["url"],
            slug=row["slug"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ─── Health-check query helpers (Phase 1) ──────────────────────────────

    def failed_guids(self, feed_id: int) -> set[str]:
        """Return GUIDs of episodes whose status is 'error' for this feed."""
        rows = self._conn.execute(
            "SELECT guid FROM episodes WHERE feed_id = ? AND status = 'error'",
            (feed_id,),
        ).fetchall()
        return {r["guid"] for r in rows}

    def pending_guids(self, feed_id: int) -> set[str]:
        """Return GUIDs of episodes whose status is 'pending' for this feed."""
        rows = self._conn.execute(
            "SELECT guid FROM episodes WHERE feed_id = ? AND status = 'pending'",
            (feed_id,),
        ).fetchall()
        return {r["guid"] for r in rows}

    def empty_feeds(self) -> list[dict[str, object]]:
        """Return feeds that have zero episode records (never synced)."""
        rows = self._conn.execute(
            "SELECT f.id, f.url, f.slug, f.title, f.created_at "
            "FROM feeds f WHERE NOT EXISTS (\n"
            "    SELECT 1 FROM episodes e WHERE e.feed_id = f.id\n"
            ")"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "url": r["url"],
                "slug": r["slug"],
                "title": r["title"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def feed_health_summary(self) -> list[dict[str, object]]:
        """
        Return a summary of each feed's episode status.

        Each row contains:
            feed_id, title, total_episodes, done_count, pending_count,
            error_count, health_status (healthy | unhealthy | empty)
        """
        rows = self._conn.execute(
            """
            SELECT
                f.id AS feed_id,
                f.title,
                COALESCE(e.total, 0) AS total_episodes,
                COALESCE(e.done_count, 0) AS done_count,
                COALESCE(e.pending_count, 0) AS pending_count,
                COALESCE(e.error_count, 0) AS error_count
            FROM feeds f
            LEFT JOIN (
                SELECT
                    feed_id,
                    SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS done_count,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_count,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count,
                    COUNT(*) AS total
                FROM episodes
                GROUP BY feed_id
            ) e ON f.id = e.feed_id
            ORDER BY f.title COLLATE NOCASE
            """
        ).fetchall()
        result = []
        for r in rows:
            total = int(r["total_episodes"])
            done = int(r["done_count"])
            pending = int(r["pending_count"])
            errors = int(r["error_count"])
            if total == 0:
                health = "empty"
            elif errors > 0 or pending > 0:
                health = "unhealthy"
            else:
                health = "healthy"
            result.append({
                "feed_id": r["feed_id"],
                "title": r["title"],
                "total_episodes": total,
                "done_count": done,
                "pending_count": pending,
                "error_count": errors,
                "health_status": health,
            })
        return result
