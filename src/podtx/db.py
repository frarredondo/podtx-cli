from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from podtx.models import Feed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_output_paths(paths: list[Path]) -> str:
    """Record output paths absolutely: readers run from a different cwd."""
    return json.dumps([str(Path(p).expanduser().resolve()) for p in paths])


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
            CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                feed_slug,
                guid UNINDEXED,
                title,
                text,
                published_at UNINDEXED,
                txt_path UNINDEXED,
                json_path UNINDEXED,
                tokenize='porter'
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
                _encode_output_paths(output_paths),
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
                _encode_output_paths(output_paths),
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

    # ─── Health-check queries behind `podtx doctor` / `podtx feeds` ─────────

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

    def last_transcribed_at(self, feed_id: int) -> str | None:
        """Return the most recent transcribed_at timestamp for a feed, if any."""
        row = self._conn.execute(
            "SELECT MAX(transcribed_at) AS m FROM episodes WHERE feed_id = ? AND status = 'done'",
            (feed_id,),
        ).fetchone()
        if row is None:  # pragma: no cover - MAX query always returns a row
            return None
        return row["m"] if row["m"] else None

    # ─── Health-check queries behind `podtx doctor` ────────────────────────

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

    # ─── Search (FTS5) ─────────────────────────────────────────────────

    def upsert_search_entry(
        self,
        *,
        feed_slug: str,
        guid: str,
        title: str,
        published_at: str | None,
        text: str,
        txt_path: str,
        json_path: str,
    ) -> None:
        """Insert or replace a transcript in the FTS index."""
        # FTS5 has no REPLACE; delete then insert.
        self._conn.execute(
            "DELETE FROM search_fts WHERE guid = ? AND feed_slug = ?",
            (guid, feed_slug),
        )
        self._conn.execute(
            "INSERT INTO search_fts (feed_slug, guid, title, text, published_at, txt_path, json_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (feed_slug, guid, title, text or "", published_at, txt_path, json_path),
        )
        self._conn.commit()

    def search_transcripts(
        self,
        query: str,
        feed: str | None = None,
        limit: int | None = 10,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, object]]:
        """Full-text search over indexed transcripts.

        Returns dicts with feed_slug, guid, title, published_at, text,
        txt_path, json_path, snippet, rank.
        """
        if not query or not query.strip():
            return []
        q = query.strip()
        # Build query: FTS5 MATCH + optional filters.
        sql = (
            "SELECT feed_slug, guid, title, published_at, text, txt_path, json_path, "
            "snippet(search_fts, 3, '', '', ' … ', 24) AS snippet, rank "
            "FROM search_fts WHERE search_fts MATCH ?"
        )
        params: list[object] = [q]
        if feed:
            sql += " AND feed_slug = ?"
            params.append(feed)
        if since:
            sql += " AND published_at IS NOT NULL AND published_at >= ?"
            params.append(since)
        if until:
            sql += " AND published_at IS NOT NULL AND published_at <= ?"
            params.append(until)
        sql += " ORDER BY rank LIMIT ?"
        params.append(int(limit) if limit is not None else 10)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:  # pragma: no cover - invalid FTS syntax fallback
            # Invalid FTS5 syntax (e.g. unmatched quotes) -> try escaped phrase
            # Fallback to simple token search.
            try:
                safe = '"' + q.replace('"', '""') + '"'
                params2: list[object] = [safe]
                sql2 = (
                    "SELECT feed_slug, guid, title, published_at, text, txt_path, json_path, "
                    "snippet(search_fts, 3, '', '', ' … ', 24) AS snippet, rank "
                    "FROM search_fts WHERE search_fts MATCH ?"
                )
                if feed:
                    sql2 += " AND feed_slug = ?"
                    params2.append(feed)
                if since:
                    sql2 += " AND published_at IS NOT NULL AND published_at >= ?"
                    params2.append(since)
                if until:
                    sql2 += " AND published_at IS NOT NULL AND published_at <= ?"
                    params2.append(until)
                sql2 += " ORDER BY rank LIMIT ?"
                params2.append(int(limit) if limit is not None else 10)
                rows = self._conn.execute(sql2, params2).fetchall()
            except sqlite3.OperationalError:  # pragma: no cover - second fallback
                return []
        out: list[dict[str, object]] = []
        for r in rows:
            out.append({
                "feed_slug": r["feed_slug"],
                "guid": r["guid"],
                "title": r["title"],
                "published_at": r["published_at"],
                "text": r["text"],
                "txt_path": r["txt_path"],
                "json_path": r["json_path"],
                "snippet": r["snippet"] if r["snippet"] else (r["text"][:160] if r["text"] else ""),
                "rank": r["rank"],
            })
        return out

    def reindex_search(self, transcripts_root: Path) -> int:
        """Rebuild FTS index from transcript JSON files on disk."""
        root = Path(transcripts_root).expanduser()
        if not root.is_dir():
            return 0
        count = 0
        for json_path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            feed_slug = json_path.parent.name
            guid = str(payload.get("guid") or json_path.stem)
            title = str(payload.get("title") or json_path.stem)
            published_at = payload.get("date")
            if published_at is not None:
                published_at = str(published_at)
            text = str(payload.get("text") or "").strip()
            if not text and payload.get("segments"):
                try:
                    text = " ".join(str(s.get("text", "")).strip() for s in payload.get("segments") or [] if s.get("text"))
                except Exception:
                    text = ""
            txt_path = str(json_path.with_suffix(".txt"))
            # Prefer absolute paths for CLI output stability
            try:
                txt_p = str(json_path.with_suffix(".txt").resolve())
                json_p = str(json_path.resolve())
            except Exception:
                txt_p = txt_path
                json_p = str(json_path)
            self.upsert_search_entry(
                feed_slug=feed_slug,
                guid=guid,
                title=title,
                published_at=published_at,
                text=text,
                txt_path=txt_p,
                json_path=json_p,
            )
            count += 1
        return count
