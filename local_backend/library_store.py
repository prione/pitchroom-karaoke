from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0
    return 0


class LibraryStore:
    def __init__(self, database_path: Path, work_root: Path) -> None:
        self.database_path = database_path
        self.work_root = work_root.resolve()
        self.cache_root = (work_root / "cache").resolve()
        self.upload_root = (work_root / "uploads").resolve()
        self.lock = Lock()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS library_entries (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_kind TEXT NOT NULL CHECK(source_kind IN ('file', 'youtube')),
                    youtube_url TEXT NOT NULL DEFAULT '',
                    youtube_id TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL CHECK(mode IN ('vocal', 'instrumental')),
                    duration REAL NOT NULL DEFAULT 0,
                    notes_json TEXT NOT NULL DEFAULT '[]',
                    accompaniment_path TEXT NOT NULL,
                    guide_path TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT '',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_played_at TEXT,
                    trashed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_library_active_recent
                    ON library_entries(trashed_at, last_played_at, updated_at);
                CREATE INDEX IF NOT EXISTS idx_library_title
                    ON library_entries(title COLLATE NOCASE);
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(library_entries)")}
            if "size_bytes" not in columns:
                connection.execute("ALTER TABLE library_entries ADD COLUMN size_bytes INTEGER NOT NULL DEFAULT 0")

    def import_entries(self, entries: list[dict]) -> None:
        for entry in entries:
            if not entry.get("id") or not entry.get("accompaniment_path"):
                continue
            self.upsert(entry, preserve_activity=True)

    def upsert(self, entry: dict, *, preserve_activity: bool = True) -> dict:
        created_at = str(entry.get("created_at") or _now())
        updated_at = str(entry.get("updated_at") or _now())
        notes_json = json.dumps(entry.get("notes") or [], ensure_ascii=False, separators=(",", ":"))
        accompaniment_path = Path(str(entry.get("accompaniment_path") or ""))
        source_path = Path(str(entry.get("source_path") or "")) if entry.get("source_path") else None
        size_bytes = _path_size(accompaniment_path.parent)
        if source_path is not None and source_path.parent != accompaniment_path.parent:
            size_bytes += _path_size(source_path.parent if source_path.name == "source.wav" else source_path)
        with self.lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at, last_played_at, trashed_at FROM library_entries WHERE id = ?",
                (entry["id"],),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO library_entries (
                    id, title, source_kind, youtube_url, youtube_id, mode, duration,
                    notes_json, accompaniment_path, guide_path, source_path, size_bytes,
                    created_at, updated_at, last_played_at, trashed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    source_kind=excluded.source_kind,
                    youtube_url=CASE WHEN excluded.youtube_url <> '' THEN excluded.youtube_url ELSE library_entries.youtube_url END,
                    youtube_id=CASE WHEN excluded.youtube_id <> '' THEN excluded.youtube_id ELSE library_entries.youtube_id END,
                    mode=excluded.mode,
                    duration=excluded.duration,
                    notes_json=CASE WHEN excluded.notes_json <> '[]' THEN excluded.notes_json ELSE library_entries.notes_json END,
                    accompaniment_path=excluded.accompaniment_path,
                    guide_path=excluded.guide_path,
                    source_path=CASE WHEN excluded.source_path <> '' THEN excluded.source_path ELSE library_entries.source_path END,
                    size_bytes=excluded.size_bytes,
                    updated_at=excluded.updated_at
                """,
                (
                    str(entry["id"]), str(entry.get("title") or "名称未設定の曲"),
                    str(entry.get("source_kind") or "file"), str(entry.get("youtube_url") or ""),
                    str(entry.get("youtube_id") or ""), str(entry.get("mode") or "instrumental"),
                    float(entry.get("duration") or 0), notes_json,
                    str(entry.get("accompaniment_path") or ""), str(entry.get("guide_path") or ""),
                    str(entry.get("source_path") or ""), size_bytes,
                    str(existing["created_at"] if existing and preserve_activity else created_at), updated_at,
                    existing["last_played_at"] if existing and preserve_activity else entry.get("last_played_at"),
                    existing["trashed_at"] if existing and preserve_activity else entry.get("trashed_at"),
                ),
            )
        return self.get(str(entry["id"]), mark_played=False, allow_trashed=True)

    def _row_to_public(self, row: sqlite3.Row, *, include_notes: bool = False) -> dict:
        public = {
            "id": row["id"], "title": row["title"], "source_kind": row["source_kind"],
            "youtube_url": row["youtube_url"], "youtube_id": row["youtube_id"],
            "mode": row["mode"], "duration": row["duration"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "last_played_at": row["last_played_at"], "trashed_at": row["trashed_at"],
            "size_bytes": int(row["size_bytes"]),
        }
        if include_notes:
            try:
                public["notes"] = json.loads(row["notes_json"])
            except json.JSONDecodeError:
                public["notes"] = []
        return public

    def list(self, *, search: str = "", source: str = "all", sort: str = "recent", trashed: bool = False, offset: int = 0, limit: int = 50) -> dict:
        where = ["trashed_at IS NOT NULL" if trashed else "trashed_at IS NULL"]
        params: list[object] = []
        if search.strip():
            where.append("title LIKE ? ESCAPE '\\' COLLATE NOCASE")
            escaped = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        if source in {"file", "youtube"}:
            where.append("source_kind = ?")
            params.append(source)
        order = {
            "title": "title COLLATE NOCASE ASC, updated_at DESC",
            "updated": "updated_at DESC",
            "size": "size_bytes DESC, updated_at DESC",
            "recent": "COALESCE(last_played_at, updated_at) DESC",
        }.get(sort, "COALESCE(last_played_at, updated_at) DESC")
        clause = " AND ".join(where)
        limit = max(1, min(50, limit))
        offset = max(0, offset)
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM library_entries WHERE {clause}", params).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM library_entries WHERE {clause} ORDER BY {order} LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        entries = [self._row_to_public(row) for row in rows]
        return {"entries": entries, "total": total, "offset": offset, "limit": limit, "has_more": offset + len(entries) < total}

    def get(self, entry_id: str, *, mark_played: bool = True, allow_trashed: bool = False) -> dict:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM library_entries WHERE id = ?", (entry_id,)).fetchone()
            if row is None or (row["trashed_at"] and not allow_trashed):
                raise KeyError(entry_id)
            if mark_played:
                connection.execute("UPDATE library_entries SET last_played_at = ? WHERE id = ?", (_now(), entry_id))
                row = connection.execute("SELECT * FROM library_entries WHERE id = ?", (entry_id,)).fetchone()
        return self._row_to_public(row, include_notes=True)

    def get_paths(self, entry_id: str, *, allow_trashed: bool = False) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM library_entries WHERE id = ?", (entry_id,)).fetchone()
        if row is None or (row["trashed_at"] and not allow_trashed):
            raise KeyError(entry_id)
        return dict(row)

    def storage_entries(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM library_entries ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def update_storage_paths(
        self,
        entry_id: str,
        *,
        accompaniment_path: str,
        guide_path: str,
        source_path: str,
    ) -> dict:
        accompaniment = Path(accompaniment_path)
        source = Path(source_path) if source_path else None
        size_bytes = _path_size(accompaniment.parent)
        if source is not None and source.parent != accompaniment.parent:
            size_bytes += _path_size(source.parent if source.name == "source.wav" else source)
        with self.lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE library_entries
                SET accompaniment_path = ?, guide_path = ?, source_path = ?, size_bytes = ?, updated_at = ?
                WHERE id = ?
                """,
                (accompaniment_path, guide_path, source_path, size_bytes, _now(), entry_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(entry_id)
        return self.get(entry_id, mark_played=False, allow_trashed=True)

    def optimization_preview(self) -> dict:
        entries = self.storage_entries()
        candidates = 0
        current_bytes = 0
        for entry in entries:
            accompaniment = Path(str(entry.get("accompaniment_path") or ""))
            cache_dir = accompaniment.parent
            needs_optimization = accompaniment.suffix.lower() == ".wav"
            needs_optimization = needs_optimization or any((cache_dir / name).is_file() for name in {
                "attack-source.wav", "continuous-source.wav", "melody-source.wav",
            })
            guide = Path(str(entry.get("guide_path") or "")) if entry.get("guide_path") else None
            needs_optimization = needs_optimization or bool(guide and guide.suffix.lower() == ".wav")
            if needs_optimization and accompaniment.is_file():
                candidates += 1
                current_bytes += _path_size(cache_dir)
        return {"items": candidates, "current_bytes": current_bytes}

    def trash(self, entry_id: str) -> dict:
        with self.lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE library_entries SET trashed_at = ?, updated_at = ? WHERE id = ? AND trashed_at IS NULL",
                (_now(), _now(), entry_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(entry_id)
        return self.get(entry_id, mark_played=False, allow_trashed=True)

    def restore(self, entry_id: str) -> dict:
        with self.lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE library_entries SET trashed_at = NULL, updated_at = ? WHERE id = ? AND trashed_at IS NOT NULL",
                (_now(), entry_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(entry_id)
        return self.get(entry_id, mark_played=False)

    def _safe_remove(self, path: Path, root: Path, *, directory: bool) -> int:
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            return 0
        if resolved == root or not resolved.exists():
            return 0
        size = _path_size(resolved)
        if directory and resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.is_file():
            resolved.unlink()
        return size

    def permanently_delete(self, entry_id: str) -> dict:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM library_entries WHERE id = ? AND trashed_at IS NOT NULL", (entry_id,)).fetchone()
            if row is None:
                raise KeyError(entry_id)
            connection.execute("DELETE FROM library_entries WHERE id = ?", (entry_id,))
            cache_refs = int(connection.execute("SELECT COUNT(*) FROM library_entries WHERE accompaniment_path = ?", (row["accompaniment_path"],)).fetchone()[0])
            source_refs = int(connection.execute("SELECT COUNT(*) FROM library_entries WHERE source_path = ? AND source_path <> ''", (row["source_path"],)).fetchone()[0])
        freed = 0
        cache_dir = Path(row["accompaniment_path"]).parent
        if cache_refs == 0:
            freed += self._safe_remove(cache_dir, self.cache_root, directory=True)
        if row["source_path"] and source_refs == 0:
            source_path = Path(row["source_path"])
            if source_path.name == "source.wav" and source_path.parent.parent.resolve() == self.upload_root:
                freed += self._safe_remove(source_path.parent, self.upload_root, directory=True)
            else:
                freed += self._safe_remove(source_path, self.upload_root, directory=False)
        return {"id": entry_id, "freed_bytes": freed}

    def stats(self) -> dict:
        with self._connect() as connection:
            active = int(connection.execute("SELECT COUNT(*) FROM library_entries WHERE trashed_at IS NULL").fetchone()[0])
            trashed = int(connection.execute("SELECT COUNT(*) FROM library_entries WHERE trashed_at IS NOT NULL").fetchone()[0])
        return {"active": active, "trashed": trashed, "cache_bytes": _path_size(self.cache_root), "upload_bytes": _path_size(self.upload_root)}

    def _unreferenced_paths(self) -> list[Path]:
        with self._connect() as connection:
            rows = connection.execute("SELECT accompaniment_path, source_path FROM library_entries").fetchall()
        protected_cache = {Path(row["accompaniment_path"]).parent.resolve() for row in rows if row["accompaniment_path"]}
        protected_uploads: set[Path] = set()
        for row in rows:
            if not row["source_path"]:
                continue
            source = Path(row["source_path"]).resolve()
            try:
                relative = source.relative_to(self.upload_root)
                protected_uploads.add(
                    (self.upload_root / relative.parts[0]).resolve()
                    if len(relative.parts) > 1
                    else source
                )
            except ValueError:
                continue
        candidates: list[Path] = []
        if self.cache_root.is_dir():
            candidates.extend(path for path in self.cache_root.iterdir() if path.resolve() not in protected_cache)
        if self.upload_root.is_dir():
            candidates.extend(path for path in self.upload_root.iterdir() if path.resolve() not in protected_uploads)
        return candidates

    def cleanup_preview(self) -> dict:
        paths = self._unreferenced_paths()
        return {"items": len(paths), "reclaimable_bytes": sum(_path_size(path) for path in paths)}

    def cleanup_unreferenced(self) -> dict:
        with self.lock:
            paths = self._unreferenced_paths()
            freed = 0
            for path in paths:
                root = self.cache_root if path.parent.resolve() == self.cache_root else self.upload_root
                freed += self._safe_remove(path, root, directory=path.is_dir())
        return {"items": len(paths), "freed_bytes": freed}
