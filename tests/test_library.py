from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_backend import main
from local_backend.library_store import LibraryStore


class LibraryTests(unittest.TestCase):
    def test_saves_a_completed_job_without_exposing_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accompaniment = root / "accompaniment.wav"
            vocals = root / "vocals.wav"
            accompaniment.write_bytes(b"RIFF-audio")
            vocals.write_bytes(b"RIFF-guide")
            test_jobs = {
                "job-1": {
                    "status": "completed",
                    "filename": "song.mp3",
                    "result": {
                        "mode": "vocal",
                        "duration": 123.4,
                        "notes": [],
                        "accompaniment_path": str(accompaniment),
                        "vocals_path": str(vocals),
                    },
                },
            }
            library_file = root / "library.json"
            store = LibraryStore(root / "library.sqlite3", root)
            with (
                patch.object(main, "jobs", test_jobs),
                patch.object(main, "library", {}),
                patch.object(main, "LIBRARY_FILE", library_file),
                patch.object(main, "library_store", store),
            ):
                notes = [{"time": 1.0, "duration": 0.2, "midi": 69, "confidence": 0.9}]
                entry = main._save_job_to_library("job-1", main._normalize_library_notes(notes))
                public = main._public_library_entry(entry, include_notes=True)

            self.assertEqual(public["title"], "song.mp3")
            self.assertEqual(public["notes"][0]["midi"], 69.0)
            self.assertNotIn("accompaniment_path", public)
            self.assertNotIn("guide_path", public)
            saved = json.loads(library_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["entries"][entry["id"]]["source_kind"], "file")

    def test_rejects_non_finite_pitch_values(self) -> None:
        with self.assertRaises(main.HTTPException):
            main._normalize_library_notes([{"time": 0, "midi": float("nan")}])

    def test_search_trash_restore_and_shared_cache_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache" / "shared"
            uploads = root / "uploads"
            cache.mkdir(parents=True)
            uploads.mkdir(parents=True)
            accompaniment = cache / "accompaniment.wav"
            accompaniment.write_bytes(b"audio" * 100)
            source = uploads / "source.mp3"
            source.write_bytes(b"source" * 100)
            store = LibraryStore(root / "library.sqlite3", root)
            base = {
                "title": "Searchable Song", "source_kind": "file", "mode": "instrumental",
                "duration": 120, "notes": [{"time": 0, "midi": 60}],
                "accompaniment_path": str(accompaniment), "guide_path": "",
                "source_path": str(source), "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
            store.upsert({**base, "id": "one"})
            store.upsert({**base, "id": "two", "title": "Second Song"})

            result = store.list(search="Searchable", limit=1)
            self.assertEqual(result["total"], 1)
            store.trash("one")
            self.assertEqual(store.stats()["trashed"], 1)
            store.restore("one")
            self.assertEqual(store.stats()["active"], 2)
            store.trash("one")
            store.permanently_delete("one")
            self.assertTrue(cache.exists(), "shared cache must remain while another entry references it")
            self.assertTrue(source.exists(), "shared source must remain while another entry references it")
            store.trash("two")
            deleted = store.permanently_delete("two")
            self.assertFalse(cache.exists())
            self.assertFalse(source.exists())
            self.assertGreater(deleted["freed_bytes"], 0)

    def test_cleanup_only_removes_unreferenced_cache_and_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            referenced_cache = root / "cache" / "referenced"
            stale_cache = root / "cache" / "stale"
            referenced_upload = root / "uploads" / "song.mp3"
            stale_upload = root / "uploads" / "old.mp3"
            referenced_cache.mkdir(parents=True)
            stale_cache.mkdir(parents=True)
            referenced_upload.parent.mkdir(parents=True)
            accompaniment = referenced_cache / "accompaniment.wav"
            accompaniment.write_bytes(b"kept-cache")
            (stale_cache / "old.wav").write_bytes(b"stale-cache")
            referenced_upload.write_bytes(b"kept-upload")
            stale_upload.write_bytes(b"stale-upload")
            store = LibraryStore(root / "library.sqlite3", root)
            store.upsert({
                "id": "kept", "title": "Kept", "source_kind": "file", "mode": "instrumental",
                "duration": 30, "notes": [], "accompaniment_path": str(accompaniment),
                "guide_path": "", "source_path": str(referenced_upload),
                "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
            })

            preview = store.cleanup_preview()
            self.assertEqual(preview["items"], 2)
            self.assertGreater(preview["reclaimable_bytes"], 0)
            result = store.cleanup_unreferenced()
            self.assertEqual(result["items"], 2)
            self.assertTrue(referenced_cache.exists())
            self.assertTrue(referenced_upload.exists())
            self.assertFalse(stale_cache.exists())
            self.assertFalse(stale_upload.exists())

    def test_cleanup_preserves_nested_youtube_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache" / "youtube"
            upload_job = root / "uploads" / "job-id"
            stale_job = root / "uploads" / "stale-job"
            cache.mkdir(parents=True)
            upload_job.mkdir(parents=True)
            stale_job.mkdir(parents=True)
            accompaniment = cache / "accompaniment.flac"
            source = upload_job / "source.m4a"
            accompaniment.write_bytes(b"flac")
            source.write_bytes(b"compressed-source")
            (stale_job / "source.m4a").write_bytes(b"stale")
            store = LibraryStore(root / "library.sqlite3", root)
            store.upsert({
                "id": "youtube", "title": "YouTube", "source_kind": "youtube", "mode": "vocal",
                "duration": 10, "notes": [], "accompaniment_path": str(accompaniment),
                "guide_path": "", "source_path": str(source),
                "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
            })

            preview = store.cleanup_preview()
            self.assertEqual(preview["items"], 1)
            store.cleanup_unreferenced()
            self.assertTrue(upload_job.exists())
            self.assertTrue(source.exists())
            self.assertFalse(stale_job.exists())


if __name__ == "__main__":
    unittest.main()
