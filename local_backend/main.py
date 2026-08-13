from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from imageio_ffmpeg import get_ffmpeg_exe
from mutagen import File as MutagenFile
from pydantic import BaseModel
from yt_dlp import DownloadError, YoutubeDL

from .analyzer import ANALYZER_VERSION, analyze_file
from .cache_optimizer import audio_media_type, compact_youtube_source, optimize_analysis_cache
from .library_store import LibraryStore

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(os.environ.get("PITCHROOM_APP_ROOT", ROOT)).resolve()
WORK_ROOT = Path(os.environ.get("PITCHROOM_DATA_ROOT", ROOT / ".pitchroom")).resolve()
UPLOAD_ROOT = WORK_ROOT / "uploads"
LIBRARY_FILE = WORK_ROOT / "library.json"
LIBRARY_DATABASE = WORK_ROOT / "library.sqlite3"
LIBRARY_MIGRATION_MARKER = WORK_ROOT / ".library-cache-migrated-v3"
LIBRARY_SQLITE_MARKER = WORK_ROOT / ".library-sqlite-migrated-v1"
MODEL_ROOT = Path(
    os.environ.get("PITCHROOM_MODEL_ROOT", ROOT / "node_modules" / "@spotify" / "basic-pitch" / "model")
).resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PitchRoom Local Engine", version=ANALYZER_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pitchroom-analysis")
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
library_lock = threading.Lock()
cache_optimization_lock = threading.Lock()


def _read_library() -> dict[str, dict]:
    if not LIBRARY_FILE.is_file():
        return {}
    try:
        payload = json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
    return entries if isinstance(entries, dict) else {}


library: dict[str, dict] = _read_library()


def _write_library() -> None:
    LIBRARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = LIBRARY_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "entries": library}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, LIBRARY_FILE)


def _public_library_entry(entry: dict, *, include_notes: bool = False) -> dict:
    public = {key: value for key, value in entry.items() if not key.endswith("_path")}
    if not include_notes:
        public.pop("notes", None)
    return public


def _normalize_library_notes(notes: list[dict]) -> list[dict]:
    if len(notes) > 100_000:
        raise HTTPException(status_code=422, detail="音程データが大きすぎます")
    normalized: list[dict] = []
    for note in notes:
        try:
            time = float(note["time"])
            midi = float(note["midi"])
            duration = float(note.get("duration", 0.1))
            confidence = float(note.get("confidence", 1.0))
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail="音程データの形式が正しくありません") from None
        if not all(math.isfinite(value) for value in (time, midi, duration, confidence)):
            raise HTTPException(status_code=422, detail="音程データに無効な値があります")
        normalized.append({
            "time": max(0.0, time),
            "duration": max(0.01, duration),
            "midi": midi,
            "confidence": max(0.0, min(1.0, confidence)),
            "attack": bool(note.get("attack", False)),
        })
    return normalized


def _save_job_to_library(job_id: str, notes: list[dict] | None = None) -> dict:
    job = jobs.get(job_id)
    if not job or job.get("status") != "completed" or not isinstance(job.get("result"), dict):
        raise HTTPException(status_code=404, detail="保存できる解析結果が見つかりません")
    with cache_optimization_lock:
        result = job["result"]
        accompaniment_path = Path(str(result.get("accompaniment_path", "")))
        if not accompaniment_path.is_file():
            raise HTTPException(status_code=404, detail="解析済み伴奏が見つかりません")
        source_kind = "youtube" if job.get("source") == "youtube" else "file"
        library_id = f"{source_kind}-{accompaniment_path.parent.name}"
        now = datetime.now(timezone.utc).isoformat()
        existing = library.get(library_id, {})
        final_notes = notes if notes is not None else result.get("notes", [])
        if notes is None and not final_notes and existing.get("notes"):
            final_notes = existing["notes"]
        final_notes = final_notes if isinstance(final_notes, list) else []
        if notes is not None:
            result["notes"] = final_notes
            result["predominant_notes"] = []
            result["note_count"] = len(final_notes)
        finalized = result.get("mode") == "vocal" or bool(final_notes)
        try:
            optimized = optimize_analysis_cache(
                result,
                keep_melody_source=result.get("mode") == "instrumental" and not finalized,
            )
            result.clear()
            result.update(optimized["result"])
            job["result"] = result
            if finalized and job.get("source_path"):
                compacted_source, _ = compact_youtube_source(Path(str(job["source_path"])), UPLOAD_ROOT)
                job["source_path"] = str(compacted_source) if compacted_source else ""
        except Exception as error:
            job["optimization_warning"] = str(error)

        accompaniment_path = Path(str(result.get("accompaniment_path", "")))
        vocals_path = Path(str(result.get("vocals_path", ""))) if result.get("vocals_path") else None
        entry = {
            "id": library_id,
            "title": str(result.get("youtube_title") or job.get("filename") or "名称未設定の曲"),
            "source_kind": source_kind,
            "youtube_url": str(job.get("url") or ""),
            "youtube_id": str(result.get("youtube_id") or ""),
            "mode": str(result.get("mode") or "instrumental"),
            "duration": float(result.get("duration") or 0),
            "notes": final_notes,
            "accompaniment_path": str(accompaniment_path.resolve()),
            "guide_path": str(vocals_path.resolve()) if result.get("mode") == "vocal" and vocals_path and vocals_path.is_file() else "",
            "source_path": str(job.get("source_path") or ""),
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        with library_lock:
            library[library_id] = entry
            _write_library()
        library_store.upsert(entry, preserve_activity=False)
        return entry


def _auto_save_job_to_library(job_id: str) -> None:
    try:
        _save_job_to_library(job_id)
    except Exception:
        # Library indexing must never turn a completed audio analysis into a failure.
        return


def _cache_key_for_source(path: Path, analyzer_version: str = ANALYZER_VERSION) -> str:
    digest = hashlib.sha256()
    digest.update(analyzer_version.encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:20]


def _cached_title(path: Path, fallback: str) -> str:
    try:
        tags = MutagenFile(path, easy=True)
        title = tags.get("title", []) if tags is not None else []
        if title and str(title[0]).strip():
            return str(title[0]).strip()
    except Exception:
        pass
    return fallback


def _migrate_cached_library() -> None:
    if LIBRARY_MIGRATION_MARKER.is_file():
        return
    cache_versions = {ANALYZER_VERSION}
    for analysis_path in (WORK_ROOT / "cache").glob("*/analysis.json"):
        try:
            cached = json.loads(analysis_path.read_text(encoding="utf-8"))
            if cached.get("version"):
                cache_versions.add(str(cached["version"]))
        except (OSError, json.JSONDecodeError):
            continue
    raw_candidates = list(UPLOAD_ROOT.glob("*.*")) + list(UPLOAD_ROOT.glob("*/source.wav"))
    candidates: list[Path] = []
    source_signatures: set[tuple[int, str]] = set()
    for source_path in raw_candidates:
        if not source_path.is_file():
            continue
        fallback = "youtube" if source_path.name == "source.wav" else source_path.suffix.lower()
        metadata_source = source_path.with_suffix(".m4a") if source_path.name == "source.wav" else source_path
        signature = (source_path.stat().st_size, _cached_title(metadata_source, fallback))
        if signature in source_signatures:
            continue
        source_signatures.add(signature)
        candidates.append(source_path)
    changed = False
    seen: set[str] = set()
    for source_path in candidates:
        matches: list[tuple[float, str, Path, dict]] = []
        versions = {ANALYZER_VERSION} if source_path.name == "source.wav" else cache_versions
        for version in versions:
            try:
                candidate_key = _cache_key_for_source(source_path, version)
            except OSError:
                break
            candidate_analysis = WORK_ROOT / "cache" / candidate_key / "analysis.json"
            if not candidate_analysis.is_file():
                continue
            try:
                candidate_result = json.loads(candidate_analysis.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            matches.append((candidate_analysis.stat().st_mtime, candidate_key, candidate_analysis, candidate_result))
        if not matches:
            continue
        _, cache_key, analysis_path, result = max(matches, key=lambda match: match[0])
        if cache_key in seen:
            continue
        seen.add(cache_key)
        accompaniment_path = Path(str(result.get("accompaniment_path", "")))
        vocals_path = Path(str(result.get("vocals_path", "")))
        if not accompaniment_path.is_file():
            continue
        source_kind = "youtube" if source_path.name == "source.wav" else "file"
        library_id = f"{source_kind}-{cache_key}"
        timestamp = datetime.fromtimestamp(analysis_path.stat().st_mtime, timezone.utc).isoformat()
        fallback = f"YouTube 解析済み曲 ({timestamp[5:10]} {timestamp[11:16]} UTC)" if source_kind == "youtube" else "解析済み音源"
        if library_id in library:
            if not library[library_id].get("source_path"):
                library[library_id]["source_path"] = str(source_path.resolve())
                changed = True
            if source_kind == "youtube" and str(library[library_id].get("title", "")).startswith("YouTube 解析済み曲"):
                library[library_id]["title"] = fallback
                changed = True
            continue
        metadata_source = source_path.with_suffix(".m4a") if source_kind == "youtube" else source_path
        library[library_id] = {
            "id": library_id,
            "title": _cached_title(metadata_source, fallback),
            "source_kind": source_kind,
            "youtube_url": "",
            "youtube_id": "",
            "mode": str(result.get("mode") or "instrumental"),
            "duration": float(result.get("duration") or 0),
            "notes": result.get("notes", []) if isinstance(result.get("notes"), list) else [],
            "accompaniment_path": str(accompaniment_path.resolve()),
            "guide_path": str(vocals_path.resolve()) if result.get("mode") == "vocal" and vocals_path.is_file() else "",
            "source_path": str(source_path.resolve()),
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        changed = True
    if changed:
        with library_lock:
            _write_library()
    LIBRARY_MIGRATION_MARKER.write_text(ANALYZER_VERSION, encoding="utf-8")


_migrate_cached_library()
library_store = LibraryStore(LIBRARY_DATABASE, WORK_ROOT)
if not LIBRARY_SQLITE_MARKER.is_file():
    library_store.import_entries(list(library.values()))
    LIBRARY_SQLITE_MARKER.write_text("1", encoding="utf-8")


def _optimize_stored_library() -> dict:
    freed_bytes = 0
    optimized_items = 0
    failed_items = 0
    changed_legacy = False
    with cache_optimization_lock:
        for row in library_store.storage_entries():
            cache_dir = Path(str(row.get("accompaniment_path") or "")).parent
            manifest = cache_dir / "analysis.json"
            if not manifest.is_file():
                failed_items += 1
                continue
            try:
                result = json.loads(manifest.read_text(encoding="utf-8"))
                notes = json.loads(str(row.get("notes_json") or "[]"))
                if notes:
                    result["notes"] = notes
                    result["predominant_notes"] = []
                    result["note_count"] = len(notes)
                finalized = row.get("mode") == "vocal" or bool(notes)
                outcome = optimize_analysis_cache(
                    result,
                    keep_melody_source=row.get("mode") == "instrumental" and not finalized,
                )
                updated_result = outcome["result"]
                source_path = Path(str(row.get("source_path") or "")) if row.get("source_path") else None
                source_freed = 0
                if finalized and source_path:
                    compacted_source, source_freed = compact_youtube_source(source_path, UPLOAD_ROOT)
                    source_path = compacted_source
                accompaniment_path = str(updated_result["accompaniment_path"])
                guide_path = str(updated_result.get("vocals_path") or "") if row.get("mode") == "vocal" else ""
                source_value = str(source_path) if source_path else ""
                library_store.update_storage_paths(
                    str(row["id"]),
                    accompaniment_path=accompaniment_path,
                    guide_path=guide_path,
                    source_path=source_value,
                )
                if row["id"] in library:
                    library[row["id"]].update({
                        "accompaniment_path": accompaniment_path,
                        "guide_path": guide_path,
                        "source_path": source_value,
                    })
                    changed_legacy = True
                freed_bytes += int(outcome["freed_bytes"]) + source_freed
                optimized_items += 1
            except Exception:
                failed_items += 1
        if changed_legacy:
            with library_lock:
                _write_library()
    return {"items": optimized_items, "failed_items": failed_items, "freed_bytes": freed_bytes}


def _set_job(job_id: str, **values: object) -> None:
    with jobs_lock:
        jobs[job_id].update(values)


def _run_job(job_id: str, source_path: Path) -> None:
    try:
        _set_job(job_id, status="processing", message="解析を開始しています")
        result = analyze_file(
            source_path,
            WORK_ROOT,
            lambda progress, message: _set_job(job_id, progress=progress, message=message),
        )
        _set_job(job_id, status="completed", progress=100, message="解析が完了しました", result=result)
        _auto_save_job_to_library(job_id)
    except Exception as error:
        _set_job(job_id, status="failed", message=f"解析に失敗しました: {error}", error=repr(error))


class YouTubeJobRequest(BaseModel):
    url: str


class LibrarySaveRequest(BaseModel):
    job_id: str
    notes: list[dict]


def _is_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme in {"http", "https"} and (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    )


def _youtube_match_filter(info: dict, *, incomplete: bool) -> str | None:
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
        return "ライブ配信は解析できません"
    duration = info.get("duration")
    if duration and duration > 3 * 60 * 60:
        return "3時間を超える動画は解析できません"
    return None


def _download_youtube_audio(
    url: str,
    download_root: Path,
    progress_hook,
    retry_callback=lambda _attempt: None,
) -> tuple[dict, Path]:
    """Download audio, retrying with independent YouTube player clients.

    YouTube changes which clients require proof-of-origin tokens over time.
    The current yt-dlp default is preferred, with explicit clients retained as
    fallbacks so one stale/blocked media URL does not fail the whole job.
    """
    node_path = os.environ.get("PITCHROOM_NODE_PATH") or shutil.which("node")
    attempts: list[tuple[str, str | None]] = [
        ("default", None),
        ("android_vr", "android_vr"),
        ("web_embedded", "web_embedded"),
    ]
    errors: list[str] = []
    for attempt_index, (label, player_client) in enumerate(attempts):
        if attempt_index:
            retry_callback(attempt_index + 1)
            for partial in download_root.glob("source.*"):
                if partial.is_file():
                    partial.unlink(missing_ok=True)
        options = {
            "format": "m4a/bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": str(download_root / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "overwrites": True,
            "socket_timeout": 20,
            "retries": 4,
            "extractor_retries": 3,
            "fragment_retries": 4,
            "file_access_retries": 3,
            "concurrent_fragment_downloads": 4,
            "max_filesize": 500 * 1024 * 1024,
            "match_filter": _youtube_match_filter,
            "progress_hooks": [progress_hook],
        }
        if node_path:
            options["js_runtimes"] = {"node": {"path": node_path}}
        if player_client:
            options["extractor_args"] = {"youtube": {"player_client": [player_client]}}
        try:
            with YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True)
                if info is None:
                    raise RuntimeError("動画情報を取得できませんでした")
                source_path = Path(downloader.prepare_filename(info))
            if not source_path.exists():
                candidates = [path for path in download_root.glob("source.*") if path.is_file()]
                if not candidates:
                    raise RuntimeError("解析用音声を保存できませんでした")
                source_path = max(candidates, key=lambda path: path.stat().st_size)
            return info, source_path
        except DownloadError as error:
            errors.append(f"{label}: {error}")
    raise RuntimeError("YouTubeから音声を取得できませんでした: " + " / ".join(errors))


def _run_youtube_job(job_id: str, url: str) -> None:
    download_root = UPLOAD_ROOT / job_id
    download_root.mkdir(parents=True, exist_ok=True)

    def progress_hook(progress: dict) -> None:
        if progress.get("status") != "downloading":
            return
        downloaded = float(progress.get("downloaded_bytes") or 0)
        total = float(progress.get("total_bytes") or progress.get("total_bytes_estimate") or 0)
        percent = downloaded / total if total > 0 else 0
        _set_job(
            job_id,
            progress=max(1, min(14, round(percent * 14))),
            message="YouTubeから解析用音声を取得しています",
        )

    try:
        _set_job(job_id, status="processing", progress=1, message="YouTube動画を確認しています")
        info, source_path = _download_youtube_audio(
            url,
            download_root,
            progress_hook,
            lambda attempt: _set_job(
                job_id,
                progress=1,
                message=f"YouTube取得経路を切り替えています ({attempt}/3)",
            ),
        )

        if source_path.stat().st_size > 500 * 1024 * 1024:
            raise RuntimeError("500MBを超える動画は解析できません")

        _set_job(job_id, progress=15, message="取得した音声を解析形式へ変換しています")
        converted_path = download_root / "source.wav"
        subprocess.run(
            [get_ffmpeg_exe(), "-y", "-i", str(source_path), "-vn", "-ac", "2", "-ar", "44100", str(converted_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        source_path = converted_path
        _set_job(job_id, source_path=str(source_path.resolve()))

        _set_job(job_id, progress=17, message="音声を取得しました。主旋律を解析します")
        result = analyze_file(
            source_path,
            WORK_ROOT,
            lambda progress, message: _set_job(
                job_id,
                progress=17 + round(progress * 0.83),
                message=message,
            ),
        )
        result["youtube_id"] = str(info.get("id") or "")
        result["youtube_title"] = str(info.get("title") or "YouTube セッション")
        _set_job(
            job_id,
            status="completed",
            progress=100,
            message="YouTube動画の解析が完了しました",
            result=result,
        )
        _auto_save_job_to_library(job_id)
    except Exception as error:
        _set_job(
            job_id,
            status="failed",
            message=f"YouTube動画の解析に失敗しました: {error}",
            error=repr(error),
        )


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine": "demucs+crepe+muscriptor+lead-graph+high-resolution-amt+melodia",
        "analyzer_version": ANALYZER_VERSION,
        "local_only": True,
    }


@app.post("/api/jobs")
def create_job(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}:
        raise HTTPException(status_code=415, detail="対応していない音源形式です")
    job_id = uuid.uuid4().hex
    source_path = UPLOAD_ROOT / f"{job_id}{suffix}"
    with source_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "status": "queued",
        "progress": 0,
        "message": "解析待ちです",
        "source_path": str(source_path.resolve()),
        "result": None,
    }
    executor.submit(_run_job, job_id, source_path)
    return {"id": job_id, "status": "queued"}


@app.post("/api/youtube-jobs")
def create_youtube_job(request: YouTubeJobRequest) -> dict:
    url = request.url.strip()
    if not _is_youtube_url(url):
        raise HTTPException(status_code=422, detail="有効なYouTube URLを入力してください")
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "id": job_id,
        "filename": None,
        "source": "youtube",
        "url": url,
        "status": "queued",
        "progress": 0,
        "message": "YouTube音声の取得待ちです",
        "source_path": "",
        "result": None,
    }
    executor.submit(_run_youtube_job, job_id, url)
    return {"id": job_id, "status": "queued"}


@app.get("/api/library")
def list_library(
    q: str = Query(default="", max_length=200),
    source: str = Query(default="all"),
    sort: str = Query(default="recent"),
    trashed: bool = Query(default=False),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=50),
) -> dict:
    if source not in {"all", "file", "youtube"}:
        raise HTTPException(status_code=422, detail="音源フィルターが正しくありません")
    if sort not in {"recent", "title", "updated", "size"}:
        raise HTTPException(status_code=422, detail="並べ替え方法が正しくありません")
    return library_store.list(search=q, source=source, sort=sort, trashed=trashed, offset=offset, limit=limit)


@app.post("/api/library")
def save_library_entry(request: LibrarySaveRequest) -> dict:
    entry = _save_job_to_library(request.job_id.strip(), _normalize_library_notes(request.notes))
    return library_store.get(entry["id"], mark_played=False)


@app.get("/api/library-stats")
def get_library_stats() -> dict:
    return library_store.stats()


@app.get("/api/library-cleanup")
def preview_library_cleanup() -> dict:
    return library_store.cleanup_preview()


@app.post("/api/library-cleanup")
def cleanup_library_cache() -> dict:
    if any(job.get("status") in {"queued", "processing"} for job in jobs.values()):
        raise HTTPException(status_code=409, detail="解析中は不要データを整理できません")
    return library_store.cleanup_unreferenced()


@app.get("/api/library-optimize")
def preview_library_optimization() -> dict:
    return library_store.optimization_preview()


@app.post("/api/library-optimize")
def optimize_library_storage() -> dict:
    if any(job.get("status") in {"queued", "processing"} for job in jobs.values()):
        raise HTTPException(status_code=409, detail="解析中は保存曲を圧縮できません")
    return _optimize_stored_library()


@app.post("/api/library/{library_id}/trash")
def trash_library_entry(library_id: str) -> dict:
    try:
        return library_store.trash(library_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="解析済み曲が見つかりません") from None


@app.post("/api/library/{library_id}/restore")
def restore_library_entry(library_id: str) -> dict:
    try:
        return library_store.restore(library_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="ゴミ箱の曲が見つかりません") from None


@app.delete("/api/library/{library_id}")
def permanently_delete_library_entry(library_id: str) -> dict:
    try:
        return library_store.permanently_delete(library_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="完全削除できる曲が見つかりません") from None


@app.get("/api/library/{library_id}")
def get_library_entry(library_id: str) -> dict:
    try:
        return library_store.get(library_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="解析済み曲が見つかりません") from None


@app.get("/api/library/{library_id}/audio/{stem}")
def get_library_audio(library_id: str, stem: str) -> FileResponse:
    if stem not in {"accompaniment", "guide"}:
        raise HTTPException(status_code=404, detail="保存済みステムが見つかりません")
    try:
        entry = library_store.get_paths(library_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="解析済み曲が見つかりません") from None
    path_value = entry.get("accompaniment_path" if stem == "accompaniment" else "guide_path", "")
    path = Path(str(path_value))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="保存済み音源が見つかりません")
    return FileResponse(path, media_type=audio_media_type(path), filename=path.name)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="解析ジョブが見つかりません")
    public_job = dict(job)
    result = public_job.get("result")
    if isinstance(result, dict):
        public_result = dict(result)
        public_result.pop("accompaniment_path", None)
        public_result.pop("vocals_path", None)
        public_result.pop("melody_source_path", None)
        public_job["result"] = public_result
    public_job.pop("error", None)
    return public_job


@app.get("/api/jobs/{job_id}/audio/{stem}")
def get_audio(job_id: str, stem: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="音源がまだ準備できていません")
    if stem not in {"accompaniment", "vocals", "melody-source"}:
        raise HTTPException(status_code=404, detail="ステムが見つかりません")
    result = job["result"]
    result_key = "melody_source_path" if stem == "melody-source" else f"{stem}_path"
    path = Path(str(result.get(result_key) or ""))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ステムが見つかりません")
    return FileResponse(path, media_type=audio_media_type(path), filename=path.name)


@app.get("/api/models/basic-pitch/{filename}")
def get_basic_pitch_model(filename: str) -> FileResponse:
    if filename not in {"model.json", "group1-shard1of1.bin"}:
        raise HTTPException(status_code=404, detail="モデルファイルが見つかりません")
    path = MODEL_ROOT / filename
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Basic Pitchモデルがインストールされていません")
    media_type = "application/json" if filename.endswith(".json") else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)
