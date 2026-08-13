from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe


INTERMEDIATE_AUDIO = {
    "attack-source.wav",
    "continuous-source.wav",
    "melody-source.wav",
}


def _path_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _convert_wav_to_flac(source: Path) -> Path:
    if source.suffix.lower() != ".wav" or not source.is_file():
        return source
    target = source.with_suffix(".flac")
    temporary = target.with_name(f"{target.stem}.optimizing.flac")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                get_ffmpeg_exe(), "-y", "-i", str(source), "-map_metadata", "-1",
                "-c:a", "flac", "-compression_level", "8", str(temporary),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"FLAC conversion produced no output: {source.name}")
        os.replace(temporary, target)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def optimize_analysis_cache(result: dict, *, keep_melody_source: bool = False) -> dict:
    """Losslessly compact a completed analysis and return its updated result.

    Original WAV files are retained until both FLAC files and the updated
    analysis manifest have been written successfully.
    """
    accompaniment = Path(str(result.get("accompaniment_path") or ""))
    if not accompaniment.is_file():
        raise FileNotFoundError("解析済み伴奏が見つかりません")
    cache_dir = accompaniment.parent.resolve()
    before_bytes = _path_size(cache_dir)
    updated = dict(result)
    converted_sources: list[Path] = []

    optimized_accompaniment = _convert_wav_to_flac(accompaniment)
    updated["accompaniment_path"] = str(optimized_accompaniment.resolve())
    if optimized_accompaniment != accompaniment:
        converted_sources.append(accompaniment)

    vocals = Path(str(result.get("vocals_path") or "")) if result.get("vocals_path") else None
    if result.get("mode") == "vocal" and vocals and vocals.is_file():
        optimized_vocals = _convert_wav_to_flac(vocals)
        updated["vocals_path"] = str(optimized_vocals.resolve())
        if optimized_vocals != vocals:
            converted_sources.append(vocals)
    elif result.get("mode") != "vocal":
        updated["vocals_path"] = ""

    melody_source = Path(str(result.get("melody_source_path") or "")) if result.get("melody_source_path") else None
    if not keep_melody_source:
        updated["melody_source_path"] = ""

    manifest = cache_dir / "analysis.json"
    temporary_manifest = cache_dir / "analysis.optimizing.json"
    temporary_manifest.write_text(json.dumps(updated, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_manifest, manifest)

    for source in converted_sources:
        if _inside(source, cache_dir):
            source.unlink(missing_ok=True)
    for filename in INTERMEDIATE_AUDIO:
        path = cache_dir / filename
        if keep_melody_source and melody_source and path.resolve() == melody_source.resolve():
            continue
        path.unlink(missing_ok=True)
    if result.get("mode") != "vocal" and vocals and _inside(vocals, cache_dir):
        vocals.unlink(missing_ok=True)

    after_bytes = _path_size(cache_dir)
    return {
        "result": updated,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "freed_bytes": max(0, before_bytes - after_bytes),
    }


def compact_youtube_source(source_path: Path, upload_root: Path) -> tuple[Path | None, int]:
    """Drop a derived source.wav while retaining the original compressed download."""
    if source_path.name != "source.wav" or not _inside(source_path, upload_root):
        return (source_path if source_path.is_file() else None), 0
    candidates = [
        path for path in source_path.parent.glob("source.*")
        if path.is_file() and path != source_path and path.suffix.lower() not in {".part", ".ytdl"}
    ]
    if not candidates:
        return (source_path if source_path.is_file() else None), 0
    original = min(candidates, key=lambda path: path.stat().st_size)
    freed = source_path.stat().st_size if source_path.is_file() else 0
    source_path.unlink(missing_ok=True)
    return original.resolve(), freed


def audio_media_type(path: Path) -> str:
    return "audio/flac" if path.suffix.lower() == ".flac" else "audio/wav"
