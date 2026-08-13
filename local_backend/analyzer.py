from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
import torch
import torchcrepe
from demucs.api import Separator
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, median_filter

from local_backend.muscriptor_adapter import transcribe_multi_instrument

Progress = Callable[[int, str], None]
ANALYZER_VERSION = "9.0"
TARGET_SR = 22_050
MODEL_NAME = "htdemucs_6s"
_separator: Separator | None = None
_model_lock = threading.Lock()
_transcriber = None
_transcriber_lock = threading.Lock()
PIANO_MODEL_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
PIANO_MODEL_SHA256 = "c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141"


@dataclass
class MelodyNote:
    time: float
    duration: float
    midi: float
    confidence: float
    attack: bool = False


def _get_separator(progress: Progress) -> Separator:
    global _separator
    with _model_lock:
        if _separator is None:
            progress(8, "音声分離モデルを読み込んでいます（初回はダウンロードします）")
            _separator = Separator(
                model=MODEL_NAME,
                device="cuda" if torch.cuda.is_available() else "cpu",
                shifts=1,
                overlap=0.25,
                split=True,
                segment=7,
                jobs=0,
                progress=False,
            )
    return _separator


def _read_audio(path: Path, limit_seconds: float | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    frames = -1 if limit_seconds is None else min(info.frames, int(limit_seconds * info.samplerate))
    audio, sample_rate = sf.read(path, frames=frames, dtype="float32", always_2d=True)
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio /= peak
    return audio, sample_rate


def _resample_mono(audio: np.ndarray, source_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    mono = np.mean(audio, axis=0, dtype=np.float32)
    if source_sr != target_sr:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=target_sr, res_type="soxr_hq")
    return np.asarray(mono, dtype=np.float32)


def _write_stem(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, sample_rate, subtype="PCM_16")


def _extract_predominant_instrumental_notes(
    audio_path: Path,
    progress: Progress,
    start_progress: int = 72,
    end_progress: int = 88,
    output_name: str | None = None,
) -> list[dict]:
    progress(start_progress, "知覚上の優勢旋律を追跡しています")
    output_path = audio_path.with_name(output_name or f"{audio_path.stem}-melody.json")
    if output_path.exists():
        notes = json.loads(output_path.read_text(encoding="utf-8"))
        progress(end_progress, "保存済みの優勢旋律輪郭を読み込みました")
        return notes
    root = Path(os.environ.get("PITCHROOM_APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
    node = os.environ.get("PITCHROOM_NODE_PATH") or shutil.which("node")
    child_environment = os.environ.copy()
    if not node:
        electron_node = root / "node_modules" / "electron" / "dist" / "electron.exe"
        if electron_node.is_file():
            node = str(electron_node)
            child_environment["ELECTRON_RUN_AS_NODE"] = "1"
    if not node:
        raise RuntimeError("Node.js runtime for the melody tracker was not found")
    if Path(node).name.lower().startswith("electron"):
        child_environment["ELECTRON_RUN_AS_NODE"] = "1"
    completed = subprocess.run(
        [node, str(root / "scripts" / "extract-melodia.cjs"), str(audio_path), str(output_path)],
        cwd=root,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"優勢旋律の追跡に失敗しました: {detail[-800:]}")
    notes = json.loads(output_path.read_text(encoding="utf-8"))
    progress(end_progress, "全音符採譜と統合する旋律輪郭を準備しました")
    return notes


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_piano_model(work_root: Path, progress: Progress) -> Path:
    model_dir = work_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "high-resolution-transcription.pth"
    if model_path.exists() and _file_sha256(model_path) == PIANO_MODEL_SHA256:
        return model_path

    # Reuse the package's conventional cache if it was downloaded previously.
    legacy_path = Path.home() / "piano_transcription_inference_data" / "note_F1=0.9677_pedal_F1=0.9186.pth"
    if legacy_path.exists() and _file_sha256(legacy_path) == PIANO_MODEL_SHA256:
        shutil.copy2(legacy_path, model_path)
        return model_path

    progress(70, "高解像度採譜モデルを取得しています（初回のみ約165MB）")
    temporary_path = model_path.with_suffix(".download")
    try:
        with urllib.request.urlopen(PIANO_MODEL_URL, timeout=90) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if _file_sha256(temporary_path) != PIANO_MODEL_SHA256:
            raise RuntimeError("採譜モデルの検証に失敗しました")
        os.replace(temporary_path, model_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return model_path


def _get_high_resolution_transcriber(work_root: Path, progress: Progress):
    global _transcriber
    with _transcriber_lock:
        if _transcriber is None:
            from piano_transcription_inference import PianoTranscription

            model_path = _ensure_piano_model(work_root, progress)
            progress(72, "高解像度採譜モデルを読み込んでいます")
            _transcriber = PianoTranscription(
                checkpoint_path=str(model_path),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
    return _transcriber


def _rank_transcription_events(audio_path: Path, events: list[dict]) -> list[dict]:
    audio, source_sr = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    if source_sr != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=TARGET_SR, res_type="soxr_hq")
    hop = 256
    cqt = np.abs(librosa.cqt(
        y=mono,
        sr=TARGET_SR,
        hop_length=hop,
        fmin=librosa.note_to_hz("C2"),
        n_bins=84,
        bins_per_octave=12,
    ))
    ranked = []
    for event in events:
        midi_index = int(event["midi_note"]) - 36
        if not 0 <= midi_index < cqt.shape[0]:
            continue
        frame = max(0, min(cqt.shape[1] - 1, round(float(event["onset_time"]) * TARGET_SR / hop)))
        local = np.max(cqt[:, max(0, frame - 1):min(cqt.shape[1], frame + 3)], axis=1)
        strength = float(local[midi_index])
        ranked.append({
            **event,
            "spectral_rank": int(np.sum(local > strength)) + 1,
            "spectral_ratio": strength / max(float(np.max(local)), 1e-9),
        })
    return ranked


def _select_high_resolution_melody(events: list[dict]) -> tuple[list[dict], dict]:
    candidates = sorted(
        (
            {
                "time": float(event["onset_time"]),
                "duration": float(event["offset_time"] - event["onset_time"]),
                "midi": int(event["midi_note"]),
                "velocity": int(event["velocity"]),
                "spectral_rank": int(event.get("spectral_rank", 1)),
                "spectral_ratio": float(event.get("spectral_ratio", 1.0)),
            }
            for event in events
            if float(event["offset_time"] - event["onset_time"]) >= 0.055
            and int(event["velocity"]) >= 35
            and 40 <= int(event["midi_note"]) <= 100
        ),
        key=lambda note: (note["time"], -note["midi"]),
    )
    groups: list[list[dict]] = []
    for note in candidates:
        if groups and note["time"] - groups[-1][0]["time"] <= 0.045:
            groups[-1].append(note)
        else:
            groups.append([note])

    # Reject model-only notes that have no corresponding spectral peak in the
    # source. Among the audible notes at each new attack, follow the skyline
    # voice; harmony notes that began earlier do not displace it.
    supported_groups = [[note for note in group if note["spectral_rank"] <= 6] for group in groups]
    selected = [
        max(group, key=lambda note: (note["midi"], note["spectral_ratio"], note["velocity"]))
        for group in supported_groups
        if group
    ]
    result = []
    for index, note in enumerate(selected):
        next_time = selected[index + 1]["time"] if index + 1 < len(selected) else math.inf
        duration = min(note["duration"], max(0.06, next_time - note["time"]))
        result.append({
            "time": round(note["time"], 4),
            "duration": round(min(duration, 2.0), 4),
            "midi": note["midi"],
            "confidence": round((note["velocity"] / 127 + note["spectral_ratio"]) / 2, 4),
            "attack": True,
        })
    maximum_time = max((group[0]["time"] for group in groups), default=0.0)
    segments = []
    for start in np.arange(0.0, maximum_time + 0.001, 10.0):
        segment_groups = [group for group in groups if start <= group[0]["time"] < start + 10.0]
        segment_notes = [note for note in selected if start <= note["time"] < start + 10.0]
        segments.append({
            "start": round(float(start), 1),
            "end": round(float(start + 10.0), 1),
            "coverage": round(len(segment_notes) / max(1, len(segment_groups)), 4),
            "density": round(len(segment_notes) / 10.0, 4),
            "top3": round(sum(note["spectral_rank"] <= 3 for note in segment_notes) / max(1, len(segment_notes)), 4),
            "median_duration": round(float(np.median([note["duration"] for note in segment_notes])), 4) if segment_notes else 0.0,
        })
    quality = {
        "coverage": round(len(selected) / max(1, len(groups)), 4),
        "top3": round(sum(note["spectral_rank"] <= 3 for note in selected) / max(1, len(selected)), 4),
        "top6": round(sum(note["spectral_rank"] <= 6 for note in selected) / max(1, len(selected)), 4),
        "segments": segments,
    }
    return result, quality


def _route_instrumental_modules(
    attack_notes: list[dict],
    continuous_notes: list[dict],
    quality: dict,
    duration: float,
    source_activity: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    attack_notes = _remove_interleaved_lower_voice(_stabilize_continuous_notes(attack_notes))
    quality_by_start = {
        round(float(segment.get("start", 0.0)), 1): segment
        for segment in (quality.get("segments") or [])
    }
    activity_by_start = {
        round(float(segment.get("start", index * 10.0)), 1): segment
        for index, segment in enumerate(source_activity or [])
    }
    segment_count = max(1, math.ceil(duration / 10.0))
    decisions = []
    for index in range(segment_count):
        start = float(index * 10)
        segment = {
            "start": start,
            "end": min(duration, start + 10.0),
            "coverage": 0.0,
            "density": 0.0,
            "top3": 0.0,
            "median_duration": 0.0,
            **quality_by_start.get(round(start, 1), {}),
        }
        # These are acoustic-shape rules rather than instrument labels. Sparse,
        # sustained, or poorly supported attacks favor continuous-F0 tracking;
        # dense, well-supported attacks favor onset transcription.
        sustained_shape = (
            segment["top3"] < 0.70
            or segment["density"] < 1.8
            or (segment["density"] < 2.5 and segment["median_duration"] >= 0.55)
        )
        activity = activity_by_start.get(round(start, 1))
        attack_count = sum(start <= float(note["time"]) < start + 10.0 for note in attack_notes)
        continuous_count = sum(start <= float(note["time"]) < start + 10.0 for note in continuous_notes)
        continuous_coverage = _note_coverage(continuous_notes, start, start + 10.0) / 10.0
        attack_profile = _continuous_profile(attack_notes, start, start + 10.0)
        continuous_profile = _continuous_profile(continuous_notes, start, start + 10.0)
        if activity:
            continuous_dominant = activity["continuous_ratio"] >= 0.30
            strongly_dominant = activity["continuous_ratio"] >= 0.80
            continuous_ready = continuous_coverage >= 0.55
            continuous = activity["continuous_active"] and continuous_dominant and (
                sustained_shape or strongly_dominant or continuous_ready
            )
        else:
            continuous = sustained_shape
        if attack_count == 0 and continuous_count > 0:
            continuous = True
        elif continuous_count == 0 and attack_count > 0:
            continuous = False
        # A polyphonic sustained tracker can lock onto cello/bass after the
        # lead moves to another family. Prefer the independently detected
        # upper contour only when it remains dense and the continuous result
        # has clearly collapsed into the accompaniment register.
        register_collapsed = (
            attack_profile["count"] >= 6
            and attack_profile["coverage"] >= 0.30
            and 0 < continuous_profile["median_pitch"] < 60
            and attack_profile["median_pitch"] >= continuous_profile["median_pitch"] + 9
            and attack_profile["jump_p90"] <= 30
        )
        if register_collapsed:
            continuous = False
        decisions.append({
            **segment,
            **(activity or {}),
            "attack_notes": attack_count,
            "continuous_notes": continuous_count,
            "continuous_coverage": round(continuous_coverage, 4),
            "attack_median_pitch": round(attack_profile["median_pitch"], 2),
            "continuous_median_pitch": round(continuous_profile["median_pitch"], 2),
            "register_collapsed": register_collapsed,
            "method": "continuous" if continuous else "attack",
        })

    # Remove one-window flicker while retaining genuine multi-window handoffs.
    for index in range(1, len(decisions) - 1):
        if decisions[index - 1]["method"] == decisions[index + 1]["method"] != decisions[index]["method"]:
            decisions[index]["method"] = decisions[index - 1]["method"]

    def method_at(time: float) -> str:
        index = min(len(decisions) - 1, max(0, int(time // 10)))
        return decisions[index]["method"] if decisions else "continuous"

    routed = [note for note in attack_notes if method_at(float(note["time"])) == "attack"]
    routed.extend(note for note in continuous_notes if method_at(float(note["time"])) == "continuous")
    routed.sort(key=lambda note: (float(note["time"]), -float(note.get("confidence", 0))))

    normalized = []
    for source in routed:
        note = {
            **source,
            "time": round(max(0.0, float(source["time"])), 4),
            "duration": round(max(0.06, min(2.0, float(source.get("duration", 0.1)))), 4),
            "midi": int(round(float(source["midi"]))),
            "attack": True,
        }
        previous = normalized[-1] if normalized else None
        if previous and abs(note["time"] - previous["time"]) <= 0.045:
            if note.get("confidence", 0) > previous.get("confidence", 0):
                normalized[-1] = note
            continue
        if previous and previous["midi"] == note["midi"] and note["time"] - (previous["time"] + previous["duration"]) <= 0.05:
            previous["duration"] = round(min(2.0, note["time"] + note["duration"] - previous["time"]), 4)
            continue
        if previous and note["time"] < previous["time"] + previous["duration"]:
            previous["duration"] = round(max(0.06, note["time"] - previous["time"]), 4)
        if note["time"] < duration:
            normalized.append(note)
    return normalized, decisions


def _note_coverage(notes: list[dict], start: float, end: float) -> float:
    intervals = sorted(
        (
            max(start, float(note["time"])),
            min(end, float(note["time"]) + float(note.get("duration", 0.1))),
        )
        for note in notes
        if float(note["time"]) < end and float(note["time"]) + float(note.get("duration", 0.1)) > start
    )
    covered = 0.0
    current_start = current_end = None
    for interval_start, interval_end in intervals:
        if interval_end <= interval_start:
            continue
        if current_end is None or interval_start > current_end:
            if current_end is not None:
                covered += current_end - current_start
            current_start, current_end = interval_start, interval_end
        else:
            current_end = max(current_end, interval_end)
    if current_end is not None:
        covered += current_end - current_start
    return covered


def _continuous_profile(notes: list[dict], start: float, end: float) -> dict:
    segment = [note for note in notes if start <= float(note["time"]) < end]
    pitches = [float(note["midi"]) for note in segment]
    jumps = [abs(pitches[index] - pitches[index - 1]) for index in range(1, len(pitches))]
    return {
        "coverage": _note_coverage(segment, start, end) / max(0.001, end - start),
        "median_pitch": float(np.median(pitches)) if pitches else 0.0,
        "jump_p90": float(np.quantile(jumps, 0.9)) if jumps else 0.0,
        "count": len(segment),
    }


def _stabilize_continuous_notes(notes: list[dict]) -> list[dict]:
    stable = [dict(note) for note in sorted(notes, key=lambda note: float(note["time"]))]
    for index in range(len(stable) - 2, 0, -1):
        previous, note, following = stable[index - 1:index + 2]
        if (
            float(note.get("duration", 0.1)) <= 0.45
            and float(following["time"]) - float(note["time"]) <= 0.9
            and abs(float(note["midi"]) - float(previous["midi"])) >= 8
            and abs(float(note["midi"]) - float(following["midi"])) >= 8
            and abs(float(previous["midi"]) - float(following["midi"])) <= 4
        ):
            stable.pop(index)
    for index in range(len(stable) - 2, 0, -1):
        previous, note, following = stable[index - 1:index + 2]
        local = [
            float(candidate["midi"])
            for candidate in stable
            if abs(float(candidate["time"]) - float(note["time"])) <= 2.0
        ]
        local_register = float(np.median(local)) if len(local) >= 5 else float(note["midi"])
        if (
            float(note["midi"]) <= local_register - 12
            and float(note.get("duration", 0.1)) <= 1.2
            and float(following["time"]) - float(previous["time"]) <= 2.0
            and float(previous["midi"]) >= float(note["midi"]) + 8
            and float(following["midi"]) >= float(note["midi"]) + 8
        ):
            stable.pop(index)
    return stable


def _remove_interleaved_lower_voice(notes: list[dict]) -> list[dict]:
    """Reject short accompaniment notes threaded through a stable upper line."""
    ordered = [dict(note) for note in sorted(notes, key=lambda note: float(note["time"]))]
    kept = []
    for note in ordered:
        time = float(note["time"])
        midi = float(note["midi"])
        previous_upper = [
            candidate for candidate in ordered
            if 0 < time - float(candidate["time"]) <= 1.8
            and float(candidate["midi"]) >= midi + 10
        ]
        following_upper = [
            candidate for candidate in ordered
            if 0 < float(candidate["time"]) - time <= 1.8
            and float(candidate["midi"]) >= midi + 10
        ]
        upper_continues = previous_upper and following_upper and (
            abs(
                float(np.median([candidate["midi"] for candidate in previous_upper]))
                - float(np.median([candidate["midi"] for candidate in following_upper]))
            ) <= 7
        )
        if upper_continues and float(note.get("duration", 0.1)) <= 0.65:
            continue
        kept.append(note)
    return kept


def _choose_continuous_candidates(
    foreground_notes: list[dict],
    residual_notes: list[dict],
    duration: float,
    foreground_activity: list[dict] | None = None,
    residual_activity: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    decisions = []
    selected = []
    for start in np.arange(0.0, duration, 10.0):
        end = min(duration, float(start + 10.0))
        foreground = _continuous_profile(foreground_notes, float(start), end)
        residual = _continuous_profile(residual_notes, float(start), end)
        index = int(start // 10)
        foreground_rms = (
            float(foreground_activity[index].get("continuous_rms", 0.0))
            if foreground_activity and index < len(foreground_activity) else 0.0
        )
        residual_rms = (
            float(residual_activity[index].get("continuous_rms", 0.0))
            if residual_activity and index < len(residual_activity) else 0.0
        )
        residual_audible = (
            True if foreground_activity is None or residual_activity is None
            else residual_rms >= max(1e-6, foreground_rms * 0.06)
        )
        # Coverage rewards a tracker that keeps hearing the phrase; large pitch
        # jumps penalize trackers that bounce between lead and accompaniment.
        foreground_score = foreground["coverage"] - foreground["jump_p90"] * 0.02
        residual_score = residual["coverage"] - residual["jump_p90"] * 0.02
        collapsed_register = (
            foreground["median_pitch"] > 0
            and residual["median_pitch"] >= foreground["median_pitch"] + 7
            and residual["coverage"] >= 0.30
            and residual["coverage"] >= foreground["coverage"] * 0.9
        )
        use_residual = residual["count"] > 0 and residual_audible and (
            residual_score >= foreground_score + 0.03 or collapsed_register
        )
        method = "residual" if use_residual else "foreground"
        decisions.append({
            "start": round(float(start), 1),
            "end": round(end, 1),
            "method": method,
            "foreground_rms": round(foreground_rms, 7),
            "residual_rms": round(residual_rms, 7),
            "residual_audible": residual_audible,
            "foreground": {key: round(value, 4) for key, value in foreground.items()},
            "residual": {key: round(value, 4) for key, value in residual.items()},
        })

    for index in range(1, len(decisions) - 1):
        if decisions[index - 1]["method"] == decisions[index + 1]["method"] != decisions[index]["method"]:
            decisions[index]["method"] = decisions[index - 1]["method"]

    for decision in decisions:
        source = residual_notes if decision["method"] == "residual" else foreground_notes
        selected.extend(
            note for note in source
            if decision["start"] <= float(note["time"]) < decision["end"]
        )
    return _stabilize_continuous_notes(selected), decisions


def _fill_detected_contour_gaps(
    contour_notes: list[dict],
    detected_notes: list[dict],
) -> tuple[list[dict], int]:
    """Fill short contour holes without copying material from another time."""
    filled = [dict(note) for note in sorted(contour_notes, key=lambda note: float(note["time"]))]
    additions: list[dict] = []
    for previous, following in zip(filled, filled[1:]):
        gap_start = float(previous["time"]) + float(previous.get("duration", 0.1))
        gap_end = float(following["time"])
        gap = gap_end - gap_start
        if gap < 0.12 or gap > 1.35:
            continue
        midpoint = (gap_start + gap_end) / 2
        for source in detected_notes:
            source_start = float(source["time"])
            source_end = source_start + float(source.get("duration", 0.1))
            start = max(gap_start, source_start)
            end = min(gap_end, source_end)
            if end - start < 0.06 or float(source.get("confidence", 0.0)) < 0.55:
                continue
            midi = float(source["midi"])
            center = (start + end) / 2
            agrees_with_edge = (
                center <= midpoint and abs(midi - float(previous["midi"])) <= 4
            ) or (
                center >= midpoint and abs(midi - float(following["midi"])) <= 4
            )
            if not agrees_with_edge:
                continue
            additions.append({
                **source,
                "time": round(start, 4),
                "duration": round(end - start, 4),
                "midi": int(round(midi)),
                "attack": True,
            })

    if not additions:
        return filled, 0
    filled.extend(additions)
    return _stabilize_continuous_notes(filled), len(additions)


def _measure_source_activity(
    attack_source: np.ndarray,
    continuous_source: np.ndarray,
    sample_rate: int,
    duration: float,
) -> list[dict]:
    segment_seconds = 10.0
    continuous_levels = []
    raw = []
    for start in np.arange(0.0, duration, segment_seconds):
        first = int(start * sample_rate)
        last = min(attack_source.shape[1], int((start + segment_seconds) * sample_rate))
        attack_rms = _energy(attack_source[:, first:last]) if last > first else 0.0
        continuous_rms = _energy(continuous_source[:, first:last]) if last > first else 0.0
        continuous_levels.append(continuous_rms)
        raw.append((float(start), attack_rms, continuous_rms))
    active_floor = max(1e-6, max(continuous_levels, default=0.0) * 0.01)
    return [{
        "start": round(start, 1),
        "attack_rms": round(attack_rms, 7),
        "continuous_rms": round(continuous_rms, 7),
        "continuous_ratio": round(continuous_rms / max(attack_rms, 1e-7), 4),
        "continuous_active": continuous_rms >= active_floor,
    } for start, attack_rms, continuous_rms in raw]


def _extract_high_resolution_instrumental_notes(
    audio_path: Path,
    work_root: Path,
    progress: Progress,
) -> tuple[list[dict], dict]:
    progress(70, "高解像度で楽器の打鍵を採譜しています")
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    if sample_rate != 16_000:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=16_000, res_type="soxr_hq")
    event_path = audio_path.with_name("high-resolution-events.json")
    if event_path.exists():
        events = json.loads(event_path.read_text(encoding="utf-8"))
    else:
        transcribed = _get_high_resolution_transcriber(work_root, progress).transcribe(mono, None)
        events = [
            {
                "onset_time": float(event["onset_time"]),
                "offset_time": float(event["offset_time"]),
                "midi_note": int(event["midi_note"]),
                "velocity": int(event["velocity"]),
            }
            for event in transcribed["est_note_events"]
        ]
        event_path.write_text(json.dumps(events), encoding="utf-8")
    progress(86, "採譜結果を原音の各アタックと照合しています")
    return _select_high_resolution_melody(_rank_transcription_events(audio_path, events))


def _energy(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def _should_use_multi_instrument(mode: str, attack_to_continuous_ratio: float) -> bool:
    """Route only non-attack-dominant instrumentals to the slower general AMT."""
    return mode == "instrumental" and attack_to_continuous_ratio < 1.30


def _fill_short_gaps(values: np.ndarray, max_gap: int = 4) -> np.ndarray:
    result = values.copy()
    index = 0
    while index < len(result):
        if np.isfinite(result[index]):
            index += 1
            continue
        end = index
        while end < len(result) and not np.isfinite(result[end]):
            end += 1
        if index > 0 and end < len(result) and end - index <= max_gap:
            left, right = result[index - 1], result[end]
            if abs(left - right) <= 2.0:
                result[index:end] = np.linspace(left, right, end - index + 2)[1:-1]
        index = end
    return result


def _stabilize_octaves(midi: np.ndarray) -> np.ndarray:
    """Remove only very short octave glitches without folding real vocal leaps.

    CREPE can occasionally lock onto a harmonic for one or two frames.  The old
    implementation always moved a large jump toward the previous pitch, which
    also destroyed genuine octave jumps.  Here an octave correction is allowed
    only when a run is at most 30 ms and the pitches immediately before and
    after it agree.
    """
    result = midi.copy()
    index = 1
    while index < len(result) - 1:
        if not np.isfinite(result[index]) or not np.isfinite(result[index - 1]):
            index += 1
            continue
        run_end = index + 1
        while (
            run_end < len(result)
            and run_end - index < 4
            and np.isfinite(result[run_end])
            and abs(result[run_end] - result[index]) < 2.5
        ):
            run_end += 1
        if run_end < len(result) and np.isfinite(result[run_end]):
            before = result[index - 1]
            after = result[run_end]
            run_pitch = float(np.nanmedian(result[index:run_end]))
            octave_delta = run_pitch - ((before + after) / 2)
            if abs(before - after) < 2.0 and 10.5 <= abs(octave_delta) <= 13.5:
                result[index:run_end] -= math.copysign(12.0, octave_delta)
        index = max(index + 1, run_end)
    return result


def _notes_from_track(
    midi: np.ndarray,
    confidence: np.ndarray,
    hop_seconds: float,
    attack_frames: np.ndarray | None = None,
) -> list[MelodyNote]:
    stable = _fill_short_gaps(_stabilize_octaves(midi))
    finite = np.isfinite(stable)
    if finite.any():
        filled = stable.copy()
        fallback = float(np.nanmedian(filled))
        filled[~finite] = fallback
        filled = median_filter(filled, size=5, mode="nearest")
        stable[finite] = filled[finite]
    quantized = np.where(np.isfinite(stable), np.round(stable), np.nan)
    attacks: set[int] = set()
    minimum_note_frames = max(1, math.ceil(0.075 / hop_seconds))
    for detected_frame in np.asarray(attack_frames if attack_frames is not None else [], dtype=int):
        # Backtracked onset frames can land one or two frames before the pitch
        # becomes measurable. Attach the attack to the first voiced frame so
        # the re-articulation survives note segmentation.
        search_end = min(len(quantized), int(detected_frame) + 6)
        for candidate in range(max(0, int(detected_frame)), search_end):
            run_end = candidate
            while (
                run_end < len(quantized)
                and np.isfinite(quantized[run_end])
                and np.isfinite(quantized[candidate])
                and abs(quantized[run_end] - quantized[candidate]) <= 0.5
            ):
                run_end += 1
            if np.isfinite(midi[candidate]) and run_end - candidate >= minimum_note_frames:
                attacks.add(candidate)
                break
    notes: list[MelodyNote] = []
    start = 0
    while start < len(quantized):
        if not np.isfinite(quantized[start]) or confidence[start] < 0.18:
            start += 1
            continue
        pitch = quantized[start]
        end = start + 1
        while (
            end < len(quantized)
            and end not in attacks
            and np.isfinite(quantized[end])
            and abs(quantized[end] - pitch) <= 0.5
        ):
            end += 1
        duration = (end - start) * hop_seconds
        segment_conf = float(np.nanmedian(confidence[start:end]))
        if duration >= 0.075 and segment_conf >= 0.22:
            note = MelodyNote(
                time=round(start * hop_seconds, 4),
                duration=round(duration, 4),
                midi=float(pitch),
                confidence=round(min(1.0, max(0.0, segment_conf)), 4),
                attack=start in attacks,
            )
            if notes and not note.attack and note.time - (notes[-1].time + notes[-1].duration) < 0.09 and note.midi == notes[-1].midi:
                notes[-1].duration = round(note.time + note.duration - notes[-1].time, 4)
                notes[-1].confidence = round((notes[-1].confidence + note.confidence) / 2, 4)
            else:
                notes.append(note)
        start = end
    return notes


def _simplify_vocal_notes(notes: list[MelodyNote]) -> list[MelodyNote]:
    """Reduce pitch-bar chatter caused by vibrato without flattening phrases."""
    if not notes:
        return []

    joined: list[MelodyNote] = []
    for source in notes:
        note = MelodyNote(source.time, source.duration, source.midi, source.confidence, source.attack)
        previous = joined[-1] if joined else None
        previous_end = previous.time + previous.duration if previous else 0.0
        if previous and note.midi == previous.midi and note.time - previous_end <= 0.12:
            total_duration = previous.duration + note.duration
            previous.confidence = round(
                (previous.confidence * previous.duration + note.confidence * note.duration)
                / max(total_duration, 1e-6),
                4,
            )
            previous.duration = round(note.time + note.duration - previous.time, 4)
        else:
            joined.append(note)

    # A short neighbouring-note excursion that immediately returns to the
    # original pitch is normally vibrato or a scoop in sung vocals. Karaoke
    # guide bars should show the perceived target note, not every oscillation.
    changed = True
    while changed and len(joined) >= 3:
        changed = False
        for index in range(1, len(joined) - 1):
            before, turn, after = joined[index - 1:index + 2]
            gap_before = turn.time - (before.time + before.duration)
            gap_after = after.time - (turn.time + turn.duration)
            if (
                before.midi == after.midi
                and 0.5 <= abs(turn.midi - before.midi) <= 2.0
                and turn.duration <= 0.16
                and gap_before <= 0.06
                and gap_after <= 0.06
            ):
                weighted_duration = before.duration + turn.duration + after.duration
                confidence = (
                    before.confidence * before.duration
                    + turn.confidence * turn.duration
                    + after.confidence * after.duration
                ) / max(weighted_duration, 1e-6)
                joined[index - 1:index + 2] = [MelodyNote(
                    time=before.time,
                    duration=round(after.time + after.duration - before.time, 4),
                    midi=before.midi,
                    confidence=round(confidence, 4),
                    attack=before.attack,
                )]
                changed = True
                break
    return joined


def _normalize_vocal_register(notes: list[MelodyNote]) -> list[MelodyNote]:
    """Repair sparse octave misses using the song's duration-weighted register.

    This is intentionally asymmetric: low outliers may be lifted one octave,
    while sustained high notes are never folded down. Only very short extreme
    high outliers are treated as harmonic detections.
    """
    if not notes:
        return notes
    ordered = sorted(notes, key=lambda note: note.midi)
    total_weight = sum(max(0.04, note.duration) * max(0.2, note.confidence) for note in ordered)
    accumulated = 0.0
    register_center = ordered[len(ordered) // 2].midi
    for note in ordered:
        accumulated += max(0.04, note.duration) * max(0.2, note.confidence)
        if accumulated >= total_weight / 2:
            register_center = note.midi
            break

    low_threshold = register_center - 11.5
    low_weight = sum(note.duration for note in notes if note.midi < low_threshold)
    voiced_duration = sum(note.duration for note in notes)
    sparse_low_cluster = low_weight / max(voiced_duration, 1e-6) < 0.12
    high_threshold = register_center + 24

    normalized: list[MelodyNote] = []
    for note in notes:
        midi = note.midi
        if sparse_low_cluster:
            while midi < low_threshold:
                midi += 12
        if midi > high_threshold and note.duration < 0.2:
            midi -= 12
        normalized.append(MelodyNote(note.time, note.duration, midi, note.confidence, note.attack))
    return normalized


def _match_frame_count(values: np.ndarray, frame_count: int) -> np.ndarray:
    if len(values) == frame_count:
        return values
    if not len(values):
        return np.zeros(frame_count, dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, len(values))
    target_positions = np.linspace(0.0, 1.0, frame_count)
    return np.interp(target_positions, source_positions, values).astype(np.float32)


def _voice_presence_mask(
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    sample_rate: int,
    target_sr: int,
    hop: int,
    frame_count: int,
) -> np.ndarray:
    """Estimate where a separated vocal stem contains an actual lead voice.

    Demucs may leave pitched instruments in the vocal stem. The gate keeps
    frames that have sufficient vocal-stem energy, are not overwhelmed by the
    accompaniment, and are reasonably centered in the stereo image.
    """
    vocal_mid = _resample_mono(vocals, sample_rate, target_sr)
    accompaniment_mid = _resample_mono(accompaniment, sample_rate, target_sr)
    if vocals.shape[0] >= 2:
        vocal_side = _resample_mono(np.stack((vocals[0], -vocals[1])), sample_rate, target_sr)
    else:
        vocal_side = np.zeros_like(vocal_mid)

    frame_length = 1024
    vocal_rms = librosa.feature.rms(y=vocal_mid, frame_length=frame_length, hop_length=hop, center=True)[0]
    accompaniment_rms = librosa.feature.rms(y=accompaniment_mid, frame_length=frame_length, hop_length=hop, center=True)[0]
    side_rms = librosa.feature.rms(y=vocal_side, frame_length=frame_length, hop_length=hop, center=True)[0]
    vocal_rms = _match_frame_count(vocal_rms, frame_count)
    accompaniment_rms = _match_frame_count(accompaniment_rms, frame_count)
    side_rms = _match_frame_count(side_rms, frame_count)

    strong_level = float(np.percentile(vocal_rms, 90))
    quiet_level = float(np.percentile(vocal_rms, 10))
    energy_floor = max(1e-5, strong_level * 0.04, min(quiet_level * 1.6, strong_level * 0.09))
    isolation_db = 20 * np.log10((vocal_rms + 1e-7) / (accompaniment_rms + 1e-7))
    center_share = vocal_rms / (vocal_rms + side_rms + 1e-7)
    presence = (vocal_rms >= energy_floor) & (isolation_db >= -20.0) & (center_share >= 0.42)

    # Bridge tiny consonant gaps, remove isolated leakage bursts, then restore
    # a small margin around each phrase so note onsets and releases survive.
    presence = binary_closing(presence, structure=np.ones(9, dtype=bool))
    presence = binary_opening(presence, structure=np.ones(7, dtype=bool))
    presence = binary_dilation(presence, structure=np.ones(9, dtype=bool))
    return np.asarray(presence, dtype=bool)


def _extract_vocal_melody(
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    sample_rate: int,
    progress: Progress,
) -> tuple[list[MelodyNote], float]:
    progress(72, "分離したボーカルから音程を追跡しています")
    crepe_sr = 16_000
    hop = 160
    mono = _resample_mono(vocals, sample_rate, crepe_sr)
    peak = float(np.max(np.abs(mono)))
    if peak > 1e-5:
        mono = mono / peak * 0.95
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tensor = torch.from_numpy(mono)[None].to(device)
    with torch.inference_mode():
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            crepe_sr,
            hop,
            float(librosa.note_to_hz("C2")),
            float(librosa.note_to_hz("C7")),
            model="full",
            decoder=torchcrepe.decode.weighted_argmax,
            return_periodicity=True,
            batch_size=1024,
            device=device,
            pad=True,
        )
        periodicity = torchcrepe.filter.median(periodicity, 5)
        pitch = torchcrepe.filter.mean(pitch, 3)
    probability = periodicity[0].detach().cpu().numpy()
    midi = librosa.hz_to_midi(pitch[0].detach().cpu().numpy())
    voice_present = _voice_presence_mask(vocals, accompaniment, sample_rate, crepe_sr, hop, len(probability))
    midi[(probability < 0.28) | ~voice_present] = np.nan
    notes = _notes_from_track(midi, probability, hop / crepe_sr)
    notes = _simplify_vocal_notes(_normalize_vocal_register(notes))
    return notes, float(np.mean(voice_present))


def _vocal_hypothesis_quality(
    notes: list[MelodyNote],
    duration: float,
    vocal_ratio: float,
    voice_gate_ratio: float,
) -> dict:
    intervals = sorted((note.time, note.time + note.duration) for note in notes)
    covered = 0.0
    if intervals:
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered += end - start
                start, end = next_start, next_end
        covered += end - start
    coverage = covered / max(duration, 0.01)
    density = len(notes) / max(duration, 0.01)
    score = (
        0.35 * min(1.0, vocal_ratio / 0.35)
        + 0.25 * min(1.0, voice_gate_ratio / 0.55)
        + 0.25 * min(1.0, coverage / 0.45)
        + 0.15 * min(1.0, density / 1.2)
    )
    accepted = (
        score >= 0.60
        and (coverage >= 0.16 or vocal_ratio >= 0.35)
        and len(notes) >= max(8, int(duration * 0.18))
    )
    return {
        "accepted": accepted,
        "score": round(score, 4),
        "coverage": round(coverage, 4),
        "density": round(density, 4),
        "note_count": len(notes),
    }


def _instrument_salience_track(audio: np.ndarray, sample_rate: int, progress: Progress) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    progress(72, "伴奏から旋律候補を探索しています")
    mono = _resample_mono(audio, sample_rate)
    harmonic, _ = librosa.effects.hpss(mono, margin=(1.0, 4.0))
    hop = 512
    bins_per_octave = 24
    cqt = np.abs(librosa.cqt(
        harmonic,
        sr=TARGET_SR,
        hop_length=hop,
        fmin=librosa.note_to_hz("C3"),
        n_bins=96,
        bins_per_octave=bins_per_octave,
    ))
    onset_envelope = librosa.onset.onset_strength(
        y=harmonic,
        sr=TARGET_SR,
        hop_length=hop,
        aggregate=np.median,
    )
    attack_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=TARGET_SR,
        hop_length=hop,
        backtrack=True,
        units="frames",
        delta=0.08,
        wait=2,
    )
    salience = np.log1p(cqt / (np.median(cqt, axis=1, keepdims=True) + 1e-6))
    frequencies = librosa.cqt_frequencies(96, fmin=librosa.note_to_hz("C3"), bins_per_octave=bins_per_octave)
    midis = librosa.hz_to_midi(frequencies)
    # Karaoke melodies are usually above the bass register. A gentle upper-register
    # prior prevents the bass line from winning without forcing a fixed octave.
    salience += ((midis - 60) / 48.0)[:, None]
    frame_floor = np.percentile(salience, 55, axis=0)
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in range(salience.shape[1]):
        column = salience[:, frame]
        peaks = np.flatnonzero((column >= np.roll(column, 1)) & (column >= np.roll(column, -1)))
        peaks = peaks[(peaks > 0) & (peaks < len(column) - 1)]
        if peaks.size:
            top = peaks[np.argsort(column[peaks])[-8:]]
            candidates.append((midis[top], column[top]))
        else:
            candidates.append((np.array([], dtype=float), np.array([], dtype=float)))
    tracked = np.full(len(candidates), np.nan)
    confidence = np.zeros(len(candidates))
    previous: float | None = None
    for frame, (pitches, strengths) in enumerate(candidates):
        if not pitches.size or float(np.max(strengths)) < frame_floor[frame]:
            previous = None if frame % 5 == 0 else previous
            continue
        normalized = strengths - float(np.median(strengths))
        if previous is None:
            scores = normalized
        else:
            distance = np.abs(pitches - previous)
            scores = normalized - 0.24 * distance - 0.5 * (distance > 9)
        best = int(np.argmax(scores))
        tracked[frame] = pitches[best]
        confidence[frame] = float(1 / (1 + math.exp(-max(-8.0, min(8.0, normalized[best])))))
        previous = float(pitches[best])
    return tracked, confidence, attack_frames, hop / TARGET_SR


def _extract_instrumental_melody(audio: np.ndarray, sample_rate: int, progress: Progress) -> list[MelodyNote]:
    midi, confidence, attack_frames, hop_seconds = _instrument_salience_track(audio, sample_rate, progress)
    return _notes_from_track(midi, confidence, hop_seconds, attack_frames)


def _cache_key(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(ANALYZER_VERSION.encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:20]


def analyze_file(
    source_path: Path,
    work_root: Path,
    progress: Progress,
    limit_seconds: float | None = None,
) -> dict:
    cache_dir = work_root / "cache" / _cache_key(source_path)
    result_path = cache_dir / "analysis.json"
    if result_path.exists() and limit_seconds is None:
        progress(100, "保存済みの解析結果を読み込みました")
        return json.loads(result_path.read_text(encoding="utf-8"))

    progress(2, "音源を読み込んでいます")
    audio, sample_rate = _read_audio(source_path, limit_seconds)
    duration = audio.shape[1] / sample_rate
    separator = _get_separator(progress)

    def separation_progress(data: dict) -> None:
        offset = float(data.get("segment_offset", 0))
        total = float(data.get("audio_length", audio.shape[1]))
        value = 12 + int(53 * min(1.0, offset / max(1.0, total)))
        progress(value, "ボーカル・ドラム・ベース・その他に分離しています")

    separator._callback = separation_progress
    progress(12, "ボーカル・ドラム・ベース・その他に分離しています")
    # Demucs' shift trick uses a random time offset. With a single shift that
    # made subtle instrumental attacks vary between otherwise identical runs,
    # occasionally turning a short A-B-A figure into one sustained harmonic.
    # Keep the quality benefit of one shift, but make its offset reproducible.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(0)
        _, stems_tensor = separator.separate_tensor(torch.from_numpy(audio), sr=sample_rate)
    stems = {name: value.detach().cpu().numpy() for name, value in stems_tensor.items()}
    model_sr = separator.samplerate
    vocals = stems["vocals"]
    stem_silence = np.zeros_like(next(iter(stems.values())))
    attack_source = stems.get("piano", stem_silence) + stems.get("guitar", stem_silence)
    continuous_source = stems["other"] + stems.get("guitar", stem_silence)
    melody_source = attack_source + stems["other"]
    vocal_ratio = _energy(vocals) / max(_energy(audio), 1e-6)
    non_vocal_accompaniment = sum(
        (stem for name, stem in stems.items() if name != "vocals"), stem_silence.copy(),
    )
    if vocal_ratio < 0.04:
        # A nearly empty vocal stem is decisive for instrumental material.
        # Avoid running full-song CREPE only to confirm silence/leakage.
        vocal_notes: list[MelodyNote] = []
        detected_voice_gate_ratio = 0.0
        vocal_hypothesis = {
            "accepted": False,
            "score": round(0.35 * min(1.0, vocal_ratio / 0.35), 4),
            "coverage": 0.0,
            "density": 0.0,
            "note_count": 0,
            "fast_rejected": True,
        }
    else:
        vocal_notes, detected_voice_gate_ratio = _extract_vocal_melody(
            vocals, non_vocal_accompaniment, model_sr, progress,
        )
        vocal_hypothesis = _vocal_hypothesis_quality(
            vocal_notes, duration, vocal_ratio, detected_voice_gate_ratio,
        )
    mode = "vocal" if vocal_hypothesis["accepted"] else "instrumental"
    accompaniment = (
        non_vocal_accompaniment
        if mode == "vocal"
        else sum(stems.values(), stem_silence.copy())
    )

    output_dir = cache_dir if limit_seconds is None else work_root / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    accompaniment_path = output_dir / "accompaniment.wav"
    vocals_path = output_dir / "vocals.wav"
    melody_source_path = output_dir / "melody-source.wav"
    attack_source_path = output_dir / "attack-source.wav"
    continuous_source_path = output_dir / "continuous-source.wav"
    _write_stem(accompaniment_path, accompaniment, model_sr)
    _write_stem(vocals_path, vocals, model_sr)
    _write_stem(melody_source_path, melody_source, model_sr)
    _write_stem(attack_source_path, attack_source, model_sr)
    _write_stem(continuous_source_path, continuous_source, model_sr)

    multi_notes: list[dict] | None = None
    multi_instrument: dict | None = None
    foreground_notes: list[dict] = []
    residual_notes: list[dict] = []
    foreground_activity: list[dict] = []
    residual_activity: list[dict] = []
    attack_to_continuous_ratio = _energy(attack_source) / max(_energy(continuous_source), 1e-7)
    # Attack-dominant arrangements are better served by the precise piano /
    # guitar AMT. MuScriptor is reserved for complex mixtures where sustained
    # sources are at least comparable and the lead can move between timbres.
    use_multi_instrument = _should_use_multi_instrument(mode, attack_to_continuous_ratio)
    if use_multi_instrument:
        foreground_notes = _extract_predominant_instrumental_notes(
            continuous_source_path, progress, start_progress=66, end_progress=67,
            output_name="continuous-melody.json",
        )
        residual_notes = _extract_predominant_instrumental_notes(
            vocals_path, progress, start_progress=67, end_progress=68,
            output_name="residual-melody.json",
        )
        silence = np.zeros_like(continuous_source)
        foreground_activity = _measure_source_activity(silence, continuous_source, model_sr, duration)
        residual_activity = _measure_source_activity(silence, vocals, model_sr, duration)
        multi_contour, contour_routing = _choose_continuous_candidates(
            foreground_notes, residual_notes, duration,
            foreground_activity, residual_activity,
        )
        multi_notes, multi_instrument = (
            transcribe_multi_instrument(
                source_path, output_dir, duration, progress, contour_notes=multi_contour,
            )
            if limit_seconds is None
            else (None, {"status": "preview_skipped"})
        )
        if multi_instrument is not None:
            multi_instrument["contour_routing"] = contour_routing

    progress(68, "主旋律の解析方法を選択しています")
    if mode == "vocal":
        notes = vocal_notes
        voice_gate_ratio = detected_voice_gate_ratio
        predominant_notes: list[dict] = []
        instrumental_method = None
        transcription_quality = None
    elif multi_notes:
        notes = multi_notes
        voice_gate_ratio = None
        predominant_notes = []
        decoder = (multi_instrument or {}).get("decoder", {})
        instrumental_method = (
            "muscriptor_contour_hybrid"
            if decoder.get("contour_anchored")
            else "muscriptor_lead_graph"
        )
        transcription_quality = {"multi_instrument": multi_instrument}
    else:
        high_resolution_notes, transcription_quality = _extract_high_resolution_instrumental_notes(
            attack_source_path, work_root, progress,
        )
        transcription_quality["analyzer_routing"] = {
            "method": "attack_modules",
            "attack_to_continuous_ratio": round(attack_to_continuous_ratio, 4),
            "multi_instrument_skipped": not use_multi_instrument,
        }
        attack_reliable = (
            len(high_resolution_notes) >= max(12, duration * 0.25)
            and transcription_quality["coverage"] >= 0.70
            and transcription_quality["top3"] >= 0.75
            and transcription_quality["top6"] >= 0.90
        )
        transcription_quality["attack_reliable"] = attack_reliable
        if duration > 0:
            progress(89, "持続旋律を追跡し、区間ごとに解析器を切り替えています")
            if not foreground_notes:
                foreground_notes = _extract_predominant_instrumental_notes(
                    continuous_source_path, progress, start_progress=87, end_progress=89,
                    output_name="continuous-melody.json",
                )
                residual_notes = _extract_predominant_instrumental_notes(
                    vocals_path, progress, start_progress=89, end_progress=91,
                    output_name="residual-melody.json",
                )
                silence = np.zeros_like(continuous_source)
                foreground_activity = _measure_source_activity(
                    silence, continuous_source, model_sr, duration,
                )
                residual_activity = _measure_source_activity(
                    silence, vocals, model_sr, duration,
                )
            continuous_notes, continuous_routing = _choose_continuous_candidates(
                foreground_notes, residual_notes, duration,
                foreground_activity, residual_activity,
            )
            continuous_notes, gap_fill_count = _fill_detected_contour_gaps(
                continuous_notes, high_resolution_notes,
            )
            # A separator may label violin-like foreground as "vocals". In an
            # instrumental piece that stem remains a valid sustained source.
            continuous_activity_source = continuous_source + vocals
            source_activity = _measure_source_activity(
                attack_source, continuous_activity_source, model_sr, duration,
            )
            notes, routing = _route_instrumental_modules(
                high_resolution_notes, continuous_notes, transcription_quality, duration, source_activity,
            )
            transcription_quality["routing"] = routing
            transcription_quality["continuous_routing"] = continuous_routing
            transcription_quality["gap_fill_count"] = gap_fill_count
            if notes:
                predominant_notes = []
                instrumental_method = "adaptive_modules"
            else:
                predominant_notes = _extract_predominant_instrumental_notes(melody_source_path, progress)
                instrumental_method = "hybrid_fallback"
        else:
            notes = []
            predominant_notes = _extract_predominant_instrumental_notes(melody_source_path, progress)
            instrumental_method = "hybrid_fallback"
        transcription_quality["multi_instrument"] = multi_instrument
        voice_gate_ratio = None

    progress(92, "カラオケ用音源を書き出しています")
    result = {
        "version": ANALYZER_VERSION,
        "mode": mode,
        "duration": round(duration, 4),
        "sample_rate": model_sr,
        "vocal_ratio": round(vocal_ratio, 4),
        "vocal_hypothesis": vocal_hypothesis,
        "attack_to_continuous_ratio": round(attack_to_continuous_ratio, 4),
        "voice_gate_applied": mode == "vocal",
        "voice_gate_ratio": round(voice_gate_ratio, 4) if voice_gate_ratio is not None else None,
        "notes": [asdict(note) for note in notes] if mode == "vocal" else notes,
        "predominant_notes": predominant_notes,
        "instrumental_method": instrumental_method,
        "transcription_quality": transcription_quality,
        "note_count": len(notes) + len(predominant_notes),
        "accompaniment_path": str(accompaniment_path.resolve()),
        "vocals_path": str(vocals_path.resolve()),
        "melody_source_path": str(melody_source_path.resolve()),
    }
    if limit_seconds is None:
        result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    progress(100, "主旋律の解析が完了しました")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PitchRoom melody analyzer")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path(".pitchroom"))
    parser.add_argument("--limit", type=float, default=None)
    arguments = parser.parse_args()

    def print_progress(value: int, message: str) -> None:
        print(f"[{value:3d}%] {message}", flush=True)

    print(json.dumps(analyze_file(arguments.audio, arguments.work_root, print_progress, arguments.limit), ensure_ascii=False, indent=2))
