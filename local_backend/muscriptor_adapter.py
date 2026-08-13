from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from local_backend.lead_melody import decode_lead_melody


Progress = Callable[[int, str], None]
_model = None
_model_lock = threading.Lock()


def _load_model():
    global _model
    with _model_lock:
        if _model is None:
            from muscriptor import TranscriptionModel

            weights = os.environ.get("PITCHROOM_MUSCRIPTOR_WEIGHTS", "small")
            _model = TranscriptionModel.load_model(weights, device="cpu", dtype="float32")
    return _model


def _read_cached_candidates(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None
    except (OSError, ValueError):
        return None


def _add_acoustic_support(audio_path: Path, candidates: list[dict]) -> None:
    """Measure whether each transcribed fundamental is present in the mix."""
    if not candidates or all("acoustic_support" in item and item["acoustic_support"] != 0.68 for item in candidates):
        return
    sample_rate = 22_050
    hop = 512
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    harmonic, _ = librosa.effects.hpss(audio, margin=(1.0, 3.0))
    magnitude = np.abs(librosa.cqt(
        harmonic,
        sr=sample_rate,
        hop_length=hop,
        fmin=librosa.note_to_hz("C1"),
        n_bins=84,
        bins_per_octave=12,
    )).astype(np.float32)
    logged = np.log1p(magnitude / (float(np.median(magnitude)) + 1e-7))
    frame_median = np.median(logged, axis=0)
    frame_high = np.percentile(logged, 90, axis=0)
    seconds_per_frame = hop / sample_rate
    for item in candidates:
        bin_index = int(round(float(item["midi"]) - 24.0))
        if not 0 <= bin_index < logged.shape[0]:
            item["acoustic_support"] = 0.0
            continue
        first = max(0, int(float(item["time"]) / seconds_per_frame))
        last = min(
            logged.shape[1],
            max(first + 1, int((float(item["time"]) + min(float(item["duration"]), 0.8)) / seconds_per_frame)),
        )
        tone = logged[bin_index, first:last]
        scale = np.maximum(0.05, frame_high[first:last] - frame_median[first:last])
        normalized = (tone - frame_median[first:last]) / scale
        item["acoustic_support"] = round(float(np.clip(np.median(normalized), 0.0, 1.0)), 4)


def transcribe_multi_instrument(
    audio_path: Path,
    output_dir: Path,
    duration: float,
    progress: Progress,
    contour_notes: list[dict] | None = None,
) -> tuple[list[dict] | None, dict]:
    """Run MuScriptor when its gated weights are available.

    A missing package, license acceptance, login, or local weight file is a
    recoverable condition. The analyzer can then use its existing local
    modules instead of failing the whole song load.
    """
    candidate_path = output_dir / "muscriptor-events.json"
    candidates = _read_cached_candidates(candidate_path)
    status = "cache" if candidates is not None else "ready"
    if candidates is None:
        try:
            progress(69, "複数楽器の採譜モデルを読み込んでいます")
            model = _load_model()
            from muscriptor import NoteEndEvent, NoteStartEvent
            from muscriptor.events import ProgressEvent

            starts: dict[int, object] = {}
            candidates = []
            # The final decoder is anchored to an acoustic predominant-melody
            # contour and does not depend on stable instrument labels across
            # chunk boundaries. Decode four chunks together on CPU: this is
            # several times faster than sequential prelude forcing while the
            # contour keeps the selected karaoke line unchanged.
            for event in model.transcribe(
                audio_path,
                instruments=None,
                prelude_forcing=False,
                batch_size=4,
            ):
                if isinstance(event, ProgressEvent):
                    fraction = event.completed / max(1, event.total)
                    progress(70 + int(15 * fraction), "複数楽器を採譜しています")
                elif isinstance(event, NoteStartEvent):
                    starts[event.index] = event
                elif isinstance(event, NoteEndEvent):
                    start = starts.pop(event.start_event_index, event.start_event)
                    note_duration = float(event.end_time) - float(start.start_time)
                    if note_duration >= 0.04:
                        candidates.append({
                            "time": round(float(start.start_time), 4),
                            "duration": round(note_duration, 4),
                            "midi": int(start.pitch),
                            "instrument": str(start.instrument),
                            "confidence": 0.68,
                            "acoustic_support": 0.68,
                        })
            _add_acoustic_support(audio_path, candidates)
            candidate_path.write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
            status = "transcribed"
        except Exception as error:  # model/auth failures must keep the app usable
            message = str(error)
            if "gated" in message.lower() or "401" in message or "authenticate" in message.lower():
                reason = "model_auth_required"
            elif isinstance(error, (ImportError, ModuleNotFoundError)):
                reason = "package_unavailable"
            else:
                reason = "model_error"
            return None, {"status": reason, "detail": message[-500:]}

    _add_acoustic_support(audio_path, candidates)
    progress(86, "楽器をまたぐ主旋律を選んでいます")
    melody, decoder = decode_lead_melody(candidates, duration, contour_notes=contour_notes)
    return melody, {
        "status": status,
        "model": "muscriptor-small",
        "decoder": decoder,
    }
