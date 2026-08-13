from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class _Candidate:
    time: float
    duration: float
    midi: float
    instrument: str
    confidence: float
    acoustic_support: float

    @property
    def end(self) -> float:
        return self.time + self.duration


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _union_coverage(notes: list[_Candidate], start: float, end: float) -> float:
    intervals = sorted(
        (max(start, note.time), min(end, note.end))
        for note in notes
        if note.time < end and note.end > start
    )
    if not intervals:
        return 0.0
    covered = 0.0
    current_start, current_end = intervals[0]
    for interval_start, interval_end in intervals[1:]:
        if interval_start <= current_end:
            current_end = max(current_end, interval_end)
        else:
            covered += current_end - current_start
            current_start, current_end = interval_start, interval_end
    return covered + current_end - current_start


def _normalize_candidates(candidates: list[dict]) -> list[_Candidate]:
    normalized: list[_Candidate] = []
    for item in candidates:
        instrument = str(item.get("instrument", "unknown")).strip().lower() or "unknown"
        if any(word in instrument for word in ("drum", "percussion", "cymbal")):
            continue
        time = float(item.get("time", item.get("start_time", 0.0)))
        duration = float(item.get("duration", 0.0))
        midi = float(item.get("midi", item.get("pitch", -1.0)))
        if not (0.0 <= time and 0.04 <= duration <= 16.0 and 21.0 <= midi <= 108.0):
            continue
        normalized.append(_Candidate(
            time=time,
            duration=duration,
            midi=midi,
            instrument=instrument,
            confidence=_clamp(float(item.get("confidence", 0.65))),
            acoustic_support=_clamp(float(item.get("acoustic_support", item.get("confidence", 0.65)))),
        ))
    return sorted(normalized, key=lambda note: (note.time, -note.midi, note.instrument))


def _instrument_window_scores(
    notes: list[_Candidate],
    duration: float,
    window_seconds: float,
) -> tuple[list[dict[str, float]], list[dict[str, dict[str, float]]]]:
    window_count = max(1, math.ceil(max(duration, max((note.end for note in notes), default=0.0)) / window_seconds))
    scores: list[dict[str, float]] = []
    profiles: list[dict[str, dict[str, float]]] = []
    for window_index in range(window_count):
        start = window_index * window_seconds
        end = min(max(duration, start + window_seconds), start + window_seconds)
        window_notes = [note for note in notes if note.time < end and note.end > start]
        by_instrument: dict[str, list[_Candidate]] = defaultdict(list)
        for note in window_notes:
            by_instrument[note.instrument].append(note)
        all_midis = sorted(note.midi for note in window_notes)
        global_median = all_midis[len(all_midis) // 2] if all_midis else 60.0
        window_scores: dict[str, float] = {}
        window_profiles: dict[str, dict[str, float]] = {}
        for instrument, instrument_notes in by_instrument.items():
            union = _union_coverage(instrument_notes, start, end)
            total = sum(max(0.0, min(end, note.end) - max(start, note.time)) for note in instrument_notes)
            coverage = _clamp(union / max(0.01, end - start))
            polyphony = total / max(union, 0.04)
            density = len([note for note in instrument_notes if start <= note.time < end]) / max(0.01, end - start)
            support = sum(note.acoustic_support for note in instrument_notes) / len(instrument_notes)
            confidence = sum(note.confidence for note in instrument_notes) / len(instrument_notes)
            ordered = sorted(instrument_notes, key=lambda note: note.time)
            motion = 0.0
            if len(ordered) > 1:
                motion = sum(min(12.0, abs(right.midi - left.midi)) for left, right in zip(ordered, ordered[1:]))
                motion = _clamp(motion / (4.0 * (len(ordered) - 1)))
            median_pitch = sorted(note.midi for note in instrument_notes)[len(instrument_notes) // 2]
            relative_register = 1.0 / (1.0 + math.exp(-(median_pitch - global_median) / 5.0))
            bass_penalty = 0.65 if "bass" in instrument and len(by_instrument) > 1 else 0.0
            score = (
                2.15 * support
                + 0.55 * confidence
                + 0.90 * min(1.0, coverage / 0.55)
                + 0.40 * motion
                + 0.32 * relative_register
                - 0.95 * max(0.0, polyphony - 1.15)
                - 0.10 * max(0.0, density - 6.0)
                - bass_penalty
            )
            window_scores[instrument] = score
            window_profiles[instrument] = {
                "coverage": round(coverage, 4),
                "polyphony": round(polyphony, 4),
                "density": round(density, 4),
                "support": round(support, 4),
                "score": round(score, 4),
            }
        scores.append(window_scores)
        profiles.append(window_profiles)
    return scores, profiles


def _decode_instrument_path(scores: list[dict[str, float]]) -> list[str | None]:
    if not scores:
        return []
    states = sorted({instrument for window in scores for instrument in window})
    if not states:
        return [None] * len(scores)
    previous_scores: dict[str | None, float] = {None: 0.0}
    backpointers: list[dict[str | None, str | None]] = []
    for window_index, local_scores in enumerate(scores):
        active_states: list[str | None] = list(local_scores) or [None]
        current_scores: dict[str | None, float] = {}
        pointers: dict[str | None, str | None] = {}
        for state in active_states:
            local = -0.25 if state is None and local_scores else (0.15 if state is None else local_scores[state])
            best_previous = None
            best_score = -math.inf
            for previous_state, previous_score in previous_scores.items():
                if state == previous_state:
                    transition = 0.55 if state is not None else 0.05
                elif previous_state is None:
                    transition = -0.05
                elif state is None:
                    transition = -0.20
                else:
                    previous_active = previous_state in scores[window_index]
                    transition = -0.30 if previous_active else 0.05
                candidate_score = previous_score + transition + local
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = previous_state
            current_scores[state] = best_score
            pointers[state] = best_previous
        previous_scores = current_scores
        backpointers.append(pointers)
    state = max(previous_scores, key=previous_scores.get)
    path: list[str | None] = []
    for pointers in reversed(backpointers):
        path.append(state)
        state = pointers.get(state)
    path.reverse()
    # Remove one-window instrument flicker unless the surrounding instrument is absent.
    for index in range(1, len(path) - 1):
        if path[index - 1] == path[index + 1] != path[index]:
            surrounding = path[index - 1]
            if surrounding in scores[index]:
                path[index] = surrounding
    return path


def _cluster_onsets(notes: list[_Candidate], tolerance: float = 0.065) -> list[list[_Candidate]]:
    clusters: list[list[_Candidate]] = []
    for note in notes:
        if clusters and note.time - clusters[-1][0].time <= tolerance:
            clusters[-1].append(note)
        else:
            clusters.append([note])
    return clusters


def _choose_monophonic_notes(
    notes: list[_Candidate],
    instrument_path: list[str | None],
    window_seconds: float,
) -> list[dict]:
    routed = [
        note for note in notes
        if instrument_path[min(len(instrument_path) - 1, int(note.time / window_seconds))] == note.instrument
    ]
    selected: list[_Candidate] = []
    for cluster in _cluster_onsets(routed):
        previous = selected[-1] if selected else None
        window_midis = sorted(note.midi for note in cluster)
        median_pitch = window_midis[len(window_midis) // 2]

        def candidate_score(note: _Candidate) -> float:
            score = 2.2 * note.acoustic_support + 0.55 * note.confidence
            score += 0.20 * min(1.0, note.duration / 0.7)
            score += 0.10 / (1.0 + math.exp(-(note.midi - median_pitch) / 3.0))
            if previous is not None:
                gap = max(0.0, note.time - previous.end)
                jump = abs(note.midi - previous.midi)
                if gap < 1.5:
                    score -= 0.07 * max(0.0, jump - 2.0)
                    score -= 0.35 if jump > 12.0 else 0.0
                if note.instrument == previous.instrument:
                    score += 0.20
                elif gap < 0.25:
                    score -= 0.20
            return score

        chosen = max(cluster, key=candidate_score)
        if (
            previous
            and chosen.instrument == previous.instrument
            and abs(chosen.midi - previous.midi) < 0.35
            and chosen.time <= previous.end + 0.09
        ):
            selected[-1] = _Candidate(
                previous.time,
                max(previous.end, chosen.end) - previous.time,
                previous.midi,
                previous.instrument,
                max(previous.confidence, chosen.confidence),
                max(previous.acoustic_support, chosen.acoustic_support),
            )
            continue
        if previous and chosen.time < previous.end:
            shortened = chosen.time - previous.time
            if shortened >= 0.06:
                selected[-1] = _Candidate(
                    previous.time, shortened, previous.midi, previous.instrument,
                    previous.confidence, previous.acoustic_support,
                )
            else:
                selected.pop()
        selected.append(chosen)

    return [{
        "time": round(note.time, 4),
        "duration": round(note.duration, 4),
        "midi": round(note.midi, 3),
        "confidence": round(0.55 * note.confidence + 0.45 * note.acoustic_support, 4),
        "attack": True,
        "instrument": note.instrument,
    } for note in selected if note.duration >= 0.06]


def _align_to_predominant_contour(notes: list[_Candidate], contour_notes: list[dict]) -> list[dict]:
    """Use a perceptual F0 contour to choose attacks from a polyphonic AMT.

    Instrument labels are deliberately ignored here: real-world transcription
    models can assign a violin phrase to a guitar or bass program while still
    detecting useful pitches and onset times.
    """
    contour = sorted(
        (
            _Candidate(
                time=float(item["time"]),
                duration=max(0.06, float(item.get("duration", 0.1))),
                midi=float(item["midi"]),
                instrument="predominant_contour",
                confidence=_clamp(float(item.get("confidence", 0.65))),
                acoustic_support=_clamp(float(item.get("confidence", 0.65))),
            )
            for item in contour_notes
            if float(item.get("duration", 0.0)) >= 0.06
        ),
        key=lambda note: note.time,
    )
    if not contour:
        return []

    aligned: list[_Candidate] = []
    used: set[int] = set()
    for anchor in contour:
        overlapping = [
            (index, note) for index, note in enumerate(notes)
            if index not in used
            and note.time < anchor.end + 0.12
            and note.end > anchor.time - 0.08
            and abs(note.midi - anchor.midi) <= 2.25
        ]
        if not overlapping:
            aligned.append(anchor)
            continue
        groups: list[list[tuple[int, _Candidate]]] = []
        for indexed in overlapping:
            if groups and indexed[1].time - groups[-1][0][1].time <= 0.055:
                groups[-1].append(indexed)
            else:
                groups.append([indexed])
        for group in groups:
            index, selected = max(
                group,
                key=lambda indexed: (
                    1.8 * indexed[1].acoustic_support
                    + 0.4 * indexed[1].confidence
                    - 0.55 * abs(indexed[1].midi - anchor.midi)
                ),
            )
            used.add(index)
            aligned.append(selected)

    aligned.sort(key=lambda note: (note.time, -note.acoustic_support, -note.midi))
    monophonic: list[_Candidate] = []
    for note in aligned:
        if monophonic and note.time - monophonic[-1].time <= 0.055:
            previous = monophonic[-1]
            previous_score = previous.acoustic_support + 0.2 * previous.confidence
            note_score = note.acoustic_support + 0.2 * note.confidence
            if note_score > previous_score:
                monophonic[-1] = note
            continue
        if (
            monophonic
            and note.instrument == monophonic[-1].instrument
            and abs(note.midi - monophonic[-1].midi) < 0.35
            and note.time <= monophonic[-1].end + 0.04
        ):
            previous = monophonic[-1]
            monophonic[-1] = _Candidate(
                previous.time,
                max(previous.end, note.end) - previous.time,
                previous.midi,
                previous.instrument,
                max(previous.confidence, note.confidence),
                max(previous.acoustic_support, note.acoustic_support),
            )
            continue
        monophonic.append(note)

    result: list[dict] = []
    for index, note in enumerate(monophonic):
        duration = note.duration
        if index + 1 < len(monophonic) and monophonic[index + 1].time < note.end:
            duration = max(0.06, monophonic[index + 1].time - note.time)
        result.append({
            "time": round(note.time, 4),
            "duration": round(duration, 4),
            "midi": round(note.midi, 3),
            "confidence": round(0.55 * note.confidence + 0.45 * note.acoustic_support, 4),
            "attack": note.instrument != "predominant_contour",
            "instrument": note.instrument,
        })
    return result


def decode_lead_melody(
    candidates: list[dict],
    duration: float,
    window_seconds: float = 2.5,
    contour_notes: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Select one human-like lead line from instrument-tagged note candidates.

    The decoder first follows the most melody-like instrument through time,
    then resolves polyphony inside that route into one note at each onset. It
    permits an instrument handoff at phrase-scale window boundaries and never
    invents or copies notes that were not transcribed.
    """
    notes = _normalize_candidates(candidates)
    if not notes:
        return [], {"candidate_count": 0, "instrument_path": [], "windows": []}
    if contour_notes:
        melody = _align_to_predominant_contour(notes, contour_notes)
        return melody, {
            "candidate_count": len(notes),
            "selected_count": len(melody),
            "contour_note_count": len(contour_notes),
            "contour_anchored": True,
            "instrument_path": [],
            "windows": [],
        }
    scores, profiles = _instrument_window_scores(notes, duration, window_seconds)
    instrument_path = _decode_instrument_path(scores)
    melody = _choose_monophonic_notes(notes, instrument_path, window_seconds)
    compact_path: list[dict] = []
    for index, instrument in enumerate(instrument_path):
        if compact_path and compact_path[-1]["instrument"] == instrument:
            compact_path[-1]["end"] = round(min(duration, (index + 1) * window_seconds), 3)
        else:
            compact_path.append({
                "start": round(index * window_seconds, 3),
                "end": round(min(duration, (index + 1) * window_seconds), 3),
                "instrument": instrument,
            })
    return melody, {
        "candidate_count": len(notes),
        "selected_count": len(melody),
        "window_seconds": window_seconds,
        "instrument_path": compact_path,
        "windows": profiles,
    }
