import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from local_backend.analyzer import MelodyNote, _cache_key, _choose_continuous_candidates, _fill_detected_contour_gaps, _measure_source_activity, _notes_from_track, _remove_interleaved_lower_voice, _route_instrumental_modules, _select_high_resolution_melody, _should_use_multi_instrument, _simplify_vocal_notes, _vocal_hypothesis_quality
from local_backend.lead_melody import decode_lead_melody


class InstrumentAttackSegmentationTest(unittest.TestCase):
    def test_same_pitch_is_split_at_detected_attacks(self) -> None:
        midi = np.full(90, 77.0, dtype=float)
        confidence = np.full(90, 0.8, dtype=float)

        notes = _notes_from_track(midi, confidence, 0.02, np.array([30, 60]))

        self.assertEqual([note.midi for note in notes], [77.0, 77.0, 77.0])
        self.assertEqual([note.attack for note in notes], [False, True, True])
        self.assertEqual([note.time for note in notes], [0.0, 0.6, 1.2])

    def test_attack_moves_to_first_voiced_frame_after_a_tiny_gap(self) -> None:
        midi = np.full(70, 77.0, dtype=float)
        midi[30] = np.nan
        confidence = np.full(70, 0.8, dtype=float)

        notes = _notes_from_track(midi, confidence, 0.02, np.array([30]))

        self.assertEqual(len(notes), 2)
        self.assertTrue(notes[1].attack)
        self.assertEqual(notes[1].time, 0.62)


class VocalNoteSimplificationTest(unittest.TestCase):
    def test_short_vibrato_turn_is_merged_into_target_note(self) -> None:
        notes = [
            MelodyNote(0.0, 0.42, 69.0, 0.91),
            MelodyNote(0.42, 0.12, 70.0, 0.75),
            MelodyNote(0.54, 0.46, 69.0, 0.90),
        ]

        result = _simplify_vocal_notes(notes)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].midi, 69.0)
        self.assertEqual(result[0].duration, 1.0)

    def test_deliberate_vocal_pitch_change_is_preserved(self) -> None:
        notes = [
            MelodyNote(0.0, 0.4, 69.0, 0.9),
            MelodyNote(0.4, 0.24, 72.0, 0.88),
            MelodyNote(0.64, 0.42, 74.0, 0.92),
        ]

        result = _simplify_vocal_notes(notes)

        self.assertEqual([note.midi for note in result], [69.0, 72.0, 74.0])

    def test_sparse_instrument_leakage_does_not_make_a_song_vocal(self) -> None:
        notes = [MelodyNote(index * 2.5, 0.3, 72, 0.8) for index in range(110)]

        quality = _vocal_hypothesis_quality(notes, 291.9, 0.1199, 0.1737)

        self.assertFalse(quality["accepted"])

    def test_dense_isolated_voice_contour_is_accepted(self) -> None:
        notes = [MelodyNote(index * 0.35, 0.3, 65 + index % 8, 0.88) for index in range(690)]

        quality = _vocal_hypothesis_quality(notes, 278.7, 0.55, 0.93)

        self.assertTrue(quality["accepted"])


class HighResolutionMelodySelectionTest(unittest.TestCase):
    def test_uses_new_skyline_attacks_without_repeating_sustained_harmony(self) -> None:
        events = [
            {"onset_time": 0.0, "offset_time": 1.5, "midi_note": 72, "velocity": 60},
            {"onset_time": 0.0, "offset_time": 1.5, "midi_note": 60, "velocity": 70},
            {"onset_time": 0.3, "offset_time": 0.8, "midi_note": 70, "velocity": 75},
            {"onset_time": 0.6, "offset_time": 1.0, "midi_note": 72, "velocity": 78},
        ]

        result, quality = _select_high_resolution_melody(events)

        self.assertEqual([note["midi"] for note in result], [72, 70, 72])
        self.assertEqual([note["time"] for note in result], [0.0, 0.3, 0.6])
        self.assertEqual(result[0]["duration"], 0.3)
        self.assertEqual(quality["coverage"], 1.0)

    def test_rejects_a_hallucinated_upper_note_without_spectral_support(self) -> None:
        events = [
            {"onset_time": 0.0, "offset_time": 0.4, "midi_note": 91, "velocity": 90, "spectral_rank": 20},
            {"onset_time": 0.0, "offset_time": 0.4, "midi_note": 84, "velocity": 70, "spectral_rank": 2},
        ]

        result, quality = _select_high_resolution_melody(events)

        self.assertEqual([note["midi"] for note in result], [84])
        self.assertEqual(quality["top3"], 1.0)

    def test_routes_attack_and_continuous_sections_from_acoustic_shape(self) -> None:
        attack_notes = [
            {"time": 1.0, "duration": 0.3, "midi": 80, "confidence": 0.9},
            {"time": 11.0, "duration": 0.3, "midi": 81, "confidence": 0.9},
        ]
        continuous_notes = [
            {"time": 1.0, "duration": 0.8, "midi": 70, "confidence": 0.8},
            {"time": 11.0, "duration": 0.8, "midi": 71, "confidence": 0.8},
        ]
        quality = {"segments": [
            {"start": 0, "end": 10, "top3": 0.95, "density": 3.2, "median_duration": 0.3},
            {"start": 10, "end": 20, "top3": 0.8, "density": 1.2, "median_duration": 1.1},
        ]}

        activity = [
            {"continuous_ratio": 0.01, "continuous_active": False},
            {"continuous_ratio": 1.2, "continuous_active": True},
        ]
        result, routing = _route_instrumental_modules(attack_notes, continuous_notes, quality, 20, activity)

        self.assertEqual([note["midi"] for note in result], [80, 71])
        self.assertEqual([segment["method"] for segment in routing], ["attack", "continuous"])

    def test_routes_a_pure_sustained_piece_without_attack_quality_segments(self) -> None:
        continuous_notes = [
            {"time": 1.0, "duration": 0.8, "midi": 70, "confidence": 0.8},
            {"time": 11.0, "duration": 0.8, "midi": 72, "confidence": 0.8},
        ]
        activity = [
            {"start": 0, "continuous_ratio": 10.0, "continuous_active": True},
            {"start": 10, "continuous_ratio": 10.0, "continuous_active": True},
        ]

        result, routing = _route_instrumental_modules([], continuous_notes, {}, 20, activity)

        self.assertEqual([note["midi"] for note in result], [70, 72])
        self.assertEqual([segment["method"] for segment in routing], ["continuous", "continuous"])
        self.assertEqual(len(routing), 2)

    def test_source_activity_is_scale_relative(self) -> None:
        attack = np.ones((2, 200), dtype=np.float32) * 0.4
        continuous = np.concatenate((
            np.zeros((2, 100), dtype=np.float32),
            np.ones((2, 100), dtype=np.float32) * 0.5,
        ), axis=1)

        activity = _measure_source_activity(attack, continuous, 10, 20)

        self.assertFalse(activity[0]["continuous_active"])
        self.assertGreater(activity[1]["continuous_ratio"], 1.0)

    def test_prefers_a_complete_stable_residual_over_a_collapsed_low_tracker(self) -> None:
        foreground = [
            {"time": 0.0, "duration": 0.8, "midi": 51, "confidence": 0.5},
            {"time": 2.0, "duration": 0.8, "midi": 53, "confidence": 0.5},
            {"time": 4.0, "duration": 0.8, "midi": 49, "confidence": 0.5},
        ]
        residual = [
            {"time": 0.0, "duration": 2.0, "midi": 70, "confidence": 0.5},
            {"time": 2.0, "duration": 2.0, "midi": 72, "confidence": 0.5},
            {"time": 4.0, "duration": 2.0, "midi": 70, "confidence": 0.5},
            {"time": 6.0, "duration": 2.0, "midi": 73, "confidence": 0.5},
        ]

        notes, routing = _choose_continuous_candidates(foreground, residual, 10)

        self.assertEqual(routing[0]["method"], "residual")
        self.assertEqual([note["midi"] for note in notes], [70, 72, 70, 73])

    def test_prefers_the_foreground_register_during_a_fading_outro(self) -> None:
        collapsed = [{"time": 0.0, "duration": 3.5, "midi": 42, "confidence": 0.5}]
        foreground = [
            {"time": 0.0, "duration": 1.5, "midi": 75, "confidence": 0.4},
            {"time": 2.0, "duration": 1.5, "midi": 77, "confidence": 0.4},
            {"time": 4.0, "duration": 1.0, "midi": 75, "confidence": 0.4},
        ]

        notes, routing = _choose_continuous_candidates(collapsed, foreground, 10)

        self.assertEqual(routing[0]["method"], "residual")
        self.assertEqual([note["midi"] for note in notes], [75, 77, 75])

    def test_continuous_coverage_can_win_in_a_mixed_attack_window(self) -> None:
        attack = [{"time": index, "duration": 0.2, "midi": 60 + index % 2, "confidence": 0.8} for index in range(10)]
        continuous = [{"time": 0.0, "duration": 2.0, "midi": 70, "confidence": 0.8},
                      {"time": 2.0, "duration": 2.0, "midi": 72, "confidence": 0.8},
                      {"time": 4.0, "duration": 2.0, "midi": 70, "confidence": 0.8},
                      {"time": 6.0, "duration": 2.0, "midi": 73, "confidence": 0.8}]
        quality = {"segments": [{"start": 0, "top3": 0.95, "density": 3.0, "median_duration": 0.3}]}
        activity = [{"start": 0, "continuous_ratio": 0.55, "continuous_active": True}]

        notes, routing = _route_instrumental_modules(attack, continuous, quality, 10, activity)

        self.assertEqual(routing[0]["method"], "continuous")
        self.assertEqual([note["midi"] for note in notes], [70, 72, 70, 73])

    def test_returns_to_upper_attacks_when_continuous_tracker_collapses_to_bass(self) -> None:
        attack = [
            {"time": float(index), "duration": 0.7, "midi": 72 + index % 3, "confidence": 0.8}
            for index in range(10)
        ]
        continuous = [
            {"time": float(index), "duration": 0.8, "midi": 49 + index % 2, "confidence": 0.6}
            for index in range(10)
        ]
        quality = {"segments": [
            {"start": 0, "top3": 0.9, "density": 2.0, "median_duration": 0.7},
        ]}
        activity = [{"start": 0, "continuous_ratio": 0.7, "continuous_active": True}]

        notes, routing = _route_instrumental_modules(attack, continuous, quality, 10, activity)

        self.assertEqual(routing[0]["method"], "attack")
        self.assertTrue(routing[0]["register_collapsed"])
        self.assertEqual([note["midi"] for note in notes], [72, 73, 74, 72, 73, 74, 72, 73, 74, 72])

    def test_removes_an_isolated_low_accompaniment_attack_after_a_handoff(self) -> None:
        attack = [
            {"time": 0.0, "duration": 0.4, "midi": 72, "confidence": 0.8},
            {"time": 0.5, "duration": 0.1, "midi": 48, "confidence": 0.8},
            {"time": 1.0, "duration": 0.4, "midi": 74, "confidence": 0.8},
        ]

        notes, _ = _route_instrumental_modules(attack, [], {}, 2, [])

        self.assertEqual([note["midi"] for note in notes], [72, 74])

    def test_removes_multiple_lower_notes_interleaved_with_an_upper_line(self) -> None:
        notes = [
            {"time": 0.0, "duration": 0.4, "midi": 72, "confidence": 0.8},
            {"time": 0.3, "duration": 0.2, "midi": 48, "confidence": 0.8},
            {"time": 0.6, "duration": 0.2, "midi": 50, "confidence": 0.8},
            {"time": 0.9, "duration": 0.4, "midi": 74, "confidence": 0.8},
        ]

        filtered = _remove_interleaved_lower_voice(notes)

        self.assertEqual([note["midi"] for note in filtered], [72, 74])

    def test_preserves_a_real_handoff_to_a_lower_register(self) -> None:
        notes = [
            {"time": 0.0, "duration": 0.4, "midi": 72, "confidence": 0.8},
            {"time": 0.8, "duration": 0.5, "midi": 55, "confidence": 0.8},
            {"time": 1.4, "duration": 0.5, "midi": 57, "confidence": 0.8},
        ]

        filtered = _remove_interleaved_lower_voice(notes)

        self.assertEqual([note["midi"] for note in filtered], [72, 55, 57])

    def test_rejects_a_dense_residual_contour_when_its_source_is_inaudible(self) -> None:
        foreground = [{"time": 0.0, "duration": 5.0, "midi": 70, "confidence": 0.5}]
        residual = [
            {"time": float(index), "duration": 0.8, "midi": 72 + index % 2, "confidence": 0.8}
            for index in range(10)
        ]
        foreground_activity = [{"start": 0, "continuous_rms": 0.02}]
        residual_activity = [{"start": 0, "continuous_rms": 0.0001}]

        notes, routing = _choose_continuous_candidates(
            foreground, residual, 10, foreground_activity, residual_activity,
        )

        self.assertEqual(routing[0]["method"], "foreground")
        self.assertFalse(routing[0]["residual_audible"])
        self.assertEqual([note["midi"] for note in notes], [70])

    def test_removes_an_isolated_bass_drop_from_a_contour(self) -> None:
        foreground = [
            {"time": 0.0, "duration": 0.5, "midi": 72, "confidence": 0.8},
            {"time": 0.5, "duration": 0.5, "midi": 74, "confidence": 0.8},
            {"time": 1.0, "duration": 0.5, "midi": 72, "confidence": 0.8},
            {"time": 1.5, "duration": 0.5, "midi": 48, "confidence": 0.8},
            {"time": 2.0, "duration": 0.5, "midi": 75, "confidence": 0.8},
            {"time": 2.5, "duration": 0.5, "midi": 74, "confidence": 0.8},
        ]

        notes, _ = _choose_continuous_candidates(foreground, [], 10)

        self.assertEqual([note["midi"] for note in notes], [72, 74, 72, 75, 74])

    def test_fills_a_short_gap_only_from_a_matching_detected_note(self) -> None:
        contour = [
            {"time": 0.0, "duration": 0.8, "midi": 72, "confidence": 0.8},
            {"time": 1.8, "duration": 0.8, "midi": 74, "confidence": 0.8},
        ]
        candidates = [
            {"time": 0.9, "duration": 0.35, "midi": 73, "confidence": 0.9},
            {"time": 1.3, "duration": 0.35, "midi": 52, "confidence": 0.9},
        ]

        filled, count = _fill_detected_contour_gaps(contour, candidates)

        self.assertEqual(count, 1)
        self.assertEqual([note["midi"] for note in filled], [72, 73, 74])

    def test_does_not_copy_notes_into_a_long_gap(self) -> None:
        contour = [
            {"time": 0.0, "duration": 0.5, "midi": 72, "confidence": 0.8},
            {"time": 3.0, "duration": 0.5, "midi": 74, "confidence": 0.8},
        ]
        candidates = [{"time": 1.0, "duration": 1.0, "midi": 73, "confidence": 0.9}]

        filled, count = _fill_detected_contour_gaps(contour, candidates)

        self.assertEqual(count, 0)
        self.assertEqual([note["midi"] for note in filled], [72, 74])


class MultiInstrumentLeadDecoderTest(unittest.TestCase):
    def test_routes_attack_dominant_music_away_from_the_slow_general_model(self) -> None:
        self.assertFalse(_should_use_multi_instrument("instrumental", 1.77))
        self.assertTrue(_should_use_multi_instrument("instrumental", 0.81))
        self.assertFalse(_should_use_multi_instrument("vocal", 0.2))

    def test_follows_a_phrase_handoff_from_piano_to_violin(self) -> None:
        candidates = []
        for index, pitch in enumerate([72, 74, 76, 77, 79]):
            candidates.append({
                "time": index * 0.5, "duration": 0.42, "midi": pitch,
                "instrument": "piano", "confidence": 0.85, "acoustic_support": 0.9,
            })
        # Piano accompaniment remains after the handoff but is chordal and weak.
        for index in range(5, 11):
            for pitch in (48, 55, 60):
                candidates.append({
                    "time": index * 0.5, "duration": 0.45, "midi": pitch,
                    "instrument": "piano", "confidence": 0.7, "acoustic_support": 0.35,
                })
        for index, pitch in enumerate([79, 81, 83, 84, 83, 81], start=5):
            candidates.append({
                "time": index * 0.5, "duration": 0.44, "midi": pitch,
                "instrument": "violin", "confidence": 0.85, "acoustic_support": 0.95,
            })

        melody, metadata = decode_lead_melody(candidates, 5.6, window_seconds=2.5)

        self.assertEqual([note["midi"] for note in melody[:5]], [72, 74, 76, 77, 79])
        self.assertEqual([note["midi"] for note in melody[5:]], [79, 81, 83, 84, 83, 81])
        self.assertEqual([segment["instrument"] for segment in metadata["instrument_path"]], ["piano", "violin"])

    def test_resolves_a_chord_to_one_detected_note_without_copying_events(self) -> None:
        candidates = [
            {"time": 0.0, "duration": 0.4, "midi": 60, "instrument": "piano", "acoustic_support": 0.3},
            {"time": 0.0, "duration": 0.4, "midi": 64, "instrument": "piano", "acoustic_support": 0.4},
            {"time": 0.0, "duration": 0.4, "midi": 67, "instrument": "piano", "acoustic_support": 0.9},
            {"time": 0.5, "duration": 0.4, "midi": 69, "instrument": "piano", "acoustic_support": 0.9},
        ]

        melody, _ = decode_lead_melody(candidates, 1.0)

        self.assertEqual([note["midi"] for note in melody], [67, 69])
        self.assertTrue(all(note["midi"] in {60, 64, 67, 69} for note in melody))

    def test_contour_anchor_ignores_wrong_instrument_labels_and_bass_arpeggios(self) -> None:
        candidates = []
        for index in range(8):
            candidates.append({
                "time": index * 0.25, "duration": 0.2, "midi": [48, 52, 55, 52][index % 4],
                "instrument": "electric_bass", "acoustic_support": 0.9,
            })
        for index, pitch in enumerate([76, 77, 79, 81]):
            candidates.append({
                "time": index * 0.5, "duration": 0.45, "midi": pitch,
                "instrument": "electric_bass", "acoustic_support": 0.8,
            })
        contour = [
            {"time": index * 0.5, "duration": 0.45, "midi": pitch, "confidence": 0.8}
            for index, pitch in enumerate([76, 77, 79, 81])
        ]

        melody, metadata = decode_lead_melody(candidates, 2.0, contour_notes=contour)

        self.assertEqual([note["midi"] for note in melody], [76, 77, 79, 81])
        self.assertTrue(metadata["contour_anchored"])

class ModelProfileCacheTest(unittest.TestCase):
    def test_analyzer_cache_changes_when_model_or_routing_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "test.wav"
            audio_path.write_bytes(b"transcription-cache-fingerprint")
            with mock.patch.dict(os.environ, {
                "PITCHROOM_MUSCRIPTOR_DEVICE": "cpu",
                "PITCHROOM_MUSCRIPTOR_WEIGHTS": "small",
                "PITCHROOM_MUSCRIPTOR_QUALITY": "balanced",
                "PITCHROOM_MUSCRIPTOR_ROUTING": "adaptive",
            }):
                baseline = _cache_key(audio_path)
            with mock.patch.dict(os.environ, {
                "PITCHROOM_MUSCRIPTOR_DEVICE": "cpu",
                "PITCHROOM_MUSCRIPTOR_WEIGHTS": "medium",
                "PITCHROOM_MUSCRIPTOR_QUALITY": "high",
                "PITCHROOM_MUSCRIPTOR_ROUTING": "always",
            }):
                upgraded = _cache_key(audio_path)
            self.assertNotEqual(baseline, upgraded)

    def test_routing_override_is_only_used_for_instrumentals(self) -> None:
        with mock.patch.dict(os.environ, {"PITCHROOM_MUSCRIPTOR_ROUTING": "always"}):
            self.assertTrue(_should_use_multi_instrument("instrumental", 5.0))
            self.assertFalse(_should_use_multi_instrument("vocal", 0.2))
        with mock.patch.dict(os.environ, {"PITCHROOM_MUSCRIPTOR_ROUTING": "off"}):
            self.assertFalse(_should_use_multi_instrument("instrumental", 0.2))


if __name__ == "__main__":
    unittest.main()
