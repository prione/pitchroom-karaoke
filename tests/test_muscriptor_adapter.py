import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_backend.muscriptor_adapter import (
    _candidate_cache_path,
    muscriptor_settings,
    transcribe_multi_instrument,
)


class MuScriptorAdapterTest(unittest.TestCase):
    @mock.patch.dict(os.environ, {
        "PITCHROOM_MUSCRIPTOR_DEVICE": "cpu",
        "PITCHROOM_MUSCRIPTOR_WEIGHTS": "small",
        "PITCHROOM_MUSCRIPTOR_QUALITY": "balanced",
    })
    def test_cached_instrument_events_are_decoded_without_reloading_the_model(self) -> None:
        events = [
            {"time": 0.0, "duration": 0.4, "midi": 72, "instrument": "piano", "acoustic_support": 0.9},
            {"time": 0.5, "duration": 0.4, "midi": 74, "instrument": "piano", "acoustic_support": 0.9},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            _candidate_cache_path(output_dir, muscriptor_settings()).write_text(
                json.dumps(events), encoding="utf-8"
            )
            notes, metadata = transcribe_multi_instrument(
                Path("audio-is-not-read-when-support-is-cached.wav"),
                output_dir,
                1.0,
                lambda _value, _message: None,
            )
        self.assertEqual(metadata["status"], "cache")
        self.assertEqual(metadata["model"], "small")
        self.assertEqual([note["midi"] for note in notes], [72, 74])

    def test_auto_device_uses_medium_on_cuda(self) -> None:
        with mock.patch.dict(os.environ, {
            "PITCHROOM_MUSCRIPTOR_DEVICE": "auto",
            "PITCHROOM_MUSCRIPTOR_WEIGHTS": "",
            "PITCHROOM_MUSCRIPTOR_QUALITY": "balanced",
        }), mock.patch("torch.cuda.is_available", return_value=True):
            settings = muscriptor_settings()
        self.assertEqual(settings["device"], "cuda")
        self.assertEqual(settings["weights"], "medium")
        self.assertEqual(settings["batch_size"], 4)
        self.assertFalse(settings["prelude_forcing"])

    def test_high_quality_preserves_chunk_preludes(self) -> None:
        with mock.patch.dict(os.environ, {
            "PITCHROOM_MUSCRIPTOR_DEVICE": "cuda",
            "PITCHROOM_MUSCRIPTOR_WEIGHTS": "large",
            "PITCHROOM_MUSCRIPTOR_QUALITY": "high",
        }):
            settings = muscriptor_settings()
        self.assertEqual(settings["weights"], "large")
        self.assertTrue(settings["prelude_forcing"])
        self.assertEqual(settings["batch_size"], 1)

    def test_cache_isolated_across_quality_profiles(self) -> None:
        with mock.patch.dict(os.environ, {
            "PITCHROOM_MUSCRIPTOR_DEVICE": "cpu",
            "PITCHROOM_MUSCRIPTOR_WEIGHTS": "small",
            "PITCHROOM_MUSCRIPTOR_QUALITY": "balanced",
        }):
            balanced = _candidate_cache_path(Path("cache"), muscriptor_settings())
        with mock.patch.dict(os.environ, {
            "PITCHROOM_MUSCRIPTOR_DEVICE": "cpu",
            "PITCHROOM_MUSCRIPTOR_WEIGHTS": "small",
            "PITCHROOM_MUSCRIPTOR_QUALITY": "high",
        }):
            high = _candidate_cache_path(Path("cache"), muscriptor_settings())
        self.assertNotEqual(balanced, high)


if __name__ == "__main__":
    unittest.main()
