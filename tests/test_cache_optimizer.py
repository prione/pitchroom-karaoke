from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from local_backend.cache_optimizer import compact_youtube_source, optimize_analysis_cache


def _write_test_wav(path: Path, seconds: float = 0.3) -> None:
    sample_rate = 16_000
    frames = bytearray()
    for index in range(round(sample_rate * seconds)):
        value = round(8_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend(struct.pack("<hh", value, value))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)


class CacheOptimizerTests(unittest.TestCase):
    def test_losslessly_compresses_playback_and_removes_intermediate_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache" / "song"
            cache.mkdir(parents=True)
            for filename in (
                "accompaniment.wav", "vocals.wav", "melody-source.wav",
                "attack-source.wav", "continuous-source.wav",
            ):
                _write_test_wav(cache / filename)
            result = {
                "version": "test", "mode": "instrumental", "duration": 0.3,
                "notes": [{"time": 0, "midi": 69}],
                "accompaniment_path": str(cache / "accompaniment.wav"),
                "vocals_path": str(cache / "vocals.wav"),
                "melody_source_path": str(cache / "melody-source.wav"),
            }
            (cache / "analysis.json").write_text(json.dumps(result), encoding="utf-8")

            outcome = optimize_analysis_cache(result)

            optimized = outcome["result"]
            self.assertTrue(Path(optimized["accompaniment_path"]).is_file())
            self.assertEqual(Path(optimized["accompaniment_path"]).suffix, ".flac")
            self.assertFalse((cache / "accompaniment.wav").exists())
            self.assertFalse((cache / "vocals.wav").exists())
            self.assertFalse((cache / "melody-source.wav").exists())
            self.assertFalse((cache / "attack-source.wav").exists())
            self.assertFalse((cache / "continuous-source.wav").exists())
            saved = json.loads((cache / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["accompaniment_path"], optimized["accompaniment_path"])
            self.assertGreater(outcome["freed_bytes"], 0)

    def test_keeps_melody_source_until_client_transcription_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache" / "song"
            cache.mkdir(parents=True)
            for filename in ("accompaniment.wav", "vocals.wav", "melody-source.wav", "attack-source.wav", "continuous-source.wav"):
                _write_test_wav(cache / filename)
            result = {
                "mode": "instrumental", "notes": [],
                "accompaniment_path": str(cache / "accompaniment.wav"),
                "vocals_path": str(cache / "vocals.wav"),
                "melody_source_path": str(cache / "melody-source.wav"),
            }

            outcome = optimize_analysis_cache(result, keep_melody_source=True)

            self.assertTrue(Path(outcome["result"]["melody_source_path"]).is_file())
            self.assertFalse((cache / "attack-source.wav").exists())
            self.assertFalse((cache / "continuous-source.wav").exists())

    def test_replaces_youtube_conversion_wav_with_original_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory) / "uploads"
            job = upload_root / "job"
            job.mkdir(parents=True)
            converted = job / "source.wav"
            original = job / "source.m4a"
            converted.write_bytes(b"wav" * 100)
            original.write_bytes(b"m4a" * 10)

            retained, freed = compact_youtube_source(converted, upload_root)

            self.assertEqual(retained, original.resolve())
            self.assertFalse(converted.exists())
            self.assertEqual(freed, 300)


if __name__ == "__main__":
    unittest.main()
