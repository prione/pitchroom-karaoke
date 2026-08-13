import json
import tempfile
import unittest
from pathlib import Path

from local_backend.muscriptor_adapter import transcribe_multi_instrument


class MuScriptorAdapterTest(unittest.TestCase):
    def test_cached_instrument_events_are_decoded_without_reloading_the_model(self) -> None:
        events = [
            {"time": 0.0, "duration": 0.4, "midi": 72, "instrument": "piano", "acoustic_support": 0.9},
            {"time": 0.5, "duration": 0.4, "midi": 74, "instrument": "piano", "acoustic_support": 0.9},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "muscriptor-events.json").write_text(json.dumps(events), encoding="utf-8")

            notes, metadata = transcribe_multi_instrument(
                Path("audio-is-not-read-when-support-is-cached.wav"),
                output_dir,
                1.0,
                lambda _value, _message: None,
            )

        self.assertEqual(metadata["status"], "cache")
        self.assertEqual([note["midi"] for note in notes], [72, 74])


if __name__ == "__main__":
    unittest.main()
