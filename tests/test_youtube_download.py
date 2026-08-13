import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yt_dlp import DownloadError

from local_backend.main import _download_youtube_audio


class YouTubeDownloadFallbackTest(unittest.TestCase):
    def test_retries_with_an_explicit_client_after_a_403(self) -> None:
        attempts = []

        class FakeDownloader:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                self.assert_download = download
                attempts.append(self.options.get("extractor_args"))
                if len(attempts) == 1:
                    raise DownloadError("HTTP Error 403: Forbidden")
                output = Path(self.options["outtmpl"].replace("%(ext)s", "m4a"))
                output.write_bytes(b"downloaded")
                return {"id": "test", "ext": "m4a"}

            def prepare_filename(self, _info):
                return self.options["outtmpl"].replace("%(ext)s", "m4a")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "local_backend.main.YoutubeDL", FakeDownloader,
        ):
            info, path = _download_youtube_audio(
                "https://www.youtube.com/watch?v=test",
                Path(temporary),
                lambda _progress: None,
            )

        self.assertEqual(info["id"], "test")
        self.assertEqual(path.name, "source.m4a")
        self.assertEqual(attempts[0], None)
        self.assertEqual(attempts[1]["youtube"]["player_client"], ["android_vr"])


if __name__ == "__main__":
    unittest.main()
