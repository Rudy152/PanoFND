from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from panoagent.dataset import PanoFNDDataset


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ("image", "audio", "video"):
            (self.root / folder).mkdir()
        (self.root / "image" / "image_0676_1.jpg").write_bytes(b"image")
        (self.root / "audio" / "audio_0676_1.mp3").write_bytes(b"audio")
        (self.root / "audio" / "audio_0676_1.txt").write_text("transcript", encoding="utf-8")
        (self.root / "video" / "video_1201_1.mp4").write_bytes(b"video")
        rows = [
            {
                "id": "0676", "modalities": ["text", "image", "audio"],
                "title": "Title", "text": "Body", "image_paths": ["image_676_1.jpg"],
                "video_paths": [], "audio_paths": ["audio_676_1.mp3"], "label": "fake",
                "source": "https://example.test/a", "publish_time": "2025/01/01",
                "description": "This is fake news",
            },
            {
                "id": "0001", "modalities": ["text"], "title": "Real", "text": "Body",
                "image_paths": [], "video_paths": [], "audio_paths": [], "label": "true",
                "source": "https://example.test/a", "publish_time": "", "description": "Summary",
            },
            {
                "id": "1201", "modalities": ["text", "video"], "title": "Video", "text": "Body",
                "image_paths": [], "video_paths": ["vedio_1201_1.mp4"], "audio_paths": [],
                "label": "true", "source": "https://example.test/video", "publish_time": "",
                "description": "Summary",
            },
        ]
        (self.root / "data.json").write_text(json.dumps(rows), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_repairs_zero_padding_and_loads_transcript(self) -> None:
        sample = next(iter(PanoFNDDataset(self.root)))
        self.assertTrue(sample.image_paths[0].is_file())
        self.assertEqual(sample.transcript, "transcript")
        self.assertEqual(len(sample.path_repairs), 2)
        self.assertNotIn("fake news", sample.detector_text().lower())

    def test_repairs_video_prefix_typo(self) -> None:
        samples = list(PanoFNDDataset(self.root))
        self.assertTrue(samples[2].video_paths[0].is_file())

    def test_grouped_split_keeps_shared_source_together(self) -> None:
        dataset = PanoFNDDataset(self.root)
        split = dataset.create_grouped_split(seed=7)
        membership = {
            sample_id: partition
            for partition, ids in split["partitions"].items()
            for sample_id in ids
        }
        self.assertEqual(membership["0676"], membership["0001"])

    def test_clean_holdout_removes_entire_used_source_group(self) -> None:
        dataset = PanoFNDDataset(self.root)
        result = dataset.create_clean_holdout(
            candidate_ids=["0676", "0001", "1201"],
            excluded_ids=["0001"],
        )
        self.assertEqual(result["partitions"]["test"], ["1201"])
        self.assertEqual(result["removed_candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
