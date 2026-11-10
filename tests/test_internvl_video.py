from pathlib import Path

from PIL import Image

from panoagent.internvl_video import (
    build_question,
    dynamic_preprocess,
    parse_binary_label,
    truncate_head_tail,
)
from panoagent.schemas import NewsSample


def _sample() -> NewsSample:
    return NewsSample(
        id="1",
        modalities=("text", "video"),
        title="Example title",
        text="Example article",
        image_paths=(),
        video_paths=(Path("clip.mp4"),),
        audio_paths=(),
        label="fake",
        source="",
        publish_time="2024-01-01",
        description="This is fake news and must never be shown to the model.",
        transcript="Spoken evidence",
    )


def test_parse_binary_label_is_strict() -> None:
    assert parse_binary_label("LABEL: real") == "REAL"
    assert parse_binary_label("FAKE") == "FAKE"
    assert parse_binary_label("possibly fake") == "ERROR"


def test_question_has_frames_but_excludes_gold_and_description() -> None:
    question = build_question(_sample(), ["Video Frame-1", "Video Frame-2"])
    assert question.count("<image>") == 2
    assert "Example title" in question
    assert "Spoken evidence" in question
    assert "must never be shown" not in question
    assert "GOLD" not in question


def test_dynamic_preprocess_limits_tiles() -> None:
    image = Image.new("RGB", (1600, 400), "white")
    tiles = dynamic_preprocess(image, max_tiles=4, use_thumbnail=True)
    assert len(tiles) == 5
    assert all(tile.size == (448, 448) for tile in tiles)


def test_truncate_head_tail_preserves_both_ends() -> None:
    value = "A" * 1000 + "B" * 1000
    truncated = truncate_head_tail(value, 300)
    assert len(truncated) == 300
    assert truncated.startswith("A")
    assert truncated.endswith("B")
    assert "MIDDLE TRUNCATED" in truncated
