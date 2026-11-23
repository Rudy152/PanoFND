from pathlib import Path

from panoagent.fact_r1_video import (
    EXPECTED_WEIGHT_FILES,
    build_user_prompt,
    parse_fact_r1_answer,
    truncate_text,
)
from panoagent.schemas import NewsSample


def _sample() -> NewsSample:
    return NewsSample(
        id="1",
        modalities=("text", "image", "video"),
        title="Example title",
        text="Private article body that is outside the Fact-R1 protocol.",
        image_paths=(Path("still.jpg"),),
        video_paths=(Path("clip.mp4"),),
        audio_paths=(Path("clip.wav"),),
        label="fake",
        source="",
        publish_time="2025-01-01",
        description="This is fake news and leaks the gold label.",
        transcript="Spoken evidence",
    )


def test_released_prompt_uses_only_title_and_transcript() -> None:
    prompt = build_user_prompt(_sample())
    assert "Example title" in prompt
    assert "Spoken evidence" in prompt
    assert "article body" not in prompt
    assert "leaks the gold" not in prompt
    assert "2025-01-01" not in prompt
    assert "Please answer yes or no" in prompt
    assert "<think> </think>" in prompt
    assert "<answer> </answer>" in prompt


def test_fact_r1_yes_means_fake_and_no_means_real() -> None:
    assert parse_fact_r1_answer(
        "<think>Mismatch.</think><answer>Yes.</answer>"
    ) == ("FAKE", True)
    assert parse_fact_r1_answer(
        "<think>Consistent.</think><answer> No. </answer>"
    ) == ("REAL", True)


def test_parser_marks_nonofficial_fallback_and_rejects_ambiguous_text() -> None:
    assert parse_fact_r1_answer("yes") == ("FAKE", False)
    assert parse_fact_r1_answer("The answer may be yes") == ("ERROR", False)
    assert parse_fact_r1_answer("<answer>maybe</answer>") == ("ERROR", False)


def test_transcript_truncation_preserves_both_ends() -> None:
    value = "A" * 500 + "B" * 500
    truncated = truncate_text(value, 200)
    assert len(truncated) == 200
    assert truncated.startswith("A")
    assert truncated.endswith("B")
    assert "MIDDLE TRUNCATED" in truncated


def test_checkpoint_manifest_pins_four_official_shards() -> None:
    assert list(EXPECTED_WEIGHT_FILES) == [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]
    assert all(len(str(row["sha256"])) == 64 for row in EXPECTED_WEIGHT_FILES.values())
