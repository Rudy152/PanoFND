from __future__ import annotations

import torch

from panoagent.miragenews_text import label_from_fake_probability, load_text_classifier


def test_label_from_fake_probability_uses_strict_author_threshold() -> None:
    assert label_from_fake_probability(0.6097, 0.6097) == "REAL"
    assert label_from_fake_probability(0.6098, 0.6097) == "FAKE"


def test_load_text_classifier_uses_safe_official_checkpoint_shape(tmp_path) -> None:
    checkpoint = tmp_path / "text.pt"
    torch.save(
        {
            "model_state_dict": {
                "linear.weight": torch.zeros((1, 768)),
                "linear.bias": torch.zeros(1),
            },
            "best_threshold": 0.6097,
        },
        checkpoint,
    )
    classifier, threshold = load_text_classifier(checkpoint, "cpu")
    output = classifier(torch.zeros((2, 768)))
    assert threshold == 0.6097
    assert output.shape == (2, 1)
    assert torch.allclose(output, torch.full((2, 1), 0.5))
