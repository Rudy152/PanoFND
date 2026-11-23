"""PanoAgent v2: manipulation-aware multimodal fact-checking.

Design (see V2_DESIGN.md):
  L0 labelled media (news images, timestamped video frames, native audio parts)
  L1 local forensic detectors (scores, not verdicts)
  L2 pairwise / unimodal evidence agents that see only the modalities they compare
  L4 a small trainable scorer over fixed evidence features (source-grouped CV)

The original `panoagent` package is left unchanged; v2 imports only its dataset
loader, split helpers and JSON parser.
"""

__version__ = "2.0.0"
