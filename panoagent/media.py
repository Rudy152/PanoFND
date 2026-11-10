from __future__ import annotations

import hashlib
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
import warnings

from .schemas import NewsSample


def _truncate_head_tail(value: str, max_chars: int | None) -> str:
    if max_chars is None or len(value) <= max_chars:
        return value
    if max_chars < 64:
        raise ValueError("text limits must be at least 64 characters")
    marker = "\n...[TRUNCATED FOR LOCAL CONTEXT BUDGET]...\n"
    remaining = max_chars - len(marker)
    head = remaining // 2
    tail = remaining - head
    return value[:head] + marker + value[-tail:]


_DECODE_LOCK = threading.Lock()


class VideoFrameExtractor:
    """Extract a deterministic, uniformly spaced frame set and cache it on disk."""

    def __init__(
        self,
        cache_root: str | Path,
        frame_count: int = 16,
        max_edge: int = 768,
        isolate: bool = True,
        timeout_seconds: float = 1800.0,
    ):
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        self.cache_root = Path(cache_root)
        self.frame_count = frame_count
        self.max_edge = max_edge
        self.isolate = isolate
        self.timeout_seconds = timeout_seconds

    def _output_paths(self, sample_id: str, video_path: Path) -> list[Path]:
        # The source-content fingerprint prevents an updated dataset from
        # silently reusing frames extracted from an older file with the same
        # sample ID and filename.
        digest = hashlib.sha256()
        with video_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprint = digest.hexdigest()[:16]
        folder = (
            self.cache_root
            / sample_id
            / f"{video_path.stem}_{fingerprint}"
            / f"n{self.frame_count}_edge{self.max_edge}"
        )
        return [folder / f"frame_{index:02d}.jpg" for index in range(self.frame_count)]

    def extract(self, sample_id: str, video_path: Path) -> tuple[Path, ...]:
        if not video_path.is_file():
            return ()
        expected = self._output_paths(sample_id, video_path)
        existing = tuple(path for path in expected if path.is_file())
        if len(existing) == self.frame_count:
            return existing

        try:
            import av
        except ImportError as exc:
            raise RuntimeError("Video frame mode requires PyAV (`pip install av`).") from exc

        folder = expected[0].parent
        folder.mkdir(parents=True, exist_ok=True)
        failed_marker = folder / "decode_failed.txt"
        if failed_marker.is_file():
            raise RuntimeError(f"video decoder previously crashed on {video_path.name}")
        # PyAV/FFmpeg decoding from several worker threads crashed the whole process with
        # a Windows access violation (0xC0000005) under --workers 2. Decode one at a time.
        with _DECODE_LOCK:
            existing = tuple(path for path in expected if path.is_file())
            if len(existing) == self.frame_count:
                return existing
            if not self.isolate:
                return self._decode(av, video_path, expected)
            # Some files (e.g. video_1669_1.mp4) segfault FFmpeg even single-threaded. A
            # native crash cannot be caught in-process and killed whole runs, so decode
            # in a child process and turn a crash into an ordinary, recoverable error.
            completed = subprocess.run(
                [
                    sys.executable, "-m", "panoagent.media",
                    str(video_path), str(folder), str(self.frame_count), str(self.max_edge),
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                failed_marker.write_text(
                    f"exit code {completed.returncode}\n{completed.stderr[-2000:]}",
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"video decoder crashed on {video_path.name} (exit {completed.returncode})"
                )
            return tuple(path.resolve() for path in expected if path.is_file())

    def _decode(self, av, video_path: Path, expected: list[Path]) -> tuple[Path, ...]:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            total = int(stream.frames or 0)
            if total <= 0:
                decoded = list(container.decode(stream))
                total = len(decoded)
                if not decoded:
                    return ()
                indices = self._uniform_indices(total)
                selected = [decoded[index] for index in indices]
            else:
                indices = set(self._uniform_indices(total))
                selected = []
                for index, frame in enumerate(container.decode(stream)):
                    if index in indices:
                        selected.append(frame)
                    if index >= max(indices):
                        break

        outputs: list[Path] = []
        for output_index, frame in enumerate(selected):
            image = frame.to_image()
            image.thumbnail((self.max_edge, self.max_edge))
            path = expected[output_index]
            image.convert("RGB").save(path, format="JPEG", quality=90)
            outputs.append(path.resolve())
        return tuple(outputs)

    def _uniform_indices(self, total: int) -> list[int]:
        if total <= self.frame_count:
            return list(range(total))
        if self.frame_count == 1:
            return [total // 2]
        return [round(index * (total - 1) / (self.frame_count - 1)) for index in range(self.frame_count)]


def prepare_sample_media(
    sample: NewsSample,
    mode: str,
    frame_extractor: VideoFrameExtractor | None = None,
    max_text_chars: int | None = None,
    max_transcript_chars: int | None = None,
) -> NewsSample:
    """Return an inference view without mutating the original dataset.

    native: send original image/audio/video media.
    transcript: send images plus text/audio transcript, omitting raw audio/video.
    frames: send images plus cached uniformly sampled video frames and transcript.
    """
    sample = replace(
        sample,
        text=_truncate_head_tail(sample.text, max_text_chars),
        transcript=_truncate_head_tail(sample.transcript, max_transcript_chars),
    )
    if mode == "native":
        return sample
    if mode == "transcript":
        return replace(sample, audio_paths=(), video_paths=())
    if mode == "frames":
        if frame_extractor is None:
            raise ValueError("frames mode requires a VideoFrameExtractor")
        frames: list[Path] = []
        for video_path in sample.video_paths:
            try:
                frames.extend(frame_extractor.extract(sample.id, video_path))
            except Exception as exc:
                # A corrupt media file must not terminate an entire evaluation
                # partition. Keep the article/transcript and any other images,
                # while making the local degradation visible in the run log.
                warnings.warn(
                    f"Skipping undecodable video for sample {sample.id}: "
                    f"{video_path.name} ({type(exc).__name__}: {exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )
        images = tuple(path for path in sample.image_paths if path.is_file()) + tuple(frames)
        return replace(sample, image_paths=images, audio_paths=(), video_paths=())
    raise ValueError(f"Unknown media mode: {mode}")


def _child_main(argv: list[str]) -> int:
    """Child-process entry: decode one video into an existing frame folder."""
    import av

    video_path, folder, frame_count, max_edge = argv
    extractor = VideoFrameExtractor(Path(folder).parent, int(frame_count), int(max_edge), isolate=False)
    expected = [Path(folder) / f"frame_{index:02d}.jpg" for index in range(int(frame_count))]
    extractor._decode(av, Path(video_path), expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv[1:]))
