"""L0: labelled, content-addressed media for v2.

Every attachment carries an explicit label ([NEWS IMAGE 1], [VIDEO FRAME 3 @ 00:42],
[AUDIO PART 1/2], [VIDEO SOUNDTRACK PART 1/1]) so the model can tell news images
from video frames. Audio is sent natively (compressed mono MP3 parts) instead of
the dataset's speech-recognition transcript. All caches are keyed by file content
(SHA-256), never by path, so an in-place file replacement can't reuse stale output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

from panoagent.schemas import NewsSample

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

_INDEX_LOCK = threading.Lock()


class ContentHasher:
    """SHA-256 of file contents, memoised by (path, size, mtime) in a JSON index."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self._index: dict[str, str] = {}
        if self.index_path.is_file():
            try:
                self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except ValueError:
                self._index = {}

    def digest(self, path: Path) -> str:
        stat = path.stat()
        key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        with _INDEX_LOCK:
            if key in self._index:
                return self._index[key]
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                sha.update(chunk)
        value = sha.hexdigest()
        with _INDEX_LOCK:
            self._index[key] = value
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._index), encoding="utf-8")
            tmp.replace(self.index_path)
        return value


@dataclass(slots=True)
class Attachment:
    label: str
    kind: str  # "image" | "audio"
    path: Path
    digest: str


@dataclass(slots=True)
class PreparedItem:
    sample: NewsSample
    text: str  # title + article + publication date; empty when no text modality
    modalities: tuple[str, ...]  # effective: text/image/audio/video actually available
    attachments: dict[str, list[Attachment]] = field(default_factory=dict)
    media_errors: list[str] = field(default_factory=list)


def _budget(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "\n...[TRUNCATED]...\n"
    head = (max_chars - len(marker)) // 2
    return value[:head] + marker + value[-(max_chars - len(marker) - head):]


def _clock(seconds: float) -> str:
    seconds = int(round(seconds or 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class MediaPreparer:
    def __init__(
        self,
        cache_root: str | Path = "cache/v2_media",
        frame_count: int = 8,
        max_edge: int = 768,
        audio_chunk_seconds: float = 1200.0,
        audio_bitrate: int = 24000,
        max_text_chars: int = 12000,
        timeout_seconds: float = 1800.0,
    ):
        self.cache_root = Path(cache_root)
        self.frame_count = frame_count
        self.max_edge = max_edge
        self.audio_chunk_seconds = audio_chunk_seconds
        self.audio_bitrate = audio_bitrate
        self.max_text_chars = max_text_chars
        self.timeout_seconds = timeout_seconds
        self.hasher = ContentHasher(self.cache_root / "content_index.json")
        self._decode_lock = threading.Lock()

    def _run_worker(self, args: list[str], out_dir: Path) -> dict:
        meta_path = out_dir / "meta.json"
        failed = out_dir / "decode_failed.txt"
        if meta_path.is_file():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        if failed.is_file():
            raise RuntimeError(failed.read_text(encoding="utf-8").splitlines()[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        with self._decode_lock:
            if meta_path.is_file():
                return json.loads(meta_path.read_text(encoding="utf-8"))
            completed = subprocess.run(
                [sys.executable, "-m", "panoagent_v2.media_worker", *args],
                cwd=str(PACKAGE_ROOT), capture_output=True, text=True,
                timeout=self.timeout_seconds,
            )
        if completed.returncode != 0 or not meta_path.is_file():
            message = f"decoder exit {completed.returncode} for {Path(args[1]).name}"
            failed.write_text(message + "\n" + completed.stderr[-2000:], encoding="utf-8")
            raise RuntimeError(message)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _audio_parts(self, source: Path, label: str) -> list[Attachment]:
        digest = self.hasher.digest(source)
        out_dir = self.cache_root / "audio" / f"{digest[:24]}_b{self.audio_bitrate}_c{int(self.audio_chunk_seconds)}"
        meta = self._run_worker(
            ["audio", str(source), str(out_dir), str(self.audio_chunk_seconds), str(self.audio_bitrate)],
            out_dir,
        )
        parts = meta.get("parts") or []
        result = []
        for index, part in enumerate(parts, 1):
            path = out_dir / part["path"]
            result.append(Attachment(
                label=f"[{label} PART {index}/{len(parts)} | {_clock(part['start'])}-{_clock(part['end'])}]",
                kind="audio", path=path, digest=self.hasher.digest(path),
            ))
        return result

    def prepare(self, sample: NewsSample) -> PreparedItem:
        errors: list[str] = []
        text_parts = []
        if sample.title.strip():
            text_parts.append(f"TITLE: {sample.title.strip()}")
        if sample.text.strip():
            text_parts.append(f"ARTICLE: {sample.text.strip()}")
        text = _budget("\n".join(text_parts), self.max_text_chars) if text_parts else ""
        if text and sample.publish_time.strip():
            text += f"\nPUBLICATION DATE: {sample.publish_time.strip()}"

        attachments: dict[str, list[Attachment]] = {}
        images = [path for path in sample.image_paths if path.is_file()]
        if images:
            attachments["image"] = [
                Attachment(f"[NEWS IMAGE {index}]", "image", path, self.hasher.digest(path))
                for index, path in enumerate(images, 1)
            ]
        audios = [path for path in sample.audio_paths if path.is_file()]
        if audios:
            parts: list[Attachment] = []
            for path in audios:
                try:
                    parts.extend(self._audio_parts(path, "NEWS AUDIO"))
                except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
                    errors.append(f"audio {path.name}: {exc}")
            if parts:
                attachments["audio"] = parts
        videos = [path for path in sample.video_paths if path.is_file()]
        if videos:
            visual: list[Attachment] = []
            for video in videos:
                digest = self.hasher.digest(video)
                out_dir = self.cache_root / "frames" / f"{digest[:24]}_n{self.frame_count}_e{self.max_edge}"
                try:
                    meta = self._run_worker(
                        ["frames", str(video), str(out_dir), str(self.frame_count), str(self.max_edge)],
                        out_dir,
                    )
                    for index, frame in enumerate(meta.get("frames") or [], 1):
                        path = out_dir / frame["path"]
                        visual.append(Attachment(
                            f"[VIDEO FRAME {index} @ {_clock(frame['time'])}]", "image", path,
                            self.hasher.digest(path),
                        ))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"video frames {video.name}: {exc}")
                try:
                    visual.extend(self._audio_parts(video, "VIDEO SOUNDTRACK"))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"video soundtrack {video.name}: {exc}")
            if visual:
                attachments["video"] = visual

        modalities = tuple(
            name for name in ("text", "image", "audio", "video")
            if (name == "text" and text) or name in attachments
        )
        return PreparedItem(sample, text, modalities, attachments, errors)
