"""Child-process media decoding.

Some PanoFND files crash FFmpeg natively (e.g. video_1669_1.mp4). A native crash
cannot be caught in Python, so every decode runs here, in a separate process; the
parent turns a non-zero exit into an ordinary, recoverable error.

Usage:
  python -m panoagent_v2.media_worker frames <video> <out_dir> <count> <max_edge>
  python -m panoagent_v2.media_worker audio  <media> <out_dir> <chunk_seconds> <bitrate>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def extract_frames(video: Path, out_dir: Path, count: int, max_edge: int) -> dict:
    import av

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        if not duration or duration <= 0:
            duration = None
        if duration:
            # Uniform timestamps strictly inside the clip; seek to each one.
            targets = [duration * (index + 0.5) / count for index in range(count)]
            for index, target in enumerate(targets):
                # Backward seek lands on the keyframe at or before the target; that
                # first decoded frame is used (exact-frame decoding is far slower).
                container.seek(int(target / stream.time_base), stream=stream, backward=True)
                for frame in container.decode(stream):
                    image = frame.to_image()
                    image.thumbnail((max_edge, max_edge))
                    path = out_dir / f"frame_{index:02d}.jpg"
                    image.convert("RGB").save(path, format="JPEG", quality=90)
                    frames.append({"path": path.name, "time": round(float(frame.time or target), 2)})
                    break
        else:
            decoded = list(container.decode(stream))
            step = max(1, len(decoded) // count)
            for index, frame in enumerate(decoded[::step][:count]):
                image = frame.to_image()
                image.thumbnail((max_edge, max_edge))
                path = out_dir / f"frame_{index:02d}.jpg"
                image.convert("RGB").save(path, format="JPEG", quality=90)
                frames.append({"path": path.name, "time": round(float(frame.time or 0.0), 2)})
    meta = {"duration": duration, "frames": frames}
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def compress_audio(media: Path, out_dir: Path, chunk_seconds: float, bitrate: int) -> dict:
    """Mono 16 kHz MP3 parts, each small enough for the 10 MB Base64 request limit."""
    import av

    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[dict] = []
    with av.open(str(media)) as src:
        if not src.streams.audio:
            meta = {"duration": 0.0, "parts": []}
            (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            return meta
        stream = src.streams.audio[0]
        resampler = av.AudioResampler(format="s16p", layout="mono", rate=16000)
        dst = None
        ostream = None
        part_index = -1
        part_start = 0.0
        last_time = 0.0

        def open_part(index: int):
            container = av.open(str(out_dir / f"part_{index:02d}.mp3"), "w", format="mp3")
            out = container.add_stream("mp3", rate=16000)
            out.layout = "mono"
            out.bit_rate = bitrate
            return container, out

        def close_part():
            if dst is None:
                return
            for packet in ostream.encode(None):
                dst.mux(packet)
            dst.close()
            parts.append({
                "path": f"part_{part_index:02d}.mp3",
                "start": round(part_start, 2),
                "end": round(last_time, 2),
            })

        for frame in src.decode(stream):
            time = float(frame.time or last_time)
            if dst is None or time - part_start >= chunk_seconds:
                close_part()
                part_index += 1
                part_start = time if part_index else 0.0
                dst, ostream = open_part(part_index)
            last_time = time + (frame.samples / float(frame.sample_rate or 16000))
            for resampled in resampler.resample(frame):
                for packet in ostream.encode(resampled):
                    dst.mux(packet)
        close_part()
    meta = {"duration": round(last_time, 2), "parts": parts}
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main(argv: list[str]) -> int:
    task = argv[0]
    if task == "frames":
        extract_frames(Path(argv[1]), Path(argv[2]), int(argv[3]), int(argv[4]))
    elif task == "audio":
        compress_audio(Path(argv[1]), Path(argv[2]), float(argv[3]), int(argv[4]))
    else:
        raise SystemExit(f"unknown task {task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
