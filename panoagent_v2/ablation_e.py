"""Table E: one-factor input ablations of the v2 single call (artifacts/ABLATION_PLAN_20261124.md).

    inspect  build every variant's request without calling the API and check it against R
    run      N and R first, a drift gate, then every other variant (resumable, cost-capped)
    score    compare each variant with R on the items it changes

Reference R is the frozen v2 single call; its cached probabilities are reused. A variant only calls the API for
items whose request differs from R. Same model, system prompt, temperature, seed and log-probability output
throughout. Only the 943 tuning items are accepted. The frozen modules are imported, never changed.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from joblib import Parallel, delayed

from panoagent import prompts as v1_prompts
from panoagent.dataset import PanoFNDDataset
from panoagent.media import VideoFrameExtractor, _truncate_head_tail, prepare_sample_media

from . import prompts
from .ablation import ROOT, SEEDS, Evaluation, check_tuning_only, single_threshold
from .agents import StageCache, p_fake_from_logprobs
from .cli import _load_rows
from .client import OmniClient, _audio_data_url, _image_data_url, build_content
from .media import MediaPreparer

PRICE_IN, PRICE_OUT = 0.8e-6, 2.7e-6  # CNY per token, checked 2026-11-24
TRANSCRIPT_CHARS = 8000  # v1 budget
TITLE_ONLY = {"audio", "image", "audio+image"}  # no text modality: v1 and v2 both still send the title
# R and N first (drift gate), then the planned primary contrasts, then the rest.
ORDER = ("R", "N", "A2", "P", "L", "S0", "A1", "S", "F", "I", "T", "D")
PRIMARY = ("A2", "P", "L")
NAMES = {"R": "v2 single call, only its failed items", "N": "repeat of R", "N2": "second repeat of R, after N", "S0": "complete v1 request", "P": "v1 user prompt and layout",
         "L": "no attachment labels", "A1": "audio -> transcript cut to 8000 chars", "A2": "audio -> full transcript",
         "S": "no video soundtrack", "F": "16 video frames", "I": "news images at original resolution",
         "D": "no publication date", "T": "no title for items without a text modality"}
MAX_DP, MAX_FLIP = 0.03, 0.05  # drift gate: mean |dP(fake)| and label flips at 0.5 between R and its repeat
# If the gate fires, N2 repeats the same items once more. Fixed before N2 was run: with D = mean |dP(fake)|, D(N,N2)/D(R,N)
# at or above 0.75 means ordinary run-to-run noise, at or below 0.5 means the service changed, anything between is unclear.
NOISE_RATIO, DRIFT_RATIO = 0.75, 0.5

UNLABELLED_USER = prompts.SINGLE_USER.replace(
    " Each attachment is preceded by a label\n"
    "such as [NEWS IMAGE 1], [VIDEO FRAME 3 @ 00:42], [NEWS AUDIO PART 1/1] or [VIDEO SOUNDTRACK PART 1/1].", "")
assert UNLABELLED_USER != prompts.SINGLE_USER

# A request is a list of parts: ("text", str) | ("image", Path, "1280" | "raw") | ("audio", Path)


def _media_parts(item, labels=True, transcripts=None, drop_soundtrack=False, image_mode="1280"):
    parts = []
    for name in item.modalities:
        if name == "text":
            continue
        if labels:
            parts.append(("text", f"\n{prompts.MODALITY_NAMES[name].upper()}:"))
        if name == "audio" and transcripts is not None:
            parts.append(("text", f"[NEWS AUDIO TRANSCRIPT]\n{transcripts}"))
            continue
        for a in item.attachments[name]:
            if drop_soundtrack and name == "video" and a.kind == "audio":
                continue
            if labels:
                parts.append(("text", a.label))
            parts.append(("image", a.path, image_mode if name == "image" else "1280") if a.kind == "image" else ("audio", a.path))
    return parts


def _v2_parts(item, user, text, **media):
    parts = [("text", user)]
    if text:
        parts.append(("text", f"\n[NEWS TEXT]\n{text}"))
    return parts + _media_parts(item, **media)


def _v1_text(sample, with_date: bool) -> str:
    out = []
    if sample.title.strip():
        out.append(f"TITLE:\n{sample.title.strip()}")
    if sample.text.strip():
        out.append(f"ARTICLE TEXT:\n{sample.text.strip()}")
    if with_date and sample.publish_time.strip():
        out.append(f"PUBLICATION DATE (metadata): {sample.publish_time.strip()}")
    return "\n\n".join(out)


class Builder:
    def __init__(self, media_cache="cache/v2_media", v1_frames="cache/video_frames"):
        self.prep = MediaPreparer(media_cache)
        self.prep16 = MediaPreparer(media_cache, frame_count=16)
        self.prep16.hasher = self.prep.hasher
        self.extractor = VideoFrameExtractor(ROOT / v1_frames, frame_count=16)

    def build(self, variant: str, sample) -> list[tuple]:
        if variant == "S0":
            return self._v1(sample)
        item = (self.prep16 if variant == "F" else self.prep).prepare(sample)
        user = prompts.SINGLE_USER.format(modalities="+".join(item.modalities))
        text = item.text
        if variant in ("R", "N", "N2", "F"):
            return _v2_parts(item, user, text)
        if variant == "P":
            user = v1_prompts.DIRECT_USER.format(modalities="+".join(item.modalities), content=_v1_text(sample, bool(text)))
            return [("text", user)] + _media_parts(item)
        if variant == "L":
            return _v2_parts(item, UNLABELLED_USER.format(modalities="+".join(item.modalities)), text, labels=False)
        if variant in ("A1", "A2"):
            if not sample.transcript.strip():
                return _v2_parts(item, user, text)
            t = sample.transcript.strip()
            return _v2_parts(item, user, text, transcripts=_truncate_head_tail(t, TRANSCRIPT_CHARS) if variant == "A1" else t)
        if variant == "S":
            return _v2_parts(item, user, text, drop_soundtrack=True)
        if variant == "I":
            big = any(self._edge(a.path) > 1280 for a in item.attachments.get("image", []))
            return _v2_parts(item, user, text, image_mode="raw" if big else "1280")
        if variant == "D":
            date = f"\nPUBLICATION DATE: {sample.publish_time.strip()}"
            return _v2_parts(item, user, text[: -len(date)] if text.endswith(date) else text)
        if variant == "T":
            if sample.modality_key not in TITLE_ONLY:
                return _v2_parts(item, user, text)
            # without the title there is no text block, so the date and the "text" modality go with it
            mods = tuple(m for m in item.modalities if m != "text")
            user = prompts.SINGLE_USER.format(modalities="+".join(mods))
            return _v2_parts(replace(item, modalities=mods), user, "")
        raise ValueError(variant)

    @staticmethod
    def _edge(path) -> int:
        from PIL import Image

        with Image.open(path) as im:
            return max(im.size)

    def _v1(self, sample):
        view = prepare_sample_media(sample, "frames", self.extractor, 12000, TRANSCRIPT_CHARS)
        user = v1_prompts.DIRECT_USER.format(modalities=sample.modality_key, content=view.detector_text())
        return [("text", user)] + [("image", p, "v1") for p in view.image_paths if p.is_file()]

    def fingerprint(self, variant: str, model: str, system: str, parts) -> str:
        seq = [[p[0], p[1]] if p[0] == "text" else [p[0], self.prep.hasher.digest(p[1]), *p[2:]] for p in parts]
        # N is a deliberate repeat, so its identity must differ from R's
        blob = json.dumps([variant if variant in ("N", "N2") else "", model, system, seq], ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _raw_image_url(path: Path) -> str:
    """Image bytes as v1 sent them: untouched when the format is supported, otherwise JPEG."""
    from PIL import Image

    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "BMP": "image/bmp"}.get((im.format or "").upper())
        if mime is None:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=90)
            raw, mime = buf.getvalue(), "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def to_content(parts) -> list[dict]:
    out = []
    for p in parts:
        if p[0] == "text":
            out.append({"type": "text", "text": p[1]})
        elif p[0] == "image":
            out.append({"type": "image_url", "image_url": {"url": _image_data_url(p[1]) if p[2] == "1280" else _raw_image_url(p[1])}})
        else:
            out.append({"type": "input_audio", "input_audio": {"data": _audio_data_url(p[1]), "format": "mp3"}})
    return out


def _same(a, b) -> bool:
    return [tuple(map(str, p)) for p in a] == [tuple(map(str, p)) for p in b]


def _cost(u: dict) -> float:
    return (u or {}).get("prompt_tokens", 0) * PRICE_IN + (u or {}).get("completion_tokens", 0) * PRICE_OUT


class Runner:
    def __init__(self, args):
        self.args = args
        self.reference = _load_rows(args.features)
        check_tuning_only([r["id"] for r in self.reference])
        self.samples = {s.id: s for s in PanoFNDDataset(args.dataset).iter_ids([r["id"] for r in self.reference])}
        self.pilot = set(map(str, json.loads((ROOT / "splits/v2_tuning180_seed42_20261121.json").read_text(encoding="utf-8"))["partitions"]["pilot"]))
        self.ref = {r["id"]: r["single_call"] for r in self.reference}
        self.builder = Builder()
        self.out = Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.client = OmniClient(args.model_id)
        self.lock = threading.Lock()
        self.spent = sum(_cost(r.get("usage")) for f in self.out.glob("*.jsonl") for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()))

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        with self.lock, (self.out / "run.log").open("a", encoding="utf-8") as h:
            h.write(line + "\n")
        print(line, flush=True)

    def status(self, stage: str, **extra) -> None:
        (self.out / "status.json").write_text(json.dumps(
            {"stage": stage, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "spent_cny": round(self.spent, 2), **extra},
            ensure_ascii=False, indent=2), encoding="utf-8")

    def wanted(self, variant: str) -> list[str]:
        """Items whose request differs from R (R: only the failed ones, N: the 180-item subset)."""
        ids = [r["id"] for r in self.reference]
        if variant == "R":
            return [i for i in ids if self.ref[i].get("p_fake") is None]
        if variant in ("N", "N2"):
            return [i for i in ids if i in self.pilot and self.ref[i].get("p_fake") is not None]
        out = []
        for i in ids:
            key = self.reference_key(i)
            if variant in ("A1", "A2") and "audio" not in key or variant in ("S", "F") and "video" not in key \
                    or variant == "I" and "image" not in key or variant == "T" and key not in TITLE_ONLY or variant == "L" and key == "text":
                continue
            if variant in ("S0", "P"):
                out.append(i)
            elif not _same(self.builder.build(variant, self.samples[i]), self.builder.build("R", self.samples[i])):
                out.append(i)
        return out

    def reference_key(self, i: str) -> str:
        return self.samples[i].modality_key

    def path(self, variant: str) -> Path:
        return self.out / f"{variant}.jsonl"

    def rows(self, variant: str) -> dict[str, dict]:
        f = self.path(variant)
        return {r["id"]: r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())} if f.is_file() else {}

    def call(self, variant: str, i: str) -> dict:
        started = time.perf_counter()
        sample = self.samples[i]
        row = {"id": i, "variant": variant, "modality_key": sample.modality_key}
        try:
            parts = self.builder.build(variant, sample)
            system = prompts.SINGLE_SYSTEM
            if self.spent > self.args.max_cost:
                return {**row, "error": "skipped: cost limit", "skipped": True}
            cache = StageCache(Path(self.args.cache) / self.args.model_id / variant)
            fp = self.builder.fingerprint(variant, self.args.model_id, system, parts)
            rec = cache.load(i, "single_call", fp)
            if rec is None or rec.get("error"):
                rec = {"fingerprint": fp}
                try:
                    res = self.client.chat(system, to_content(parts), 4, logprobs=True)
                    rec.update(raw=res.text, usage=res.usage, model=res.model, request_id=res.request_id,
                               top_logprobs=res.first_token_logprobs, error=None)
                    rec["p_fake"] = p_fake_from_logprobs(res.first_token_logprobs or [], res.text)
                    with self.lock:
                        self.spent += _cost(res.usage)
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:600]
                cache.save(i, "single_call", rec)
                row["cached"] = False
            else:
                row["cached"] = True
            usage = rec.get("usage") if not row["cached"] else {}  # a resumed run must not count tokens twice
            row.update(p_fake=rec.get("p_fake"), raw=rec.get("raw"), usage=usage, error=rec.get("error"),
                       n_parts=len(parts))
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the run
            row["error"] = f"{type(exc).__name__}: {exc}"[:600]
        row["latency_seconds"] = round(time.perf_counter() - started, 2)
        with self.lock, self.path(variant).open("a", encoding="utf-8") as h:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def run_variant(self, variant: str, limit: int | None = None) -> None:
        ids = self.wanted(variant)
        if limit:
            ids = ids[:: max(1, len(ids) // limit)][:limit]
        for attempt in (1, 2):
            have = self.rows(variant)
            todo = [i for i in ids if i not in have or have[i].get("error")
                    and not ("data_inspection_failed" in str(have[i]["error"]) and attempt == 2)]
            if attempt == 1:
                self.log(f"{variant} ({NAMES[variant]}): {len(ids)} items, {len(todo)} to run, spent {self.spent:.2f} CNY")
            if not todo:
                break
            self.status(f"{variant}: running", attempt=attempt, remaining=len(todo))
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                for n, fut in enumerate(as_completed([pool.submit(self.call, variant, i) for i in todo]), 1):
                    r = fut.result()
                    if n % 20 == 0 or n == len(todo) or r.get("error"):
                        print(f"{variant} {n}/{len(todo)} {r['id']} {r['modality_key']:17} p={r.get('p_fake')} spent={self.spent:.2f} "
                              f"err={str(r.get('error'))[:80]} {r.get('latency_seconds')}s", flush=True)
                        self.status(f"{variant}: running", attempt=attempt, done=n, of=len(todo))
        have = self.rows(variant)
        self.log(f"{variant}: finished, {sum(1 for i in ids if i in have and not have[i].get('error'))}/{len(ids)} ok, spent {self.spent:.2f} CNY")

    def repeat_decision(self) -> dict:
        n, n2 = self.rows("N"), self.rows("N2")
        ids = [i for i in n if n[i].get("p_fake") is not None and (n2.get(i) or {}).get("p_fake") is not None and self.ref[i].get("p_fake") is not None]
        r, a, b = (np.array([f(i) for i in ids]) for f in (lambda i: self.ref[i]["p_fake"], lambda i: n[i]["p_fake"], lambda i: n2[i]["p_fake"]))
        d = {k: float(np.abs(x - y).mean()) for k, (x, y) in {"R-N": (r, a), "N-N2": (a, b), "R-N2": (r, b)}.items()}
        ratio = d["N-N2"] / d["R-N"]
        out = {"n": len(ids), **{f"mean_abs_dp {k}": round(v, 4) for k, v in d.items()}, "ratio": round(ratio, 3),
               "signed N-R": round(float((a - r).mean()), 4), "signed N2-N": round(float((b - a).mean()), 4),
               "flips N-N2": round(float(((a >= .5) != (b >= .5)).mean()), 4),
               "decision": "noise" if ratio >= NOISE_RATIO else "drift" if ratio <= DRIFT_RATIO else "unclear"}
        return out

    def drift(self) -> dict:
        rep = self.rows("N")
        pairs = [(self.ref[i]["p_fake"], r["p_fake"]) for i, r in rep.items() if not r.get("error") and r.get("p_fake") is not None]
        a, b = np.array(pairs).T
        return {"n": len(pairs), "mean_abs_dp": round(float(np.abs(a - b).mean()), 4), "max_abs_dp": round(float(np.abs(a - b).max()), 4),
                "flips_at_0.5": round(float(((a >= .5) != (b >= .5)).mean()), 4), "identical": round(float((a == b).mean()), 4)}


def command_inspect(args) -> None:
    r = Runner(args)
    bad = 0
    for i, s in list(r.samples.items())[:: max(1, len(r.samples) // args.sample)]:  # R must equal the frozen request exactly
        item = r.builder.prep.prepare(s)
        from .agents import _sections

        tb, se = _sections(item, item.modalities)
        want = build_content(prompts.SINGLE_USER.format(modalities="+".join(item.modalities)), se, tb)
        if to_content(r.builder.build("R", s)) != want:
            bad += 1
            print("R MISMATCH", i)
    print(f"R equals the frozen v2 request on sampled items: {'yes' if not bad else f'{bad} mismatches'}")
    for v in ORDER:
        ids = r.wanted(v)
        keys = Counter(r.reference_key(i) for i in ids)
        print(f"== {v} {NAMES[v]}: {len(ids)} items {dict(sorted(keys.items()))}")
        for i in ids[:: max(1, len(ids) // 2)][:2]:
            parts = r.builder.build(v, r.samples[i])
            s = r.samples[i]
            blob = " ".join(p[1] for p in parts if p[0] == "text")
            leak = [n for n, x in (("description", s.description[:60]), ("source", s.source)) if x and x in blob]
            print("  ", i, r.reference_key(i), [(p[1][:60].replace("\n", " ") if p[0] == "text" else f"<{p[0]}{':' + p[2] if len(p) > 2 else ''}>") for p in parts][:9],
                  "parts", len(parts), "LEAK" if leak else "no leak", leak)


def self_decision(r) -> str | None:
    f = r.out / "repeat_decision.json"
    return json.loads(f.read_text(encoding="utf-8"))["decision"] if f.is_file() else None


def command_run(args) -> None:
    if sys.platform == "win32":  # no idle sleep while this process runs; no setting is changed
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    r = Runner(args)
    lock_file = r.out / "run.lock"
    if lock_file.exists():
        import psutil

        pid = int(lock_file.read_text() or 0)
        if pid and psutil.pid_exists(pid) and "ablation_e" in " ".join(psutil.Process(pid).cmdline()):
            raise SystemExit(f"table E is already running as pid {pid}")
    lock_file.write_text(str(os.getpid()))
    r.log(f"table E started; already spent {r.spent:.2f} CNY; cap {args.max_cost} CNY")
    try:
        if args.only:
            r.run_variant(args.only)
            if args.only == "N2":
                d = r.repeat_decision()
                (r.out / "repeat_decision.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
                r.log(f"repeat decision: {d}")
                r.status("N2 finished", decision=d)
            return
        if args.smoke:
            for v in ORDER:
                r.run_variant(v, limit=2)
            r.status("smoke finished")
            return
        for v in ORDER:
            r.run_variant(v)
            if v == "N":
                d = r.drift()
                r.log(f"drift check R vs its repeat: {d}")
                (r.out / "drift.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
                repeat = self_decision(r)
                if (d["mean_abs_dp"] > MAX_DP or d["flips_at_0.5"] > MAX_FLIP) and repeat != "noise":
                    r.status("stopped: drift gate", drift=d)
                    r.log("drift gate: the repeat differs from R too much, so R must be re-run before the other variants")
                    return
            if r.spent > args.max_cost:
                r.status("stopped: cost limit")
                r.log("cost limit reached")
                return
        r.status("finished")
        r.log("all variants finished")
    finally:
        lock_file.unlink(missing_ok=True)


def _threshold_job(name, rows, groups, seed):
    return name, seed, single_threshold(False)(rows, groups, seed)


def _logit(p):
    q = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def rerun_band(y, base, keys, pool, draws=2000) -> float:
    """95th percentile of |dAUC| between two runs that differ only by rerun noise.

    ``pool`` holds logit differences between repeated identical requests, by modality. Both runs get half of the variance,
    with random signs, so the difference is centred on zero. Item-wise independence is assumed: a shift common to a whole
    day is not covered.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    everything = np.concatenate(list(pool.values()))
    logit = _logit(base)
    keys = np.asarray(keys)

    def noisy():
        e = np.empty(len(logit))
        for k in set(keys.tolist()):
            at = np.where(keys == k)[0]
            src = pool.get(k, everything)
            e[at] = rng.choice(src, len(at)) * rng.choice([-1.0, 1.0], len(at))
        return roc_auc_score(y, logit + e / np.sqrt(2))

    return float(np.percentile(np.abs([noisy() - noisy() for _ in range(draws)]), 95))


def command_score(args) -> None:
    r = Runner(args)
    samples = {s.id: s for s in PanoFNDDataset(args.dataset)}
    rows = r.reference
    ids = [x["id"] for x in rows]
    groups = [PanoFNDDataset.group_key(samples[i]) for i in ids]
    gold = [x["gold"] for x in rows]
    rr = r.rows("R")
    p_ref = {i: (r.ref[i].get("p_fake") if r.ref[i].get("p_fake") is not None else (rr.get(i) or {}).get("p_fake")) for i in ids}
    ok_ref = {i: p_ref[i] is not None for i in ids}

    def vector(v):
        got, want = r.rows(v), set(r.wanted(v))
        p = dict(p_ref)
        ok = dict(ok_ref)
        for i in want:
            x = got.get(i)
            ok[i] = bool(x and not x.get("error") and x.get("p_fake") is not None)
            p[i] = x["p_fake"] if ok[i] else None
        return p, ok, want

    variants = [v for v in ORDER if v not in ("R", "N") and r.path(v).is_file()]
    vec = {"R": (p_ref, ok_ref, set(ids)), **{v: vector(v) for v in variants}}
    run_rows = {n: [{"features": {"single_p_fake": (p[i] if p[i] is not None else 0.5)}, "modality_key": x["modality_key"], "gold": x["gold"]}
                    for i, x in zip(ids, rows)] for n, (p, _, _) in vec.items()}
    done = Parallel(n_jobs=-2)(delayed(_threshold_job)(n, rw, groups, s) for n, rw in run_rows.items() for s in SEEDS)
    runs = {n: [x for _, x in sorted([(s, x) for m, s, x in done if m == n], key=lambda t: t[0])] for n in vec}
    g = np.array(gold)
    rep = {"drift": r.drift() if r.path("N").is_file() else None, "n_reference_ok": sum(ok_ref.values()),
           "seeds": list(SEEDS), "variants": {}}
    # rerun noise: R, N and N2 are the same request on the same 180 items
    nn, n2 = r.rows("N"), r.rows("N2")
    trip = [i for i in ids if all((t.get(i) or {}).get("p_fake") is not None for t in (nn, n2)) and r.ref[i].get("p_fake") is not None]
    same = {"R": np.array([r.ref[i]["p_fake"] for i in trip]), "N": np.array([nn[i]["p_fake"] for i in trip]), "N2": np.array([n2[i]["p_fake"] for i in trip])}
    tk = np.array([samples[i].modality_key for i in trip])
    pairs = (("R", "N"), ("R", "N2"), ("N", "N2"))
    pool = {k: np.concatenate([(_logit(same[a]) - _logit(same[b]))[tk == k] for a, b in pairs]) for k in sorted(set(tk.tolist()))}
    gi = np.array([g[ids.index(i)] for i in trip])
    evn = Evaluation(gi.tolist(), [groups[ids.index(i)] for i in trip], auc_boot=2000)
    hard = lambda p: [(p, np.where(p >= 0.5, "FAKE", "REAL"))]
    rep["rerun_noise"] = {"n": len(trip), "pairs": {f"{a}-{b}": {
        "mean_abs_dp": float(np.abs(same[a] - same[b]).mean()), "signed": float((same[a] - same[b]).mean()),
        "flips_at_0.5": float(((same[a] >= .5) != (same[b] >= .5)).mean()), **evn.paired(hard(same[a]), hard(same[b]))} for a, b in pairs},
        "auroc": {k: evn.summarise(hard(v))["auroc"]["mean"] for k, v in same.items()},
        "mean_abs_dp_by_modality": {k: {"n": int((tk == k).sum()), "mean_abs_dp": float(np.mean([np.abs(same[a] - same[b])[tk == k].mean() for a, b in pairs]))}
                                    for k in pool}}
    for v in variants:
        p, ok, want = vec[v]
        mask_all = np.array([i in want for i in ids])
        mask = np.array([i in want and ok[i] and ok_ref[i] for i in ids])
        entry = {"name": NAMES[v], "affected": int(mask_all.sum()), "both_ok": int(mask.sum()),
                 "affected_by_modality": dict(Counter(samples[i].modality_key for i in want))}
        whole = np.array([ok[i] and ok_ref[i] for i in ids])
        for label, m in (("primary", mask), ("intent_to_treat", mask_all), ("all_tuning_items", whole)):
            idx = np.where(m)[0]
            if not len(idx):
                continue
            ev = Evaluation(g[idx].tolist(), [groups[k] for k in idx], auc_boot=2000)
            sub = lambda n: [(np.array([vec[n][0][ids[k]] if vec[n][0][ids[k]] is not None else 0.5 for k in idx]), np.asarray(pr)[idx]) for _, pr in runs[n]]
            raw = lambda n: [(np.array([vec[n][0][ids[k]] if vec[n][0][ids[k]] is not None else 0.5 for k in idx]),
                              np.where(np.array([vec[n][0][ids[k]] if vec[n][0][ids[k]] is not None else 0.5 for k in idx]) >= 0.5, "FAKE", "REAL"))]
            entry[label] = {"n": len(idx), "fake": int((g[idx] == "FAKE").sum()), "R": ev.summarise(sub("R")), v: ev.summarise(sub(v)),
                            "R_minus_variant": ev.paired(sub("R"), sub(v)), "R_minus_variant_raw_0.5": ev.paired(raw("R"), raw(v)),
                            "raw_0.5": {"R": ev.summarise(raw("R"))["balanced_accuracy"]["mean"], v: ev.summarise(raw(v))["balanced_accuracy"]["mean"]}}
        a = np.array([(p_ref[i], p[i]) for i in ids if i in want and ok[i] and ok_ref[i]], dtype=float)
        entry["mean_abs_dp"], entry["flips_at_0.5"] = float(np.abs(a[:, 0] - a[:, 1]).mean()), float(((a[:, 0] >= .5) != (a[:, 1] >= .5)).mean())
        y = (g == "FAKE").astype(int)[mask]
        base = [p_ref[i] for i, m in zip(ids, mask) if m]
        entry["rerun_band_auc95"] = rerun_band(y, base, [samples[i].modality_key for i, m in zip(ids, mask) if m], pool)
        seen = [i for i in trip if i in want and ok[i]]
        entry["against_repeat"] = {"n": len(seen), "mean_abs_dp_variant": float(np.mean([abs(p_ref[i] - p[i]) for i in seen])),
                                   "mean_abs_dp_repeat": float(np.mean([abs(nn[i]["p_fake"] - r.ref[i]["p_fake"]) for i in seen]))} if seen else None
        entry["by_modality"] = {}
        for k in sorted({samples[i].modality_key for i in want}):
            idx = np.array([j for j, i in enumerate(ids) if mask[j] and samples[i].modality_key == k])
            if len(idx) >= 40 and len(set(g[idx])) == 2:
                ev = Evaluation(g[idx].tolist(), [groups[j] for j in idx], auc_boot=1000)
                pa, pb = (np.array([q[ids[j]] for j in idx]) for q in (p_ref, p))
                entry["by_modality"][k] = {"n": len(idx), **{x: w for x, w in ev.paired(hard(pa), hard(pb)).items() if x.startswith("auc")}}
        d = entry["primary"]["R_minus_variant"]
        entry["resolved"] = bool((d["auc_ci95"][0] > 0 or d["auc_ci95"][1] < 0) and abs(d["auc_diff"]) > entry["rerun_band_auc95"])
        rep["variants"][v] = entry
    if "A1" in vec and "A2" in vec:  # the effect of truncation alone, only where the two requests differ
        idx = np.array([j for j, i in enumerate(ids) if i in vec["A1"][2] and vec["A1"][1][i] and vec["A2"][1][i]
                        and len(samples[i].transcript.strip()) > TRANSCRIPT_CHARS])
        ev = Evaluation(g[idx].tolist(), [groups[j] for j in idx], auc_boot=2000)
        pa, pb = (np.array([vec[n][0][ids[j]] for j in idx]) for n in ("A2", "A1"))
        rep["A2_minus_A1_truncated_items"] = {"n": len(idx), "mean_abs_dp": float(np.abs(pa - pb).mean()), **ev.paired(hard(pa), hard(pb))}
    if "S0" in variants:  # the S0 outputs against the v1 run of 2026-11-07 (hard labels)
        v1 = {x["id"]: x["prediction"] for x in map(json.loads, (ROOT / "runs/qwen38_omni_flash_full_1950_20261107.jsonl").read_text(encoding="utf-8").splitlines())}
        p, ok, _ = vec["S0"]
        both = [i for i in ids if ok[i] and v1.get(i) in ("REAL", "FAKE")]
        rep["S0_vs_v1_20261107"] = {"n": len(both), "same_label_at_0.5": float(np.mean([(p[i] >= 0.5) == (v1[i] == "FAKE") for i in both]))}
    d1 = json.loads((ROOT / "artifacts/v2_results_20261122/ablation_d1_20261124.json").read_text(encoding="utf-8"))
    pvals = {"D1 structure": d1["subsets"]["three-modality, both succeeded (primary)"]["pairwise_minus_joint"]["A0 plain"]["auc_p_boot"]}
    pvals.update({v: rep["variants"][v]["primary"]["R_minus_variant"]["auc_p_boot"] for v in PRIMARY if v in rep["variants"]})
    ranked = sorted(pvals.items(), key=lambda t: t[1])
    holm, prev = {}, 0.0
    for k, (name, p) in enumerate(ranked):
        prev = max(prev, min(1.0, (len(ranked) - k) * p))
        holm[name] = prev
    rep["holm_adjusted_primary"] = holm
    out = Path(args.output_dir)
    (out / "ablation_e.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "variants"}, ensure_ascii=False, indent=1))
    for v, e in rep["variants"].items():
        q = e["primary"]
        d = q["R_minus_variant"]
        print(f"{v:3} {e['name']:45} n={q['n']}/{e['affected']} dAUC(R-v) {d['auc_diff']:+.3f} [{d['auc_ci95'][0]:+.3f},{d['auc_ci95'][1]:+.3f}] p={d['auc_p_boot']:.3f} "
              f"dBA {d['ba_diff']:+.1%} [{d['ba_ci95'][0]:+.1%},{d['ba_ci95'][1]:+.1%}] |dp| {e['mean_abs_dp']:.3f} flips {e['flips_at_0.5']:.1%} "
              f"rerun band {e['rerun_band_auc95']:.3f} resolved {e['resolved']}")


def command_sensitivity(args) -> None:
    """Post hoc, added after seeing the scores: the 180-item subset with R replaced by its repeats N and N2, and S0 against v1."""
    r = Runner(args)
    samples = r.samples
    gold = {x["id"]: x["gold"] for x in r.reference}
    hard = lambda p: [(p, np.where(p >= 0.5, "FAKE", "REAL"))]
    runs = {"R": {i: x.get("p_fake") for i, x in r.ref.items()}, **{n: {i: x.get("p_fake") for i, x in r.rows(n).items()} for n in ("N", "N2")}}
    out = {"reference_on_180": {}, "variants": {}}
    for v in ORDER[2:]:
        got = {i: x.get("p_fake") for i, x in r.rows(v).items()}
        ids = [i for i in runs["N"] if got.get(i) is not None and all(runs[n].get(i) is not None for n in runs)]
        ev = Evaluation([gold[i] for i in ids], [PanoFNDDataset.group_key(samples[i]) for i in ids], auc_boot=2000)
        arr = lambda d: np.array([d[i] for i in ids])
        out["variants"][v] = {"n": len(ids), **{n: {k: w for k, w in ev.paired(hard(arr(d)), hard(arr(got))).items() if k.startswith("auc")} for n, d in runs.items()}}
    v1 = {x["id"]: x["prediction"] for x in map(json.loads, (ROOT / "runs/qwen38_omni_flash_full_1950_20261107.jsonl").read_text(encoding="utf-8").splitlines())}
    s0 = {i: x.get("p_fake") for i, x in r.rows("S0").items()}
    ids = [i for i in gold if s0.get(i) is not None and v1.get(i) in ("REAL", "FAKE")]
    g, a, b = (np.array(x) for x in ([gold[i] for i in ids], [v1[i] for i in ids], ["FAKE" if s0[i] >= 0.5 else "REAL" for i in ids]))
    ba = lambda p: float(np.mean([(p[g == c] == c).mean() for c in ("REAL", "FAKE")]))
    out["S0_vs_v1_20261107"] = {"n": len(ids), "ba_v1": ba(a), "ba_S0_at_0.5": ba(b), "same_label": float((a == b).mean())}
    (Path(args.output_dir) / "ablation_e_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="panoagent_v2.ablation_e")
    ap.add_argument("command", choices=("inspect", "run", "score", "sensitivity"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-id", default="qwen3.8-omni-flash")
    ap.add_argument("--cache", default="cache/v2_ablation_e")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", choices=(*ORDER, "N2"))
    args = ap.parse_args(argv)
    {"inspect": command_inspect, "run": command_run, "score": command_score, "sensitivity": command_sensitivity}[args.command](args)


if __name__ == "__main__":
    main()
