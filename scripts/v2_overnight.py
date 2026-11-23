"""Unattended v2 run: finish the 943 tuning set, freeze the scorer, then run the 765 holdout once.

    .venv\\Scripts\\python.exe scripts\\v2_overnight.py

Stages (each is skipped if its output already exists, so the script can be restarted):
  1. wait for any running tuning extraction, then resume it until all 943 items are present
  2. one repair pass: items with failed calls (network drops, invalid JSON) are re-run;
     successful calls come from the stage cache, so only the failed calls are repeated
  3. cross-validated score on the tuning set (report only)
  4. freeze the scorer on the tuning set (primary system fixed in advance: all 17 features)
  5. extract the holdout with the frozen code (extraction refuses if code changed), same repair pass
  6. apply the frozen scorer once and write metrics

Progress goes to runs/v2_overnight_status.json and runs/v2_overnight.log.
The process asks Windows not to idle-sleep while it runs (no setting is changed).
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
DATASET = str(ROOT / "fakenewsDataSet")
TAG = "20261121"
TUNING = ("splits/half_eval_seed42.json", "train", f"runs/v2_tuning943_features_{TAG}.jsonl")
HOLDOUT = ("splits/final_holdout_clean_765_20261031.json", "test", f"runs/v2_holdout765_features_{TAG}.jsonl")
FROZEN = f"runs/v2_frozen_{TAG}/frozen.json"
TUNING_SCORE = f"runs/v2_tuning943_score_{TAG}"
HOLDOUT_EVAL = f"runs/v2_holdout765_eval_{TAG}"
BASELINES = [
    ("single call qwen3.8-omni-flash (v1 inputs)", "runs/qwen38_omni_flash_full_1950_20261107.jsonl"),
    ("single call qwen3.5-27b (v1 inputs)", "runs/qwen35_27b_full_1950_20261107.jsonl"),
]
STATUS = ROOT / "runs" / "v2_overnight_status.json"
LOG = ROOT / "runs" / "v2_overnight.log"


def log(message: str, stage: str | None = None) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)
    if stage:
        STATUS.write_text(json.dumps({"stage": stage, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "message": message}, ensure_ascii=False, indent=2), encoding="utf-8")


def keep_awake() -> None:
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def extraction_running(output: str) -> bool:
    name = Path(output).name
    query = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { $_.CommandLine }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True).stdout
    return any("panoagent_v2.cli extract" in line and name in line for line in out.splitlines())


def ids_of(split: str, partition: str) -> list[str]:
    payload = json.loads((ROOT / split).read_text(encoding="utf-8"))
    return [str(v) for v in payload["partitions"][partition]]


def latest_rows(output: str) -> dict[str, dict]:
    path = ROOT / output
    rows: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def has_failed_call(row: dict) -> bool:
    return bool(row.get("fatal_error") or (row.get("single_call") or {}).get("error")
                or any(c.get("error") for c in row.get("calls", [])))


def run_extract(split: str, partition: str, output: str, frozen: str | None) -> int:
    command = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "run_v2.ps1"),
               "-Split", split, "-Partition", partition, "-Output", output, "-Workers", "4"]
    if frozen:
        command += ["-Frozen", frozen]
    console = ROOT / (output.replace(".jsonl", "") + "_console.log")
    with console.open("a", encoding="utf-8") as handle:
        return subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def complete(split: str, partition: str, output: str, frozen: str | None, label: str) -> None:
    wanted = ids_of(split, partition)
    while extraction_running(output):
        rows = latest_rows(output)
        log(f"{label}: extraction already running, {len(rows)}/{len(wanted)} rows", stage=f"{label}: extracting")
        time.sleep(600)
    for attempt in range(4):
        rows = latest_rows(output)
        missing = [i for i in wanted if i not in rows or rows[i].get("fatal_error")]
        if not missing:
            break
        log(f"{label}: {len(missing)} items missing or fatal, extraction attempt {attempt + 1}",
            stage=f"{label}: extracting")
        code = run_extract(split, partition, output, frozen)
        log(f"{label}: extraction exited with code {code}")
    rows = latest_rows(output)
    failed = [i for i in wanted if i in rows and has_failed_call(rows[i])]
    marker = ROOT / (output + ".repaired")
    if failed and not marker.exists():
        log(f"{label}: repair pass for {len(failed)} items with failed calls: {failed}", stage=f"{label}: repair")
        path = ROOT / output
        shutil.copyfile(path, str(path) + ".before_repair")
        keep = [json.dumps(rows[i], ensure_ascii=False) for i in rows if i not in set(failed)]
        path.write_text("\n".join(keep) + "\n", encoding="utf-8")
        for attempt in range(3):
            code = run_extract(split, partition, output, frozen)
            rows = latest_rows(output)
            if all(i in rows and not rows[i].get("fatal_error") for i in wanted):
                break
            log(f"{label}: repair extraction attempt {attempt + 1} exited {code}, still incomplete")
        marker.write_text(json.dumps({"repaired": failed}, ensure_ascii=False), encoding="utf-8")
    rows = latest_rows(output)
    still = sum(1 for i in wanted if i in rows and has_failed_call(rows[i]))
    missing = sum(1 for i in wanted if i not in rows)
    log(f"{label}: done, {len(rows)} rows, {missing} missing, {still} with failed calls after repair")


def cli(*args: str) -> None:
    command = [PY, "-m", "panoagent_v2.cli", *args]
    log("run: " + " ".join(args[:1] + args[3:5]))
    with LOG.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode:
        raise SystemExit(f"{args[0]} failed with exit code {result.returncode}")


def baseline_args() -> list[str]:
    out: list[str] = []
    for name, path in BASELINES:
        out += ["--baseline", name, path]
    return out


def main() -> None:
    keep_awake()
    log("overnight run started", stage="start")
    complete(*TUNING, frozen=None, label="tuning943")
    if not (ROOT / TUNING_SCORE / "metrics.json").exists():
        cli("score", "--dataset", DATASET, "--features", TUNING[2], "--output-dir", TUNING_SCORE, *baseline_args())
    if not (ROOT / FROZEN).exists():
        log("freezing scorer on the tuning set", stage="freeze")
        cli("freeze", "--dataset", DATASET, "--features", TUNING[2], "--output", FROZEN)
    complete(*HOLDOUT, frozen=FROZEN, label="holdout765")
    if not (ROOT / HOLDOUT_EVAL / "metrics.json").exists():
        log("applying frozen scorer to the holdout", stage="evaluate")
        cli("evaluate", "--dataset", DATASET, "--features", HOLDOUT[2], "--frozen", FROZEN,
            "--output-dir", HOLDOUT_EVAL, *baseline_args())
    log("all stages finished", stage="finished")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        log(f"stopped: {type(exc).__name__}: {exc}", stage="stopped with error")
        raise
