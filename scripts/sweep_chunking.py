"""Sweep chunking parameters through index -> ingest -> eval.

Usage:
    python3 scripts/sweep_chunking.py --phase smoke
    python3 scripts/sweep_chunking.py --phase axis
    python3 scripts/sweep_chunking.py --phase custom --combo '{"MAX_CHUNK_CHARS": 1200}'

Results are appended to results/chunking_sweep/sweep_chunking.csv.
Credentials are intentionally read from environment only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "chunking_sweep" / "sweep_chunking.csv"

PARAM_KEYS = (
    "MAX_CHUNK_CHARS",
    "OVERLAP_MESSAGE_COUNT",
    "OVERLAP_CONTEXT_CHARS",
    "MAX_TIME_GAP_SECONDS",
    "SPLIT_MESSAGE_CHAR_THRESHOLD",
    "SPLIT_SEGMENT_TARGET_CHARS",
)

DEFAULTS = {
    "MAX_CHUNK_CHARS": 1800,
    "OVERLAP_MESSAGE_COUNT": 2,
    "OVERLAP_CONTEXT_CHARS": 500,
    "MAX_TIME_GAP_SECONDS": 10800,
    "SPLIT_MESSAGE_CHAR_THRESHOLD": 1200,
    "SPLIT_SEGMENT_TARGET_CHARS": 700,
}


def shell(
    cmd: list[str],
    env: dict[str, str] | None = None,
    *,
    check: bool = True,
    capture: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    full_env = {**os.environ}
    if env:
        full_env.update(env)
    kwargs: dict[str, Any] = {"env": full_env, "cwd": str(ROOT)}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["text"] = True
    if timeout is not None:
        kwargs["timeout"] = timeout
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        out = result.stdout if capture else ""
        raise RuntimeError(f"cmd {cmd} failed rc={result.returncode}\n{out}")
    return result


def wait_for_health(url: str, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"service not healthy after {timeout}s: {url}")


def restart_index(params: dict[str, int]) -> None:
    env = {key: str(value) for key, value in params.items()}
    shell(
        ["docker", "compose", "up", "-d", "--force-recreate", "--no-deps", "index"],
        env=env,
        capture=True,
        timeout=120,
    )
    wait_for_health("http://localhost:8001/health")


def run_ingest(data_path: str) -> str:
    env = {
        "DATA_PATH": data_path,
        "RESET_COLLECTION": "1",
        "INDEX_URL": "http://localhost:8001",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION_NAME": "evaluation",
        "BATCH_SIZE": "16",
    }
    result = shell(["python3", "eval/ingest.py"], env=env, capture=True, timeout=300)
    return result.stdout or ""


METRIC_RE = {
    "n": re.compile(r"^N\s*=\s*(\d+)", re.MULTILINE),
    "row": re.compile(r"^final\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", re.MULTILINE),
    "legacy_recall": re.compile(r"^Recall@\d+\s*=\s*([0-9.]+)", re.MULTILINE),
    "legacy_ndcg": re.compile(r"^nDCG@\d+\s*=\s*([0-9.]+)", re.MULTILINE),
    "legacy_score": re.compile(r"^score\s*=\s*([0-9.]+)", re.MULTILINE),
}


def run_eval(dataset_path: str, k: int = 50) -> dict[str, Any]:
    env = {"SEARCH_URL": "http://localhost:8002"}
    result = shell(
        ["python3", "eval/run.py", "--dataset", dataset_path, "--k", str(k)],
        env=env,
        capture=True,
        timeout=600,
    )
    out = result.stdout or ""
    metrics: dict[str, Any] = {"_tail": out[-400:]}
    if match := METRIC_RE["n"].search(out):
        metrics["n"] = float(match.group(1))
    if match := METRIC_RE["row"].search(out):
        metrics["recall"] = float(match.group(1))
        metrics["ndcg"] = float(match.group(2))
        metrics["score"] = float(match.group(3))
        return metrics
    if (
        (recall := METRIC_RE["legacy_recall"].search(out))
        and (ndcg := METRIC_RE["legacy_ndcg"].search(out))
        and (score := METRIC_RE["legacy_score"].search(out))
    ):
        metrics["recall"] = float(recall.group(1))
        metrics["ndcg"] = float(ndcg.group(1))
        metrics["score"] = float(score.group(1))
    return metrics


def parse_chunk_count(ingest_stdout: str) -> int:
    match = re.search(r"->\s*(\d+)\s+chunks", ingest_stdout)
    return int(match.group(1)) if match else -1


def append_row(row: dict[str, Any]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", *PARAM_KEYS, "chunks", "recall@50", "ndcg@50", "score", "note"]
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **{key: row.get(key) for key in PARAM_KEYS},
                "chunks": row.get("chunks"),
                "recall@50": row.get("recall"),
                "ndcg@50": row.get("ndcg"),
                "score": row.get("score"),
                "note": row.get("note", ""),
            }
        )


def make_combo(**overrides: int) -> dict[str, int]:
    combo = dict(DEFAULTS)
    combo.update(overrides)
    return combo


def run_one(params: dict[str, int], *, note: str, data_path: str, eval_path: str) -> dict[str, Any]:
    print(f"\n=== {params}  note={note!r}")
    restart_index(params)
    ingest_out = run_ingest(data_path)
    metrics = run_eval(eval_path)
    row = {**params, **metrics, "chunks": parse_chunk_count(ingest_out), "note": note}
    append_row(row)
    print(
        f"   chunks={row['chunks']}  R@50={row.get('recall')}  "
        f"nDCG@50={row.get('ndcg')}  score={row.get('score')}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "axis", "custom"], default="axis")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--combo", type=str, default=None, help="JSON dict of param overrides for custom phase")
    parser.add_argument("--data-path", default="data/Dataset_sweep.json")
    parser.add_argument("--eval-path", default="data/Dataset_sweep_questions.jsonl")
    args = parser.parse_args()

    if args.dry:
        print("PARAMS:", PARAM_KEYS)
        print("DEFAULTS:", DEFAULTS)
        return

    if args.phase == "smoke":
        run_one(make_combo(), note="baseline", data_path=args.data_path, eval_path=args.eval_path)
        return

    if args.phase == "custom":
        overrides = json.loads(args.combo) if args.combo else {}
        run_one(make_combo(**overrides), note=f"custom:{overrides}", data_path=args.data_path, eval_path=args.eval_path)
        return

    runs: list[tuple[dict[str, int], str]] = [(make_combo(), "baseline")]
    for value in (600, 900, 1200, 1800, 2400, 3600):
        runs.append((make_combo(MAX_CHUNK_CHARS=value), f"axis:MAX_CHUNK_CHARS={value}"))
    for overlap_messages, overlap_chars in [(0, 0), (1, 500), (3, 1000), (5, 2000)]:
        runs.append(
            (
                make_combo(OVERLAP_MESSAGE_COUNT=overlap_messages, OVERLAP_CONTEXT_CHARS=overlap_chars),
                f"axis:OVERLAP({overlap_messages},{overlap_chars})",
            )
        )
    for value in (3600, 7200, 10800, 86400):
        runs.append((make_combo(MAX_TIME_GAP_SECONDS=value), f"axis:MAX_TIME_GAP_SECONDS={value}"))
    for combo, note in runs:
        try:
            run_one(combo, note=note, data_path=args.data_path, eval_path=args.eval_path)
        except Exception as exc:
            print(f"!! run failed for {combo} ({exc})")
            append_row({**combo, "recall": None, "ndcg": None, "score": None, "chunks": -1, "note": f"FAIL:{exc}"})


if __name__ == "__main__":
    main()
