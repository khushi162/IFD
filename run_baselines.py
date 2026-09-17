#!/usr/bin/env python3
"""
Baseline Experiment Orchestrator  IFD-PART2
=============================================

Runs Clean + Attack Robustness suite for B1 (FedAvg) and B2 (Krum),
saving results to:

    results/baselines/<baseline_name>/<experiment_label>.json

10 experiments per baseline = 20 total.

Skip logic: if the output JSON already exists AND contains non-empty
metrics_distributed (i.e. was completed with full metric logging),
the experiment is skipped. This lets you safely re-run after a crash.

Usage:
    python run_baselines.py

Environment variables:
    RESULTS_DIR   default: ./results
    NUM_CLIENTS   default: 10
    NUM_ROUNDS    default: 50
    NROWS         default: 150000
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "10"))
NUM_ROUNDS  = int(os.environ.get("NUM_ROUNDS",  "50"))
NROWS       = int(os.environ.get("NROWS",       "150000"))

BATCH_SIZE = 512
EPOCHS     = 10

RATIOS  = [0.10, 0.20, 0.40]

ATTACKS = ["sign_flip", "label_flip", "model_replace"]

ATTACK_LABELS = {
    "sign_flip":     "SignFlip",
    "label_flip":    "LabelFlip",
    "model_replace": "ModelReplace",
}

# Only B1 and B2
BASELINES = ["b1_fedavg", "b2_krum"]


# ============================================================================
# Paths
# ============================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(RESULTS_DIR, "baselines_experiment_log.txt")

BASE_CMD = [
    sys.executable, "train.py",
    "--num-clients",      str(NUM_CLIENTS),
    "--num-rounds",       str(NUM_ROUNDS),
    "--nrows",            str(NROWS),
    "--batch-size",       str(BATCH_SIZE),
    "--epochs-per-round", str(EPOCHS),
]


# ============================================================================
# Bookkeeping
# ============================================================================

results_summary = []


# ============================================================================
# Utility
# ============================================================================

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_log(message: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def n_adv(ratio: float) -> int:
    return max(1, round(NUM_CLIENTS * ratio))


def pct(ratio: float) -> str:
    return f"{int(round(ratio * 100))}pct"


def is_complete(path: str) -> bool:
    """
    Return True only if the JSON at path exists AND has non-empty
    metrics_distributed � meaning it was run with full metric logging.
    Experiments completed before the metrics fix will be re-run.
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("metrics_distributed"))
    except Exception:
        return False


# ============================================================================
# Ray cleanup
# ============================================================================

def cleanup_local_ray(reason: str = "") -> None:
    if os.environ.get("RAY_ADDRESS", "").strip():
        return
    ray_executable = shutil.which("ray")
    if ray_executable is None:
        return
    prefix = f"[Ray cleanup] {reason}" if reason else "[Ray cleanup]"
    print(f"{prefix} stopping local Ray...", flush=True)
    write_log(f"{prefix} stopping local Ray...")
    try:
        proc = subprocess.run(
            [ray_executable, "stop", "--force"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = proc.stdout.strip()
        if output:
            print(output, flush=True)
            write_log(output)
    except Exception as exc:
        write_log(f"[Ray cleanup] exception: {exc}")


# ============================================================================
# Run one experiment
# ============================================================================

def run(baseline: str, label: str, extra_args: list) -> bool:
    baseline_results_dir = os.path.join(RESULTS_DIR, "baselines", baseline)
    os.makedirs(baseline_results_dir, exist_ok=True)

    out_path = os.path.join(baseline_results_dir, f"{label}.json")

    if is_complete(out_path):
        msg = f"[{baseline}/{label}] SKIPPED (already complete with metrics)"
        print(msg, flush=True)
        write_log(msg)
        results_summary.append(msg)
        return True

    cleanup_local_ray(reason=f"before {baseline}/{label}")

    cmd = BASE_CMD + [
        "--baseline",    baseline,
        "--run-label",   label,
        "--results-dir", baseline_results_dir,
    ] + extra_args

    separator = "=" * 78
    header = (
        f"\n{separator}\n"
        f"  BASELINE:   {baseline}\n"
        f"  EXPERIMENT: {label}\n"
        f"  STARTED:    {timestamp()}\n"
        f"  CMD:        {' '.join(cmd)}\n"
        f"{separator}\n"
    )
    print(header, flush=True)
    write_log(header.rstrip())

    start_time = time.time()

    try:
        process = subprocess.run(cmd, check=False)
        return_code = process.returncode
    except Exception as exc:
        elapsed = time.time() - start_time
        summary = (
            f"[{baseline}/{label}] FAILED "
            f"(exception: {type(exc).__name__}: {exc}) "
            f"� {elapsed / 60:.1f} min"
        )
        print(f"\n{summary}\n", flush=True)
        write_log(summary)
        results_summary.append(summary)
        cleanup_local_ray(reason=f"after exception in {baseline}/{label}")
        return False

    elapsed = time.time() - start_time
    cleanup_local_ray(reason=f"after {baseline}/{label}")

    status  = "SUCCESS" if return_code == 0 else f"FAILED (rc={return_code})"
    success = return_code == 0

    summary = (
        f"[{baseline}/{label}] {status} � "
        f"{elapsed / 60:.1f} min � "
        f"finished {timestamp()}"
    )
    print(f"\n{summary}\n", flush=True)
    write_log(summary)
    results_summary.append(summary)
    return success


# ============================================================================
# Main
# ============================================================================

def main() -> int:

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 78 + "\n")
        f.write("Baseline Experiment Orchestrator � RESUMED/STARTED\n")
        f.write(f"Started:      {timestamp()}\n")
        f.write(f"Baselines:    {', '.join(BASELINES)}\n")
        f.write(f"Results dir:  {RESULTS_DIR}\n")
        f.write(f"Num clients:  {NUM_CLIENTS}\n")
        f.write(f"Num rounds:   {NUM_ROUNDS}\n")
        f.write(f"NROWS:        {NROWS}\n")
        f.write("=" * 78 + "\n\n")

    for baseline in BASELINES:

        print(
            f"\n\n{'#' * 78}\n"
            f"  BASELINE: {baseline.upper()}\n"
            f"{'#' * 78}\n",
            flush=True,
        )
        write_log(f"\n### BASELINE: {baseline} ###")

        # Section A: Clean run
        run(baseline, "CleanRun_Simple", [])

        # Section B: Attack robustness
        for attack in ATTACKS:
            for ratio in RATIOS:
                label = f"Attack_{ATTACK_LABELS[attack]}_{pct(ratio)}"
                run(
                    baseline,
                    label,
                    [
                        "--num-adversaries", str(n_adv(ratio)),
                        "--attack-type",     attack,
                    ],
                )

    # --------------------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------------------
    total     = len(results_summary)
    skipped   = sum("SKIPPED"  in line for line in results_summary)
    failed    = sum("FAILED"   in line for line in results_summary)
    succeeded = sum("SUCCESS"  in line for line in results_summary)

    print("\n\n=== ALL BASELINE EXPERIMENTS COMPLETE ===", flush=True)
    print(f"Total:     {total}", flush=True)
    print(f"Succeeded: {succeeded}", flush=True)
    print(f"Skipped:   {skipped}",   flush=True)
    print(f"Failed:    {failed}",    flush=True)

    write_log("\n" + "=" * 78)
    write_log("FINAL SUMMARY")
    write_log(f"Finished:  {timestamp()}")
    write_log(f"Total:     {total}")
    write_log(f"Succeeded: {succeeded}")
    write_log(f"Skipped:   {skipped}")
    write_log(f"Failed:    {failed}")
    write_log("")

    for line in results_summary:
        print(line, flush=True)
        write_log(line)

    return 1 if failed else 0


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", flush=True)
        try:
            cleanup_local_ray(reason="KeyboardInterrupt")
            write_log(f"\nINTERRUPTED: {timestamp()}")
        except Exception:
            pass
        sys.exit(130)
    except Exception as exc:
        print(f"\n\nFATAL ERROR: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
