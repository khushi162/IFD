#!/usr/bin/env python3
"""
Seed sweep runner.

Runs run_experiments.py once per seed (45-55).

Features:
    - Seed-level resume.
    - Skips seeds with COMPLETE marker.
    - Per-seed stdout/stderr log.
    - Seed timeout.
    - Process-group termination.
    - Continues after failed/timed-out seeds.
    - Sweep-level summary.

Usage:
    python run_seed_sweep.py

Optional environment variables:
    NUM_CLIENTS
    NUM_ROUNDS
    NROWS
    EXPERIMENT_TIMEOUT_MINUTES
    SEED_TIMEOUT_MINUTES
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


# ============================================================================
# Configuration
# ============================================================================

SEEDS = list(
    range(45, 56)
)

BASE_RESULTS = "./results/Seed_Sweep2"

SWEEP_LOG = os.path.join(
    BASE_RESULTS,
    "seed_sweep_log.txt",
)

SEED_TIMEOUT_MINUTES = float(
    os.environ.get(
        "SEED_TIMEOUT_MINUTES",
        "3600",
    )
)

SEED_TIMEOUT_SECONDS = (
    SEED_TIMEOUT_MINUTES * 60
)


# ============================================================================
# Utility
# ============================================================================

def ts():

    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def log(
    message,
    fh=None,
):

    print(
        message,
        flush=True,
    )

    if fh:

        fh.write(
            message + "\n"
        )

        fh.flush()


def seed_complete(
    results_dir,
):

    return os.path.isfile(
        os.path.join(
            results_dir,
            "COMPLETE",
        )
    )


# ============================================================================
# Process termination
# ============================================================================

def terminate_process_tree(
    process,
    seed_log,
):

    if process.poll() is not None:
        return

    try:

        seed_log.write(
            f"\n[{ts()}] "
            "Terminating seed process tree.\n"
        )

        seed_log.flush()

        if os.name == "posix":

            os.killpg(
                process.pid,
                signal.SIGTERM,
            )

        else:

            process.terminate()

        try:

            process.wait(
                timeout=15
            )

        except subprocess.TimeoutExpired:

            seed_log.write(
                f"[{ts()}] "
                "Forcing seed process termination.\n"
            )

            seed_log.flush()

            if os.name == "posix":

                os.killpg(
                    process.pid,
                    signal.SIGKILL,
                )

            else:

                process.kill()

            process.wait(
                timeout=15
            )

    except ProcessLookupError:
        pass

    except Exception as exc:

        seed_log.write(
            f"[{ts()}] "
            f"Termination error: "
            f"{type(exc).__name__}: {exc}\n"
        )

        seed_log.flush()


# ============================================================================
# Run one seed
# ============================================================================

def run_seed(
    seed,
    sweep_log,
):

    results_dir = os.path.join(
        BASE_RESULTS,
        f"seed_{seed}",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------------

    if seed_complete(
        results_dir
    ):

        message = (
            f"Seed {seed:3d}: "
            "SKIPPED — already complete."
        )

        log(
            message,
            sweep_log,
        )

        return message

    # ------------------------------------------------------------------------
    # Per-seed log
    # ------------------------------------------------------------------------

    seed_log_path = os.path.join(
        results_dir,
        "seed.log",
    )

    log(
        f"\n--- Seed {seed} ---",
        sweep_log,
    )

    log(
        f"Directory: {results_dir}",
        sweep_log,
    )

    log(
        f"Started: {ts()}",
        sweep_log,
    )

    start = time.time()

    try:

        with open(
            seed_log_path,
            "a",
            encoding="utf-8",
        ) as seed_log:

            seed_log.write(
                "\n"
                + "=" * 80
                + "\n"
                + f"SEED {seed} STARTED {ts()}\n"
                + f"TIMEOUT: "
                f"{SEED_TIMEOUT_MINUTES:.1f} min\n"
                + "=" * 80
                + "\n"
            )

            seed_log.flush()

            env = os.environ.copy()

            env["RESULTS_DIR"] = (
                results_dir
            )

            env["SEED"] = str(seed)

            cmd = [
                sys.executable,
                "run_experiments.py",
            ]

            kwargs = {
                "env": env,
                "stdout": seed_log,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
            }

            if os.name == "posix":

                kwargs[
                    "start_new_session"
                ] = True

            else:

                kwargs[
                    "creationflags"
                ] = (
                    subprocess
                    .CREATE_NEW_PROCESS_GROUP
                )

            process = subprocess.Popen(
                cmd,
                **kwargs,
            )

            try:

                return_code = process.wait(
                    timeout=SEED_TIMEOUT_SECONDS
                )

            except subprocess.TimeoutExpired:

                elapsed = (
                    time.time() - start
                )

                message = (
                    f"Seed {seed:3d}: TIMEOUT — "
                    f"{elapsed / 60:.1f} min"
                )

                seed_log.write(
                    f"\n[{ts()}] "
                    f"{message}\n"
                )

                seed_log.flush()

                terminate_process_tree(
                    process,
                    seed_log,
                )

                log(
                    message,
                    sweep_log,
                )

                return message

            elapsed = (
                time.time() - start
            )

            # ----------------------------------------------------------------
            # Verify seed completion marker.
            # ----------------------------------------------------------------

            if (
                return_code == 0
                and seed_complete(results_dir)
            ):

                message = (
                    f"Seed {seed:3d}: SUCCESS — "
                    f"{elapsed / 60:.1f} min — "
                    f"finished {ts()}"
                )

            elif return_code == 0:

                message = (
                    f"Seed {seed:3d}: FAILED — "
                    "run_experiments.py returned 0 "
                    "but COMPLETE marker is missing."
                )

            else:

                message = (
                    f"Seed {seed:3d}: FAILED "
                    f"(rc={return_code}) — "
                    f"{elapsed / 60:.1f} min — "
                    f"finished {ts()}"
                )

            seed_log.write(
                f"\n[{ts()}] {message}\n"
            )

            seed_log.flush()

            log(
                message,
                sweep_log,
            )

            return message

    except Exception as exc:

        elapsed = (
            time.time() - start
        )

        message = (
            f"Seed {seed:3d}: CRASHED — "
            f"{type(exc).__name__}: {exc} — "
            f"{elapsed / 60:.1f} min"
        )

        log(
            message,
            sweep_log,
        )

        return message


# ============================================================================
# Main
# ============================================================================

def main():

    os.makedirs(
        BASE_RESULTS,
        exist_ok=True,
    )

    summaries = []

    with open(
        SWEEP_LOG,
        "a",
        encoding="utf-8",
    ) as sweep_log:

        log(
            "\n" + "=" * 80,
            sweep_log,
        )

        log(
            f"SEED SWEEP STARTED {ts()}",
            sweep_log,
        )

        log(
            f"Seeds: {SEEDS}",
            sweep_log,
        )

        log(
            f"Seed timeout: "
            f"{SEED_TIMEOUT_MINUTES:.1f} min",
            sweep_log,
        )

        log(
            "=" * 80,
            sweep_log,
        )

        for seed in SEEDS:

            summary = run_seed(
                seed,
                sweep_log,
            )

            summaries.append(
                summary
            )

        # --------------------------------------------------------------------
        # Final sweep summary
        # --------------------------------------------------------------------

        log(
            "\n" + "=" * 80,
            sweep_log,
        )

        log(
            f"SEED SWEEP COMPLETE {ts()}",
            sweep_log,
        )

        log(
            "=" * 80,
            sweep_log,
        )

        for summary in summaries:

            log(
                summary,
                sweep_log,
            )

    # ------------------------------------------------------------------------
    # Exit nonzero if any seed is incomplete.
    # ------------------------------------------------------------------------

    incomplete = []

    for seed in SEEDS:

        results_dir = os.path.join(
            BASE_RESULTS,
            f"seed_{seed}",
        )

        if not seed_complete(
            results_dir
        ):

            incomplete.append(
                seed
            )

    if incomplete:

        print(
            "\nSweep finished with incomplete seeds:",
            flush=True,
        )

        for seed in incomplete:

            print(
                f"  - seed {seed}",
                flush=True,
            )

        return 1

    print(
        "\nSUCCESS: All seeds completed.",
        flush=True,
    )

    return 0


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nSeed sweep interrupted.",
            flush=True,
        )

        sys.exit(130)

    except Exception as exc:

        print(
            f"\nFATAL ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        sys.exit(1)
