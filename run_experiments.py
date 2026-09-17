#!/usr/bin/env python3
"""
IEEE TIFS Experiment Orchestrator — IFD-PART2

22 experiments:

Section A:
    1 Clean baseline

Section B:
    9 attack experiments
    3 attacks x 3 ratios

Section C:
    12 ablations
    3 layers x 4 conditions

Features:
    - Experiment-level resume.
    - Seed-level completion marker.
    - Per-experiment logs.
    - Experiment timeout.
    - Process-group termination.
    - Ray cleanup.
    - Continues after failures.
    - Atomic completion markers.
    - Per-experiment model checkpoints.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "./results",
)

NUM_CLIENTS = int(
    os.environ.get("NUM_CLIENTS", "10")
)

NUM_ROUNDS = int(
    os.environ.get("NUM_ROUNDS", "50")
)

NROWS = int(
    os.environ.get("NROWS", "150000")
)

SAVE_MODEL_ROOT = os.environ.get(
    "SAVE_MODEL_ROOT",
    "",
)

SEED = int(
    os.environ.get("SEED", "44")
)

BATCH_SIZE = 512
EPOCHS = 10

EXPERIMENT_TIMEOUT_MINUTES = float(
    os.environ.get(
        "EXPERIMENT_TIMEOUT_MINUTES",
        "180",
    )
)

EXPERIMENT_TIMEOUT_SECONDS = (
    EXPERIMENT_TIMEOUT_MINUTES * 60
)

RATIOS = [
    0.10,
    0.20,
    0.40,
]

ATTACKS = [
    "sign_flip",
    "label_flip",
    "model_replace",
]

ATTACK_LABELS = {
    "sign_flip": "SignFlip",
    "label_flip": "LabelFlip",
    "model_replace": "ModelReplace",
}


# ============================================================================
# Paths
# ============================================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

LOG_PATH = os.path.join(
    RESULTS_DIR,
    "experiment_log.txt",
)

BOOKKEEPING_ROOT = os.path.join(
    RESULTS_DIR,
    "_experiments",
)

SEED_COMPLETE_MARKER = os.path.join(
    RESULTS_DIR,
    "COMPLETE",
)


# ============================================================================
# Utility
# ============================================================================

def timestamp():
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def write_log(message):

    with open(
        LOG_PATH,
        "a",
        encoding="utf-8",
    ) as fh:

        fh.write(
            message + "\n"
        )


def n_adv(ratio):

    return max(
        1,
        round(
            NUM_CLIENTS * ratio
        ),
    )


def pct(ratio):

    return (
        f"{int(round(ratio * 100))}pct"
    )


def validate_config():

    if NUM_CLIENTS < 1:
        raise ValueError(
            "NUM_CLIENTS must be >= 1"
        )

    if NUM_ROUNDS < 1:
        raise ValueError(
            "NUM_ROUNDS must be >= 1"
        )

    if NROWS < 1:
        raise ValueError(
            "NROWS must be >= 1"
        )

    if EXPERIMENT_TIMEOUT_MINUTES <= 0:
        raise ValueError(
            "EXPERIMENT_TIMEOUT_MINUTES must be > 0"
        )


# ============================================================================
# Experiment definitions
# ============================================================================

def build_experiments():

    experiments = []

    # ------------------------------------------------------------------------
    # Section A
    # ------------------------------------------------------------------------

    experiments.append(
        (
            "CleanRun_Simple",
            [],
        )
    )

    # ------------------------------------------------------------------------
    # Section B
    # ------------------------------------------------------------------------

    for attack in ATTACKS:

        for ratio in RATIOS:

            label = (
                f"Attack_"
                f"{ATTACK_LABELS[attack]}_"
                f"{pct(ratio)}"
            )

            experiments.append(
                (
                    label,
                    [
                        "--num-adversaries",
                        str(n_adv(ratio)),
                        "--attack-type",
                        attack,
                    ],
                )
            )

    # ------------------------------------------------------------------------
    # Section C
    # ------------------------------------------------------------------------

    layers = [
        (
            "NoL1",
            "--disable-layer1",
        ),
        (
            "NoL2",
            "--disable-layer2",
        ),
        (
            "NoL3",
            "--disable-layer3",
        ),
    ]

    conditions = [
        (
            "Clean",
            [],
        ),
        (
            "SignFlip20pct",
            [
                "--num-adversaries",
                str(n_adv(0.20)),
                "--attack-type",
                "sign_flip",
            ],
        ),
        (
            "LabelFlip20pct",
            [
                "--num-adversaries",
                str(n_adv(0.20)),
                "--attack-type",
                "label_flip",
            ],
        ),
        (
            "ModelReplace20pct",
            [
                "--num-adversaries",
                str(n_adv(0.20)),
                "--attack-type",
                "model_replace",
            ],
        ),
    ]

    for layer_name, layer_flag in layers:

        for condition_name, condition_args in conditions:

            experiments.append(
                (
                    f"Ablation_"
                    f"{layer_name}_"
                    f"{condition_name}",

                    [layer_flag]
                    + condition_args,
                )
            )

    return experiments


# ============================================================================
# Completion markers
# ============================================================================

def bookkeeping_dir(label):

    return os.path.join(
        BOOKKEEPING_ROOT,
        label,
    )


def complete_marker(label):

    return os.path.join(
        bookkeeping_dir(label),
        "COMPLETE",
    )


def experiment_complete(label):

    return os.path.isfile(
        complete_marker(label)
    )


def write_complete(label):

    directory = bookkeeping_dir(
        label
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    marker = complete_marker(
        label
    )

    tmp = marker + ".tmp"

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as fh:

        fh.write(
            f"Experiment: {label}\n"
            f"Seed: {SEED}\n"
            f"Completed: {timestamp()}\n"
        )

        fh.flush()
        os.fsync(
            fh.fileno()
        )

    os.replace(
        tmp,
        marker,
    )


def write_seed_complete():

    tmp = (
        SEED_COMPLETE_MARKER
        + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as fh:

        fh.write(
            f"Seed: {SEED}\n"
            f"Completed: {timestamp()}\n"
        )

        fh.flush()
        os.fsync(
            fh.fileno()
        )

    os.replace(
        tmp,
        SEED_COMPLETE_MARKER,
    )


# ============================================================================
# Ray cleanup
# ============================================================================

def cleanup_local_ray(reason=""):

    ray_address = os.environ.get(
        "RAY_ADDRESS",
        "",
    ).strip()

    if ray_address:

        message = (
            "[Ray cleanup] skipped: "
            f"RAY_ADDRESS={ray_address!r}"
        )

        print(
            message,
            flush=True,
        )

        write_log(message)

        return

    ray = shutil.which("ray")

    if ray is None:

        message = (
            "[Ray cleanup] ray executable "
            "not found."
        )

        print(
            message,
            flush=True,
        )

        write_log(message)

        return

    prefix = (
        f"[Ray cleanup] {reason}"
        if reason
        else "[Ray cleanup]"
    )

    print(
        f"{prefix} stopping local Ray...",
        flush=True,
    )

    write_log(
        f"{prefix} stopping local Ray..."
    )

    try:

        proc = subprocess.run(
            [
                ray,
                "stop",
                "--force",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=60,
        )

        if proc.stdout:

            print(
                proc.stdout.strip(),
                flush=True,
            )

            write_log(
                proc.stdout.strip()
            )

        if proc.returncode == 0:

            print(
                "[Ray cleanup] local Ray stopped.",
                flush=True,
            )

            write_log(
                "[Ray cleanup] local Ray stopped."
            )

        else:

            message = (
                "[Ray cleanup] ray stop returned "
                f"rc={proc.returncode}"
            )

            print(
                message,
                flush=True,
            )

            write_log(message)

    except Exception as exc:

        message = (
            "[Ray cleanup] exception: "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            message,
            flush=True,
        )

        write_log(message)


# ============================================================================
# Process cleanup
# ============================================================================

def terminate_process_tree(
    process,
    experiment_log,
):

    if process.poll() is not None:
        return

    try:

        experiment_log.write(
            f"\n[{timestamp()}] "
            "Terminating process tree.\n"
        )

        experiment_log.flush()

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

            experiment_log.write(
                f"[{timestamp()}] "
                "Graceful termination timed out; "
                "forcing termination.\n"
            )

            experiment_log.flush()

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

        experiment_log.write(
            f"[{timestamp()}] "
            f"Termination error: "
            f"{type(exc).__name__}: {exc}\n"
        )

        experiment_log.flush()


# ============================================================================
# Run one experiment
# ============================================================================

def run(label, extra_args):

    # ------------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------------

    if experiment_complete(label):

        message = (
            f"[{label}] SKIPPED — already complete."
        )

        print(
            message,
            flush=True,
        )

        write_log(message)

        return True

    # ------------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------------

    book_dir = bookkeeping_dir(
        label
    )

    os.makedirs(
        book_dir,
        exist_ok=True,
    )

    experiment_log_path = os.path.join(
        book_dir,
        "experiment.log",
    )

    # ------------------------------------------------------------------------
    # Model path
    # ------------------------------------------------------------------------

    if SAVE_MODEL_ROOT:

        model_dir = os.path.join(
            SAVE_MODEL_ROOT,
            "models",
        )

    else:

        model_dir = os.path.join(
            RESULTS_DIR,
            "models",
        )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    model_path = os.path.join(
        model_dir,
        f"{label}.pt",
    )

    # ------------------------------------------------------------------------
    # Ray cleanup before experiment
    # ------------------------------------------------------------------------

    cleanup_local_ray(
        reason=f"before {label}"
    )

    # ------------------------------------------------------------------------
    # Command
    # ------------------------------------------------------------------------

    cmd = [
        sys.executable,
        "train.py",

        "--num-clients",
        str(NUM_CLIENTS),

        "--num-rounds",
        str(NUM_ROUNDS),

        "--nrows",
        str(NROWS),

        "--batch-size",
        str(BATCH_SIZE),

        "--epochs-per-round",
        str(EPOCHS),

        "--results-dir",
        RESULTS_DIR,

        "--save-model",
        model_path,

        "--seed",
        str(SEED),

        "--run-label",
        label,
    ] + extra_args

    header = (
        "\n"
        + "=" * 78
        + "\n"
        + f"EXPERIMENT: {label}\n"
        + f"SEED:       {SEED}\n"
        + f"STARTED:    {timestamp()}\n"
        + f"TIMEOUT:    "
        f"{EXPERIMENT_TIMEOUT_MINUTES:.1f} min\n"
        + "COMMAND:\n"
        + " ".join(cmd)
        + "\n"
        + "=" * 78
        + "\n"
    )

    print(
        header,
        flush=True,
    )

    write_log(
        header.rstrip()
    )

    start = time.time()

    # ------------------------------------------------------------------------
    # Run process
    # ------------------------------------------------------------------------

    try:

        with open(
            experiment_log_path,
            "a",
            encoding="utf-8",
        ) as experiment_log:

            experiment_log.write(
                header
            )

            experiment_log.flush()

            env = os.environ.copy()

            env["RESULTS_DIR"] = RESULTS_DIR
            env["SEED"] = str(SEED)

            kwargs = {
                "env": env,
                "stdout": experiment_log,
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
                    timeout=EXPERIMENT_TIMEOUT_SECONDS
                )

            except subprocess.TimeoutExpired:

                elapsed = (
                    time.time() - start
                )

                message = (
                    f"[{label}] TIMEOUT — "
                    f"{elapsed / 60:.1f} min"
                )

                experiment_log.write(
                    f"\n[{timestamp()}] "
                    f"{message}\n"
                )

                experiment_log.flush()

                terminate_process_tree(
                    process,
                    experiment_log,
                )

                cleanup_local_ray(
                    reason=f"after timeout {label}"
                )

                print(
                    f"\n{message}\n",
                    flush=True,
                )

                write_log(message)

                return False

    except Exception as exc:

        elapsed = (
            time.time() - start
        )

        message = (
            f"[{label}] CRASHED — "
            f"{type(exc).__name__}: {exc} — "
            f"{elapsed / 60:.1f} min"
        )

        print(
            f"\n{message}\n",
            flush=True,
        )

        write_log(message)

        cleanup_local_ray(
            reason=f"after crash {label}"
        )

        return False

    # ------------------------------------------------------------------------
    # Always clean Ray
    # ------------------------------------------------------------------------

    elapsed = (
        time.time() - start
    )

    cleanup_local_ray(
        reason=f"after {label}"
    )

    # ------------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------------

    if return_code == 0:

        # train.py has atomically written the JSON.
        json_path = os.path.join(
            RESULTS_DIR,
            f"{label}.json",
        )

        if not os.path.isfile(
            json_path
        ):

            message = (
                f"[{label}] FAILED — "
                "train.py returned 0 but expected "
                f"metrics file is missing: {json_path}"
            )

            print(
                message,
                flush=True,
            )

            write_log(message)

            return False

        # Only now is the experiment durable.
        write_complete(label)

        message = (
            f"[{label}] SUCCESS — "
            f"{elapsed / 60:.1f} min — "
            f"finished {timestamp()}"
        )

        print(
            f"\n{message}\n",
            flush=True,
        )

        write_log(message)

        return True

    # ------------------------------------------------------------------------
    # Failure
    # ------------------------------------------------------------------------

    message = (
        f"[{label}] FAILED "
        f"(rc={return_code}) — "
        f"{elapsed / 60:.1f} min — "
        f"finished {timestamp()}"
    )

    print(
        f"\n{message}\n",
        flush=True,
    )

    write_log(message)

    return False


# ============================================================================
# Main
# ============================================================================

def main():

    validate_config()

    experiments = build_experiments()

    if len(experiments) != 22:

        raise RuntimeError(
            f"Expected 22 experiments, "
            f"generated {len(experiments)}."
        )

    # ------------------------------------------------------------------------
    # Already-complete seed
    # ------------------------------------------------------------------------

    if os.path.isfile(
        SEED_COMPLETE_MARKER
    ):

        message = (
            f"Seed {SEED} already complete. "
            "Nothing to run."
        )

        print(
            message,
            flush=True,
        )

        write_log(message)

        return 0

    # ------------------------------------------------------------------------
    # Suite header
    # ------------------------------------------------------------------------

    write_log("")
    write_log("=" * 78)
    write_log(
        "IEEE TIFS Experiment Orchestrator"
    )
    write_log(
        f"Started: {timestamp()}"
    )
    write_log(
        f"Seed: {SEED}"
    )
    write_log(
        f"Results: {RESULTS_DIR}"
    )
    write_log(
        f"Experiments: {len(experiments)}"
    )
    write_log(
        f"Timeout: "
        f"{EXPERIMENT_TIMEOUT_MINUTES:.1f} min"
    )
    write_log("=" * 78)

    cleanup_local_ray(
        reason="initial"
    )

    # ------------------------------------------------------------------------
    # Execute suite
    # ------------------------------------------------------------------------

    for index, (label, args) in enumerate(
        experiments,
        start=1,
    ):

        print(
            "\n"
            + "#" * 78
            + f"\n# Experiment "
            f"{index}/{len(experiments)}: "
            f"{label}"
            + "\n"
            + "#" * 78
            + "\n",
            flush=True,
        )

        run(
            label,
            args,
        )

    # ------------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------------

    cleanup_local_ray(
        reason="final"
    )

    # ------------------------------------------------------------------------
    # Determine durable state
    # ------------------------------------------------------------------------

    completed = [
        label
        for label, _ in experiments
        if experiment_complete(label)
    ]

    incomplete = [
        label
        for label, _ in experiments
        if not experiment_complete(label)
    ]

    write_log("")
    write_log("=" * 78)
    write_log("FINAL STATUS")
    write_log("=" * 78)
    write_log(
        f"Finished: {timestamp()}"
    )
    write_log(
        f"Completed: {len(completed)}/{len(experiments)}"
    )
    write_log(
        f"Incomplete: {len(incomplete)}"
    )

    if incomplete:

        write_log(
            "Incomplete experiments:"
        )

        for label in incomplete:
            write_log(
                f"  - {label}"
            )

        print(
            "\nIncomplete experiments:",
            flush=True,
        )

        for label in incomplete:

            print(
                f"  - {label}",
                flush=True,
            )

        return 1

    # ------------------------------------------------------------------------
    # All 22 durable
    # ------------------------------------------------------------------------

    write_seed_complete()

    message = (
        f"\nSUCCESS: All {len(experiments)} "
        f"experiments completed for seed {SEED}."
    )

    print(
        message,
        flush=True,
    )

    write_log(message)

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
            "\nInterrupted by user.",
            flush=True,
        )

        cleanup_local_ray(
            reason="KeyboardInterrupt"
        )

        write_log(
            f"INTERRUPTED: {timestamp()}"
        )

        sys.exit(130)

    except Exception as exc:

        message = (
            f"FATAL ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            message,
            flush=True,
        )

        cleanup_local_ray(
            reason="fatal error"
        )

        write_log(message)

        sys.exit(1)
