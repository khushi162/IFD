#!/usr/bin/env python3
"""
Training entrypoint for IFD-PART2.

Features:
    - Reproducible training via --seed.
    - Explicit Ray shutdown.
    - Atomic metrics JSON writes.
    - Optional final model checkpoint.
    - Supports CascadeRouter and baseline strategies.
    - Supports L1/L2/L3 ablations.
"""

import argparse
import json
import os
import sys
import time

# WSL2 workaround. Proper Ray cleanup is still performed.
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

import numpy as np
import torch
import flwr as fl
import ray

from torch.utils.data import DataLoader, Subset

from data.loader import load_ieee_cis_data
from data.partitioner import GeographicPartitioner

from experiment.client import (
    IFDClient,
    FraudMLP,
    set_parameters,
)

from orchestration.flower_strategy import CascadeRouter


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="IFD-PART2 Federated Training"
    )

    p.add_argument(
        "--data-dir",
        default="./data/raw",
    )

    p.add_argument(
        "--nrows",
        type=int,
        default=None,
    )

    p.add_argument(
        "--num-clients",
        type=int,
        default=10,
    )

    p.add_argument(
        "--num-rounds",
        type=int,
        default=20,
    )

    p.add_argument(
        "--num-adversaries",
        type=int,
        default=0,
    )

    p.add_argument(
        "--attack-type",
        type=str,
        default=None,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    p.add_argument(
        "--epochs-per-round",
        type=int,
        default=1,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=44,
    )

    p.add_argument(
        "--run-label",
        type=str,
        default=None,
    )

    p.add_argument(
        "--save-model",
        type=str,
        default=None,
    )

    p.add_argument(
        "--results-dir",
        type=str,
        default="./results",
    )

    p.add_argument(
        "--disable-layer1",
        action="store_true",
    )

    p.add_argument(
        "--disable-layer2",
        action="store_true",
    )

    p.add_argument(
        "--disable-layer3",
        action="store_true",
    )

    p.add_argument(
        "--baseline",
        type=str,
        default=None,
    )

    return p.parse_args()


# ============================================================================
# Ray cleanup
# ============================================================================

def shutdown_ray():
    """
    Shut down the local Ray runtime.

    Safe to call if Ray was never initialized or was already shut down.
    """

    try:

        if ray.is_initialized():

            print(
                "\n[Ray cleanup] Shutting down Ray...",
                flush=True,
            )

            ray.shutdown(
                _exiting_interpreter=False,
                wait_for_processes=True,
            )

            print(
                "[Ray cleanup] Ray shutdown complete.",
                flush=True,
            )

        else:

            print(
                "\n[Ray cleanup] Ray is not initialized.",
                flush=True,
            )

    except Exception as exc:

        print(
            "[Ray cleanup] WARNING: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


# ============================================================================
# Atomic JSON writer
# ============================================================================

def atomic_json_dump(data, path):
    """
    Write JSON atomically.

    The temporary file prevents an interrupted process from leaving behind
    a partially-written result that could be mistaken for a valid result.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = path + ".tmp"

    try:

        with open(
            tmp_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                default=str,
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp_path,
            path,
        )

    except Exception:

        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

        raise


# ============================================================================
# Main
# ============================================================================

def main():

    args = parse_args()

    # ------------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------------

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    if args.num_clients < 1:
        raise ValueError(
            "--num-clients must be >= 1"
        )

    if args.num_rounds < 1:
        raise ValueError(
            "--num-rounds must be >= 1"
        )

    if args.num_adversaries < 0:
        raise ValueError(
            "--num-adversaries must be >= 0"
        )

    if args.num_adversaries > args.num_clients:
        raise ValueError(
            "--num-adversaries cannot exceed "
            "--num-clients"
        )

    if args.nrows is not None and args.nrows < 1:
        raise ValueError(
            "--nrows must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.epochs_per_round < 1:
        raise ValueError(
            "--epochs-per-round must be >= 1"
        )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"Device: {device}",
        flush=True,
    )

    if device.type == "cuda":

        print(
            f"  GPU: "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )

        print(
            f"  VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )

    # Ray actors already provide process isolation.
    num_workers = 0
    pin_memory = False

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    print(
        f"\nLoading data from: {args.data_dir}",
        flush=True,
    )

    try:

        train_ds, test_ds = load_ieee_cis_data(
            data_dir=args.data_dir,
            nrows=args.nrows,
            synthetic_fallback=False,
        )

    except FileNotFoundError:

        print(
            f"ERROR: IEEE-CIS CSV files not found in "
            f"'{args.data_dir}'.",
            flush=True,
        )

        return 1

    print(
        f"  Train samples: {len(train_ds)}",
        flush=True,
    )

    print(
        f"  Test samples:  {len(test_ds)}",
        flush=True,
    )

    input_dim = train_ds[0][0].shape[0]

    print(
        f"  Input dim: {input_dim}",
        flush=True,
    )

    # ------------------------------------------------------------------------
    # Partition data
    # ------------------------------------------------------------------------

    labels = train_ds.y.numpy()

    partitioner = GeographicPartitioner(
        num_clients=args.num_clients,
        seed=args.seed,
    )

    client_indices, _ = partitioner.partition(
        labels
    )

    # ------------------------------------------------------------------------
    # Flower client factory
    # ------------------------------------------------------------------------

    def client_fn(cid: str):

        client_idx = int(cid)

        indices = client_indices[client_idx]

        client_train_ds = Subset(
            train_ds,
            indices,
        )

        train_loader = DataLoader(
            client_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        is_adv = (
            client_idx < args.num_adversaries
        )

        client = IFDClient(
            cid=cid,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            epochs=args.epochs_per_round,
            lr=args.lr,
            is_adversary=is_adv,
            attack_type=(
                args.attack_type
                if is_adv
                else None
            ),
        )

        return client.to_client()

    # ------------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------------

    def weighted_average(metrics):

        if not metrics:
            return {}

        total_examples = sum(
            n
            for n, _ in metrics
        )

        if total_examples == 0:
            return {}

        keys = metrics[0][1].keys()

        return {
            key: sum(
                n * values[key]
                for n, values in metrics
            ) / total_examples
            for key in keys
        }

    # ------------------------------------------------------------------------
    # Ablation layers
    # ------------------------------------------------------------------------

    from layers.layer1_norm_cosine import (
        Layer1NormCosine,
    )

    from layers.layer2_spectral import (
        Layer2Spectral,
    )

    from layers.layer3_temporal import (
        Layer3Temporal,
    )

    class _PassL1(Layer1NormCosine):

        def score(self, gradients):

            n = len(gradients)

            return (
                torch.ones(n),
                torch.ones(n),
            )

    class _PassL2(Layer2Spectral):

        def score(self, gradients):

            n = len(gradients)

            return (
                torch.ones(n),
                torch.ones(n),
            )

    class _PassL3(Layer3Temporal):

        def score(
            self,
            gradients,
            client_ids=None,
        ):

            n = len(gradients)

            return (
                torch.ones(n),
                torch.ones(n),
            )

    layer1 = (
        _PassL1()
        if args.disable_layer1
        else None
    )

    layer2 = (
        _PassL2()
        if args.disable_layer2
        else None
    )

    layer3 = (
        _PassL3()
        if args.disable_layer3
        else None
    )

    # ------------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------------

    if args.baseline:

        from orchestration.baseline_strategy import (
            BaselineStrategy,
        )

        strategy = BaselineStrategy(
            baseline_name=args.baseline,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        print(
            f"Strategy: BaselineStrategy "
            f"({args.baseline})",
            flush=True,
        )

    else:

        strategy = CascadeRouter(
            layer1=layer1,
            layer2=layer2,
            layer3=layer3,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        print(
            "Strategy: CascadeRouter",
            flush=True,
        )

    # ------------------------------------------------------------------------
    # Experiment information
    # ------------------------------------------------------------------------

    print(
        "\nStarting FL simulation:",
        flush=True,
    )

    print(
        f"  Clients:      {args.num_clients}",
        flush=True,
    )

    print(
        f"  Rounds:       {args.num_rounds}",
        flush=True,
    )

    print(
        f"  Adversaries:  {args.num_adversaries}",
        flush=True,
    )

    print(
        f"  Attack:       {args.attack_type}",
        flush=True,
    )

    # ------------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------------

    t0 = time.time()

    history = None

    try:

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(
                num_rounds=args.num_rounds
            ),
            strategy=strategy,
            client_resources={
                "num_cpus": 10,
                "num_gpus": 0.5,
            },
            ray_init_args={
                "ignore_reinit_error": True,
            },
        )

        elapsed = time.time() - t0

        print(
            f"\nTraining complete in "
            f"{elapsed:.1f}s "
            f"({elapsed / max(1, args.num_rounds):.1f}s/round)",
            flush=True,
        )

    except Exception as exc:

        elapsed = time.time() - t0

        print(
            f"\nERROR: FL simulation failed after "
            f"{elapsed:.1f}s",
            flush=True,
        )

        print(
            f"  {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise

    finally:

        shutdown_ray()

    # ------------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------------

    if history is None:

        print(
            "ERROR: Simulation returned no history.",
            flush=True,
        )

        return 1

    # ------------------------------------------------------------------------
    # Save metrics atomically
    # ------------------------------------------------------------------------

    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    if args.run_label:

        filename = (
            f"{args.run_label}.json"
        )

    else:

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        run_type = (
            f"attack_{args.attack_type}_"
            f"{args.num_adversaries}adv"
            if args.num_adversaries > 0
            else "clean"
        )

        filename = (
            f"history_{run_type}_"
            f"{timestamp}.json"
        )

    metrics_path = os.path.join(
        args.results_dir,
        filename,
    )

    metrics = {
        "seed": args.seed,
        "run_label": args.run_label,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "num_adversaries": args.num_adversaries,
        "attack_type": args.attack_type,

        "losses_distributed":
            history.losses_distributed,

        "metrics_distributed":
            history.metrics_distributed,

        "metrics_centralized":
            history.metrics_centralized,
    }

    atomic_json_dump(
        metrics,
        metrics_path,
    )

    print(
        f"Metrics saved to: {metrics_path}",
        flush=True,
    )

    # ------------------------------------------------------------------------
    # Optional model checkpoint
    # ------------------------------------------------------------------------

    if args.save_model:

        save_dir = os.path.dirname(
            args.save_model
        )

        if save_dir:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )

        final_model = FraudMLP(
            input_dim=input_dim
        ).to(device)

        if (
            hasattr(
                strategy,
                "latest_aggregated_ndarrays",
            )
            and strategy.latest_aggregated_ndarrays
        ):

            set_parameters(
                final_model,
                strategy.latest_aggregated_ndarrays,
            )

            tmp_model = (
                args.save_model + ".tmp"
            )

            torch.save(
                final_model.state_dict(),
                tmp_model,
            )

            os.replace(
                tmp_model,
                args.save_model,
            )

            print(
                f"Model checkpoint saved to: "
                f"{args.save_model}",
                flush=True,
            )

        else:

            print(
                "WARNING: No aggregated parameters "
                "available; model checkpoint not saved.",
                flush=True,
            )

    return 0


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    try:

        sys.exit(main())

    except KeyboardInterrupt:

        print(
            "\nTraining interrupted by user.",
            flush=True,
        )

        shutdown_ray()

        sys.exit(130)

    except Exception as exc:

        print(
            f"\nFATAL ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        shutdown_ray()

        sys.exit(1)
