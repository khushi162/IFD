"""
BaselineStrategy: Flower Strategy wrapping BaselineAdapter (B1-B9).

Provides the same aggregate_fit interface as CascadeRouter so that
train.py can swap it in via --baseline without any other changes.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from baselines.adapter import BaselineAdapter


class BaselineStrategy(FedAvg):
    """
    Flower strategy that delegates aggregation to a single baseline (B1-B9).

    Reuses the same flatten/reconstruct logic as CascadeRouter so results
    are directly comparable. No cascade layers, no reputation tracking.
    """

    def __init__(
        self,
        baseline_name: str,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **baseline_kwargs,
    ):
        super().__init__(
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.adapter = BaselineAdapter(baseline_name, **baseline_kwargs)
        self.latest_aggregated_ndarrays: Optional[List[np.ndarray]] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using the chosen baseline, then reconstruct per-layer arrays."""

        if not results:
            return None, {}

        # ------------------------------------------------------------------
        # 1. Extract client IDs and parameter arrays
        # ------------------------------------------------------------------
        client_ids: List[str] = []
        client_param_ndarrays: List[List[np.ndarray]] = []

        for client_proxy, fit_res in results:
            client_ids.append(str(client_proxy.cid))
            client_param_ndarrays.append(
                parameters_to_ndarrays(fit_res.parameters)
            )

        # ------------------------------------------------------------------
        # 2. Flatten each client's parameters into one float32 vector
        # ------------------------------------------------------------------
        flattened: List[torch.Tensor] = []
        shapes: List[tuple] = []
        dtypes: List[np.dtype] = []

        # Capture shapes/dtypes from first client (all clients share same architecture)
        for arr in client_param_ndarrays[0]:
            shapes.append(arr.shape)
            dtypes.append(arr.dtype)

        for ndarrays in client_param_ndarrays:
            parts = [
                np.asarray(arr, dtype=np.float32).reshape(-1)
                for arr in ndarrays
            ]
            flat = np.concatenate(parts, axis=0)
            flattened.append(torch.from_numpy(flat.copy()))

        print(
            f"[{self.adapter.name}] Round {server_round}: "
            f"{len(client_ids)} clients, "
            f"{flattened[0].numel()} parameters/client",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 3. Aggregate with the baseline
        #    FLTrust (b6) falls back to mean-of-clients as server_gradient
        #    when no root dataset is available.
        # ------------------------------------------------------------------
        agg_flat: torch.Tensor = self.adapter.aggregate(
            flattened,
            client_ids=client_ids,
        )

        # ------------------------------------------------------------------
        # 4. Reconstruct per-layer arrays from the flat aggregated vector
        # ------------------------------------------------------------------
        agg_numpy = agg_flat.detach().cpu().numpy().astype(np.float32)

        aggregated_ndarrays: List[np.ndarray] = []
        offset = 0
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            size = int(np.prod(shape))
            layer_flat = agg_numpy[offset : offset + size]
            offset += size

            if np.issubdtype(dtype, np.floating):
                aggregated_ndarrays.append(
                    layer_flat.reshape(shape).astype(dtype)
                )
            else:
                # Non-floating buffers: copy from first client (same as CascadeRouter)
                aggregated_ndarrays.append(
                    client_param_ndarrays[0][i].copy()
                )

        self.latest_aggregated_ndarrays = aggregated_ndarrays

        return ndarrays_to_parameters(aggregated_ndarrays), {
            "server_round": server_round,
        }
