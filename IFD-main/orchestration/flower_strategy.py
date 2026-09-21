"""
CascadeRouter: Flower Strategy Adapter for Gated Cascade Defense
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import logging

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

from layers.layer1_norm_cosine import Layer1NormCosine
from layers.layer2_spectral import Layer2Spectral
from layers.layer3_temporal import Layer3Temporal
from orchestration.reputation import (
    ReputationTracker,
    reputation_weighted_aggregate,
)
from orchestration.threshold_controller import ThresholdController


class CascadeRouter(FedAvg):
    """
    Flower Strategy implementing the Three-Layer Gated Cascade Defense.

    Sequentially routes client updates through:
      Layer 1: Norm/Cosine Filter
      Layer 2: Spectral LOO Anomaly Detector
      Layer 3: Temporal Consistency Tracker

    Dynamically escalates thresholds via ThresholdController and tracks client
    reputations via ReputationTracker for reputation-weighted aggregation.
    """

    def __init__(
        self,
        layer1: Optional[Layer1NormCosine] = None,
        layer2: Optional[Layer2Spectral] = None,
        layer3: Optional[Layer3Temporal] = None,
        threshold_controller: Optional[ThresholdController] = None,
        reputation_tracker: Optional[ReputationTracker] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layer1 = layer1 or Layer1NormCosine()
        self.layer2 = layer2 or Layer2Spectral()
        self.layer3 = layer3 or Layer3Temporal()

        self.threshold_controller = (
            threshold_controller
            or ThresholdController()
        )

        self.reputation_tracker = (
            reputation_tracker
            or ReputationTracker()
        )

        self.all_seen_cids: Set[str] = set()
        self.consecutive_new_cid_rounds: int = 0

        self.latest_aggregated_ndarrays: Optional[
            List[np.ndarray]
        ] = None

    # ========================================================================
    # Parameter validation
    # ========================================================================

    @staticmethod
    def _validate_client_parameters(
        client_ids: List[str],
        client_param_ndarrays: List[List[np.ndarray]],
    ) -> None:
        """
        Verify that every client returned the exact same parameter structure.

        This MUST happen before the cascade layers run.

        A federated model must have identical parameter tensor shapes on every
        client. If one client sends even one tensor with a different shape,
        the update cannot safely participate in the same aggregation.
        """

        if not client_param_ndarrays:
            raise ValueError(
                "No client parameter arrays were received."
            )

        reference = client_param_ndarrays[0]
        reference_cid = client_ids[0]

        # --------------------------------------------------------------------
        # Number of tensors
        # --------------------------------------------------------------------

        expected_num_arrays = len(reference)

        for client_idx, arrays in enumerate(
            client_param_ndarrays
        ):

            cid = client_ids[client_idx]

            if len(arrays) != expected_num_arrays:

                raise ValueError(
                    "HETEROGENEOUS CLIENT PARAMETERS\n"
                    f"Reference client: {reference_cid}\n"
                    f"Reference tensors: {expected_num_arrays}\n"
                    f"Client {cid} tensors: {len(arrays)}"
                )

        # --------------------------------------------------------------------
        # Tensor shapes and dtypes
        # --------------------------------------------------------------------

        for client_idx, arrays in enumerate(
            client_param_ndarrays
        ):

            cid = client_ids[client_idx]

            for layer_idx, array in enumerate(arrays):

                reference_array = reference[layer_idx]

                if array.shape != reference_array.shape:

                    raise ValueError(
                        "HETEROGENEOUS CLIENT PARAMETER SHAPE\n"
                        f"Reference client: {reference_cid}\n"
                        f"Bad client:       {cid}\n"
                        f"Parameter index:  {layer_idx}\n"
                        f"Reference shape:  {reference_array.shape}\n"
                        f"Client shape:     {array.shape}\n"
                    )

                # Dtype mismatches are normally harmless for aggregation,
                # so log them rather than treating them as fatal.
                if array.dtype != reference_array.dtype:

                    logging.warning(
                        "Client %s parameter %d dtype differs: "
                        "reference=%s, client=%s",
                        cid,
                        layer_idx,
                        reference_array.dtype,
                        array.dtype,
                    )

    # ========================================================================
    # Flatten parameters
    # ========================================================================

    @staticmethod
    def _flatten_parameters(
        client_ids: List[str],
        client_param_ndarrays: List[List[np.ndarray]],
    ) -> List[torch.Tensor]:
        """
        Flatten each client's complete parameter set into one vector.

        Every parameter array is included and flattened in the exact order
        supplied by Flower.
        """

        flattened_tensors: List[torch.Tensor] = []

        for client_idx, ndarrays in enumerate(
            client_param_ndarrays
        ):

            cid = client_ids[client_idx]

            flat_parts = []

            for parameter_idx, array in enumerate(ndarrays):

                # The model parameters should be numeric arrays.
                if not np.issubdtype(
                    array.dtype,
                    np.number,
                ):

                    raise TypeError(
                        "NON-NUMERIC CLIENT PARAMETER\n"
                        f"Client: {cid}\n"
                        f"Parameter index: {parameter_idx}\n"
                        f"Dtype: {array.dtype}"
                    )

                flat_parts.append(
                    np.asarray(
                        array,
                        dtype=np.float32,
                    ).reshape(-1)
                )

            if not flat_parts:

                raise ValueError(
                    f"Client {cid} returned zero parameter arrays."
                )

            flat_vec = np.concatenate(
                flat_parts,
                axis=0,
            )

            flattened_tensors.append(
                torch.from_numpy(
                    flat_vec.copy()
                )
            )

        # --------------------------------------------------------------------
        # Final flattened-size validation
        # --------------------------------------------------------------------

        expected_size = flattened_tensors[0].numel()

        mismatches = []

        for idx, tensor in enumerate(
            flattened_tensors
        ):

            actual_size = tensor.numel()

            if actual_size != expected_size:

                mismatches.append(
                    (
                        client_ids[idx],
                        expected_size,
                        actual_size,
                    )
                )

        if mismatches:

            details = "\n".join(
                f"  client={cid}: expected={expected}, "
                f"actual={actual}"
                for cid, expected, actual in mismatches
            )

            raise ValueError(
                "INCONSISTENT FLATTENED PARAMETER SIZE\n"
                f"Reference size: {expected_size}\n"
                f"Mismatches:\n{details}"
            )

        return flattened_tensors

    # ========================================================================
    # Aggregate fit
    # ========================================================================

    def aggregate_fit(
        self,
        server_round: int,
        results: List[
            Tuple[ClientProxy, FitRes]
        ],
        failures: List[
            Union[
                Tuple[ClientProxy, FitRes],
                BaseException,
            ]
        ],
    ) -> Tuple[
        Optional[Parameters],
        Dict[str, Scalar],
    ]:

        """Aggregate fit results using the Gated Cascade Defense pipeline."""

        if not results:
            return None, {}

        # --------------------------------------------------------------------
        # 1. Extract client IDs and parameters
        # --------------------------------------------------------------------

        client_ids: List[str] = []
        client_param_ndarrays: List[
            List[np.ndarray]
        ] = []

        for client_proxy, fit_res in results:

            client_id = str(
                client_proxy.cid
            )

            client_ids.append(client_id)

            ndarrays = parameters_to_ndarrays(
                fit_res.parameters
            )

            client_param_ndarrays.append(
                ndarrays
            )

        # --------------------------------------------------------------------
        # 2. Validate parameter structure BEFORE flattening
        # --------------------------------------------------------------------

        self._validate_client_parameters(
            client_ids,
            client_param_ndarrays,
        )

        # --------------------------------------------------------------------
        # 3. Flatten complete parameter vectors
        # --------------------------------------------------------------------

        flattened_tensors = (
            self._flatten_parameters(
                client_ids,
                client_param_ndarrays,
            )
        )

        # Diagnostic information
        vector_sizes = [
            tensor.numel()
            for tensor in flattened_tensors
        ]

        logging.info(
            "Round %d: %d clients, flattened parameter size=%d",
            server_round,
            len(client_ids),
            vector_sizes[0],
        )

        print(
            f"[CascadeRouter] Round {server_round}: "
            f"{len(client_ids)} clients, "
            f"{vector_sizes[0]} parameters/client",
            flush=True,
        )

        # --------------------------------------------------------------------
        # Instrument client ID persistence
        # --------------------------------------------------------------------

        num_clients = len(client_ids)

        if num_clients > 0:

            new_cids = [
                cid
                for cid in client_ids
                if cid not in self.all_seen_cids
            ]

            fraction_new = (
                len(new_cids) / num_clients
            )

            if fraction_new > 0.9:
                self.consecutive_new_cid_rounds += 1
            else:
                self.consecutive_new_cid_rounds = 0

            if self.consecutive_new_cid_rounds > 3:

                logging.warning(
                    "Client ID persistence may be broken: "
                    f"{fraction_new * 100:.1f}% new client IDs "
                    f"for {self.consecutive_new_cid_rounds} "
                    "consecutive rounds. "
                    "Reputation and Layer 3 temporal state "
                    "may be resetting every round."
                )

            self.all_seen_cids.update(
                client_ids
            )

        # --------------------------------------------------------------------
        # 4. Run sequential cascade layers
        # --------------------------------------------------------------------

        a1, c1 = self.layer1.score(
            flattened_tensors
        )

        a2, c2 = self.layer2.score(
            flattened_tensors
        )

        a3, c3 = self.layer3.score(
            flattened_tensors,
            client_ids,
        )

        # Combined acceptance score per client
        a_combined = torch.min(
            torch.min(a1, a2),
            a3,
        )

        # --------------------------------------------------------------------
        # 5. Update threshold controller
        # --------------------------------------------------------------------

        thresholds = (
            self.threshold_controller.update(
                {
                    "layer1": a1,
                    "layer2": a2,
                    "layer3": a3,
                }
            )
        )

        t1 = thresholds.get(
            "layer1",
            0.0,
        )

        t2 = thresholds.get(
            "layer2",
            0.0,
        )

        t3 = thresholds.get(
            "layer3",
            0.0,
        )

        hard_reject = (
            (a1 < t1)
            | (a2 < t2)
            | (a3 < t3)
        )

        a_combined_gated = torch.where(
            hard_reject,
            torch.zeros_like(
                a_combined
            ),
            a_combined,
        )

        # --------------------------------------------------------------------
        # 6. Update reputation tracker
        # --------------------------------------------------------------------

        reputations_dict = (
            self.reputation_tracker.update(
                client_ids,
                a_combined_gated,
            )
        )

        reputations_tensor = torch.tensor(
            [
                reputations_dict[cid]
                for cid in client_ids
            ],
            dtype=torch.float32,
        )

        # --------------------------------------------------------------------
        # 7. Reputation-weighted aggregation
        # --------------------------------------------------------------------

        rep_weights_tensor = (
            self.reputation_tracker.get_weights(
                client_ids
            )
        )

        rep_weights_tensor = torch.where(
            hard_reject,
            torch.zeros_like(
                rep_weights_tensor
            ),
            rep_weights_tensor,
        )

        rep_weights = (
            rep_weights_tensor
            .detach()
            .cpu()
            .numpy()
        )

        sum_rep = float(
            np.sum(rep_weights)
        )

        if sum_rep <= 1e-8:

            logging.warning(
                "Round %d: all clients rejected or "
                "zero-weighted; falling back to "
                "uniform aggregation.",
                server_round,
            )

            rep_weights = (
                np.ones(
                    num_clients,
                    dtype=np.float64,
                )
                / max(1, num_clients)
            )

            sum_rep = 1.0

        # --------------------------------------------------------------------
        # 8. Aggregate arrays layer-by-layer
        # --------------------------------------------------------------------

        if not client_param_ndarrays:
            return None, {}

        num_layers = len(
            client_param_ndarrays[0]
        )

        aggregated_ndarrays: List[
            np.ndarray
        ] = []

        for layer_idx in range(num_layers):

            reference_array = (
                client_param_ndarrays[0][
                    layer_idx
                ]
            )

            orig_dtype = (
                reference_array.dtype
            )

            if np.issubdtype(
                orig_dtype,
                np.floating,
            ):

                layer_sum = np.zeros_like(
                    reference_array,
                    dtype=np.float64,
                )

                for client_idx, client_ndarrays in enumerate(
                    client_param_ndarrays
                ):

                    layer_sum += (
                        client_ndarrays[
                            layer_idx
                        ]
                        * rep_weights[client_idx]
                    )

                agg_layer = (
                    layer_sum / sum_rep
                ).astype(orig_dtype)

            else:

                # Non-floating tensors are copied from the first client.
                #
                # This preserves the original behavior. If your model has
                # integer buffers such as BatchNorm counters, those should
                # ideally be handled explicitly in a future revision.
                agg_layer = (
                    reference_array.copy()
                )

            aggregated_ndarrays.append(
                agg_layer
            )

        # --------------------------------------------------------------------
        # 9. Store latest aggregate
        # --------------------------------------------------------------------

        self.latest_aggregated_ndarrays = (
            aggregated_ndarrays
        )

        aggregated_parameters = (
            ndarrays_to_parameters(
                aggregated_ndarrays
            )
        )

        # --------------------------------------------------------------------
        # 10. Metrics
        # --------------------------------------------------------------------

        metrics: Dict[str, Scalar] = {

            "server_round":
                server_round,

            "layer1_reject_rate_raw":
                float(
                    (a1 < 0.5)
                    .float()
                    .mean()
                    .item()
                ),

            "layer1_reject_rate_at_threshold":
                float(
                    (a1 < t1)
                    .float()
                    .mean()
                    .item()
                ),

            "layer2_reject_rate_raw":
                float(
                    (a2 < 0.5)
                    .float()
                    .mean()
                    .item()
                ),

            "layer2_reject_rate_at_threshold":
                float(
                    (a2 < t2)
                    .float()
                    .mean()
                    .item()
                ),

            "layer3_reject_rate_raw":
                float(
                    (a3 < 0.5)
                    .float()
                    .mean()
                    .item()
                ),

            "layer3_reject_rate_at_threshold":
                float(
                    (a3 < t3)
                    .float()
                    .mean()
                    .item()
                ),

            "mean_reputation":
                float(
                    reputations_tensor
                    .mean()
                    .item()
                ),

            "min_reputation":
                float(
                    reputations_tensor
                    .min()
                    .item()
                ),

            "layer1_threshold":
                float(t1),

            "layer2_threshold":
                float(t2),

            "layer3_threshold":
                float(t3),
        }

        return (
            aggregated_parameters,
            metrics,
        )
