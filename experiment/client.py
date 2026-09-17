"""
Flower NumPyClient and PyTorch FraudMLP Model
"""

from typing import Dict, List, Optional, Tuple, Union, cast

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import flwr as fl
from flwr.common import Scalar


class FraudMLP(nn.Module):
    """PyTorch MLP model for credit card fraud detection."""

    def __init__(self, input_dim: int = 30, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """
    Extract the complete model state_dict in deterministic order.

    This includes both trainable parameters and persistent buffers such as
    BatchNorm running statistics and num_batches_tracked.
    """

    return [
        value.detach().cpu().numpy().copy()
        for _, value in net.state_dict().items()
    ]


def set_parameters(
    net: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """
    Load a complete model state_dict from Flower parameters.

    Validates both the number and shape of tensors before loading.
    """

    state = net.state_dict()
    keys = list(state.keys())

    if len(parameters) != len(keys):
        raise ValueError(
            "CLIENT PARAMETER COUNT MISMATCH\n"
            f"Model expects {len(keys)} tensors, "
            f"but Flower supplied {len(parameters)} tensors.\n"
            f"Model keys: {keys}"
        )

    converted = {}

    for idx, (key, reference) in enumerate(state.items()):
        incoming = np.asarray(parameters[idx])

        if incoming.shape != tuple(reference.shape):
            raise ValueError(
                "CLIENT PARAMETER SHAPE MISMATCH\n"
                f"Parameter index: {idx}\n"
                f"Parameter name:  {key}\n"
                f"Expected shape: {tuple(reference.shape)}\n"
                f"Received shape: {incoming.shape}\n"
                f"Expected dtype: {reference.dtype}\n"
                f"Received dtype: {incoming.dtype}"
            )

        tensor = torch.from_numpy(
            incoming.copy()
        ).to(
            dtype=reference.dtype
        )

        converted[key] = tensor

    net.load_state_dict(
        converted,
        strict=True,
    )


# Module-level counter tracking how many evaluate() calls client "0" has made.
# Used by the SAVE_PROBS_ROUND feature to identify the communication round.
# A list is used so the value can be mutated from inside evaluate().
_IFDClient__eval_round_counter = [0]


class IFDClient(fl.client.NumPyClient):
    """Flower client representing a bank node in federated learning."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int = 30,
        epochs: int = 1,
        lr: float = 1e-3,
        is_adversary: bool = False,
        attack_type: Optional[str] = None,
    ):
        self.cid = str(cid)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.net = FraudMLP(
            input_dim=input_dim
        ).to(self.device)

        self.epochs = epochs
        self.lr = lr

        self.is_adversary = is_adversary
        self.attack_type = attack_type

    def get_parameters(
        self,
        config: Dict[str, Scalar],
    ) -> List[np.ndarray]:

        return get_parameters(self.net)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[
        List[np.ndarray],
        int,
        Dict[str, Scalar],
    ]:

        set_parameters(
            self.net,
            parameters,
        )

        is_gpu = next(
            self.net.parameters()
        ).is_cuda

        if is_gpu:
            device_name = torch.cuda.get_device_name(0)
        else:
            device_name = "CPU"

        print(
            f"[Client {self.cid}] Training on {device_name}",
            flush=True,
        )

        optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
        )

        # Class-weighted loss: up-weight the minority fraud class so the
        # model is penalised more for missing a fraud than a false alarm.
        # pos_weight = (# negatives) / (# positives), clamped to [1, 100].
        _all_labels = self.train_loader.dataset.dataset.y[
            self.train_loader.dataset.indices
        ] if hasattr(self.train_loader.dataset, "indices") else (
            self.train_loader.dataset.y
        )
        _n_pos = float(_all_labels.sum())
        _n_neg = float(len(_all_labels) - _n_pos)
        _pos_weight = float(np.clip(_n_neg / max(_n_pos, 1.0), 1.0, 100.0))
        _weight_tensor = torch.tensor(
            [_pos_weight], dtype=torch.float32, device=self.device
        )
        criterion = nn.BCELoss(reduction="none")

        def _weighted_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            per_sample = criterion(preds, targets)
            weights = torch.where(targets == 1, _weight_tensor, torch.ones_like(targets))
            return (per_sample * weights).mean()

        self.net.train()

        for epoch in range(self.epochs):

            for X_batch, y_batch in self.train_loader:

                if len(X_batch) <= 1:
                    continue

                X_batch = X_batch.to(
                    self.device,
                    non_blocking=False,
                )

                y_batch = y_batch.to(
                    self.device,
                    non_blocking=False,
                )

                # Label flipping is applied only to the local training labels.
                if (
                    self.is_adversary
                    and self.attack_type == "label_flip"
                ):
                    y_batch = 1.0 - y_batch

                optimizer.zero_grad(
                    set_to_none=True
                )

                preds = self.net(X_batch)

                loss = _weighted_loss(
                    preds,
                    y_batch,
                )

                loss.backward()

                optimizer.step()

        updated_params = get_parameters(
            self.net
        )

        # ------------------------------------------------------------
        # Byzantine attacks
        # ------------------------------------------------------------

        if (
            self.is_adversary
            and self.attack_type is not None
        ):

            if self.attack_type == "sign_flip":

                updated_params = [
                    -p
                    if np.issubdtype(
                        p.dtype,
                        np.floating,
                    )
                    else p
                    for p in updated_params
                ]

            elif self.attack_type == "model_replace":

                scale = 10.0

                updated_params = [
                    p * scale
                    if np.issubdtype(
                        p.dtype,
                        np.floating,
                    )
                    else p
                    for p in updated_params
                ]

            elif self.attack_type == "gaussian_noise":

                floating = [
                    p
                    for p in updated_params
                    if np.issubdtype(
                        p.dtype,
                        np.floating,
                    )
                    and p.size > 0
                ]

                if floating:

                    norms = [
                        np.linalg.norm(p)
                        for p in floating
                    ]

                    sigma = (
                        0.1 * max(norms)
                        if norms
                        else 1e-4
                    )

                    noisy_params = []

                    for p in updated_params:

                        if np.issubdtype(
                            p.dtype,
                            np.floating,
                        ):

                            noise = np.random.normal(
                                0.0,
                                sigma,
                                p.shape,
                            ).astype(
                                p.dtype
                            )

                            noisy_params.append(
                                p + noise
                            )

                        else:
                            noisy_params.append(p)

                    updated_params = noisy_params

            elif self.attack_type == "label_flip":
                # Already applied during local training.
                pass

            else:
                raise ValueError(
                    f"Unknown attack type: {self.attack_type}"
                )

        dataset = cast(
            Dataset,
            self.train_loader.dataset,
        )

        num_samples = len(dataset)

        return (
            updated_params,
            num_samples,
            {},
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[
        float,
        int,
        Dict[str, Scalar],
    ]:

        set_parameters(
            self.net,
            parameters,
        )

        criterion = nn.BCELoss()

        self.net.eval()

        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_labels = []

        import sklearn.metrics as skm

        with torch.no_grad():

            for X_batch, y_batch in self.val_loader:

                X_batch = X_batch.to(
                    self.device
                )

                y_batch = y_batch.to(
                    self.device
                )

                preds = self.net(X_batch)

                loss = criterion(
                    preds,
                    y_batch,
                )

                total_loss += (
                    loss.item()
                    * len(y_batch)
                )

                total_samples += len(
                    y_batch
                )

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_labels.extend(
                    y_batch.cpu().numpy()
                )

        avg_loss = (
            total_loss
            / max(1, total_samples)
        )

        all_labels = np.asarray(
            all_labels
        )

        all_preds = np.asarray(
            all_preds
        )

        binary_preds = (
            all_preds >= 0.5
        ).astype(float)

        if total_samples > 0:

            accuracy = (
                skm.accuracy_score(
                    all_labels,
                    binary_preds,
                )
            )

        else:
            accuracy = 0.0

        try:

            auc = skm.roc_auc_score(
                all_labels,
                all_preds,
            )

        except ValueError:

            auc = 0.5

        precision = skm.precision_score(
            all_labels,
            binary_preds,
            zero_division=0,
        )

        recall = skm.recall_score(
            all_labels,
            binary_preds,
            zero_division=0,
        )

        f1 = skm.f1_score(
            all_labels,
            binary_preds,
            zero_division=0,
        )

        metrics = {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        # ------------------------------------------------------------------
        # Optional per-sample probability export (threshold sweep support)
        #
        # Set SAVE_PROBS_PATH to an .npz filepath to capture y_true and
        # y_score for the threshold sweep.  Only client "0" writes the
        # file so there are no race conditions in distributed evaluate.
        #
        # SAVE_PROBS_ROUND (optional): save only on this round number.
        # Flower passes config={} to evaluate(), so we track rounds with
        # a module-level counter instead of reading from config.
        # ------------------------------------------------------------------
        _probs_path = os.environ.get("SAVE_PROBS_PATH")
        if _probs_path and self.cid == "0":
            # Increment the per-process evaluate round counter.
            _IFDClient__eval_round_counter[0] += 1
            _target_round_str = os.environ.get("SAVE_PROBS_ROUND", "")
            _save_now = True
            if _target_round_str:
                try:
                    _save_now = (
                        _IFDClient__eval_round_counter[0]
                        == int(_target_round_str)
                    )
                except (ValueError, TypeError):
                    _save_now = True
            if _save_now:
                np.savez(_probs_path, y_true=all_labels, y_score=all_preds)

        return (
            float(avg_loss),
            total_samples,
            metrics,
        )