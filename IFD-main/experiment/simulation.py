"""
Simulation Launcher using flwr.simulation.start_simulation()
"""

from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, Subset

import flwr as fl
from data.loader import IEEEFraudDataset, load_ieee_cis_data
from data.partitioner import GeographicPartitioner
from experiment.client import IFDClient, FraudMLP
from orchestration.flower_strategy import CascadeRouter


def client_fn_factory(
    client_indices: Dict[int, Any],
    train_dataset: IEEEFraudDataset,
    test_dataset: IEEEFraudDataset,
    input_dim: int,
    num_adversaries: int = 0,
    attack_type: Optional[str] = None,
):
    """Factory creating client_fn closure for Flower simulation."""
    
    def client_fn(cid: str) -> fl.client.Client:
        client_idx = int(cid)
        indices = client_indices[client_idx]
        
        client_train_ds = Subset(train_dataset, indices)
        train_loader = DataLoader(client_train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        is_adv = client_idx < num_adversaries

        client = IFDClient(
            cid=cid,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            epochs=1,
            is_adversary=is_adv,
            attack_type=attack_type if is_adv else None,
        )
        return client.to_client()

    return client_fn


def run_simulation(
    num_clients: int = 5,
    num_rounds: int = 2,
    num_adversaries: int = 0,
    attack_type: Optional[str] = None,
    strategy: Optional[fl.server.strategy.Strategy] = None,
) -> fl.server.history.History:
    """
    Run Flower simulation for federated learning fraud detection.
    """
    train_ds, test_ds = load_ieee_cis_data(synthetic_fallback=True)
    input_dim = train_ds[0][0].shape[0]

    labels = train_ds.y.numpy()
    partitioner = GeographicPartitioner(num_clients=num_clients, seed=42)
    client_indices, _ = partitioner.partition(labels)

    if strategy is None:
        strategy = CascadeRouter()

    client_fn = client_fn_factory(
        client_indices=client_indices,
        train_dataset=train_ds,
        test_dataset=test_ds,
        input_dim=input_dim,
        num_adversaries=num_adversaries,
        attack_type=attack_type,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    return history
