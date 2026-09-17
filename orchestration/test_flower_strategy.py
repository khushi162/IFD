"""
Numerical Verification for Chunk 5: CascadeRouter Flower Strategy Adapter
"""

from unittest.mock import MagicMock
import numpy as np
import torch

from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from orchestration.flower_strategy import CascadeRouter


def create_mock_client(cid: str) -> MagicMock:
    client = MagicMock(spec=ClientProxy)
    client.cid = cid
    return client


def test_cascade_router_execution():
    print("\n--- Test: CascadeRouter Aggregate Fit ---")
    strategy = CascadeRouter()

    # Create 5 mock client fit results across 3 rounds
    weights = [np.random.randn(10, 5).astype(np.float32), np.random.randn(5).astype(np.float32)]
    fit_res = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(weights),
        num_examples=100,
        metrics={},
    )

    results = [(create_mock_client(f"client_{i}"), fit_res) for i in range(5)]

    for r in range(1, 4):
        parameters, metrics = strategy.aggregate_fit(server_round=r, results=results, failures=[])
        assert "layer1_reject_rate_raw" in metrics, "Missing layer1_reject_rate_raw in metrics"
        assert "layer1_reject_rate_at_threshold" in metrics, "Missing layer1_reject_rate_at_threshold in metrics"
        assert "layer2_reject_rate_raw" in metrics, "Missing layer2_reject_rate_raw in metrics"
        assert "layer2_reject_rate_at_threshold" in metrics, "Missing layer2_reject_rate_at_threshold in metrics"
        assert "layer3_reject_rate_raw" in metrics, "Missing layer3_reject_rate_raw in metrics"
        assert "layer3_reject_rate_at_threshold" in metrics, "Missing layer3_reject_rate_at_threshold in metrics"
        print(f"Round {r} Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        # Verify structure of aggregated parameters
        agg_ndarrays = parameters_to_ndarrays(parameters)
        assert len(agg_ndarrays) == len(weights), "Aggregated weights layer count mismatch"
        assert agg_ndarrays[0].shape == weights[0].shape, "Aggregated shape mismatch"

    print("✓ CascadeRouter aggregate_fit test PASSED")


def test_cascade_router_empty_results():
    print("\n--- Test: CascadeRouter Empty Results ---")
    strategy = CascadeRouter()
    parameters, metrics = strategy.aggregate_fit(server_round=1, results=[], failures=[])
    assert parameters is None
    assert metrics == {}
    print("✓ CascadeRouter empty results test PASSED")


if __name__ == "__main__":
    test_cascade_router_execution()
    test_cascade_router_empty_results()
    print("\n==========================================")
    print("ALL CHUNK 5 FLOWER STRATEGY TESTS PASSED ✓")
    print("==========================================")
