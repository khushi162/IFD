"""
Unified Baseline Adapter wrapping all baselines B1–B9 into a common interface.
"""

from typing import List, Optional, Union
import torch

from baselines.b1_fedavg import fedavg
from baselines.b2_krum import krum
from baselines.b3_median import coordinate_median
from baselines.b4_trimmed_mean import trimmed_mean
from baselines.b5_bulyan import bulyan
from baselines.b6_fltrust import fltrust
from baselines.b7_foolsgold import foolsgold
from baselines.b8_dpfl import dp_fl
from baselines.b9_fldetector import FLDetector


class BaselineAdapter:
    """
    Unified adapter for baseline federated aggregation defenses B1–B9.
    
    Supported baseline names:
      - "b1_fedavg" / "fedavg"
      - "b2_krum" / "krum"
      - "b2_multikrum" / "multikrum"
      - "b3_median" / "median"
      - "b4_trimmed_mean" / "trimmed_mean"
      - "b5_bulyan" / "bulyan"
      - "b6_fltrust" / "fltrust"
      - "b7_foolsgold" / "foolsgold"
      - "b8_dpfl" / "dpfl"
      - "b9_fldetector" / "fldetector"
    """

    def __init__(self, name: str, **kwargs):
        self.name = name.lower()
        self.kwargs = kwargs
        if self.name in ["b9_fldetector", "fldetector"]:
            self.fldetector = FLDetector(**kwargs)

    def aggregate(
        self,
        gradients: List[torch.Tensor],
        client_ids: Optional[List[str]] = None,
        server_gradient: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate client gradients using chosen baseline.
        
        Returns:
            g_aggregated: 1D torch Tensor
        """
        if not gradients:
            raise ValueError("Gradients list cannot be empty.")

        n = self.name
        if n in ["b1_fedavg", "fedavg"]:
            return fedavg(gradients)
        elif n in ["b2_krum", "krum"]:
            return krum(gradients, multi_krum=False, **self.kwargs)
        elif n in ["b2_multikrum", "multikrum"]:
            return krum(gradients, multi_krum=True, **self.kwargs)
        elif n in ["b3_median", "median"]:
            return coordinate_median(gradients)
        elif n in ["b4_trimmed_mean", "trimmed_mean"]:
            return trimmed_mean(gradients, **self.kwargs)
        elif n in ["b5_bulyan", "bulyan"]:
            return bulyan(gradients, **self.kwargs)
        elif n in ["b6_fltrust", "fltrust"]:
            return fltrust(gradients, server_gradient=server_gradient)
        elif n in ["b7_foolsgold", "foolsgold"]:
            return foolsgold(gradients)
        elif n in ["b8_dpfl", "dpfl"]:
            return dp_fl(gradients, **self.kwargs)
        elif n in ["b9_fldetector", "fldetector"]:
            return self.fldetector.aggregate(gradients, client_ids=client_ids)
        else:
            raise ValueError(f"Unknown baseline name: '{self.name}'. Supported: B1–B9.")
