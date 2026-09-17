"""
Baselines module for IFD-Fintech.

Provides baseline aggregation algorithms B1–B9:
  B1: FedAvg
  B2: Krum / Multi-Krum
  B3: Coordinate Median
  B4: Trimmed Mean
  B5: Bulyan
  B6: FLTrust
  B7: FoolsGold
  B8: DP-FL
  B9: FLDetector
"""

from baselines.adapter import BaselineAdapter
from baselines.b1_fedavg import fedavg
from baselines.b2_krum import krum
from baselines.b3_median import coordinate_median
from baselines.b4_trimmed_mean import trimmed_mean
from baselines.b5_bulyan import bulyan
from baselines.b6_fltrust import fltrust
from baselines.b7_foolsgold import foolsgold
from baselines.b8_dpfl import dp_fl
from baselines.b9_fldetector import FLDetector

__all__ = [
    "BaselineAdapter",
    "fedavg",
    "krum",
    "coordinate_median",
    "trimmed_mean",
    "bulyan",
    "fltrust",
    "foolsgold",
    "dp_fl",
    "FLDetector",
]
