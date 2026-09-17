"""
Attacks module for IFD-Fintech.

Includes locked scope attacks:
- A1: Oracle White-Box PGD
- A2: Temporal Grinding
- A3: Spectral Matching
"""

from attacks.a1_oracle_whitebox import OracleWhiteBoxPGD
from attacks.a2_grinding import TemporalGrinding
from attacks.a3_spectral_matching import SpectralMatching

__all__ = [
    "OracleWhiteBoxPGD",
    "TemporalGrinding",
    "SpectralMatching",
]
