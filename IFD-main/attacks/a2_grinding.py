"""
A2: Temporal Grinding Attack

Adversary stays stealthy during calibration rounds, slowly rotating their gradient
trajectory toward target payload direction to poison the historical EMA trajectory
without triggering sudden cosine consistency drops in Layer 3.
"""

import math
import torch


class TemporalGrinding:
    """
    Temporal Grinding Attack.
    
    Gradually shifts gradient direction during warmup rounds to poison EMA trajectory
    before injecting full target payload.
    """

    def __init__(self, drift_angle_deg: float = 5.0, warmup_rounds: int = 15):
        """
        Args:
            drift_angle_deg: Maximum rotation angle (in degrees) per round during warmup
            warmup_rounds: Number of calibration rounds to grind trajectory
        """
        self.drift_angle_deg = drift_angle_deg
        self.warmup_rounds = warmup_rounds
        self.drift_angle_rad = math.radians(drift_angle_deg)

    def generate_gradient(
        self,
        round_num: int,
        base_gradient: torch.Tensor,
        target_direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate attack gradient for current round.
        
        Args:
            round_num: Current federated learning round index (1-indexed)
            base_gradient: Honest gradient vector for current round
            target_direction: Malicious target direction vector
            
        Returns:
            g_adv: 1D torch Tensor gradient vector
        """
        g_base = base_gradient.flatten().float()
        v_target = target_direction.flatten().float()
        norm_base = torch.norm(g_base) + 1e-8
        
        unit_base = g_base / norm_base
        unit_target = v_target / (torch.norm(v_target) + 1e-8)

        if round_num >= self.warmup_rounds:
            # Post warmup: inject full target payload direction with base gradient magnitude
            return unit_target * norm_base

        # Warmup phase: interpolate angle toward target_direction by (round_num * drift_angle)
        current_max_angle = round_num * self.drift_angle_rad

        # Compute angle between base and target
        cos_theta = torch.clamp(torch.sum(unit_base * unit_target), -1.0, 1.0)
        total_angle = torch.acos(cos_theta).item()

        if total_angle < 1e-6:
            return unit_target * norm_base

        # Interpolation factor alpha_angle
        actual_angle = min(current_max_angle, total_angle)
        t = actual_angle / total_angle

        # Spherical linear interpolation (SLERP)
        sin_total = math.sin(total_angle)
        w_base = math.sin((1 - t) * total_angle) / sin_total
        w_target = math.sin(t * total_angle) / sin_total

        unit_blended = w_base * unit_base + w_target * unit_target
        unit_blended = unit_blended / (torch.norm(unit_blended) + 1e-8)

        return unit_blended * norm_base
