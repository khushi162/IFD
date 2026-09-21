"""
Adaptive Threshold Escalation Controller

Tracks per-layer rejection rates and dynamically adjusts acceptance thresholds
to defend against coordinated attacks while minimizing false positives during
normal operation.

Mathematical Specification:
---------------------------
Track rejection rate (fraction with a_i < 0.5) per layer per round:
    rejection_rate_layer_k = mean(a_k < 0.5)

Exponential moving average of rejection rate:
    EMA_reject_k = β · rejection_rate_k + (1-β) · EMA_reject_k^(t-1)

Escalation trigger:
    if EMA_reject_k > threshold_trigger:
        threshold_k = min(threshold_k + escalation_step, max_threshold)
    else:
        threshold_k = max(threshold_k - decay_step, base_threshold)

Default Hyperparameters (placeholders needing tuning):
------------------------------------------------------
- β = 0.2                   # EMA weight for rejection rate tracking
- threshold_trigger = 0.15  # Trigger escalation if >15% rejection rate
- escalation_step = 0.05    # Increase threshold by 0.05 per round under attack
- decay_step = 0.01         # Decrease threshold by 0.01 per round when calm
- base_threshold = 0.5      # Default acceptance threshold
- max_threshold = 0.8       # Cap on escalated threshold
"""

"""
Real bug found: a single round with zero scored clients silently and permanently disables escalation for that layer.

python
empty_scores = torch.tensor([])
tc.update({'layer1': empty_scores})
# → ema_reject['layer1'] = nan

tc.update({'layer1': torch.ones(50)})  # normal round afterward
# → ema_reject['layer1'] is STILL nan

rejection_mask.float().mean() on an empty tensor is NaN, and once ema_reject is NaN, self.beta * rate + (1-self.beta) * NaN is NaN forever — it never recovers. Worse, the escalation check if ema_reject > threshold_trigger is silently always False for NaN (no exception raised), so the controller doesn't crash — it just quietly stops escalating for that layer, permanently, for the rest of training, with zero indication anything went wrong. This is the kind of failure that's very easy to miss in experiments since nothing errors out.

This can genuinely happen in FL: a round where a layer receives no clients (e.g. all clients dropped by an earlier cascade stage, or a straggler round with 0 participants for that layer).

Fix: guard against empty/degenerate input before the EMA update — e.g. skip the update entirely (keep previous EMA) if scores.numel() == 0, or clamp/reset if NaN is ever detected. One line:

python
if scores.numel() == 0:
    updated_thresholds[layer_name] = self.current_threshold[layer_name]
    continue
"""

import torch
from typing import Dict


class ThresholdController:
    """
    Adaptive per-layer threshold controller with exponential moving average
    of rejection rates.
    
    Escalates thresholds when attack is detected (high rejection rate),
    decays back to baseline when system is calm.
    """
    
    def __init__(
        self,
        beta: float = 0.2,
        threshold_trigger: float = 0.15,
        escalation_step: float = 0.05,
        decay_step: float = 0.01,
        base_threshold: float = 0.5,
        max_threshold: float = 0.8,
    ):
        """
        Initialize threshold controller.
        
        Args:
            beta: EMA weight for rejection rate tracking (placeholder)
            threshold_trigger: EMA rejection rate that triggers escalation (placeholder)
            escalation_step: Amount to increase threshold per round (placeholder)
            decay_step: Amount to decrease threshold per round (placeholder)
            base_threshold: Default/minimum acceptance threshold
            max_threshold: Maximum allowed threshold
        """
        self.beta = beta
        self.threshold_trigger = threshold_trigger
        self.escalation_step = escalation_step
        self.decay_step = decay_step
        self.base_threshold = base_threshold
        self.max_threshold = max_threshold
        
        # Per-layer state: EMA of rejection rate and current threshold
        self.ema_reject: Dict[str, float] = {}
        self.current_threshold: Dict[str, float] = {}
    
    def update(self, layer_scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update thresholds based on layer-wise acceptance scores.
        
        Args:
            layer_scores: Dictionary mapping layer names to acceptance score tensors
                         e.g., {"layer1": a1_scores, "layer2": a2_scores, ...}
                         Each tensor has shape [num_clients] with scores in [0, 1]
        
        Returns:
            Dictionary mapping layer names to updated thresholds
        """
        updated_thresholds = {}
        
        for layer_name, scores in layer_scores.items():
            # Initialize layer state on first encounter
            if layer_name not in self.ema_reject:
                self.ema_reject[layer_name] = 0.0
                self.current_threshold[layer_name] = self.base_threshold
            
            if scores.numel() == 0:
                updated_thresholds[layer_name] = self.current_threshold[layer_name]
                continue
            
            # Compute rejection rate: fraction of clients with score < 0.5
            # Using 0.5 as the rejection boundary per spec
            rejection_mask = scores < 0.5
            rejection_rate = rejection_mask.float().mean().item()
            
            # Update EMA of rejection rate
            # EMA_reject_k = β · rejection_rate_k + (1-β) · EMA_reject_k^(t-1)
            prev_ema = self.ema_reject[layer_name]
            self.ema_reject[layer_name] = (
                self.beta * rejection_rate + (1 - self.beta) * prev_ema
            )
            
            # Escalate or decay threshold based on EMA rejection rate
            current_thresh = self.current_threshold[layer_name]
            
            if self.ema_reject[layer_name] > self.threshold_trigger:
                # Attack detected: escalate threshold
                new_thresh = min(
                    current_thresh + self.escalation_step,
                    self.max_threshold
                )
            else:
                # System calm: decay back to baseline
                new_thresh = max(
                    current_thresh - self.decay_step,
                    self.base_threshold
                )
            
            self.current_threshold[layer_name] = new_thresh
            updated_thresholds[layer_name] = new_thresh
        
        return updated_thresholds
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds for all tracked layers."""
        return self.current_threshold.copy()
    
    def get_ema_reject_rates(self) -> Dict[str, float]:
        """Get current EMA rejection rates for all tracked layers."""
        return self.ema_reject.copy()
    
    def reset(self):
        """Reset all layer state to initial values."""
        self.ema_reject.clear()
        self.current_threshold.clear()
