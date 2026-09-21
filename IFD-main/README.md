# IFD-Fintech: Robust Federated Learning for Credit Card Identity Fraud Detection

**A temporally-aware gated cascade defense framework against adaptive poisoning attacks in cross-silo financial federated learning.**

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-Flower_PyTorch-green.svg)](https://flower.ai/)
[![Target](https://img.shields.io/badge/target-IEEE_TIFS-orange.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10206)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What This Is About](#what-this-is-about)
- [Methodology and Solutions](#methodology-and-solutions)
  - [Layer 1: Norm/Cosine Filtering](#layer-1-normcosine-filtering)
  - [Layer 2: Spectral Anomaly Detection](#layer-2-spectral-anomaly-detection)
  - [Layer 3: Temporal Consistency Scoring](#layer-3-temporal-consistency-scoring)
  - [Adaptive Threshold Escalation](#adaptive-threshold-escalation)
  - [EWMA Reputation and Aggregation](#ewma-reputation-and-aggregation)
  - [Formal Guarantees](#formal-guarantees)
- [Attack Models](#attack-models)
- [Baselines](#baselines)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Experiments](#running-experiments)
- [Codebase Architecture](#codebase-architecture)
- [Paper and Results](#paper-and-results)
- [Citation](#citation)

---

## Problem Statement

Cross-silo federated learning (FL) enables financial institutions and banks to collaboratively train fraud detection models without sharing raw transaction data. However, financially motivated adversaries who compromise a client bank can poison the global model to evade detection of their own fraud patterns.

**The core vulnerability:** Existing Byzantine-robust aggregation methods—Krum, trimmed mean, Bulyan, FLTrust, FoolsGold, FLDetector, and others—are primarily **stateless**: each training round is evaluated independently, making them blind to **temporally-adaptive attackers** who alternate between malicious and benign behavior across rounds. A malicious client that behaves honestly for three rounds, performs a targeted attack on the fourth, and pauses on the fifth can evade per-round static defenses.

Additionally, the financial FL domain presents unique constraints:
- **Non-stationary concept drift** — fraud patterns evolve across retraining cycles
- **Multi-round attack scheduling** — adversaries can spread a single attack across many rounds
- **Regulatory requirements** — auditability, explainability, and AML/CFT override obligations
- **High cost of false positives** — a wrongly-flagged honest bank damages consortium trust

---

## What This Is About

This project presents a **gated cascade defense framework** comprising three complementary detection layers connected by an **adaptive threshold escalation policy**, designed specifically for the financial FL setting. The framework is evaluated against implemented attack models (A1–A3) and baseline defenses (FedAvg, Bulyan, FLDetector, DP-FL), alongside ablation configurations isolating each component's contribution.

**Key Design Objectives:**
- Full defense achieves robust protection against coordinated and adaptive poisoning (ASR < 0.25)
- Cascaded structure resolves ~80% of benign updates at the lightweight $O(d)$ layer
- Adaptive thresholds tighten during attack phases ($\eta_{\text{attack}}$) and relax during steady states ($\eta_{\text{relax}}$)
- Reputation-weighted aggregation prevents permanent exclusion of transiently anomalous honest participants

## Caution
This paper is an ongoing research project and is under active development. Specifications, bounds, and implementation details reflect the current codebase state.

---

## Methodology and Solutions

The defense is organized as a **gated cascade** — most updates are resolved by the cheapest layer (Layer 1, $O(d)$ per update), and only ambiguous cases escalate to deeper analysis.

```
Incoming Gradient g_i
        │
        ▼
┌─────────────────────────────────┐
│ Layer 1: Norm/Cosine Filter     │── Confidence ≥ τ₁ ──► Accept / Reject (Score a₁)
└─────────────────────────────────┘
        │ Confidence < τ₁ (Escalate)
        ▼
┌─────────────────────────────────┐
│ Layer 2: Spectral LOO Detector  │── Confidence ≥ τ₂ ──► Accept / Reject (Score a₂)
└─────────────────────────────────┘
        │ Confidence < τ₂ (Escalate)
        ▼
┌─────────────────────────────────┐
│ Layer 3: Temporal Consistency   │─────────────────────► Final Score a₃
└─────────────────────────────────┘
```

---

### Layer 1: Norm/Cosine Filtering

**Purpose:** Rapidly evaluate updates based on gradient magnitude and alignment with the consensus direction.

**Mechanism:**
1. **Warm-up Statistics:** During initial clean rounds, running statistics of honest update norms ($\mu_{\text{norm}}, \sigma_{\text{norm}}$) and cosine similarities ($\mu_{\text{cos}}, \sigma_{\text{cos}}$) are estimated.
2. **Norm Anomaly Score:** Evaluated via a two-sided z-score:
   $$z_{\text{norm}} = \frac{|\|g_i\|_2 - \mu_{\text{norm}}|}{\sigma_{\text{norm}}}$$
   $$s_{\text{norm}} = 1 - \sigma(z_{\text{norm}} - 3.0)$$
3. **Cosine Similarity Score:** Using the coordinate-wise median reference vector $\text{ref} = \text{median}_j(g_j)$:
   $$\cos(g_i, \text{ref}) = \frac{\langle g_i, \text{ref} \rangle}{\|g_i\|_2 \|\text{ref}\|_2}$$
   $$z_{\text{cos}} = \frac{\mu_{\text{cos}} - \cos(g_i, \text{ref})}{\sigma_{\text{cos}}}$$
   $$s_{\text{cos}} = 1 - \sigma(z_{\text{cos}} - 3.0)$$
4. **Combined Score & Confidence:**
   $$a_1 = \min(s_{\text{norm}}, s_{\text{cos}}), \quad c_1 = 2 \cdot |a_1 - 0.5|$$

```python
from layers import Layer1NormCosine

filter_l1 = Layer1NormCosine()
anomaly_score, confidence = filter_l1.score(gradients)
```

---

### Layer 2: Spectral Anomaly Detection (Leave-One-Out Robust PCA)

**Purpose:** Detect **coordinated multi-client attacks** (e.g., $m \ge 3$ colluding clients) that evade per-client norm/cosine filtering by identifying anomalous subspace projections.

**Mechanism:**
1. **Leave-One-Out (LOO) Peer Subspace:** To prevent malicious clients from contaminating the subspace against which they are tested, the principal subspace is computed solely from the peer set $\mathcal{P}_{-i} = \{g_j \mid j \neq i\}$:
   $$R_{-i} = G_{-i} - \mathbf{1} \mu_{-i}^\top, \quad R_{-i} = U \Sigma V^\top$$
   The number of retained principal components $k$ is selected dynamically to explain $\gamma = 95\%$ of peer variance ($k = \max(1, \min(k_{\text{var}}, N-2, d))$).
2. **Reconstruction Residual:**
   $$e_i = \| (g_i - \mu_{-i}) - V_k V_k^\top (g_i - \mu_{-i}) \|_2$$
3. **MAD-Normalized Robust Z-Score:** Comparing target residual $e_i$ against peer residuals $\{e_j\}_{j \in \mathcal{P}_{-i}}$:
   $$\text{med}_{\text{res}} = \text{median}(\{e_j\}), \quad \text{MAD} = \text{median}(|\{e_j\} - \text{med}_{\text{res}}|)$$
   $$z_i = \frac{e_i - \text{med}_{\text{res}}}{1.4826 \cdot \text{MAD} + 10^{-8}}$$
   $$a_2 = 1.0 - \sigma(z_i - 3.0)$$
4. **Confidence:**
   $$c_2 = 2 \cdot |a_2 - 0.5| \cdot \min\left(1.0, \frac{N-1}{\max(k+1, 2)}\right)$$

```python
from layers import Layer2Spectral

filter_l2 = Layer2Spectral(variance_ratio=0.95)
anomaly_score, confidence = filter_l2.score(gradients)
```

---

### Layer 3: Temporal Consistency Scoring

**Purpose:** Detect **slow, adaptive poisoning** (gradual perturbation drift across consecutive rounds) by evaluating consistency against each client's historical trajectory.

**Mechanism:**
1. **Historical Update EMA:** Maintains an exponential moving average $\bar{g}_i^{(t)}$ of previous updates for each client:
   $$\bar{g}_i^{(t)} = (1 - \alpha) \bar{g}_i^{(t-1)} + \alpha g_i^{(t)}$$
2. **Cosine Consistency Score:**
   $$\text{consistency} = \frac{\langle g_i, \bar{g}_i^{(t-1)} \rangle}{\|g_i\|_2 \|\bar{g}_i^{(t-1)}\|_2}$$
   $$a_3 = \frac{\text{consistency} + 1.0}{2.0} \in [0, 1]$$
3. **Maturity-Based Confidence:**
   $$c_3 = \min\left(\frac{\text{rounds\_seen}_i}{\text{maturity\_rounds}}, 1.0\right)$$

```python
from layers import Layer3Temporal

filter_l3 = Layer3Temporal(alpha=0.1, maturity_rounds=20)
anomaly_score, confidence = filter_l3.score(gradients, client_ids)
```

---

### Adaptive Threshold Escalation

The orchestrator routes updates through the cascade using dynamic confidence gates $\tau_1^{(t)}$ and $\tau_2^{(t)}$:

1. Evaluate Layer 1: If $c_1(g_i) \ge \tau_1^{(t)}$, decide using $a_1(g_i)$.
2. Otherwise escalate to Layer 2: If $c_2(g_i) \ge \tau_2^{(t)}$, decide using $a_2(g_i)$.
3. Otherwise escalate to Layer 3: Decide using $a_3(g_i)$.

**Dynamic Adaptation:**
The estimated round attack rate $\rho^{(t)}$ (fraction of updates with $a < \theta_{\text{reject}}$) adjusts the thresholds:
$$\tau_k^{(t+1)} = \text{clamp}\left(\tau_k^{(t)} + \eta_{\text{attack}}(\rho^{(t)} - \rho_0) \cdot \mathbf{1}_{\{\rho^{(t)} > \rho_0\}} - \eta_{\text{relax}} \cdot \mathbf{1}_{\{\rho^{(t)} \le \rho_0\}}, \, \tau_{k,\min}, \, \tau_{k,\max}\right)$$

---

### EWMA Reputation and Aggregation

Client reputations $R_i^{(t)} \in [0, 1]$ track long-term historical reliability:
$$R_i^{(t+1)} = (1 - \alpha) R_i^{(t)} + \alpha a_i^{(t)} + \gamma (R_{\text{SS}} - R_i^{(t)})$$
where $R_{\text{SS}} = 0.85$ serves as the steady-state anchor, preventing permanent lockout of honest participants.

**Reputation-Weighted Aggregation:**
Updates are aggregated according to their reputation weight $w_i$:
$$w_i = \begin{cases} 
R_i & \text{if } a_i \ge \theta_{\text{accept}} \\
R_i \cdot \text{suspicious\_weight} & \text{if } \theta_{\text{reject}} \le a_i < \theta_{\text{accept}} \\
0 & \text{if } a_i < \theta_{\text{reject}}
\end{cases}$$
$$g_{\text{global}} = \frac{\sum_i w_i g_i}{\sum_i w_i}$$

```python
from orchestration import CascadeRouter
from layers import Layer1NormCosine, Layer2Spectral, Layer3Temporal

router = CascadeRouter(
    layer1=Layer1NormCosine(),
    layer2=Layer2Spectral(),
    layer3=Layer3Temporal(),
)

# In Flower, you pass this strategy to the server:
# flwr.server.start_simulation(..., strategy=router)
```

---

### Formal Guarantees

- **Cascade False Positive Rate:**
  $$\text{FPR}_{\text{cascade}} = \text{FPR}_1 + e_1 \cdot [\text{FPR}_2 + e_2 \cdot \text{FPR}_3] \le \text{FPR}_1 + \text{FPR}_2 + \text{FPR}_3$$
  where $e_k$ represents the escalation rate past layer $k$.
- **Spectral Stability:** Bounded under honest gradient covariance drift via the Davis-Kahan $\sin\Theta$ theorem.
- **Reputation Recovery:** The reputation recurrence guarantees $\liminf R_i \ge 0.1417$ under persistent honest behavior, enabling recovery from transient anomalies within ~15 rounds.

---

## Attack Models

| ID | Attack | Description | Primary Layer Targeted |
|:---|:-------|:------------|:-----------------------|
| **A1** | Oracle White-Box PGD | Gradient optimization directly targeting decision boundaries | Layer 1 (Norm/Cosine) |
| **A2** | Temporal Grinding | Multi-phase drift (Burn-in $\to$ Subliminal $\to$ Active $\to$ Cooldown) | Layer 3 (Temporal) |
| **A3** | Spectral Matching | Coordinated perturbation aligned to dominant peer eigenvectors | Layer 2 (Spectral) |
| **A4** | Temporal Collusion | Combined multi-client coordination and temporal scheduling | Full Cascade |
| **A5** | Synthetic Transaction Injection | Local dataset poisoning generating subtle distribution shifts | Layer 3 |
| **A6** | Cascade-Aware Envelope | Adversary with defense knowledge operating within perturbation limits | Operational Envelope |

---

## Baselines

| ID | Baseline | Type | Description |
|:---|:---------|:-----|:------------|
| **B1** | FedAvg | Unprotected | Standard coordinate mean aggregation |
| **B2** | Krum / Multi-Krum | Geometric | Distance-based outlier rejection |
| **B3** | Coordinate Median | Statistical | Component-wise median |
| **B4** | Trimmed Mean | Statistical | Component-wise extreme value trimming |
| **B5** | Bulyan | Hybrid | Combined Krum selection with trimmed averaging |
| **B6** | FLTrust | Reference-based | Server-held validation gradient bootstrapping |
| **B7** | FoolsGold | Similarity | Multi-round cosine divergence penalization |
| **B8** | DP-FL | Privacy-Preserving | Norm clipping with calibrated Gaussian noise |
| **B9** | FLDetector | History-based | Cauchy-loss gradient prediction tracking |

---

## Getting Started

### Prerequisites

You can run this project locally via Python, or completely containerized via Docker (Recommended for Windows / GPU passthrough).

- **Docker Method:** Docker Desktop (with WSL2 backend for Windows GPU support)
- **Local Method:** Python 3.10+ (Tested up to Python 3.14) and `uv` package manager.

### Method 1: Docker (Recommended)

The easiest way to run the simulation with full GPU support (especially on Windows) is using Docker Compose.

1. Download the [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection/data) from Kaggle.
2. Place `train_transaction.csv` and `train_identity.csv` in `./data/raw/`.
3. Build the container:
```bash
docker compose build
```
4. Run the full simulation (results will be saved to `./results/`):
```bash
docker compose run --rm train
```
*Tip: You can pass any standard `train.py` arguments to the Docker run command (e.g., `docker compose run --rm train python train.py --num-adversaries 3 --attack-type sign_flip`).*

### Method 2: Local Installation (uv)

```bash
# Clone the repository
git clone <repo-url>
cd IFD-PART2

# Install dependencies and sync the environment
uv sync

# Run the test suite (35 verification tests across layers and baselines)
uv run pytest -q

# Run the full simulation
uv run python train.py --num-clients 10 --num-rounds 50
```

---

## Codebase Architecture

```
IFD-PART2/
├── train.py                     # Unified CLI entrypoint for simulations
├── Dockerfile & docker-compose  # GPU-enabled containerization
│
├── layers/                      # Detection Layer Implementations
│   ├── __init__.py              # Layer exports
│   ├── layer1_norm_cosine.py    # Layer 1: Norm & cosine z-score filter
│   ├── layer2_spectral.py       # Layer 2: Leave-one-out SVD / MAD detector
│   └── layer3_temporal.py       # Layer 3: Multi-round EMA consistency
│
├── orchestration/               # Cascade & Defense Management
│   ├── __init__.py              # CascadeRouter orchestrator
│   ├── threshold_controller.py  # Adaptive threshold (tau_1, tau_2) policy
│   ├── reputation.py            # ReputationManager tracking R_i state
│   └── flower_strategy.py       # Flower Strategy integration
│
├── attacks/                     # Adversarial Attack Models
│   ├── __init__.py              # Attack registry & base class
│   ├── a1_oracle_whitebox.py    # White-box evasion attack
│   ├── a2_grinding.py           # 4-phase temporal grinding attack
│   └── a3_spectral_matching.py  # Coordinated spectral evasion attack
│
├── baselines/                   # Comparison Defenses
│   ├── bulyan.py                # Bulyan implementation
│   ├── dpfl.py                  # DP-FL baseline
│   ├── fldetector.py            # FLDetector baseline
│   └── adapter.py               # Baseline-to-Cascade adapter wrapper
│
├── data/                        # Dataset Loaders & Splitters
│   ├── __init__.py              # Non-IID Dirichlet & geographic partitioners
│   ├── dataset.py               # PyTorch FraudDataset wrapper
│   └── loader.py                # IEEE-CIS and synthetic data loader
│
└── experiment/                  # Evaluation & Simulation Tools
    ├── ablation.py              # Ablation suite configurations
    ├── client.py                # Flower client wrapper
    ├── metrics.py               # Fraud & defense evaluation metrics
    └── simulation.py            # Full FL simulation runner
```
---

## Citation

If you find this codebase or research methodology useful, please consider citing:

```bibtex
@article{ifd_fintech2026,
  title={Temporally-Aware Gated Cascade Defense Against Adaptive Poisoning in Cross-Silo Financial Federated Learning},
  author={IFD-Fintech Research Team},
  journal={Targeted for IEEE Transactions on Information Forensics and Security (TIFS)},
  year={2026}
}
```