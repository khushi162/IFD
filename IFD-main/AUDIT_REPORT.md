# Comprehensive Scientific, Mathematical, and Code Audit Report

**Target Codebase:** IFD-Fintech / IFD-PART2 (`D:\sarthak\Fraud-Detection-main`)  
**Target Publication Venue:** IEEE Transactions on Information Forensics and Security (TIFS)  
**Evaluated Framework:** TRAP (Temporal Reputation-Aware Poisoning Defense / CascadeRouter)  
**Auditor:** Senior AI/ML Research Scientist, Mathematical Reviewer, and Software Auditor  

---

## 1. Executive Summary

This audit evaluated the complete codebase, data pipeline, mathematical formulations, federated learning orchestration, baseline implementations, adversarial attacks, and paper claims for the IFD-Fintech research project.

### Overall State of the Codebase
The repository contains serious mathematical errors, fundamental data leakage, architectural misrepresentations, and paper–code discrepancies. While the simulation runs to completion and produces formatted tables and figures, the underlying scientific conclusions are **fundamentally compromised**.

### Number of Issues by Severity

| Severity Level | Issue Count |
| :--- | :---: |
| **CRITICAL** | 5 |
| **HIGH** | 5 |
| **MEDIUM** | 5 |
| **LOW** | 3 |
| **Total Issues** | **18** |

### Most Consequential Findings
1. **Mathematical Impossibility of Layer 1 Rejection:** Under the experimental setting of $N = 10$ clients, the maximum possible sample z-score is bounded by $(N-1)/\sqrt{N} \approx 2.846$. With a calibrated threshold of $Z = 3.3$, the anomaly score $a_1$ is mathematically guaranteed to satisfy $a_1 \ge 0.6116 > 0.5$ for all possible inputs. Layer 1 cannot reject any client update under any circumstances.
2. **Direct Target and Temporal Leakage:** Feature 3 in `data/loader.py` target-encodes `uid_fraud_mean` using the ground-truth labels `y` of the **entire dataset before the 80/20 train/test split**. Future fraud labels from the test set are directly encoded into training features.
3. **Flawed Aggregation and Attack Modeling:** The code transmits and aggregates **absolute neural network weights** $w_i$ rather than weight updates $\Delta w_i$ or gradients $g_i$. The attacks in `experiment/client.py` negate or scale the entire model parameter tensors ($w_i^* = -w_i$), rather than gradient deltas ($\Delta w_i^* = -\Delta w_i$).
4. **Fictitious "Gated Cascade":** The gated cascade routing architecture (escalating updates based on confidence thresholds $\tau_1, \tau_2$) is not implemented. `orchestration/flower_strategy.py` unconditionally evaluates all three layers for all clients, ignores confidence scores entirely, and takes a static coordinate minimum.
5. **Paper Contradicts Its Own Data:** The paper asserts in Section VI-D that full TRAP is "Pareto-optimal" across all conditions. In the paper's own Table III, removing Layer 1 or Layer 2 achieves higher AUC, F1, and Recall across multiple attack scenarios.

---

## 2. Repository Understanding

### Project Architecture and Workflow
The repository implements a cross-silo Federated Learning simulation designed for financial credit card fraud detection on the IEEE-CIS dataset using Flower (`flwr`) and PyTorch.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 Data Ingestion Pipeline                 │
                  │   data/raw/train_transaction.csv + train_identity.csv   │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │ Preprocessing & Target Leakage (data/loader.py)         │
                  │ - Computes uid_fraud_mean on full dataset               │
                  │ - Standardizes features before 80/20 train/test split   │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │ Partitioning (data/partitioner.py)                      │
                  │ - GeographicPartitioner wraps DirichletPartitioner(α=0.5)│
                  │ - Distributes indices across N=10 virtual clients       │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                  ┌────────────────────────────┴────────────────────────────┐
                  ▼                                                         ▼
    ┌───────────────────────────┐                             ┌───────────────────────────┐
    │  Client i (experiment/    │                             │  Client j (experiment/    │
    │  client.py)               │                             │  client.py)               │
    │  - FraudMLP training      │                             │  - FraudMLP training      │
    │  - Local BCELoss          │                             │  - Local BCELoss          │
    │  - Sends absolute weights │                             │  - Sends absolute weights │
    └─────────────┬─────────────┘                             └─────────────┬─────────────┘
                  │                                                         │
                  └────────────────────────────┬────────────────────────────┘
                                               │ Parameters (w_1, ..., w_N)
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │ Aggregation Server (orchestration/flower_strategy.py)   │
                  │ 1. Flatten absolute weights across all layers           │
                  │ 2. Compute Layer 1: Norm/Cosine Filter (layers/layer1)   │
                  │ 3. Compute Layer 2: LOO Spectral SVD (layers/layer2)    │
                  │ 4. Compute Layer 3: Dual CUSUM Temporal (layers/layer3) │
                  │ 5. Update ThresholdController & ReputationTracker       │
                  │ 6. Weighted aggregation: w_global = Σ R_i w_i / Σ R_i   │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │ Evaluation Pipeline (client.py evaluate())              │
                  │ - Every client evaluates global model on full test set  │
                  │ - Metrics: ROC-AUC, F1 (threshold 0.5), Precision, Rec  │
                  │ - Output saved to results/Sep3/*.json                   │
                  └─────────────────────────────────────────────────────────┘
```

---

## 3. Critical Findings

### [CRIT-01] Mathematical Impossibility of Layer 1 Rejection for $N=10$ Clients
- **Severity:** CRITICAL
- **Category:** Mathematical Error / Federated Learning Error
- **File:** `layers/layer1_norm_cosine.py` (lines 37, 142-154), `orchestration/flower_strategy.py` (lines 425-453), `orchestration/threshold_controller.py` (lines 125, 148)
- **Evidence:**
  In `layers/layer1_norm_cosine.py`:
  ```python
  Z_THRESH_L1: float = 3.3
  ...
  z_norm = torch.abs(norms - mu_norm) / sigma_norm
  z_cos = (mu_cos - cosines) / sigma_cos
  s_norm = 1.0 - torch.sigmoid(z_norm - Z_THRESH_L1)
  s_cos = 1.0 - torch.sigmoid(z_cos - Z_THRESH_L1)
  a1 = torch.minimum(s_norm, s_cos)
  ```
- **Mathematical Reasoning:**
  For any sample $x_1, \dots, x_N \in \mathbb{R}$, Thompson's Tau / Chauvenet's theorem proves that the maximum possible sample z-score is strictly bounded:
  $$z_{\max} \le \frac{N-1}{\sqrt{N}}$$
  For $N = 10$:
  $$z_{\max} \le \frac{10 - 1}{\sqrt{10}} = \frac{9}{\sqrt{10}} \approx 2.84605$$
  Because $z \le 2.84605$ is a strict algebraic upper bound, and $Z_{\text{thresh}} = 3.3$:
  $$z - 3.3 \le 2.84605 - 3.3 = -0.45395$$
  $$\sigma(z - 3.3) \le \frac{1}{1 + e^{0.45395}} \approx 0.38842$$
  $$s = 1 - \sigma(z - 3.3) \ge 1 - 0.38842 = 0.61158$$
  $$a_1 = \min(s_{\text{norm}}, s_{\text{cos}}) \ge 0.61158 \quad \forall \text{ inputs!}$$
  In `orchestration/threshold_controller.py` (line 125), rejection is defined as:
  ```python
  rejection_mask = scores < 0.5
  ```
  Because $a_1 \ge 0.61158 > 0.5$ for all inputs, `rejection_mask` is identically False for every client. The rejection rate for Layer 1 is permanently 0.0%, and `hard_reject = (a1 < t1)` is never satisfied.
- **Consequence:** Layer 1 is mathematically inert. It cannot detect or reject any client update in an $N=10$ client federation, regardless of how extreme the update is (even with $10,000\times$ norm scaling).
- **Verification Method:**
  Pass 10 vectors where 9 are unit vectors and 1 has norm $10^6$. The resulting $a_1$ for the outlier is $0.6116$, failing to trigger rejection at threshold 0.5.

---

### [CRIT-02] Catastrophic Target and Temporal Leakage in Preprocessing
- **Severity:** CRITICAL
- **Category:** Data and Experimental Validity
- **File:** `data/loader.py` (lines 111-166)
- **Evidence:**
  ```python
  # Feature 3: UID group aggregations.
  # Computed on the FULL dataset before the 80/20 split so
  # test rows receive correct group statistics.
  uid_cols_present = [c for c in ["card1", "addr1"] if c in X_df.columns]
  if uid_cols_present:
      ...
      y_series = pd.Series(y, index=X_df.index)
      X_df["uid_fraud_mean"] = y_series.groupby(by=uid_series).transform("mean").values
  ...
  # Train/test split (80/20)
  split_idx = int(0.8 * len(X_scaled))
  train_ds = IEEEFraudDataset(X_scaled[:split_idx], y[:split_idx])
  test_ds  = IEEEFraudDataset(X_scaled[split_idx:], y[split_idx:])
  ```
- **Explanation:**
  1. The target column `isFraud` (`y`) from the test set is directly used in `y_series.groupby(by=uid_series).transform("mean")` on line 132 before the 80/20 split. Test set labels are directly encoded into the feature `uid_fraud_mean` in the training set.
  2. The IEEE-CIS dataset is ordered by `TransactionDT`. Aggregating across the full dataset uses future fraud outcomes to predict past transactions.
  3. Feature selection (`corrwith(y)` on line 101), median imputation, and standard scaling are also computed on the combined train and test sets.
- **Consequence:** Reported test performance is inflated by label leakage. The evaluation does not reflect genuine generalization.
- **Verification Method:** Inspect `data/loader.py` lines 132 and 163. The split occurs after target encoding.

---

### [CRIT-03] Transmitting and Aggregating Absolute Model Weights Instead of Gradient Updates
- **Severity:** CRITICAL
- **Category:** Federated Learning / Mathematical Formulation
- **File:** `experiment/client.py` (lines 248-285), `orchestration/flower_strategy.py` (lines 332-337, 562-580)
- **Evidence:**
  In `experiment/client.py`:
  ```python
  updated_params = get_parameters(self.net) # Full state_dict
  if self.is_adversary and self.attack_type == "sign_flip":
      updated_params = [-p if np.issubdtype(p.dtype, np.floating) else p for p in updated_params]
  ```
  In `orchestration/flower_strategy.py`:
  ```python
  for client_idx, client_ndarrays in enumerate(client_param_ndarrays):
      layer_sum += client_ndarrays[layer_idx] * rep_weights[client_idx]
  agg_layer = (layer_sum / sum_rep).astype(orig_dtype)
  ```
- **Mathematical Reasoning:**
  In standard FL, the client update is $\Delta w_i^{(t)} = w_i^{(t)} - w^{(t-1)}$.
  The paper specifies in Equation (1):
  $$\Delta w_i^* = -\Delta w_i$$
  The code negates the absolute weights:
  $$w_i^* = -w_i$$
  In real FL, absolute weights are dominated by the global model base $w^{(t-1)}$. The cosine similarity between absolute model weights across clients is almost always $> 0.999$, even when client gradients point in opposite directions. The defense only detected "sign flipping" because the adversary negated the base weights, which is not an FL gradient poisoning attack.
- **Consequence:** The defense evaluates the static geometry of model weights rather than optimization trajectories. A true update sign flip ($\Delta w_i^* = -\Delta w_i$) yields $w_i^* = w^{(t-1)} - \Delta w_i \approx w^{(t-1)}$ and would completely bypass the defense.
- **Verification Method:**
  Compute $\cos(w^{(t-1)} - \Delta w, w^{(t-1)} + \Delta w)$ with $\|w^{(t-1)}\| \gg \|\Delta w\|$. The cosine similarity is approximately $+1.0$, not $-1.0$.

---

### [CRIT-04] Fictitious "Gated Cascade" Architecture (Escalation Gates Not Implemented)
- **Severity:** CRITICAL
- **Category:** Research Integrity / Software Architecture
- **File:** `orchestration/flower_strategy.py` (lines 401-418)
- **Evidence:**
  In `orchestration/flower_strategy.py`:
  ```python
  a1, c1 = self.layer1.score(flattened_tensors)
  a2, c2 = self.layer2.score(flattened_tensors)
  a3, c3 = self.layer3.score(flattened_tensors, client_ids)

  a_combined = torch.min(torch.min(a1, a2), a3)
  ```
- **Explanation:**
  The README and paper abstract claim a "gated cascade defense" where updates are resolved at Layer 1 ($O(d)$), and only escalate to Layer 2 or Layer 3 if confidence is below dynamic thresholds $\tau_1, \tau_2$.
  In code, confidence scores `c1`, `c2`, and `c3` are completely ignored. Every layer is executed unconditionally for every client in every round. Escalation routing logic does not exist.
- **Consequence:** The claimed $O(d)$ computational efficiency, formal cascade false positive rate guarantees, and dynamic gating mechanisms are not present in the code.
- **Verification Method:** Search for `c1` and `c2` in `orchestration/flower_strategy.py`. They are returned by `.score()` but never referenced again.

---

### [CRIT-05] Paper Narrative Contradicts Its Own Published Ablation Data
- **Severity:** CRITICAL
- **Category:** Research Integrity / Experimental Consistency
- **File:** `paper/paper.tex` (lines 598-651)
- **Evidence:**
  In `paper/paper.tex` (line 648):
  > `\textbf{Finding~4 --- Full \sysname{} is Pareto-optimal.}`  
  > Full \sysname{} achieves the highest AUC, F1, and Recall simultaneously across all four attack conditions, confirming the layers are complementary.
  
  Direct comparison with Table III in `paper/paper.tex` (lines 609-625):
  
  | Condition | Metric | Full TRAP | No Layer 1 | No Layer 2 | No Layer 3 | Better than Full TRAP? |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Clean** | AUC | 0.702 | **0.705** | 0.678 | 0.700 | **Yes (No L1)** |
  | **SF 20%** | AUC | 0.669 | **0.686** | 0.672 | 0.808 | **Yes (No L1)** |
  | **SF 20%** | F1 | 0.304 | **0.335** | 0.305 | 0.000 | **Yes (No L1)** |
  | **SF 20%** | Recall | 0.185 | **0.208** | 0.185 | 0.000 | **Yes (No L1)** |
  | **MR 20%** | F1 | 0.270 | **0.312** | **0.367** | **0.335** | **Yes (All 3 ablations beat Full TRAP)** |
  | **MR 20%** | Recall | 0.158 | **0.190** | **0.235** | **0.207** | **Yes (All 3 ablations beat Full TRAP)** |

- **Consequence:** The claim of Pareto optimality is directly contradicted by the numbers presented in the paper's own table.
- **Verification Method:** Compare line 648 of `paper/paper.tex` with Table III lines 609–625.

---

## 4. High-Severity Findings

### [HIGH-01] Mathematical Mischaracterization of ROC-AUC
- **Severity:** HIGH
- **Category:** Mathematical / Statistical Error
- **File:** `paper/paper.tex` (lines 499, 508-510, 537-543, 627)
- **Evidence:**
  Paper line 499: `\small $^\dagger$ All-benign collapse: model predicts all transactions as legitimate (Recall=0). High AUC reflects benign-class only.`
- **Mathematical Reasoning:**
  ROC-AUC is the Wilcoxon-Mann-Whitney statistic: $\text{AUC} = P(\hat{y}_{\text{fraud}} > \hat{y}_{\text{benign}})$. It measures the probability that a randomly selected positive sample has a higher score than a randomly selected negative sample. It cannot reflect the "benign class only." An AUC of 0.807 indicates that the model separates the classes with 80.7% ranking accuracy. Recall=0 occurred purely because the classification threshold was fixed at 0.5 while all probabilities shifted below 0.5.
- **Consequence:** The paper's core narrative explaining baseline failure is mathematically invalid.

---

### [HIGH-02] Post-Processing Decision Threshold Tuning on Test Set Probabilities
- **Severity:** HIGH
- **Category:** Evaluation Error / Data Snooping
- **File:** `threshold_sweep.py` (lines 4-10, 48-74), `experiment/metrics.py` (lines 38-40), `probs_clean.npz`, `probs_signflip20.npz`
- **Evidence:**
  `threshold_sweep.py` sweeps decision thresholds on test set probabilities exported by `experiment/client.py` (line 515).
  In `experiment/metrics.py` (lines 38-40):
  ```python
  # Binary metrics at threshold 0.5
  # Chenged to 0.33
  y_pred_bin = (y_pred_prob >= 0.33).astype(int)
  ```
- **Consequence:** Tuning decision thresholds on test set probabilities violates valid evaluation methodology.

---

### [HIGH-03] Flawed Baseline Implementations (B2 Krum, B6 FLTrust, B7 FoolsGold, B8 DP-FL, B9 FLDetector)
- **Severity:** HIGH
- **Category:** Baseline Validity
- **File:** `baselines/b2_krum.py` (line 8), `baselines/b6_fltrust.py` (lines 19-21), `baselines/b7_foolsgold.py` (lines 32-38), `baselines/b8_dpfl.py` (lines 8, 22)
- **Evidence:**
  1. **Krum (`b2_krum.py`):** Hardcodes $f=1$. When 4 out of 10 clients are adversaries (40%), Krum assumes $f=1$. Furthermore, when $N \le 2f+2$, Krum's theoretical bound is violated.
  2. **FLTrust (`b6_fltrust.py`):** When `server_gradient` is None (which is always true in `BaselineStrategy`), it sets $g_0 = \text{mean}(g_i)$. It trusts the poisoned average of clients.
  3. **FoolsGold (`b7_foolsgold.py`):** Lacks multi-round historical accumulation. Weights are computed as $\text{softmax}((1 - \max \text{sim}) \times 5)$. Adversaries with orthogonal noise receive higher weights than honest clients.
  4. **DP-FL (`b8_dpfl.py`):** Clips the flattened absolute weights vector with `clip_norm = 1.0`. The actual weight vector norm is 5–20, so model weights are severely scaled down every round.
  5. **FLDetector (`b9_fldetector.py`):** Replaces Zhang et al.'s Cauchy loss prediction tracking with a 4-line check for negative cosine similarity.
- **Consequence:** Baseline comparisons are unrepresentative of the actual published algorithms.

---

### [HIGH-04] BatchNorm Buffers Flattened and Linearly Averaged
- **Severity:** HIGH
- **Category:** Machine Learning Error
- **File:** `experiment/client.py` (lines 41-52), `orchestration/flower_strategy.py` (lines 557-585)
- **Evidence:**
  `FraudMLP` includes `nn.BatchNorm1d(hidden_dim)`.
  In `experiment/client.py` (lines 41-52), `get_parameters` dumps the full `state_dict`, including `running_mean`, `running_var`, and `num_batches_tracked`.
  In `orchestration/flower_strategy.py` (lines 562-580), `running_var` is averaged as a linear combination across clients.
  In lines 584–585, non-floating buffers (`num_batches_tracked`) are copied from Client 0. In adversarial experiments, Client 0 is an adversary.
- **Consequence:** Linearly averaging sample variances under non-IID distributions is statistically invalid, and copying non-floating state from Client 0 allows the adversary to control server buffers.

---

### [HIGH-05] Client Drift Caused by Heterogeneous Local Loss Formulations
- **Severity:** HIGH
- **Category:** Machine Learning / Optimization Error
- **File:** `experiment/client.py` (lines 189-206)
- **Evidence:**
  ```python
  _n_pos = float(_all_labels.sum())
  _n_neg = float(len(_all_labels) - _n_pos)
  _pos_weight = float(np.clip(_n_neg / max(_n_pos, 1.0), 1.0, 100.0))
  ```
- **Explanation:**
  Under Dirichlet non-IID partitioning, each client has a different fraud ratio, resulting in different values for `_pos_weight`. Each client optimizes a different weighted objective function, inducing client drift independent of adversarial poisoning.
  Under `label_flip`, 96.5% of samples become positive, and the loss multiplier ($\approx 27\times$) amplifies the flipped loss.
- **Consequence:** Optimization objectives diverge across clients, destabilizing global aggregation.

---

## 5. Medium-Severity Findings

### [MED-01] Fictitious Geographic Partitioning
- **Severity:** MEDIUM
- **Category:** Research Integrity / Experimental Validity
- **File:** `data/partitioner.py` (lines 54-87)
- **Evidence:**
  `GeographicPartitioner` claims to model cross-silo banks across North America, Europe, Asia-Pacific, and Latin America. In `data/partitioner.py` (lines 83-86), it computes `c % 4`, assigns region names, and calls `DirichletPartitioner(alpha=0.5)` without using the region assignments.
- **Consequence:** The claimed "geographic regional fraud rate skews" are absent from the implementation.

---

### [MED-02] Single-Run Conclusions Without Error Bars
- **Severity:** MEDIUM
- **Category:** Statistical Methodology
- **File:** `paper/paper.tex` (lines 360, 463-502, 603-629), `run_seed_sweep.py`
- **Evidence:**
  All values in Table II and Table III are single point estimates from seed 44. No standard deviations or confidence intervals are reported.

---

### [MED-03] Layer 2 MAD Collapse on Low Residual Variance
- **Severity:** MEDIUM
- **Category:** Mathematical / Anomaly Detection Robustness
- **File:** `layers/layer2_spectral.py` (lines 144-151)
- **Evidence:**
  ```python
  if MAD < self.epsilon:
      return torch.ones(N, device=device, dtype=dtype), torch.ones(N, device=device, dtype=dtype)
  ```
  If $\ge 50\%$ of client residuals are near zero, MAD evaluates to 0, causing Layer 2 to assign $a_2 = 1.0$ to all clients (including adversaries).

---

### [MED-04] Advanced Attack Models (A1, A2, A3) Are Dead Code
- **Severity:** MEDIUM
- **Category:** Code Integrity
- **File:** `attacks/a1_oracle_whitebox.py`, `attacks/a2_grinding.py`, `attacks/a3_spectral_matching.py`
- **Evidence:**
  None of these modules are imported or used in `train.py` or `run_experiments.py`. The experiments only run basic parameter negation and scaling.

---

### [MED-05] 10-Fold Redundant Test Set Evaluation
- **Severity:** MEDIUM
- **Category:** Computational Efficiency
- **File:** `train.py` (lines 425-431), `experiment/client.py` (lines 388-422)
- **Evidence:**
  Every virtual client loads the full test set into `val_loader`. In each round, all 10 clients evaluate the identical model on the same 30,000 test samples.

---

## 6. Low-Severity Findings

### [LOW-01] Non-Existent Future Package Versions in `pyproject.toml`
- **Severity:** LOW
- **Category:** Reproducibility / Environment
- **File:** `pyproject.toml` (lines 6-16)
- **Evidence:**
  Specifies `requires-python = ">=3.14, < 4.0"`, `torch>=2.13.0`, `scikit-learn>=1.9.0`, `pandas>=3.0.5`, `numpy>=2.5.2`, `pytest>=9.1.1`, and `ray>=2.57.0`. Standard `uv sync` or `pip install` fails.

---

### [LOW-02] Typo in Ablation Result Filename (`LignFlip`)
- **Severity:** LOW
- **Category:** Software Engineering
- **File:** `results/Sep3/Ablation_NoL3_LignFlip20pct.json`
- **Evidence:** Filename contains `LignFlip` instead of `LabelFlip`.

---

### [LOW-03] Epoch Hyperparameter Inconsistency Across Files
- **Severity:** LOW
- **Category:** Configuration Inconsistency
- **File:** `train.py` (line 95, `default=1`), `Dockerfile` (line 82, `--epochs-per-round 2`), `run_experiments.py` (line 70, `EPOCHS = 10`), `paper/paper.tex` (line 357, `Local epochs: 10`).

---

## 7. Mathematical Audit

| Equation / Component | Paper / Expected Formulation | Actual Code Implementation | Mathematical Discrepancy | Consequence |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1 Sample Z-Score** | $z = \frac{\|g_i\|_2 - \mu}{\sigma}$, threshold $Z=3.3$ | Sample mean and sample std computed over $N=10$ gradients | Sample z-score on $N=10$ is strictly bounded by $z_{\max} \le \frac{9}{\sqrt{10}} \approx 2.846 < 3.3$ | $a_1 \ge 0.6116$ for all inputs. Layer 1 cannot reject any update. |
| **ROC-AUC Definition** | Wilcoxon-Mann-Whitney ranking probability $P(\hat{y}_+ > \hat{y}_-)$ | `skm.roc_auc_score(all_labels, all_preds)` | Claimed in paper: "High AUC reflects benign-class only" | Mathematically invalid claim. High AUC indicates ranking ability despite threshold shift. |
| **FL Parameter Delta** | $\Delta w_i = w_i^{(t)} - w^{(t-1)}$ | Absolute parameters $w_i$ dumped via `state_dict()` | Does not subtract previous global model $w^{(t-1)}$ | Compares raw model weights rather than gradient updates; $\cos(w_i, w_j) \approx 1$. |
| **Sign-Flip Attack** | $\Delta w_i^* = -\Delta w_i$ | $w_i^* = -w_i$ | Negates entire model weight tensors, biases, and running stats | Artificially distorts weights; does not reflect gradient sign-flip attacks. |
| **Model Replacement** | $\Delta w_i^* = \gamma \Delta w_i$ | $w_i^* = \gamma w_i$ | Scales absolute weights by $\gamma = 10$ | Distorts global model base rather than scaling the optimization step. |
| **FoolsGold Diversity** | $\alpha_i = 1 - \max_{j \neq i} \cos(H_i, H_j)$ using historical updates | $\alpha_i = 1 - \max_{j \neq i} \cos(w_i, w_j)$; $\text{softmax}(5 \alpha)$ | Computed statelessly on raw weights; inverts weighting logic | Assigns highest aggregation weights to divergent adversaries. |
| **DP-FL Gradient Clipping** | $\|g_i\|_2 \le C$ | Clamps absolute weights with $C = 1.0$ | Model weights norm is $5\text{--}20$, so weights are squashed | Global model weights collapse toward zero every round. |
| **Reputation Recovery** | $\liminf R_i \ge 0.1417$ under steady state | Streaks with $R_i < 0.1$ for $K=5$ set weight to $0.0$ | Recovery takes $\ge 27$ rounds of honest updates | Misaligned with claimed 15-round recovery bound. |

---

## 8. Federated Learning Audit

- **Client Logic:** Clients train locally using PyTorch. However, `optim.Adam` is re-instantiated every round in `fit()`, discarding momentum and second-moment buffers ($m_t, v_t$).
- **Server Logic:** Server flattens heterogeneous parameter types (trainable weights, biases, BatchNorm running statistics) into a single 1D tensor for scoring and aggregation.
- **Aggregation Procedure:** Aggregation operates on absolute weights. Non-floating buffers are copied from Client 0 (an adversary in attack scenarios).
- **Client Selection & Communication:** All 10 clients participate in every round ($C = 1.0$).
- **Data Partitioning:** Implemented via Dirichlet distribution ($\alpha=0.5$). The claimed "Geographic Regional" partitioning is a wrapper that applies Dirichlet without regional stratification.
- **Privacy & Security Assumptions:** Clients share entire model state dictionaries without encryption or differential privacy. When DP is evaluated in baseline B8, clipping is misapplied to model weights.

---

## 9. Data Leakage & Experimental Validity Audit

1. **Target Leakage:** `data/loader.py` (line 132) computes target encoding (`uid_fraud_mean`) across the full dataset prior to the 80/20 train/test split, directly leaking test set labels into the training set.
2. **Temporal Leakage:** IEEE-CIS is a time-ordered transaction dataset. Future fraud labels are averaged into historical transactions.
3. **Preprocessing Leakage:** Feature selection correlations (`corrwith(y)`) and standard scaling parameters are computed on the full dataset before splitting.
4. **Test Set Contamination:** Every virtual client receives the complete global test set, evaluating on 30,000 samples each round.

---

## 10. Reproducibility Audit

1. **Dependencies:** `pyproject.toml` references non-existent package versions (`torch>=2.13.0`, `pandas>=3.0.5`, `ray>=2.57.0`).
2. **Ray Process Seeding:** While seeds are set in the main process, Ray worker processes do not explicitly re-seed PyTorch or NumPy, leaving mini-batch shuffling non-deterministic.
3. **Hardware & Environment:** The Docker setup targets an RTX 3060 with CUDA 12.1, but the base image setup uses unstable package versions.

---

## 11. Paper ↔ Code Consistency Matrix

| Paper / Claimed Behavior | Actual Code Implementation | Consistent? | Evidence | Impact |
| :--- | :--- | :---: | :--- | :--- |
| **Gated Cascade Routing** with confidence escalation gates $\tau_1, \tau_2$ | Unconditional evaluation of all 3 layers; confidence scores ignored | **NO** | `flower_strategy.py` (lines 401-418) | Core algorithmic concept not implemented. |
| **Full TRAP is Pareto-Optimal** across all scenarios | No-L1 and No-L2 achieve higher AUC, F1, and Recall in Table III | **NO** | `paper.tex` (lines 609-650) | Paper conclusions contradict reported results. |
| **Layer 1 Norm/Cosine Filter** with $Z=3.3$ rejects anomalies | $z \le 2.846$ for $N=10$; $a_1 \ge 0.6116 > 0.5$ | **NO** | `layer1_norm_cosine.py` (lines 142-154) | Layer 1 cannot reject any update. |
| **Sign-Flip Attack:** $\Delta w_i^* = -\Delta w_i$ | Absolute weights negated: $w_i^* = -w_i$ | **NO** | `client.py` (lines 263-270) | Evaluates inverted models rather than poisoned updates. |
| **Model Replacement:** $\Delta w_i^* = 10 \Delta w_i$ | Absolute weights scaled: $w_i^* = 10 w_i$ | **NO** | `client.py` (lines 277-285) | Artificially inflates model parameters. |
| **Geographic Partitioning** across 4 global regions | Assigns region labels by modulo arithmetic, then runs standard Dirichlet | **NO** | `partitioner.py` (lines 68-86) | Claimed domain-specific partitioning is absent. |
| **Evaluation of Baselines B1–B9** | Only B1 (FedAvg) and B2 (Krum) are evaluated in the paper | **NO** | `run_baselines.py` (line 59) | Baselines B3–B9 omitted from reported experiments. |
| **Evaluated Attacks A1–A6** | Only Sign-Flip, Label-Flip, and Model-Replacement run in experiments | **NO** | `run_experiments.py` (lines 89-93) | A1–A3 are standalone dead code; A4–A6 not implemented. |
| **"High AUC reflects benign-class only"** under collapse | Standard ROC-AUC ranking probability $P(\hat{y}_+ > \hat{y}_-)$ | **NO** | `paper.tex` (line 499) | Mathematically invalid explanation of baseline metrics. |

---

## 12. Issue Matrix

| ID | Severity | Category | Location | Issue | Impact | Confidence |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **ISS-01** | CRITICAL | Math / FL | `layers/layer1_norm_cosine.py:37` | Sample z-score for $N=10$ bounded by $2.846 < 3.3$; $a_1 \ge 0.6116$ | Layer 1 cannot reject any update | 100% |
| **ISS-02** | CRITICAL | Data Leakage | `data/loader.py:132-166` | Target encoding of `uid_fraud_mean` computed on full dataset before train/test split | Severe test label leakage into training | 100% |
| **ISS-03** | CRITICAL | FL / ML | `experiment/client.py:261` | Absolute weights negated/scaled rather than gradient updates $\Delta w_i$ | Defense evaluates raw weights; invalidates threat model | 100% |
| **ISS-04** | CRITICAL | Architecture | `orchestration/flower_strategy.py:401` | Gated cascade routing and confidence thresholds not implemented | Core architectural claim is unfulfilled | 100% |
| **ISS-05** | CRITICAL | Integrity | `paper/paper.tex:648` | Paper claims Pareto optimality, but Table III shows ablations beat full TRAP | Falsified scientific claim in text | 100% |
| **ISS-06** | HIGH | Math / Stats | `paper/paper.tex:499` | Paper claims high AUC "reflects benign class only" | Mathematical misinterpretation of ROC-AUC | 100% |
| **ISS-07** | HIGH | Evaluation | `threshold_sweep.py:48` | Decision threshold tuned on test set probabilities | Test set snooping | 100% |
| **ISS-08** | HIGH | Baselines | `baselines/b2_krum.py:8` | Krum hardcoded to $f=1$ under $f=4$; B6, B7, B8, B9 algorithmically broken | Unfair and broken baseline comparisons | 100% |
| **ISS-09** | HIGH | ML / Systems | `orchestration/flower_strategy.py:562` | BatchNorm variances linearly averaged; buffer copied from adversary Client 0 | Invalid statistics; adversarial buffer control | 100% |
| **ISS-10** | HIGH | ML / Optimization| `experiment/client.py:189` | Heterogeneous local loss weighting ($w_i \in [1, 100]$) across clients | Induces severe client drift | 100% |
| **ISS-11** | MEDIUM | Integrity | `data/partitioner.py:83` | `GeographicPartitioner` assigns cosmetic region names, then calls Dirichlet | Fictitious partitioning claim | 100% |
| **ISS-12** | MEDIUM | Statistics | `paper/paper.tex:Table II` | All results based on single run (seed 44) without error bars | Statistically unsupported conclusions | 100% |
| **ISS-13** | MEDIUM | Anomaly Detection | `layers/layer2_spectral.py:148` | If MAD $< 10^{-8}$, returns all ones, accepting adversaries with score 1.0 | Vulnerable to collusion / low variance | 100% |
| **ISS-14** | MEDIUM | Software Eng | `attacks/a1_oracle_whitebox.py` | Attacks A1, A2, A3 are dead code never run in experiments | Unverified attack claims | 100% |
| **ISS-15** | MEDIUM | Performance | `train.py:425` | All 10 clients evaluate global test set redundantly | 10-fold redundant computation | 100% |
| **ISS-16** | LOW | Packaging | `pyproject.toml:6` | References non-existent package versions (`torch>=2.13.0`, etc.) | Broken environment installation | 100% |
| **ISS-17** | LOW | Software Eng | `results/Sep3/Ablation...` | Typo in filename: `LignFlip` instead of `LabelFlip` | Naming inconsistency | 100% |
| **ISS-18** | LOW | Configuration | `train.py:95` vs `Dockerfile:82` | Epoch count mismatch between default args, Docker, and paper | Hyperparameter inconsistency | 100% |

---

## 13. Fundamental Validity Threats

### 1. Threat to Algorithmic Claims
- The claimed "gated cascade" is an unconditional ensemble taking coordinate-wise minima.
- Layer 1 is mathematically incapable of rejecting any update under $N=10$ clients.
- The defense operates on raw neural network weights rather than gradient updates.

### 2. Threat to Experimental Results
- Target encoding leaks test set ground truth into training features.
- Baseline implementations are severely degraded (Krum hardcoded to $f=1$; FLTrust averages poisoned clients; FoolsGold weights outliers).
- Results are reported from a single seed without variance estimates.

### 3. Threat to Scientific Interpretation
- The paper asserts that FedAvg collapses because "AUC reflects the benign class only," demonstrating a fundamental misunderstanding of ROC-AUC.
- The paper claims full TRAP is Pareto-optimal, while its own Table III shows that removing layers yields superior F1 and Recall across multiple settings.

---

## 14. What Appears Correct

1. **Closed-Form Submatrix Slicing in Layer 2:** The pre-computation of the Gram matrix $K = G G^\top$ and closed-form LOO centering in `layers/layer2_spectral.py` (lines 70-92) correctly implements kernel PCA Leave-One-Out residuals.
2. **Dual-Channel CUSUM Logic:** The mathematical formulation of the local trajectory and global consensus CUSUM accumulators in `layers/layer3_temporal.py` (lines 131-158) is correctly implemented.
3. **Atomic File Operations:** `train.py` (lines 202-248) and `run_experiments.py` (lines 350-373) use temporary files and atomic `os.replace` calls to avoid partial writes.
4. **Flower Simulation Pipeline:** The use of Flower's NumPyClient and Strategy abstraction correctly facilitates multi-round federated training execution.

---

## 15. Verification Plan

To verify these findings, execute the following concrete tests:

### Test 1: Layer 1 Mathematical Inactivity Test
```python
import torch
from layers.layer1_norm_cosine import Layer1NormCosine

layer1 = Layer1NormCosine()
# 9 honest clients, 1 extreme outlier with 10,000x norm
G = torch.randn(10, 1000)
G[0] *= 10000.0
a1, _ = layer1.score(G)
assert (a1 < 0.5).sum().item() == 0, "Layer 1 should reject, but cannot!"
print(f"Outlier score: {a1[0].item():.4f} >= 0.6116 (Cannot reject!)")
```

### Test 2: Data Leakage Verification Test
```python
from data.loader import load_ieee_cis_data
train_ds, test_ds = load_ieee_cis_data(data_dir="./data/raw", nrows=10000, synthetic_fallback=False)
# Check correlation of uid_fraud_mean with test labels
# The feature in train_ds directly reflects target labels from test_ds
```

### Test 3: True Update Sign-Flip Evasion Test
```python
import torch
from layers.layer1_norm_cosine import Layer1NormCosine

l1 = Layer1NormCosine()
w_base = torch.randn(1000) * 5.0 # Global model weights
delta_w = torch.randn(1000) * 0.05 # Honest gradient update

# Honest clients send w_base + delta_w
# Attacker flips delta_w: w_base - delta_w
G = torch.stack([w_base + delta_w + torch.randn(1000)*0.01 for _ in range(9)] + [w_base - delta_w])
a1, _ = l1.score(G)
print(f"True update sign flip score: {a1[-1].item():.4f} (Defense completely blind!)")
```

---

## 16. Final Evaluation & Verdict

### Final Questions Answered:

1. **Is the codebase mathematically and scientifically sound?**
   **No.** The codebase contains fatal mathematical flaws (Layer 1 cannot reject outliers for $N=10$; ROC-AUC is misinterpreted), severe data leakage (target encoding using test set labels before splitting), and fundamental domain errors (aggregating absolute model weights rather than update deltas).

2. **Are the reported experimental results valid and reproducible?**
   **No.** The reported results are derived from a pipeline contaminated by test label leakage, evaluated against crippled and misconfigured baselines, and based on single-seed runs. Furthermore, `pyproject.toml` lists non-existent future package versions, preventing standard reproduction.

3. **Is the implementation consistent with the paper's claims?**
   **No.** The claimed gated cascade escalation is not implemented; advanced attacks A1–A3 are dead code; geographic partitioning is fictitious; and the paper's textual claim of Pareto optimality directly contradicts its own published ablation table.

**Recommendation:** The submission requires a major mathematical correction, clean re-implementation of the federated update protocol, removal of data leakage, and a complete re-run of experiments before it can be considered scientifically defensible.
