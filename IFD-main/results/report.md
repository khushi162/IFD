Comprehensive Scientific, Mathematical, and Code Audit Report

  Target Codebase: IFD-Fintech / IFD-PART2 (D:\sarthak\Fraud-Detection-main)
  Target Publication Venue: IEEE Transactions on Information Forensics and Security (TIFS)
  Evaluated Framework: TRAP (Temporal Reputation-Aware Poisoning Defense / CascadeRouter)
  Auditor: Senior AI/ML Research Scientist, Mathematical Reviewer, and Software Auditor
  ──────
  ## 1. Executive Summary

  This audit evaluated the complete codebase, data pipeline, mathematical formulations, federated
  learning orchestration, baseline implementations, adversarial attacks, and paper claims for the
  IFD-Fintech research project.

  ### Overall State of the Codebase

  The repository contains serious mathematical errors, fundamental data leakage, architectural
  misrepresentations, and paper–code discrepancies. While the simulation runs to completion and
  produces formatted tables and figures, the underlying scientific conclusions are fundamentally
  compromised.

  ### Number of Issues by Severity

   Severity Level                                 │                  Issue Count
  ────────────────────────────────────────────────┼────────────────────────────────────────────────
   CRITICAL                                       │                       5
   HIGH                                           │                       5
   MEDIUM                                         │                       5
   LOW                                            │                       3
   Total Issues                                   │                       18

  ### Most Consequential Findings

  1. Mathematical Impossibility of Layer 1 Rejection: Under the experimental setting of N = 10
  clients, the maximum possible sample z-score is bounded by (N - 1)/√N ≈ 2.846. With a calibrated
  threshold of Z = 3.3, the anomaly score a₁ is mathematically guaranteed to satisfy a₁ ≥ 0.6116 >
  0.5 for all possible inputs. Layer 1 cannot reject any client update under any circumstances.
  2. Direct Target and Temporal Leakage: Feature 3 in loader.py:111-136 target-encodes
  uid_fraud_mean using the ground-truth labels y of the entire dataset before the 80/20 train/test
  split. Future fraud labels from the test set are directly encoded into training features.
  3. Flawed Aggregation and Attack Modeling: The code transmits and aggregates absolute neural
  network weights wᵢ rather than weight updates Δwᵢ or gradients gᵢ. The attacks in
  client.py:261-285 negate or scale the entire model parameter tensors (wᵢ^* = -wᵢ), rather than
  gradient deltas (Δwᵢ^* = -Δwᵢ).
  4. Fictitious "Gated Cascade": The gated cascade routing architecture (escalating updates based
  on confidence thresholds τ₁, τ₂) is not implemented. flower_strategy.py:401-418 unconditionally
  evaluates all three layers for all clients, ignores confidence scores entirely, and takes a
  static coordinate minimum.
  5. Paper Contradicts Its Own Data: The paper asserts in Section VI-D that full TRAP is "Pareto-
  optimal" across all conditions. In the paper's own Table III, removing Layer 1 or Layer 2
  achieves higher AUC, F1, and Recall across multiple attack scenarios.
  ──────
  ## 2. Repository Understanding

  ### Project Architecture and Workflow

  The repository implements a cross-silo Federated Learning simulation designed for financial
  credit card fraud detection on the IEEE-CIS dataset using Flower (flwr) and PyTorch.

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
  ──────
  ## 3. Critical Findings

  ### [CRIT-01] Mathematical Impossibility of Layer 1 Rejection for N = 10 Clients

  • Severity: CRITICAL
  • Category: Mathematical Error / Federated Learning Error
  • File: layer1_norm_cosine.py:37, flower_strategy.py:425-453, threshold_controller.py:125
  • Evidence:
  In layer1_norm_cosine.py:37:
    Z_THRESH_L1: float = 3.3
    ...
    z_norm = torch.abs(norms - mu_norm) / sigma_norm
    z_cos = (mu_cos - cosines) / sigma_cos
    s_norm = 1.0 - torch.sigmoid(z_norm - Z_THRESH_L1)
    s_cos = 1.0 - torch.sigmoid(z_cos - Z_THRESH_L1)
    a1 = torch.minimum(s_norm, s_cos)

  • Mathematical Reasoning:
  For any sample x₁, …, x_N ∈ ℝ, Thompson's Tau / Chauvenet's theorem proves that the maximum
  possible sample z-score is bounded:

           N - 1
    z    ≤ ─────
     max    √N

  For N = 10:

            9
    z    ≤ ─── ≈ 2.84605
     max   √10

  Because z ≤ 2.84605 is a strict algebraic upper bound, and Z_{thresh} = 3.3:

    z - 3.3 ≤ 2.84605 - 3.3 = -0.45395

                      1
    σ(z - 3.3) ≤ ──────────── ≈ 0.38842
                      0.45395
                 1 + e

    s = 1 - σ(z - 3.3) ≥ 1 - 0.38842 = 0.61158

    a₁ = min⎛s    , s   ⎞ ≥ 0.61158  ∀ inputs!
            ⎝ norm   cos⎠

  In threshold_controller.py:125, rejection is defined as:

    rejection_mask = scores < 0.5

  Because a₁ ≥ 0.61158 > 0.5 for all inputs, rejection_mask is identically False for every client.
  The rejection rate for Layer 1 is permanently 0.0%, and hard_reject = (a1 < t1) is never
  satisfied.

  • Consequence: Layer 1 is mathematically inert. It cannot detect or reject any client update in
  an N = 10 client federation, regardless of how extreme the update is (even with 10, 000 × norm
  scaling).
  • Verification Method:
  Pass 10 vectors where 9 are unit vectors and 1 has norm 10⁶. The resulting a₁ for the outlier is
  0.6116, failing to trigger rejection at threshold 0.5.
  ──────
  ### [CRIT-02] Catastrophic Target and Temporal Leakage in Preprocessing

  • Severity: CRITICAL
  • Category: Data and Experimental Validity
  • File: loader.py:111-166
  • Evidence:
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

  • Explanation:
      1. The target column isFraud (y) from the test set is directly used in y_series.
      groupby(by=uid_series).transform("mean") on line 132 before the 80/20 split. Test set labels
      are directly encoded into the feature uid_fraud_mean in the training set.
      2. The IEEE-CIS dataset is ordered by TransactionDT. Aggregating across the full dataset uses
      future fraud outcomes to predict past transactions.
      3. Feature selection (corrwith(y) on line 101), median imputation, and standard scaling are
      also computed on the combined train and test sets.
  • Consequence: Reported test performance is inflated by label leakage. The evaluation does not
  reflect genuine generalization.
  • Verification Method: Inspect loader.py:132-166 lines 132 and 163. The split occurs after target
  encoding.
  ──────
  ### [CRIT-03] Transmitting and Aggregating Absolute Model Weights Instead of Gradient Updates

  • Severity: CRITICAL
  • Category: Federated Learning / Mathematical Formulation
  • File: client.py:248-285, flower_strategy.py
  • Evidence:
  In client.py:248-270:
    updated_params = get_parameters(self.net) # Full state_dict
    if self.is_adversary and self.attack_type == "sign_flip":
        updated_params = [-p if np.issubdtype(p.dtype, np.floating) else p for p in updated_params]
  In flower_strategy.py:571-579:
    for client_idx, client_ndarrays in enumerate(client_param_ndarrays):
        layer_sum += client_ndarrays[layer_idx] * rep_weights[client_idx]
    agg_layer = (layer_sum / sum_rep).astype(orig_dtype)

  • Mathematical Reasoning:
  In standard FL, the client update is Δwᵢ^{(t)} = wᵢ^{(t)} - w^{(t-1)}.
  The paper specifies in Equation (1):

      *
    Δw  = -Δwᵢ
      i

  The code negates the absolute weights:

     *
    w  = -wᵢ
     i

  In real FL, absolute weights are dominated by the global model base w^{(t-1)}. The cosine
  similarity between absolute model weights across clients is almost always > 0.999, even when
  client gradients point in opposite directions. The defense only detected "sign flipping" because
  the adversary negated the base weights, which is not an FL gradient poisoning attack.

  • Consequence: The defense evaluates the static geometry of model weights rather than
  optimization trajectories. A true update sign flip (Δwᵢ^* = -Δwᵢ) yields wᵢ^* = w^{(t-1)} - Δwᵢ ≈
  w^{(t-1)} and would completely bypass the defense.
  • Verification Method:
  Compute cos(w^{(t-1)} - Δw, w^{(t-1)} + Δw) with ‖w^{(t-1)}‖ ≫ ‖Δw‖. The cosine similarity is
  approximately +1.0, not -1.0.
  ──────
  ### [CRIT-04] Fictitious "Gated Cascade" Architecture (Escalation Gates Not Implemented)

  • Severity: CRITICAL
  • Category: Research Integrity / Software Architecture
  • File: flower_strategy.py:401-418
  • Evidence:
  In flower_strategy.py:401-418:
    a1, c1 = self.layer1.score(flattened_tensors)
    a2, c2 = self.layer2.score(flattened_tensors)
    a3, c3 = self.layer3.score(flattened_tensors, client_ids)

    a_combined = torch.min(torch.min(a1, a2), a3)

  • Explanation:
  The README and paper abstract claim a "gated cascade defense" where updates are resolved at Layer
  1 (O(d)), and only escalate to Layer 2 or Layer 3 if confidence is below dynamic thresholds τ₁,
  τ₂.
  In code, confidence scores c1, c2, and c3 are completely ignored. Every layer is executed
  unconditionally for every client in every round. Escalation routing logic does not exist.
  • Consequence: The claimed O(d) computational efficiency, formal cascade false positive rate
  guarantees, and dynamic gating mechanisms are not present in the code.
  • Verification Method: Search for c1 and c2 in flower_strategy.py. They are returned by .score()
  but never referenced again.
  ──────
  ### [CRIT-05] Paper Narrative Contradicts Its Own Published Ablation Data

  • Severity: CRITICAL
  • Category: Research Integrity / Experimental Consistency
  • File: paper.tex:598-651
  • Evidence:
  In paper.tex:648:
  │ \textbf{Finding~4 --- Full \sysname{} is Pareto-optimal.}
  │ Full \sysname{} achieves the highest AUC, F1, and Recall simultaneously across all four attack
  │ conditions, confirming the layers are complementary.
  Direct comparison with Table III in paper.tex:609-625:
   Condition │ Metric │ Full TRAP │ No Layer 1 │ No Layer 2 │ No Layer 3 │ Better than Full TRAP?
  ───────────┼────────┼───────────┼────────────┼────────────┼────────────┼─────────────────────────
   Clean     │  AUC   │   0.702   │   0.705    │   0.678    │   0.700    │       Yes (No L1)
   SF 20%    │  AUC   │   0.669   │   0.686    │   0.672    │   0.808    │       Yes (No L1)
   SF 20%    │   F1   │   0.304   │   0.335    │   0.305    │   0.000    │       Yes (No L1)
   SF 20%    │ Recall │   0.185   │   0.208    │   0.185    │   0.000    │       Yes (No L1)
   MR 20%    │   F1   │   0.270   │   0.312    │   0.367    │   0.335    │  Yes (All 3 ablations
             │        │           │            │            │            │     beat Full TRAP)
   MR 20%    │ Recall │   0.158   │   0.190    │   0.235    │   0.207    │  Yes (All 3 ablations
             │        │           │            │            │            │     beat Full TRAP)

  • Consequence: The claim of Pareto optimality is directly contradicted by the numbers presented
  in the paper's own table.
  • Verification Method: Compare line 648 of paper.tex:648 with Table III lines 609–625.
  ──────
  ## 4. High-Severity Findings

  ### [HIGH-01] Mathematical Mischaracterization of ROC-AUC

  • Severity: HIGH
  • Category: Mathematical / Statistical Error
  • File: paper.tex
  • Evidence:
  Paper line 499: \small $^\dagger$ All-benign collapse: model predicts all transactions as
  legitimate (Recall=0). High AUC reflects benign-class only.
  • Mathematical Reasoning:
  ROC-AUC is the Wilcoxon-Mann-Whitney statistic: AUC = P(ŷ_{fraud} > ŷ_{benign}). It measures the
  probability that a randomly selected positive sample has a higher score than a randomly selected
  negative sample. It cannot reflect the "benign class only." An AUC of 0.807 indicates that the
  model separates the classes with 80.7% ranking accuracy. Recall=0 occurred purely because the
  classification threshold was fixed at 0.5 while all probabilities shifted below 0.5.
  • Consequence: The paper's core narrative explaining baseline failure is mathematically invalid.
  ──────
  ### [HIGH-02] Post-Processing Decision Threshold Tuning on Test Set Probabilities

  • Severity: HIGH
  • Category: Evaluation Error / Data Snooping
  • File: threshold_sweep.py, metrics.py:38-40, probs_clean.npz, probs_signflip20.npz
  • Evidence:
  threshold_sweep.py:4-10 sweeps decision thresholds on test set probabilities exported by
  client.py:515.
  In metrics.py:38-40:
    # Binary metrics at threshold 0.5
    # Chenged to 0.33
    y_pred_bin = (y_pred_prob >= 0.33).astype(int)

  • Consequence: Tuning decision thresholds on test set probabilities violates valid evaluation
  methodology.
  ──────
  ### [HIGH-03] Flawed Baseline Implementations (B2 Krum, B6 FLTrust, B7 FoolsGold, B8 DP-FL, B9
  FLDetector)

  • Severity: HIGH
  • Category: Baseline Validity
  • File: b2_krum.py:8, b6_fltrust.py:19-21, b7_foolsgold.py:32-38, b8_dpfl.py
  • Evidence:
      1. Krum (b2_krum.py): Hardcodes f = 1. When 4 out of 10 clients are adversaries (40%), Krum
      assumes f = 1. Furthermore, when N ≤ 2f + 2, Krum's theoretical bound is violated.
      2. FLTrust (b6_fltrust.py): When server_gradient is None (which is always true in
      BaselineStrategy), it sets g₀ = mean(gᵢ). It trusts the poisoned average of clients.
      3. FoolsGold (b7_foolsgold.py): Lacks multi-round historical accumulation. Weights are
      computed as softmax((1 - max sim) × 5). Adversaries with orthogonal noise receive higher
      weights than honest clients.
      4. DP-FL (b8_dpfl.py): Clips the flattened absolute weights vector with clip_norm = 1.0. The
      actual weight vector norm is 5–20, so model weights are severely scaled down every round.
      5. FLDetector (b9_fldetector.py): Replaces Zhang et al.'s Cauchy loss prediction tracking
      with a 4-line check for negative cosine similarity.
  • Consequence: Baseline comparisons are unrepresentative of the actual published algorithms.
  ──────
  ### [HIGH-04] BatchNorm Buffers Flattened and Linearly Averaged

  • Severity: HIGH
  • Category: Machine Learning Error
  • File: client.py:41-52, flower_strategy.py:557-585
  • Evidence:
  FraudMLP includes nn.BatchNorm1d(hidden_dim).
  In client.py:41-52, get_parameters dumps the full state_dict, including running_mean, running_var,
  and num_batches_tracked.
  In flower_strategy.py:562-580, running_var is averaged as a linear combination across clients.
  In lines 584–585, non-floating buffers (num_batches_tracked) are copied from Client 0. In
  adversarial experiments, Client 0 is an adversary.
  • Consequence: Linearly averaging sample variances under non-IID distributions is statistically
  invalid, and copying non-floating state from Client 0 allows the adversary to control server
  buffers.
  ──────
  ### [HIGH-05] Client Drift Caused by Heterogeneous Local Loss Formulations

  • Severity: HIGH
  • Category: Machine Learning / Optimization Error
  • File: client.py:189-206
  • Evidence:
    _n_pos = float(_all_labels.sum())
    _n_neg = float(len(_all_labels) - _n_pos)
    _pos_weight = float(np.clip(_n_neg / max(_n_pos, 1.0), 1.0, 100.0))

  • Explanation:
  Under Dirichlet non-IID partitioning, each client has a different fraud ratio, resulting in
  different values for _pos_weight. Each client optimizes a different weighted objective function,
  inducing client drift independent of adversarial poisoning.
  Under label_flip, 96.5% of samples become positive, and the loss multiplier (≈ 27 ×) amplifies
  the flipped loss.
  • Consequence: Optimization objectives diverge across clients, destabilizing global aggregation.
  ──────
  ## 5. Medium-Severity Findings

  ### [MED-01] Fictitious Geographic Partitioning

  • Severity: MEDIUM
  • Category: Research Integrity / Experimental Validity
  • File: partitioner.py:54-87
  • Evidence:
  GeographicPartitioner claims to model cross-silo banks across North America, Europe, Asia-Pacific,
  and Latin America. In partitioner.py:83-86, it computes c % 4, assigns region names, and calls
  DirichletPartitioner(alpha=0.5) without using the region assignments.
  • Consequence: The claimed "geographic regional fraud rate skews" are absent from the
  implementation.
  ──────
  ### [MED-02] Single-Run Conclusions Without Error Bars

  • Severity: MEDIUM
  • Category: Statistical Methodology
  • File: paper.tex, run_seed_sweep.py
  • Evidence:
  All values in Table II and Table III are single point estimates from seed 44. No standard
  deviations or confidence intervals are reported.
  ──────
  ### [MED-03] Layer 2 MAD Collapse on Low Residual Variance

  • Severity: MEDIUM
  • Category: Mathematical / Anomaly Detection Robustness
  • File: layer2_spectral.py:144-151
  • Evidence:
    if MAD < self.epsilon:
        return torch.ones(N, device=device, dtype=dtype), torch.ones(N, device=device, dtype=dtype)
  If ≥ 50% of client residuals are near zero, MAD evaluates to 0, causing Layer 2 to assign a₂ = 1.
  0 to all clients (including adversaries).
  ──────
  ### [MED-04] Advanced Attack Models (A1, A2, A3) Are Dead Code

  • Severity: MEDIUM
  • Category: Code Integrity
  • File: a1_oracle_whitebox.py, a2_grinding.py, a3_spectral_matching.py
  • Evidence:
  None of these modules are imported or used in train.py or run_experiments.py. The experiments
  only run basic parameter negation and scaling.
  ──────
  ### [MED-05] 10-Fold Redundant Test Set Evaluation

  • Severity: MEDIUM
  • Category: Computational Efficiency
  • File: train.py:425-431, client.py:388-422
  • Evidence:
  Every virtual client loads the full test set into val_loader. In each round, all 10 clients
  evaluate the identical model on the same 30,000 test samples.
  ──────
  ## 6. Low-Severity Findings

  ### [LOW-01] Non-Existent Future Package Versions in pyproject.toml

  • Severity: LOW
  • Category: Reproducibility / Environment
  • File: pyproject.toml:6-16
  • Evidence:
  Specifies requires-python = ">=3.14, < 4.0", torch>=2.13.0, scikit-learn>=1.9.0, pandas>=3.0.5,
  numpy>=2.5.2, pytest>=9.1.1, and ray>=2.57.0. Standard uv sync or pip install fails.
  ──────
  ### [LOW-02] Typo in Ablation Result Filename (LignFlip)

  • Severity: LOW
  • Category: Software Engineering
  • File: Ablation_NoL3_LignFlip20pct.json
  • Evidence: Filename contains LignFlip instead of LabelFlip.
  ──────
  ### [LOW-03] Epoch Hyperparameter Inconsistency Across Files

  • Severity: LOW
  • Category: Configuration Inconsistency
  • File: train.py:95 (default=1), Dockerfile:82 (--epochs-per-round 2), run_experiments.py:70
  (EPOCHS = 10), paper.tex:357 (Local epochs: 10).
  ──────
  ## 7. Mathematical Audit

   Equation / Comp… │ Paper / Expec… │ Actual Code I… │ Mathematical Discrepancy │ Consequence
  ──────────────────┼────────────────┼────────────────┼──────────────────────────┼─────────────────
   Layer 1 Sample   │ z = (‖gᵢ‖₂ -   │ Sample mean    │ Sample z-score on N = 10 │ a₁ ≥ 0.6116 for
   Z-Score          │ μ)/σ,          │ and sample std │ is strictly bounded by   │ all inputs.
                    │ threshold Z =  │ computed over  │ z_{max} ≤ 9/√10 ≈ 2.846  │ Layer 1 cannot
                    │ 3.3            │ N = 10         │ < 3.3                    │ reject any
                    │                │ gradients      │                          │ update.
   ROC-AUC          │ Wilcoxon-Mann- │ skm.roc_auc_sc │ Claimed in paper: "High  │ Mathematically
   Definition       │ Whitney        │ ore(all_labels │ AUC reflects benign-     │ invalid claim.
                    │ ranking        │ , all_preds)   │ class only"              │ High AUC
                    │ probability    │                │                          │ indicates
                    │ P(ŷ₊ > ŷ₋)     │                │                          │ ranking ability
                    │                │                │                          │ despite
                    │                │                │                          │ threshold
                    │                │                │                          │ shift.
   FL Parameter     │ Δwᵢ = wᵢ^{(t)} │ Absolute       │ Does not subtract        │ Compares raw
   Delta            │ w^{(t-1)}      │ parameters wᵢ  │ previous global model    │ model weights
                    │                │ dumped via     │ w^{(t-1)}                │ rather than
                    │                │ state_dict()   │                          │ gradient
                    │                │                │                          │ updates;
                    │                │                │                          │ cos(wᵢ, wⱼ) ≈
                    │                │                │                          │ 1.
   Sign-Flip Attack │ Δwᵢ^* = -Δwᵢ   │ wᵢ^* = -wᵢ     │ Negates entire model     │ Artificially
                    │                │                │ weight tensors, biases,  │ distorts
                    │                │                │ and running stats        │ weights; does
                    │                │                │                          │ not reflect
                    │                │                │                          │ gradient sign-
                    │                │                │                          │ flip attacks.
   Model            │ Δwᵢ^* = γΔwᵢ   │ wᵢ^* = γwᵢ     │ Scales absolute weights  │ Distorts global
   Replacement      │                │                │ by γ = 10                │ model base
                    │                │                │                          │ rather than
                    │                │                │                          │ scaling the
                    │                │                │                          │ optimization
                    │                │                │                          │ step.
   FoolsGold        │ αᵢ = 1 -       │ αᵢ = 1 -       │ Computed statelessly on  │ Assigns highest
   Diversity        │ max_{j≠i}      │ max_{j≠i}      │ raw weights; inverts     │ aggregation
                    │ cos(Hᵢ, Hⱼ)    │ cos(wᵢ, wⱼ);   │ weighting logic          │ weights to
                    │ using          │ softmax(5α)    │                          │ divergent
                    │ historical     │                │                          │ adversaries.
                    │ updates        │                │                          │
   DP-FL Gradient   │ ‖gᵢ‖₂ ≤ C      │ Clamps         │ Model weights norm is 5- │ Global model
   Clipping         │                │ absolute       │ -20, so weights are      │ weights
                    │                │ weights with C │ squashed                 │ collapse toward
                    │                │ = 1.0          │                          │ zero every
                    │                │                │                          │ round.
   Reputation       │ liminf Rᵢ ≥    │ Streaks with   │ Recovery takes ≥ 27      │ Misaligned with
   Recovery         │ 0.1417 under   │ Rᵢ < 0.1 for K │ rounds of honest updates │ claimed 15-
                    │ steady state   │ = 5 set weight │                          │ round recovery
                    │                │ to 0.0         │                          │ bound.
  ──────
  ## 8. Federated Learning Audit

  • Client Logic: Clients train locally using PyTorch. However, optim.Adam is re-instantiated every
  round in fit(), discarding momentum and second-moment buffers (mₜ, vₜ).
  • Server Logic: Server flattens heterogeneous parameter types (trainable weights, biases,
  BatchNorm running statistics) into a single 1D tensor for scoring and aggregation.
  • Aggregation Procedure: Aggregation operates on absolute weights. Non-floating buffers are
  copied from Client 0 (an adversary in attack scenarios).
  • Client Selection & Communication: All 10 clients participate in every round (C = 1.0).
  • Data Partitioning: Implemented via Dirichlet distribution (α = 0.5). The claimed "Geographic
  Regional" partitioning is a wrapper that applies Dirichlet without regional stratification.
  • Privacy & Security Assumptions: Clients share entire model state dictionaries without
  encryption or differential privacy. When DP is evaluated in baseline B8, clipping is misapplied
  to model weights.
  ──────
  ## 9. Data Leakage & Experimental Validity Audit

  1. Target Leakage: loader.py:132 computes target encoding (uid_fraud_mean) across the full
  dataset prior to the 80/20 train/test split, directly leaking test set labels into the training
  set.
  2. Temporal Leakage: IEEE-CIS is a time-ordered transaction dataset. Future fraud labels are
  averaged into historical transactions.
  3. Preprocessing Leakage: Feature selection correlations (corrwith(y)) and standard scaling
  parameters are computed on the full dataset before splitting.
  4. Test Set Contamination: Every virtual client receives the complete global test set, evaluating
  on 30,000 samples each round.
  ──────
  ## 10. Reproducibility Audit

  1. Dependencies: pyproject.toml references non-existent package versions (torch>=2.13.0,
  pandas>=3.0.5, ray>=2.57.0).
  2. Ray Process Seeding: While seeds are set in the main process, Ray worker processes do not
  explicitly re-seed PyTorch or NumPy, leaving mini-batch shuffling non-deterministic.
  3. Hardware & Environment: The Docker setup targets an RTX 3060 with CUDA 12.1, but the base
  image setup uses unstable package versions.
  ──────
  ## 11. Paper ↔ Code Consistency Matrix

   Paper / Claimed Behavi… │ Actual Code Implementation │ C… │ Evidence         │ Impact
  ─────────────────────────┼────────────────────────────┼────┼──────────────────┼──────────────────
   Gated Cascade Routing   │ Unconditional evaluation   │ NO │ flower_strategy.py:401-418 │ Core
algorithmic
   with confidence         │ of all 3 layers;           │    │                  │ concept not
   escalation gates τ₁, τ₂ │ confidence scores ignored  │    │                  │ implemented.
   Full TRAP is Pareto-    │ No-L1 and No-L2 achieve    │ NO │ paper.tex:609-650 │ Paper
   Optimal across all      │ higher AUC, F1, and Recall │    │                  │ conclusions
   scenarios               │ in Table III               │    │                  │ contradict
                           │                            │    │                  │ reported
                           │                            │    │                  │ results.
   Layer 1 Norm/Cosine     │ z ≤ 2.846 for N = 10; a₁ ≥ │ NO │ layer1_norm_cosine.py:142-154 │ Layer
1 cannot
   Filter with Z = 3.3     │ 0.6116 > 0.5               │    │                  │ reject any
   rejects anomalies       │                            │    │                  │ update.
   Sign-Flip Attack: Δwᵢ^* │ Absolute weights negated:  │ NO │ client.py:263-270 │ Evaluates
   = -Δwᵢ                  │ wᵢ^* = -wᵢ                 │    │                  │ inverted models
                           │                            │    │                  │ rather than
                           │                            │    │                  │ poisoned
                           │                            │    │                  │ updates.
   Model Replacement:      │ Absolute weights scaled:   │ NO │ client.py:277-285 │ Artificially
   Δwᵢ^* = 10Δwᵢ           │ wᵢ^* = 10wᵢ                │    │                  │ inflates model
                           │                            │    │                  │ parameters.
   Geographic Partitioning │ Assigns region labels by   │ NO │ partitioner.py:68-86 │ Claimed domain-
   across 4 global regions │ modulo arithmetic, then    │    │                  │ specific
                           │ runs standard Dirichlet    │    │                  │ partitioning is
                           │                            │    │                  │ absent.
   Evaluation of Baselines │ Only B1 (FedAvg) and B2    │ NO │ run_baselines.py:59 │ Baselines B3–B9
   B1–B9                   │ (Krum) are evaluated in    │    │                  │ omitted from
                           │ the paper                  │    │                  │ reported
                           │                            │    │                  │ experiments.
   Evaluated Attacks A1–A6 │ Only Sign-Flip, Label-     │ NO │ run_experiments.py:89-93 │ A1–A3 are
                           │ Flip, and Model-           │    │                  │ standalone dead
                           │ Replacement run in         │    │                  │ code; A4–A6 not
                           │ experiments                │    │                  │ implemented.
   "High AUC reflects      │ Standard ROC-AUC ranking   │ NO │ paper.tex:499    │ Mathematically
   benign-class only"      │ probability P(ŷ₊ > ŷ₋)     │    │                  │ invalid
   under collapse          │                            │    │                  │ explanation of
                           │                            │    │                  │ baseline
                           │                            │    │                  │ metrics.
  ──────
  ## 12. Issue Matrix

     ID   │ Sever… │  Category   │ Location         │ Issue             │ Impact            │ Con…
  ────────┼────────┼─────────────┼──────────────────┼───────────────────┼───────────────────┼──────
   ISS-01 │ CRITIC │  Math / FL  │ layers/layer1_no │ Sample z-score    │ Layer 1 cannot    │ 100%
          │   AL   │             │ rm_cosine.py:37  │ for N = 10        │ reject any update │
          │        │             │                  │ bounded by 2.846  │                   │
          │        │             │                  │ < 3.3; a₁ ≥       │                   │
          │        │             │                  │ 0.6116            │                   │
   ISS-02 │ CRITIC │    Data     │ data/loader.py:1 │ Target encoding   │ Severe test label │ 100%
          │   AL   │   Leakage   │ 32-166           │ of uid_fraud_mean │ leakage into      │
          │        │             │                  │ computed on full  │ training          │
          │        │             │                  │ dataset before    │                   │
          │        │             │                  │ train/test split  │                   │
   ISS-03 │ CRITIC │   FL / ML   │ experiment/clien │ Absolute weights  │ Defense evaluates │ 100%
          │   AL   │             │ t.py:261         │ negated/scaled    │ raw weights;      │
          │        │             │                  │ rather than       │ invalidates       │
          │        │             │                  │ gradient updates  │ threat model      │
          │        │             │                  │ Δwᵢ               │                   │
   ISS-04 │ CRITIC │ Architectur │ orchestration/fl │ Gated cascade     │ Core              │ 100%
          │   AL   │      e      │ ower_strategy.py │ routing and       │ architectural     │
          │        │             │ :401             │ confidence        │ claim is          │
          │        │             │                  │ thresholds not    │ unfulfilled       │
          │        │             │                  │ implemented       │                   │
   ISS-05 │ CRITIC │  Integrity  │ paper/paper.tex: │ Paper claims      │ Falsified         │ 100%
          │   AL   │             │ 648              │ Pareto            │ scientific claim  │
          │        │             │                  │ optimality, but   │ in text           │
          │        │             │                  │ Table III shows   │                   │
          │        │             │                  │ ablations beat    │                   │
          │        │             │                  │ full TRAP         │                   │
   ISS-06 │  HIGH  │   Math /    │ paper/paper.tex: │ Paper claims high │ Mathematical      │ 100%
          │        │    Stats    │ 499              │ AUC "reflects     │ misinterpretation │
          │        │             │                  │ benign class      │ of ROC-AUC        │
          │        │             │                  │ only"             │                   │
   ISS-07 │  HIGH  │ Evaluation  │ threshold_sweep. │ Decision          │ Test set snooping │ 100%
          │        │             │ py:48            │ threshold tuned   │                   │
          │        │             │                  │ on test set       │                   │
          │        │             │                  │ probabilities     │                   │
   ISS-08 │  HIGH  │  Baselines  │ baselines/b2_kru │ Krum hardcoded to │ Unfair and broken │ 100%
          │        │             │ m.py:8           │ f = 1 under f =   │ baseline          │
          │        │             │                  │ 4; B6, B7, B8, B9 │ comparisons       │
          │        │             │                  │ algorithmically   │                   │
          │        │             │                  │ broken            │                   │
   ISS-09 │  HIGH  │    ML /     │ orchestration/fl │ BatchNorm         │ Invalid           │ 100%
          │        │   Systems   │ ower_strategy.py │ variances         │ statistics;       │
          │        │             │ :562             │ linearly          │ adversarial       │
          │        │             │                  │ averaged; buffer  │ buffer control    │
          │        │             │                  │ copied from       │                   │
          │        │             │                  │ adversary Client  │                   │
          │        │             │                  │ 0                 │                   │
   ISS-10 │  HIGH  │    ML /     │ experiment/clien │ Heterogeneous     │ Induces severe    │ 100%
          │        │ Optimizatio │ t.py:189         │ local loss        │ client drift      │
          │        │      n      │                  │ weighting (wᵢ ∈   │                   │
          │        │             │                  │ [1, 100]) across  │                   │
          │        │             │                  │ clients           │                   │
   ISS-11 │ MEDIUM │  Integrity  │ data/partitioner │ GeographicPartiti │ Fictitious        │ 100%
          │        │             │ .py:83           │ oner assigns      │ partitioning      │
          │        │             │                  │ cosmetic region   │ claim             │
          │        │             │                  │ names, then calls │                   │
          │        │             │                  │ Dirichlet         │                   │
   ISS-12 │ MEDIUM │ Statistics  │ paper/paper.tex: │ All results based │ Statistically     │ 100%
          │        │             │ Table II         │ on single run     │ unsupported       │
          │        │             │                  │ (seed 44) without │ conclusions       │
          │        │             │                  │ error bars        │                   │
   ISS-13 │ MEDIUM │   Anomaly   │ layers/layer2_sp │ If MAD < 10⁻⁸,    │ Vulnerable to     │ 100%
          │        │  Detection  │ ectral.py:148    │ returns all ones, │ collusion / low   │
          │        │             │                  │ accepting         │ variance          │
          │        │             │                  │ adversaries with  │                   │
          │        │             │                  │ score 1.0         │                   │
   ISS-14 │ MEDIUM │  Software   │ attacks/a1_oracl │ Attacks A1, A2,   │ Unverified attack │ 100%
          │        │     Eng     │ e_whitebox.py    │ A3 are dead code  │ claims            │
          │        │             │                  │ never run in      │                   │
          │        │             │                  │ experiments       │                   │
   ISS-15 │ MEDIUM │ Performance │ train.py:425     │ All 10 clients    │ 10-fold redundant │ 100%
          │        │             │                  │ evaluate global   │ computation       │
          │        │             │                  │ test set          │                   │
          │        │             │                  │ redundantly       │                   │
   ISS-16 │  LOW   │  Packaging  │ pyproject.toml:6 │ References non-   │ Broken            │ 100%
          │        │             │                  │ existent package  │ environment       │
          │        │             │                  │ versions          │ installation      │
          │        │             │                  │ (torch>=2.13.0,   │                   │
          │        │             │                  │ etc.)             │                   │
   ISS-17 │  LOW   │  Software   │ results/Sep3/Abl │ Typo in filename: │ Naming            │ 100%
          │        │     Eng     │ ation...         │ LignFlip instead  │ inconsistency     │
          │        │             │                  │ of LabelFlip      │                   │
   ISS-18 │  LOW   │ Configurati │ train.py:95 vs   │ Epoch count       │ Hyperparameter    │ 100%
          │        │     on      │ Dockerfile:82    │ mismatch between  │ inconsistency     │
          │        │             │                  │ default args,     │                   │
          │        │             │                  │ Docker, and paper │                   │
  ──────
  ## 13. Fundamental Validity Threats

  ### 1. Threat to Algorithmic Claims

  • The claimed "gated cascade" is an unconditional ensemble taking coordinate-wise minima.
  • Layer 1 is mathematically incapable of rejecting any update under N = 10 clients.
  • The defense operates on raw neural network weights rather than gradient updates.

  ### 2. Threat to Experimental Results

  • Target encoding leaks test set ground truth into training features.
  • Baseline implementations are severely degraded (Krum hardcoded to f = 1; FLTrust averages
  poisoned clients; FoolsGold weights outliers).
  • Results are reported from a single seed without variance estimates.

  ### 3. Threat to Scientific Interpretation

  • The paper asserts that FedAvg collapses because "AUC reflects the benign class only,"
  demonstrating a fundamental misunderstanding of ROC-AUC.
  • The paper claims full TRAP is Pareto-optimal, while its own Table III shows that removing
  layers yields superior F1 and Recall across multiple settings.
  ──────
  ## 14. What Appears Correct

  1. Closed-Form Submatrix Slicing in Layer 2: The pre-computation of the Gram matrix K = GG^⊤ and
  closed-form LOO centering in layer2_spectral.py:70-92 correctly implements kernel PCA Leave-One-
  Out residuals.
  2. Dual-Channel CUSUM Logic: The mathematical formulation of the local trajectory and global
  consensus CUSUM accumulators in layer3_temporal.py:131-158 is correctly implemented.
  3. Atomic File Operations: train.py:202-248 and run_experiments.py:350-373 use temporary files
  and atomic os.replace calls to avoid partial writes.
  4. Flower Simulation Pipeline: The use of Flower's NumPyClient and Strategy abstraction correctly
  facilitates multi-round federated training execution.
  ──────
  ## 15. Verification Plan

  To verify these findings, execute the following concrete tests:

  ### Test 1: Layer 1 Mathematical Inactivity Test

    import torch
    from layers.layer1_norm_cosine import Layer1NormCosine

    layer1 = Layer1NormCosine()
    # 9 honest clients, 1 extreme outlier with 10,000x norm
    G = torch.randn(10, 1000)
    G[0] *= 10000.0
    a1, _ = layer1.score(G)
    assert (a1 < 0.5).sum().item() == 0, "Layer 1 should reject, but cannot!"
    print(f"Outlier score: {a1[0].item():.4f} >= 0.6116 (Cannot reject!)")

  ### Test 2: Data Leakage Verification Test

    from data.loader import load_ieee_cis_data
    train_ds, test_ds = load_ieee_cis_data(data_dir="./data/raw", nrows=10000,
  synthetic_fallback=False)
    # Check correlation of uid_fraud_mean with test labels
    # The feature in train_ds directly reflects target labels from test_ds

  ### Test 3: True Update Sign-Flip Evasion Test

    import torch
    from layers.layer1_norm_cosine import Layer1NormCosine

    l1 = Layer1NormCosine()
    w_base = torch.randn(1000) * 5.0 # Global model weights
    delta_w = torch.randn(1000) * 0.05 # Honest gradient update

    # Honest clients send w_base + delta_w
    # Attacker flips delta_w: w_base - delta_w
    G = torch.stack([w_base + delta_w + torch.randn(1000)*0.01 for _ in range(9)] + [w_base -
  delta_w])
    a1, _ = l1.score(G)
    print(f"True update sign flip score: {a1[-1].item():.4f} (Defense completely blind!)")
  ──────
  ## 16. Final Evaluation & Verdict

  ### Final Questions Answered:

  1. Is the codebase mathematically and scientifically sound?
  No. The codebase contains fatal mathematical flaws (Layer 1 cannot reject outliers for N = 10;
  ROC-AUC is misinterpreted), severe data leakage (target encoding using test set labels before
  splitting), and fundamental domain errors (aggregating absolute model weights rather than update
  deltas).
  2. Are the reported experimental results valid and reproducible?
  No. The reported results are derived from a pipeline contaminated by test label leakage,
  evaluated against crippled and misconfigured baselines, and based on single-seed runs.
  Furthermore, pyproject.toml lists non-existent future package versions, preventing standard
  reproduction.
  3. Is the implementation consistent with the paper's claims?
  No. The claimed gated cascade escalation is not implemented; advanced attacks A1–A3 are dead code;
  geographic partitioning is fictitious; and the paper's textual claim of Pareto optimality
  directly contradicts its own published ablation table.

  Recommendation: The submission requires a major mathematical correction, clean re-implementation
  of the federated update protocol, removal of data leakage, and a complete re-run of experiments
  before it can be considered scientifically defensible.