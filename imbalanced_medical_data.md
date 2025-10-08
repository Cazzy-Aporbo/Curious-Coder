# Class Imbalance in Medical Diagnosis
## When 99% Accuracy Means Clinical Failure

### Intent
Medical datasets often exhibit severe class imbalance (1:100 to 1:10,000 for rare diseases). Standard algorithms optimize overall accuracy, missing rare positive cases that matter most clinically. This document provides mathematically grounded approaches for handling imbalance while maintaining clinical utility.

### The Imbalance Problem: Mathematical Perspective

Given dataset D = {(x₁, y₁), ..., (xₙ, yₙ)} where:
- n₊ = |{i : yᵢ = 1}| (positive/disease cases)
- n₋ = |{i : yᵢ = 0}| (negative/healthy cases)
- Imbalance ratio: ρ = n₋/n₊ (often 100-10,000 in medical contexts)

**Standard loss minimization:**
```
L_standard = (1/n) Σᵢ ℓ(f(xᵢ), yᵢ)
```
Biases toward majority class since n₋ >> n₊

### Core Approaches: Theory & Trade-offs

#### 1. Cost-Sensitive Learning
**Formulation:** Weight errors by clinical importance

```
L_weighted = (1/n) Σᵢ wᵢ · ℓ(f(xᵢ), yᵢ)

where wᵢ = {
    C₊ if yᵢ = 1  (false negative cost)
    C₋ if yᵢ = 0  (false positive cost)
}
```

**Optimal weights (balanced):**
- C₊ = n/(2·n₊) = (1 + ρ)/2
- C₋ = n/(2·n₋) = (1 + ρ)/(2ρ)

**Clinical adjustment:**
```python
# Incorporate actual costs (e.g., missed diagnosis vs unnecessary test)
cost_matrix = [[0, cost_FP],      # TN, FP
               [cost_FN, 0]]       # FN, TP

# Example: Missing cancer (FN) 100x worse than false alarm (FP)
C₊ = cost_FN * (n/n₊)
C₋ = cost_FP * (n/n₋)
```

#### 2. Resampling Strategies

##### SMOTE (Synthetic Minority Over-sampling)
**Algorithm:** Generate synthetic samples along feature space lines

For each minority sample xᵢ:
1. Find k nearest minority neighbors
2. Select random neighbor x̃
3. Generate synthetic: x_new = xᵢ + λ(x̃ - xᵢ), λ ~ U(0,1)

**Mathematical justification:** Assumes class-conditional density is locally linear

**Boundary-focused variant (Borderline-SMOTE):**
```python
# Only synthesize near decision boundary
borderline = {x ∈ minority : |N_majority(x)| ≈ |N_minority(x)|}
```

##### Tomek Links Removal
**Definition:** (xᵢ, xⱼ) is Tomek link if:
1. yᵢ ≠ yⱼ (different classes)
2. ∄ xₖ : d(xᵢ, xₖ) < d(xᵢ, xⱼ) or d(xⱼ, xₖ) < d(xᵢ, xⱼ)

**Action:** Remove majority class member of each link
**Effect:** Cleans decision boundary, removes overlap

#### 3. Algorithm-Level Solutions

##### Threshold Optimization
Standard decision: ŷ = 𝟙[P(y=1|x) > 0.5]

**Optimal threshold for imbalanced data:**
```
τ* = arg max F_β(τ) = arg max [(1 + β²) · precision(τ) · recall(τ)] / 
                                [β² · precision(τ) + recall(τ)]
```

For medical diagnosis, typically β > 1 (favor recall over precision)

**Theoretical optimal (cost-sensitive):**
```
τ_bayes = C₋·P(y=0) / [C₋·P(y=0) + C₊·P(y=1)]
        = C₋·ρ / (C₋·ρ + C₊)
```

##### Ensemble Methods for Imbalance

**BalancedBagging:**
```
for b in 1..B:
    Sample n₊ from minority (with replacement)
    Sample n₊ from majority (without replacement)  # Undersample
    Train base_model_b on balanced sample
    
Prediction: ŷ = majority_vote({base_model_b})
```

**RUSBoost (Random UnderSampling + AdaBoost):**
```
Initialize: w₁(i) = 1/n
for t in 1..T:
    Undersample majority based on weights wₜ
    Train weak learner hₜ on balanced sample
    εₜ = Σᵢ wₜ(i)·𝟙[hₜ(xᵢ) ≠ yᵢ]
    αₜ = ½ log[(1-εₜ)/εₜ]
    wₜ₊₁(i) = wₜ(i)·exp(-αₜ·yᵢ·hₜ(xᵢ))
    
Final: H(x) = sign[Σₜ αₜ·hₜ(x)]
```

### Evaluation Metrics for Imbalanced Medical Data

#### Beyond Accuracy: Clinical Metrics

| Metric | Formula | Use Case | Limitation |
|--------|---------|----------|------------|
| **Sensitivity (Recall)** | TP/(TP+FN) | Screen for disease | Ignores FP rate |
| **Specificity** | TN/(TN+FP) | Confirm healthy | Ignores FN rate |
| **PPV (Precision)** | TP/(TP+FP) | Treatment decision | Prevalence-dependent |
| **NPV** | TN/(TN+FN) | Rule out disease | Prevalence-dependent |
| **F₂ Score** | 5·Prec·Rec/(4·Prec+Rec) | Favor sensitivity | Arbitrary β choice |
| **MCC** | (TP·TN-FP·FN)/√[(TP+FP)(TP+FN)(TN+FP)(TN+FN)] | Balanced measure | Hard to interpret |
| **AUPRC** | ∫ Precision(Recall) dRecall | Ranking quality | Focuses on positives |

**Prevalence adjustment for PPV/NPV:**
```
PPV_adjusted = (Sens·Prev) / (Sens·Prev + (1-Spec)·(1-Prev))
NPV_adjusted = (Spec·(1-Prev)) / ((1-Sens)·Prev + Spec·(1-Prev))
```

### Biological & Medical Failure Modes

| Scenario | Problem | Solution |
|----------|---------|----------|
| **Rare genetic variants** | 1:100,000 imbalance | Anomaly detection framework |
| **Cancer screening** | High FP unacceptable (anxiety, cost) | Two-stage: sensitive first, specific second |
| **ICU mortality** | Temporal imbalance (most survive initially) | Time-varying weights |
| **Drug side effects** | Multiple rare outcomes | Multi-label with hierarchical penalties |
| **Pediatric diseases** | Age-varying prevalence | Stratified sampling by age |

### Statistical Power Considerations

**Minimum sample size for rare events:**
```
n₊_min ≈ 10·p / min(sensitivity, specificity)
```
where p = number of predictors

**Rule of thumb:** Need ≥10 events per predictor for stable estimates

**Power calculation for imbalanced design:**
```python
from statsmodels.stats.power import tt_ind_solve_power

# Effect size adjusted for imbalance
cohen_d_adjusted = cohen_d * √(n₊·n₋/n²)
power = tt_ind_solve_power(effect_size=cohen_d_adjusted, 
                           ratio=ρ,
                           alpha=0.05)
```

### Implementation: Practical Pipeline

```python
def medical_imbalance_pipeline(X, y, clinical_costs):
    """
    Complete pipeline for imbalanced medical data
    """
    # 1. Assess imbalance
    imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    
    # 2. Clean boundaries
    if imbalance_ratio > 10:
        X_clean, y_clean = remove_tomek_links(X, y)
    
    # 3. Generate synthetic samples if extreme imbalance
    if imbalance_ratio > 100:
        X_balanced, y_balanced = SMOTE(
            X_clean, y_clean,
            sampling_strategy=0.1,  # Don't fully balance
            k_neighbors=min(5, (y == 1).sum() - 1)
        )
    
    # 4. Train with clinical weights
    weights = compute_clinical_weights(y_balanced, clinical_costs)
    model = train_weighted_model(X_balanced, y_balanced, weights)
    
    # 5. Optimize decision threshold
    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = optimize_threshold(
        y_val, val_probs,
        metric='f2',  # Favor sensitivity
        constraints={'specificity': 0.7}  # Minimum specificity
    )
    
    # 6. Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(
        model, method='isotonic', cv=3
    )
    
    return calibrated_model, threshold
```

### Cross-Validation for Imbalanced Data

**Stratified K-Fold:** Maintains class proportions
```python
# Ensures each fold has same imbalance ratio
skf = StratifiedKFold(n_splits=5, shuffle=True)

# For extreme imbalance, ensure minimum positive samples
min_samples_per_fold = 20
n_splits = min(5, (y == 1).sum() // min_samples_per_fold)
```

**Nested CV for Threshold Selection:**
```python
# Outer: Model selection
# Inner: Threshold optimization
for train_idx, test_idx in outer_cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    
    # Inner CV for threshold
    best_threshold = 0.5
    for val_train, val_test in inner_cv.split(X[train_idx], y[train_idx]):
        probs = model.predict_proba(X[val_test])[:, 1]
        thresholds = np.linspace(0.1, 0.9, 20)
        scores = [f2_score(y[val_test], probs > t) for t in thresholds]
        best_threshold = thresholds[np.argmax(scores)]
```

### Decision Framework

```
if imbalance_ratio < 3:
    use_standard_methods()
    
elif 3 <= imbalance_ratio < 10:
    use_class_weights()
    optimize_threshold()
    
elif 10 <= imbalance_ratio < 100:
    if n_positive > 100:
        use_SMOTE() + class_weights()
    else:
        use_BalancedBagging()
    carefully_optimize_threshold()
    
else:  # > 100:1
    if clinical_priority == "screening":
        use_anomaly_detection()
    else:
        use_cost_sensitive_ensemble()
        consider_two_stage_classifier()
```

### Clinical Integration Checklist

- [ ] Calculate Number Needed to Screen (NNS = 1/[Sens × Prev])
- [ ] Estimate false positive burden (FP rate × screening population)
- [ ] Validate on temporal holdout (not just random split)
- [ ] Check calibration (predicted vs actual probabilities)
- [ ] Subgroup analysis (age, sex, comorbidities)
- [ ] Decision curve analysis (net benefit across thresholds)
- [ ] Failure mode analysis (when does model fail dangerously?)

### References
- He, H. & Garcia, E.A. (2009). Learning from imbalanced data
- Wallace, B.C. et al. (2011). Class imbalance, redux (medical focus)
- Fernández, A. et al. (2018). Learning from imbalanced data sets