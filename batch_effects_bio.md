# Batch Effects in High-Throughput Biological Data
## When Technical Variation Drowns Biological Signal

### Intent
Batch effects—systematic technical variation between measurement groups—can dominate biological signals in genomics, proteomics, and metabolomics. This document provides mathematical frameworks for detecting, quantifying, and removing batch effects while preserving biological variation.

### The Batch Effect Problem

Given measurements **Y** ∈ ℝ^(g×n) (g genes/features, n samples):

```
Y_ij = μ_i + X_j^T β_i + γ_ib(j) + δ_ik(j) + ε_ij

where:
- μ_i: baseline expression for feature i
- X_j: biological covariates (disease, age, sex)
- β_i: biological effects
- γ_i: additive batch effect
- δ_i: multiplicative batch effect
- b(j): batch indicator for sample j
- k(j): other technical factors (RNA quality, processing time)
- ε_ij: random noise
```

**The challenge:** Separate γ and δ from β when batches correlate with biology.

### Mathematical Frameworks for Batch Correction

#### 1. ComBat: Empirical Bayes Batch Correction

**Model:** Location-scale adjustment with empirical Bayes

```
Y*_ij = (Y_ij - α̂_i - X_j^T β̂_i) / σ̂_ib + X_j^T β̂_i

Parametric priors:
γ_i ~ N(γ̄_b, τ²_b)  # Batch mean shifts
δ²_i ~ InverseGamma(λ_b, θ_b)  # Batch variances
```

**Empirical Bayes estimates:**
```python
# Hyperparameter estimation
γ̄_b = mean(γ̂_i) across genes in batch b
τ²_b = var(γ̂_i)

# Shrinkage estimates
γ*_i = (n_b·τ²_b·γ̂_i + σ²_i·γ̄_b) / (n_b·τ²_b + σ²_i)
δ*_i = (θ_b + 0.5·Σ_j∈b (Y_ij - α̂_i - X_j^T β̂_i - γ*_i)²) / (λ_b + n_b/2)
```

**Protection of biological variation:**
- Design matrix **X** preserves biological covariates
- Only removes variation orthogonal to **X**

#### 2. RUV (Remove Unwanted Variation)

**Approach:** Use negative control genes or samples

**RUV-2 (control genes):**
```
Y = Xβ + Wα + ε

where W estimated from control genes:
Y_control = Wα_control + ε_control
Ŵ = top k singular vectors of Y_control
```

**RUV-4 (replicate samples):**
```
# Use technical replicates to estimate W
Y_replicates = μ + Wα + ε
W = eigenvectors of Cov(Y_replicates - μ̂)
```

**Mathematical guarantee:** If controls truly unaffected by biology, W ⊥ X

#### 3. Surrogate Variable Analysis (SVA)

**Philosophy:** Discover hidden factors affecting many genes

**Algorithm:**
1. **Estimate biological signal:**
   ```
   H₀: Y = Xβ + ε
   H₁: Y = Xβ + Γν + ε  (Γ = surrogate variables)
   ```

2. **Iterative discovery:**
   ```python
   for k in 1..K:
       # Residuals after removing known effects
       R = Y - X(X^TX)^(-1)X^TY
       
       # Find strongest residual pattern
       u, d, v = SVD(R)
       sv_k = v[:, 0]  # First right singular vector
       
       # Test significance
       p_values = [permutation_test(gene_i, sv_k) for gene_i]
       if significant_genes < threshold:
           break
       
       # Add to model
       Γ = [Γ, sv_k]
   ```

3. **Final correction:**
   ```
   Y_corrected = Y - Γ(Γ^TΓ)^(-1)Γ^TY
   ```

### Batch Effect Detection & Quantification

#### 1. Principal Variance Component Analysis (PVCA)

Partition total variance into sources:
```
Var(Y) = Σ_factors σ²_factor + σ²_residual

# Weighted average of variance explained
PVCA_batch = (σ²_batch / Var(Y)) × PC_weight
```

#### 2. Guided PCA (gPCA)

Test if batches drive principal components:
```
δ = Σ_k λ_k · |cor(PC_k, batch)|²

p-value via permutation of batch labels
```

#### 3. Silhouette Coefficient for Batch Mixing

```python
def batch_ASW(data, batch_labels, bio_labels):
    """Average silhouette width for batch mixing within biological groups"""
    ASW_per_bio = []
    for bio_group in unique(bio_labels):
        mask = bio_labels == bio_group
        data_bio = data[mask]
        batch_bio = batch_labels[mask]
        
        # Silhouette: (between-batch - within-batch) / max
        s = silhouette_score(data_bio, batch_bio)
        ASW_per_bio.append(1 - abs(s))  # 1 = perfect mixing
    
    return mean(ASW_per_bio)
```

### Biological Contexts & Failure Modes

| Data Type | Common Batch Sources | Typical Magnitude | Correction Challenge |
|-----------|---------------------|-------------------|---------------------|
| **RNA-seq** | Library prep, sequencing run | 20-50% variance | GC content, length bias |
| **Proteomics** | MS run, digestion batch | 30-60% variance | Missing values, dynamic range |
| **Microarray** | Chip, hybridization date | 10-30% variance | Probe effects, saturation |
| **Single-cell** | Capture date, cell viability | 40-70% variance | Zero inflation, cell cycle |
| **Metabolomics** | Extraction, instrument drift | 20-40% variance | Matrix effects, retention time |
| **ATAC-seq** | Transposition efficiency | 25-45% variance | Fragment size, ploidy |

### Design Strategies to Minimize Batch Effects

#### 1. Randomized Block Design
```
Optimal allocation minimizes E[(β̂ - β)²]:

For K batches, N samples, G groups:
n_kg = N/(K·G) samples from group g in batch k

Constraint: n_kg ≥ 2 for variance estimation
```

#### 2. Reference Sample Distribution
```python
def distribute_references(n_batches, n_refs_per_batch=3):
    """Distribute technical replicates across batches"""
    # Same biological sample in each batch
    # Enables direct batch effect estimation
    
    reference_matrix = np.zeros((n_batches, n_refs_per_batch))
    for batch in range(n_batches):
        # Include at beginning, middle, end to detect drift
        positions = [0, batch_size//2, batch_size-1]
        reference_matrix[batch] = positions
    
    return reference_matrix
```

### Method Selection Framework

```python
def select_batch_correction(data, metadata):
    """Decision tree for batch correction method"""
    
    n_batches = metadata['batch'].nunique()
    n_samples_per_batch = metadata.groupby('batch').size()
    has_replicates = check_technical_replicates(metadata)
    has_controls = check_negative_controls(data)
    batch_bio_confounded = cramers_v(metadata['batch'], metadata['condition']) > 0.3
    
    if n_batches == 1:
        return "No batch correction needed"
    
    elif batch_bio_confounded:
        if has_replicates:
            return "RUV-4 (uses replicates, handles confounding)"
        elif has_controls:
            return "RUV-2 (uses control genes)"
        else:
            return "SVA (discovers latent factors) + careful validation"
    
    elif n_samples_per_batch.min() < 3:
        return "Mean-centering only (insufficient samples for variance)"
    
    elif has_controls:
        return "RUV-2 or RUV-3"
    
    else:
        return "ComBat (robust empirical Bayes)"
```

### Validation of Batch Correction

#### 1. Preservation of Biological Signal
```python
def bio_preservation_score(Y_original, Y_corrected, bio_labels):
    """Measure preservation of biological differences"""
    
    # Pairwise distances between biological groups
    dist_original = pairwise_distances(
        group_means(Y_original, bio_labels)
    )
    dist_corrected = pairwise_distances(
        group_means(Y_corrected, bio_labels)
    )
    
    # Correlation of distance matrices
    preservation = spearmanr(
        dist_original.flatten(),
        dist_corrected.flatten()
    )
    
    return preservation.correlation
```

#### 2. Removal of Technical Variation
```python
def batch_removal_score(Y_corrected, batch_labels):
    """Proportion of variance NOT explained by batch"""
    
    # Linear model: Y ~ batch
    _, _, r_value, _, _ = stats.linregress(
        batch_labels,
        Y_corrected.mean(axis=0)
    )
    
    return 1 - r_value**2  # Want this close to 1
```

### Common Pitfalls & Solutions

| Pitfall | Consequence | Solution |
|---------|------------|----------|
| **Overcorrection** | Removes real biology | Preserve known covariates; validate with holdout |
| **Batch-bio confounding** | Can't separate effects | Use time-series or dose-response to break confounding |
| **Small batches** | Unstable variance estimates | Pool variance; increase regularization |
| **Non-linear effects** | Linear methods fail | Deep learning methods (e.g., scVI for single-cell) |
| **Missing values** | Breaks matrix operations | Impute within batch; use method-specific handling |
| **Outlier samples** | Dominate correction | Robust methods; outlier removal pre-correction |

### Advanced Topics

#### 1. Multi-Omic Integration with Batch Effects
```python
# MOFA+ approach: Factor analysis aware of batches
Y_modality_m = W_m × Z + batch_effects_m + ε_m

# Different batches per modality
# Shared factors Z across modalities
```

#### 2. Longitudinal Batch Effects
```
Y_it = μ_i + β_i·t + γ_i·batch(t) + subject_i + ε_it

# Mixed effects model separating:
# - Time trends (β_i·t)
# - Batch effects (γ_i)
# - Subject effects (random)
```

#### 3. Spatial Batch Effects (Tissue Sections)
```python
# Gaussian process for spatial smoothing
Y_ij = f(x_i, y_i) + batch_j + ε_ij

where f ~ GP(0, K_spatial)
K_spatial(s, s') = σ² exp(-||s - s'||²/2ℓ²)
```

### Practical Implementation Checklist

**Before Experiment:**
- [ ] Randomize samples across batches
- [ ] Include technical replicates (≥3 per batch)
- [ ] Include biological replicates across batches
- [ ] Record all technical variables (time, operator, reagent lot)
- [ ] Plan for 20-30% extra samples for batch effects

**During Analysis:**
- [ ] Visualize batches (PCA colored by batch)
- [ ] Quantify batch contribution (PVCA)
- [ ] Test for batch-biology confounding
- [ ] Apply correction method
- [ ] Validate on holdout samples
- [ ] Document correction parameters

**Reporting:**
- [ ] State batch correction method and parameters
- [ ] Show before/after PCA plots
- [ ] Report variance explained by batch
- [ ] Provide batch-corrected data
- [ ] Include uncorrected data for reproducibility

### Code Template
```python
def comprehensive_batch_correction(data, metadata, method='auto'):
    """Full pipeline for batch effect handling"""
    
    # 1. Detection
    batch_variance = PVCA(data, metadata)
    print(f"Batch explains {batch_variance:.1%} of variance")
    
    if batch_variance < 0.05:
        return data, "No significant batch effect"
    
    # 2. Method selection
    if method == 'auto':
        method = select_batch_correction(data, metadata)
    
    # 3. Correction
    if method == 'ComBat':
        data_corrected = ComBat(data, metadata['batch'], 
                              mod=metadata[['age', 'sex', 'condition']])
    elif method == 'SVA':
        surrogates = SVA(data, metadata[['condition']], metadata['batch'])
        data_corrected = remove_surrogates(data, surrogates)
    # ... other methods
    
    # 4. Validation
    bio_preserved = bio_preservation_score(data, data_corrected, metadata['condition'])
    batch_removed = batch_removal_score(data_corrected, metadata['batch'])
    
    print(f"Biological signal preserved: {bio_preserved:.1%}")
    print(f"Batch effect removed: {batch_removed:.1%}")
    
    return data_corrected, method
```

### References
- Johnson, W.E. et al. (2007). Adjusting batch effects in microarray data using empirical Bayes methods
- Leek, J.T. & Storey, J.D. (2007). Capturing heterogeneity in gene expression studies by surrogate variable analysis
- Gagnon-Bartsch, J.A. & Speed, T.P. (2012). Using control genes to correct for unwanted variation
- Haghverdi, L. et al. (2018). Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors