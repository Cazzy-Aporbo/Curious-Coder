# Principal Component Analysis vs Independent Component Analysis
## When Orthogonality Matters Less Than Independence

### Intent
PCA finds orthogonal directions of maximum variance. ICA finds maximally independent components. In biological systems where signals mix non-orthogonally (EEG, fMRI, gene expression), independence often matters more than variance.

### Mathematical Formulation

#### PCA: Variance Maximization
**Objective:** Find orthogonal projections maximizing variance

Given data matrix **X** ∈ ℝ^(n×p) with n samples, p features:

```
maximize    Var(Xw) = w^T Cov(X) w
subject to  ||w||₂ = 1
            w_i ⊥ w_j for i ≠ j
```

**Solution:** Eigendecomposition of covariance matrix
- C = X^T X / (n-1)
- C = VΛV^T where V contains eigenvectors, Λ contains eigenvalues
- Principal components: Y = XV

#### ICA: Independence Maximization  
**Objective:** Find components with maximum statistical independence

Assume: **X** = **AS** where A is mixing matrix, S contains independent sources

```
maximize    J(W) = Σᵢ H(yᵢ) - H(y)
where       y = Wx (unmixed signals)
            H(·) is differential entropy
```

**Practical surrogate:** Maximize non-Gaussianity (via negentropy or kurtosis)
```
J(y) ≈ [E{G(y)} - E{G(v)}]²
```
where G is nonlinear function (e.g., G(u) = log cosh(u)), v ~ N(0,1)

### Critical Assumptions

| Aspect | PCA | ICA |
|--------|-----|-----|
| **Statistical** | Second-order sufficiency (covariance captures all structure) | Higher-order statistics matter |
| **Signal model** | X = signal + Gaussian noise | X = AS, sources are independent |
| **Identifiability** | Unique up to sign | Unique up to permutation & scaling |
| **Gaussianity** | Works for Gaussian data | Fails if >1 source is Gaussian |
| **Orthogonality** | Components orthogonal by construction | Components generally non-orthogonal |

### When to Prefer Each

#### Use PCA when:
- **Dimensionality reduction** is primary goal (preserving global variance)
- **Gaussian assumption** reasonable (thermal noise, measurement error)
- **Interpretability** through variance explained is valuable
- **Computational efficiency** critical (O(min(n²p, np²)) vs iterative)

#### Use ICA when:
- **Source separation** needed (unmixing signals)
- **Non-Gaussian** structure present (super/sub-Gaussian distributions)
- **Independence** more meaningful than orthogonality
- **Biological mixing** suspected (neural, metabolic, genetic pathways)

### Biological Applications & Failure Modes

| Application | PCA Success | PCA Failure | ICA Success |
|-------------|-------------|-------------|-------------|
| **EEG/MEG** | Global trends (sleep stages) | Source localization | Artifact removal (eye blinks, heartbeat) |
| **fMRI** | Motion correction | Network identification | Default mode/task networks |
| **Gene expression** | Batch effects | Pathway decomposition | Cell type deconvolution |
| **Mass spectrometry** | Technical variance | Metabolite identification | Isotope pattern separation |

### Diagnostics & Validation

#### PCA Diagnostics:
```python
# 1. Variance explained
var_explained = eigenvalues / eigenvalues.sum()
cumsum_var = np.cumsum(var_explained)

# 2. Reconstruction error
X_reconstructed = Y[:, :k] @ V[:k, :].T  # k components
mse = ||X - X_reconstructed||²_F / (n*p)

# 3. Component stability (bootstrap)
stability = std(V_bootstrap) / |V_original|
```

#### ICA Diagnostics:
```python
# 1. Independence test (mutual information)
MI(y_i, y_j) = ∫∫ p(y_i, y_j) log[p(y_i, y_j)/(p(y_i)p(y_j))] dy_i dy_j

# 2. Non-Gaussianity (kurtosis)
kurt(y) = E[(y - μ)⁴]/σ⁴ - 3  # Should be ≠ 0

# 3. Convergence (gradient norm)
||∇J(W)||₂ < ε  # Typically ε = 10⁻⁵
```

### Implementation Considerations

#### Preprocessing Pipeline:
1. **Centering:** Always required (subtract mean)
2. **Scaling:** 
   - PCA: Optional (changes relative importance)
   - ICA: Critical (prevents scale dominance)
3. **Whitening:**
   - PCA: Not needed (builds it in)
   - ICA: Often helps convergence (decorrelates first)

#### Computational Complexity:
- **PCA:** O(min(n²p, np²)) for full decomposition
- **ICA FastICA:** O(npc) where c is iterations (typically 10-100)
- **ICA Infomax:** O(n²p) per iteration

### The Biology-Driven Choice

In biological systems, independence often reflects true underlying processes better than orthogonality:

1. **Neural signals** mix through volume conduction (non-orthogonal but independent sources)
2. **Gene regulatory networks** operate independently but share transcription machinery
3. **Metabolic pathways** run in parallel with shared cofactors

**Key insight:** When biological sources are mixed linearly but operate independently, ICA recovers meaningful components even when PCA shows uninterpretable variance spread across many components.

### Neighbor Methods to Consider

- **Sparse PCA:** When seeking interpretable, sparse loadings
- **Kernel PCA:** For nonlinear manifolds  
- **Non-negative Matrix Factorization:** When sources are non-negative (concentrations, counts)
- **Factor Analysis:** When explicit noise model needed
- **Dictionary Learning:** Over-complete basis for sparse coding

### Final Decision Framework

```
if task == "compression":
    if data.is_gaussian():
        use_PCA()
    else:
        consider_autoencoder()
        
elif task == "source_separation":
    if sources.are_independent() and not all_gaussian():
        use_ICA()
    elif sources.are_sparse():
        use_sparse_coding()
    elif sources.are_nonnegative():
        use_NMF()
        
elif task == "visualization":
    if preserve_global_structure:
        use_PCA()
    elif preserve_local_structure:
        use_tSNE() or UMAP()
```

### References for Deeper Study
- Hyvärinen, A. & Oja, E. (2000). Independent component analysis: algorithms and applications
- Beckmann, C.F. & Smith, S.M. (2004). Probabilistic ICA for functional magnetic resonance imaging
- Brunet, J.P. et al. (2004). Metagenes and molecular pattern discovery using matrix factorization