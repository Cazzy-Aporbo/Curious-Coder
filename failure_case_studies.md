# Failure Case Studies in Biological Data Analysis
## Learning from Catastrophic Mistakes in Real Studies

### Case Study 1: The Duke Cancer Scandal
**What Happened:** Genomic signatures for chemotherapy response were reversed

#### The Technical Failure
```python
# The original error (simplified)
def original_duke_analysis():
    """
    Actual bug that led to patient harm
    """
    # Gene expression data
    sensitive_samples = load_data("sensitive_to_drug.csv")
    resistant_samples = load_data("resistant_to_drug.csv")
    
    # THE BUG: Off-by-one error in sample indexing
    # Sensitive samples labeled as resistant and vice versa
    labels = ["resistant"] * len(sensitive_samples) + ["sensitive"] * len(resistant_samples)
    data = np.vstack([sensitive_samples, resistant_samples])
    
    # Build predictor - appears to work great!
    # (Because it learned the reverse pattern)
    model = train_model(data, labels)
    return model
```

#### Why It Wasn't Caught
1. **Validation on same reversed data** - looked perfect
2. **Biological plausibility** - genes made sense post-hoc
3. **P-values looked great** - reversal doesn't affect significance
4. **Complex pipeline** - error hidden in preprocessing

#### The Forensic Analysis
```python
def forensic_reproduction():
    """
    How Baggerly & Coombes detected the error
    """
    # 1. Tried to reproduce from methods → failed
    # 2. Reverse-engineered from results → found inconsistencies
    # 3. Checked gene lists → overlap where shouldn't be
    # 4. Traced sample names → found label swaps
    
    # Key insight: Reproducible ≠ Correct
```

#### Lessons for Your Pipeline
```python
class ValidationFramework:
    """
    Checks that would have caught this
    """
    def __init__(self):
        self.checks = []
    
    def biological_sense_check(self, genes, known_biology):
        """Do results align with prior knowledge?"""
        pass
    
    def label_permutation_test(self, data, labels, model_performance):
        """Is performance too good to be true?"""
        pass
    
    def independent_validation(self, model, external_data):
        """Test on data you didn't touch during development"""
        pass
    
    def data_provenance_tracking(self):
        """Track every transformation"""
        pass
```

### Case Study 2: The Microbiome Composition Fallacy

#### The Mathematical Trap
```python
def compositional_data_mistake():
    """
    Common error: Treating microbiome proportions as independent
    """
    # Relative abundances sum to 1
    microbiome_proportions = np.array([
        [0.3, 0.3, 0.4],  # Sample 1
        [0.5, 0.2, 0.3],  # Sample 2
    ])
    
    # WRONG: Standard correlation
    correlation_matrix = np.corrcoef(microbiome_proportions.T)
    # This will show spurious negative correlations!
    
    # Why: If species A increases, others must decrease (sum=1)
```

#### The Correct Approach
```python
def compositional_analysis():
    """
    Proper handling of compositional data
    """
    # 1. Log-ratio transformations
    def clr_transform(X):
        """Centered log-ratio"""
        geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
        return np.log(X / geometric_mean[:, None])
    
    # 2. Aitchison distance instead of Euclidean
    def aitchison_distance(x, y):
        clr_x = clr_transform(x)
        clr_y = clr_transform(y)
        return np.linalg.norm(clr_x - clr_y)
    
    # 3. Special methods for zeros (common in microbiome)
    def multiplicative_replacement(X, delta=1e-10):
        X_nonzero = X.copy()
        X_nonzero[X == 0] = delta
        X_nonzero = X_nonzero / X_nonzero.sum(axis=1)[:, None]
        return X_nonzero
```

### Case Study 3: Simpson's Paradox in Clinical Trials

#### The Paradox Manifested
```python
def simpsons_paradox_example():
    """
    Treatment appears harmful overall but beneficial in each subgroup
    Real example: Kidney stone treatment
    """
    # Overall results
    treatment_A = {"success": 273, "total": 350}  # 78% success
    treatment_B = {"success": 289, "total": 350}  # 83% success
    # B looks better!
    
    # But stratified by stone size:
    small_stones = {
        "A": {"success": 81, "total": 87},   # 93% success
        "B": {"success": 234, "total": 270},  # 87% success
    }
    large_stones = {
        "A": {"success": 192, "total": 263},  # 73% success
        "B": {"success": 55, "total": 80},    # 69% success
    }
    # A is better for both small AND large stones!
    
    # The resolution: Confounding by indication
    # Severe cases got treatment A more often
```

#### Detection and Resolution
```python
def detect_simpsons_paradox(data, treatment, outcome, potential_confounders):
    """
    Systematic check for paradox
    """
    results = {}
    
    # Overall effect
    overall_effect = compute_effect(data, treatment, outcome)
    results['overall'] = overall_effect
    
    # Stratified effects
    for confounder in potential_confounders:
        strata = data[confounder].unique()
        stratified_effects = []
        
        for stratum in strata:
            mask = data[confounder] == stratum
            effect = compute_effect(data[mask], treatment, outcome)
            stratified_effects.append(effect)
        
        # Check for reversal
        if all(e * overall_effect < 0 for e in stratified_effects):
            print(f"WARNING: Simpson's paradox detected with {confounder}")
            
        results[confounder] = stratified_effects
    
    return results
```

### Case Study 4: The Batch Effect Mask

#### Real Example: Cancer Subtype Discovery
```python
def batch_masked_biology():
    """
    TCGA pancreatic cancer: Batches perfectly aligned with subtypes
    Making it impossible to distinguish technical from biological
    """
    # The problematic design
    # Batch 1: All samples from Hospital A (mostly subtype 1)
    # Batch 2: All samples from Hospital B (mostly subtype 2)
    
    # Attempted analysis
    pca_results = PCA(expression_data)
    # PC1 separates batches... or subtypes? Can't tell!
    
    # The tragedy: Real biological differences discarded as batch effects
```

#### Prevention Strategy
```python
def balanced_batch_design(samples, biological_factors, n_batches):
    """
    Ensure biological factors are distributed across batches
    """
    from scipy.optimize import linear_sum_assignment
    
    # Create cost matrix: penalty for unbalanced batches
    cost_matrix = np.zeros((len(samples), n_batches))
    
    for i, sample in enumerate(samples):
        for b in range(n_batches):
            # Cost = imbalance created by adding this sample to batch b
            current_batch_composition = get_batch_composition(b)
            imbalance = calculate_imbalance(
                current_batch_composition, 
                sample.biological_factors
            )
            cost_matrix[i, b] = imbalance
    
    # Optimal assignment
    sample_idx, batch_idx = linear_sum_assignment(cost_matrix)
    
    return sample_idx, batch_idx
```

### Case Study 5: P-Hacking in Multiple Testing

#### The Garden of Forking Paths
```python
def multiple_testing_failure():
    """
    Real study: 1000 brain regions × 20 behaviors × 5 statistics = 100,000 tests
    "Found" significant associations that don't replicate
    """
    # Simulate the problem
    n_regions = 1000
    n_behaviors = 20
    n_subjects = 50
    
    # All null data - no real associations
    brain_data = np.random.randn(n_subjects, n_regions)
    behavior_data = np.random.randn(n_subjects, n_behaviors)
    
    # Test everything
    p_values = []
    for i in range(n_regions):
        for j in range(n_behaviors):
            _, p = stats.pearsonr(brain_data[:, i], behavior_data[:, j])
            p_values.append(p)
    
    # "Significant" findings at p < 0.05
    significant = sum(p < 0.05 for p in p_values)
    print(f"Found {significant} 'significant' associations (expect ~5000 by chance)")
    
    # Bonferroni would require p < 0.05/100000 = 5e-7
    # FDR is more reasonable but still conservative
```

#### Proper Approaches
```python
def hierarchical_fdr_control():
    """
    Tree-based FDR: Test hypotheses in meaningful groups
    """
    # Level 1: Brain lobes (5 tests)
    # Level 2: Regions within significant lobes (50 tests)  
    # Level 3: Voxels within significant regions (1000 tests)
    
    # This preserves power while controlling error
    pass

def stability_selection_alternative():
    """
    Require findings to be stable across subsamples
    """
    stable_features = []
    
    for _ in range(100):
        subsample = np.random.choice(n_subjects, n_subjects//2)
        selected = run_analysis(data[subsample])
        stable_features.append(selected)
    
    # Only keep features selected >60% of time
    stability_scores = compute_stability(stable_features)
    return stability_scores > 0.6
```

### Case Study 6: The Survival Bias Trap

#### Conditioning on the Future
```python
def immortal_time_bias():
    """
    Classic error: Lung cancer patients who got treatment lived longer
    But: You must survive long enough to get treatment!
    """
    # WRONG ANALYSIS
    def naive_survival_analysis(patients):
        treated = patients[patients.got_treatment == True]
        untreated = patients[patients.got_treatment == False]
        
        # Treated group has guaranteed survival until treatment start
        # This creates artificial survival advantage
        km_treated = KaplanMeier(treated.survival_time)
        km_untreated = KaplanMeier(untreated.survival_time)
        # Treated appears to live longer (but it's bias!)
    
    # CORRECT: Time-dependent covariate
    def proper_survival_analysis(patients):
        # Treatment status changes over time
        # Person contributes to "untreated" until treatment starts
        # Then contributes to "treated" group
        
        extended_data = create_counting_process_format(patients)
        cox_model = CoxPH()
        cox_model.fit(
            extended_data,
            duration_col='stop',
            start_col='start', 
            event_col='event',
            show_progress=True
        )
```

### Meta-Analysis: Common Patterns in Failures

```python
def failure_pattern_analysis():
    """
    Extract common themes from disasters
    """
    patterns = {
        "data_leakage": [
            "Using test data in preprocessing",
            "Feature selection on full dataset",
            "Imputation before splitting"
        ],
        "wrong_assumptions": [
            "Independence when correlated",
            "Gaussian when heavy-tailed",
            "Missing at random when not"
        ],
        "confounding": [
            "Batch-biology alignment",
            "Temporal trends",
            "Selection bias"
        ],
        "multiple_testing": [
            "Cherry-picking significant results",
            "Testing until significant",
            "Post-hoc hypothesis generation"
        ],
        "overfitting": [
            "Too many parameters",
            "Evaluated on training data",
            "Optimizing for wrong metric"
        ]
    }
    
    return patterns
```

### Creating Your Own Failure Catalog

For each project, document:

1. **What you tried that failed**
   - Hypothesis
   - Method
   - Why it seemed reasonable
   - How it failed
   - Root cause

2. **Near misses**
   - Errors caught late
   - Results that seemed too good
   - Assumptions barely held

3. **Uncertainty points**
   - Decisions you're unsure about
   - Alternative analyses considered
   - Results sensitive to choices

### The Pre-Mortem Exercise

Before running any analysis:
```python
def pre_mortem(planned_analysis):
    """
    Imagine your analysis failed spectacularly.
    Work backwards: what went wrong?
    """
    failure_modes = []
    
    # Data quality failures
    if "missing_data" in planned_analysis:
        failure_modes.append("Imputation introduced bias")
    
    # Statistical failures
    if "multiple_testing" in planned_analysis:
        failure_modes.append("False discoveries from many tests")
    
    # Biological failures
    if "batch_correction" in planned_analysis:
        failure_modes.append("Removed real biological variation")
    
    # For each failure mode, add a check
    for failure in failure_modes:
        add_diagnostic_check(failure)
    
    return failure_modes
```

### Learning from Others' Mistakes

Resources to study:
1. **Retracted papers** - Read the retraction notices
2. **Failed replications** - Understand why they failed
3. **Correction notices** - See what was fixed
4. **Negative results** - Published "failed" studies
5. **Methods papers** - Often motivated by common errors

### Your Personal Failure Journal

Template:
```markdown
## Date: [DATE]
## Analysis: [WHAT I WAS TRYING]

### What Went Wrong
[Specific description]

### Why I Didn't Catch It
[Blind spots, assumptions]

### How I Found It
[The revealing moment]

### Mathematical Root Cause
[The actual mathematical/statistical error]

### Biological Consequence
[What this means for interpretation]

### Prevention for Next Time
[Specific check or method change]

### Similar Failures to Watch For
[Related problems this suggests]
```

Remember: Every failure is a discovered edge case that makes your methods more robust.