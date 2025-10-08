# Causal Inference in Biological Systems
## Moving Beyond Correlation in Complex Living Systems

### Intent
Biological systems are inherently causal - genes regulate proteins, proteins catalyze reactions, drugs affect pathways. Yet most analyses only capture correlations. This document provides rigorous frameworks for inferring causality from observational and experimental biological data, where randomized controls are often impossible or unethical.

### The Fundamental Problem of Causal Inference

**Potential Outcomes Framework (Rubin Causal Model):**

For unit i with treatment T ∈ {0,1}:
- Y_i(1) = potential outcome if treated
- Y_i(0) = potential outcome if untreated
- Causal effect: τ_i = Y_i(1) - Y_i(0)

**The fundamental problem:** We only observe Y_i(T_i), never both potential outcomes.

**Average Treatment Effect (ATE):**
```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

**The challenge in biology:** Confounding is ubiquitous
```
E[Y|T=1] - E[Y|T=0] = ATE + Bias
where Bias = E[Y(0)|T=1] - E[Y(0)|T=0]
```

### Causal Graphs and Biological Networks

#### Directed Acyclic Graphs (DAGs) in Biology

```python
def biological_dag_example():
    """
    Gene → mRNA → Protein → Phenotype
      ↓                ↑
    Epigenetics → Transcription Factor
    """
    
    # Formal representation
    nodes = ['Gene', 'Epigenetics', 'mRNA', 'TF', 'Protein', 'Phenotype']
    edges = [
        ('Gene', 'mRNA'),
        ('Gene', 'Epigenetics'),
        ('Epigenetics', 'TF'),
        ('TF', 'mRNA'),
        ('mRNA', 'Protein'),
        ('Protein', 'Phenotype')
    ]
    
    # D-separation rules determine conditional independence
    # Gene ⊥ Phenotype | Protein (blocked by conditioning)
    # Gene ⊥ TF | Epigenetics (blocked by conditioning)
```

#### Identifying Causal Effects from DAGs

**Backdoor Criterion:** A set of variables Z satisfies the backdoor criterion if:
1. No node in Z is a descendant of treatment T
2. Z blocks all backdoor paths from T to Y

```python
def check_backdoor_criterion(dag, treatment, outcome, adjustment_set):
    """
    Verify if adjustment set satisfies backdoor criterion
    """
    # Check no descendants of treatment
    descendants = find_descendants(dag, treatment)
    if any(node in descendants for node in adjustment_set):
        return False
    
    # Check blocks all backdoor paths
    backdoor_paths = find_backdoor_paths(dag, treatment, outcome)
    for path in backdoor_paths:
        if not is_blocked(path, adjustment_set):
            return False
    
    return True
```

### Methods for Causal Inference

#### 1. Instrumental Variables (IV) for Mendelian Randomization

**Setup:** Gene variant Z affects outcome Y only through exposure X

```
Z → X → Y
    ↑   ↑
    U ──┘  (Unobserved confounder)
```

**IV Assumptions:**
1. Relevance: Z strongly associated with X
2. Exclusion: Z affects Y only through X
3. Independence: Z independent of unmeasured confounders

**Two-Stage Least Squares (2SLS):**
```python
def mendelian_randomization_2sls(genotype, exposure, outcome, covariates=None):
    """
    Estimate causal effect of exposure on outcome using genetic IV
    """
    from statsmodels.sandbox.regression.gmm import IV2SLS
    
    # Stage 1: Regress exposure on instrument
    # X = γZ + ε
    
    # Stage 2: Regress outcome on predicted exposure
    # Y = βX̂ + ε
    
    # Combined estimation
    if covariates is not None:
        exog = np.column_stack([exposure, covariates])
        instruments = np.column_stack([genotype, covariates])
    else:
        exog = exposure.reshape(-1, 1)
        instruments = genotype.reshape(-1, 1)
    
    model = IV2SLS(outcome, exog, instruments)
    results = model.fit()
    
    # Weak instrument test
    f_stat = results.first_stage_f_statistic
    if f_stat < 10:
        print("Warning: Weak instrument (F={:.2f} < 10)".format(f_stat))
    
    # Causal effect estimate
    causal_effect = results.params[0]
    se = results.bse[0]
    
    return causal_effect, se, f_stat
```

#### 2. Propensity Score Methods

**Propensity Score:** e(X) = P(T=1|X)

**Key Theorem (Rosenbaum & Rubin):** If treatment assignment is ignorable given X, then it's ignorable given e(X).

```python
def propensity_score_matching(X, treatment, outcome, caliper=0.1):
    """
    Estimate causal effect via propensity score matching
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    
    # Step 1: Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, treatment)
    propensity_scores = ps_model.predict_proba(X)[:, 1]
    
    # Check overlap (common support)
    ps_treated = propensity_scores[treatment == 1]
    ps_control = propensity_scores[treatment == 0]
    
    overlap = (ps_control.min() <= ps_treated.max()) and \
              (ps_treated.min() <= ps_control.max())
    
    if not overlap:
        print("Warning: No overlap in propensity scores!")
    
    # Step 2: Match treated to control units
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    
    # For each treated unit, find nearest control
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(propensity_scores[control_idx].reshape(-1, 1))
    
    matched_controls = []
    for t_idx in treated_idx:
        ps_t = propensity_scores[t_idx].reshape(1, -1)
        dist, c_idx = nn.kneighbors(ps_t)
        
        # Caliper: Only match if close enough
        if dist[0][0] <= caliper:
            matched_controls.append(control_idx[c_idx[0][0]])
        else:
            matched_controls.append(None)
    
    # Step 3: Estimate treatment effect
    ate = 0
    n_matched = 0
    
    for t_idx, c_idx in zip(treated_idx, matched_controls):
        if c_idx is not None:
            ate += outcome[t_idx] - outcome[c_idx]
            n_matched += 1
    
    ate = ate / n_matched if n_matched > 0 else np.nan
    
    print(f"Matched {n_matched}/{len(treated_idx)} treated units")
    
    return ate, propensity_scores
```

#### 3. Difference-in-Differences (DiD) for Longitudinal Data

**Setup:** Treatment affects some units at time t*

```
Y_it = α + β(Treated_i × Post_t) + γ_i + δ_t + ε_it

where:
- γ_i: unit fixed effects
- δ_t: time fixed effects
- β: causal effect
```

```python
def difference_in_differences(data, unit_col, time_col, outcome_col, 
                            treatment_col, treatment_time):
    """
    Estimate causal effect using DiD
    """
    # Create post-treatment indicator
    data['post'] = (data[time_col] >= treatment_time).astype(int)
    
    # Create DiD interaction term
    data['did'] = data[treatment_col] * data['post']
    
    # Fixed effects regression
    import statsmodels.formula.api as smf
    
    formula = f'{outcome_col} ~ did + C({unit_col}) + C({time_col})'
    model = smf.ols(formula, data=data)
    results = model.fit()
    
    # Parallel trends test (pre-treatment)
    pre_data = data[data[time_col] < treatment_time]
    trends_formula = f'{outcome_col} ~ {treatment_col}*{time_col} + C({unit_col})'
    trends_model = smf.ols(trends_formula, data=pre_data)
    trends_results = trends_model.fit()
    
    interaction_p = trends_results.pvalues[f'{treatment_col}:{time_col}']
    
    if interaction_p < 0.05:
        print(f"Warning: Parallel trends violated (p={interaction_p:.3f})")
    
    return results.params['did'], results.bse['did'], interaction_p
```

#### 4. Regression Discontinuity Design (RDD)

**Setup:** Treatment assigned based on threshold of running variable

```python
def regression_discontinuity(running_var, outcome, threshold, bandwidth=None):
    """
    Estimate causal effect at discontinuity
    
    Treatment: T = 1 if running_var >= threshold else 0
    """
    treatment = (running_var >= threshold).astype(int)
    
    if bandwidth is None:
        # Optimal bandwidth (Imbens-Kalyanaraman)
        bandwidth = optimal_bandwidth_ik(running_var, outcome, threshold)
    
    # Local linear regression on each side
    mask = np.abs(running_var - threshold) <= bandwidth
    X_local = running_var[mask]
    Y_local = outcome[mask]
    T_local = treatment[mask]
    
    # Center running variable at threshold
    X_centered = X_local - threshold
    
    # Estimate with interaction
    # Y = α + τT + β₁X + β₂TX + ε
    from sklearn.linear_model import LinearRegression
    
    features = np.column_stack([
        T_local,
        X_centered,
        T_local * X_centered
    ])
    
    model = LinearRegression()
    model.fit(features, Y_local)
    
    # Causal effect at threshold
    tau = model.coef_[0]
    
    # Manipulation test (McCrary density test)
    density_left = np.sum(running_var < threshold) / len(running_var)
    density_right = np.sum(running_var >= threshold) / len(running_var)
    manipulation_ratio = density_right / density_left
    
    if manipulation_ratio > 1.5 or manipulation_ratio < 0.67:
        print(f"Warning: Potential manipulation (density ratio={manipulation_ratio:.2f})")
    
    return tau, bandwidth, manipulation_ratio
```

### Causal Discovery from Data

#### PC Algorithm (Peter-Clark)

```python
def pc_algorithm(data, alpha=0.05):
    """
    Discover causal structure from observational data
    """
    n_vars = data.shape[1]
    
    # Step 1: Start with complete undirected graph
    skeleton = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sep_sets = {}
    
    # Step 2: Remove edges via conditional independence tests
    for level in range(n_vars - 1):
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton[i, j] == 0:
                    continue
                
                # Find potential conditioning sets
                neighbors = get_neighbors(skeleton, i, j)
                
                for cond_set in combinations(neighbors, level):
                    # Conditional independence test
                    p_value = conditional_independence_test(
                        data[:, i], data[:, j], 
                        data[:, list(cond_set)]
                    )
                    
                    if p_value > alpha:
                        skeleton[i, j] = skeleton[j, i] = 0
                        sep_sets[(i, j)] = cond_set
                        break
    
    # Step 3: Orient edges using v-structures
    dag = orient_v_structures(skeleton, sep_sets)
    
    # Step 4: Apply orientation rules
    dag = apply_meek_rules(dag)
    
    return dag
```

### Biological Applications and Pitfalls

| Method | Biological Application | Key Assumptions | Common Violations |
|--------|----------------------|-----------------|-------------------|
| **Mendelian Randomization** | Gene → Disease causation | No pleiotropy | Genes affect multiple traits |
| **Propensity Scores** | Treatment effect in EHR | No unmeasured confounding | Hidden disease severity |
| **DiD** | Policy effects on health | Parallel trends | Differential health trends |
| **RDD** | Age/dose thresholds | No manipulation | Patients gaming thresholds |
| **IV** | Drug compliance effects | Exclusion restriction | Direct placebo effects |

### Time-Varying Treatments and Confounders

```python
def marginal_structural_model(data, treatment_history, outcome, 
                            time_varying_confounders):
    """
    Handle time-varying confounding via inverse probability weighting
    
    L(t) → A(t) → Y(t+1)
     ↑      ↓
     └──────┘
    """
    n_times = len(treatment_history[0])
    n_units = len(treatment_history)
    
    # Calculate stabilized weights
    weights = np.ones(n_units)
    
    for t in range(n_times):
        # Denominator: P(A(t) | A_bar(t-1), L_bar(t))
        X_denom = build_history_features(
            treatment_history, time_varying_confounders, t
        )
        ps_denom = estimate_propensity(
            X_denom, treatment_history[:, t]
        )
        
        # Numerator: P(A(t) | A_bar(t-1))
        X_num = build_history_features(
            treatment_history, None, t
        )
        ps_num = estimate_propensity(
            X_num, treatment_history[:, t]
        )
        
        # Update weights
        treated = treatment_history[:, t] == 1
        weights[treated] *= ps_num[treated] / ps_denom[treated]
        weights[~treated] *= (1 - ps_num[~treated]) / (1 - ps_denom[~treated])
    
    # Weighted outcome regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(treatment_history, outcome, sample_weight=weights)
    
    # Check weight distribution
    if weights.max() / weights.min() > 100:
        print("Warning: Extreme weights detected")
    
    return model.coef_[0], weights
```

### Sensitivity Analysis for Unmeasured Confounding

```python
def sensitivity_to_hidden_bias(treatment_effect, gamma_range):
    """
    Rosenbaum bounds for sensitivity to hidden bias
    
    Gamma: Odds ratio of hidden bias
    If gamma=2, hidden bias could double odds of treatment
    """
    results = []
    
    for gamma in gamma_range:
        # Bounds on p-value under hidden bias
        p_lower = wilcoxon_signed_rank_pvalue(treatment_effect / gamma)
        p_upper = wilcoxon_signed_rank_pvalue(treatment_effect * gamma)
        
        results.append({
            'gamma': gamma,
            'p_lower': p_lower,
            'p_upper': p_upper,
            'significant': p_upper < 0.05
        })
    
    # Find breaking point
    breaking_gamma = None
    for r in results:
        if not r['significant']:
            breaking_gamma = r['gamma']
            break
    
    print(f"Effect remains significant up to gamma={breaking_gamma}")
    
    return results
```

### Causal Inference Validation Framework

```python
class CausalValidation:
    """
    Validate causal inference assumptions and results
    """
    
    def __init__(self, data, treatment, outcome):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
    
    def check_positivity(self):
        """
        Verify sufficient overlap in covariate distributions
        """
        ps = self.estimate_propensity_scores()
        
        # Common support
        ps_treated = ps[self.treatment == 1]
        ps_control = ps[self.treatment == 0]
        
        overlap = min(ps_treated.max(), ps_control.max()) - \
                 max(ps_treated.min(), ps_control.min())
        
        return overlap > 0.1  # At least 10% overlap
    
    def test_balance(self, weights=None):
        """
        Check covariate balance after adjustment
        """
        standardized_differences = []
        
        for col in self.data.columns:
            if weights is None:
                mean_t = self.data[self.treatment == 1][col].mean()
                mean_c = self.data[self.treatment == 0][col].mean()
                std_pooled = self.data[col].std()
            else:
                mean_t = np.average(
                    self.data[self.treatment == 1][col], 
                    weights=weights[self.treatment == 1]
                )
                mean_c = np.average(
                    self.data[self.treatment == 0][col],
                    weights=weights[self.treatment == 0]
                )
                std_pooled = np.sqrt(
                    np.average((self.data[col] - self.data[col].mean())**2,
                              weights=weights)
                )
            
            std_diff = abs(mean_t - mean_c) / std_pooled
            standardized_differences.append(std_diff)
        
        # Good balance: all standardized differences < 0.1
        max_imbalance = max(standardized_differences)
        return max_imbalance < 0.1
    
    def placebo_test(self, placebo_outcome):
        """
        Test on outcome that shouldn't be affected
        """
        # If method finds effect on placebo, something's wrong
        effect = self.estimate_effect(placebo_outcome)
        se = self.estimate_standard_error(placebo_outcome)
        
        z_score = abs(effect / se)
        return z_score < 1.96  # No significant placebo effect
    
    def negative_control_test(self, negative_control_treatment):
        """
        Test with exposure that shouldn't affect outcome
        """
        # Similar to placebo but swaps treatment instead of outcome
        pass
```

### Practical Implementation Guide

```python
def causal_analysis_pipeline(data, treatment, outcome, method='auto'):
    """
    Complete causal inference pipeline
    """
    
    # Step 1: Diagnostic checks
    diagnostics = {
        'n_treated': sum(treatment),
        'n_control': sum(1 - treatment),
        'outcome_mean_treated': outcome[treatment == 1].mean(),
        'outcome_mean_control': outcome[treatment == 0].mean(),
        'naive_difference': outcome[treatment == 1].mean() - 
                           outcome[treatment == 0].mean()
    }
    
    # Step 2: Select method based on data structure
    if method == 'auto':
        if has_instrument(data):
            method = 'iv'
        elif has_discontinuity(data):
            method = 'rdd'
        elif has_panel_structure(data):
            method = 'did'
        else:
            method = 'propensity'
    
    # Step 3: Apply chosen method
    if method == 'propensity':
        effect, se = propensity_score_analysis(data, treatment, outcome)
    elif method == 'iv':
        effect, se = instrumental_variable_analysis(data, treatment, outcome)
    # ... etc
    
    # Step 4: Sensitivity analyses
    sensitivity_results = {
        'hidden_bias': sensitivity_to_hidden_bias(effect),
        'functional_form': test_functional_forms(data, treatment, outcome),
        'measurement_error': bootstrap_measurement_error(data, treatment, outcome)
    }
    
    # Step 5: Report with appropriate caveats
    return {
        'effect': effect,
        'se': se,
        'ci_95': (effect - 1.96*se, effect + 1.96*se),
        'method': method,
        'diagnostics': diagnostics,
        'sensitivity': sensitivity_results,
        'interpretation': generate_interpretation(effect, se, method, sensitivity_results)
    }
```

### Key Principles for Biological Causal Inference

1. **Biology provides structure** - Use known pathways to inform DAGs
2. **Experiments when possible** - CRISPR, knockouts provide gold standard
3. **Multiple lines of evidence** - Combine observational and experimental
4. **Temporal ordering** - Biology has natural time scales
5. **Dose-response** - Biological effects often show gradients
6. **Mechanism matters** - Causal effects should have biological explanation
7. **Heterogeneity is real** - Effects vary across cell types, individuals
8. **Validation essential** - Test predictions in independent systems

### References
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Hernán, M.A. & Robins, J.M. (2020). Causal Inference: What If
- Davey Smith, G. & Ebrahim, S. (2003). Mendelian randomization
- VanderWeele, T.J. (2015). Explanation in Causal Inference