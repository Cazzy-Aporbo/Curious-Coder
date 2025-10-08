# My Journey Into the Abyss of Missing Data
## How I Learned That Absence of Evidence Is Evidence

### Week 1: The Innocent Beginning

I'm analyzing metabolomics data. 40% of values are missing. No problem, I'll just impute with the mean:

```python
# How hard could it be?
def my_first_imputation(data):
    for col in range(data.shape[1]):
        mean_val = np.nanmean(data[:, col])
        missing_mask = np.isnan(data[:, col])
        data[missing_mask, col] = mean_val
    return data

imputed_data = my_first_imputation(metabolomics_data.copy())
print("Done! Moving on to analysis...")
```

My downstream analysis finds 47 significant metabolites. I'm excited. My advisor asks one question that destroys everything:

"Are the missing values random, or do they mean something?"

### Week 2: The Horrible Realization

I investigate why values are missing:

```python
def investigate_missingness(data, detection_limits):
    """
    Oh no. Oh no no no.
    """
    # Plot intensity vs missingness rate
    for metabolite in range(data.shape[1]):
        present_values = data[~np.isnan(data[:, metabolite]), metabolite]
        missing_rate = np.isnan(data[:, metabolite]).mean()
        
        if len(present_values) > 0:
            mean_intensity = np.mean(present_values)
            print(f"Metabolite {metabolite}: Missing {missing_rate:.1%}, Mean intensity when present: {mean_intensity:.2f}")
    
    # The pattern that made my stomach drop:
    # High intensity metabolites: 5% missing
    # Medium intensity: 20% missing  
    # Low intensity: 60% missing
    
    # IT'S NOT RANDOM. It's below detection limit!
```

Missing values aren't missing randomly. They're missing because the metabolite concentration is below the detection threshold. By imputing with the mean, I'm saying "undetectable = average," which is completely wrong.

### Week 3: Understanding the Three Types of Missingness

I need to understand this properly:

```python
def demonstrate_missingness_types():
    """
    MCAR vs MAR vs MNAR - why it matters
    """
    true_data = np.random.lognormal(0, 2, (100, 1))
    
    # Type 1: Missing Completely At Random (MCAR)
    # Missingness unrelated to any values
    mcar_data = true_data.copy()
    mcar_mask = np.random.random(100) < 0.3  # 30% missing
    mcar_data[mcar_mask] = np.nan
    
    # Type 2: Missing At Random (MAR)
    # Missingness depends on observed values
    mar_data = true_data.copy()
    covariate = np.random.randn(100)
    mar_prob = 1 / (1 + np.exp(-covariate))  # Depends on covariate
    mar_mask = np.random.random(100) < mar_prob
    mar_data[mar_mask] = np.nan
    
    # Type 3: Missing Not At Random (MNAR) 
    # Missingness depends on the missing value itself!
    mnar_data = true_data.copy()
    detection_limit = np.percentile(true_data, 30)
    mnar_data[true_data < detection_limit] = np.nan  # Can't detect low values
    
    # Compare imputation effects
    for data, name in [(mcar_data, 'MCAR'), (mar_data, 'MAR'), (mnar_data, 'MNAR')]:
        observed_mean = np.nanmean(data)
        true_mean = np.mean(true_data)
        bias = observed_mean - true_mean
        
        print(f"{name}: True mean={true_mean:.2f}, Observed mean={observed_mean:.2f}, Bias={bias:.2f}")
    
    # Output:
    # MCAR: True mean=2.84, Observed mean=2.81, Bias=-0.03
    # MAR:  True mean=2.84, Observed mean=2.79, Bias=-0.05  
    # MNAR: True mean=2.84, Observed mean=4.92, Bias=2.08  # HUGE BIAS!
```

My metabolomics data is MNAR. Standard methods will fail catastrophically.

### Week 4: Little's MCAR Test

Maybe I can at least test if data is MCAR:

```python
def little_mcar_test(data):
    """
    Little's MCAR test - the theory is beautiful
    H0: Data is MCAR
    H1: Data is not MCAR
    """
    # The idea: If MCAR, then missing pattern shouldn't relate to values
    # Split data by missing patterns
    
    patterns = {}
    for i, row in enumerate(data):
        pattern = tuple(np.isnan(row).astype(int))
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(i)
    
    # For each pattern, compute mean and covariance
    # Under MCAR, all patterns should have same parameters
    
    # This gets complex fast...
    # Chi-squared test statistic with complex degrees of freedom
    
    # But here's what I learned: This test has NO POWER to detect MNAR!
    # It can only distinguish MCAR from MAR
```

### Week 5: The Tobit Model Revelation

For data that's missing due to detection limits, there's a better way:

```python
def tobit_regression_discovery(y, X, detection_limit):
    """
    The Tobit model - designed for censored data
    
    Latent model: y* = Xβ + ε
    Observed: y = max(y*, detection_limit)
    """
    
    def tobit_log_likelihood(params, y, X, limit):
        beta = params[:-1]
        sigma = params[-1]
        
        y_pred = X @ beta
        
        # For observed values: normal likelihood
        observed_mask = y > limit
        ll_observed = -0.5 * np.log(2*np.pi*sigma**2) - \
                      0.5 * ((y[observed_mask] - y_pred[observed_mask])**2) / sigma**2
        
        # For censored values: probability of being below limit
        censored_mask = y <= limit
        z = (limit - y_pred[censored_mask]) / sigma
        ll_censored = np.log(norm.cdf(z))
        
        return -np.sum(ll_observed) - np.sum(ll_censored)
    
    # This completely changed how I think about "missing" values
    # They're not missing - they're bounds on the true value!
```

### Week 6: Multiple Imputation - A Deeper Understanding

I thought I understood multiple imputation. I didn't:

```python
def multiple_imputation_deep_dive(data):
    """
    What MI actually does - it's about uncertainty, not just filling gaps
    """
    
    # Step 1: Impute multiple times
    n_imputations = 50
    imputed_datasets = []
    
    for m in range(n_imputations):
        # Each imputation draws from posterior predictive distribution
        # Not just point estimates!
        
        # MICE algorithm - the real version
        imputed = data.copy()
        
        for iteration in range(10):  # Usually converges in <10
            for col in range(data.shape[1]):
                # Use all other columns to predict this one
                missing_mask = np.isnan(data[:, col])
                
                if missing_mask.any():
                    observed_mask = ~missing_mask
                    
                    # Train model on observed
                    X_train = np.delete(imputed[observed_mask], col, axis=1)
                    y_train = data[observed_mask, col]
                    
                    # Predict missing with uncertainty
                    X_pred = np.delete(imputed[missing_mask], col, axis=1)
                    
                    # Bayesian regression for proper uncertainty
                    model = BayesianRidge()
                    model.fit(X_train, y_train)
                    
                    # Sample from posterior predictive
                    y_pred_mean, y_pred_std = model.predict(X_pred, return_std=True)
                    y_pred = np.random.normal(y_pred_mean, y_pred_std)
                    
                    imputed[missing_mask, col] = y_pred
        
        imputed_datasets.append(imputed)
    
    # Step 2: Analyze each dataset
    results = []
    for imputed in imputed_datasets:
        # Run your analysis
        result = analyze(imputed)
        results.append(result)
    
    # Step 3: Pool results using Rubin's rules
    # This is the key insight - we're not averaging data, we're averaging inferences!
    
    pooled_estimate = np.mean(results)
    
    # Within-imputation variance
    within_var = np.mean([r.variance for r in results])
    
    # Between-imputation variance  
    between_var = np.var([r.estimate for r in results])
    
    # Total variance (Rubin's formula)
    total_var = within_var + between_var * (1 + 1/n_imputations)
    
    print(f"Pooled estimate: {pooled_estimate:.3f} ± {np.sqrt(total_var):.3f}")
    print(f"Variance inflation from missingness: {between_var/within_var:.1%}")
```

### Week 7: Sensitivity Analysis - Embracing Uncertainty

I can't know the true missing mechanism. So I need to check everything:

```python
def sensitivity_to_missing_mechanism(data, outcome):
    """
    What if my assumptions are wrong?
    """
    
    results = {}
    
    # Assumption 1: MCAR - Missing completely at random
    data_mcar = data.copy()
    data_mcar[np.isnan(data)] = np.nanmean(data, axis=0)
    results['MCAR'] = analyze(data_mcar, outcome)
    
    # Assumption 2: MAR - Missing at random given observables
    imputer_mar = IterativeImputer(random_state=0)
    data_mar = imputer_mar.fit_transform(data)
    results['MAR'] = analyze(data_mar, outcome)
    
    # Assumption 3: MNAR - Missing = below detection
    # Impute with half the minimum observed value
    data_mnar = data.copy()
    for col in range(data.shape[1]):
        min_observed = np.nanmin(data[:, col])
        data_mnar[np.isnan(data[:, col]), col] = min_observed / 2
    results['MNAR_low'] = analyze(data_mnar, outcome)
    
    # Assumption 4: MNAR - Missing = above detection  
    # (Saturation in some assays)
    data_mnar_high = data.copy()
    for col in range(data.shape[1]):
        max_observed = np.nanmax(data[:, col])
        data_mnar_high[np.isnan(data[:, col]), col] = max_observed * 2
    results['MNAR_high'] = analyze(data_mnar_high, outcome)
    
    # Pattern mixture model - different mechanism per variable
    # This is where it gets really complex...
    
    # Compare all results
    for mechanism, result in results.items():
        print(f"{mechanism}: Effect size = {result['effect']:.3f}, p = {result['pvalue']:.4f}")
    
    # If conclusions change dramatically, you have a problem
    if max(r['effect'] for r in results.values()) / min(r['effect'] for r in results.values()) > 2:
        print("WARNING: Results highly sensitive to missing data assumptions!")
```

### Week 8: The Pattern Discovery

Missing patterns themselves carry information:

```python
def missingness_as_information(data, clinical_outcomes):
    """
    The pattern of what's missing tells us about the patient
    """
    
    # Create missingness indicators
    missing_pattern = np.isnan(data).astype(int)
    
    # Can we predict outcome from just the missingness pattern?
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier()
    scores = cross_val_score(rf, missing_pattern, clinical_outcomes, cv=5)
    
    print(f"Prediction from missingness alone: {scores.mean():.3f}")
    
    # Often surprisingly high! Why?
    # - Sicker patients have more tests ordered (less missing)
    # - Certain diseases affect what can be measured
    # - Treatment protocols determine what gets measured
    
    # Feature importance
    rf.fit(missing_pattern, clinical_outcomes)
    importance = rf.feature_importances_
    
    for i, imp in enumerate(importance):
        if imp > 0.05:  # Important features
            print(f"Missing {metabolite_names[i]}: Importance = {imp:.3f}")
    
    # This completely changed my perspective
    # Missingness is data, not just absence of data
```

### Week 9: Building My Own Imputation for Metabolomics

Understanding the biology leads to better methods:

```python
class MetabolomicsImputer:
    """
    Domain-specific imputation using biological knowledge
    """
    
    def __init__(self, pathway_database, detection_limits):
        self.pathways = pathway_database
        self.limits = detection_limits
        
    def impute(self, data):
        """
        Use metabolic pathway information for imputation
        """
        imputed = data.copy()
        
        for metabolite_idx in range(data.shape[1]):
            missing_mask = np.isnan(data[:, metabolite_idx])
            
            if not missing_mask.any():
                continue
            
            # Find metabolites in same pathway
            metabolite_name = self.metabolite_names[metabolite_idx]
            pathway_partners = self.find_pathway_partners(metabolite_name)
            
            if pathway_partners:
                # Use pathway partners for prediction
                partner_indices = [self.metabolite_names.index(p) 
                                 for p in pathway_partners 
                                 if p in self.metabolite_names]
                
                if partner_indices:
                    # Biological constraint: Metabolites in same pathway correlate
                    X = data[:, partner_indices]
                    
                    # But also consider detection limits
                    if self.is_below_detection(metabolite_idx, data):
                        # Sample from truncated distribution
                        # This metabolite is likely below detection
                        
                        # Estimate parameters from observed values
                        observed = data[~missing_mask, metabolite_idx]
                        if len(observed) > 2:
                            mu, sigma = norm.fit(np.log(observed + 1e-10))
                            
                            # Sample from truncated log-normal
                            limit = self.limits[metabolite_idx]
                            samples = self.sample_truncated_lognormal(
                                mu, sigma, limit, size=missing_mask.sum()
                            )
                            imputed[missing_mask, metabolite_idx] = samples
                    else:
                        # MAR imputation using pathway information
                        self.pathway_based_imputation(
                            imputed, metabolite_idx, partner_indices, missing_mask
                        )
            else:
                # No pathway info - fall back to careful statistical imputation
                self.statistical_imputation(imputed, metabolite_idx, missing_mask)
        
        return imputed
    
    def is_below_detection(self, metabolite_idx, data):
        """
        Determine if missing is likely due to detection limit
        """
        # Check if missing values occur mainly in samples with low overall intensity
        missing_mask = np.isnan(data[:, metabolite_idx])
        
        if missing_mask.sum() < 2:
            return False
        
        # Compare total intensity in samples with/without this metabolite
        total_intensity_missing = np.nansum(data[missing_mask], axis=1)
        total_intensity_present = np.nansum(data[~missing_mask], axis=1)
        
        # If samples with missing values have lower total intensity,
        # likely below detection
        return np.median(total_intensity_missing) < np.median(total_intensity_present)
```

### Week 10: The Information Theory Perspective

How much information do we lose to missingness?

```python
def information_loss_from_missingness(complete_data, missing_mask):
    """
    Quantify information loss using entropy
    """
    
    # Entropy of complete data
    # H(X) = -Σ p(x) log p(x)
    
    # Discretize for entropy calculation
    n_bins = int(np.sqrt(len(complete_data)))
    
    # Complete data entropy
    hist_complete, _ = np.histogram(complete_data, bins=n_bins)
    p_complete = hist_complete / hist_complete.sum()
    p_complete = p_complete[p_complete > 0]  # Remove zeros
    H_complete = -np.sum(p_complete * np.log2(p_complete))
    
    # Observed data entropy (with missing)
    observed_data = complete_data[~missing_mask]
    hist_observed, _ = np.histogram(observed_data, bins=n_bins)
    p_observed = hist_observed / hist_observed.sum()
    p_observed = p_observed[p_observed > 0]
    H_observed = -np.sum(p_observed * np.log2(p_observed))
    
    # Information loss
    info_loss = H_complete - H_observed
    
    print(f"Complete data entropy: {H_complete:.2f} bits")
    print(f"Observed data entropy: {H_observed:.2f} bits")
    print(f"Information loss: {info_loss:.2f} bits ({info_loss/H_complete:.1%})")
    
    # But wait - if missing is informative (MNAR), we might have MORE information!
    # The absence pattern adds information
    
    # Joint entropy of (observed values, missing pattern)
    # This is where it gets philosophically interesting
```

### Week 11: The Causal Perspective

When does imputation create or destroy causal relationships?

```python
def causal_implications_of_imputation():
    """
    Imputation can create spurious relationships or hide real ones
    """
    
    # True causal model:
    # X -> Y
    # X -> M (missingness in Y)
    
    n = 1000
    X = np.random.randn(n)
    Y_true = 2 * X + np.random.randn(n)
    
    # Missingness depends on X (MAR)
    miss_prob = 1 / (1 + np.exp(-X))  # Higher X -> more likely missing
    missing_mask = np.random.random(n) < miss_prob
    
    Y_observed = Y_true.copy()
    Y_observed[missing_mask] = np.nan
    
    # Different imputation strategies
    
    # 1. Mean imputation
    Y_mean = Y_observed.copy()
    Y_mean[missing_mask] = np.nanmean(Y_observed)
    
    # 2. Regression imputation using X
    Y_regression = Y_observed.copy()
    model = LinearRegression().fit(X[~missing_mask].reshape(-1, 1), 
                                  Y_observed[~missing_mask])
    Y_regression[missing_mask] = model.predict(X[missing_mask].reshape(-1, 1))
    
    # Compare causal effects
    true_effect = np.cov(X, Y_true)[0, 1] / np.var(X)
    mean_imp_effect = np.cov(X, Y_mean)[0, 1] / np.var(X)
    reg_imp_effect = np.cov(X, Y_regression)[0, 1] / np.var(X)
    
    print(f"True causal effect: {true_effect:.3f}")
    print(f"After mean imputation: {mean_imp_effect:.3f}")  # Biased toward zero
    print(f"After regression imputation: {reg_imp_effect:.3f}")  # Less biased
    
    # The deep question: What assumptions about causality does each method make?
```

### Week 12: My Current Understanding

```python
class MissingDataPhilosophy:
    """
    What I now believe about missing data
    """
    
    principles = [
        "Missing data is not a technical problem, it's a scientific one",
        "The mechanism matters more than the method",
        "Sometimes the best imputation is no imputation",
        "Missingness patterns are data",
        "Uncertainty from missingness must be propagated",
        "Biology determines missingness more than statistics",
        "Validation must include sensitivity analysis",
        "Document why data is missing, not just that it is"
    ]
    
    def decision_framework(self, data, context):
        """
        How I now approach missing data
        """
        
        # 1. Understand the mechanism
        mechanism = self.diagnose_mechanism(data, context)
        
        # 2. Consider the scientific question
        # Sometimes missingness IS the answer
        if self.is_missingness_informative(data, context):
            return "Analyze missingness pattern directly"
        
        # 3. Match method to mechanism
        if mechanism == "MCAR":
            return "Complete case or simple imputation acceptable"
        
        elif mechanism == "MAR":
            return "Multiple imputation or inverse probability weighting"
        
        elif mechanism == "MNAR":
            if context == "detection_limit":
                return "Tobit or survival models"
            elif context == "dropout":
                return "Pattern mixture or selection models"
            else:
                return "Sensitivity analysis across assumptions"
        
        # 4. Always quantify uncertainty
        return "Whatever method, bootstrap the whole pipeline"
```

### What I'm Still Learning

The deeper I go, the more questions emerge:

```python
# Current exploration: Optimal design to minimize missingness impact
def optimal_experimental_design():
    """
    Can we design experiments that are robust to missingness?
    
    Ideas I'm exploring:
    1. Planned missingness - deliberately skip some measurements
    2. Adaptive sampling - measure more where uncertainty is high
    3. Multiple cheap measurements vs few expensive ones
    4. Validation cohorts specifically for missing mechanisms
    """
    pass

# The question that keeps me up at night:
# If we perfectly imputed all missing values, 
# would we be gaining information or destroying it?
```

This journey taught me that missing data isn't about filling in blanks. It's about understanding what the blanks mean.