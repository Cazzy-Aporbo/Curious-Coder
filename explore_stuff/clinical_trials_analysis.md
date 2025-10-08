# Clinical Trial Design and Analysis
## Statistical Rigor When Lives Depend on the Answer

### Intent
Clinical trials determine which treatments reach patients. This document provides comprehensive frameworks for designing trials that answer the right questions, analyzing data that respects clinical complexity, and making decisions where statistical significance and clinical significance diverge.

### The Fundamental Framework

**Clinical Trial Equation:**
```
Truth = Observed Effect ± Bias ± Random Error

Our job: Minimize bias, quantify random error, interpret clinically
```

### Phase-Specific Design Considerations

#### Phase I: Dose-Finding (n=20-100)

**Objective:** Find Maximum Tolerated Dose (MTD)

**3+3 Design (Traditional):**
```
Algorithm:
1. Treat 3 patients at dose level i
2. If 0/3 have DLT → escalate to i+1
3. If 1/3 has DLT → treat 3 more at level i
   - If 1/6 has DLT → escalate
   - If ≥2/6 have DLT → de-escalate
4. If ≥2/3 have DLT → de-escalate

MTD = highest dose with DLT rate <33%
```

**Continual Reassessment Method (CRM) - Bayesian:**
```python
def crm_dose_finding(toxicity_targets, prior_skeleton):
    """
    Model-based dose finding
    
    Toxicity model: P(DLT at dose d) = d^exp(β)
    """
    
    def posterior_toxicity_rate(dose_level, observed_dlts, n_patients, beta_prior):
        """
        Bayesian update of toxicity estimate
        """
        # Likelihood: Binomial
        # Prior: Normal on log-odds scale
        
        def log_posterior(beta):
            # Prior
            log_prior = norm.logpdf(beta, beta_prior['mean'], beta_prior['std'])
            
            # Likelihood
            p_dlt = dose_level ** np.exp(beta)
            log_likelihood = binom.logpmf(observed_dlts, n_patients, p_dlt)
            
            return log_prior + log_likelihood
        
        # Sample from posterior using MCMC
        posterior_samples = mcmc_sample(log_posterior)
        
        # Posterior mean toxicity rate
        posterior_rates = dose_level ** np.exp(posterior_samples)
        return posterior_rates.mean(), posterior_rates.std()
    
    # Select next dose closest to target toxicity
    def select_next_dose(current_data, target_toxicity=0.25):
        posterior_rates = []
        for dose in dose_levels:
            rate, uncertainty = posterior_toxicity_rate(dose, current_data)
            posterior_rates.append(rate)
        
        # Choose dose closest to target
        distances = [abs(rate - target_toxicity) for rate in posterior_rates]
        return dose_levels[np.argmin(distances)]
```

#### Phase II: Efficacy Signal (n=50-300)

**Simon's Two-Stage Design:**
```
Minimize expected sample size under H₀
H₀: response rate ≤ p₀ (not promising)
H₁: response rate ≥ p₁ (promising)

Stage 1: Treat n₁ patients
- If ≤r₁ responses → STOP for futility
- If >r₁ responses → continue

Stage 2: Treat n₂ additional patients  
- If ≤r responses total → drug ineffective
- If >r responses → drug promising

Optimal design minimizes E[N|H₀]
```

```python
def simon_two_stage_design(p0, p1, alpha=0.05, beta=0.20):
    """
    Find optimal two-stage design
    """
    from scipy.stats import binom
    
    designs = []
    
    for n1 in range(1, 100):
        for n in range(n1, 200):
            n2 = n - n1
            
            for r1 in range(n1 + 1):
                for r in range(r1, n + 1):
                    
                    # Type I error (reject H₀ when true)
                    prob_pass_stage1_h0 = 1 - binom.cdf(r1, n1, p0)
                    prob_pass_both_h0 = sum([
                        binom.pmf(x1, n1, p0) * (1 - binom.cdf(r - x1, n2, p0))
                        for x1 in range(r1 + 1, min(n1, r) + 1)
                    ])
                    alpha_actual = prob_pass_both_h0
                    
                    # Power (reject H₀ when false)
                    prob_pass_stage1_h1 = 1 - binom.cdf(r1, n1, p1)
                    power = sum([
                        binom.pmf(x1, n1, p1) * (1 - binom.cdf(r - x1, n2, p1))
                        for x1 in range(r1 + 1, min(n1, r) + 1)
                    ])
                    
                    # Expected sample size under H₀
                    en_h0 = n1 + n2 * prob_pass_stage1_h0
                    
                    if alpha_actual <= alpha and power >= 1 - beta:
                        designs.append({
                            'n1': n1, 'r1': r1, 'n': n, 'r': r,
                            'alpha': alpha_actual, 'power': power,
                            'en_h0': en_h0
                        })
    
    # Select optimal (minimum expected N under H₀)
    optimal = min(designs, key=lambda x: x['en_h0'])
    return optimal
```

#### Phase III: Confirmatory (n=300-3000)

**Sample Size Calculation:**

```python
def sample_size_superiority(effect_size, alpha=0.025, power=0.80, 
                           allocation_ratio=1, dropout_rate=0.1):
    """
    Sample size for superiority trial
    
    For continuous outcome:
    n = (σ²(z_α + z_β)²(1 + 1/r)) / δ²
    
    For binary outcome:
    n = (p₁(1-p₁)/r + p₂(1-p₂))(z_α + z_β)² / (p₁ - p₂)²
    """
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha)  # One-sided test
    z_beta = norm.ppf(power)
    
    if outcome_type == 'continuous':
        # Standardized effect size
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
    elif outcome_type == 'binary':
        p1, p2 = effect_size  # (treatment rate, control rate)
        p_bar = (p1 + p2) / 2
        
        n_per_group = (p_bar * (1 - p_bar) * (z_alpha + z_beta)**2) / (p1 - p2)**2
    
    # Adjust for unequal allocation
    n_treatment = n_per_group * allocation_ratio / (1 + allocation_ratio)
    n_control = n_per_group / (1 + allocation_ratio)
    
    # Inflate for dropout
    n_total = (n_treatment + n_control) / (1 - dropout_rate)
    
    return int(np.ceil(n_total))
```

### Randomization Strategies

#### 1. Block Randomization

```python
def block_randomization(n_patients, block_sizes=[4, 6], treatments=['A', 'B']):
    """
    Maintain balance throughout recruitment
    """
    allocation = []
    remaining = n_patients
    
    while remaining > 0:
        # Random block size
        block_size = np.random.choice(block_sizes)
        block_size = min(block_size, remaining)
        
        # Create balanced block
        n_per_treatment = block_size // len(treatments)
        block = treatments * n_per_treatment
        
        # Handle odd block sizes
        if block_size % len(treatments) != 0:
            extra = np.random.choice(treatments, 
                                    block_size % len(treatments), 
                                    replace=False)
            block.extend(extra)
        
        # Randomize within block
        np.random.shuffle(block)
        allocation.extend(block)
        
        remaining -= block_size
    
    return allocation[:n_patients]
```

#### 2. Stratified Randomization

```python
def stratified_randomization(patients, stratification_factors):
    """
    Balance treatment groups within strata
    """
    
    # Create strata
    strata = {}
    for patient in patients:
        stratum_key = tuple(patient[factor] for factor in stratification_factors)
        if stratum_key not in strata:
            strata[stratum_key] = []
        strata[stratum_key].append(patient)
    
    # Randomize within each stratum
    allocations = {}
    for stratum_key, stratum_patients in strata.items():
        stratum_allocation = block_randomization(len(stratum_patients))
        for patient, treatment in zip(stratum_patients, stratum_allocation):
            allocations[patient['id']] = treatment
    
    return allocations
```

#### 3. Adaptive Randomization

```python
def response_adaptive_randomization(current_responses):
    """
    Update allocation ratio based on observed responses
    """
    
    # Bayesian update
    successes_a = current_responses['A']['successes']
    failures_a = current_responses['A']['failures']
    successes_b = current_responses['B']['successes']
    failures_b = current_responses['B']['failures']
    
    # Posterior Beta distributions
    # Prior: Beta(1, 1) - uniform
    posterior_a = beta(1 + successes_a, 1 + failures_a)
    posterior_b = beta(1 + successes_b, 1 + failures_b)
    
    # Thompson sampling
    sample_a = posterior_a.rvs()
    sample_b = posterior_b.rvs()
    
    # Probability of assigning to A
    prob_a = sample_a / (sample_a + sample_b)
    
    # Add some exploration (don't go to extremes)
    prob_a = 0.1 + 0.8 * prob_a  # Keep between 10% and 90%
    
    return 'A' if np.random.random() < prob_a else 'B'
```

### Interim Analyses and Stopping Rules

#### Group Sequential Design

```python
def group_sequential_boundaries(n_analyses, alpha=0.025, beta=0.10, 
                               method='obrien-fleming'):
    """
    Calculate stopping boundaries for interim analyses
    """
    from scipy.stats import norm
    
    K = n_analyses  # Number of analyses
    
    if method == 'obrien-fleming':
        # Conservative early, aggressive late
        # α spending: α(t) = 2 - 2Φ(z_α/√t)
        
        information_fractions = np.arange(1, K+1) / K
        
        boundaries = []
        for k, t_k in enumerate(information_fractions, 1):
            if k == K:
                # Final analysis uses remaining α
                z_k = norm.ppf(1 - alpha)
            else:
                # O'Brien-Fleming boundary
                z_k = norm.ppf(1 - alpha) / np.sqrt(t_k)
            
            boundaries.append({
                'analysis': k,
                'information': t_k,
                'z_boundary': z_k,
                'p_boundary': 1 - norm.cdf(z_k)
            })
    
    elif method == 'pocock':
        # Constant boundary (aggressive early stopping)
        # Find c such that overall α is preserved
        
        from scipy.optimize import minimize_scalar
        
        def alpha_spent(c):
            # Probability of crossing any boundary
            prob = 0
            for k in range(1, K+1):
                t_k = k / K
                # Complex calculation involving multivariate normal
                # Simplified here
                prob += norm.cdf(-c) * (1 - prob)
            return abs(prob - alpha)
        
        result = minimize_scalar(alpha_spent)
        c_pocock = result.x
        
        boundaries = []
        for k in range(1, K+1):
            boundaries.append({
                'analysis': k,
                'information': k/K,
                'z_boundary': c_pocock,
                'p_boundary': 1 - norm.cdf(c_pocock)
            })
    
    return boundaries
```

#### Conditional Power and Sample Size Re-estimation

```python
def conditional_power(observed_effect, observed_n, planned_n, 
                     target_effect, variance):
    """
    Probability of success given current data
    """
    
    # Information fraction
    info_current = observed_n / planned_n
    info_remaining = 1 - info_current
    
    # Current Z-score
    z_current = observed_effect / np.sqrt(variance / observed_n)
    
    # Drift parameter
    theta = target_effect * np.sqrt(planned_n / variance)
    
    # Conditional power
    z_final = (z_current * np.sqrt(info_current) + 
               theta * np.sqrt(info_remaining))
    
    from scipy.stats import norm
    cp = norm.cdf(z_final - norm.ppf(0.975))  # Two-sided test
    
    # Sample size re-estimation
    if cp < 0.30:
        # Futility - consider stopping
        recommendation = "STOP_FUTILITY"
        
    elif cp < 0.80:
        # Increase sample size to achieve 80% conditional power
        
        # Required final Z
        z_required = norm.ppf(0.975) + norm.ppf(0.80)
        
        # Required total information
        info_required = ((z_required - z_current * np.sqrt(info_current)) / 
                        theta) ** 2
        
        new_n = int(np.ceil(info_required * variance / target_effect**2))
        recommendation = f"INCREASE_N_TO_{new_n}"
        
    else:
        recommendation = "CONTINUE_AS_PLANNED"
    
    return cp, recommendation
```

### Handling Missing Data in Clinical Trials

#### Multiple Imputation Under Different Assumptions

```python
def clinical_trial_missing_data(data, treatment, outcome, missing_mechanism):
    """
    Handle missing data with clinical context
    """
    
    if missing_mechanism == "MAR":
        # Standard multiple imputation
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        imputer = IterativeImputer(random_state=0, n_nearest_features=5)
        imputed_datasets = []
        
        for m in range(20):  # 20 imputations
            imputer.set_params(random_state=m)
            imputed = imputer.fit_transform(data)
            imputed_datasets.append(imputed)
    
    elif missing_mechanism == "MNAR_jumped_to_reference":
        # Jump to reference (J2R) - conservative for efficacy
        # Assume patients who drop out revert to control group behavior
        
        control_outcomes = data[treatment == 0][outcome]
        control_mean = np.nanmean(control_outcomes)
        control_std = np.nanstd(control_outcomes)
        
        imputed_datasets = []
        for m in range(20):
            imputed = data.copy()
            missing_mask = np.isnan(data[outcome])
            
            # Impute from control distribution
            n_missing = missing_mask.sum()
            imputed_values = np.random.normal(control_mean, control_std, n_missing)
            imputed[outcome][missing_mask] = imputed_values
            
            imputed_datasets.append(imputed)
    
    elif missing_mechanism == "MNAR_tipping_point":
        # Tipping point analysis - sensitivity analysis
        delta_values = np.linspace(-0.5, 0.5, 11)  # Shift parameters
        
        results = []
        for delta in delta_values:
            # Impute with shift
            imputed = data.copy()
            missing_mask = np.isnan(data[outcome])
            
            # MAR imputation then shift
            base_imputed = simple_imputation(data)
            imputed[outcome][missing_mask] = base_imputed[outcome][missing_mask] + delta
            
            # Analyze
            treatment_effect = analyze_trial(imputed)
            results.append({'delta': delta, 'effect': treatment_effect})
        
        return results
    
    # Analyze each imputed dataset and pool
    results = []
    for imputed in imputed_datasets:
        effect, variance = analyze_single_dataset(imputed)
        results.append({'effect': effect, 'variance': variance})
    
    # Rubin's rules
    pooled_effect = np.mean([r['effect'] for r in results])
    within_variance = np.mean([r['variance'] for r in results])
    between_variance = np.var([r['effect'] for r in results])
    total_variance = within_variance + (1 + 1/len(results)) * between_variance
    
    return pooled_effect, np.sqrt(total_variance)
```

### Multiplicity Adjustment

#### Hierarchical Testing Strategy

```python
def hierarchical_testing(p_values, hierarchy, alpha=0.025):
    """
    Control Type I error with multiple endpoints
    """
    
    results = {}
    remaining_alpha = alpha
    
    for level in hierarchy:
        level_rejected = False
        
        for endpoint in level:
            if p_values[endpoint] <= remaining_alpha:
                results[endpoint] = 'SIGNIFICANT'
                level_rejected = True
            else:
                results[endpoint] = 'NOT_SIGNIFICANT'
                # Stop testing in this family
                break
        
        if not level_rejected:
            # Can't proceed to next level
            break
    
    return results
```

### Subgroup Analysis

```python
def subgroup_analysis_with_interaction(data, treatment, outcome, subgroup_var):
    """
    Proper subgroup analysis with interaction testing
    """
    import statsmodels.api as sm
    
    # First: Test for interaction
    model_formula = f'{outcome} ~ {treatment} * {subgroup_var}'
    interaction_model = sm.OLS.from_formula(model_formula, data).fit()
    
    interaction_p = interaction_model.pvalues[f'{treatment}:{subgroup_var}']
    
    results = {
        'interaction_p': interaction_p,
        'interaction_significant': interaction_p < 0.05
    }
    
    # Subgroup-specific effects
    for subgroup_level in data[subgroup_var].unique():
        subgroup_data = data[data[subgroup_var] == subgroup_level]
        
        # Effect in subgroup
        treat_outcomes = subgroup_data[subgroup_data[treatment] == 1][outcome]
        control_outcomes = subgroup_data[subgroup_data[treatment] == 0][outcome]
        
        effect = treat_outcomes.mean() - control_outcomes.mean()
        
        # Confidence interval
        se = np.sqrt(treat_outcomes.var()/len(treat_outcomes) + 
                    control_outcomes.var()/len(control_outcomes))
        ci = (effect - 1.96*se, effect + 1.96*se)
        
        results[f'effect_{subgroup_level}'] = {
            'estimate': effect,
            'ci': ci,
            'n': len(subgroup_data)
        }
    
    # Forest plot data
    results['forest_plot'] = create_forest_plot_data(results)
    
    return results
```

### Bayesian Methods in Clinical Trials

```python
def bayesian_adaptive_trial(prior_params, data_so_far, decision_thresholds):
    """
    Bayesian adaptive design with predictive probabilities
    """
    
    # Update posterior
    posterior = update_posterior(prior_params, data_so_far)
    
    # Posterior probability of success
    prob_success = 1 - posterior.cdf(0)  # P(effect > 0)
    
    # Predictive probability of trial success
    remaining_n = planned_n - len(data_so_far)
    
    predictive_success = 0
    n_simulations = 10000
    
    for _ in range(n_simulations):
        # Sample from posterior
        true_effect = posterior.rvs()
        
        # Simulate remaining patients
        future_data = simulate_patients(remaining_n, true_effect)
        
        # Combine with current data
        combined_data = np.concatenate([data_so_far, future_data])
        
        # Final analysis
        final_posterior = update_posterior(prior_params, combined_data)
        final_success = 1 - final_posterior.cdf(0) > 0.975  # High confidence
        
        predictive_success += final_success
    
    predictive_success /= n_simulations
    
    # Decision rules
    if prob_success > decision_thresholds['stop_success']:
        return 'STOP_FOR_SUCCESS'
    elif predictive_success < decision_thresholds['stop_futility']:
        return 'STOP_FOR_FUTILITY'
    else:
        return 'CONTINUE'
```

### Non-Inferiority and Equivalence Trials

```python
def non_inferiority_analysis(treatment_effect, se, margin, alpha=0.025):
    """
    Test if new treatment is not worse than standard by more than margin
    
    H₀: μ_new - μ_standard ≤ -margin (new is inferior)
    H₁: μ_new - μ_standard > -margin (new is non-inferior)
    """
    
    # Shift the null hypothesis
    z = (treatment_effect + margin) / se
    
    from scipy.stats import norm
    p_value = 1 - norm.cdf(z)
    
    # Confidence interval
    ci_lower = treatment_effect - norm.ppf(1-alpha) * se
    ci_upper = float('inf')  # One-sided for non-inferiority
    
    non_inferior = ci_lower > -margin
    
    # Check for superiority (if non-inferior)
    if non_inferior:
        superior = ci_lower > 0
    else:
        superior = False
    
    return {
        'effect': treatment_effect,
        'ci': (ci_lower, ci_upper),
        'margin': margin,
        'non_inferior': non_inferior,
        'superior': superior,
        'p_value': p_value
    }
```

### Adaptive Designs

#### SMART Design (Sequential Multiple Assignment Randomized Trial)

```python
def smart_design(patient, stage=1):
    """
    Adaptive treatment strategy
    """
    
    if stage == 1:
        # Initial randomization
        treatment = np.random.choice(['A', 'B'])
        patient['stage1_treatment'] = treatment
        
        # Observe response after period 1
        response = observe_response(patient, treatment)
        patient['stage1_response'] = response
        
    if stage == 2:
        # Re-randomize based on stage 1 response
        if patient['stage1_response'] == 'responder':
            # Responders: continue vs. switch
            if patient['stage1_treatment'] == 'A':
                treatment = np.random.choice(['A', 'C'])
            else:
                treatment = np.random.choice(['B', 'D'])
                
        else:  # Non-responder
            # Non-responders: intensify vs. switch
            if patient['stage1_treatment'] == 'A':
                treatment = np.random.choice(['A+E', 'F'])
            else:
                treatment = np.random.choice(['B+E', 'G'])
        
        patient['stage2_treatment'] = treatment
    
    return patient
```

### Platform Trials

```python
class PlatformTrial:
    """
    Master protocol testing multiple treatments
    """
    
    def __init__(self, control_arm, alpha_total=0.05):
        self.control_arm = control_arm
        self.treatment_arms = {}
        self.alpha_total = alpha_total
        self.alpha_spent = 0
        
    def add_treatment_arm(self, treatment_name, start_time):
        """Add new treatment to ongoing trial"""
        
        # Allocate α for new comparison
        remaining_alpha = self.alpha_total - self.alpha_spent
        n_active_arms = len([a for a in self.treatment_arms.values() 
                           if a['status'] == 'active'])
        
        alpha_allocation = remaining_alpha / (n_active_arms + 2)  # Conservative
        
        self.treatment_arms[treatment_name] = {
            'start_time': start_time,
            'alpha': alpha_allocation,
            'status': 'active',
            'patients': []
        }
    
    def adaptive_randomization(self, current_time):
        """Response-adaptive randomization across all active arms"""
        
        active_arms = [self.control_arm] + [
            name for name, arm in self.treatment_arms.items() 
            if arm['status'] == 'active'
        ]
        
        # Calculate allocation probabilities
        arm_performances = {}
        for arm in active_arms:
            if len(arm['patients']) > 10:  # Minimum data
                response_rate = calculate_response_rate(arm['patients'])
                arm_performances[arm] = response_rate
            else:
                arm_performances[arm] = 0.5  # Prior
        
        # Thompson sampling for allocation
        allocation_probs = thompson_sampling(arm_performances)
        
        return np.random.choice(active_arms, p=allocation_probs)
    
    def interim_analysis(self, arm_name):
        """Arm-specific stopping rules"""
        
        arm = self.treatment_arms[arm_name]
        
        # Efficacy boundary (adjusted for multiplicity)
        efficacy_boundary = calculate_efficacy_boundary(
            arm['alpha'], 
            information_fraction(arm)
        )
        
        # Futility boundary
        futility_boundary = calculate_futility_boundary(
            conditional_power_threshold=0.20
        )
        
        test_statistic = calculate_test_statistic(
            arm['patients'], 
            self.control_arm['patients']
        )
        
        if test_statistic > efficacy_boundary:
            arm['status'] = 'graduated'
            self.alpha_spent += arm['alpha']
            return 'STOP_SUCCESS'
            
        elif test_statistic < futility_boundary:
            arm['status'] = 'dropped'
            # Don't spend α
            return 'STOP_FUTILITY'
            
        else:
            return 'CONTINUE'
```

### Reporting and Interpretation

```python
def create_trial_report(trial_data, primary_outcome, secondary_outcomes):
    """
    Generate comprehensive trial report following CONSORT guidelines
    """
    
    report = {
        'participant_flow': create_consort_diagram(trial_data),
        'baseline_characteristics': create_baseline_table(trial_data),
        'primary_analysis': analyze_primary_endpoint(trial_data, primary_outcome),
        'secondary_analyses': {},
        'safety_analysis': analyze_safety(trial_data),
        'subgroup_analyses': {}
    }
    
    # Number Needed to Treat (NNT)
    if primary_outcome['type'] == 'binary':
        arr = calculate_absolute_risk_reduction(trial_data, primary_outcome)
        nnt = 1 / arr if arr > 0 else float('inf')
        report['nnt'] = {
            'estimate': nnt,
            'ci': calculate_nnt_ci(arr, trial_data)
        }
    
    # Effect sizes with clinical interpretation
    effect_size = calculate_standardized_effect(trial_data, primary_outcome)
    report['clinical_significance'] = interpret_effect_size(
        effect_size, 
        minimal_clinically_important_difference=0.3
    )
    
    return report
```

### Common Pitfalls and Solutions

| Pitfall | Impact | Solution |
|---------|--------|----------|
| **Underpowered for subgroups** | False negatives in key populations | Pre-specify subgroups, stratify randomization |
| **Multiple testing inflation** | False positives | Hierarchical testing, gate-keeping |
| **Informative censoring** | Biased survival estimates | Sensitivity analyses, joint modeling |
| **Protocol violations** | Diluted treatment effect | ITT and per-protocol analyses |
| **Baseline imbalances** | Confounding | Covariate adjustment, stratification |
| **Missing not at random** | Biased conclusions | Multiple sensitivity analyses |

### References
- ICH E9: Statistical Principles for Clinical Trials
- Friedman, Furberg, DeMets (2015). Fundamentals of Clinical Trials
- Berry et al. (2010). Bayesian Adaptive Methods for Clinical Trials
- Piantadosi (2017). Clinical Trials: A Methodologic Perspective