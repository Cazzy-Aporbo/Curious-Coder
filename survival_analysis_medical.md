# Survival Analysis in Medicine and Biology
## When Time-to-Event Matters More Than Binary Outcomes

### Intent
Survival analysis handles time-to-event data with censoring—when we observe some but not all event times. Critical for clinical trials, disease progression, and biological processes where timing carries mechanistic information that binary outcomes discard.

### Mathematical Foundation

#### Core Concepts

**Survival Function:** S(t) = P(T > t) = probability of surviving past time t

**Hazard Function:** h(t) = lim_{Δt→0} P(t ≤ T < t+Δt | T ≥ t)/Δt
- Instantaneous risk of event at time t, given survival until t
- h(t) = f(t)/S(t) = -d[log S(t)]/dt

**Cumulative Hazard:** H(t) = ∫₀ᵗ h(u)du = -log S(t)

**Fundamental Relationship:**
```
S(t) = exp(-H(t)) = exp(-∫₀ᵗ h(u)du)
```

#### The Censoring Problem

Dataset: {(Tᵢ, δᵢ, Xᵢ)}ⁿᵢ₌₁ where:
- Tᵢ = min(T*ᵢ, Cᵢ) observed time
- T*ᵢ = true event time
- Cᵢ = censoring time
- δᵢ = 𝟙[T*ᵢ ≤ Cᵢ] event indicator
- Xᵢ = covariates

**Types of Censoring:**
1. **Right censoring:** Event hasn't occurred by study end
2. **Left censoring:** Event occurred before observation began
3. **Interval censoring:** Event occurred between observations
4. **Informative censoring:** Censoring related to outcome (violates assumptions)

### Core Methods

#### 1. Kaplan-Meier Estimator (Non-parametric)

**Formulation:**
```
Ŝ(t) = ∏_{tⱼ≤t} (1 - dⱼ/nⱼ)

where:
- tⱼ: ordered unique event times
- dⱼ: number of events at tⱼ
- nⱼ: number at risk just before tⱼ
```

**Variance (Greenwood's formula):**
```
Var[Ŝ(t)] = Ŝ(t)² × Σ_{tⱼ≤t} [dⱼ/(nⱼ(nⱼ-dⱼ))]
```

**Confidence Intervals (log-log transformation for boundedness):**
```
CI = exp{±z_{α/2} × √Var[log(-log Ŝ(t))]}
```

#### 2. Cox Proportional Hazards Model (Semi-parametric)

**Model:**
```
h(t|X) = h₀(t) × exp(X^T β)

where:
- h₀(t): baseline hazard (unspecified)
- exp(X^T β): relative hazard
```

**Partial Likelihood (Cox, 1975):**
```
L(β) = ∏ᵢ:δᵢ=1 [exp(Xᵢ^T β) / Σ_{j∈R(tᵢ)} exp(Xⱼ^T β)]

where R(t) = {j: Tⱼ ≥ t} is the risk set
```

**No need to specify h₀(t)!** This is the key innovation.

**Parameter Estimation:**
```python
# Newton-Raphson for β
β_new = β_old + I(β_old)⁻¹ × U(β_old)

where:
U(β) = ∂log L/∂β  # Score function
I(β) = -∂²log L/∂β∂β^T  # Information matrix
```

#### 3. Accelerated Failure Time Models (Parametric)

**Model:** log T = X^T β + σε

**Common Distributions:**

| Distribution | Hazard Shape | Use Case | Survival Function |
|--------------|--------------|----------|-------------------|
| **Exponential** | Constant | Memoryless processes | S(t) = exp(-λt) |
| **Weibull** | Monotonic | Aging/wear processes | S(t) = exp(-(t/λ)^k) |
| **Log-normal** | Non-monotonic | Biological growth | S(t) = 1 - Φ(log(t)-μ)/σ) |
| **Log-logistic** | Non-monotonic | Immunological response | S(t) = 1/(1+(t/α)^β) |

**Parameter interpretation:**
- Cox: exp(β) = hazard ratio
- AFT: exp(β) = time ratio (acceleration factor)

### Assumptions & Diagnostics

#### 1. Proportional Hazards Assumption (Cox Model)

**Mathematical Statement:** h₁(t)/h₂(t) = constant over time

**Schoenfeld Residuals Test:**
```python
def test_proportional_hazards(cox_model, data):
    """
    H₀: β(t) = β (constant over time)
    """
    residuals = cox_model.compute_schoenfeld_residuals()
    
    for covariate in covariates:
        # Residuals should have zero slope over time
        correlation = spearmanr(residuals[covariate], event_times)
        p_value = correlation.pvalue
        
        if p_value < 0.05:
            print(f"{covariate} violates PH assumption")
            
    # Visual check
    plot_schoenfeld(residuals, transform='km')  # Should be horizontal
```

**Solutions for PH Violations:**
1. **Stratification:** h(t|X, stratum=s) = h₀ₛ(t) × exp(X^T β)
2. **Time-varying coefficients:** β(t) = β₀ + β₁ × g(t)
3. **Time-dependent covariates:** X(t)

#### 2. Independent Censoring Assumption

**Requirement:** P(C > t | T* = s, X) = P(C > t | X) for all s, t

**Violations & Solutions:**

| Scenario | Problem | Solution |
|----------|---------|----------|
| **Competing risks** | Death from other causes | Cause-specific or subdistribution hazards |
| **Dependent censoring** | Sicker patients drop out | Inverse probability weighting |
| **Cure fraction** | Subset never experiences event | Mixture cure models |

### Advanced Methods for Complex Biology

#### 1. Time-Dependent Covariates

**Extended Cox Model:**
```
h(t|X(t)) = h₀(t) × exp[Σⱼ βⱼXⱼ(t)]
```

**Implementation Challenges:**
```python
# Data structure: Start-stop format
# Each row represents an interval for one subject
data_extended = [
    (id=1, tstart=0,  tstop=30, X_bp=120, event=0),
    (id=1, tstart=30, tstop=65, X_bp=140, event=1),
    # ...
]

# Partial likelihood modification
L(β) = ∏ᵢ:δᵢ=1 [exp(Xᵢ(tᵢ)^T β) / Σ_{j∈R(tᵢ)} exp(Xⱼ(tᵢ)^T β)]
```

#### 2. Competing Risks

**Cause-Specific Hazard:**
```
h_k(t) = lim_{Δt→0} P(t ≤ T < t+Δt, cause=k | T ≥ t)/Δt
```

**Cumulative Incidence Function:**
```
CIF_k(t) = P(T ≤ t, cause=k) = ∫₀ᵗ S(u) × h_k(u) du

where S(t) = exp(-Σₖ H_k(t))
```

**Fine-Gray Model (Subdistribution hazard):**
```
h*_k(t) = lim_{Δt→0} P(t ≤ T < t+Δt, cause=k | T ≥ t or (T < t and cause ≠ k))/Δt
```

#### 3. Recurrent Events

**Counting Process Formulation:**
```
N(t) = number of events by time t
dN(t) = N(t) - N(t⁻)

Intensity: λ(t) = E[dN(t) | history]
```

**Models:**
- **Andersen-Gill:** All events contribute equally
- **Prentice-Williams-Peterson:** Risk set changes after each event
- **Wei-Lin-Weissfeld:** Marginal model for ordered events

#### 4. Joint Models for Longitudinal and Survival Data

**Application:** Biomarker trajectory affects survival

```
Longitudinal: Y_i(t) = X_i(t)^T β + Z_i(t)^T b_i + ε_i(t)
Survival: h_i(t) = h₀(t) × exp(W_i^T γ + αf(Y_i(t)))

where f(·) links biomarker to hazard
```

### Medical Applications & Pitfalls

| Clinical Context | Key Considerations | Common Errors |
|-----------------|-------------------|---------------|
| **Cancer trials** | Delayed treatment effect | Using logrank when hazards cross |
| **Cardiovascular** | Multiple events (MI, stroke, death) | Ignoring recurrent events |
| **Transplantation** | Time-dependent exposure | Immortal time bias |
| **Infectious disease** | Cure fraction | Standard Cox assumes everyone susceptible |
| **Pharmacokinetics** | Interval censoring | Treating as right-censored |
| **Genetic studies** | Age-dependent penetrance | Not accounting for ascertainment |

### Sample Size Calculation

**Logrank Test Power:**
```python
def sample_size_survival(hr, p_event_control, alpha=0.05, power=0.8, ratio=1):
    """
    Schoenfeld formula for Cox/logrank
    
    hr: hazard ratio
    p_event_control: probability of event in control
    ratio: allocation ratio (treatment:control)
    """
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # Number of events needed
    d = ((z_alpha + z_beta)**2 * (1 + ratio)**2) / (ratio * log(hr)**2)
    
    # Total sample size
    n = d / (p_event_control * (1 + ratio*hr)/(1 + ratio))
    
    return ceil(n)
```

**Accounting for Censoring:**
```
n_adjusted = n / (1 - dropout_rate)
```

### Implementation Template

```python
class ClinicalSurvivalAnalysis:
    """Complete survival analysis pipeline for clinical data"""
    
    def __init__(self, data, time_col, event_col, covariate_cols):
        self.data = data
        self.time = time_col
        self.event = event_col
        self.covariates = covariate_cols
        
    def exploratory_analysis(self):
        """Initial exploration"""
        # 1. Censoring proportion
        censoring_rate = 1 - self.data[self.event].mean()
        
        # 2. Follow-up time distribution
        median_followup = self.data.loc[
            self.data[self.event] == 0, self.time
        ].median()
        
        # 3. Event time distribution
        km_fit = KaplanMeierFitter()
        km_fit.fit(self.data[self.time], self.data[self.event])
        median_survival = km_fit.median_survival_time_
        
        return {
            'censoring_rate': censoring_rate,
            'median_followup': median_followup,
            'median_survival': median_survival
        }
    
    def fit_models(self):
        """Fit hierarchy of models"""
        results = {}
        
        # 1. Non-parametric (Kaplan-Meier)
        km = KaplanMeierFitter()
        km.fit(self.data[self.time], self.data[self.event])
        results['km'] = km
        
        # 2. Semi-parametric (Cox)
        cox = CoxPHFitter()
        cox.fit(self.data[[self.time, self.event] + self.covariates],
                self.time, self.event)
        results['cox'] = cox
        
        # 3. Test proportional hazards
        ph_test = proportional_hazard_test(cox, self.data)
        if ph_test.p_value < 0.05:
            # Use stratified Cox or time-varying
            cox_strat = CoxPHFitter()
            cox_strat.fit(..., strata=problematic_vars)
            results['cox_stratified'] = cox_strat
        
        # 4. Parametric alternatives
        for dist_name in ['weibull', 'lognormal', 'loglogistic']:
            aft = ParametricSurvival(dist_name)
            aft.fit(self.data)
            results[f'aft_{dist_name}'] = aft
            
        return results
    
    def validate_model(self, model, method='bootstrap', n_iterations=100):
        """Internal validation"""
        if method == 'bootstrap':
            c_indices = []
            for _ in range(n_iterations):
                boot_idx = np.random.choice(len(self.data), len(self.data))
                boot_data = self.data.iloc[boot_idx]
                
                model_boot = model.__class__()
                model_boot.fit(boot_data)
                c_indices.append(model_boot.concordance_index_)
            
            return {
                'c_index_mean': np.mean(c_indices),
                'c_index_std': np.std(c_indices),
                'optimism': model.concordance_index_ - np.mean(c_indices)
            }
    
    def clinical_predictions(self, model, times=[1, 3, 5]):
        """Clinically relevant predictions"""
        predictions = {}
        
        for t in times:
            # Survival probability at time t
            predictions[f'{t}yr_survival'] = model.predict_survival_function(
                self.data[self.covariates], times=[t*365]
            )
            
            # Risk stratification
            risk_scores = model.predict_partial_hazard(self.data[self.covariates])
            tertiles = pd.qcut(risk_scores, 3, labels=['Low', 'Medium', 'High'])
            predictions[f'{t}yr_risk_groups'] = tertiles
            
        return predictions
```

### Reporting Guidelines

**Essential Elements (per STROBE):**
- [ ] Number at risk at key timepoints
- [ ] Median follow-up time
- [ ] Censoring proportion and pattern
- [ ] Survival curve with confidence bands
- [ ] Number of events per covariate (≥10 rule)
- [ ] Assumption checks (PH, linearity)
- [ ] Handling of missing data
- [ ] Sensitivity analyses

**Advanced Reporting:**
- [ ] Time-dependent ROC/AUC
- [ ] Calibration plots
- [ ] Decision curve analysis
- [ ] Restricted mean survival time
- [ ] Life-years gained

### References
- Cox, D.R. (1972). Regression models and life-tables
- Therneau, T.M. & Grambsch, P.M. (2000). Modeling Survival Data
- Royston, P. & Parmar, M.K. (2013). Restricted mean survival time
- Putter, H. et al. (2007). Tutorial in biostatistics: competing risks and multi-state models