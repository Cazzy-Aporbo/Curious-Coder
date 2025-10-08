# Systems Biology and Mathematical Modeling
## From Molecules to Networks: Quantitative Understanding of Life

### Intent
Biological systems are networks of interacting components with emergent properties. This document provides mathematical frameworks for modeling gene regulatory networks, metabolic pathways, signaling cascades, and whole-cell dynamics, bridging molecular mechanisms and systems-level behavior.

### The Core Principle: Dynamics Over Statics

```
Biology is not architecture, it's choreography
Static snapshot → Temporal dynamics → System behavior

dx/dt = f(x, parameters, inputs)
```

### Ordinary Differential Equations (ODEs) for Biological Systems

#### 1. Mass Action Kinetics

```python
def mass_action_kinetics():
    """
    Foundation of chemical reaction modeling
    
    A + B → C with rate constant k
    Rate = k[A][B]
    """
    
    def reaction_system(y, t, params):
        """
        Example: Protein synthesis and degradation
        mRNA → mRNA + Protein (rate: k_translate)
        Protein → ∅ (rate: k_degrade)
        """
        mRNA, protein = y
        k_translate, k_degrade = params
        
        # ODEs
        dmRNA_dt = 0  # Assuming constant mRNA
        dprotein_dt = k_translate * mRNA - k_degrade * protein
        
        return [dmRNA_dt, dprotein_dt]
    
    # Solve
    from scipy.integrate import odeint
    
    y0 = [1.0, 0.0]  # Initial: mRNA=1, protein=0
    params = [0.5, 0.1]  # Translation and degradation rates
    t = np.linspace(0, 50, 100)
    
    solution = odeint(reaction_system, y0, t, args=(params,))
    
    # Steady state (analytical)
    protein_ss = params[0] * y0[0] / params[1]
    
    return solution, protein_ss
```

#### 2. Michaelis-Menten Enzyme Kinetics

```python
def michaelis_menten_full():
    """
    Full enzyme kinetics model
    
    E + S ⇌ ES → E + P
    """
    
    def enzyme_kinetics(y, t, params):
        E, S, ES, P = y
        k1, k_1, k2 = params  # Forward, reverse, catalytic
        
        # Formation of ES complex
        v_bind = k1 * E * S - k_1 * ES
        
        # Product formation
        v_cat = k2 * ES
        
        # ODEs
        dE_dt = -v_bind + v_cat
        dS_dt = -v_bind
        dES_dt = v_bind - v_cat
        dP_dt = v_cat
        
        return [dE_dt, dS_dt, dES_dt, dP_dt]
    
    # Quasi-steady-state approximation
    def michaelis_menten_qssa(S, Vmax, Km):
        """
        Simplified MM equation
        v = Vmax * S / (Km + S)
        """
        return Vmax * S / (Km + S)
    
    # Hill equation for cooperativity
    def hill_equation(S, Vmax, K, n):
        """
        Cooperative binding
        v = Vmax * S^n / (K^n + S^n)
        """
        return Vmax * S**n / (K**n + S**n)
```

### Gene Regulatory Networks

#### 1. Boolean Networks

```python
class BooleanNetwork:
    """
    Discrete model of gene regulation
    """
    
    def __init__(self, n_genes):
        self.n_genes = n_genes
        self.state = np.random.randint(0, 2, n_genes)
        
    def define_rules(self):
        """
        Logic rules for each gene
        """
        self.rules = {
            # Gene 0: activated by gene 1 AND NOT gene 2
            0: lambda state: state[1] and not state[2],
            
            # Gene 1: activated by gene 0 OR gene 2
            1: lambda state: state[0] or state[2],
            
            # Gene 2: activated by NOT gene 0
            2: lambda state: not state[0]
        }
    
    def update_synchronous(self):
        """
        Update all genes simultaneously
        """
        new_state = np.zeros(self.n_genes, dtype=int)
        
        for gene, rule in self.rules.items():
            new_state[gene] = int(rule(self.state))
        
        self.state = new_state
        
    def update_asynchronous(self):
        """
        Update one random gene at a time
        """
        gene = np.random.randint(self.n_genes)
        self.state[gene] = int(self.rules[gene](self.state))
    
    def find_attractors(self, max_steps=1000):
        """
        Find steady states and cycles
        """
        visited_states = []
        
        for _ in range(max_steps):
            state_tuple = tuple(self.state)
            
            if state_tuple in visited_states:
                # Found cycle
                cycle_start = visited_states.index(state_tuple)
                cycle = visited_states[cycle_start:]
                return {'type': 'cycle', 'states': cycle}
            
            visited_states.append(state_tuple)
            self.update_synchronous()
        
        return {'type': 'unknown', 'states': visited_states}
```

#### 2. Continuous Gene Regulation Models

```python
def gene_regulatory_ode():
    """
    ODE model of gene regulatory network
    """
    
    def grn_dynamics(y, t, params):
        """
        Example: Toggle switch (mutual inhibition)
        """
        x1, x2 = y  # Protein concentrations
        
        # Parameters
        alpha1, alpha2 = params['production']
        beta1, beta2 = params['degradation']
        K1, K2 = params['dissociation']
        n1, n2 = params['cooperativity']
        
        # Hill function repression
        dx1_dt = alpha1 / (1 + (x2/K2)**n2) - beta1 * x1
        dx2_dt = alpha2 / (1 + (x1/K1)**n1) - beta2 * x2
        
        return [dx1_dt, dx2_dt]
    
    # Find steady states
    def find_steady_states(params):
        from scipy.optimize import fsolve
        
        def steady_state_equations(y):
            dydt = grn_dynamics(y, 0, params)
            return dydt
        
        # Multiple initial guesses to find all steady states
        steady_states = []
        for init in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            ss = fsolve(steady_state_equations, init)
            
            # Check if actually steady
            if np.allclose(steady_state_equations(ss), 0):
                # Check if already found
                if not any(np.allclose(ss, s) for s in steady_states):
                    steady_states.append(ss)
        
        return steady_states
    
    # Stability analysis
    def analyze_stability(steady_state, params):
        """
        Linear stability analysis via Jacobian
        """
        
        def jacobian(y):
            eps = 1e-8
            n = len(y)
            J = np.zeros((n, n))
            
            for i in range(n):
                y_plus = y.copy()
                y_minus = y.copy()
                y_plus[i] += eps
                y_minus[i] -= eps
                
                f_plus = grn_dynamics(y_plus, 0, params)
                f_minus = grn_dynamics(y_minus, 0, params)
                
                J[:, i] = (np.array(f_plus) - np.array(f_minus)) / (2 * eps)
            
            return J
        
        J = jacobian(steady_state)
        eigenvalues = np.linalg.eigvals(J)
        
        # Stable if all eigenvalues have negative real parts
        stable = all(e.real < 0 for e in eigenvalues)
        
        return {
            'stable': stable,
            'eigenvalues': eigenvalues,
            'type': classify_fixed_point(eigenvalues)
        }
    
    def classify_fixed_point(eigenvalues):
        """
        Classify type of fixed point
        """
        real_parts = [e.real for e in eigenvalues]
        imag_parts = [e.imag for e in eigenvalues]
        
        if all(r < 0 for r in real_parts):
            if all(i == 0 for i in imag_parts):
                return 'stable node'
            else:
                return 'stable spiral'
        elif all(r > 0 for r in real_parts):
            if all(i == 0 for i in imag_parts):
                return 'unstable node'
            else:
                return 'unstable spiral'
        else:
            return 'saddle point'
```

### Stochastic Models: When Molecule Numbers Matter

#### 1. Gillespie Algorithm

```python
def gillespie_algorithm(initial_state, reactions, propensities, 
                       stoichiometry, t_max):
    """
    Exact stochastic simulation
    """
    
    # Initialize
    t = 0
    state = initial_state.copy()
    trajectory = [(t, state.copy())]
    
    while t < t_max:
        # Calculate propensities for all reactions
        props = [prop(state) for prop in propensities]
        prop_sum = sum(props)
        
        if prop_sum == 0:
            break  # No more reactions possible
        
        # Time to next reaction (exponential distribution)
        tau = -np.log(np.random.random()) / prop_sum
        t += tau
        
        # Which reaction occurs? (weighted by propensities)
        r = np.random.random() * prop_sum
        cumsum = 0
        for i, prop in enumerate(props):
            cumsum += prop
            if r < cumsum:
                reaction_idx = i
                break
        
        # Update state
        state += stoichiometry[reaction_idx]
        trajectory.append((t, state.copy()))
    
    return trajectory

# Example: Birth-death process
def birth_death_example():
    """
    Simple gene expression model
    ∅ → mRNA (rate: k_transcribe)
    mRNA → mRNA + Protein (rate: k_translate)
    mRNA → ∅ (rate: γ_mRNA)
    Protein → ∅ (rate: γ_protein)
    """
    
    initial_state = np.array([0, 0])  # [mRNA, Protein]
    
    # Define reactions
    propensities = [
        lambda s: 10.0,           # Transcription (constant)
        lambda s: 0.5 * s[0],     # Translation
        lambda s: 0.1 * s[0],     # mRNA decay
        lambda s: 0.01 * s[1],    # Protein decay
    ]
    
    stoichiometry = [
        np.array([1, 0]),   # Transcription: +1 mRNA
        np.array([0, 1]),   # Translation: +1 Protein
        np.array([-1, 0]),  # mRNA decay: -1 mRNA
        np.array([0, -1]),  # Protein decay: -1 Protein
    ]
    
    trajectory = gillespie_algorithm(
        initial_state, None, propensities, stoichiometry, t_max=1000
    )
    
    return trajectory
```

#### 2. Chemical Master Equation

```python
def chemical_master_equation(n_max, rates):
    """
    Solve the CME for probability distribution
    
    dP/dt = MP where M is the transition matrix
    """
    
    # Create transition matrix
    # Example: Simple birth-death process
    k_birth, k_death = rates
    
    # State space: n = 0, 1, 2, ..., n_max molecules
    M = np.zeros((n_max+1, n_max+1))
    
    # Birth transitions: n → n+1
    for n in range(n_max):
        M[n+1, n] = k_birth
        M[n, n] -= k_birth
    
    # Death transitions: n → n-1
    for n in range(1, n_max+1):
        M[n-1, n] = k_death * n
        M[n, n] -= k_death * n
    
    # Steady-state distribution
    # Solve Mp = 0 with Σp = 1
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    
    # Find zero eigenvalue (steady state)
    idx = np.argmin(np.abs(eigenvalues))
    steady_state = np.real(eigenvectors[:, idx])
    steady_state = steady_state / steady_state.sum()
    
    # Analytical solution for birth-death (Poisson)
    lambda_param = k_birth / k_death
    analytical = np.array([
        np.exp(-lambda_param) * lambda_param**n / np.math.factorial(n)
        for n in range(n_max+1)
    ])
    
    return steady_state, analytical
```

### Metabolic Networks

#### 1. Flux Balance Analysis (FBA)

```python
def flux_balance_analysis(S, v_min, v_max, c):
    """
    Optimize metabolic flux distribution
    
    maximize: c^T v
    subject to: Sv = 0 (steady state)
                v_min ≤ v ≤ v_max (capacity constraints)
    
    S: stoichiometric matrix (metabolites × reactions)
    v: flux vector
    c: objective coefficients (e.g., biomass production)
    """
    from scipy.optimize import linprog
    
    n_metabolites, n_reactions = S.shape
    
    # Equality constraints: Sv = 0
    A_eq = S
    b_eq = np.zeros(n_metabolites)
    
    # Bounds
    bounds = list(zip(v_min, v_max))
    
    # Solve (minimize -c^T v to maximize c^T v)
    result = linprog(
        c=-c,  # Minimize negative for maximization
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )
    
    if result.success:
        optimal_flux = result.x
        growth_rate = -result.fun
        
        # Shadow prices (dual variables)
        shadow_prices = result.eqlin.marginals if hasattr(result, 'eqlin') else None
        
        return {
            'flux': optimal_flux,
            'growth_rate': growth_rate,
            'shadow_prices': shadow_prices
        }
    else:
        return None
```

#### 2. Elementary Flux Modes

```python
def find_elementary_modes(S, reversibilities):
    """
    Find all elementary flux modes
    Minimal sets of reactions that can operate at steady state
    """
    
    # Convert to irreversible reactions
    S_irrev, rev_mapping = make_irreversible(S, reversibilities)
    
    # Double description method
    def double_description_method(matrix):
        """
        Find extreme rays of polyhedral cone
        """
        # Implementation of DD algorithm
        # This is complex - using simplified version
        from scipy.spatial import ConvexHull
        
        # Find vertices of dual polytope
        # Then convert back to rays
        pass
    
    # Each elementary mode satisfies:
    # 1. Sv = 0 (steady state)
    # 2. v ≥ 0 (thermodynamically feasible)
    # 3. v is minimal (cannot be decomposed)
    
    return elementary_modes
```

### Signaling Networks

#### 1. Phosphorylation Cascades

```python
def mapk_cascade():
    """
    MAPK signaling cascade model
    Raf → MEK → ERK with dual phosphorylation
    """
    
    def cascade_odes(y, t, params):
        """
        Variables: [Raf, Raf*, MEK, MEK*, MEK**, ERK, ERK*, ERK**]
        * = phosphorylated, ** = doubly phosphorylated
        """
        Raf, RafP, MEK, MEKP, MEKPP, ERK, ERKP, ERKPP = y
        
        # Extract parameters
        k = params['kinase_rates']
        p = params['phosphatase_rates']
        
        # Raf activation (simplified)
        v1 = k[0] * Raf - p[0] * RafP
        
        # MEK phosphorylation by Raf*
        v2 = k[1] * RafP * MEK / (params['Km'][0] + MEK)
        v3 = k[2] * RafP * MEKP / (params['Km'][1] + MEKP)
        
        # MEK dephosphorylation
        v4 = p[1] * MEKP
        v5 = p[2] * MEKPP
        
        # ERK phosphorylation by MEK**
        v6 = k[3] * MEKPP * ERK / (params['Km'][2] + ERK)
        v7 = k[4] * MEKPP * ERKP / (params['Km'][3] + ERKP)
        
        # ERK dephosphorylation
        v8 = p[3] * ERKP
        v9 = p[4] * ERKPP
        
        # ODEs
        dRaf_dt = -v1
        dRafP_dt = v1
        dMEK_dt = -v2 + v4
        dMEKP_dt = v2 - v3 - v4 + v5
        dMEKPP_dt = v3 - v5
        dERK_dt = -v6 + v8
        dERKP_dt = v6 - v7 - v8 + v9
        dERKPP_dt = v7 - v9
        
        return [dRaf_dt, dRafP_dt, dMEK_dt, dMEKP_dt, dMEKPP_dt,
                dERK_dt, dERKP_dt, dERKPP_dt]
    
    # Ultrasensitivity analysis
    def dose_response(input_range, params):
        """
        Calculate input-output relationship
        """
        outputs = []
        
        for input_level in input_range:
            # Set Raf activation level
            params['kinase_rates'][0] = input_level
            
            # Find steady state
            y0 = np.ones(8) * 0.1
            t = np.linspace(0, 1000, 2)
            
            # Solve to steady state
            sol = odeint(cascade_odes, y0, t, args=(params,))
            steady_state = sol[-1]
            
            # Output is ERK**
            outputs.append(steady_state[7])
        
        # Calculate Hill coefficient (ultrasensitivity)
        from scipy.optimize import curve_fit
        
        def hill_func(x, vmax, k, n):
            return vmax * x**n / (k**n + x**n)
        
        popt, _ = curve_fit(hill_func, input_range, outputs)
        hill_coefficient = popt[2]
        
        return outputs, hill_coefficient
```

#### 2. Calcium Oscillations

```python
def calcium_oscillator():
    """
    Calcium-induced calcium release model
    """
    
    def calcium_dynamics(y, t, params):
        Ca_cyt, Ca_er, IP3 = y
        
        # Parameters
        v_in = params['influx']
        v_out = params['efflux']
        k_er = params['er_leak']
        v_rel = params['release_max']
        k_ip3 = params['ip3_sensitivity']
        k_ca = params['ca_sensitivity']
        k_pump = params['pump_rate']
        
        # IP3 receptor (simplified)
        h_inf = k_ip3 / (k_ip3 + IP3)
        
        # Calcium release from ER
        J_release = v_rel * (IP3/(IP3+k_ip3)) * (Ca_cyt/(Ca_cyt+k_ca)) * h_inf * Ca_er
        
        # SERCA pump
        J_pump = k_pump * Ca_cyt**2 / (1 + Ca_cyt**2)
        
        # Leak from ER
        J_leak = k_er * (Ca_er - Ca_cyt)
        
        # ODEs
        dCa_cyt_dt = v_in - v_out*Ca_cyt + J_release - J_pump + J_leak
        dCa_er_dt = -J_release + J_pump - J_leak
        dIP3_dt = params['ip3_production'] - params['ip3_decay'] * IP3
        
        return [dCa_cyt_dt, dCa_er_dt, dIP3_dt]
    
    # Bifurcation analysis
    def bifurcation_analysis(param_range, param_name, params):
        """
        Track steady states as parameter varies
        """
        
        steady_states = []
        stability = []
        
        for param_value in param_range:
            params[param_name] = param_value
            
            # Find steady states
            from scipy.optimize import fsolve
            
            # Multiple initial conditions
            for init in [[0.1, 1, 0.1], [0.5, 5, 0.5], [1, 10, 1]]:
                ss = fsolve(lambda y: calcium_dynamics(y, 0, params), init)
                
                # Check if valid
                if all(s >= 0 for s in ss):
                    # Check stability
                    J = numerical_jacobian(calcium_dynamics, ss, params)
                    eigenvalues = np.linalg.eigvals(J)
                    is_stable = all(e.real < 0 for e in eigenvalues)
                    
                    steady_states.append((param_value, ss[0]))  # Ca_cyt
                    stability.append(is_stable)
        
        return steady_states, stability
```

### Spatial Models: Reaction-Diffusion

```python
def reaction_diffusion_1d(L, nx, T, dt, D, reaction_func):
    """
    Solve reaction-diffusion PDE
    ∂u/∂t = D∇²u + f(u)
    """
    
    dx = L / nx
    x = np.linspace(0, L, nx)
    
    # Initial condition
    u = np.exp(-((x - L/2)**2) / (2*0.1**2))  # Gaussian
    
    # Diffusion matrix (second derivative)
    diffusion_matrix = np.diag(-2*np.ones(nx)) + \
                      np.diag(np.ones(nx-1), 1) + \
                      np.diag(np.ones(nx-1), -1)
    diffusion_matrix = D * diffusion_matrix / dx**2
    
    # Time evolution
    u_history = [u.copy()]
    
    for _ in range(int(T/dt)):
        # Reaction term
        reaction = reaction_func(u)
        
        # Diffusion term (finite differences)
        diffusion = diffusion_matrix @ u
        
        # Update (forward Euler)
        u = u + dt * (diffusion + reaction)
        
        # Boundary conditions (no-flux)
        u[0] = u[1]
        u[-1] = u[-2]
        
        u_history.append(u.copy())
    
    return np.array(u_history)

# Turing patterns
def turing_pattern_2d():
    """
    Reaction-diffusion system forming patterns
    Activator-inhibitor model
    """
    
    def gray_scott_model(u, v, params):
        """
        Gray-Scott model
        u: substrate, v: autocatalyst
        """
        F, k = params['feed'], params['kill']
        
        reaction_u = -u * v**2 + F * (1 - u)
        reaction_v = u * v**2 - (F + k) * v
        
        return reaction_u, reaction_v
    
    # 2D simulation
    def simulate_2d(nx, ny, T, dt, Du, Dv, params):
        # Initialize
        u = np.ones((nx, ny))
        v = np.zeros((nx, ny))
        
        # Add perturbation
        center = (nx//2, ny//2)
        u[center[0]-5:center[0]+5, center[1]-5:center[1]+5] = 0.5
        v[center[0]-5:center[0]+5, center[1]-5:center[1]+5] = 0.25
        
        # Laplacian operator
        def laplacian_2d(field):
            laplacian = np.zeros_like(field)
            laplacian[1:-1, 1:-1] = (
                field[2:, 1:-1] + field[:-2, 1:-1] +
                field[1:-1, 2:] + field[1:-1, :-2] -
                4 * field[1:-1, 1:-1]
            )
            return laplacian
        
        # Time evolution
        for _ in range(int(T/dt)):
            # Reaction terms
            reaction_u, reaction_v = gray_scott_model(u, v, params)
            
            # Diffusion terms
            diffusion_u = Du * laplacian_2d(u)
            diffusion_v = Dv * laplacian_2d(v)
            
            # Update
            u += dt * (diffusion_u + reaction_u)
            v += dt * (diffusion_v + reaction_v)
        
        return u, v
```

### Parameter Estimation and Model Fitting

```python
def parameter_estimation_framework(model, data, method='mle'):
    """
    Estimate parameters from experimental data
    """
    
    if method == 'mle':
        # Maximum Likelihood Estimation
        def negative_log_likelihood(params, data):
            # Simulate model with parameters
            simulation = model(params)
            
            # Assume Gaussian noise
            residuals = data - simulation
            sigma = np.std(residuals)
            
            # Log-likelihood
            log_likelihood = -0.5 * len(data) * np.log(2*np.pi*sigma**2)
            log_likelihood -= 0.5 * np.sum(residuals**2) / sigma**2
            
            return -log_likelihood
        
        from scipy.optimize import minimize
        
        result = minimize(
            negative_log_likelihood,
            initial_guess,
            args=(data,),
            method='L-BFGS-B'
        )
        
        return result.x
    
    elif method == 'bayesian':
        # Bayesian inference using MCMC
        import emcee
        
        def log_posterior(params, data):
            # Prior
            log_prior = 0
            for i, p in enumerate(params):
                if prior_bounds[i][0] <= p <= prior_bounds[i][1]:
                    log_prior += 0  # Uniform prior
                else:
                    return -np.inf
            
            # Likelihood
            simulation = model(params)
            residuals = data - simulation
            sigma = 0.1  # Assumed noise level
            
            log_likelihood = -0.5 * np.sum((residuals/sigma)**2)
            
            return log_prior + log_likelihood
        
        # MCMC sampling
        n_walkers = 32
        n_dim = len(initial_guess)
        
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_posterior, args=(data,)
        )
        
        # Initialize walkers
        pos = initial_guess + 1e-4 * np.random.randn(n_walkers, n_dim)
        
        # Run MCMC
        sampler.run_mcmc(pos, n_steps=5000)
        
        # Get posterior samples
        samples = sampler.get_chain(discard=1000, flat=True)
        
        return {
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'samples': samples
        }
```

### Sensitivity Analysis

```python
def sensitivity_analysis(model, params, method='local'):
    """
    Analyze parameter sensitivity
    """
    
    if method == 'local':
        # Local sensitivity (derivatives)
        sensitivities = {}
        baseline = model(params)
        
        for param_name, param_value in params.items():
            # Perturb parameter
            delta = 0.01 * param_value
            params_plus = params.copy()
            params_plus[param_name] = param_value + delta
            
            output_plus = model(params_plus)
            
            # Sensitivity coefficient
            S = (output_plus - baseline) / (delta * baseline) * param_value
            sensitivities[param_name] = S
        
        return sensitivities
    
    elif method == 'global':
        # Sobol sensitivity indices
        from SALib.sample import sobol
        from SALib.analyze import sobol as sobol_analyze
        
        # Define parameter ranges
        problem = {
            'num_vars': len(params),
            'names': list(params.keys()),
            'bounds': [[p*0.5, p*1.5] for p in params.values()]
        }
        
        # Generate samples
        param_samples = sobol.sample(problem, 1024)
        
        # Run model for each sample
        outputs = []
        for sample in param_samples:
            outputs.append(model(dict(zip(params.keys(), sample))))
        
        # Calculate Sobol indices
        Si = sobol_analyze.analyze(problem, np.array(outputs))
        
        return {
            'first_order': Si['S1'],  # Main effects
            'total_order': Si['ST']    # Including interactions
        }
```

### Model Reduction Techniques

```python
def model_reduction(full_model, threshold=0.01):
    """
    Reduce model complexity while preserving behavior
    """
    
    # 1. Time-scale separation
    def identify_time_scales(jacobian):
        """
        Identify fast and slow variables
        """
        eigenvalues = np.linalg.eigvals(jacobian)
        time_scales = 1 / np.abs(eigenvalues.real)
        
        # Separate fast and slow
        fast_threshold = np.percentile(time_scales, 25)
        fast_vars = time_scales < fast_threshold
        slow_vars = ~fast_vars
        
        return fast_vars, slow_vars
    
    # 2. Quasi-steady-state approximation
    def apply_qssa(model, fast_vars):
        """
        Set fast variables to steady state
        """
        def reduced_model(slow_vars, t, params):
            # Solve for fast variables algebraically
            # This is problem-specific
            pass
        
        return reduced_model
    
    # 3. Balanced truncation
    def balanced_truncation(A, B, C):
        """
        Reduce linear system while preserving input-output behavior
        """
        from scipy.linalg import solve_lyapunov
        
        # Controllability Gramian
        Wc = solve_lyapunov(A, -B @ B.T)
        
        # Observability Gramian
        Wo = solve_lyapunov(A.T, -C.T @ C)
        
        # Hankel singular values
        hankel = np.sqrt(np.linalg.eigvals(Wc @ Wo))
        
        # Keep states with large singular values
        n_keep = np.sum(hankel > threshold)
        
        # Transformation matrix
        # ... (complex implementation)
        
        return reduced_system
```

### Whole-Cell Models

```python
class WholeCellModel:
    """
    Integrate multiple cellular processes
    """
    
    def __init__(self):
        self.modules = {
            'metabolism': MetabolismModule(),
            'gene_expression': GeneExpressionModule(),
            'dna_replication': DNAReplicationModule(),
            'cell_division': CellDivisionModule()
        }
        
    def simulate(self, t_max, dt):
        """
        Multi-scale simulation
        """
        t = 0
        state = self.initialize_state()
        
        while t < t_max:
            # Different time scales for different processes
            
            # Fast: Metabolism (milliseconds)
            for _ in range(10):
                state = self.modules['metabolism'].update(state, dt/10)
            
            # Medium: Gene expression (minutes)
            state = self.modules['gene_expression'].update(state, dt)
            
            # Slow: DNA replication (hours)
            if t % 3600 == 0:  # Every hour
                state = self.modules['dna_replication'].update(state, 3600)
            
            # Check for cell division
            if self.check_division_criteria(state):
                state = self.modules['cell_division'].divide(state)
            
            t += dt
        
        return state
```

### Common Pitfalls and Solutions

| Pitfall | Consequence | Solution |
|---------|------------|----------|
| **Ignoring stochasticity** | Missing noise-driven behaviors | Use Gillespie for small numbers |
| **Wrong time scales** | Numerical instability | Adaptive time stepping |
| **Over-parameterization** | Non-identifiable models | Sensitivity analysis, reduce parameters |
| **Assuming mass action** | Wrong kinetics | Use appropriate rate laws |
| **Neglecting space** | Missing patterns | Add diffusion terms |
| **Point estimates only** | Ignoring uncertainty | Bayesian inference |

### References
- Alon (2019). An Introduction to Systems Biology
- Klipp et al. (2016). Systems Biology: A Textbook
- Strogatz (2018). Nonlinear Dynamics and Chaos
- Wilkinson (2011). Stochastic Modelling for Systems Biology