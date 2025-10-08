# Microbiome Analysis: Compositional Data in Ecological Context
## When Your Data Sums to One and Everything Correlates with Everything

### Intent
Microbiome data violates nearly every assumption of standard statistics: it's compositional (sums to constant), sparse (mostly zeros), overdispersed (variance >> mean), and phylogenetically structured. This document provides rigorous frameworks for analyzing microbial communities while respecting these constraints.

### The Fundamental Problem: Compositional Data

**The Closure Problem:**
```
Observed: p_i = c_i / Σc_j (relative abundance)
True: c_i = actual cell counts (unknown)

Key insight: If species A doubles, all others appear to decrease!
This creates spurious negative correlations.
```

### Mathematical Framework for Compositional Data

#### 1. The Simplex and Aitchison Geometry

**Data lives on the simplex:**
```
S^D = {x ∈ ℝ^D : x_i > 0, Σx_i = 1}
```

**Aitchison operations:**
```python
def compositional_operations():
    """
    Operations that respect the simplex
    """
    
    # Perturbation (⊕) - multiplication in simplex
    def perturb(x, y):
        return closure(x * y)
    
    # Power transformation (⊙) - scaling in simplex  
    def power(x, a):
        return closure(x ** a)
    
    # Closure - project back to simplex
    def closure(x):
        return x / x.sum()
    
    # Aitchison inner product
    def aitchison_inner_product(x, y):
        log_x = np.log(x)
        log_y = np.log(y)
        centered_x = log_x - log_x.mean()
        centered_y = log_y - log_y.mean()
        return np.dot(centered_x, centered_y)
    
    # Aitchison norm (distance to neutral element)
    def aitchison_norm(x):
        return np.sqrt(aitchison_inner_product(x, x))
    
    # Aitchison distance
    def aitchison_distance(x, y):
        return aitchison_norm(perturb(x, power(y, -1)))
```

#### 2. Log-Ratio Transformations

**Centered Log-Ratio (CLR):**
```python
def clr_transform(X):
    """
    CLR: clr(x) = log(x/g(x))
    where g(x) = geometric mean
    """
    # Add pseudocount for zeros
    X_pseudo = X + 1e-10
    
    # Geometric mean per sample
    geometric_mean = np.exp(np.log(X_pseudo).mean(axis=1))
    
    # CLR transform
    X_clr = np.log(X_pseudo / geometric_mean[:, np.newaxis])
    
    return X_clr
```

**Additive Log-Ratio (ALR):**
```python
def alr_transform(X, reference_idx=-1):
    """
    ALR: alr(x) = log(x_i/x_ref) for i ≠ ref
    """
    X_pseudo = X + 1e-10
    
    # Use last component as reference
    X_ref = X_pseudo[:, reference_idx:reference_idx+1]
    
    # Remove reference from numerator
    X_num = np.delete(X_pseudo, reference_idx, axis=1)
    
    # ALR transform
    X_alr = np.log(X_num / X_ref)
    
    return X_alr
```

**Isometric Log-Ratio (ILR):**
```python
def ilr_transform(X):
    """
    ILR: Orthonormal basis for simplex
    """
    n_samples, n_features = X.shape
    
    # Create orthonormal basis (Gram-Schmidt)
    def create_ilr_basis(D):
        basis = np.zeros((D, D-1))
        for i in range(D-1):
            # Helmert contrast
            basis[:i+1, i] = 1/np.sqrt((i+1)*(i+2))
            basis[i+1, i] = -(i+1)/np.sqrt((i+1)*(i+2))
        return basis
    
    V = create_ilr_basis(n_features)
    
    # ILR transform
    X_pseudo = X + 1e-10
    X_log = np.log(X_pseudo)
    X_ilr = X_log @ V
    
    return X_ilr, V
```

### Zero Handling in Microbiome Data

#### 1. Types of Zeros

```python
def classify_zeros(count_matrix, metadata):
    """
    Distinguish biological vs technical zeros
    """
    
    zeros_classification = {}
    
    for i, taxon in enumerate(count_matrix.columns):
        zero_samples = count_matrix[taxon] == 0
        
        # Structural zero: Never present in this environment
        if metadata['environment'].isin(['extreme_pH', 'high_temp']).any():
            if taxon in thermophile_database:
                zeros_classification[taxon] = 'structural'
                
        # Sampling zero: Below detection
        elif zero_samples.sum() / len(count_matrix) < 0.3:
            # Present in most samples -> likely sampling zeros
            zeros_classification[taxon] = 'sampling'
            
        # Count zero: True absence
        else:
            # Use hurdle model to test
            from statsmodels.discrete.count_model import ZeroInflatedPoisson
            
            model = ZeroInflatedPoisson(
                count_matrix[taxon],
                exog=metadata[['pH', 'temperature']],
                exog_infl=metadata[['sequencing_depth']]
            )
            result = model.fit()
            
            if result.params['inflate_const'] > 2:
                zeros_classification[taxon] = 'excess_zeros'
            else:
                zeros_classification[taxon] = 'count_zeros'
    
    return zeros_classification
```

#### 2. Zero Replacement Strategies

```python
def zero_replacement(X, method='multiplicative', delta=None):
    """
    Handle zeros before log-ratio transformation
    """
    
    if method == 'pseudocount':
        # Simple but biased
        return X + 1
    
    elif method == 'multiplicative':
        # Martín-Fernández method
        if delta is None:
            delta = 0.65 * X[X > 0].min()
        
        X_imputed = X.copy()
        for i in range(len(X)):
            if (X[i] == 0).any():
                n_zeros = (X[i] == 0).sum()
                n_nonzeros = len(X[i]) - n_zeros
                
                # Replace zeros
                X_imputed[i][X[i] == 0] = delta
                
                # Adjust non-zeros to maintain closure
                X_imputed[i][X[i] > 0] *= (1 - n_zeros * delta) / X[i][X[i] > 0].sum()
        
        return X_imputed
    
    elif method == 'gbm':
        # Geometric Bayesian multiplicative
        from scipy.stats import dirichlet
        
        # Prior from non-zero samples
        prior = X[X.sum(axis=1) > 0].mean(axis=0) + 0.5
        
        X_imputed = []
        for x in X:
            if (x == 0).any():
                # Sample from posterior
                posterior = dirichlet.rvs(alpha=x + prior, size=1)[0]
                X_imputed.append(posterior)
            else:
                X_imputed.append(x)
        
        return np.array(X_imputed)
```

### Alpha Diversity: Within-Sample Diversity

```python
def calculate_alpha_diversity(counts, metrics=['shannon', 'simpson', 'observed']):
    """
    Calculate various diversity indices
    """
    
    results = {}
    
    # Convert to proportions
    props = counts / counts.sum()
    props = props[props > 0]  # Remove zeros for log
    
    if 'shannon' in metrics:
        # H = -Σ p_i * ln(p_i)
        results['shannon'] = -np.sum(props * np.log(props))
    
    if 'simpson' in metrics:
        # D = 1 - Σ p_i²
        results['simpson'] = 1 - np.sum(props ** 2)
    
    if 'observed' in metrics:
        # Species richness
        results['observed'] = (counts > 0).sum()
    
    if 'chao1' in metrics:
        # Chao1 estimator for true richness
        S_obs = (counts > 0).sum()
        f1 = (counts == 1).sum()  # Singletons
        f2 = (counts == 2).sum()  # Doubletons
        
        if f2 > 0:
            results['chao1'] = S_obs + (f1**2) / (2*f2)
        else:
            results['chao1'] = S_obs + f1*(f1-1)/2
    
    if 'faith_pd' in metrics:
        # Faith's Phylogenetic Diversity
        # Requires phylogenetic tree
        results['faith_pd'] = calculate_faith_pd(counts, tree)
    
    # Hill numbers (general diversity)
    for q in [0, 1, 2]:
        if q == 0:
            results[f'hill_q{q}'] = (counts > 0).sum()
        elif q == 1:
            results[f'hill_q{q}'] = np.exp(results['shannon'])
        else:
            results[f'hill_q{q}'] = (np.sum(props ** q)) ** (1/(1-q))
    
    return results
```

### Beta Diversity: Between-Sample Diversity

```python
def calculate_beta_diversity(count_matrix, metric='braycurtis', 
                           tree=None, rarefaction_depth=None):
    """
    Calculate pairwise dissimilarities
    """
    
    # Rarefaction for equal sampling depth
    if rarefaction_depth:
        count_matrix = rarefy(count_matrix, rarefaction_depth)
    
    if metric == 'braycurtis':
        # Bray-Curtis: Σ|x_i - y_i| / Σ(x_i + y_i)
        from scipy.spatial.distance import braycurtis
        dist_matrix = pdist(count_matrix, metric=braycurtis)
        
    elif metric == 'jaccard':
        # Binary Jaccard
        binary_matrix = (count_matrix > 0).astype(int)
        dist_matrix = pdist(binary_matrix, metric='jaccard')
        
    elif metric == 'aitchison':
        # Compositional distance
        clr_matrix = clr_transform(count_matrix)
        dist_matrix = pdist(clr_matrix, metric='euclidean')
        
    elif metric == 'unifrac' and tree:
        # UniFrac: phylogenetic distance
        dist_matrix = calculate_unifrac(count_matrix, tree, weighted=True)
        
    elif metric == 'philr' and tree:
        # Phylogenetic ILR
        philr_matrix = philr_transform(count_matrix, tree)
        dist_matrix = pdist(philr_matrix, metric='euclidean')
    
    return squareform(dist_matrix)
```

### Differential Abundance Testing

#### 1. DESeq2-style Approach

```python
def deseq2_for_microbiome(count_matrix, metadata, formula):
    """
    Negative binomial GLM with size factors
    """
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    
    # Estimate size factors (geometric means)
    def estimate_size_factors(counts):
        # Remove zeros for geometric mean
        pseudo_counts = counts.replace(0, np.nan)
        
        # Geometric mean per taxon
        geo_means = np.exp(np.log(pseudo_counts).mean(axis=0))
        
        # Size factor per sample
        size_factors = []
        for i in range(len(counts)):
            non_zero = counts.iloc[i] > 0
            if non_zero.any():
                ratio = counts.iloc[i][non_zero] / geo_means[non_zero]
                size_factors.append(np.median(ratio))
            else:
                size_factors.append(1.0)
        
        return np.array(size_factors)
    
    size_factors = estimate_size_factors(count_matrix)
    
    # Fit negative binomial for each taxon
    results = {}
    
    for taxon in count_matrix.columns:
        counts = count_matrix[taxon].values
        
        # Create design matrix
        design = sm.add_constant(metadata[formula.split('~')[1].strip()])
        
        # Offset for size factors
        offset = np.log(size_factors)
        
        try:
            # Negative binomial regression
            model = sm.GLM(
                counts,
                design,
                family=NegativeBinomial(),
                offset=offset
            )
            result = model.fit()
            
            results[taxon] = {
                'coefficients': result.params,
                'pvalues': result.pvalues,
                'log2FC': result.params[1] / np.log(2)  # Convert to log2
            }
        except:
            results[taxon] = None
    
    # Multiple testing correction
    from statsmodels.stats.multitest import multipletests
    
    pvalues = [r['pvalues'][1] for r in results.values() if r]
    rejected, padj, _, _ = multipletests(pvalues, method='fdr_bh')
    
    # Add adjusted p-values
    for i, (taxon, result) in enumerate(results.items()):
        if result:
            result['padj'] = padj[i]
            result['significant'] = rejected[i]
    
    return results
```

#### 2. ANCOM-BC: Bias-Corrected Composition Analysis

```python
def ancom_bc(count_matrix, metadata, formula):
    """
    Analysis of Compositions of Microbiomes with Bias Correction
    """
    
    # Log-ratio transformation with bias correction
    log_counts = np.log(count_matrix + 1)
    
    # Estimate sample-specific bias
    def estimate_bias(log_counts):
        # Median of log-ratios as reference
        sample_medians = log_counts.median(axis=1)
        
        # Bias per sample
        bias = []
        for i in range(len(log_counts)):
            # Regression on median
            slope, intercept = np.polyfit(
                sample_medians,
                log_counts.iloc[i],
                deg=1
            )
            bias.append(intercept)
        
        return np.array(bias)
    
    bias = estimate_bias(log_counts)
    
    # Correct for bias
    log_counts_corrected = log_counts - bias[:, np.newaxis]
    
    # Linear model on corrected data
    from sklearn.linear_model import LinearRegression
    
    results = {}
    for taxon in count_matrix.columns:
        y = log_counts_corrected[taxon]
        X = metadata[formula.split('~')[1].strip().split('+')]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Bootstrap for p-values
        n_bootstrap = 1000
        bootstrap_coefs = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), len(X), replace=True)
            model_boot = LinearRegression()
            model_boot.fit(X.iloc[idx], y.iloc[idx])
            bootstrap_coefs.append(model_boot.coef_)
        
        bootstrap_coefs = np.array(bootstrap_coefs)
        
        # P-value from bootstrap
        pvalues = []
        for j in range(len(model.coef_)):
            p = (bootstrap_coefs[:, j] * np.sign(model.coef_[j]) < 0).mean() * 2
            pvalues.append(min(p, 1.0))
        
        results[taxon] = {
            'coefficients': model.coef_,
            'pvalues': pvalues,
            'bias_correction': bias.mean()
        }
    
    return results
```

#### 3. ALDEx2: Compositional Approach

```python
def aldex2_analysis(count_matrix, conditions):
    """
    ANOVA-Like Differential Expression for compositions
    """
    
    # Monte Carlo sampling from Dirichlet
    n_mc = 128  # Number of MC samples
    
    # Add uniform prior
    count_matrix_prior = count_matrix + 0.5
    
    # Generate MC instances
    mc_samples = []
    for _ in range(n_mc):
        # Dirichlet sampling for each sample
        dirichlet_samples = []
        for i in range(len(count_matrix_prior)):
            alpha = count_matrix_prior.iloc[i].values
            sample = np.random.dirichlet(alpha)
            dirichlet_samples.append(sample)
        
        mc_samples.append(np.array(dirichlet_samples))
    
    # CLR transform each MC instance
    clr_samples = []
    for mc in mc_samples:
        clr = clr_transform(mc)
        clr_samples.append(clr)
    
    # Test each taxon
    results = {}
    
    for j, taxon in enumerate(count_matrix.columns):
        # Collect CLR values across MC samples
        taxon_clr = np.array([clr[:, j] for clr in clr_samples])
        
        # Between-group vs within-group variance
        between_var = []
        within_var = []
        
        for mc_idx in range(n_mc):
            values = taxon_clr[mc_idx]
            
            # ANOVA components
            groups = [values[conditions == c] for c in conditions.unique()]
            
            # Between-group variance
            group_means = [g.mean() for g in groups]
            grand_mean = values.mean()
            between = sum(len(g) * (m - grand_mean)**2 
                         for g, m in zip(groups, group_means))
            
            # Within-group variance
            within = sum(np.sum((g - m)**2) 
                        for g, m in zip(groups, group_means))
            
            between_var.append(between / (len(conditions.unique()) - 1))
            within_var.append(within / (len(values) - len(conditions.unique())))
        
        # Effect size
        effect_size = np.mean(between_var) / (np.mean(within_var) + 1e-10)
        
        # P-value via permutation
        null_effects = []
        for _ in range(1000):
            perm_conditions = np.random.permutation(conditions)
            # ... repeat ANOVA with permuted labels
        
        p_value = (null_effects >= effect_size).mean()
        
        results[taxon] = {
            'effect_size': effect_size,
            'p_value': p_value,
            'between_variance': np.mean(between_var),
            'within_variance': np.mean(within_var)
        }
    
    return results
```

### Longitudinal Microbiome Analysis

```python
def longitudinal_microbiome_analysis(counts_timeline, timepoints, subject_ids):
    """
    Analyze temporal dynamics of microbiome
    """
    
    # 1. Volatility: temporal instability
    def calculate_volatility(counts, times, subjects):
        volatility = {}
        
        for subject in subjects.unique():
            subject_mask = subjects == subject
            subject_counts = counts[subject_mask]
            subject_times = times[subject_mask]
            
            # Sort by time
            time_order = np.argsort(subject_times)
            ordered_counts = subject_counts.iloc[time_order]
            
            # Aitchison distance between consecutive timepoints
            distances = []
            for i in range(len(ordered_counts)-1):
                dist = aitchison_distance(
                    ordered_counts.iloc[i],
                    ordered_counts.iloc[i+1]
                )
                time_diff = subject_times[time_order[i+1]] - subject_times[time_order[i]]
                distances.append(dist / time_diff)  # Rate of change
            
            volatility[subject] = np.mean(distances) if distances else 0
        
        return volatility
    
    # 2. Auto-correlation and periodicity
    def detect_periodicity(counts, times, max_lag=30):
        # CLR transform
        clr_counts = clr_transform(counts)
        
        autocorrelations = []
        for lag in range(1, max_lag):
            if lag < len(times):
                # Samples at lag distance
                pairs = []
                for i in range(len(times) - lag):
                    if times[i+lag] - times[i] == lag:  # Exact lag
                        pairs.append((clr_counts[i], clr_counts[i+lag]))
                
                if pairs:
                    pairs = np.array(pairs)
                    corr = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
                    autocorrelations.append((lag, corr))
        
        return autocorrelations
    
    # 3. Dynamic time warping for trajectory comparison
    def dtw_distance(seq1, seq2):
        from scipy.spatial.distance import euclidean
        
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.inf * np.ones((n+1, m+1))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = euclidean(seq1[i-1], seq2[j-1])
                
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        return dtw_matrix[n, m]
    
    # 4. State-space modeling
    from filterpy.kalman import KalmanFilter
    
    def kalman_smooth_microbiome(observations):
        """
        Smooth noisy observations using Kalman filter
        """
        n_taxa = observations.shape[1]
        
        kf = KalmanFilter(dim_x=n_taxa, dim_z=n_taxa)
        
        # State transition (assume small changes)
        kf.F = np.eye(n_taxa) * 0.99
        
        # Measurement function
        kf.H = np.eye(n_taxa)
        
        # Process noise
        kf.Q = np.eye(n_taxa) * 0.01
        
        # Measurement noise
        kf.R = np.eye(n_taxa) * 0.1
        
        # Initial state
        kf.x = observations[0]
        kf.P = np.eye(n_taxa)
        
        # Filter
        smoothed = []
        for obs in observations:
            kf.predict()
            kf.update(obs)
            smoothed.append(kf.x.copy())
        
        return np.array(smoothed)
    
    return {
        'volatility': calculate_volatility(counts_timeline, timepoints, subject_ids),
        'periodicity': detect_periodicity(counts_timeline, timepoints),
        'smoothed': kalman_smooth_microbiome(counts_timeline.values)
    }
```

### Network Analysis

```python
def construct_microbial_network(count_matrix, method='sparcc'):
    """
    Infer microbial interaction networks
    """
    
    if method == 'sparcc':
        # SparCC: correlation for compositional data
        def sparcc(counts, n_iterations=20):
            # Variance of log-ratios
            n_samples, n_taxa = counts.shape
            
            # Initialize correlation estimate
            cor_matrix = np.eye(n_taxa)
            
            for iteration in range(n_iterations):
                # Variance of log-ratios
                var_mat = np.zeros((n_taxa, n_taxa))
                
                for i in range(n_taxa):
                    for j in range(i+1, n_taxa):
                        log_ratio = np.log((counts[:, i] + 1) / (counts[:, j] + 1))
                        var_mat[i, j] = var_mat[j, i] = np.var(log_ratio)
                
                # Solve for correlations
                # System of equations: Var(log(X_i/X_j)) = Var(X_i) + Var(X_j) - 2*Cov(X_i, X_j)
                
                # ... numerical solution ...
                
            return cor_matrix
    
    elif method == 'spieceasi':
        # SPIEC-EASI: Sparse Inverse Covariance
        from sklearn.covariance import GraphicalLassoCV
        
        # CLR transform
        clr_data = clr_transform(count_matrix)
        
        # Estimate sparse precision matrix
        model = GraphicalLassoCV(cv=5)
        model.fit(clr_data)
        
        # Precision matrix represents direct interactions
        precision = model.precision_
        
        # Convert to adjacency matrix
        adjacency = (np.abs(precision) > 0.01).astype(int)
        np.fill_diagonal(adjacency, 0)
        
        return adjacency
    
    elif method == 'flashweave':
        # FlashWeave: sensitive and fast
        # Conditional mutual information
        pass
    
    # Network statistics
    import networkx as nx
    
    G = nx.from_numpy_array(adjacency)
    
    network_stats = {
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'clustering_coefficient': nx.average_clustering(G),
        'modularity': nx.algorithms.community.modularity(
            G, nx.algorithms.community.greedy_modularity_communities(G)
        ),
        'centrality': nx.degree_centrality(G)
    }
    
    return adjacency, network_stats
```

### Machine Learning for Microbiome

```python
class MicrobiomeMLPipeline:
    """
    ML pipeline respecting compositional nature
    """
    
    def __init__(self, transform='clr'):
        self.transform = transform
        
    def preprocess(self, counts):
        """Transform and handle zeros"""
        
        # Zero replacement
        counts_nz = zero_replacement(counts, method='multiplicative')
        
        # Compositional transformation
        if self.transform == 'clr':
            return clr_transform(counts_nz)
        elif self.transform == 'ilr':
            return ilr_transform(counts_nz)[0]
        elif self.transform == 'philr':
            return philr_transform(counts_nz, self.tree)
    
    def feature_selection(self, X, y, method='selbal'):
        """
        Compositional feature selection
        """
        
        if method == 'selbal':
            # Balance selection for microbiome
            # Find two groups of taxa that maximize discrimination
            
            from itertools import combinations
            
            best_balance = None
            best_score = -np.inf
            
            # Try different balance sizes
            for size_num in range(1, min(10, X.shape[1]//2)):
                for size_den in range(1, min(10, X.shape[1]//2)):
                    
                    # Try combinations
                    for numerator in combinations(range(X.shape[1]), size_num):
                        for denominator in combinations(
                            set(range(X.shape[1])) - set(numerator), 
                            size_den
                        ):
                            # Create balance
                            balance = np.log(
                                X[:, numerator].mean(axis=1) / 
                                X[:, denominator].mean(axis=1)
                            )
                            
                            # Score (e.g., ROC-AUC)
                            from sklearn.metrics import roc_auc_score
                            score = roc_auc_score(y, balance)
                            
                            if score > best_score:
                                best_score = score
                                best_balance = (numerator, denominator)
            
            return best_balance
    
    def cross_validation(self, X, y, model, cv_method='repeated_random'):
        """
        Proper CV for microbiome data
        """
        
        if cv_method == 'repeated_random':
            # Account for compositionality in CV
            from sklearn.model_selection import RepeatedStratifiedKFold
            
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
            scores = []
            
            for train_idx, test_idx in cv.split(X, y):
                # Transform using only training data parameters
                X_train = X[train_idx]
                X_test = X[test_idx]
                
                # Recompute CLR using training geometric mean
                train_geom_mean = np.exp(np.log(X_train + 1e-10).mean())
                
                X_train_clr = np.log(X_train / train_geom_mean)
                X_test_clr = np.log(X_test / train_geom_mean)
                
                model.fit(X_train_clr, y[train_idx])
                score = model.score(X_test_clr, y[test_idx])
                scores.append(score)
            
            return np.mean(scores), np.std(scores)
```

### Functional Prediction

```python
def predict_metagenome_function(otu_table, reference_genomes):
    """
    PICRUSt-style functional prediction
    """
    
    # 1. Normalize by 16S copy number
    copy_numbers = load_16s_copy_numbers()
    normalized_otus = otu_table / copy_numbers
    
    # 2. Predict metagenomes
    kegg_pathways = load_kegg_reference()
    
    predicted_functions = {}
    
    for otu in normalized_otus.columns:
        if otu in reference_genomes:
            genome = reference_genomes[otu]
            
            for pathway in genome['pathways']:
                if pathway not in predicted_functions:
                    predicted_functions[pathway] = 0
                
                # Add contribution from this OTU
                predicted_functions[pathway] += normalized_otus[otu] * genome['pathways'][pathway]
    
    # 3. Nearest Sequenced Taxon Index (NSTI)
    # Measure reliability of predictions
    nsti_scores = []
    for otu in otu_table.columns:
        if otu in reference_genomes:
            distance = reference_genomes[otu]['distance_to_reference']
            abundance = otu_table[otu].sum()
            nsti_scores.append(distance * abundance)
    
    nsti = np.sum(nsti_scores) / otu_table.sum().sum()
    
    return predicted_functions, nsti
```

### Integration with Host Data

```python
def integrate_microbiome_host(microbiome_data, host_data, method='cca'):
    """
    Multi-omics integration
    """
    
    if method == 'cca':
        # Canonical Correlation Analysis
        from sklearn.cross_decomposition import CCA
        
        # Transform microbiome data
        microbiome_clr = clr_transform(microbiome_data)
        
        # Standardize host data
        host_scaled = (host_data - host_data.mean()) / host_data.std()
        
        # CCA
        cca = CCA(n_components=5)
        micro_scores, host_scores = cca.fit_transform(microbiome_clr, host_scaled)
        
        # Correlation of canonical variates
        correlations = [
            np.corrcoef(micro_scores[:, i], host_scores[:, i])[0, 1]
            for i in range(5)
        ]
        
        return {
            'microbiome_scores': micro_scores,
            'host_scores': host_scores,
            'correlations': correlations,
            'microbiome_loadings': cca.x_loadings_,
            'host_loadings': cca.y_loadings_
        }
    
    elif method == 'mediation':
        # Test if microbiome mediates host phenotype
        import statsmodels.api as sm
        
        results = {}
        
        for taxon in microbiome_data.columns:
            for metabolite in host_data.columns:
                # Path a: Treatment → Mediator
                model_a = sm.OLS(microbiome_data[taxon], treatment)
                result_a = model_a.fit()
                a = result_a.params[1]
                
                # Path b: Mediator → Outcome (controlling for treatment)
                X = sm.add_constant(pd.DataFrame({
                    'treatment': treatment,
                    'mediator': microbiome_data[taxon]
                }))
                model_b = sm.OLS(host_data[metabolite], X)
                result_b = model_b.fit()
                b = result_b.params['mediator']
                
                # Indirect effect
                indirect_effect = a * b
                
                # Sobel test for significance
                se_a = result_a.bse[1]
                se_b = result_b.bse['mediator']
                sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
                sobel_z = indirect_effect / sobel_se
                
                results[f'{taxon}_{metabolite}'] = {
                    'indirect_effect': indirect_effect,
                    'sobel_z': sobel_z,
                    'p_value': 2 * (1 - norm.cdf(abs(sobel_z)))
                }
        
        return results
```

### Common Pitfalls

| Pitfall | Consequence | Solution |
|---------|------------|----------|
| **Ignoring compositionality** | Spurious correlations | Use log-ratio transforms |
| **Rarefying unnecessarily** | Loss of information | Use proper count models |
| **Testing all taxa** | Multiple testing burden | Pre-filter or use multivariate methods |
| **Ignoring phylogeny** | Missing biological signal | Use phylogenetic methods |
| **Assuming independence** | Pseudo-replication | Mixed models for repeated measures |
| **Over-interpreting networks** | False interactions | Validate experimentally |

### References
- Gloor et al. (2017). Microbiome datasets are compositional
- McMurdie & Holmes (2013). phyloseq: An R package for reproducible interactive analysis
- Mandal et al. (2015). Analysis of composition of microbiomes (ANCOM)
- Knight et al. (2018). Best practices for analysing microbiomes