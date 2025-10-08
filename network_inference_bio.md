# Network Inference and Graph Analysis in Biological Systems
## From Correlation to Causation in Complex Biological Networks

### Intent
Biological systems are networks - genes regulate each other, proteins interact, metabolites participate in pathways, neurons connect in circuits. This document provides rigorous methods for inferring network structure from data and analyzing the resulting graphs to understand biological function.

### Mathematical Framework for Biological Networks

**Graph Representation:**
G = (V, E, W) where:
- V = {v₁, ..., vₙ}: nodes (genes, proteins, metabolites, cells)
- E ⊆ V × V: edges (interactions, regulations, connections)
- W: E → ℝ: edge weights (interaction strength, correlation, flux)

**Adjacency Matrix:**
```
A_ij = {
    w_ij if (i,j) ∈ E
    0     otherwise
}
```

**Biological Network Types:**

| Network Type | Nodes | Edges | Directed | Weighted | Signed |
|-------------|-------|--------|----------|----------|--------|
| Gene Regulatory | Genes/TFs | Regulation | Yes | Yes | Yes (+/-) |
| Protein-Protein | Proteins | Physical binding | No | Yes | No |
| Metabolic | Metabolites | Reactions | Yes | Yes (flux) | No |
| Signaling | Proteins | Phosphorylation | Yes | Yes | Yes |
| Neural | Neurons | Synapses | Yes | Yes | Yes |
| Ecological | Species | Interactions | Yes/No | Yes | Yes |

### Network Inference Methods

#### 1. Correlation-Based Methods

**Pearson Correlation Network:**
```python
def correlation_network(expression_data, threshold=0.5):
    """
    Simple correlation-based network
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(expression_data.T)
    
    # Threshold to create adjacency matrix
    adjacency = np.abs(corr_matrix) > threshold
    np.fill_diagonal(adjacency, 0)  # No self-loops
    
    # Problem: Correlation ≠ Direct interaction
    # A → B → C gives correlation between A and C
    
    return adjacency, corr_matrix
```

**Partial Correlation (Gaussian Graphical Model):**
```python
def partial_correlation_network(expression_data, alpha=0.01):
    """
    Partial correlation removes indirect effects
    
    Partial correlation between i and j given all others:
    ρ_ij|rest = -Ω_ij / √(Ω_ii * Ω_jj)
    
    where Ω = Σ^(-1) is precision matrix
    """
    from sklearn.covariance import GraphicalLassoCV
    
    # Estimate sparse precision matrix
    model = GraphicalLassoCV(alphas=10, cv=5)
    model.fit(expression_data)
    
    precision_matrix = model.precision_
    
    # Convert to partial correlations
    diag = np.sqrt(np.diag(precision_matrix))
    partial_corr = -precision_matrix / np.outer(diag, diag)
    np.fill_diagonal(partial_corr, 1)
    
    # Threshold for significance
    n_samples = len(expression_data)
    threshold = np.sqrt(1/(n_samples - 3)) * stats.norm.ppf(1 - alpha/2)
    
    adjacency = np.abs(partial_corr) > threshold
    np.fill_diagonal(adjacency, 0)
    
    return adjacency, partial_corr
```

#### 2. Information-Theoretic Methods

**Mutual Information Networks:**
```python
def mutual_information_network(expression_data, n_bins=10):
    """
    MI captures non-linear relationships
    
    MI(X,Y) = ΣΣ p(x,y) log[p(x,y)/(p(x)p(y))]
    """
    n_genes = expression_data.shape[1]
    mi_matrix = np.zeros((n_genes, n_genes))
    
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            # Discretize for MI estimation
            x_discrete = pd.cut(expression_data[:, i], bins=n_bins, labels=False)
            y_discrete = pd.cut(expression_data[:, j], bins=n_bins, labels=False)
            
            # Compute MI
            mi = mutual_info_score(x_discrete, y_discrete)
            
            # Normalize by entropy
            h_x = entropy(x_discrete)
            h_y = entropy(y_discrete)
            normalized_mi = 2 * mi / (h_x + h_y)
            
            mi_matrix[i, j] = mi_matrix[j, i] = normalized_mi
    
    return mi_matrix
```

**ARACNE (Algorithm for Reconstruction of Accurate Cellular Networks):**
```python
def aracne(mi_matrix, epsilon=0.05):
    """
    Remove indirect interactions using Data Processing Inequality (DPI)
    
    If X → Y → Z, then MI(X,Z) ≤ min(MI(X,Y), MI(Y,Z))
    """
    n_genes = mi_matrix.shape[0]
    adjacency = mi_matrix.copy()
    
    # For each triplet
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            for k in range(j+1, n_genes):
                # Check DPI for triplet (i,j,k)
                mi_ij = mi_matrix[i, j]
                mi_jk = mi_matrix[j, k]
                mi_ik = mi_matrix[i, k]
                
                # Find weakest edge
                edges = [(mi_ij, i, j), (mi_jk, j, k), (mi_ik, i, k)]
                weakest = min(edges, key=lambda x: x[0])
                
                # Remove if significantly weaker
                if weakest[0] < min(edges[1][0], edges[2][0]) * (1 - epsilon):
                    adjacency[weakest[1], weakest[2]] = 0
                    adjacency[weakest[2], weakest[1]] = 0
    
    return adjacency
```

#### 3. Regression-Based Methods

**GENIE3 (GEne Network Inference with Ensemble of trees):**
```python
def genie3(expression_data, gene_names=None):
    """
    Random Forest importance for network inference
    """
    from sklearn.ensemble import RandomForestRegressor
    
    n_genes = expression_data.shape[1]
    importance_matrix = np.zeros((n_genes, n_genes))
    
    for target_gene in range(n_genes):
        # Predict target from all others
        y = expression_data[:, target_gene]
        X = np.delete(expression_data, target_gene, axis=1)
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=1000,
            max_features='sqrt',
            random_state=42
        )
        rf.fit(X, y)
        
        # Feature importance = regulatory strength
        importances = rf.feature_importances_
        
        # Map back to full matrix
        idx = 0
        for regulator in range(n_genes):
            if regulator != target_gene:
                importance_matrix[regulator, target_gene] = importances[idx]
                idx += 1
    
    return importance_matrix
```

#### 4. Dynamic Bayesian Networks

**For Time-Series Data:**
```python
def dynamic_bayesian_network(time_series_data, max_lag=3):
    """
    Infer time-delayed regulatory relationships
    
    Model: X_t = f(X_{t-1}, X_{t-2}, ..., X_{t-lag}) + ε
    """
    n_timepoints, n_genes = time_series_data.shape
    adjacency = np.zeros((n_genes * max_lag, n_genes))
    
    # Create lagged features
    X_lagged = []
    y_current = []
    
    for lag in range(1, max_lag + 1):
        if lag < n_timepoints:
            X_lagged.append(time_series_data[:-lag])
            
    X_lagged = np.hstack(X_lagged)
    y_current = time_series_data[max_lag:]
    
    # Infer network for each target gene
    for target in range(n_genes):
        # Elastic net for sparse selection
        from sklearn.linear_model import ElasticNetCV
        
        model = ElasticNetCV(cv=5, max_iter=1000)
        model.fit(X_lagged, y_current[:, target])
        
        # Non-zero coefficients indicate regulation
        adjacency[:, target] = np.abs(model.coef_) > 0
    
    return adjacency.reshape(max_lag, n_genes, n_genes)
```

### Network Analysis Metrics

#### Node Centrality Measures

```python
def compute_centralities(adjacency_matrix):
    """
    Various centrality measures for biological importance
    """
    import networkx as nx
    
    G = nx.from_numpy_array(adjacency_matrix)
    
    centralities = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
        'pagerank': nx.pagerank(G)
    }
    
    # Biological interpretation:
    # - Degree: How many interactions (hub genes)
    # - Betweenness: Bottlenecks in information flow
    # - Closeness: Quick signal propagation
    # - Eigenvector: Connected to important nodes
    # - PageRank: Regulatory influence
    
    return centralities
```

#### Network Motifs

```python
def find_network_motifs(adjacency, motif_size=3):
    """
    Over-represented subgraphs (regulatory patterns)
    """
    import networkx as nx
    from itertools import combinations
    
    G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    
    # Common biological motifs
    motif_counts = {
        'feedforward': 0,
        'feedback': 0,
        'mutual': 0
    }
    
    # Check all triplets
    for nodes in combinations(G.nodes(), motif_size):
        subgraph = G.subgraph(nodes)
        
        # Classify motif type
        edges = list(subgraph.edges())
        
        if len(edges) == 3:
            # Feed-forward loop
            if is_feedforward(subgraph):
                motif_counts['feedforward'] += 1
        
        # Check for cycles (feedback)
        if nx.is_directed_acyclic_graph(subgraph) == False:
            motif_counts['feedback'] += 1
    
    # Statistical significance via random networks
    significance = compute_motif_significance(G, motif_counts)
    
    return motif_counts, significance
```

#### Community Detection

```python
def detect_network_modules(adjacency_matrix, method='louvain'):
    """
    Find functional modules/pathways
    """
    import networkx as nx
    from networkx.algorithms import community
    
    G = nx.from_numpy_array(adjacency_matrix)
    
    if method == 'louvain':
        # Louvain method for modularity optimization
        communities = community.louvain_communities(G)
        
    elif method == 'spectral':
        # Spectral clustering on graph Laplacian
        from sklearn.cluster import SpectralClustering
        
        n_clusters = estimate_optimal_clusters(adjacency_matrix)
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed'
        )
        labels = clustering.fit_predict(adjacency_matrix)
        
        communities = [[] for _ in range(n_clusters)]
        for node, label in enumerate(labels):
            communities[label].append(node)
    
    elif method == 'infomap':
        # Information-theoretic approach
        import infomap
        
        im = infomap.Infomap()
        for i, j in zip(*np.nonzero(adjacency_matrix)):
            im.add_link(i, j, adjacency_matrix[i, j])
        
        im.run()
        communities = im.get_modules()
    
    # Compute modularity
    modularity = nx.algorithms.community.modularity(G, communities)
    
    return communities, modularity
```

### Graph Signal Processing

```python
class GraphSignalProcessor:
    """
    Analyze signals (e.g., expression) on network structure
    """
    
    def __init__(self, adjacency_matrix):
        self.A = adjacency_matrix
        self.n_nodes = len(adjacency_matrix)
        
        # Compute graph Laplacian
        degree = np.sum(adjacency_matrix, axis=1)
        self.D = np.diag(degree)
        self.L = self.D - self.A  # Unnormalized Laplacian
        
        # Normalized Laplacian
        D_sqrt_inv = np.diag(1 / np.sqrt(degree + 1e-10))
        self.L_norm = D_sqrt_inv @ self.L @ D_sqrt_inv
        
        # Eigendecomposition (graph Fourier basis)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.L_norm)
    
    def graph_fourier_transform(self, signal):
        """
        Transform signal to graph frequency domain
        """
        return self.eigenvectors.T @ signal
    
    def graph_filtering(self, signal, filter_func):
        """
        Apply spectral filter to graph signal
        """
        # Transform to frequency domain
        signal_hat = self.graph_fourier_transform(signal)
        
        # Apply filter in frequency domain
        filtered_hat = filter_func(self.eigenvalues) * signal_hat
        
        # Transform back
        filtered_signal = self.eigenvectors @ filtered_hat
        
        return filtered_signal
    
    def smoothness(self, signal):
        """
        Measure how smooth signal is on graph
        
        Smooth signals have similar values on connected nodes
        """
        return signal.T @ self.L @ signal
    
    def diffusion_distance(self, t=1):
        """
        Distance based on random walk probability
        """
        # Heat kernel
        H_t = expm(-t * self.L)
        
        # Diffusion distance matrix
        dist_matrix = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                diff = H_t[i, :] - H_t[j, :]
                dist_matrix[i, j] = np.sqrt(np.sum(diff**2))
        
        return dist_matrix
```

### Causal Network Inference

```python
def pc_stable_algorithm(data, alpha=0.05, max_cond_set_size=3):
    """
    PC-stable algorithm for causal discovery
    Handles observational data with causal assumptions
    """
    n_vars = data.shape[1]
    
    # Start with complete graph
    adjacency = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    separation_sets = {}
    
    # Phase 1: Skeleton discovery
    for cond_set_size in range(max_cond_set_size + 1):
        edge_list = list(zip(*np.where(adjacency)))
        
        for (i, j) in edge_list:
            # Find neighbors for conditioning
            neighbors_i = np.where(adjacency[i, :])[0]
            neighbors_j = np.where(adjacency[j, :])[0]
            possible_cond = np.union1d(neighbors_i, neighbors_j)
            possible_cond = possible_cond[~np.isin(possible_cond, [i, j])]
            
            if len(possible_cond) >= cond_set_size:
                # Test conditional independence
                from itertools import combinations
                
                for cond_set in combinations(possible_cond, cond_set_size):
                    p_val = conditional_independence_test(
                        data[:, i], 
                        data[:, j],
                        data[:, list(cond_set)]
                    )
                    
                    if p_val > alpha:
                        # Remove edge
                        adjacency[i, j] = adjacency[j, i] = 0
                        separation_sets[(i, j)] = cond_set
                        break
    
    # Phase 2: Orient v-structures
    oriented = orient_v_structures(adjacency, separation_sets)
    
    # Phase 3: Apply orientation rules
    final_dag = apply_orientation_rules(oriented)
    
    return final_dag

def conditional_independence_test(x, y, z):
    """
    Test if X ⊥ Y | Z using partial correlation
    """
    if z.shape[1] == 0:
        # Marginal independence
        return stats.pearsonr(x, y)[1]
    
    # Regress out Z from both X and Y
    from sklearn.linear_model import LinearRegression
    
    reg_x = LinearRegression().fit(z, x)
    res_x = x - reg_x.predict(z)
    
    reg_y = LinearRegression().fit(z, y)
    res_y = y - reg_y.predict(z)
    
    # Test correlation of residuals
    return stats.pearsonr(res_x, res_y)[1]
```

### Network Perturbation and Control

```python
def network_controllability(adjacency_matrix):
    """
    Determine minimum set of driver nodes for full control
    Liu et al. Nature 2011
    """
    import networkx as nx
    
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    
    # Maximum matching to find unmatched nodes
    matching = nx.algorithms.bipartite.maximum_matching(G)
    
    # Driver nodes = unmatched nodes
    matched_targets = set([v for u, v in matching])
    all_nodes = set(G.nodes())
    driver_nodes = all_nodes - matched_targets
    
    # Controllability metrics
    n_drivers = len(driver_nodes)
    n_total = len(all_nodes)
    controllability = 1 - (n_drivers / n_total)
    
    return driver_nodes, controllability

def target_control_problem(adjacency, initial_state, target_state, max_controls=5):
    """
    Find minimal intervention to drive network to target state
    """
    from scipy.optimize import minimize
    
    # Dynamics: dx/dt = Ax + Bu
    # where u is control input
    
    def objective(control_sequence):
        """
        Minimize control effort + distance to target
        """
        state = initial_state.copy()
        control_cost = 0
        
        for t, control in enumerate(control_sequence.reshape(-1, len(initial_state))):
            # Apply dynamics
            state = adjacency @ state + control
            control_cost += np.sum(control**2)
        
        # Final state error
        target_error = np.sum((state - target_state)**2)
        
        return target_error + 0.1 * control_cost
    
    # Optimize control sequence
    initial_controls = np.zeros(max_controls * len(initial_state))
    result = minimize(objective, initial_controls, method='L-BFGS-B')
    
    optimal_controls = result.x.reshape(max_controls, -1)
    
    return optimal_controls
```

### Biological Validation and Enrichment

```python
def validate_network_biologically(inferred_network, known_interactions, 
                                 gene_names=None):
    """
    Compare inferred network to known biology
    """
    # Load known interactions (e.g., from STRING, BioGRID)
    known_edges = set(known_interactions)
    
    # Convert inferred network to edges
    inferred_edges = set()
    for i, j in zip(*np.where(inferred_network)):
        if gene_names:
            edge = (gene_names[i], gene_names[j])
        else:
            edge = (i, j)
        inferred_edges.add(edge)
    
    # Compute overlap statistics
    true_positives = len(inferred_edges & known_edges)
    false_positives = len(inferred_edges - known_edges)
    false_negatives = len(known_edges - inferred_edges)
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)
    
    # Enrichment test
    n_possible = len(gene_names) * (len(gene_names) - 1)
    expected_overlap = len(inferred_edges) * len(known_edges) / n_possible
    
    enrichment_score = true_positives / expected_overlap
    
    # Hypergeometric test for significance
    from scipy.stats import hypergeom
    
    p_value = hypergeom.sf(
        true_positives - 1,  # Successes in sample
        n_possible,          # Population size
        len(known_edges),    # Success states in population
        len(inferred_edges)  # Sample size
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'enrichment': enrichment_score,
        'p_value': p_value
    }
```

### Dynamic Network Analysis

```python
def time_varying_network(time_series_data, window_size=10, step_size=1):
    """
    Infer how network changes over time
    """
    n_timepoints, n_genes = time_series_data.shape
    networks = []
    
    for t in range(0, n_timepoints - window_size, step_size):
        # Extract window
        window_data = time_series_data[t:t+window_size]
        
        # Infer network for this window
        network_t = partial_correlation_network(window_data)
        networks.append(network_t)
    
    # Analyze network evolution
    network_tensor = np.array(networks)
    
    # Compute network similarity over time
    similarities = []
    for t in range(len(networks) - 1):
        sim = np.corrcoef(
            networks[t].flatten(),
            networks[t+1].flatten()
        )[0, 1]
        similarities.append(sim)
    
    # Identify rewiring events
    rewiring_scores = 1 - np.array(similarities)
    rewiring_events = np.where(rewiring_scores > np.percentile(rewiring_scores, 95))[0]
    
    return network_tensor, rewiring_events
```

### Implementation Pipeline

```python
class BiologicalNetworkAnalysis:
    """
    Complete pipeline for network inference and analysis
    """
    
    def __init__(self, expression_data, gene_names=None):
        self.data = expression_data
        self.gene_names = gene_names if gene_names else range(expression_data.shape[1])
        self.network = None
        
    def infer_network(self, method='ensemble', **kwargs):
        """
        Infer network using specified method or ensemble
        """
        if method == 'ensemble':
            # Combine multiple methods
            networks = []
            
            # Correlation
            corr_net = correlation_network(self.data, **kwargs)[0]
            networks.append(corr_net)
            
            # Partial correlation
            pcorr_net = partial_correlation_network(self.data, **kwargs)[0]
            networks.append(pcorr_net)
            
            # MI/ARACNE
            mi_net = mutual_information_network(self.data)
            aracne_net = aracne(mi_net)
            networks.append(aracne_net)
            
            # GENIE3
            genie_net = genie3(self.data)
            networks.append(genie_net)
            
            # Consensus network (majority vote)
            consensus = np.mean(networks, axis=0)
            self.network = consensus > 0.5
            
        else:
            # Single method
            self.network = self._run_method(method, **kwargs)
        
        return self.network
    
    def analyze_topology(self):
        """
        Comprehensive topological analysis
        """
        results = {}
        
        # Basic statistics
        results['n_nodes'] = len(self.network)
        results['n_edges'] = np.sum(self.network)
        results['density'] = results['n_edges'] / (results['n_nodes']**2)
        
        # Centralities
        results['centralities'] = compute_centralities(self.network)
        
        # Modules
        communities, modularity = detect_network_modules(self.network)
        results['n_modules'] = len(communities)
        results['modularity'] = modularity
        
        # Motifs
        motif_counts, significance = find_network_motifs(self.network)
        results['motifs'] = motif_counts
        
        # Controllability
        drivers, control = network_controllability(self.network)
        results['n_driver_nodes'] = len(drivers)
        results['controllability'] = control
        
        return results
    
    def export_to_cytoscape(self, filename='network.txt'):
        """
        Export for visualization
        """
        with open(filename, 'w') as f:
            f.write("Source\tTarget\tWeight\n")
            for i, j in zip(*np.where(self.network)):
                weight = self.network[i, j]
                f.write(f"{self.gene_names[i]}\t{self.gene_names[j]}\t{weight}\n")
```

### References
- Marbach, D. et al. (2012). Wisdom of crowds for robust gene network inference
- Huynh-Thu, V.A. et al. (2010). Inferring regulatory networks from expression data using tree-based methods
- Margolin, A.A. et al. (2006). ARACNE: An algorithm for reconstruction of gene regulatory networks
- Liu, Y.Y. et al. (2011). Controllability of complex networks