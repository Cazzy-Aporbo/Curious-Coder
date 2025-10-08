# Spatial Transcriptomics Analysis
## Integrating Gene Expression with Tissue Architecture

### Intent
Spatial transcriptomics measures gene expression while preserving spatial context. This document provides mathematical frameworks for analyzing data where location and expression are equally important, addressing unique challenges like spatial autocorrelation, tissue domains, and cell-cell communication inference.

### The Spatial Transcriptomics Data Structure

```
Data: (E, S, H) where:
- E ∈ ℝ^(n×g): Expression matrix (n spots/cells, g genes)
- S ∈ ℝ^(n×2/3): Spatial coordinates (2D or 3D)
- H: Histology image (optional, RGB values)

Key challenge: E and S are not independent!
```

### Mathematical Framework

#### 1. Spatial Autocorrelation: Moran's I

**Global spatial autocorrelation:**
```
I = (n/W) × (Σᵢ Σⱼ wᵢⱼ(xᵢ - x̄)(xⱼ - x̄)) / Σᵢ(xᵢ - x̄)²

where:
- wᵢⱼ: spatial weight between spots i and j
- W = Σᵢ Σⱼ wᵢⱼ: sum of all weights
- x̄: mean expression
```

**Expected value under null (no spatial pattern):**
```
E[I] = -1/(n-1)
Var[I] = complex, depends on weight matrix
```

**Local spatial autocorrelation (Local Moran's I):**
```
Iᵢ = (xᵢ - x̄)/σ² × Σⱼ wᵢⱼ(xⱼ - x̄)

Identifies spatial clusters and outliers:
- High-High: Iᵢ > 0, xᵢ > x̄ (hot spots)
- Low-Low: Iᵢ > 0, xᵢ < x̄ (cold spots)
- High-Low: Iᵢ < 0, xᵢ > x̄ (spatial outliers)
```

#### 2. Spatial Variable Gene Detection

**SpatialDE Framework:**
Model gene expression as Gaussian Process:
```
y ~ GP(μ, K)
where K = σ²_s K_SE + σ²_n I

K_SE(dᵢⱼ) = exp(-dᵢⱼ²/2ℓ²)  # Squared exponential kernel
```

**Likelihood ratio test:**
```
H₀: y ~ N(μ, σ²I)  # No spatial pattern
H₁: y ~ GP(μ, K)   # Spatial pattern

LR = 2(log L₁ - log L₀) ~ χ²(2)
```

#### 3. Spatial Domain Detection

**Hidden Markov Random Field (HMRF):**
```
P(zᵢ = k | z₋ᵢ, θ) ∝ exp(Σⱼ∈Nᵢ βI(zⱼ = k) + log π_k + log P(xᵢ | θ_k))

where:
- zᵢ: domain assignment for spot i
- Nᵢ: spatial neighbors of i
- β: spatial smoothing parameter
- π_k: domain prior probability
- θ_k: expression parameters for domain k
```

**Optimization via EM:**
```python
def spatial_domain_em(expression, coordinates, n_domains, beta=1.0):
    """
    EM algorithm for spatial domain detection
    """
    # E-step: Update domain probabilities
    def e_step(params):
        log_probs = np.zeros((n_spots, n_domains))
        for k in range(n_domains):
            # Expression likelihood
            log_probs[:, k] = multivariate_normal.logpdf(
                expression, params['mean'][k], params['cov'][k]
            )
            # Spatial prior (Potts model)
            for i in range(n_spots):
                neighbors = find_neighbors(i, coordinates)
                spatial_term = beta * np.sum(assignments[neighbors] == k)
                log_probs[i, k] += spatial_term
        
        # Normalize
        return softmax(log_probs, axis=1)
    
    # M-step: Update parameters
    def m_step(responsibilities):
        params = {}
        for k in range(n_domains):
            weights = responsibilities[:, k]
            params['mean'][k] = np.average(expression, weights=weights, axis=0)
            params['cov'][k] = np.cov(expression.T, aweights=weights)
            params['pi'][k] = weights.mean()
        return params
```

### Spatial Statistical Challenges

#### 1. The Modifiable Areal Unit Problem (MAUP)

```python
def demonstrate_maup(expression, coordinates, bin_sizes=[10, 20, 50]):
    """
    Results change with spatial resolution!
    """
    results = {}
    
    for bin_size in bin_sizes:
        # Aggregate spots into bins
        x_bins = np.arange(coordinates[:, 0].min(), 
                          coordinates[:, 0].max(), bin_size)
        y_bins = np.arange(coordinates[:, 1].min(), 
                          coordinates[:, 1].max(), bin_size)
        
        # Average expression per bin
        binned_expression = bin_2d(expression, coordinates, x_bins, y_bins)
        
        # Compute spatial statistics
        morans_i = compute_morans_i(binned_expression)
        results[bin_size] = morans_i
    
    # Different aggregations → different conclusions!
    return results
```

#### 2. Spatial Confounding

```
True model: Expression = f(cell_type) + ε
Observed: Expression = f(cell_type) + g(location) + ε

Problem: Cell types cluster spatially
Can't separate cell type effects from location effects
```

**Solution: Spatial mixed models:**
```
y = Xβ + Zb + ε

where:
- Xβ: fixed effects (cell type)
- Zb: random spatial effects, b ~ N(0, σ²K)
- K: spatial covariance matrix
```

### Cell-Cell Communication Inference

#### 1. Ligand-Receptor Analysis with Spatial Constraints

```python
def spatial_ligand_receptor_inference(expression, coordinates, lr_pairs):
    """
    Infer cell communication considering spatial proximity
    """
    
    communication_scores = {}
    
    for ligand, receptor in lr_pairs:
        ligand_expr = expression[:, ligand]
        receptor_expr = expression[:, receptor]
        
        # Spatial kernel (communication decays with distance)
        distances = pdist(coordinates)
        spatial_kernel = np.exp(-distances / length_scale)
        
        # Communication potential
        # C_ij = L_i × R_j × K(d_ij)
        comm_matrix = np.outer(ligand_expr, receptor_expr) * \
                     squareform(spatial_kernel)
        
        # Permutation test for significance
        null_distribution = []
        for _ in range(1000):
            # Shuffle spatial locations
            shuffled_coords = np.random.permutation(coordinates)
            null_kernel = compute_kernel(shuffled_coords)
            null_comm = np.outer(ligand_expr, receptor_expr) * null_kernel
            null_distribution.append(null_comm.sum())
        
        p_value = (null_distribution >= comm_matrix.sum()).mean()
        
        communication_scores[(ligand, receptor)] = {
            'score': comm_matrix.sum(),
            'p_value': p_value,
            'spatial_range': estimate_communication_range(comm_matrix, coordinates)
        }
    
    return communication_scores
```

#### 2. Spatial Niche Analysis

```python
def identify_spatial_niches(expression, coordinates, radius=100):
    """
    Find recurring cellular neighborhoods
    """
    
    # Build spatial graph
    from sklearn.neighbors import radius_neighbors_graph
    spatial_graph = radius_neighbors_graph(coordinates, radius)
    
    # Characterize each neighborhood
    neighborhood_profiles = []
    for i in range(len(expression)):
        neighbors = spatial_graph[i].indices
        
        # Include self
        neighborhood = np.concatenate([[i], neighbors])
        
        # Aggregate neighborhood expression
        profile = {
            'mean_expression': expression[neighborhood].mean(axis=0),
            'composition': compute_cell_type_proportions(neighborhood),
            'spatial_metrics': {
                'density': len(neighborhood) / (np.pi * radius**2),
                'clustering': compute_clustering_coefficient(i, spatial_graph)
            }
        }
        neighborhood_profiles.append(profile)
    
    # Cluster neighborhoods to find niches
    from sklearn.cluster import DBSCAN
    niche_features = extract_features(neighborhood_profiles)
    niches = DBSCAN(eps=0.3).fit_predict(niche_features)
    
    return niches, neighborhood_profiles
```

### Deconvolution with Spatial Information

#### Spatial Deconvolution Model

```
Y_s = W_s × C × H + ε

where:
- Y_s: observed expression at spot s
- W_s: cell type proportions at spot s
- C: cell type expression signatures
- H: spot-specific technical factors
```

**Adding spatial constraints:**
```python
def spatial_deconvolution(spot_expression, coordinates, reference_signatures):
    """
    Deconvolve spot expression with spatial smoothness
    """
    
    n_spots = len(spot_expression)
    n_cell_types = reference_signatures.shape[1]
    
    # Objective: ||Y - WC||² + λ Σᵢⱼ wᵢⱼ ||W_i - W_j||²
    # Second term encourages spatial smoothness
    
    def objective(W_flat):
        W = W_flat.reshape(n_spots, n_cell_types)
        
        # Reconstruction error
        reconstruction = W @ reference_signatures.T
        recon_error = np.sum((spot_expression - reconstruction)**2)
        
        # Spatial smoothness penalty
        spatial_penalty = 0
        for i in range(n_spots):
            for j in range(i+1, n_spots):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                if dist < neighborhood_radius:
                    weight = np.exp(-dist / length_scale)
                    spatial_penalty += weight * np.sum((W[i] - W[j])**2)
        
        return recon_error + lambda_spatial * spatial_penalty
    
    # Optimize with constraints: W >= 0, sum(W_s) = 1
    from scipy.optimize import minimize
    
    constraints = [
        {'type': 'eq', 'fun': lambda W: np.sum(W.reshape(n_spots, n_cell_types), axis=1) - 1}
    ]
    bounds = [(0, 1)] * (n_spots * n_cell_types)
    
    result = minimize(objective, W_init.flatten(), 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x.reshape(n_spots, n_cell_types)
```

### Integration with Histology

#### Image-Guided Spatial Analysis

```python
def integrate_histology(expression, coordinates, histology_image):
    """
    Use histology to guide spatial analysis
    """
    
    # Extract image features at each spot
    image_features = []
    for coord in coordinates:
        # Extract patch around spot
        patch = extract_patch(histology_image, coord, patch_size=64)
        
        # Compute features
        features = {
            'morphology': extract_morphology_features(patch),
            'texture': compute_texture_features(patch),
            'color': compute_color_histogram(patch)
        }
        image_features.append(features)
    
    # Joint embedding of expression and morphology
    from sklearn.cross_decomposition import CCA
    
    # Canonical Correlation Analysis
    cca = CCA(n_components=10)
    expr_canonical, morph_canonical = cca.fit_transform(
        expression, 
        np.array([f['morphology'] for f in image_features])
    )
    
    # Multimodal clustering
    combined_features = np.hstack([expr_canonical, morph_canonical])
    clusters = hierarchical_clustering(combined_features)
    
    return clusters, cca
```

### Spatial Trajectory Inference

```python
def spatial_trajectory_analysis(expression, coordinates, start_region, end_region):
    """
    Infer developmental trajectories with spatial constraints
    """
    
    # Build spatial-expression graph
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    
    # Combine expression and spatial distances
    expr_distances = pdist(expression, metric='euclidean')
    spatial_distances = pdist(coordinates, metric='euclidean')
    
    # Weighted combination
    combined_distances = alpha * expr_distances + (1-alpha) * spatial_distances
    
    # Build k-NN graph
    n_neighbors = 10
    adjacency = knn_graph(combined_distances, n_neighbors)
    
    # Find shortest paths from start to end region
    start_nodes = find_nodes_in_region(coordinates, start_region)
    end_nodes = find_nodes_in_region(coordinates, end_region)
    
    # Compute geodesic distances
    dist_matrix = shortest_path(adjacency, directed=False)
    
    # Extract trajectory
    trajectories = []
    for start in start_nodes:
        for end in end_nodes:
            path = reconstruct_path(dist_matrix, start, end)
            trajectories.append(path)
    
    # Compute pseudotime
    pseudotime = compute_pseudotime_from_trajectories(trajectories)
    
    return pseudotime, trajectories
```

### Quality Control and Diagnostics

#### 1. Spatial Batch Effects

```python
def detect_spatial_batch_effects(expression, coordinates, batch_info):
    """
    Detect if batch effects have spatial structure
    """
    
    # Spatial autocorrelation of batch indicators
    batch_encoded = pd.get_dummies(batch_info)
    
    spatial_batch_correlation = {}
    for batch in batch_encoded.columns:
        morans_i = compute_morans_i(batch_encoded[batch].values, coordinates)
        
        # Permutation test
        null_distribution = []
        for _ in range(1000):
            shuffled = np.random.permutation(batch_encoded[batch])
            null_i = compute_morans_i(shuffled, coordinates)
            null_distribution.append(null_i)
        
        p_value = (null_distribution >= morans_i).mean()
        spatial_batch_correlation[batch] = {
            'morans_i': morans_i,
            'p_value': p_value
        }
    
    return spatial_batch_correlation
```

#### 2. Spatial Resolution Assessment

```python
def assess_spatial_resolution(expression, coordinates, gene_sets):
    """
    Determine if spatial resolution captures biological structures
    """
    
    results = {}
    
    for gene_set_name, genes in gene_sets.items():
        # Compute spatial coherence of gene set
        gene_expr = expression[:, genes]
        
        # Average correlation between spatial neighbors
        spatial_coherence = []
        for i in range(len(coordinates)):
            neighbors = find_k_nearest(coordinates, i, k=6)
            
            # Correlation between spot and neighbors
            correlations = [
                pearsonr(gene_expr[i], gene_expr[j])[0] 
                for j in neighbors
            ]
            spatial_coherence.append(np.mean(correlations))
        
        # Compare to random gene sets
        random_coherences = []
        for _ in range(100):
            random_genes = np.random.choice(expression.shape[1], len(genes))
            random_expr = expression[:, random_genes]
            # ... compute coherence for random set
        
        results[gene_set_name] = {
            'coherence': np.mean(spatial_coherence),
            'random_mean': np.mean(random_coherences),
            'z_score': (np.mean(spatial_coherence) - np.mean(random_coherences)) / np.std(random_coherences)
        }
    
    return results
```

### Platform-Specific Considerations

| Platform | Resolution | Molecular Capture | Key Challenges |
|----------|-----------|-------------------|----------------|
| **10x Visium** | 55μm spots | ~1-10 cells | Deconvolution needed |
| **Slide-seq** | 10μm beads | <1 cell | Sparsity, technical noise |
| **MERFISH** | Subcellular | ~500 genes | Limited gene panel |
| **Stereo-seq** | 0.5μm bins | High | Data size, computation |
| **Xenium** | Subcellular | ~400 genes | Integration with omics |

### Common Pitfalls and Solutions

| Pitfall | Consequence | Solution |
|---------|------------|----------|
| **Ignoring spatial autocorrelation** | False positive DE genes | Use spatial-aware statistics |
| **Over-smoothing** | Loss of fine structure | Cross-validate smoothing parameters |
| **Edge effects** | Biased analysis at tissue boundaries | Use edge correction or exclude |
| **Assuming isotropy** | Missing directional patterns | Test for anisotropy |
| **Batch effects in arrays** | Spurious spatial patterns | Include array position as covariate |

### Advanced Methods

#### 1. Optimal Transport for Spatial Alignment

```python
def spatial_optimal_transport(source_expr, source_coords, target_expr, target_coords):
    """
    Align spatial transcriptomics across sections/samples
    """
    import ot
    
    # Cost matrix combines expression and spatial distances
    n_source = len(source_expr)
    n_target = len(target_expr)
    
    # Expression cost
    expr_cost = cdist(source_expr, target_expr, metric='euclidean')
    
    # Spatial cost
    spatial_cost = cdist(source_coords, target_coords, metric='euclidean')
    
    # Combined cost
    cost_matrix = alpha * expr_cost + beta * spatial_cost
    
    # Solve optimal transport
    transport_plan = ot.emd(
        np.ones(n_source) / n_source,  # Uniform source distribution
        np.ones(n_target) / n_target,  # Uniform target distribution
        cost_matrix
    )
    
    return transport_plan
```

#### 2. Spatial Gene Regulatory Networks

```python
def infer_spatial_grn(expression, coordinates, tf_list):
    """
    Infer gene regulatory networks with spatial context
    """
    
    # Spatial weights for regression
    spatial_weights = compute_spatial_weights(coordinates)
    
    grn_edges = []
    
    for target_gene in range(expression.shape[1]):
        if target_gene in tf_list:
            continue  # Skip TF-TF for simplicity
        
        # Spatially weighted regression
        y = expression[:, target_gene]
        X = expression[:, tf_list]
        
        # Geographically Weighted Regression (GWR)
        for spot_i in range(len(expression)):
            # Weight neighbors by distance
            weights = spatial_weights[spot_i]
            
            # Weighted least squares
            model = sm.WLS(y, X, weights=weights)
            results = model.fit()
            
            # Significant TF-target relationships
            significant_tfs = np.where(results.pvalues < 0.05)[0]
            
            for tf_idx in significant_tfs:
                grn_edges.append({
                    'tf': tf_list[tf_idx],
                    'target': target_gene,
                    'location': coordinates[spot_i],
                    'coefficient': results.params[tf_idx]
                })
    
    return grn_edges
```

### Validation and Benchmarking

```python
def spatial_cross_validation(expression, coordinates, model, n_folds=5):
    """
    Spatially aware cross-validation
    """
    
    # Spatial blocking to avoid leakage
    from sklearn.cluster import KMeans
    
    # Cluster spatial coordinates
    spatial_blocks = KMeans(n_clusters=n_folds).fit_predict(coordinates)
    
    cv_scores = []
    for fold in range(n_folds):
        # Hold out one spatial block
        train_mask = spatial_blocks != fold
        test_mask = spatial_blocks == fold
        
        # Ensure no spatial neighbors across train/test
        train_coords = coordinates[train_mask]
        test_coords = coordinates[test_mask]
        min_distance = cdist(train_coords, test_coords).min()
        
        if min_distance < spatial_resolution:
            print(f"Warning: Spatial leakage in fold {fold}")
        
        # Train and evaluate
        model.fit(expression[train_mask], coordinates[train_mask])
        score = model.score(expression[test_mask], coordinates[test_mask])
        cv_scores.append(score)
    
    return cv_scores
```

### Future Directions

1. **3D Spatial Transcriptomics** - Extending methods to volumetric data
2. **Temporal-Spatial** - Time course with spatial resolution
3. **Multi-modal Integration** - Combining spatial transcriptomics with proteomics, metabolomics
4. **Sub-cellular Resolution** - Analyzing RNA localization patterns
5. **Spatial Perturbations** - Inferring causal relationships from spatial CRISPR screens

### References
- Ståhl et al. (2016). Visualization and analysis of gene expression in tissue sections by spatial transcriptomics
- Svensson et al. (2018). SpatialDE: identification of spatially variable genes
- Dries et al. (2021). Giotto: a toolbox for integrative analysis and visualization of spatial expression data
- Cable et al. (2022). Robust decomposition of cell type mixtures in spatial transcriptomics