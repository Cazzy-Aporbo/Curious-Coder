# Manifold Learning and Nonlinear Dimension Reduction for Biological Data
## Discovering Low-Dimensional Structure in High-Dimensional Life

### Intent
Biological data often lies on low-dimensional manifolds embedded in high-dimensional space. A cell's transcriptome has ~20,000 dimensions but follows developmental trajectories through a much lower-dimensional state space. This document provides rigorous mathematical frameworks for discovering and exploiting these manifold structures.

### The Manifold Hypothesis in Biology

**Mathematical Foundation:**

Data points {x₁, ..., xₙ} ⊂ ℝᵖ lie on or near a d-dimensional manifold M where d << p.

**Biological Examples:**
- Cell differentiation trajectories (p~20,000 genes, d~3-5 developmental axes)
- Protein conformational landscapes (p~1000s atoms, d~10s collective motions)
- Neural population dynamics (p~1000s neurons, d~10s behavioral states)
- Morphogenesis (p~1000s cells, d~few developmental fields)

### Core Mathematical Concepts

#### Manifold Definition
A d-dimensional manifold M is a topological space where every point has a neighborhood homeomorphic to ℝᵈ.

**Key Properties for Data Analysis:**
```python
def manifold_properties(data_points):
    """
    Essential properties we care about in practice
    """
    properties = {
        'intrinsic_dimension': d,  # True dimensionality
        'smoothness': C_k,  # k-times differentiable
        'geodesic_distance': d_M(x, y),  # Distance along manifold
        'tangent_space': T_x(M),  # Local linear approximation
        'curvature': K(x),  # How manifold bends
    }
    return properties
```

#### Geodesic Distance vs Euclidean Distance

```python
def illustrate_geodesic_vs_euclidean():
    """
    Swiss roll example: Points nearby in ℝ³ but far on manifold
    """
    # Generate Swiss roll
    n_points = 1000
    t = 3 * np.pi * (1 + 2 * np.random.rand(n_points))
    height = 20 * np.random.rand(n_points)
    
    X = np.zeros((n_points, 3))
    X[:, 0] = t * np.cos(t)
    X[:, 1] = height
    X[:, 2] = t * np.sin(t)
    
    # Two points: close in Euclidean, far in geodesic
    p1, p2 = 0, 500
    euclidean_dist = np.linalg.norm(X[p1] - X[p2])
    geodesic_dist = abs(t[p1] - t[p2])  # True distance along spiral
    
    print(f"Euclidean distance: {euclidean_dist:.2f}")
    print(f"Geodesic distance: {geodesic_dist:.2f}")
    print(f"Ratio: {geodesic_dist/euclidean_dist:.2f}")
    
    return X, t
```

### Classical Methods: Linear Approximations

#### Principal Component Analysis (PCA) - Linear Manifold
Already covered extensively, but key limitation:
- Assumes manifold is a linear subspace
- Preserves global distances, not local structure

#### Multi-Dimensional Scaling (MDS)
**Objective:** Preserve pairwise distances

```
minimize Σᵢⱼ (d_ij - ||yᵢ - yⱼ||²)²

where d_ij = distance in original space
      yᵢ = embedding in low dimension
```

**Classical MDS Algorithm:**
```python
def classical_mds(distance_matrix, n_components=2):
    """
    Distance-preserving embedding
    """
    n = distance_matrix.shape[0]
    
    # Double centering
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    B = -0.5 * H @ (distance_matrix**2) @ H  # Double-centered matrix
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by eigenvalue
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Keep top components
    Lambda = np.diag(np.sqrt(np.maximum(eigenvalues[:n_components], 0)))
    V = eigenvectors[:, :n_components]
    
    # Low-dimensional embedding
    Y = V @ Lambda
    
    return Y
```

### Nonlinear Manifold Learning Methods

#### 1. Isomap: Geodesic Distance Preservation

**Key Insight:** Approximate geodesic distances via shortest paths in neighborhood graph

**Algorithm:**
```python
def isomap(X, n_neighbors=5, n_components=2):
    """
    Isometric feature mapping
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import shortest_path
    
    n_samples = X.shape[0]
    
    # Step 1: Construct neighborhood graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Build adjacency matrix
    graph = np.full((n_samples, n_samples), np.inf)
    for i in range(n_samples):
        graph[i, indices[i]] = distances[i]
        graph[indices[i], i] = distances[i]  # Symmetric
    
    # Step 2: Compute shortest paths (geodesic distances)
    geodesic_distances = shortest_path(graph, directed=False)
    
    # Step 3: Apply classical MDS to geodesic distances
    embedding = classical_mds(geodesic_distances, n_components)
    
    return embedding, geodesic_distances
```

**Theoretical Guarantee:** For smooth manifolds with sufficient sampling:
```
||d_M(xᵢ, xⱼ) - d_G(xᵢ, xⱼ)|| = O(1/n^(2/d))

where d_M = true geodesic distance
      d_G = graph shortest path
```

#### 2. Locally Linear Embedding (LLE): Local Geometry Preservation

**Assumption:** Manifold is locally linear

**Objective:** Preserve local linear relationships
```
minimize Σᵢ ||xᵢ - Σⱼ∈N(i) wᵢⱼxⱼ||²

subject to Σⱼ wᵢⱼ = 1
```

```python
def lle(X, n_neighbors=10, n_components=2):
    """
    Locally Linear Embedding
    """
    n_samples = X.shape[0]
    
    # Step 1: Find neighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Step 2: Compute reconstruction weights
    W = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        # Local covariance
        Xi = X[indices[i, 1:]] - X[i]  # Centered neighbors
        C = Xi @ Xi.T
        
        # Regularization for numerical stability
        C += np.eye(n_neighbors - 1) * 1e-3 * np.trace(C)
        
        # Solve for weights
        w = np.linalg.solve(C, np.ones(n_neighbors - 1))
        w /= w.sum()
        
        # Store in sparse matrix
        W[i, indices[i, 1:]] = w
    
    # Step 3: Find embedding that preserves weights
    M = (np.eye(n_samples) - W).T @ (np.eye(n_samples) - W)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # Skip first eigenvector (all ones)
    embedding = eigenvectors[:, 1:n_components+1]
    
    return embedding
```

#### 3. t-SNE: Probability Distribution Matching

**Objective:** Preserve local neighborhoods via probability distributions

**High-dimensional similarities:**
```
p_ij = exp(-||xᵢ - xⱼ||² / 2σᵢ²) / Σₖ≠ᵢ exp(-||xᵢ - xₖ||² / 2σᵢ²)
```

**Low-dimensional similarities (t-distribution):**
```
q_ij = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σₖ≠ₗ (1 + ||yₖ - yₗ||²)⁻¹
```

**KL Divergence minimization:**
```
minimize KL(P||Q) = Σᵢⱼ p_ij log(p_ij / q_ij)
```

```python
def tsne_gradient(Y, P, learning_rate=200, momentum=0.8):
    """
    One iteration of t-SNE gradient descent
    """
    n = Y.shape[0]
    
    # Compute pairwise distances
    sum_Y = np.sum(Y**2, axis=1)
    D = sum_Y[np.newaxis, :] + sum_Y[:, np.newaxis] - 2 * Y @ Y.T
    
    # Student-t distribution
    Q = 1 / (1 + D)
    np.fill_diagonal(Q, 0)
    Q = Q / Q.sum()
    Q = np.maximum(Q, 1e-12)
    
    # Gradient
    PQ_diff = P - Q
    gradient = np.zeros_like(Y)
    
    for i in range(n):
        diff = Y[i] - Y
        gradient[i] = 4 * np.sum(
            (PQ_diff[i, :] * Q[i, :])[:, np.newaxis] * diff, 
            axis=0
        )
    
    return gradient

def compute_perplexity(distances, sigma):
    """
    Perplexity = 2^H(P) where H is Shannon entropy
    Controls effective number of neighbors
    """
    P = np.exp(-distances**2 / (2 * sigma**2))
    P = P / P.sum()
    
    H = -np.sum(P * np.log2(P + 1e-10))
    perplexity = 2**H
    
    return perplexity, P
```

#### 4. UMAP: Topological Structure Preservation

**Theoretical Foundation:** Category theory and Riemannian geometry

**Local connectivity via fuzzy simplicial sets:**
```
μ(xᵢ, xⱼ) = exp(-(d(xᵢ, xⱼ) - ρᵢ) / σᵢ)

where ρᵢ = distance to nearest neighbor
      σᵢ chosen to achieve desired n_neighbors
```

```python
def umap_loss(Y, graph, a=1.929, b=0.7915):
    """
    UMAP cross-entropy loss
    
    Attractive: -Σ w_ij log(f(||yᵢ - yⱼ||))
    Repulsive: -Σ (1-w_ij) log(1 - f(||yᵢ - yⱼ||))
    
    where f(d) = (1 + a*d^(2b))^(-1)
    """
    n_samples = Y.shape[0]
    
    # Compute low-dimensional distances
    pdist = pairwise_distances(Y)
    
    # Low-dimensional probabilities
    Q = 1 / (1 + a * pdist**(2 * b))
    
    # Cross-entropy
    CE = -graph * np.log(Q + 1e-10) - (1 - graph) * np.log(1 - Q + 1e-10)
    
    return CE.sum()
```

### Diffusion Maps: Markov Chain on Manifold

**Key Idea:** Random walk reveals manifold geometry

**Transition probability:**
```
P(x → y) = K(x, y) / Σz K(x, z)

where K(x, y) = exp(-||x - y||² / ε)
```

**Diffusion distance after t steps:**
```
D_t²(x, y) = Σz (P^t(x, z) - P^t(y, z))² / π(z)
```

```python
def diffusion_maps(X, epsilon='auto', n_components=2, t=1):
    """
    Diffusion maps embedding
    """
    n_samples = X.shape[0]
    
    # Compute kernel matrix
    if epsilon == 'auto':
        # Median heuristic
        distances = pairwise_distances(X)
        epsilon = np.median(distances)**2
    
    K = np.exp(-pairwise_distances(X)**2 / epsilon)
    
    # Normalize to get transition matrix
    D = np.diag(K.sum(axis=1))
    P = np.linalg.inv(D) @ K
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(P)
    
    # Sort by eigenvalue
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    
    # Diffusion embedding
    # Use eigenvectors scaled by eigenvalues^t
    embedding = eigenvectors[:, 1:n_components+1] * (eigenvalues[1:n_components+1]**t)
    
    return embedding, eigenvalues
```

### Biological Applications and Method Selection

| Biological System | Best Method | Why | Key Parameters |
|------------------|------------|-----|----------------|
| **Cell differentiation** | Diffusion Maps / UMAP | Continuous trajectories | t (time), n_neighbors |
| **Single-cell RNA-seq** | UMAP / t-SNE | Preserve local clusters | perplexity, min_dist |
| **Protein dynamics** | Isomap / Diffusion | Preserve distances | epsilon, n_neighbors |
| **Brain connectivity** | Isomap / MDS | Global structure | k (neighbors) |
| **Morphogenesis** | LLE / LTSA | Local linearity | n_neighbors, reg |
| **Metabolic states** | t-SNE / UMAP | Discrete clusters | perplexity, n_neighbors |

### Intrinsic Dimensionality Estimation

```python
def estimate_intrinsic_dimension(X, methods=['mle', 'correlation', 'pca']):
    """
    Multiple methods to estimate manifold dimension
    """
    results = {}
    
    if 'mle' in methods:
        # Maximum Likelihood Estimation (Levina-Bickel)
        k = 10  # neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:, 1:]  # Remove self
        
        # MLE dimension estimate
        d_mle = []
        for i in range(len(X)):
            r_k = distances[i, -1]
            if r_k > 0:
                d_est = k / np.sum(np.log(r_k / distances[i, :-1]))
                d_mle.append(d_est)
        
        results['mle'] = np.median(d_mle)
    
    if 'correlation' in methods:
        # Correlation dimension
        r_vals = np.logspace(-2, 0, 20)
        C_r = []
        
        for r in r_vals:
            count = np.sum(pairwise_distances(X) < r)
            C_r.append(count / (len(X) * (len(X) - 1)))
        
        # Fit line in log-log plot
        log_r = np.log(r_vals[5:15])
        log_C = np.log(C_r[5:15] + 1e-10)
        
        slope, _ = np.polyfit(log_r, log_C, 1)
        results['correlation'] = slope
    
    if 'pca' in methods:
        # PCA-based estimate (elbow method)
        pca = PCA()
        pca.fit(X)
        
        # Find elbow
        explained_var = pca.explained_variance_ratio_
        cumsum = np.cumsum(explained_var)
        
        # 90% variance threshold
        d_pca = np.argmax(cumsum >= 0.9) + 1
        results['pca'] = d_pca
    
    return results
```

### Validation and Quality Assessment

```python
class ManifoldQuality:
    """
    Assess quality of manifold embedding
    """
    
    def __init__(self, X_high, X_low):
        self.X_high = X_high
        self.X_low = X_low
    
    def trustworthiness(self, k=10):
        """
        Measure if neighbors in low-D were neighbors in high-D
        """
        n = len(self.X_high)
        
        # Get neighbors in both spaces
        nbrs_high = NearestNeighbors(n_neighbors=k+1)
        nbrs_high.fit(self.X_high)
        _, indices_high = nbrs_high.kneighbors(self.X_high)
        
        nbrs_low = NearestNeighbors(n_neighbors=k+1)
        nbrs_low.fit(self.X_low)
        _, indices_low = nbrs_low.kneighbors(self.X_low)
        
        trust = 0
        for i in range(n):
            # Points that are neighbors in low-D but not in high-D
            false_neighbors = set(indices_low[i]) - set(indices_high[i])
            
            for j in false_neighbors:
                # Rank of j as neighbor of i in high-D
                rank_high = np.where(indices_high[i] == j)[0]
                if len(rank_high) > 0:
                    trust += rank_high[0] - k
        
        trust = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * trust
        return trust
    
    def continuity(self, k=10):
        """
        Measure if neighbors in high-D remain neighbors in low-D
        """
        # Similar to trustworthiness but reversed
        pass
    
    def stress(self):
        """
        Normalized stress (distance preservation)
        """
        D_high = pairwise_distances(self.X_high)
        D_low = pairwise_distances(self.X_low)
        
        stress = np.sqrt(np.sum((D_high - D_low)**2) / np.sum(D_high**2))
        return stress
    
    def local_property_preservation(self, property_func):
        """
        Check if local properties are preserved
        E.g., density, curvature, etc.
        """
        prop_high = property_func(self.X_high)
        prop_low = property_func(self.X_low)
        
        correlation = np.corrcoef(prop_high, prop_low)[0, 1]
        return correlation
```

### Trajectory Inference on Manifolds

```python
def pseudotime_ordering(embedding, root_cell=None):
    """
    Order cells along developmental trajectory
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=10)
    nbrs.fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    
    # Create weighted adjacency matrix
    n_cells = len(embedding)
    graph = np.full((n_cells, n_cells), np.inf)
    
    for i in range(n_cells):
        graph[i, indices[i]] = distances[i]
    
    # Minimum spanning tree
    mst = minimum_spanning_tree(graph).toarray()
    
    # Diffusion pseudotime from root
    if root_cell is None:
        # Use cell with lowest first PC as root
        root_cell = np.argmin(embedding[:, 0])
    
    # Compute distances from root
    from scipy.sparse.csgraph import shortest_path
    pseudotime = shortest_path(mst, indices=root_cell, directed=False)
    
    return pseudotime
```

### Handling Noise and Outliers

```python
def robust_manifold_learning(X, contamination=0.1):
    """
    Manifold learning robust to outliers
    """
    from sklearn.covariance import EllipticEnvelope
    
    # Step 1: Detect outliers in high-dimensional space
    outlier_detector = EllipticEnvelope(contamination=contamination)
    outlier_detector.fit(X)
    outlier_mask = outlier_detector.predict(X) == -1
    
    # Step 2: Compute embedding on clean data
    X_clean = X[~outlier_mask]
    
    # Use UMAP with custom metric
    from sklearn.manifold import TSNE
    embedding_clean = TSNE(n_components=2).fit_transform(X_clean)
    
    # Step 3: Project outliers
    # Use k-NN regression to place outliers
    nbrs = NearestNeighbors(n_neighbors=5)
    nbrs.fit(X_clean)
    
    embedding = np.zeros((len(X), 2))
    embedding[~outlier_mask] = embedding_clean
    
    for idx in np.where(outlier_mask)[0]:
        distances, indices = nbrs.kneighbors([X[idx]])
        # Weighted average of neighbor positions
        weights = 1 / (distances[0] + 1e-10)
        weights /= weights.sum()
        embedding[idx] = weights @ embedding_clean[indices[0]]
    
    return embedding, outlier_mask
```

### Advanced Topics

#### Riemannian Optimization on Manifolds
```python
def geodesic_gradient_descent(X, objective, manifold_constraint):
    """
    Optimization respecting manifold geometry
    """
    # Project gradient onto tangent space
    grad = compute_gradient(objective, X)
    tangent_grad = project_to_tangent_space(grad, X, manifold_constraint)
    
    # Move along geodesic
    X_new = exponential_map(X, -learning_rate * tangent_grad)
    
    return X_new
```

#### Multi-Scale Manifold Learning
```python
def multiscale_embedding(X, scales=[0.1, 1.0, 10.0]):
    """
    Capture structure at multiple scales
    """
    embeddings = []
    
    for scale in scales:
        # Adjust neighborhood size
        n_neighbors = int(scale * np.sqrt(len(X)))
        
        # Compute embedding at this scale
        embedding = umap.UMAP(n_neighbors=n_neighbors).fit_transform(X)
        embeddings.append(embedding)
    
    # Combine multi-scale information
    combined = np.concatenate(embeddings, axis=1)
    
    # Final embedding
    final = PCA(n_components=2).fit_transform(combined)
    
    return final
```

### References
- Lee, J.A. & Verleysen, M. (2007). Nonlinear Dimensionality Reduction
- McInnes, L. et al. (2018). UMAP: Uniform Manifold Approximation and Projection
- Coifman, R.R. & Lafon, S. (2006). Diffusion maps
- Tenenbaum, J.B. et al. (2000). A global geometric framework for nonlinear dimensionality reduction