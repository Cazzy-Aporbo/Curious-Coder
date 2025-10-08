# Multi-Omics Data Integration: Unified Analysis Across Molecular Layers
## Combining Genomics, Transcriptomics, Proteomics, and Beyond

### Intent
Biological systems operate across multiple molecular layers - DNA variations affect RNA expression, which influences protein abundance, metabolite levels, and ultimately phenotypes. This document provides mathematically rigorous methods for integrating multi-omics data, addressing challenges of different scales, noise characteristics, and missing modalities.

### Mathematical Framework for Multi-Omics

**Data Structure:**
```
X^(1), X^(2), ..., X^(M) where:
- M = number of omics layers
- X^(m) ∈ ℝ^(n × p_m)
- n = samples (may differ across modalities)
- p_m = features in modality m

Challenges:
- p_genomics ~ 10⁶ (SNPs)
- p_transcriptomics ~ 10⁴ (genes)  
- p_proteomics ~ 10³ (proteins)
- p_metabolomics ~ 10² (metabolites)
```

**Integration Objectives:**
1. **Dimension reduction**: Find shared low-dimensional space
2. **Feature selection**: Identify cross-omic associations
3. **Prediction**: Use all omics for phenotype prediction
4. **Network inference**: Build multi-layer networks
5. **Clustering**: Find multi-omic subtypes

### Canonical Correlation Analysis (CCA) Family

#### 1. Classical CCA for Two Omics

```python
def canonical_correlation_analysis(X1, X2, n_components=2):
    """
    Find maximally correlated linear combinations
    
    Objective: maximize corr(U, V) where U = X1·W1, V = X2·W2
    """
    import numpy as np
    from scipy import linalg
    
    n = X1.shape[0]
    p1, p2 = X1.shape[1], X2.shape[1]
    
    # Center data
    X1 = X1 - X1.mean(axis=0)
    X2 = X2 - X2.mean(axis=0)
    
    # Covariance matrices
    C11 = X1.T @ X1 / (n - 1)
    C22 = X2.T @ X2 / (n - 1)  
    C12 = X1.T @ X2 / (n - 1)
    
    # Add regularization for stability
    reg = 1e-4
    C11 += reg * np.eye(p1)
    C22 += reg * np.eye(p2)
    
    # Solve generalized eigenvalue problem
    # C11^(-1/2) C12 C22^(-1) C21 C11^(-1/2) w = ρ² w
    
    C11_inv_sqrt = linalg.fractional_matrix_power(C11, -0.5)
    C22_inv_sqrt = linalg.fractional_matrix_power(C22, -0.5)
    
    # SVD of normalized cross-covariance
    K = C11_inv_sqrt @ C12 @ C22_inv_sqrt
    U, s, Vt = linalg.svd(K, full_matrices=False)
    
    # Canonical weights
    W1 = C11_inv_sqrt @ U[:, :n_components]
    W2 = C22_inv_sqrt @ Vt.T[:, :n_components]
    
    # Canonical variables
    U = X1 @ W1
    V = X2 @ W2
    
    # Canonical correlations
    cancorrs = s[:n_components]
    
    return U, V, W1, W2, cancorrs

def sparse_cca(X1, X2, n_components=2, l1_ratio=0.5):
    """
    Sparse CCA for high-dimensional omics
    """
    from sklearn.cross_decomposition import CCA
    
    # Use penalized CCA for sparsity
    class SparseCCA:
        def __init__(self, n_components, c1, c2):
            self.n_components = n_components
            self.c1 = c1  # L1 penalty for X1
            self.c2 = c2  # L1 penalty for X2
            
        def fit(self, X1, X2):
            n, p1 = X1.shape
            p2 = X2.shape[1]
            
            # Initialize
            W1 = np.random.randn(p1, self.n_components)
            W2 = np.random.randn(p2, self.n_components)
            
            for comp in range(self.n_components):
                converged = False
                
                while not converged:
                    # Update W1 (with L1 penalty)
                    u_old = X1 @ W1[:, comp]
                    v = X2 @ W2[:, comp]
                    
                    w1_new = X1.T @ v
                    w1_new = self.soft_threshold(w1_new, self.c1)
                    w1_new = w1_new / (np.linalg.norm(w1_new) + 1e-10)
                    
                    # Update W2 (with L1 penalty)
                    u = X1 @ w1_new
                    w2_new = X2.T @ u
                    w2_new = self.soft_threshold(w2_new, self.c2)
                    w2_new = w2_new / (np.linalg.norm(w2_new) + 1e-10)
                    
                    # Check convergence
                    u_new = X1 @ w1_new
                    converged = np.abs(np.corrcoef(u_old, u_new)[0, 1]) > 0.99
                    
                    W1[:, comp] = w1_new
                    W2[:, comp] = w2_new
            
            self.W1 = W1
            self.W2 = W2
            
            return self
        
        def soft_threshold(self, x, threshold):
            """
            Soft thresholding for L1 penalty
            """
            return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    model = SparseCCA(n_components, c1=l1_ratio, c2=l1_ratio)
    model.fit(X1, X2)
    
    return model.W1, model.W2
```

#### 2. Multi-CCA for Multiple Omics

```python
def multi_cca(data_list, n_components=2):
    """
    Generalized CCA for multiple data modalities
    
    Objective: maximize Σ_ij corr(U_i, U_j)
    """
    M = len(data_list)  # Number of modalities
    n = data_list[0].shape[0]  # Samples
    
    # Stack all data
    X_concat = np.hstack(data_list)
    p_total = X_concat.shape[1]
    
    # Block structure
    block_indices = []
    start = 0
    for X in data_list:
        end = start + X.shape[1]
        block_indices.append((start, end))
        start = end
    
    # Generalized eigenvalue problem
    # Maximize trace(W^T C W) subject to W^T W = I
    
    # Cross-covariance matrix
    C = np.zeros((p_total, p_total))
    
    for i, X_i in enumerate(data_list):
        for j, X_j in enumerate(data_list):
            if i != j:
                start_i, end_i = block_indices[i]
                start_j, end_j = block_indices[j]
                
                C[start_i:end_i, start_j:end_j] = X_i.T @ X_j / (n - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Top components
    idx = eigenvalues.argsort()[-n_components:][::-1]
    W = eigenvectors[:, idx]
    
    # Extract weights for each modality
    weights = []
    for start, end in block_indices:
        weights.append(W[start:end, :])
    
    # Compute latent variables
    latent = []
    for X, w in zip(data_list, weights):
        latent.append(X @ w)
    
    return latent, weights
```

### Matrix Factorization Methods

#### 1. MOFA (Multi-Omics Factor Analysis)

```python
def mofa_integration(data_dict, n_factors=10, convergence_mode='slow'):
    """
    Bayesian factor analysis for multi-omics
    
    Model: X^(m) = W^(m) Z + ε^(m)
    """
    from mofapy2 import run_mofa
    
    class MOFA:
        def __init__(self, n_factors):
            self.K = n_factors
            
        def fit(self, data_dict):
            """
            data_dict: {modality_name: data_matrix}
            """
            M = len(data_dict)  # Number of views
            N = list(data_dict.values())[0].shape[0]  # Samples
            
            # Initialize parameters
            self.Z = np.random.randn(N, self.K)  # Latent factors
            self.W = {}  # Loadings per modality
            self.tau = {}  # Noise precision
            self.alpha = {}  # ARD parameters
            
            for m, (name, X) in enumerate(data_dict.items()):
                D_m = X.shape[1]
                self.W[name] = np.random.randn(D_m, self.K)
                self.tau[name] = 1.0
                self.alpha[name] = np.ones(self.K)
            
            # Variational Bayes inference
            for iteration in range(1000):
                # E-step: Update latent factors
                precision_z = np.eye(self.K)
                mean_z = np.zeros((N, self.K))
                
                for name, X in data_dict.items():
                    W_m = self.W[name]
                    tau_m = self.tau[name]
                    
                    precision_z += tau_m * W_m.T @ W_m
                    mean_z += tau_m * X @ W_m
                
                # Solve for each sample
                for n in range(N):
                    self.Z[n] = np.linalg.solve(precision_z, mean_z[n])
                
                # M-step: Update loadings
                for name, X in data_dict.items():
                    # Update W
                    precision_w = self.tau[name] * (self.Z.T @ self.Z)
                    precision_w += np.diag(self.alpha[name])
                    
                    for d in range(X.shape[1]):
                        mean_w = self.tau[name] * self.Z.T @ X[:, d]
                        self.W[name][d] = np.linalg.solve(precision_w, mean_w)
                    
                    # Update tau (noise)
                    residuals = X - self.Z @ self.W[name].T
                    self.tau[name] = N * X.shape[1] / np.sum(residuals**2)
                    
                    # Update alpha (ARD)
                    for k in range(self.K):
                        self.alpha[name][k] = X.shape[1] / np.sum(self.W[name][:, k]**2)
                
                # Check convergence
                if iteration % 10 == 0:
                    elbo = self.compute_elbo(data_dict)
                    if iteration > 0 and abs(elbo - prev_elbo) < 1e-4:
                        break
                    prev_elbo = elbo
            
            # Compute variance explained
            self.compute_variance_explained(data_dict)
            
            return self
        
        def compute_variance_explained(self, data_dict):
            """
            R² per factor per modality
            """
            self.r2 = {}
            
            for name, X in data_dict.items():
                r2_factors = []
                
                for k in range(self.K):
                    # Variance explained by factor k
                    factor_pred = np.outer(self.Z[:, k], self.W[name][:, k])
                    ss_factor = np.sum(factor_pred**2)
                    ss_total = np.sum((X - X.mean())**2)
                    
                    r2_factors.append(ss_factor / ss_total)
                
                self.r2[name] = r2_factors
    
    model = MOFA(n_factors)
    model.fit(data_dict)
    
    return model
```

#### 2. Joint NMF (Non-negative Matrix Factorization)

```python
def joint_nmf(data_list, n_components=10, alpha=1.0):
    """
    Joint NMF with shared factors across omics
    
    Minimize: Σ_m ||X^(m) - W^(m)H||² + α||H^(m) - H*||²
    """
    from sklearn.decomposition import NMF
    
    M = len(data_list)
    n_samples = data_list[0].shape[0]
    
    # Initialize shared H and individual W
    H_shared = np.random.rand(n_components, n_samples)
    W_list = []
    H_list = []
    
    for X in data_list:
        W = np.random.rand(X.shape[1], n_components)
        H = H_shared.copy()
        W_list.append(W)
        H_list.append(H)
    
    # Alternating optimization
    for iteration in range(100):
        # Update W for each modality
        for m, X in enumerate(data_list):
            # W update (keeping H fixed)
            numerator = X @ H_list[m].T
            denominator = W_list[m] @ H_list[m] @ H_list[m].T + 1e-10
            W_list[m] *= numerator / denominator
        
        # Update H for each modality
        for m, X in enumerate(data_list):
            # H update (with coupling to shared H)
            numerator = W_list[m].T @ X + alpha * H_shared
            denominator = W_list[m].T @ W_list[m] @ H_list[m] + alpha * H_list[m] + 1e-10
            H_list[m] *= numerator / denominator
        
        # Update shared H
        H_shared = np.mean(H_list, axis=0)
        
        # Check convergence
        if iteration % 10 == 0:
            loss = 0
            for m, X in enumerate(data_list):
                reconstruction = W_list[m] @ H_list[m]
                loss += np.sum((X.T - reconstruction)**2)
                loss += alpha * np.sum((H_list[m] - H_shared)**2)
            
            if iteration > 0 and abs(loss - prev_loss) / prev_loss < 1e-4:
                break
            prev_loss = loss
    
    return W_list, H_shared
```

### Deep Learning Integration Methods

#### 1. Variational Autoencoders for Multi-Omics

```python
def multi_omics_vae(data_shapes, latent_dim=20):
    """
    VAE with shared latent space for multi-omics
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, losses
    
    class MultiOmicsVAE(tf.keras.Model):
        def __init__(self, data_shapes, latent_dim):
            super(MultiOmicsVAE, self).__init__()
            self.latent_dim = latent_dim
            self.data_shapes = data_shapes
            
            # Encoders for each modality
            self.encoders = {}
            for name, shape in data_shapes.items():
                self.encoders[name] = self.build_encoder(shape, latent_dim)
            
            # Shared latent layer
            self.z_mean_layer = layers.Dense(latent_dim)
            self.z_log_var_layer = layers.Dense(latent_dim)
            
            # Decoders for each modality
            self.decoders = {}
            for name, shape in data_shapes.items():
                self.decoders[name] = self.build_decoder(latent_dim, shape)
        
        def build_encoder(self, input_dim, latent_dim):
            """
            Encoder network for one modality
            """
            encoder = models.Sequential([
                layers.Dense(512, activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu')
            ])
            return encoder
        
        def build_decoder(self, latent_dim, output_dim):
            """
            Decoder network for one modality
            """
            decoder = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(512, activation='relu'),
                layers.Dense(output_dim)
            ])
            return decoder
        
        def encode(self, inputs_dict):
            """
            Encode all modalities to shared latent space
            """
            # Concatenate encoded representations
            encoded = []
            
            for name, x in inputs_dict.items():
                if x is not None:  # Handle missing modalities
                    enc = self.encoders[name](x)
                    encoded.append(enc)
            
            # Combine encodings (mean pooling)
            if len(encoded) > 1:
                combined = tf.reduce_mean(tf.stack(encoded), axis=0)
            else:
                combined = encoded[0]
            
            # Latent distribution parameters
            z_mean = self.z_mean_layer(combined)
            z_log_var = self.z_log_var_layer(combined)
            
            return z_mean, z_log_var
        
        def reparameterize(self, z_mean, z_log_var):
            """
            Reparameterization trick
            """
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        def decode(self, z):
            """
            Decode latent to all modalities
            """
            reconstructed = {}
            
            for name, decoder in self.decoders.items():
                reconstructed[name] = decoder(z)
            
            return reconstructed
        
        def call(self, inputs_dict):
            z_mean, z_log_var = self.encode(inputs_dict)
            z = self.reparameterize(z_mean, z_log_var)
            reconstructed = self.decode(z)
            
            # Add KL loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            self.add_loss(kl_loss)
            
            return reconstructed
    
    return MultiOmicsVAE(data_shapes, latent_dim)
```

#### 2. Attention-Based Integration

```python
def attention_integration_network(modality_dims, output_dim):
    """
    Multi-modal integration with cross-modal attention
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    class CrossModalAttention(layers.Layer):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.query = layers.Dense(dim)
            self.key = layers.Dense(dim)
            self.value = layers.Dense(dim)
            
        def call(self, x1, x2):
            """
            x1 attends to x2
            """
            Q = self.query(x1)
            K = self.key(x2)
            V = self.value(x2)
            
            # Scaled dot-product attention
            scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(self.dim))
            attention_weights = tf.nn.softmax(scores)
            
            attended = tf.matmul(attention_weights, V)
            
            return attended + x1  # Residual connection
    
    # Build model
    inputs = {}
    encoded = {}
    
    # Encode each modality
    for name, dim in modality_dims.items():
        inputs[name] = layers.Input(shape=(dim,), name=f'input_{name}')
        
        # Modality-specific encoder
        x = layers.Dense(256, activation='relu')(inputs[name])
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        encoded[name] = x
    
    # Cross-modal attention
    attention_layer = CrossModalAttention(128)
    
    # Each modality attends to others
    attended = {}
    for name1 in modality_dims:
        attended_sum = encoded[name1]
        
        for name2 in modality_dims:
            if name1 != name2:
                attended_sum = attention_layer(attended_sum, encoded[name2])
        
        attended[name1] = attended_sum
    
    # Combine all modalities
    combined = layers.Concatenate()(list(attended.values()))
    
    # Final prediction
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(output_dim, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model
```

### Network-Based Integration

```python
def similarity_network_fusion(data_list, n_neighbors=20, n_iterations=20):
    """
    SNF: Fuse multiple similarity networks
    """
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    
    def make_affinity_matrix(dist_matrix, n_neighbors):
        """
        Create affinity matrix from distance matrix
        """
        n = dist_matrix.shape[0]
        affinity = np.zeros_like(dist_matrix)
        
        for i in range(n):
            # Find k nearest neighbors
            neighbors = np.argsort(dist_matrix[i])[:n_neighbors+1]
            
            # Gaussian kernel weighted by distance
            for j in neighbors:
                if i != j:
                    # Average distance to neighbors
                    avg_dist = (dist_matrix[i, neighbors].mean() + 
                               dist_matrix[j, neighbors].mean()) / 2
                    
                    affinity[i, j] = np.exp(-dist_matrix[i, j]**2 / (avg_dist * 0.5))
        
        return affinity
    
    def normalize_network(W):
        """
        Normalize network for fusion
        """
        D = np.sum(W, axis=1)
        D[D == 0] = 1e-10
        D_sqrt = np.diag(1 / np.sqrt(D))
        
        return D_sqrt @ W @ D_sqrt
    
    # Create similarity networks
    networks = []
    
    for data in data_list:
        # Distance matrix
        dist = euclidean_distances(data)
        
        # Affinity matrix
        affinity = make_affinity_matrix(dist, n_neighbors)
        
        # Normalize
        normalized = normalize_network(affinity)
        
        networks.append(normalized)
    
    # Fusion iterations
    n_modalities = len(networks)
    fused = networks.copy()
    
    for iteration in range(n_iterations):
        updated = []
        
        for i in range(n_modalities):
            # Average other networks
            others = []
            for j in range(n_modalities):
                if i != j:
                    others.append(fused[j])
            
            P_others = np.mean(others, axis=0)
            
            # Update rule
            S_i = networks[i]  # Original network
            P_i_new = S_i @ P_others @ S_i.T
            
            # Normalize
            P_i_new = normalize_network(P_i_new)
            
            updated.append(P_i_new)
        
        fused = updated
    
    # Final fused network
    W_fused = np.mean(fused, axis=0)
    
    # Spectral clustering on fused network
    from sklearn.cluster import SpectralClustering
    
    clustering = SpectralClustering(
        n_clusters=3,  # Or determine automatically
        affinity='precomputed'
    )
    
    labels = clustering.fit_predict(W_fused)
    
    return W_fused, labels

def multiplex_network_analysis(networks_dict):
    """
    Analyze multi-layer biological networks
    """
    import networkx as nx
    
    # Create multiplex network
    multiplex = {}
    
    for layer_name, adjacency in networks_dict.items():
        G = nx.from_numpy_array(adjacency)
        multiplex[layer_name] = G
    
    # Inter-layer connections
    n_nodes = list(networks_dict.values())[0].shape[0]
    
    # Compute multiplex metrics
    metrics = {}
    
    # 1. Multiplex degree
    multiplex_degree = np.zeros(n_nodes)
    for G in multiplex.values():
        for node in range(n_nodes):
            multiplex_degree[node] += G.degree(node)
    
    metrics['multiplex_degree'] = multiplex_degree
    
    # 2. Participation coefficient
    participation = np.zeros(n_nodes)
    
    for node in range(n_nodes):
        k_total = multiplex_degree[node]
        
        if k_total > 0:
            p = 0
            for G in multiplex.values():
                k_layer = G.degree(node)
                p += (k_layer / k_total)**2
            
            participation[node] = 1 - p
    
    metrics['participation'] = participation
    
    # 3. Multi-layer modularity
    from sklearn.cluster import SpectralClustering
    
    # Aggregate adjacency
    aggregated = np.mean(list(networks_dict.values()), axis=0)
    
    clustering = SpectralClustering(n_clusters=5, affinity='precomputed')
    communities = clustering.fit_predict(aggregated)
    
    metrics['communities'] = communities
    
    return metrics
```

### Statistical Testing for Multi-Omics

```python
def multi_omics_association_testing(omics_dict, phenotype, method='cca'):
    """
    Test associations between multi-omics and phenotype
    """
    from scipy import stats
    from sklearn.cross_decomposition import PLSRegression
    
    if method == 'cca':
        # Canonical correlation with phenotype
        X_combined = np.hstack(list(omics_dict.values()))
        
        U, V, W_x, W_y, cancorrs = canonical_correlation_analysis(
            X_combined, phenotype.reshape(-1, 1)
        )
        
        # Permutation test for significance
        n_perms = 1000
        null_cancorrs = []
        
        for _ in range(n_perms):
            phenotype_perm = np.random.permutation(phenotype)
            _, _, _, _, cancorr_perm = canonical_correlation_analysis(
                X_combined, phenotype_perm.reshape(-1, 1)
            )
            null_cancorrs.append(cancorr_perm[0])
        
        p_value = np.mean(null_cancorrs >= cancorrs[0])
        
    elif method == 'pls':
        # Partial Least Squares
        X_combined = np.hstack(list(omics_dict.values()))
        
        pls = PLSRegression(n_components=5)
        pls.fit(X_combined, phenotype)
        
        # Variance explained
        r2 = pls.score(X_combined, phenotype)
        
        # Permutation test
        null_r2 = []
        for _ in range(1000):
            phenotype_perm = np.random.permutation(phenotype)
            pls_perm = PLSRegression(n_components=5)
            pls_perm.fit(X_combined, phenotype_perm)
            null_r2.append(pls_perm.score(X_combined, phenotype_perm))
        
        p_value = np.mean(null_r2 >= r2)
    
    elif method == 'kernel':
        # Kernel-based test (HSIC)
        from sklearn.metrics.pairwise import rbf_kernel
        
        # Compute kernel for each omics
        kernels = []
        for name, X in omics_dict.items():
            K = rbf_kernel(X)
            # Center kernel
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            K_centered = H @ K @ H
            kernels.append(K_centered)
        
        # Combined kernel
        K_combined = np.mean(kernels, axis=0)
        
        # Kernel for phenotype
        K_y = rbf_kernel(phenotype.reshape(-1, 1))
        K_y_centered = H @ K_y @ H
        
        # HSIC statistic
        hsic = np.trace(K_combined @ K_y_centered) / (n - 1)**2
        
        # Permutation test
        null_hsic = []
        for _ in range(1000):
            perm_idx = np.random.permutation(n)
            K_y_perm = K_y_centered[perm_idx][:, perm_idx]
            hsic_perm = np.trace(K_combined @ K_y_perm) / (n - 1)**2
            null_hsic.append(hsic_perm)
        
        p_value = np.mean(null_hsic >= hsic)
    
    return p_value

def differential_correlation_analysis(omics1, omics2, condition):
    """
    Find correlations that change between conditions
    """
    # Split by condition
    mask_0 = condition == 0
    mask_1 = condition == 1
    
    # Compute correlations in each condition
    corr_0 = np.corrcoef(omics1[mask_0].T, omics2[mask_0].T)
    corr_1 = np.corrcoef(omics1[mask_1].T, omics2[mask_1].T)
    
    n1, n2 = omics1.shape[1], omics2.shape[1]
    
    # Extract cross-correlation blocks
    corr_0_cross = corr_0[:n1, n1:]
    corr_1_cross = corr_1[:n1, n1:]
    
    # Difference in correlations
    corr_diff = corr_1_cross - corr_0_cross
    
    # Fisher z-transformation for testing
    z_0 = np.arctanh(corr_0_cross)
    z_1 = np.arctanh(corr_1_cross)
    
    n_0 = mask_0.sum()
    n_1 = mask_1.sum()
    
    # Standard error
    se = np.sqrt(1/(n_0 - 3) + 1/(n_1 - 3))
    
    # Z-scores
    z_scores = (z_1 - z_0) / se
    
    # P-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # FDR correction
    from statsmodels.stats.multitest import fdrcorrection
    
    p_flat = p_values.flatten()
    _, q_values = fdrcorrection(p_flat)
    q_values = q_values.reshape(p_values.shape)
    
    return corr_diff, z_scores, q_values
```

### Complete Pipeline

```python
class MultiOmicsIntegrationPipeline:
    """
    Complete pipeline for multi-omics integration
    """
    
    def __init__(self, data_dict, sample_metadata=None):
        """
        data_dict: {modality_name: data_matrix}
        """
        self.data = data_dict
        self.metadata = sample_metadata
        self.n_samples = list(data_dict.values())[0].shape[0]
        self.modalities = list(data_dict.keys())
        
    def preprocess(self, scale=True, impute_missing=True):
        """
        Preprocessing per modality
        """
        processed = {}
        
        for name, X in self.data.items():
            # Handle missing values
            if impute_missing and np.isnan(X).any():
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Scaling
            if scale:
                if name in ['genomics', 'methylation']:
                    # Don't scale binary/categorical
                    pass
                elif name in ['transcriptomics', 'proteomics']:
                    # Log transform and scale
                    X = np.log1p(X)
                    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                elif name == 'metabolomics':
                    # Pareto scaling
                    X = (X - X.mean(axis=0)) / (np.sqrt(X.std(axis=0)) + 1e-8)
            
            processed[name] = X
        
        self.processed_data = processed
        
        return processed
    
    def integrate(self, method='mofa', **kwargs):
        """
        Apply integration method
        """
        if method == 'mofa':
            self.model = mofa_integration(self.processed_data, **kwargs)
            self.integrated = self.model.Z
            
        elif method == 'cca':
            # Pairwise CCA for all modalities
            latent, weights = multi_cca(
                list(self.processed_data.values()), 
                **kwargs
            )
            self.integrated = np.hstack(latent)
            
        elif method == 'snf':
            W_fused, labels = similarity_network_fusion(
                list(self.processed_data.values()),
                **kwargs
            )
            self.integrated = W_fused
            self.clusters = labels
            
        elif method == 'vae':
            vae = multi_omics_vae(
                {name: X.shape[1] for name, X in self.processed_data.items()},
                **kwargs
            )
            # Train VAE
            # ... training code ...
            self.model = vae
            
        return self.integrated
    
    def downstream_analysis(self, phenotype=None):
        """
        Downstream analyses on integrated data
        """
        results = {}
        
        # Clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3)
        results['clusters'] = kmeans.fit_predict(self.integrated)
        
        # If phenotype provided
        if phenotype is not None:
            # Classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            rf = RandomForestClassifier(n_estimators=100)
            scores = cross_val_score(rf, self.integrated, phenotype, cv=5)
            results['classification_accuracy'] = scores.mean()
            
            # Feature importance
            rf.fit(self.integrated, phenotype)
            results['feature_importance'] = rf.feature_importances_
        
        # Pathway analysis
        results['enrichment'] = self.pathway_enrichment()
        
        return results
    
    def visualize(self):
        """
        Comprehensive visualization
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # t-SNE of integrated data
        tsne = TSNE(n_components=2)
        embedded = tsne.fit_transform(self.integrated)
        
        axes[0, 0].scatter(embedded[:, 0], embedded[:, 1])
        axes[0, 0].set_title('Integrated Data (t-SNE)')
        
        # Variance explained per modality
        if hasattr(self.model, 'r2'):
            for i, (name, r2) in enumerate(self.model.r2.items()):
                axes[0, 1].bar(range(len(r2)), r2, label=name, alpha=0.7)
            axes[0, 1].set_title('Variance Explained per Factor')
            axes[0, 1].legend()
        
        # Correlation heatmap
        corr = np.corrcoef(self.integrated.T)
        im = axes[0, 2].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 2].set_title('Factor Correlations')
        plt.colorbar(im, ax=axes[0, 2])
        
        # Per-modality contributions
        # ... additional visualizations ...
        
        plt.tight_layout()
        return fig
```

### References
- Argelaguet, R. et al. (2018). Multi-Omics Factor Analysis
- Rohart, F. et al. (2017). mixOmics: An R package for 'omics feature selection and multiple data integration
- Wang, B. et al. (2014). Similarity network fusion for aggregating data types on a genomic scale
- Hasin, Y. et al. (2017). Multi-omics approaches to disease