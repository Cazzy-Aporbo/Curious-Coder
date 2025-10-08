# Single-Cell Analysis: From Raw Counts to Biological Insights
## Deconstructing Cellular Heterogeneity at Individual Cell Resolution

### Intent
Single-cell technologies have revolutionized biology by revealing cellular heterogeneity invisible to bulk measurements. This document provides rigorous frameworks for analyzing single-cell data, from quality control through trajectory inference, addressing unique challenges like sparsity, technical noise, and the curse of dimensionality at cellular resolution.

### Mathematical Framework for Single-Cell Data

**Data Structure:**
```
X ∈ ℕ^(n×p) where:
- n = number of cells (10³ - 10⁶)
- p = number of genes (10⁴ - 10⁵)
- X_ij = UMI/read count for gene j in cell i
- ~85-95% zeros (dropout + biological zeros)
```

**Generative Model:**
```
True expression: λ_ij ~ LogNormal(μ_j, σ²_j)
Observed counts: X_ij ~ Poisson(s_i · ε_ij · λ_ij)

where:
- s_i = size factor (sequencing depth)
- ε_ij ~ Bernoulli(p_ij) = dropout indicator
- p_ij = logistic(α + β·log(λ_ij)) = dropout probability
```

### Quality Control and Preprocessing

#### 1. Cell and Gene Filtering

```python
def quality_control_pipeline(adata, min_genes=200, min_cells=3, 
                            max_mt_percent=5, max_genes=2500):
    """
    Comprehensive QC for single-cell data
    
    adata: AnnData object with raw counts
    """
    import scanpy as sc
    import numpy as np
    
    print(f"Starting with {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Calculate QC metrics
    # Mitochondrial genes (indicates dying cells)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    
    # Ribosomal genes (batch effects)
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    
    # Calculate percentages
    sc.pp.calculate_qc_metrics(
        adata, 
        qc_vars=['mt', 'ribo'], 
        percent_top=None, 
        log1p=False, 
        inplace=True
    )
    
    # Cell-level QC
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
    adata.obs['log_counts'] = np.log1p(adata.obs['total_counts'])
    
    # Identify outliers using MAD-based method
    def is_outlier(adata, metric, n_mads=5):
        M = adata.obs[metric]
        median = np.median(M)
        mad = np.median(np.abs(M - median))
        
        # Upper and lower bounds
        upper = median + n_mads * mad
        lower = median - n_mads * mad
        
        return (M < lower) | (M > upper)
    
    # Apply filters
    adata.obs['outlier_n_genes'] = is_outlier(adata, 'n_genes')
    adata.obs['outlier_counts'] = is_outlier(adata, 'log_counts')
    adata.obs['outlier_mt'] = adata.obs['pct_counts_mt'] > max_mt_percent
    
    # Doublet detection
    adata.obs['potential_doublet'] = (adata.obs['n_genes'] > max_genes)
    
    # Combined filter
    keep_cells = ~(
        adata.obs['outlier_n_genes'] |
        adata.obs['outlier_counts'] |
        adata.obs['outlier_mt'] |
        adata.obs['potential_doublet'] |
        (adata.obs['n_genes'] < min_genes)
    )
    
    # Gene filtering
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Apply filters
    adata = adata[keep_cells, :]
    
    print(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Removed {(~keep_cells).sum()} low-quality cells")
    
    return adata

def detect_doublets(adata, method='scrublet'):
    """
    Computational doublet detection
    """
    if method == 'scrublet':
        import scrublet as scr
        
        # Initialize scrublet
        scrub = scr.Scrublet(
            adata.X,
            expected_doublet_rate=0.06,  # Typical for 10x
            sim_doublet_ratio=2.0,
            n_neighbors=30
        )
        
        # Run scrublet
        doublet_scores, predicted_doublets = scrub.scrub_doublets(
            min_counts=2,
            min_cells=3,
            min_gene_variability_pctl=85,
            n_prin_comps=30
        )
        
        adata.obs['doublet_score'] = doublet_scores
        adata.obs['predicted_doublet'] = predicted_doublets
        
    elif method == 'doubletfinder':
        # Alternatively, use DoubletFinder logic
        # Simulate artificial doublets
        n_real = adata.n_obs
        n_doublets = int(n_real * 0.5)
        
        # Create artificial doublets
        doublet_idx1 = np.random.choice(n_real, n_doublets)
        doublet_idx2 = np.random.choice(n_real, n_doublets)
        
        doublets = adata.X[doublet_idx1] + adata.X[doublet_idx2]
        
        # Combined data
        combined = np.vstack([adata.X, doublets])
        labels = np.array([0] * n_real + [1] * n_doublets)
        
        # Classify using kNN
        from sklearn.neighbors import KNeighborsClassifier
        
        # PCA first
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        combined_pca = pca.fit_transform(combined)
        
        # kNN classification
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(combined_pca[n_real:], labels[n_real:])  # Train on doublets
        
        # Predict on real cells
        doublet_scores = knn.predict_proba(combined_pca[:n_real])[:, 1]
        
        adata.obs['doublet_score'] = doublet_scores
        adata.obs['predicted_doublet'] = doublet_scores > 0.5
    
    return adata
```

#### 2. Normalization Methods

```python
def normalization_comparison(adata):
    """
    Compare different normalization strategies
    """
    import scanpy as sc
    
    # Store different normalizations
    normalizations = {}
    
    # 1. Library size normalization (CPM)
    adata_cpm = adata.copy()
    sc.pp.normalize_total(adata_cpm, target_sum=1e6)
    normalizations['CPM'] = adata_cpm.X
    
    # 2. Log normalization
    adata_log = adata.copy()
    sc.pp.normalize_total(adata_log, target_sum=1e4)
    sc.pp.log1p(adata_log)
    normalizations['LogNorm'] = adata_log.X
    
    # 3. Scran pooling normalization
    def scran_normalization(adata):
        """
        Pool-based size factor estimation
        """
        # Rough clustering for pooling
        adata_pp = adata.copy()
        sc.pp.normalize_total(adata_pp)
        sc.pp.log1p(adata_pp)
        sc.pp.pca(adata_pp, n_comps=50)
        sc.pp.neighbors(adata_pp)
        sc.tl.leiden(adata_pp, resolution=0.5)
        
        # Compute size factors per cluster
        size_factors = np.ones(adata.n_obs)
        
        for cluster in adata_pp.obs['leiden'].unique():
            mask = adata_pp.obs['leiden'] == cluster
            cluster_data = adata.X[mask]
            
            # Pool cells and compute factors
            pool_size = min(50, cluster_data.shape[0] // 2)
            if pool_size > 5