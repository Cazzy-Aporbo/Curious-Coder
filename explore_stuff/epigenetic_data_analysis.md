# Epigenetic Data Analysis
## Decoding the Regulatory Landscape Above DNA

### Intent
Epigenetics determines how the genome is read, not what it says. This document provides mathematical frameworks for analyzing DNA methylation, histone modifications, chromatin accessibility, and 3D genome organization, integrating these layers to understand gene regulation.

### The Fundamental Concept: Information Beyond Sequence

```
Genome = Hardware (same in all cells)
Epigenome = Software (different in each cell type)

Key insight: Same DNA → Different epigenomes → Different cell fates
```

### DNA Methylation Analysis

#### 1. Bisulfite Sequencing Data Processing

```python
def bisulfite_sequencing_analysis():
    """
    Analyze CpG methylation from bisulfite sequencing
    
    Bisulfite converts: C → U (unmethylated)
                        mC → C (methylated)
    """
    
    def calculate_methylation_level(methylated_counts, unmethylated_counts):
        """
        Beta values: proportion methylated
        """
        total = methylated_counts + unmethylated_counts
        
        # Avoid division by zero
        beta = methylated_counts / (total + 1e-10)
        
        # M-values for statistical analysis (logit transformation)
        # More appropriate for statistical tests (homoscedastic)
        M = np.log2((beta + 0.01) / (1 - beta + 0.01))
        
        return beta, M
    
    def smooth_methylation_signal(positions, methylation_levels, bandwidth=100):
        """
        Local regression smoothing for methylation patterns
        """
        from scipy.interpolate import UnivariateSpline
        
        # Weight by coverage
        weights = np.sqrt(coverage)  # Higher coverage = more reliable
        
        # Fit spline
        spline = UnivariateSpline(positions, methylation_levels, 
                                 w=weights, s=bandwidth)
        
        # Smooth signal
        smooth_positions = np.linspace(positions.min(), positions.max(), 1000)
        smooth_methylation = spline(smooth_positions)
        
        return smooth_positions, smooth_methylation
    
    def detect_dmrs(methylation_data, groups, min_cpgs=3, min_diff=0.2):
        """
        Detect Differentially Methylated Regions (DMRs)
        """
        dmrs = []
        
        # Sliding window approach
        window_size = 500  # bp
        step_size = 100
        
        for chrom in methylation_data.chromosomes:
            chrom_data = methylation_data[chrom]
            
            for start in range(0, len(chrom_data), step_size):
                end = start + window_size
                window_cpgs = chrom_data[start:end]
                
                if len(window_cpgs) < min_cpgs:
                    continue
                
                # Compare groups
                group1_meth = window_cpgs[groups == 0].mean()
                group2_meth = window_cpgs[groups == 1].mean()
                
                diff = group2_meth - group1_meth
                
                # Statistical test (t-test on M-values)
                from scipy import stats
                
                m_values_1 = calculate_methylation_level(
                    window_cpgs[groups == 0]['C'],
                    window_cpgs[groups == 0]['T']
                )[1]
                
                m_values_2 = calculate_methylation_level(
                    window_cpgs[groups == 1]['C'],
                    window_cpgs[groups == 1]['T']
                )[1]
                
                t_stat, p_value = stats.ttest_ind(m_values_1, m_values_2)
                
                if abs(diff) > min_diff and p_value < 0.05:
                    dmrs.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'n_cpgs': len(window_cpgs),
                        'diff': diff,
                        'p_value': p_value
                    })
        
        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests
        p_values = [dmr['p_value'] for dmr in dmrs]
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        
        for i, dmr in enumerate(dmrs):
            dmr['q_value'] = p_adjusted[i]
        
        return dmrs
```

#### 2. Methylation Array Analysis (450K/EPIC)

```python
def methylation_array_analysis():
    """
    Process Illumina methylation arrays
    """
    
    def normalize_methylation_array(signal_intensities, method='swan'):
        """
        Normalization methods for methylation arrays
        """
        
        if method == 'swan':
            # Subset-quantile Within Array Normalization
            
            # Separate Type I and Type II probes
            type1_probes = signal_intensities['Type_I']
            type2_probes = signal_intensities['Type_II']
            
            # Match distributions
            from sklearn.preprocessing import QuantileTransformer
            qt = QuantileTransformer(output_distribution='uniform')
            
            type1_normalized = qt.fit_transform(type1_probes)
            type2_normalized = qt.fit_transform(type2_probes)
            
            return type1_normalized, type2_normalized
        
        elif method == 'quantile':
            # Quantile normalization across samples
            
            def quantile_normalize(df):
                """
                Make distributions identical across samples
                """
                rank_mean = df.stack().groupby(
                    df.rank(method='first').stack().astype(int)
                ).mean()
                
                return df.rank(method='min').stack().astype(int).map(
                    rank_mean
                ).unstack()
            
            return quantile_normalize(signal_intensities)
        
        elif method == 'functional':
            # Functional normalization using control probes
            # Uses PCA on control probes to remove technical variation
            pass
    
    def cell_type_deconvolution(methylation_matrix, reference_profiles):
        """
        Estimate cell type proportions from methylation
        
        Model: Y = X·P + ε
        Y: observed methylation (CpGs × samples)
        X: reference profiles (CpGs × cell types)
        P: proportions (cell types × samples)
        """
        
        from sklearn.linear_model import LinearRegression
        
        # Constrained regression (proportions sum to 1, all ≥ 0)
        from scipy.optimize import nnls
        
        proportions = []
        
        for sample in methylation_matrix.T:
            # Non-negative least squares
            props, _ = nnls(reference_profiles, sample)
            
            # Normalize to sum to 1
            props = props / props.sum()
            
            proportions.append(props)
        
        return np.array(proportions)
    
    def epigenetic_clock(methylation_data, clock='horvath'):
        """
        Predict biological age from methylation
        """
        
        if clock == 'horvath':
            # 353 CpG sites
            cpg_sites = load_horvath_cpgs()
            coefficients = load_horvath_coefficients()
            
            # Select relevant CpGs
            clock_methylation = methylation_data[cpg_sites]
            
            # Transform (Horvath uses custom transformation)
            def transform_age(x):
                adult_age = 20
                if x <= adult_age:
                    return np.log(x + 1) - np.log(adult_age + 1)
                else:
                    return (x - adult_age) / (adult_age + 1)
            
            # Linear model
            predicted_transform = clock_methylation @ coefficients
            
            # Inverse transform
            def inverse_transform(x):
                adult_age = 20
                if x < 0:
                    return np.exp(x + np.log(adult_age + 1)) - 1
                else:
                    return x * (adult_age + 1) + adult_age
            
            predicted_age = inverse_transform(predicted_transform)
            
            return predicted_age
```

### Histone Modification Analysis (ChIP-seq)

```python
def chip_seq_analysis():
    """
    Chromatin Immunoprecipitation sequencing analysis
    """
    
    def call_peaks(treatment_bam, control_bam, method='macs2'):
        """
        Identify regions of histone enrichment
        """
        
        if method == 'macs2':
            # Model-based Analysis of ChIP-seq
            
            # Estimate fragment size
            fragment_size = estimate_fragment_size(treatment_bam)
            
            # Build signal tracks
            treatment_pileup = build_pileup(treatment_bam, fragment_size)
            control_pileup = build_pileup(control_bam, fragment_size)
            
            # Local lambda for background
            def calculate_local_lambda(position, window_sizes=[1000, 10000]):
                lambdas = []
                
                for window in window_sizes:
                    start = position - window // 2
                    end = position + window // 2
                    
                    control_reads = control_pileup[start:end].sum()
                    lambda_bg = control_reads / window * fragment_size
                    lambdas.append(lambda_bg)
                
                # Take maximum (most conservative)
                return max(lambdas)
            
            # Peak calling with Poisson test
            peaks = []
            
            for position in range(len(treatment_pileup)):
                signal = treatment_pileup[position]
                lambda_bg = calculate_local_lambda(position)
                
                # Poisson p-value
                from scipy.stats import poisson
                p_value = 1 - poisson.cdf(signal, lambda_bg)
                
                if p_value < 1e-5:
                    peaks.append({
                        'position': position,
                        'signal': signal,
                        'background': lambda_bg,
                        'p_value': p_value
                    })
            
            # Merge nearby peaks
            merged_peaks = merge_peaks(peaks, distance=fragment_size)
            
            return merged_peaks
    
    def differential_binding_analysis(peaks_condition1, peaks_condition2):
        """
        Find differential histone marks between conditions
        """
        
        # Create consensus peak set
        all_peaks = peaks_condition1 + peaks_condition2
        consensus_peaks = merge_overlapping_peaks(all_peaks)
        
        # Count reads in each peak
        counts_matrix = []
        
        for peak in consensus_peaks:
            counts1 = count_reads_in_peak(peak, condition1_bams)
            counts2 = count_reads_in_peak(peak, condition2_bams)
            counts_matrix.append(counts1 + counts2)
        
        counts_matrix = np.array(counts_matrix)
        
        # DESeq2-style analysis
        from scipy.stats import nbinom
        
        # Size factor normalization
        size_factors = estimate_size_factors(counts_matrix)
        normalized_counts = counts_matrix / size_factors
        
        # Negative binomial test
        differential_peaks = []
        
        for i, peak in enumerate(consensus_peaks):
            mean1 = normalized_counts[i, :len(condition1_bams)].mean()
            mean2 = normalized_counts[i, len(condition1_bams):].mean()
            
            # Estimate dispersion
            var1 = normalized_counts[i, :len(condition1_bams)].var()
            var2 = normalized_counts[i, len(condition1_bams):].var()
            
            # Test for difference
            # ... complex NB test ...
            
            fold_change = mean2 / (mean1 + 1)
            
            differential_peaks.append({
                'peak': peak,
                'fold_change': fold_change,
                'p_value': p_value
            })
        
        return differential_peaks
    
    def chromatin_state_segmentation(histone_marks, method='chromhmm'):
        """
        Segment genome into chromatin states
        """
        
        if method == 'chromhmm':
            # Hidden Markov Model for chromatin states
            
            n_states = 15  # Typical number
            n_marks = len(histone_marks)
            
            # Binarize signals
            binarized = []
            for mark in histone_marks:
                threshold = np.percentile(mark, 90)
                binarized.append(mark > threshold)
            
            binarized = np.array(binarized).T
            
            # Train HMM
            from hmmlearn import hmm
            
            model = hmm.GaussianHMM(n_components=n_states)
            model.fit(binarized)
            
            # Predict states
            states = model.predict(binarized)
            
            # Characterize states by mark enrichment
            state_profiles = []
            for state in range(n_states):
                state_mask = states == state
                profile = binarized[state_mask].mean(axis=0)
                state_profiles.append(profile)
            
            # Assign biological labels
            state_labels = assign_chromatin_labels(state_profiles)
            
            return states, state_labels
```

### Chromatin Accessibility (ATAC-seq/DNase-seq)

```python
def chromatin_accessibility_analysis():
    """
    Analyze open chromatin regions
    """
    
    def call_accessible_regions(atac_bam):
        """
        Identify open chromatin from ATAC-seq
        """
        
        # ATAC-seq specific: account for Tn5 insertion
        tn5_offset = 9  # Tn5 creates 9bp duplication
        
        # Shift reads to account for Tn5 binding
        def shift_reads(bam):
            shifted_reads = []
            
            for read in bam:
                if read.is_reverse:
                    # Negative strand: shift 3' end
                    read.pos -= tn5_offset
                else:
                    # Positive strand: shift 5' end  
                    read.pos += tn5_offset
                
                shifted_reads.append(read)
            
            return shifted_reads
        
        # Call peaks on shifted reads
        peaks = call_peaks_for_accessibility(shifted_reads)
        
        return peaks
    
    def nucleosome_positioning(atac_fragment_sizes):
        """
        Infer nucleosome positions from fragment size distribution
        """
        
        # Fragment size categories
        nucleosome_free = []  # <100bp
        mononucleosome = []   # 180-247bp  
        dinucleosome = []     # 315-473bp
        
        for fragment in atac_fragment_sizes:
            size = abs(fragment['end'] - fragment['start'])
            
            if size < 100:
                nucleosome_free.append(fragment)
            elif 180 <= size <= 247:
                mononucleosome.append(fragment)
            elif 315 <= size <= 473:
                dinucleosome.append(fragment)
        
        # Build signal tracks
        nfr_signal = build_signal_track(nucleosome_free)
        mono_signal = build_signal_track(mononucleosome)
        
        # Nucleosome positions: peaks in mono, valleys in NFR
        nucleosome_positions = find_peaks(mono_signal)
        
        # Phasing: regular spacing of nucleosomes
        def detect_phasing(positions):
            distances = np.diff(positions)
            
            # Expected ~200bp spacing
            phased = np.abs(distances - 200) < 20
            phasing_score = phased.sum() / len(distances)
            
            return phasing_score
        
        return {
            'positions': nucleosome_positions,
            'phasing': detect_phasing(nucleosome_positions)
        }
    
    def footprinting_analysis(atac_data, motifs):
        """
        Identify TF binding from accessibility patterns
        """
        
        footprints = []
        
        for motif in motifs:
            # Find motif occurrences
            motif_sites = scan_genome_for_motif(motif)
            
            # Aggregate ATAC signal around motifs
            aggregate_signal = []
            
            for site in motif_sites:
                # Window around motif
                window = atac_data[site['pos']-50:site['pos']+50]
                aggregate_signal.append(window)
            
            aggregate_signal = np.mean(aggregate_signal, axis=0)
            
            # Footprint: depletion at motif, peaks on flanks
            motif_center = len(aggregate_signal) // 2
            motif_width = len(motif['sequence'])
            
            center_signal = aggregate_signal[
                motif_center-motif_width//2:motif_center+motif_width//2
            ].mean()
            
            flank_signal = np.concatenate([
                aggregate_signal[:motif_center-motif_width//2],
                aggregate_signal[motif_center+motif_width//2:]
            ]).mean()
            
            footprint_depth = (flank_signal - center_signal) / flank_signal
            
            footprints.append({
                'motif': motif['name'],
                'depth': footprint_depth,
                'profile': aggregate_signal
            })
        
        return footprints
```

### 3D Genome Organization (Hi-C/ChIA-PET)

```python
def three_d_genome_analysis():
    """
    Analyze chromatin interactions and 3D structure
    """
    
    def process_hic_matrix(raw_matrix, resolution=10000):
        """
        Process Hi-C contact matrix
        """
        
        # Remove unmappable regions
        mappable = raw_matrix.sum(axis=0) > 0
        filtered_matrix = raw_matrix[mappable][:, mappable]
        
        # ICE normalization (Iterative Correction and Eigenvector decomposition)
        def ice_normalization(matrix, max_iter=100, eps=1e-4):
            """
            Balance Hi-C matrix
            """
            n = matrix.shape[0]
            bias = np.ones(n)
            
            for iteration in range(max_iter):
                # Calculate column sums
                col_sums = matrix.sum(axis=0) * bias
                
                # Avoid division by zero
                col_sums[col_sums == 0] = 1
                
                # Update bias
                new_bias = bias / col_sums
                new_bias[np.isnan(new_bias)] = 1
                
                # Check convergence
                if np.abs(new_bias - bias).max() < eps:
                    break
                
                bias = new_bias
            
            # Apply bias
            bias_matrix = np.outer(bias, bias)
            normalized = matrix * bias_matrix
            
            return normalized, bias
        
        normalized_matrix, bias_vector = ice_normalization(filtered_matrix)
        
        # Distance normalization (expected counts by genomic distance)
        def distance_normalize(matrix):
            n = matrix.shape[0]
            expected = np.zeros(n)
            
            # Calculate expected counts for each distance
            for d in range(n):
                diagonal = np.diagonal(matrix, offset=d)
                expected[d] = np.median(diagonal[diagonal > 0])
            
            # Normalize by expected
            normalized = np.zeros_like(matrix)
            for i in range(n):
                for j in range(n):
                    dist = abs(i - j)
                    if expected[dist] > 0:
                        normalized[i, j] = matrix[i, j] / expected[dist]
            
            return normalized
        
        distance_normalized = distance_normalize(normalized_matrix)
        
        return distance_normalized
    
    def detect_tads(hic_matrix, method='insulation'):
        """
        Topologically Associating Domains
        """
        
        if method == 'insulation':
            # Insulation score method
            window_size = 10  # In bins
            
            insulation_scores = []
            
            for i in range(window_size, len(hic_matrix) - window_size):
                # Square around diagonal
                upstream = hic_matrix[i-window_size:i, i-window_size:i]
                downstream = hic_matrix[i:i+window_size, i:i+window_size]
                cross = hic_matrix[i-window_size:i, i:i+window_size]
                
                # Insulation score
                within = (upstream.mean() + downstream.mean()) / 2
                between = cross.mean()
                
                score = np.log2((within + 1) / (between + 1))
                insulation_scores.append(score)
            
            # Find boundaries (local minima)
            from scipy.signal import find_peaks
            
            boundaries = find_peaks(-np.array(insulation_scores))[0]
            
            # TADs are regions between boundaries
            tads = []
            for i in range(len(boundaries) - 1):
                tads.append({
                    'start': boundaries[i],
                    'end': boundaries[i+1],
                    'insulation': insulation_scores[boundaries[i]:boundaries[i+1]]
                })
            
            return tads
        
        elif method == 'directionality':
            # Directionality index
            # ... implementation ...
            pass
    
    def detect_loops(hic_matrix, expected_matrix):
        """
        Find chromatin loops
        """
        
        # Calculate observed/expected
        oe_matrix = hic_matrix / (expected_matrix + 1e-10)
        
        # Find local maxima (peaks)
        from scipy.ndimage import maximum_filter
        
        # Peak calling in 2D
        neighborhood_size = 5
        local_max = maximum_filter(oe_matrix, size=neighborhood_size)
        peaks = (oe_matrix == local_max) & (oe_matrix > 2)  # 2-fold enrichment
        
        # Extract loop anchors
        loops = []
        peak_coords = np.where(peaks)
        
        for i, j in zip(peak_coords[0], peak_coords[1]):
            if j > i:  # Upper triangle only
                loops.append({
                    'anchor1': i,
                    'anchor2': j,
                    'strength': oe_matrix[i, j],
                    'distance': abs(i - j)
                })
        
        return loops
    
    def compartment_analysis(hic_matrix):
        """
        A/B compartment identification
        """
        
        # Correlation matrix
        correlation_matrix = np.corrcoef(hic_matrix)
        
        # First eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
        first_eigenvector = eigenvectors[:, 0]
        
        # Sign of eigenvector indicates compartment
        # Correlate with gene density to assign A/B
        
        compartments = np.sign(first_eigenvector)
        
        # Smooth compartment calls
        from scipy.ndimage import median_filter
        compartments_smooth = median_filter(compartments, size=5)
        
        return compartments_smooth
```

### Multi-Omics Integration

```python
def integrate_epigenomic_data():
    """
    Combine multiple epigenetic data types
    """
    
    def create_regulatory_potential_score(
        methylation, 
        accessibility, 
        h3k27ac,  # Active enhancer
        h3k4me3,  # Active promoter
        expression
    ):
        """
        Integrate signals to predict regulatory activity
        """
        
        # Weight different marks
        weights = {
            'methylation': -0.5,  # Negative correlation
            'accessibility': 1.0,
            'h3k27ac': 1.5,
            'h3k4me3': 2.0
        }
        
        # Combine signals
        regulatory_score = (
            weights['methylation'] * (1 - methylation) +  # Unmethylated
            weights['accessibility'] * accessibility +
            weights['h3k27ac'] * h3k27ac +
            weights['h3k4me3'] * h3k4me3
        )
        
        # Validate with expression
        correlation = np.corrcoef(regulatory_score, expression)[0, 1]
        
        return regulatory_score, correlation
    
    def enhancer_gene_linking(enhancers, genes, hic_data, correlation_data):
        """
        Link enhancers to target genes
        """
        
        links = []
        
        for enhancer in enhancers:
            # Physical proximity from Hi-C
            nearby_genes = []
            
            for gene in genes:
                # Check if in same TAD
                if same_tad(enhancer, gene, tads):
                    # Get Hi-C contact frequency
                    contact = hic_data[enhancer['bin'], gene['bin']]
                    
                    if contact > threshold:
                        nearby_genes.append(gene)
            
            # Activity correlation
            for gene in nearby_genes:
                # Correlate enhancer activity with gene expression
                enhancer_activity = h3k27ac_signal[enhancer['pos']]
                gene_expression = rna_seq[gene['id']]
                
                correlation = np.corrcoef(enhancer_activity, gene_expression)[0, 1]
                
                if correlation > 0.3:
                    links.append({
                        'enhancer': enhancer,
                        'gene': gene,
                        'distance': abs(enhancer['pos'] - gene['tss']),
                        'contact_frequency': contact,
                        'correlation': correlation
                    })
        
        return links
```

### Machine Learning for Epigenomics

```python
def epigenetic_prediction_models():
    """
    ML models for epigenetic data
    """
    
    def predict_enhancers(sequence, epigenetic_marks):
        """
        Deep learning for enhancer prediction
        """
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        
        # Sequence branch
        seq_input = layers.Input(shape=(1000, 4))  # One-hot encoded
        conv1 = layers.Conv1D(64, 8, activation='relu')(seq_input)
        pool1 = layers.MaxPooling1D(2)(conv1)
        conv2 = layers.Conv1D(128, 8, activation='relu')(pool1)
        pool2 = layers.MaxPooling1D(2)(conv2)
        seq_features = layers.Flatten()(pool2)
        
        # Epigenetic branch
        epi_input = layers.Input(shape=(len(epigenetic_marks),))
        epi_dense = layers.Dense(64, activation='relu')(epi_input)
        
        # Combine
        combined = layers.concatenate([seq_features, epi_dense])
        dense1 = layers.Dense(128, activation='relu')(combined)
        dropout = layers.Dropout(0.5)(dense1)
        output = layers.Dense(1, activation='sigmoid')(dropout)
        
        model = Model(inputs=[seq_input, epi_input], outputs=output)
        
        return model
    
    def impute_missing_marks(observed_marks, mark_to_impute):
        """
        Impute missing epigenetic marks from others
        """
        
        # Random Forest regression
        from sklearn.ensemble import RandomForestRegressor
        
        # Train on regions with complete data
        complete_mask = ~np.isnan(observed_marks).any(axis=1)
        
        X_train = observed_marks[complete_mask]
        y_train = mark_to_impute[complete_mask]
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Impute missing regions
        missing_mask = np.isnan(mark_to_impute)
        X_missing = observed_marks[missing_mask]
        
        imputed = model.predict(X_missing)
        
        return imputed
```

### Common Pitfalls and Solutions

| Pitfall | Consequence | Solution |
|---------|------------|----------|
| **Batch effects in methylation** | False positives | ComBat, functional normalization |
| **PCR duplicates in ChIP-seq** | Overestimated peaks | Remove duplicates |
| **Mappability bias** | Missing repetitive regions | Use longer reads, better aligners |
| **Cell type heterogeneity** | Averaged signals | Deconvolution or single-cell |
| **Antibody specificity** | Wrong targets | Validate with knockouts |
| **Copy number effects** | Confounded signals | Normalize by input |

### References
- Laird (2010). Principles and challenges of genome-wide DNA methylation analysis
- Park (2009). ChIP-seq: advantages and challenges of a maturing technology
- Buenrostro et al. (2013). Transposition of native chromatin for fast and sensitive epigenomic profiling
- Lieberman-Aiden et al. (2009). Comprehensive mapping of long-range interactions