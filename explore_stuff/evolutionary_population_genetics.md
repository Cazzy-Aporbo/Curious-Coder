# Evolutionary and Population Genetics
## Mathematical Frameworks for Understanding Genetic Variation and Change

### Intent
Evolution is population genetics over time. This document provides rigorous mathematical frameworks for analyzing genetic variation, selection, drift, and adaptation, bridging molecular data with evolutionary theory and population dynamics.

### The Fundamental Equation: Hardy-Weinberg and Its Violations

**Hardy-Weinberg Equilibrium:**
```
p² + 2pq + q² = 1

where:
- p = frequency of allele A
- q = frequency of allele a
- p² = frequency of AA genotype
- 2pq = frequency of Aa genotype
- q² = frequency of aa genotype

Assumptions (all violated in reality):
1. No mutation
2. No selection
3. No migration
4. Random mating
5. Infinite population size
```

### Wright-Fisher Model: The Foundation

```python
def wright_fisher_simulation(N, p0, generations, selection=0, mutation_rate=0):
    """
    Classic Wright-Fisher model with selection and mutation
    
    N: Population size (diploid individuals)
    p0: Initial allele frequency
    """
    
    p = p0
    trajectory = [p]
    
    for gen in range(generations):
        # Selection
        if selection != 0:
            # Relative fitnesses: AA=1+s, Aa=1+hs, aa=1
            w_AA = 1 + selection
            w_Aa = 1 + selection * 0.5  # h=0.5 (codominance)
            w_aa = 1
            
            # Mean fitness
            w_bar = p**2 * w_AA + 2*p*(1-p) * w_Aa + (1-p)**2 * w_aa
            
            # Frequency after selection
            p_prime = (p**2 * w_AA + p*(1-p) * w_Aa) / w_bar
        else:
            p_prime = p
        
        # Mutation
        if mutation_rate > 0:
            # A → a at rate μ, a → A at rate μ
            p_prime = p_prime * (1 - mutation_rate) + (1 - p_prime) * mutation_rate
        
        # Genetic drift (binomial sampling)
        n_alleles = np.random.binomial(2*N, p_prime)
        p = n_alleles / (2*N)
        
        trajectory.append(p)
        
        # Fixation or loss
        if p == 0 or p == 1:
            break
    
    return np.array(trajectory)

def fixation_probability(N, p0, s):
    """
    Kimura's fixation probability formula
    
    P(fix) = (1 - e^(-4Nsp0)) / (1 - e^(-4Ns))
    
    For neutral: P(fix) = p0
    For new mutation: P(fix) ≈ 2s (if s > 0 and Ns >> 1)
    """
    if s == 0:
        return p0
    
    numerator = 1 - np.exp(-4*N*s*p0)
    denominator = 1 - np.exp(-4*N*s)
    
    return numerator / denominator
```

### Coalescent Theory: Looking Backward in Time

```python
class Coalescent:
    """
    Simulate genealogies under the coalescent
    """
    
    def __init__(self, n_samples, Ne):
        """
        n_samples: Number of sampled lineages
        Ne: Effective population size
        """
        self.n = n_samples
        self.Ne = Ne
        
    def simulate_tree(self):
        """
        Generate coalescent tree
        """
        # Initialize lineages
        lineages = list(range(self.n))
        tree = {}
        time = 0
        node_id = self.n
        
        while len(lineages) > 1:
            k = len(lineages)
            
            # Time to next coalescence (exponential distribution)
            rate = k * (k - 1) / (4 * self.Ne)
            t_coal = np.random.exponential(1 / rate)
            time += t_coal
            
            # Choose two lineages to coalesce
            i, j = np.random.choice(k, 2, replace=False)
            coal_lineages = [lineages[i], lineages[j]]
            
            # Create new node
            tree[node_id] = {
                'children': coal_lineages,
                'time': time
            }
            
            # Update lineages
            lineages = [l for idx, l in enumerate(lineages) if idx not in [i, j]]
            lineages.append(node_id)
            node_id += 1
        
        return tree
    
    def expected_pairwise_differences(self, mutation_rate):
        """
        Watterson's theta and nucleotide diversity
        """
        # Watterson's estimator
        n = self.n
        a_n = sum(1/i for i in range(1, n))
        theta_w = 4 * self.Ne * mutation_rate
        S_expected = theta_w * a_n  # Expected segregating sites
        
        # Nucleotide diversity (π)
        pi = theta_w  # For infinite sites model
        
        # Tajima's D expectation (0 under neutrality)
        
        return {
            'theta_w': theta_w,
            'S_expected': S_expected,
            'pi': pi
        }
```

### Selection Detection Methods

#### 1. Tajima's D

```python
def tajimas_d(sequences):
    """
    Test for departures from neutrality
    
    D = (π - θ_w) / sqrt(Var(π - θ_w))
    
    D < 0: Excess rare variants (positive selection or expansion)
    D > 0: Excess intermediate variants (balancing selection or contraction)
    """
    
    n = len(sequences)
    
    # Count segregating sites
    S = 0
    for site in range(len(sequences[0])):
        alleles = set(seq[site] for seq in sequences)
        if len(alleles) > 1:
            S += 1
    
    if S == 0:
        return 0  # No variation
    
    # Watterson's theta
    a1 = sum(1/i for i in range(1, n))
    theta_w = S / a1
    
    # Pairwise differences (π)
    total_diff = 0
    n_pairs = 0
    
    for i in range(n):
        for j in range(i+1, n):
            diff = sum(sequences[i][k] != sequences[j][k] 
                      for k in range(len(sequences[0])))
            total_diff += diff
            n_pairs += 1
    
    pi = total_diff / n_pairs
    
    # Variance calculation (complex formula)
    a2 = sum(1/i**2 for i in range(1, n))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    
    c1 = b1 - 1/a1
    c2 = b2 - (n + 2)/(a1 * n) + a2/a1**2
    
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)
    
    var_d = e1 * S + e2 * S * (S - 1)
    
    # Tajima's D
    D = (pi - theta_w) / np.sqrt(var_d)
    
    return D
```

#### 2. McDonald-Kreitman Test

```python
def mcdonald_kreitman_test(ingroup_seqs, outgroup_seqs):
    """
    Test for adaptive evolution using divergence and polymorphism
    
    Compares ratio of non-synonymous to synonymous changes
    within species (polymorphism) vs between species (divergence)
    """
    
    def classify_mutations(seq1, seq2, genetic_code):
        """
        Classify as synonymous or non-synonymous
        """
        syn = 0
        nonsyn = 0
        
        for i in range(0, len(seq1), 3):  # Codons
            codon1 = seq1[i:i+3]
            codon2 = seq2[i:i+3]
            
            if codon1 != codon2:
                aa1 = genetic_code.get(codon1, 'X')
                aa2 = genetic_code.get(codon2, 'X')
                
                if aa1 == aa2:
                    syn += 1
                else:
                    nonsyn += 1
        
        return syn, nonsyn
    
    # Count polymorphisms within ingroup
    Ps = 0  # Synonymous polymorphisms
    Pn = 0  # Non-synonymous polymorphisms
    
    for i in range(len(ingroup_seqs)):
        for j in range(i+1, len(ingroup_seqs)):
            s, n = classify_mutations(ingroup_seqs[i], ingroup_seqs[j], genetic_code)
            Ps += s
            Pn += n
    
    # Count fixed differences (divergence)
    Ds = 0  # Synonymous divergence
    Dn = 0  # Non-synonymous divergence
    
    for in_seq in ingroup_seqs:
        for out_seq in outgroup_seqs:
            s, n = classify_mutations(in_seq, out_seq, genetic_code)
            Ds += s
            Dn += n
    
    # Normalize by number of comparisons
    Ps /= len(ingroup_seqs) * (len(ingroup_seqs) - 1) / 2
    Pn /= len(ingroup_seqs) * (len(ingroup_seqs) - 1) / 2
    Ds /= len(ingroup_seqs) * len(outgroup_seqs)
    Dn /= len(ingroup_seqs) * len(outgroup_seqs)
    
    # 2x2 contingency table
    #       Polymorphic  Fixed
    # Syn      Ps         Ds
    # Nonsyn   Pn         Dn
    
    # Neutrality index
    NI = (Pn/Ps) / (Dn/Ds) if Ps > 0 and Ds > 0 else np.nan
    
    # Alpha (proportion of adaptive substitutions)
    alpha = 1 - (Ps * Dn) / (Pn * Ds) if Pn > 0 and Ds > 0 else np.nan
    
    # Fisher's exact test
    from scipy.stats import fisher_exact
    
    contingency = [[Ps, Ds], [Pn, Dn]]
    odds_ratio, p_value = fisher_exact(contingency)
    
    return {
        'NI': NI,
        'alpha': alpha,
        'p_value': p_value,
        'contingency_table': contingency
    }
```

#### 3. Extended Haplotype Homozygosity (EHH)

```python
def calculate_ehh(haplotypes, core_snp_position, genetic_map):
    """
    Detect recent positive selection via extended haplotypes
    """
    
    # Identify core alleles
    core_alleles = [hap[core_snp_position] for hap in haplotypes]
    allele_types = list(set(core_alleles))
    
    ehh_results = {}
    
    for allele in allele_types:
        # Get haplotypes with this core allele
        core_haplotypes = [hap for hap, a in zip(haplotypes, core_alleles) if a == allele]
        
        # Calculate EHH at increasing distances
        ehh_decay = []
        
        for distance in range(0, 500000, 10000):  # Up to 500kb
            # Find SNPs at this distance
            snps_at_distance = [i for i, pos in enumerate(genetic_map) 
                               if abs(pos - genetic_map[core_snp_position]) == distance]
            
            if not snps_at_distance:
                continue
            
            # Calculate haplotype homozygosity
            n = len(core_haplotypes)
            homozygosity = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    # Check if haplotypes are identical at this distance
                    identical = all(core_haplotypes[i][snp] == core_haplotypes[j][snp] 
                                  for snp in snps_at_distance)
                    if identical:
                        homozygosity += 2  # Count both (i,j) and (j,i)
            
            ehh = homozygosity / (n * (n - 1))
            ehh_decay.append((distance, ehh))
        
        # Integrated EHH (iHH)
        ihh = np.trapz([e for d, e in ehh_decay], [d for d, e in ehh_decay])
        
        ehh_results[allele] = {
            'ehh_decay': ehh_decay,
            'ihh': ihh
        }
    
    # Calculate iHS (integrated haplotype score)
    if len(allele_types) == 2:
        ihh_ancestral = ehh_results[allele_types[0]]['ihh']
        ihh_derived = ehh_results[allele_types[1]]['ihh']
        
        ihs = np.log(ihh_ancestral / ihh_derived)
        
        # Standardize
        # In practice, would standardize across genome
        ihs_standardized = ihs  # Placeholder
        
        return ihs_standardized
    
    return ehh_results
```

### Population Structure Analysis

#### 1. F-statistics

```python
def calculate_fst(populations):
    """
    Wright's fixation index - population differentiation
    
    F_ST = (H_T - H_S) / H_T
    
    where:
    H_T = expected heterozygosity in total population
    H_S = expected heterozygosity in subpopulations
    """
    
    all_alleles = []
    subpop_heterozygosities = []
    
    for pop in populations:
        # Allele frequencies in this population
        n = len(pop)
        allele_counts = {}
        
        for individual in pop:
            for allele in individual:
                allele_counts[allele] = allele_counts.get(allele, 0) + 1
        
        # Convert to frequencies
        total_alleles = sum(allele_counts.values())
        allele_freqs = {a: c/total_alleles for a, c in allele_counts.items()}
        
        # Expected heterozygosity (H_S)
        h_s = 1 - sum(f**2 for f in allele_freqs.values())
        subpop_heterozygosities.append(h_s)
        
        # Add to total population
        all_alleles.extend([allele for ind in pop for allele in ind])
    
    # Total population heterozygosity (H_T)
    total_allele_counts = {}
    for allele in all_alleles:
        total_allele_counts[allele] = total_allele_counts.get(allele, 0) + 1
    
    total_freqs = {a: c/len(all_alleles) for a, c in total_allele_counts.items()}
    h_t = 1 - sum(f**2 for f in total_freqs.values())
    
    # Average H_S
    h_s_avg = np.mean(subpop_heterozygosities)
    
    # F_ST
    f_st = (h_t - h_s_avg) / h_t if h_t > 0 else 0
    
    return f_st

def calculate_f_statistics(genotypes, populations):
    """
    Complete F-statistics: F_IS, F_ST, F_IT
    
    F_IS: Inbreeding within subpopulations
    F_ST: Differentiation among subpopulations
    F_IT: Total inbreeding
    
    (1 - F_IT) = (1 - F_IS)(1 - F_ST)
    """
    
    # Observed and expected heterozygosities
    H_I = []  # Individual (observed)
    H_S = []  # Subpopulation (expected)
    H_T = 0   # Total (expected)
    
    # Calculate for each locus
    for locus in range(genotypes.shape[1]):
        # ... complex calculation ...
        pass
    
    F_IS = (H_S - H_I) / H_S
    F_ST = (H_T - H_S) / H_T
    F_IT = (H_T - H_I) / H_T
    
    return F_IS, F_ST, F_IT
```

#### 2. STRUCTURE Algorithm

```python
class STRUCTUREModel:
    """
    Bayesian clustering of populations
    """
    
    def __init__(self, K, alpha=1.0):
        """
        K: Number of populations
        alpha: Dirichlet parameter for admixture
        """
        self.K = K
        self.alpha = alpha
        
    def fit(self, genotypes, n_iterations=10000, burnin=1000):
        """
        MCMC to infer population structure
        """
        n_individuals, n_loci = genotypes.shape
        
        # Initialize
        # Q: admixture proportions (individuals × populations)
        Q = np.random.dirichlet([self.alpha]*self.K, n_individuals)
        
        # P: allele frequencies (populations × loci)
        P = np.random.beta(1, 1, (self.K, n_loci))
        
        # Z: population assignments for each allele copy
        Z = np.zeros((n_individuals, n_loci, 2), dtype=int)
        
        log_likelihoods = []
        
        for iteration in range(n_iterations):
            # Update Z (population assignments)
            for i in range(n_individuals):
                for j in range(n_loci):
                    for allele_copy in range(2):
                        # Probability of assignment to each population
                        allele = genotypes[i, j, allele_copy]
                        
                        probs = Q[i] * (P[:, j] if allele == 1 else 1 - P[:, j])
                        probs /= probs.sum()
                        
                        # Sample assignment
                        Z[i, j, allele_copy] = np.random.choice(self.K, p=probs)
            
            # Update Q (admixture proportions)
            for i in range(n_individuals):
                # Count assignments to each population
                counts = np.zeros(self.K)
                for k in range(self.K):
                    counts[k] = np.sum(Z[i] == k)
                
                # Sample from Dirichlet posterior
                Q[i] = np.random.dirichlet(counts + self.alpha)
            
            # Update P (allele frequencies)
            for k in range(self.K):
                for j in range(n_loci):
                    # Count alleles assigned to population k
                    assigned = Z[:, j, :] == k
                    alleles = genotypes[:, j, :][assigned]
                    
                    n1 = np.sum(alleles == 1)
                    n0 = np.sum(alleles == 0)
                    
                    # Sample from Beta posterior
                    P[k, j] = np.random.beta(n1 + 1, n0 + 1)
            
            # Calculate log likelihood
            if iteration >= burnin:
                log_lik = self.calculate_log_likelihood(genotypes, Q, P)
                log_likelihoods.append(log_lik)
        
        return Q, P, log_likelihoods
```

### Phylogenetic Reconstruction

```python
def build_phylogenetic_tree(sequences, method='neighbor_joining'):
    """
    Reconstruct evolutionary relationships
    """
    
    if method == 'neighbor_joining':
        # Calculate distance matrix
        n = len(sequences)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Jukes-Cantor distance
                p = sum(sequences[i][k] != sequences[j][k] 
                       for k in range(len(sequences[0]))) / len(sequences[0])
                
                if p < 0.75:  # Avoid log of negative
                    d = -3/4 * np.log(1 - 4*p/3)
                else:
                    d = np.inf
                
                dist_matrix[i, j] = dist_matrix[j, i] = d
        
        # Neighbor-joining algorithm
        tree = {}
        active_nodes = list(range(n))
        next_node = n
        
        while len(active_nodes) > 2:
            # Calculate Q matrix
            m = len(active_nodes)
            Q = np.zeros((m, m))
            
            for i in range(m):
                for j in range(i+1, m):
                    Q[i, j] = (m - 2) * dist_matrix[active_nodes[i], active_nodes[j]]
                    Q[i, j] -= sum(dist_matrix[active_nodes[i], active_nodes[k]] 
                                  for k in range(m))
                    Q[i, j] -= sum(dist_matrix[active_nodes[j], active_nodes[k]] 
                                  for k in range(m))
                    Q[j, i] = Q[i, j]
            
            # Find minimum Q
            i, j = np.unravel_index(np.argmin(Q), Q.shape)
            node_i = active_nodes[i]
            node_j = active_nodes[j]
            
            # Calculate branch lengths
            delta = sum(dist_matrix[node_i, active_nodes[k]] - 
                       dist_matrix[node_j, active_nodes[k]] 
                       for k in range(m)) / (m - 2)
            
            length_i = 0.5 * dist_matrix[node_i, node_j] + 0.5 * delta
            length_j = dist_matrix[node_i, node_j] - length_i
            
            # Create new internal node
            tree[next_node] = {
                'children': [(node_i, length_i), (node_j, length_j)]
            }
            
            # Update distance matrix
            for k in active_nodes:
                if k not in [node_i, node_j]:
                    new_dist = 0.5 * (dist_matrix[node_i, k] + 
                                     dist_matrix[node_j, k] - 
                                     dist_matrix[node_i, node_j])
                    
                    # Extend matrix if needed
                    if next_node >= len(dist_matrix):
                        new_size = next_node + 1
                        new_matrix = np.zeros((new_size, new_size))
                        new_matrix[:len(dist_matrix), :len(dist_matrix)] = dist_matrix
                        dist_matrix = new_matrix
                    
                    dist_matrix[next_node, k] = dist_matrix[k, next_node] = new_dist
            
            # Update active nodes
            active_nodes.remove(node_i)
            active_nodes.remove(node_j)
            active_nodes.append(next_node)
            next_node += 1
        
        # Connect last two nodes
        if len(active_nodes) == 2:
            tree[next_node] = {
                'children': [(active_nodes[0], dist_matrix[active_nodes[0], active_nodes[1]]/2),
                            (active_nodes[1], dist_matrix[active_nodes[0], active_nodes[1]]/2)]
            }
        
        return tree
    
    elif method == 'maximum_likelihood':
        # Felsenstein's pruning algorithm
        pass
```

### Molecular Evolution Models

```python
def jukes_cantor_model(t, mu):
    """
    Simplest substitution model (equal rates)
    
    P(i→j|t) = 1/4 - 1/4 * exp(-4μt)  for i≠j
    P(i→i|t) = 1/4 + 3/4 * exp(-4μt)
    """
    
    P = np.zeros((4, 4))  # A, C, G, T
    
    off_diagonal = 0.25 - 0.25 * np.exp(-4 * mu * t)
    diagonal = 0.25 + 0.75 * np.exp(-4 * mu * t)
    
    for i in range(4):
        for j in range(4):
            P[i, j] = diagonal if i == j else off_diagonal
    
    return P

def gtr_model(t, rates, frequencies):
    """
    General Time Reversible model
    Most general model for nucleotide substitution
    """
    
    # Rate matrix Q
    Q = np.zeros((4, 4))
    
    # Fill rate matrix
    rate_params = ['AC', 'AG', 'AT', 'CG', 'CT', 'GT']
    # ... complex implementation ...
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    
    # P(t) = exp(Qt)
    P_t = eigenvectors @ np.diag(np.exp(eigenvalues * t)) @ np.linalg.inv(eigenvectors)
    
    return P_t
```

### Linkage Disequilibrium and Recombination

```python
def calculate_ld(genotypes):
    """
    Calculate linkage disequilibrium statistics
    """
    
    n_samples, n_snps = genotypes.shape
    
    # Pairwise LD
    ld_matrix = np.zeros((n_snps, n_snps))
    
    for i in range(n_snps):
        for j in range(i+1, n_snps):
            # Allele frequencies
            p_A = genotypes[:, i].mean()
            p_B = genotypes[:, j].mean()
            
            # Haplotype frequency
            p_AB = ((genotypes[:, i] == 1) & (genotypes[:, j] == 1)).mean()
            
            # D = p_AB - p_A * p_B
            D = p_AB - p_A * p_B
            
            # D' (normalized D)
            if D >= 0:
                D_max = min(p_A * (1 - p_B), (1 - p_A) * p_B)
            else:
                D_max = min(p_A * p_B, (1 - p_A) * (1 - p_B))
            
            D_prime = D / D_max if D_max > 0 else 0
            
            # r² (correlation coefficient)
            denominator = p_A * (1 - p_A) * p_B * (1 - p_B)
            r_squared = D**2 / denominator if denominator > 0 else 0
            
            ld_matrix[i, j] = r_squared
            ld_matrix[j, i] = r_squared
    
    return ld_matrix

def estimate_recombination_rate(genotypes, positions, method='likelihood'):
    """
    Estimate recombination rates between markers
    """
    
    if method == 'likelihood':
        # Hudson's two-locus sampling formula
        # Complex implementation involving coalescent with recombination
        pass
    
    elif method == 'ldhat':
        # McVean's composite likelihood approach
        
        def composite_likelihood(rho, genotypes):
            """
            Composite likelihood over all pairs
            """
            log_lik = 0
            
            for i in range(len(genotypes[0]) - 1):
                # Two-locus likelihood
                two_locus_data = genotypes[:, [i, i+1]]
                
                # Count haplotype configurations
                n00 = ((two_locus_data[:, 0] == 0) & (two_locus_data[:, 1] == 0)).sum()
                n01 = ((two_locus_data[:, 0] == 0) & (two_locus_data[:, 1] == 1)).sum()
                n10 = ((two_locus_data[:, 0] == 1) & (two_locus_data[:, 1] == 0)).sum()
                n11 = ((two_locus_data[:, 0] == 1) & (two_locus_data[:, 1] == 1)).sum()
                
                # Likelihood given recombination rate
                # Uses pre-computed likelihood lookup tables
                # ... complex implementation ...
                
            return log_lik
```

### Genome-Wide Association Studies (GWAS)

```python
def gwas_analysis(genotypes, phenotypes, covariates=None):
    """
    Test association between genetic variants and traits
    """
    
    n_samples, n_snps = genotypes.shape
    results = []
    
    for snp_idx in range(n_snps):
        genotype = genotypes[:, snp_idx]
        
        # Basic linear model
        if covariates is not None:
            # Include covariates (population structure, etc.)
            X = np.column_stack([genotype, covariates])
        else:
            X = genotype.reshape(-1, 1)
        
        X = sm.add_constant(X)
        
        # Fit model
        if phenotypes.dtype == bool:
            # Binary trait - logistic regression
            model = sm.Logit(phenotypes, X)
        else:
            # Quantitative trait - linear regression
            model = sm.OLS(phenotypes, X)
        
        try:
            result = model.fit(disp=0)
            
            # Extract statistics
            beta = result.params[1]  # Effect size
            se = result.bse[1]       # Standard error
            p_value = result.pvalues[1]
            
            results.append({
                'snp': snp_idx,
                'beta': beta,
                'se': se,
                'p_value': p_value
            })
        except:
            results.append({
                'snp': snp_idx,
                'beta': np.nan,
                'se': np.nan,
                'p_value': 1.0
            })
    
    # Multiple testing correction
    p_values = [r['p_value'] for r in results]
    
    # Bonferroni
    bonferroni_threshold = 0.05 / n_snps
    
    # FDR (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    rejected, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
    
    for i, result in enumerate(results):
        result['p_adjusted'] = p_adjusted[i]
        result['significant'] = rejected[i]
    
    return results

def genomic_control(gwas_results):
    """
    Adjust for population stratification using genomic control
    """
    
    # Calculate inflation factor
    chi2_stats = [(r['beta'] / r['se'])**2 for r in gwas_results]
    median_chi2 = np.median(chi2_stats)
    
    # Expected median under null
    from scipy.stats import chi2
    expected_median = chi2.ppf(0.5, df=1)
    
    # Inflation factor lambda
    lambda_gc = median_chi2 / expected_median
    
    # Adjust test statistics
    for result in gwas_results:
        result['chi2_adjusted'] = (result['beta'] / result['se'])**2 / lambda_gc
        result['p_adjusted'] = chi2.sf(result['chi2_adjusted'], df=1)
    
    return lambda_gc
```

### Demographic Inference

```python
def infer_demographic_history(sfs, method='dadi'):
    """
    Infer population size changes from site frequency spectrum
    """
    
    if method == 'dadi':
        # Diffusion approximation for demographic inference
        
        def model_func(params, ns):
            """
            Example: Population expansion model
            """
            nu, T = params  # Final size, time
            
            # Grid points for frequency spectrum
            pts_l = [40, 50, 60]
            
            # Initial spectrum (equilibrium)
            fs = dadi.Demographics1D.snm(ns)
            
            # Instantaneous size change
            fs = dadi.Demographics1D.two_epoch(fs, nu, T)
            
            return fs
        
        # Optimize parameters
        from scipy.optimize import minimize
        
        def objective(params):
            model_sfs = model_func(params, len(sfs))
            # Poisson likelihood
            ll = -np.sum(sfs * np.log(model_sfs) - model_sfs)
            return -ll
        
        result = minimize(objective, x0=[2.0, 0.1])
        
        return result.x
```

### Common Pitfalls and Solutions

| Pitfall | Consequence | Solution |
|---------|------------|----------|
| **Ignoring population structure** | False positive associations | Include PCA as covariates |
| **Assuming infinite sites** | Wrong mutation model | Use finite sites models |
| **Ignoring recombination** | Wrong genealogies | Use ARG or SMC |
| **Batch effects in sequencing** | Spurious population structure | Careful QC and normalization |
| **Reference bias** | Missing variation | Use graph genomes |
| **Ascertainment bias** | Skewed allele frequencies | Model discovery process |

### References
- Wakeley (2008). Coalescent Theory: An Introduction
- Nielsen & Slatkin (2013). An Introduction to Population Genetics
- Hartl & Clark (2006). Principles of Population Genetics
- Durbin et al. (2010). A map of human genome variation from population-scale sequencing