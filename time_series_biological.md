# Time Series Analysis for Biological Rhythms and Dynamics
## Extracting Temporal Patterns from Living Systems

### Intent
Biological systems are inherently dynamic - circadian rhythms oscillate, cell cycles progress, diseases evolve, populations fluctuate. This document provides rigorous methods for analyzing biological time series data, from detecting periodicity to forecasting dynamics and inferring regulatory mechanisms from temporal patterns.

### Mathematical Framework for Biological Time Series

**Time Series Representation:**
```
X(t) = μ(t) + S(t) + ε(t)

where:
- μ(t): trend component (developmental progression)
- S(t): seasonal/periodic component (circadian, cell cycle)
- ε(t): stochastic component (biological noise)
```

**Biological Time Scales:**

| System | Time Scale | Sampling Rate | Key Challenges |
|--------|------------|---------------|----------------|
| Molecular dynamics | ps - ns | Continuous | Computational cost |
| Metabolic flux | Seconds - minutes | 30s - 5min | Technical noise |
| Gene expression | Minutes - hours | 15min - 2hr | Destructive sampling |
| Circadian rhythms | 24 hours | 2-4 hours | Multiple periods |
| Cell cycle | Hours - days | 30min - 2hr | Asynchrony |
| Development | Days - weeks | Daily | Non-stationarity |
| Evolution | Years - millennia | Generational | Incomplete fossil record |

### Periodicity Detection and Analysis

#### 1. Fourier Analysis for Biological Rhythms

```python
def biological_fourier_analysis(time_series, sampling_rate, 
                               min_period=None, max_period=None):
    """
    Detect periodic components in biological data
    """
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import periodogram
    
    n = len(time_series)
    
    # Remove trend
    time = np.arange(n)
    z = np.polyfit(time, time_series, 1)
    trend = np.poly1d(z)
    detrended = time_series - trend(time)
    
    # Compute power spectral density
    frequencies, power = periodogram(detrended, sampling_rate, 
                                    scaling='density')
    
    # Convert to periods
    periods = 1 / frequencies[1:]  # Skip DC component
    power = power[1:]
    
    # Filter by biological relevance
    if min_period and max_period:
        mask = (periods >= min_period) & (periods <= max_period)
        periods = periods[mask]
        power = power[mask]
    
    # Find significant peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(power, 
                                  height=np.percentile(power, 95),
                                  distance=5)
    
    significant_periods = periods[peaks]
    peak_powers = power[peaks]
    
    # Test significance via permutation
    significance = test_periodicity_significance(
        detrended, significant_periods, n_permutations=1000
    )
    
    return {
        'periods': significant_periods,
        'powers': peak_powers,
        'p_values': significance,
        'power_spectrum': (periods, power)
    }

def test_periodicity_significance(time_series, test_periods, n_permutations=1000):
    """
    Permutation test for periodicity significance
    """
    observed_power = []
    null_powers = []
    
    for period in test_periods:
        # Fit sinusoid
        t = np.arange(len(time_series))
        omega = 2 * np.pi / period
        
        X = np.column_stack([
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones(len(t))
        ])
        
        # Observed R²
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, time_series)
        observed_r2 = model.score(X, time_series)
        observed_power.append(observed_r2)
        
        # Null distribution
        null_r2 = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(time_series)
            model.fit(X, shuffled)
            null_r2.append(model.score(X, shuffled))
        
        null_powers.append(null_r2)
    
    # Compute p-values
    p_values = []
    for obs, null_dist in zip(observed_power, null_powers):
        p = np.mean(null_dist >= obs)
        p_values.append(p)
    
    return p_values
```

#### 2. Wavelet Analysis for Non-Stationary Rhythms

```python
def wavelet_transform_biological(time_series, sampling_rate, 
                                wavelet='morlet', scales=None):
    """
    Time-frequency analysis for changing rhythms
    """
    import pywt
    
    if scales is None:
        # Scales corresponding to periods of interest
        # For circadian: 16-32 hours
        # For ultradian: 1-8 hours
        min_period = 1  # hours
        max_period = 48  # hours
        
        # Convert to scales
        frequencies = np.logspace(
            np.log10(1/max_period),
            np.log10(1/min_period),
            100
        )
        scales = sampling_rate / (2 * frequencies)
    
    # Continuous wavelet transform
    coefficients, frequencies = pywt.cwt(
        time_series, scales, wavelet, 
        sampling_period=1/sampling_rate
    )
    
    # Power spectrum
    power = np.abs(coefficients)**2
    
    # Global wavelet spectrum (average over time)
    global_power = np.mean(power, axis=1)
    
    # Scale-averaged wavelet power (for specific period bands)
    def scale_average(period_range):
        period_mask = (1/frequencies >= period_range[0]) & \
                     (1/frequencies <= period_range[1])
        return np.mean(power[period_mask, :], axis=0)
    
    # Detect rhythm changes
    circadian_power = scale_average([20, 28])  # 20-28 hour band
    
    # Find phase
    phase = np.angle(coefficients)
    
    return {
        'coefficients': coefficients,
        'power': power,
        'frequencies': frequencies,
        'periods': 1/frequencies,
        'global_power': global_power,
        'circadian_power': circadian_power,
        'phase': phase
    }
```

#### 3. Empirical Mode Decomposition (EMD)

```python
def empirical_mode_decomposition(time_series, max_imfs=10):
    """
    Decompose signal into Intrinsic Mode Functions (IMFs)
    Adaptive for non-linear, non-stationary biological signals
    """
    from PyEMD import EMD
    
    emd = EMD()
    IMFs = emd(time_series)
    
    # Analyze each IMF
    imf_properties = []
    
    for i, imf in enumerate(IMFs):
        # Instantaneous frequency via Hilbert transform
        from scipy.signal import hilbert
        
        analytic = hilbert(imf)
        instantaneous_phase = np.unwrap(np.angle(analytic))
        instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi)
        
        # Mean period
        mean_period = 1 / np.mean(instantaneous_freq[instantaneous_freq > 0])
        
        # Energy
        energy = np.sum(imf**2)
        
        imf_properties.append({
            'imf': imf,
            'mean_period': mean_period,
            'energy': energy,
            'instantaneous_freq': instantaneous_freq
        })
    
    return IMFs, imf_properties
```

### Phase and Synchronization Analysis

```python
def phase_synchronization_analysis(signal1, signal2, method='hilbert'):
    """
    Detect synchronization between biological oscillators
    """
    
    if method == 'hilbert':
        # Hilbert transform for phase
        from scipy.signal import hilbert
        
        analytic1 = hilbert(signal1)
        analytic2 = hilbert(signal2)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
    elif method == 'wavelet':
        # Wavelet-based phase
        coeffs1 = wavelet_transform_biological(signal1)
        coeffs2 = wavelet_transform_biological(signal2)
        
        # Use dominant frequency
        dominant_idx = np.argmax(coeffs1['global_power'])
        phase1 = coeffs1['phase'][dominant_idx, :]
        phase2 = coeffs2['phase'][dominant_idx, :]
    
    # Phase difference
    phase_diff = phase1 - phase2
    phase_diff_wrapped = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
    
    # Synchronization indices
    
    # 1. Phase Locking Value (PLV)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    # 2. Shannon entropy of phase difference
    hist, _ = np.histogram(phase_diff_wrapped, bins=20)
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    max_entropy = np.log(20)
    
    # Synchronization index (0 = random, 1 = perfect sync)
    sync_index = 1 - entropy/max_entropy
    
    # 3. Phase coherence
    n = len(phase1)
    coherence = np.abs(np.sum(np.exp(1j * phase_diff))) / n
    
    # Statistical significance
    # Surrogate test
    n_surrogates = 1000
    surrogate_plv = []
    
    for _ in range(n_surrogates):
        # Phase randomization
        surrogate2 = np.roll(signal2, np.random.randint(len(signal2)))
        
        # Recompute PLV
        if method == 'hilbert':
            analytic_surr = hilbert(surrogate2)
            phase_surr = np.angle(analytic_surr)
        
        phase_diff_surr = phase1 - phase_surr
        plv_surr = np.abs(np.mean(np.exp(1j * phase_diff_surr)))
        surrogate_plv.append(plv_surr)
    
    p_value = np.mean(surrogate_plv >= plv)
    
    return {
        'phase1': phase1,
        'phase2': phase2,
        'phase_difference': phase_diff_wrapped,
        'PLV': plv,
        'synchronization_index': sync_index,
        'coherence': coherence,
        'p_value': p_value
    }
```

### State Space Reconstruction

```python
def delay_embedding(time_series, embedding_dim=3, time_delay=None):
    """
    Takens' embedding theorem for attractor reconstruction
    """
    
    if time_delay is None:
        # Estimate optimal delay via mutual information minimum
        time_delay = estimate_time_delay(time_series)
    
    n = len(time_series)
    n_vectors = n - (embedding_dim - 1) * time_delay
    
    embedded = np.zeros((n_vectors, embedding_dim))
    
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i*time_delay:i*time_delay + n_vectors]
    
    return embedded, time_delay

def estimate_time_delay(time_series, max_lag=100):
    """
    Find optimal delay via first minimum of mutual information
    """
    from sklearn.metrics import mutual_info_score
    
    mi_values = []
    
    for lag in range(1, max_lag):
        # Compute MI between x(t) and x(t+lag)
        x_t = time_series[:-lag]
        x_lag = time_series[lag:]
        
        # Discretize for MI calculation
        n_bins = int(np.sqrt(len(x_t)))
        x_t_discrete = pd.cut(x_t, bins=n_bins, labels=False)
        x_lag_discrete = pd.cut(x_lag, bins=n_bins, labels=False)
        
        mi = mutual_info_score(x_t_discrete, x_lag_discrete)
        mi_values.append(mi)
    
    # Find first local minimum
    from scipy.signal import argrelextrema
    minima = argrelextrema(np.array(mi_values), np.less)[0]
    
    if len(minima) > 0:
        optimal_delay = minima[0] + 1
    else:
        # Default to autocorrelation zero crossing
        autocorr = np.correlate(time_series, time_series, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        zero_crossing = np.where(np.diff(np.sign(autocorr)))[0]
        optimal_delay = zero_crossing[0] if len(zero_crossing) > 0 else 10
    
    return optimal_delay

def estimate_embedding_dimension(time_series, max_dim=10):
    """
    False Nearest Neighbors for embedding dimension
    """
    from sklearn.neighbors import NearestNeighbors
    
    time_delay = estimate_time_delay(time_series)
    fnn_fractions = []
    
    for dim in range(1, max_dim):
        # Embed in dim and dim+1
        embedded_d = delay_embedding(time_series, dim, time_delay)[0]
        embedded_d1 = delay_embedding(time_series, dim+1, time_delay)[0]
        
        # Find nearest neighbors in dim
        nbrs = NearestNeighbors(n_neighbors=2)
        nbrs.fit(embedded_d)
        distances_d, indices = nbrs.kneighbors(embedded_d)
        
        # Check if still neighbors in dim+1
        false_nn_count = 0
        
        for i in range(len(embedded_d)):
            nn_idx = indices[i, 1]  # Nearest neighbor (not self)
            
            dist_d = distances_d[i, 1]
            dist_d1 = np.linalg.norm(embedded_d1[i] - embedded_d1[nn_idx])
            
            # False nearest neighbor criterion
            if (dist_d1 - dist_d) / dist_d > 10:
                false_nn_count += 1
        
        fnn_fraction = false_nn_count / len(embedded_d)
        fnn_fractions.append(fnn_fraction)
        
        # Stop if FNN fraction is low enough
        if fnn_fraction < 0.01:
            return dim
    
    return max_dim
```

### Nonlinear Dynamics and Chaos Detection

```python
def lyapunov_exponent(time_series, embedding_dim=None, time_delay=None):
    """
    Estimate largest Lyapunov exponent
    Positive = chaotic, 0 = periodic, negative = fixed point
    """
    
    if embedding_dim is None:
        embedding_dim = estimate_embedding_dimension(time_series)
    
    if time_delay is None:
        time_delay = estimate_time_delay(time_series)
    
    # Reconstruct attractor
    embedded, _ = delay_embedding(time_series, embedding_dim, time_delay)
    
    n = len(embedded)
    
    # Find nearest neighbors for each point
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(embedded)
    distances, indices = nbrs.kneighbors(embedded)
    
    # Track divergence
    divergences = []
    
    for i in range(n - 10):  # Leave room for evolution
        # Initial separation
        j = indices[i, 1]  # Nearest neighbor
        initial_dist = distances[i, 1]
        
        if initial_dist > 0:
            # Evolve both trajectories
            for k in range(1, min(10, n-max(i,j))):
                final_dist = np.linalg.norm(embedded[i+k] - embedded[j+k])
                
                if final_dist > 0:
                    divergence_rate = np.log(final_dist / initial_dist) / k
                    divergences.append(divergence_rate)
    
    # Average divergence rate = Lyapunov exponent
    lyapunov = np.mean(divergences) if divergences else 0
    
    # Determine system type
    if lyapunov > 0.01:
        dynamics_type = "Chaotic"
    elif abs(lyapunov) < 0.01:
        dynamics_type = "Periodic/Quasi-periodic"
    else:
        dynamics_type = "Fixed point/Limit cycle"
    
    return lyapunov, dynamics_type
```

### Granger Causality for Regulatory Networks

```python
def granger_causality_test(x, y, max_lag=5):
    """
    Test if x Granger-causes y
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Prepare data
    data = np.column_stack([y, x])
    
    # Test for each lag
    results = grangercausalitytests(data, max_lag, verbose=False)
    
    # Extract p-values
    p_values = []
    for lag in range(1, max_lag + 1):
        p_val = results[lag][0]['ssr_ftest'][1]
        p_values.append(p_val)
    
    # Optimal lag (minimum p-value)
    optimal_lag = np.argmin(p_values) + 1
    significant = p_values[optimal_lag - 1] < 0.05
    
    return {
        'causes': significant,
        'optimal_lag': optimal_lag,
        'p_value': p_values[optimal_lag - 1],
        'all_p_values': p_values
    }

def transfer_entropy(source, target, lag=1, bins=10):
    """
    Information-theoretic causality measure
    """
    # Create lagged variables
    n = len(source) - lag
    
    target_future = target[lag:]
    target_past = target[:-lag]
    source_past = source[:-lag]
    
    # Discretize
    t_future_discrete = pd.cut(target_future, bins=bins, labels=False)
    t_past_discrete = pd.cut(target_past, bins=bins, labels=False)
    s_past_discrete = pd.cut(source_past, bins=bins, labels=False)
    
    # Transfer entropy: I(Y_future ; X_past | Y_past)
    # Using chain rule: H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    
    from sklearn.metrics import mutual_info_score
    
    # Conditional entropies (approximated)
    h_future_given_past = conditional_entropy(
        t_future_discrete, t_past_discrete
    )
    
    h_future_given_both = conditional_entropy(
        t_future_discrete, 
        np.column_stack([t_past_discrete, s_past_discrete])
    )
    
    te = h_future_given_past - h_future_given_both
    
    # Normalized TE (0 to 1)
    h_future = entropy(t_future_discrete)
    normalized_te = te / h_future if h_future > 0 else 0
    
    return te, normalized_te

def conditional_entropy(x, y):
    """
    H(X|Y) = H(X,Y) - H(Y)
    """
    from scipy.stats import entropy
    
    # Joint distribution
    xy = np.column_stack([x, y.reshape(-1, 1) if y.ndim == 1 else y])
    joint_dist = np.histogramdd(xy)[0]
    joint_dist = joint_dist / joint_dist.sum()
    
    # Marginal of Y
    if y.ndim == 1:
        y_dist = np.histogram(y)[0]
    else:
        y_dist = np.histogramdd(y)[0]
    y_dist = y_dist / y_dist.sum()
    
    # Conditional entropy
    h_xy = entropy(joint_dist.flatten())
    h_y = entropy(y_dist.flatten())
    
    return h_xy - h_y
```

### Forecasting Biological Dynamics

```python
class BiologicalForecaster:
    """
    Forecasting methods for biological time series
    """
    
    def __init__(self, time_series, metadata=None):
        self.ts = time_series
        self.metadata = metadata  # Temperature, nutrients, etc.
        
    def seasonal_decomposition(self, period=24):
        """
        STL decomposition for biological rhythms
        """
        from statsmodels.tsa.seasonal import STL
        
        stl = STL(self.ts, period=period, seasonal=13)
        result = stl.fit()
        
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        }
    
    def prophet_biological(self, periods_ahead=48, 
                          include_seasonality=True):
        """
        Facebook Prophet adapted for biological data
        """
        from fbprophet import Prophet
        
        # Prepare data
        df = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', 
                               periods=len(self.ts), 
                               freq='H'),
            'y': self.ts
        })
        
        # Configure model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=include_seasonality,
            seasonality_mode='multiplicative'
        )
        
        # Add custom seasonalities
        if include_seasonality:
            # Circadian rhythm
            model.add_seasonality(
                name='circadian',
                period=24,
                fourier_order=5
            )
            
            # Ultradian rhythms
            model.add_seasonality(
                name='ultradian',
                period=4,
                fourier_order=3
            )
        
        # Add external regressors if available
        if self.metadata is not None:
            for col in self.metadata.columns:
                model.add_regressor(col)
                df[col] = self.metadata[col].values
        
        # Fit
        model.fit(df)
        
        # Forecast
        future = model.make_future_dataframe(periods=periods_ahead, freq='H')
        
        if self.metadata is not None:
            # Extend metadata (simple forward fill for demo)
            for col in self.metadata.columns:
                future[col] = df[col].iloc[-1]
        
        forecast = model.predict(future)
        
        return forecast
    
    def lstm_forecast(self, lookback=24, forecast_horizon=12):
        """
        LSTM for nonlinear biological dynamics
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        # Prepare sequences
        X, y = [], []
        
        for i in range(lookback, len(self.ts) - forecast_horizon):
            X.append(self.ts[i-lookback:i])
            y.append(self.ts[i:i+forecast_horizon])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, 
                 input_shape=(lookback, 1)),
            LSTM(50, activation='relu'),
            Dense(forecast_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(X, y, epochs=100, batch_size=32, verbose=0)
        
        # Forecast
        last_sequence = self.ts[-lookback:].reshape((1, lookback, 1))
        forecast = model.predict(last_sequence)
        
        return forecast[0]
```

### Application Pipeline

```python
class BiologicalTimeSeriesAnalysis:
    """
    Complete pipeline for biological time series
    """
    
    def __init__(self, data, sampling_rate, data_type='gene_expression'):
        self.data = data
        self.sampling_rate = sampling_rate
        self.data_type = data_type
        
    def full_analysis(self):
        """
        Comprehensive temporal analysis
        """
        results = {}
        
        # 1. Basic statistics
        results['mean'] = np.mean(self.data)
        results['std'] = np.std(self.data)
        results['cv'] = results['std'] / results['mean']
        
        # 2. Periodicity analysis
        fourier = biological_fourier_analysis(
            self.data, 
            self.sampling_rate,
            min_period=2,
            max_period=100
        )
        results['dominant_periods'] = fourier['periods']
        
        # 3. Wavelet analysis
        wavelet = wavelet_transform_biological(
            self.data, 
            self.sampling_rate
        )
        results['time_frequency'] = wavelet
        
        # 4. Phase analysis
        if len(self.data.shape) > 1:  # Multiple time series
            sync = phase_synchronization_analysis(
                self.data[:, 0], 
                self.data[:, 1]
            )
            results['synchronization'] = sync
        
        # 5. Nonlinear dynamics
        lyap, dynamics = lyapunov_exponent(self.data)
        results['lyapunov'] = lyap
        results['dynamics_type'] = dynamics
        
        # 6. Causality (if multiple series)
        if len(self.data.shape) > 1:
            gc = granger_causality_test(
                self.data[:, 0], 
                self.data[:, 1]
            )
            results['granger_causality'] = gc
        
        # 7. Forecasting
        forecaster = BiologicalForecaster(self.data)
        results['forecast'] = forecaster.lstm_forecast()
        
        return results
    
    def visualize_results(self, results):
        """
        Comprehensive visualization
        """
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        
        # Time series
        axes[0, 0].plot(self.data)
        axes[0, 0].set_title('Original Time Series')
        
        # Power spectrum
        if 'dominant_periods' in results:
            periods, power = results['time_frequency']['power_spectrum']
            axes[0, 1].semilogy(periods, power)
            axes[0, 1].set_title('Power Spectrum')
        
        # Wavelet scalogram
        if 'time_frequency' in results:
            im = axes[1, 0].imshow(results['time_frequency']['power'], 
                                   aspect='auto')
            axes[1, 0].set_title('Wavelet Scalogram')
            
        # Phase portrait
        if len(self.data) > 3:
            embedded, _ = delay_embedding(self.data)
            axes[1, 1].plot(embedded[:, 0], embedded[:, 1], 'b-', alpha=0.5)
            axes[1, 1].set_title('Phase Portrait')
        
        # Continue with other visualizations...
        
        plt.tight_layout()
        return fig
```

### References
- Winfree, A.T. (2001). The Geometry of Biological Time
- Glass, L. & Mackey, M.C. (1988). From Clocks to Chaos: The Rhythms of Life
- Kantz, H. & Schreiber, T. (2003). Nonlinear Time Series Analysis
- Bar-Joseph, Z. et al. (2012). Studying and modelling dynamic biological processes using time-series gene expression data