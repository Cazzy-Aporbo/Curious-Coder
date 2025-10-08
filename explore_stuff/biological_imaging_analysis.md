# Biological Imaging Analysis
## Extracting Quantitative Information from Microscopy and Medical Images

### Intent
Images are high-dimensional measurements of biological systems. This document provides mathematical frameworks for preprocessing, segmentation, tracking, and quantitative analysis of biological images, from subcellular structures to whole organisms.

### The Fundamental Challenge: Images as Noisy, Indirect Measurements

```
True Biology → Physical Interaction → Detection → Digitization → Image

Each step adds noise, artifacts, and distortions
Our job: Invert this process to recover biology
```

### Image Formation and Point Spread Functions

```python
def image_formation_model():
    """
    How microscopes create images
    
    Image = (Object ⊗ PSF) × Illumination + Noise
    
    where ⊗ denotes convolution
    """
    
    def point_spread_function(wavelength, NA, n=1.0):
        """
        Airy disk for diffraction-limited optics
        
        Resolution (Rayleigh): d = 0.61λ/NA
        Resolution (Abbe): d = λ/(2NA)
        """
        
        # Lateral PSF (2D Airy pattern)
        def airy_2d(x, y):
            r = np.sqrt(x**2 + y**2)
            k = 2 * np.pi / wavelength
            v = k * NA * r / n
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                psf = (2 * j1(v) / v)**2  # j1 is Bessel function
                psf[v == 0] = 1
            
            return psf
        
        # Axial PSF (along z)
        def airy_axial(z):
            u = k * NA**2 * z / (2 * n)
            psf_z = (np.sin(u) / u)**2
            return psf_z
        
        return airy_2d, airy_axial
    
    def simulate_imaging(ground_truth, psf, noise_model='poisson'):
        """
        Forward model of image formation
        """
        from scipy.ndimage import convolve
        
        # Convolution with PSF
        blurred = convolve(ground_truth, psf, mode='reflect')
        
        # Add noise
        if noise_model == 'poisson':
            # Photon shot noise
            image = np.random.poisson(blurred * 100) / 100
        elif noise_model == 'gaussian':
            # Read noise
            image = blurred + np.random.normal(0, 0.01, blurred.shape)
        elif noise_model == 'mixed':
            # Both shot and read noise
            image = np.random.poisson(blurred * 100) / 100
            image += np.random.normal(0, 0.01, image.shape)
        
        return np.clip(image, 0, 1)
```

### Deconvolution: Reversing the Blurring

```python
def deconvolution_methods():
    """
    Restore resolution by inverting PSF convolution
    """
    
    def richardson_lucy(image, psf, iterations=10):
        """
        ML estimation for Poisson noise
        """
        from scipy.ndimage import convolve
        
        estimate = np.ones_like(image) * image.mean()
        psf_mirror = psf[::-1, ::-1]
        
        for _ in range(iterations):
            # RL update
            blur_estimate = convolve(estimate, psf, mode='reflect')
            ratio = image / (blur_estimate + 1e-10)
            gradient = convolve(ratio, psf_mirror, mode='reflect')
            estimate *= gradient
            
            # Non-negativity
            estimate = np.maximum(estimate, 0)
        
        return estimate
    
    def wiener_deconvolution(image, psf, noise_power=0.01):
        """
        Frequency domain deconvolution with regularization
        """
        from scipy import fft
        
        # Fourier transforms
        image_fft = fft.fft2(image)
        psf_fft = fft.fft2(psf, s=image.shape)
        
        # Wiener filter
        psf_fft_conj = np.conj(psf_fft)
        denominator = np.abs(psf_fft)**2 + noise_power
        
        deconv_fft = image_fft * psf_fft_conj / denominator
        deconvolved = np.real(fft.ifft2(deconv_fft))
        
        return deconvolved
    
    def regularized_deconvolution(image, psf, reg_param=0.01):
        """
        Total variation regularization
        """
        from scipy.optimize import minimize
        
        def objective(x_flat):
            x = x_flat.reshape(image.shape)
            
            # Data fidelity term
            from scipy.ndimage import convolve
            residual = image - convolve(x, psf, mode='reflect')
            fidelity = np.sum(residual**2)
            
            # Total variation regularization
            grad_x = np.diff(x, axis=0)
            grad_y = np.diff(x, axis=1)
            tv = np.sum(np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2))
            
            return fidelity + reg_param * tv
        
        # Optimize
        result = minimize(objective, image.flatten(), method='L-BFGS-B')
        return result.x.reshape(image.shape)
```

### Segmentation: Finding Objects

#### 1. Classical Methods

```python
def segmentation_classical(image):
    """
    Traditional segmentation approaches
    """
    
    def otsu_threshold(image):
        """
        Optimal threshold by maximizing between-class variance
        """
        # Histogram
        hist, bins = np.histogram(image.flatten(), 256)
        hist = hist.astype(float) / hist.sum()
        
        # Cumulative distributions
        cumsum = hist.cumsum()
        mean_cumsum = (hist * np.arange(256)).cumsum()
        
        # Between-class variance for each threshold
        global_mean = mean_cumsum[-1]
        
        variances = []
        for t in range(256):
            w0 = cumsum[t]
            w1 = 1 - w0
            
            if w0 == 0 or w1 == 0:
                variances.append(0)
                continue
            
            mu0 = mean_cumsum[t] / w0
            mu1 = (global_mean - mean_cumsum[t]) / w1
            
            variance = w0 * w1 * (mu0 - mu1)**2
            variances.append(variance)
        
        threshold = np.argmax(variances) / 255.0
        return image > threshold
    
    def watershed_segmentation(image):
        """
        Watershed for touching objects
        """
        from scipy import ndimage
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        
        # Distance transform
        distance = ndimage.distance_transform_edt(image)
        
        # Find local maxima as markers
        local_maxima = peak_local_max(
            distance,
            min_distance=10,
            indices=False
        )
        markers = ndimage.label(local_maxima)[0]
        
        # Watershed
        labels = watershed(-distance, markers, mask=image)
        
        return labels
    
    def active_contours(image, init_contour):
        """
        Snake/Level set methods
        """
        from skimage.segmentation import active_contour
        
        # Image forces
        from skimage.filters import gaussian
        
        # External energy (image gradient)
        edge = gaussian(image, 1, mode='reflect')
        
        # Evolve contour
        snake = active_contour(
            edge,
            init_contour,
            alpha=0.01,  # Smoothness
            beta=10,     # Rigidity
            gamma=0.001  # Step size
        )
        
        return snake
```

#### 2. Machine Learning Segmentation

```python
def ml_segmentation():
    """
    Learning-based segmentation
    """
    
    def pixel_classification(image, training_data):
        """
        Random forest pixel classifier
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Extract features for each pixel
        def extract_pixel_features(image, x, y):
            features = []
            
            # Intensity
            features.append(image[x, y])
            
            # Local statistics (different scales)
            for radius in [3, 5, 7]:
                x_min = max(0, x - radius)
                x_max = min(image.shape[0], x + radius + 1)
                y_min = max(0, y - radius)
                y_max = min(image.shape[1], y + radius + 1)
                
                patch = image[x_min:x_max, y_min:y_max]
                
                features.extend([
                    patch.mean(),
                    patch.std(),
                    np.percentile(patch, 25),
                    np.percentile(patch, 75)
                ])
            
            # Gradient magnitude
            from scipy.ndimage import sobel
            grad_x = sobel(image, axis=0)[x, y]
            grad_y = sobel(image, axis=1)[x, y]
            features.append(np.sqrt(grad_x**2 + grad_y**2))
            
            return features
        
        # Train classifier
        X_train, y_train = training_data
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        
        # Predict for all pixels
        predictions = np.zeros_like(image)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                features = extract_pixel_features(image, x, y)
                predictions[x, y] = clf.predict([features])[0]
        
        return predictions
    
    def unet_architecture():
        """
        U-Net for biomedical segmentation
        """
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        
        def conv_block(inputs, filters):
            x = layers.Conv2D(filters, 3, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            return x
        
        # Encoder
        inputs = layers.Input((256, 256, 1))
        
        c1 = conv_block(inputs, 64)
        p1 = layers.MaxPooling2D(2)(c1)
        
        c2 = conv_block(p1, 128)
        p2 = layers.MaxPooling2D(2)(c2)
        
        c3 = conv_block(p2, 256)
        p3 = layers.MaxPooling2D(2)(c3)
        
        c4 = conv_block(p3, 512)
        p4 = layers.MaxPooling2D(2)(c4)
        
        # Bottleneck
        c5 = conv_block(p4, 1024)
        
        # Decoder
        u6 = layers.UpSampling2D(2)(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = conv_block(u6, 512)
        
        u7 = layers.UpSampling2D(2)(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = conv_block(u7, 256)
        
        u8 = layers.UpSampling2D(2)(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = conv_block(u8, 128)
        
        u9 = layers.UpSampling2D(2)(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = conv_block(u9, 64)
        
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
        
        model = Model(inputs, outputs)
        
        return model
```

### Object Tracking

```python
def tracking_algorithms():
    """
    Follow objects through time
    """
    
    def nearest_neighbor_tracking(detections_t1, detections_t2, max_distance=50):
        """
        Simple frame-to-frame association
        """
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        
        # Centroids
        centroids_t1 = [get_centroid(obj) for obj in detections_t1]
        centroids_t2 = [get_centroid(obj) for obj in detections_t2]
        
        # Cost matrix (distances)
        cost_matrix = cdist(centroids_t1, centroids_t2)
        
        # Set impossible matches to high cost
        cost_matrix[cost_matrix > max_distance] = 1e6
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Build tracks
        tracks = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_distance:
                tracks.append((r, c, cost_matrix[r, c]))
        
        return tracks
    
    def kalman_tracking():
        """
        Kalman filter for smooth tracking with prediction
        """
        from filterpy.kalman import KalmanFilter
        
        class CellTracker:
            def __init__(self):
                self.kf = KalmanFilter(dim_x=4, dim_z=2)
                
                # State: [x, y, vx, vy]
                self.kf.x = np.array([0, 0, 0, 0])
                
                # State transition (constant velocity)
                self.kf.F = np.array([
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                
                # Measurement function (observe position)
                self.kf.H = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]
                ])
                
                # Covariances
                self.kf.R = np.eye(2) * 5  # Measurement noise
                self.kf.Q = np.eye(4) * 0.1  # Process noise
                self.kf.P = np.eye(4) * 100  # Initial uncertainty
                
            def predict(self):
                self.kf.predict()
                return self.kf.x[:2]
            
            def update(self, measurement):
                self.kf.update(measurement)
                return self.kf.x[:2]
        
        return CellTracker
    
    def particle_filter_tracking(n_particles=100):
        """
        For non-linear, non-Gaussian tracking
        """
        
        class ParticleFilter:
            def __init__(self, n_particles):
                self.n_particles = n_particles
                self.particles = None
                self.weights = np.ones(n_particles) / n_particles
                
            def predict(self, motion_model):
                """
                Propagate particles through motion model
                """
                for i in range(self.n_particles):
                    # Add noise for diversity
                    noise = np.random.normal(0, 1, 2)
                    self.particles[i] = motion_model(self.particles[i]) + noise
            
            def update(self, measurement, measurement_model):
                """
                Update weights based on measurement
                """
                for i in range(self.n_particles):
                    predicted_measurement = measurement_model(self.particles[i])
                    
                    # Likelihood (Gaussian)
                    distance = np.linalg.norm(measurement - predicted_measurement)
                    self.weights[i] = np.exp(-distance**2 / (2 * 5**2))
                
                # Normalize weights
                self.weights /= self.weights.sum()
                
                # Resample if effective sample size is low
                n_eff = 1 / np.sum(self.weights**2)
                if n_eff < self.n_particles / 2:
                    self.resample()
            
            def resample(self):
                """
                Systematic resampling
                """
                cumsum = np.cumsum(self.weights)
                positions = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
                
                indices = np.searchsorted(cumsum, positions)
                self.particles = self.particles[indices]
                self.weights = np.ones(self.n_particles) / self.n_particles
            
            def estimate(self):
                """
                Weighted mean of particles
                """
                return np.average(self.particles, weights=self.weights, axis=0)
        
        return ParticleFilter
```

### Feature Extraction

```python
def morphological_features(mask):
    """
    Extract quantitative features from segmented objects
    """
    from skimage import measure
    from scipy import ndimage
    
    features = {}
    
    # Basic measurements
    features['area'] = mask.sum()
    features['perimeter'] = measure.perimeter(mask)
    
    # Shape descriptors
    features['circularity'] = 4 * np.pi * features['area'] / features['perimeter']**2
    features['eccentricity'] = measure.regionprops(mask.astype(int))[0].eccentricity
    features['solidity'] = measure.regionprops(mask.astype(int))[0].solidity
    
    # Moments
    moments = measure.moments(mask)
    features['centroid'] = (moments[1, 0] / moments[0, 0], 
                           moments[0, 1] / moments[0, 0])
    
    # Central moments (invariant to translation)
    hu_moments = measure.moments_hu(moments)
    for i, hu in enumerate(hu_moments):
        features[f'hu_moment_{i}'] = hu
    
    # Texture (if intensity image available)
    def texture_features(image, mask):
        from skimage.feature import graycomatrix, graycoprops
        
        # Gray level co-occurrence matrix
        masked_image = image * mask
        glcm = graycomatrix(
            masked_image.astype(np.uint8),
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Haralick features
        features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
        features['energy'] = graycoprops(glcm, 'energy')[0, 0]
        features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]
        
        return features
    
    return features
```

### Registration and Alignment

```python
def image_registration():
    """
    Align images across time, samples, or modalities
    """
    
    def rigid_registration(fixed, moving):
        """
        Translation and rotation only
        """
        from skimage.registration import phase_cross_correlation
        
        # FFT-based registration
        shift, error, phasediff = phase_cross_correlation(fixed, moving)
        
        # Apply transformation
        from scipy.ndimage import shift as nd_shift
        registered = nd_shift(moving, shift)
        
        return registered, shift
    
    def affine_registration(fixed, moving):
        """
        Includes scaling and shearing
        """
        from skimage.transform import AffineTransform, warp
        from scipy.optimize import minimize
        
        def cost_function(params):
            # Build affine matrix
            transform = AffineTransform(
                matrix=params[:4].reshape(2, 2),
                translation=params[4:]
            )
            
            # Warp image
            warped = warp(moving, transform.inverse)
            
            # Similarity metric (negative correlation)
            return -np.corrcoef(fixed.flatten(), warped.flatten())[0, 1]
        
        # Optimize
        initial_params = np.array([1, 0, 0, 1, 0, 0])  # Identity
        result = minimize(cost_function, initial_params)
        
        # Apply optimal transformation
        optimal_transform = AffineTransform(
            matrix=result.x[:4].reshape(2, 2),
            translation=result.x[4:]
        )
        registered = warp(moving, optimal_transform.inverse)
        
        return registered, optimal_transform
    
    def elastic_registration(fixed, moving):
        """
        Non-rigid deformation
        """
        # Demons algorithm
        from scipy.ndimage import gaussian_filter
        
        def demons_iteration(fixed, moving, sigma_fluid=1.0):
            # Compute update field
            diff = fixed - moving
            
            # Gradient of fixed image
            grad = np.gradient(fixed)
            grad_mag = grad[0]**2 + grad[1]**2
            
            # Demon forces
            scale = diff / (grad_mag + diff**2)
            
            u = scale * grad[0]
            v = scale * grad[1]
            
            # Smooth deformation field
            u = gaussian_filter(u, sigma_fluid)
            v = gaussian_filter(v, sigma_fluid)
            
            return u, v
        
        # Iterate
        u_total = np.zeros_like(fixed)
        v_total = np.zeros_like(fixed)
        
        for _ in range(100):
            # Warp moving image
            from scipy.interpolate import RectBivariateSpline
            
            y, x = np.mgrid[0:fixed.shape[0], 0:fixed.shape[1]]
            interp = RectBivariateSpline(
                np.arange(moving.shape[0]),
                np.arange(moving.shape[1]),
                moving
            )
            
            warped = interp.ev(y - v_total, x - u_total)
            
            # Compute update
            u, v = demons_iteration(fixed, warped)
            
            # Accumulate deformation
            u_total += u
            v_total += v
        
        return warped, (u_total, v_total)
```

### Super-Resolution

```python
def super_resolution_methods():
    """
    Exceed diffraction limit
    """
    
    def structured_illumination(images, patterns):
        """
        SIM reconstruction
        """
        # Fourier domain processing
        from scipy import fft
        
        n_patterns = len(patterns)
        n_orientations = 3
        
        # Extract high-frequency information
        reconstructed = np.zeros(
            (images[0].shape[0]*2, images[0].shape[1]*2),
            dtype=complex
        )
        
        for orientation in range(n_orientations):
            # Images with different pattern phases
            imgs = images[orientation*n_patterns:(orientation+1)*n_patterns]
            
            # Separate frequency components
            # ... complex math involving pattern frequency ...
            
            # Place in extended Fourier space
            pass
        
        # Inverse Fourier transform
        super_res = np.real(fft.ifft2(reconstructed))
        
        return super_res
    
    def storm_palm_reconstruction(localizations):
        """
        Single molecule localization microscopy
        """
        
        # Render localizations to image
        pixel_size = 10  # nm
        img_size = 2048
        
        image = np.zeros((img_size, img_size))
        
        for loc in localizations:
            x, y = loc['x'] / pixel_size, loc['y'] / pixel_size
            
            # Add Gaussian based on localization precision
            sigma = loc['precision'] / pixel_size
            
            x_int, y_int = int(x), int(y)
            
            if 0 <= x_int < img_size and 0 <= y_int < img_size:
                # Simple addition (could use proper Gaussian)
                image[y_int, x_int] += 1
        
        # Gaussian blur based on average precision
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=1)
        
        return image
```

### Quality Control Metrics

```python
def image_quality_metrics(image, reference=None):
    """
    Assess image quality
    """
    
    metrics = {}
    
    # Signal-to-noise ratio
    from scipy import stats
    
    # Estimate noise (MAD in background)
    background = image[image < np.percentile(image, 10)]
    noise = stats.median_abs_deviation(background)
    signal = np.percentile(image, 95) - np.median(background)
    
    metrics['snr'] = signal / noise if noise > 0 else np.inf
    
    # Contrast
    metrics['contrast'] = (image.max() - image.min()) / (image.max() + image.min())
    
    # Focus quality (gradient-based)
    from scipy.ndimage import sobel
    grad_mag = np.sqrt(sobel(image, 0)**2 + sobel(image, 1)**2)
    metrics['focus_score'] = grad_mag.var()
    
    if reference is not None:
        # Full reference metrics
        
        # Mean squared error
        metrics['mse'] = np.mean((image - reference)**2)
        
        # Peak signal-to-noise ratio
        metrics['psnr'] = 10 * np.log10(1.0 / metrics['mse'])
        
        # Structural similarity
        from skimage.metrics import structural_similarity
        metrics['ssim'] = structural_similarity(image, reference)
    
    return metrics
```

### Colocalization Analysis

```python
def colocalization_analysis(channel1, channel2, mask=None):
    """
    Quantify spatial overlap of two channels
    """
    
    if mask is not None:
        channel1 = channel1 * mask
        channel2 = channel2 * mask
    
    # Pearson correlation
    pearson_r = np.corrcoef(channel1.flatten(), channel2.flatten())[0, 1]
    
    # Manders coefficients
    threshold1 = np.percentile(channel1[channel1 > 0], 10)
    threshold2 = np.percentile(channel2[channel2 > 0], 10)
    
    mask1 = channel1 > threshold1
    mask2 = channel2 > threshold2
    
    M1 = np.sum(channel1[mask2]) / np.sum(channel1)  # Fraction of ch1 overlapping ch2
    M2 = np.sum(channel2[mask1]) / np.sum(channel2)  # Fraction of ch2 overlapping ch1
    
    # Li's ICQ (Intensity Correlation Quotient)
    mean1 = channel1.mean()
    mean2 = channel2.mean()
    
    pdm = (channel1 - mean1) * (channel2 - mean2)
    icq = (pdm > 0).sum() / pdm.size - 0.5
    
    # Costes significance test
    def costes_randomization(ch1, ch2, n_iterations=100):
        """
        Test significance by randomization
        """
        original_r = pearson_r
        
        random_rs = []
        for _ in range(n_iterations):
            # Random shift
            shift_x = np.random.randint(ch1.shape[0])
            shift_y = np.random.randint(ch1.shape[1])
            
            ch2_shifted = np.roll(np.roll(ch2, shift_x, axis=0), shift_y, axis=1)
            
            r = np.corrcoef(ch1.flatten(), ch2_shifted.flatten())[0, 1]
            random_rs.append(r)
        
        p_value = (np.array(random_rs) >= original_r).sum() / n_iterations
        
        return p_value
    
    return {
        'pearson': pearson_r,
        'manders_1': M1,
        'manders_2': M2,
        'icq': icq,
        'p_value': costes_randomization(channel1, channel2)
    }
```

### Common Pitfalls and Solutions

| Pitfall | Consequence | Solution |
|---------|------------|----------|
| **Photobleaching** | Intensity decay over time | Correction models, reduce exposure |
| **Motion blur** | Poor segmentation | Registration, deconvolution |
| **Uneven illumination** | Segmentation errors | Flat-field correction |
| **Saturated pixels** | Lost information | HDR imaging, reduce gain |
| **Z-drift** | Misalignment over time | Autofocus, fiducial markers |
| **Refractive index mismatch** | Spherical aberration | Correct immersion media |

### References
- Szeliski (2010). Computer Vision: Algorithms and Applications
- Bankhead (2022). Introduction to Bioimage Analysis
- Murphy (2012). An active role for machine learning in drug discovery
- Carpenter et al. (2006). CellProfiler: image analysis software