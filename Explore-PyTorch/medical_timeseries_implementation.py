import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
from adaptive_gradient_scheduler import (
    AdaptiveGradientScheduler,
    ImbalancedDatasetSampler,
    FeatureSpaceRegularizer,
    GradientStatisticsMonitor
)


class TemporalAttentionBlock(nn.Module):
    """
    Self-attention mechanism for capturing temporal dependencies in 
    physiological time series data. Particularly useful for irregular
    sampling rates common in medical monitoring.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection  
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class MedicalTimeSeriesEncoder(nn.Module):
    """
    Encoder for multivariate medical time series with varying sampling rates
    and missing data patterns. Handles common sensor data like ECG, PPG,
    respiration rate, and accelerometer readings.
    """
    
    def __init__(self, 
                 input_channels: int,
                 sequence_length: int,
                 embed_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
        # Convolutional feature extraction
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # Positional encoding for temporal awareness
        self.positional_encoding = nn.Parameter(torch.randn(1, sequence_length, embed_dim))
        
        # Stack of temporal attention blocks
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling strategies
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Convolutional encoding
        conv_features = self.conv_encoder(x)
        
        # Reshape for attention layers
        features = conv_features.transpose(1, 2)
        features = features + self.positional_encoding
        
        # Apply temporal attention blocks
        for block in self.temporal_blocks:
            features = block(features, mask)
        
        # Global pooling for classification
        features_transposed = features.transpose(1, 2)
        max_pooled = self.global_max_pool(features_transposed).squeeze(-1)
        avg_pooled = self.global_avg_pool(features_transposed).squeeze(-1)
        global_features = torch.cat([max_pooled, avg_pooled], dim=1)
        
        return global_features, features


class PatientRiskStratificationModel(nn.Module):
    """
    Complete model for patient risk stratification using multimodal
    physiological data. Incorporates adaptive training components for
    handling class imbalance common in medical datasets.
    """
    
    def __init__(self,
                 input_channels: int = 5,
                 sequence_length: int = 256,
                 num_risk_levels: int = 4,
                 embed_dim: int = 128,
                 num_layers: int = 3):
        
        super().__init__()
        self.encoder = MedicalTimeSeriesEncoder(
            input_channels, sequence_length, embed_dim, num_layers
        )
        
        # Risk classification head
        self.risk_classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_risk_levels)
        )
        
        # Auxiliary task: vital sign prediction
        self.vital_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_channels)
        )
        
        self.feature_dim = embed_dim * 2
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        global_features, temporal_features = self.encoder(x, mask)
        
        risk_logits = self.risk_classifier(global_features)
        vital_predictions = self.vital_predictor(global_features)
        
        return {
            'risk_logits': risk_logits,
            'vital_predictions': vital_predictions,
            'features': global_features,
            'temporal_features': temporal_features
        }


class MedicalDataAugmentation:
    """
    Domain-specific data augmentation for physiological signals.
    Includes realistic noise patterns and artifacts commonly found
    in medical sensor data.
    """
    
    def __init__(self, 
                 noise_level: float = 0.05,
                 dropout_prob: float = 0.1,
                 time_warp_factor: float = 0.2):
        
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        self.time_warp_factor = time_warp_factor
        
    def add_baseline_wander(self, signal: torch.Tensor) -> torch.Tensor:
        """Simulate baseline wander common in ECG/PPG signals."""
        
        batch_size, channels, length = signal.shape
        time = torch.linspace(0, 1, length).unsqueeze(0).unsqueeze(0)
        
        # Low frequency sinusoidal drift
        frequency = torch.rand(batch_size, channels, 1) * 0.5
        phase = torch.rand(batch_size, channels, 1) * 2 * np.pi
        wander = 0.1 * torch.sin(2 * np.pi * frequency * time + phase)
        
        return signal + wander.to(signal.device)
    
    def add_motion_artifacts(self, signal: torch.Tensor) -> torch.Tensor:
        """Simulate motion artifacts in wearable sensor data."""
        
        batch_size, channels, length = signal.shape
        
        # Random spike locations
        num_artifacts = torch.randint(0, 5, (batch_size,))
        
        for i in range(batch_size):
            if num_artifacts[i] > 0:
                artifact_positions = torch.randint(0, length, (num_artifacts[i],))
                artifact_magnitudes = torch.randn(num_artifacts[i]) * 0.5
                
                for pos, mag in zip(artifact_positions, artifact_magnitudes):
                    window_size = min(20, length - pos)
                    artifact_window = torch.exp(-torch.arange(window_size).float() / 5)
                    signal[i, :, pos:pos+window_size] += mag * artifact_window
        
        return signal
    
    def apply_time_warping(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply temporal warping to simulate varying heart rates."""
        
        batch_size, channels, length = signal.shape
        
        # Generate smooth warping function
        control_points = torch.randn(batch_size, 10) * self.time_warp_factor
        control_points = torch.nn.functional.interpolate(
            control_points.unsqueeze(1), size=length, mode='linear'
        ).squeeze(1)
        
        # Create warped time indices
        original_indices = torch.arange(length).float()
        warped_indices = original_indices + control_points
        warped_indices = torch.clamp(warped_indices, 0, length - 1)
        
        # Apply warping
        warped_signal = torch.zeros_like(signal)
        for i in range(batch_size):
            for c in range(channels):
                warped_signal[i, c] = torch.nn.functional.interpolate(
                    signal[i:i+1, c:c+1], 
                    size=length, 
                    mode='linear'
                ).squeeze()
        
        return warped_signal
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline."""
        
        if torch.rand(1) > 0.5:
            signal = self.add_baseline_wander(signal)
        
        if torch.rand(1) > 0.5:
            signal = self.add_motion_artifacts(signal)
            
        if torch.rand(1) > 0.7:
            signal = self.apply_time_warping(signal)
        
        # Add Gaussian noise
        noise = torch.randn_like(signal) * self.noise_level
        signal = signal + noise
        
        return signal


class TrainingPipeline:
    """
    Complete training pipeline with adaptive components for medical data.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 num_classes: int = 4,
                 learning_rate: float = 0.001):
        
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        # Initialize adaptive components
        self.optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
        self.scheduler = AdaptiveGradientScheduler(self.optimizer, base_lr=learning_rate)
        self.regularizer = FeatureSpaceRegularizer(
            model.feature_dim, num_classes
        ).to(device)
        
        # Loss functions
        self.risk_criterion = nn.CrossEntropyLoss()
        self.vital_criterion = nn.MSELoss()
        
        # Data augmentation
        self.augmentation = MedicalDataAugmentation()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """Single training step with adaptive components."""
        
        signals, risk_labels, next_vitals = batch
        signals = signals.to(self.device)
        risk_labels = risk_labels.to(self.device)
        next_vitals = next_vitals.to(self.device)
        
        # Apply augmentation
        signals_aug = self.augmentation(signals)
        
        # Forward pass
        outputs = self.model(signals_aug)
        
        # Compute losses
        risk_loss = self.risk_criterion(outputs['risk_logits'], risk_labels)
        vital_loss = self.vital_criterion(outputs['vital_predictions'], next_vitals[:, :, -1])
        reg_loss = self.regularizer(outputs['features'], risk_labels)
        
        total_loss = risk_loss + 0.2 * vital_loss + 0.1 * reg_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Adaptive learning rate update
        current_lr = self.scheduler.compute_adaptive_lr(risk_loss.item(), self.model)
        self.learning_rates.append(current_lr)
        
        # Optimizer step
        self.optimizer.step()
        
        return total_loss.item()
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Validation with metrics computation."""
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                signals, risk_labels, next_vitals = batch
                signals = signals.to(self.device)
                risk_labels = risk_labels.to(self.device)
                
                outputs = self.model(signals)
                loss = self.risk_criterion(outputs['risk_logits'], risk_labels)
                
                total_loss += loss.item()
                predictions = outputs['risk_logits'].argmax(dim=1)
                correct += (predictions == risk_labels).sum().item()
                total += risk_labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        self.model.train()
        return avg_loss, accuracy


def generate_synthetic_medical_data(num_samples: int = 1000, 
                                   sequence_length: int = 256,
                                   num_channels: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic physiological data for testing.
    Channels: ECG, PPG, Respiration, Temperature, Movement
    """
    
    # Base frequencies for different physiological signals
    ecg_freq = 1.2  # Hz (72 bpm)
    ppg_freq = 1.2  # Hz
    resp_freq = 0.25  # Hz (15 breaths/min)
    
    time = torch.linspace(0, 10, sequence_length)
    signals = torch.zeros(num_samples, num_channels, sequence_length)
    
    for i in range(num_samples):
        # ECG-like signal
        heart_rate_var = 1 + 0.1 * torch.randn(1)
        signals[i, 0] = torch.sin(2 * np.pi * ecg_freq * heart_rate_var * time)
        signals[i, 0] += 0.3 * torch.sin(4 * np.pi * ecg_freq * heart_rate_var * time)
        
        # PPG signal
        signals[i, 1] = torch.sin(2 * np.pi * ppg_freq * heart_rate_var * time + 0.5)
        
        # Respiration
        signals[i, 2] = torch.sin(2 * np.pi * resp_freq * time)
        
        # Temperature (slow drift)
        signals[i, 3] = 36.5 + 0.5 * torch.sin(0.1 * time) + 0.1 * torch.randn_like(time)
        
        # Movement/accelerometer
        signals[i, 4] = 0.1 * torch.randn_like(time)
    
    # Risk labels (0: low, 1: moderate, 2: high, 3: critical)
    risk_labels = torch.randint(0, 4, (num_samples,))
    
    # Next vital signs for auxiliary task
    next_vitals = signals.clone()
    next_vitals = torch.roll(next_vitals, -10, dims=2)
    
    return signals, risk_labels, next_vitals


if __name__ == "__main__":
    print("Initializing patient risk stratification system...")
    
    # Model configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PatientRiskStratificationModel(
        input_channels=5,
        sequence_length=256,
        num_risk_levels=4,
        embed_dim=128,
        num_layers=3
    )
    
    # Generate synthetic data for testing
    print("Generating synthetic medical data...")
    train_signals, train_labels, train_vitals = generate_synthetic_medical_data(1000)
    val_signals, val_labels, val_vitals = generate_synthetic_medical_data(200)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_signals, train_labels, train_vitals)
    val_dataset = torch.utils.data.TensorDataset(val_signals, val_labels, val_vitals)
    
    # Use adaptive sampler for imbalanced data
    sampler = ImbalancedDatasetSampler(train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(model, device, num_classes=4)
    
    # Training loop
    print("Starting adaptive training...")
    for epoch in range(10):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            loss = pipeline.train_step(batch)
            epoch_loss += loss
            
            # Update sampler weights based on performance
            if batch_idx % 10 == 0:
                signals, labels, _ = batch
                with torch.no_grad():
                    outputs = model(signals.to(device))
                    sampler.update_weights(outputs['risk_logits'].cpu(), labels)
        
        # Validation
        val_loss, val_acc = pipeline.validate(val_loader)
        
        # Get diagnostics
        diagnostics = pipeline.scheduler.get_diagnostics()
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss/len(train_loader):.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, "
              f"LR={diagnostics['current_lr']:.6f}")
    
    print("\nTraining complete. System operational.")
