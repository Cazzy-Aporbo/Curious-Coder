import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple
import math


class GradientStatisticsMonitor:
    """
    Tracks gradient flow statistics across network layers to inform
    adaptive learning rate decisions. Particularly useful for detecting
    vanishing or exploding gradients in deep architectures.
    """
    
    def __init__(self, window_size: int = 100, percentile_range: Tuple[float, float] = (5, 95)):
        self.window_size = window_size
        self.percentile_range = percentile_range
        self.gradient_history = {}
        self.variance_history = {}
        self.flow_metrics = {}
        
    def update(self, model: nn.Module, global_step: int):
        """Compute and store gradient statistics for current step."""
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data.cpu().numpy().flatten()
                
                if name not in self.gradient_history:
                    self.gradient_history[name] = deque(maxlen=self.window_size)
                    self.variance_history[name] = deque(maxlen=self.window_size)
                    self.flow_metrics[name] = {
                        'mean_magnitude': deque(maxlen=self.window_size),
                        'sparsity': deque(maxlen=self.window_size),
                        'entropy': deque(maxlen=self.window_size)
                    }
                
                grad_magnitude = np.abs(grad_data).mean()
                grad_variance = np.var(grad_data)
                grad_sparsity = np.sum(np.abs(grad_data) < 1e-8) / len(grad_data)
                
                self.gradient_history[name].append(grad_magnitude)
                self.variance_history[name].append(grad_variance)
                self.flow_metrics[name]['mean_magnitude'].append(grad_magnitude)
                self.flow_metrics[name]['sparsity'].append(grad_sparsity)
                
                # Calculate gradient entropy as measure of information content
                if grad_magnitude > 0:
                    normalized_grad = np.abs(grad_data) / (np.sum(np.abs(grad_data)) + 1e-10)
                    entropy = -np.sum(normalized_grad * np.log(normalized_grad + 1e-10))
                    self.flow_metrics[name]['entropy'].append(entropy)
                else:
                    self.flow_metrics[name]['entropy'].append(0)
    
    def get_layer_health_score(self, layer_name: str) -> float:
        """
        Compute health score for a specific layer based on gradient flow.
        Returns value between 0 (unhealthy) and 1 (healthy).
        """
        
        if layer_name not in self.gradient_history or len(self.gradient_history[layer_name]) < 10:
            return 0.5
        
        magnitudes = list(self.gradient_history[layer_name])
        variances = list(self.variance_history[layer_name])
        
        # Check for vanishing gradients
        recent_magnitude = np.mean(magnitudes[-10:])
        vanishing_score = 1.0 - np.exp(-recent_magnitude * 1000)
        
        # Check for exploding gradients
        variance_trend = np.polyfit(range(len(variances[-20:])), variances[-20:], 1)[0] if len(variances) >= 20 else 0
        exploding_score = 1.0 / (1.0 + np.exp(variance_trend * 100))
        
        # Check gradient consistency
        if len(magnitudes) >= 20:
            consistency_score = 1.0 - (np.std(magnitudes[-20:]) / (np.mean(magnitudes[-20:]) + 1e-10))
            consistency_score = max(0, min(1, consistency_score))
        else:
            consistency_score = 0.5
        
        # Combine scores
        health_score = 0.4 * vanishing_score + 0.3 * exploding_score + 0.3 * consistency_score
        return health_score


class AdaptiveGradientScheduler:
    """
    Dynamically adjusts learning rates based on gradient flow statistics
    and training dynamics. Uses statistical measures to detect training
    instabilities and adapt accordingly.
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 base_lr: float = 0.001,
                 warmup_steps: int = 1000,
                 cooldown_factor: float = 0.95,
                 spike_threshold: float = 3.0,
                 recovery_patience: int = 50):
        
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cooldown_factor = cooldown_factor
        self.spike_threshold = spike_threshold
        self.recovery_patience = recovery_patience
        
        self.gradient_monitor = GradientStatisticsMonitor()
        self.loss_history = deque(maxlen=100)
        self.lr_history = []
        self.global_step = 0
        self.spike_detected_step = None
        self.current_lr_multiplier = 1.0
        
    def compute_adaptive_lr(self, loss: float, model: nn.Module) -> float:
        """Calculate adaptive learning rate based on current training state."""
        
        self.global_step += 1
        self.loss_history.append(loss)
        self.gradient_monitor.update(model, self.global_step)
        
        # Warmup phase
        if self.global_step < self.warmup_steps:
            warmup_factor = self.global_step / self.warmup_steps
            return self.base_lr * warmup_factor * self.current_lr_multiplier
        
        # Detect loss spikes
        if len(self.loss_history) >= 20:
            recent_mean = np.mean(list(self.loss_history)[-10:])
            historical_mean = np.mean(list(self.loss_history)[-20:-10])
            
            if historical_mean > 0 and recent_mean / historical_mean > self.spike_threshold:
                self.spike_detected_step = self.global_step
                self.current_lr_multiplier *= 0.5
                print(f"Loss spike detected at step {self.global_step}. Reducing LR multiplier to {self.current_lr_multiplier:.4f}")
        
        # Recovery from spike
        if self.spike_detected_step and self.global_step - self.spike_detected_step > self.recovery_patience:
            self.current_lr_multiplier = min(1.0, self.current_lr_multiplier * 1.1)
            self.spike_detected_step = None
        
        # Compute layer-wise health scores
        health_scores = []
        for name, _ in model.named_parameters():
            score = self.gradient_monitor.get_layer_health_score(name)
            health_scores.append(score)
        
        if health_scores:
            avg_health = np.mean(health_scores)
            health_multiplier = 0.5 + 0.5 * avg_health
        else:
            health_multiplier = 1.0
        
        # Loss-based adaptation
        if len(self.loss_history) >= 50:
            loss_trend = np.polyfit(range(50), list(self.loss_history)[-50:], 1)[0]
            if loss_trend > 0:  # Loss increasing
                trend_multiplier = self.cooldown_factor
            else:  # Loss decreasing
                trend_multiplier = 1.0
        else:
            trend_multiplier = 1.0
        
        # Combine all factors
        adaptive_lr = self.base_lr * self.current_lr_multiplier * health_multiplier * trend_multiplier
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adaptive_lr
        
        self.lr_history.append(adaptive_lr)
        return adaptive_lr
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostic information about training dynamics."""
        
        diagnostics = {
            'global_step': self.global_step,
            'current_lr': self.lr_history[-1] if self.lr_history else self.base_lr,
            'lr_multiplier': self.current_lr_multiplier,
            'recent_loss_mean': np.mean(list(self.loss_history)[-10:]) if len(self.loss_history) >= 10 else None,
            'loss_variance': np.var(list(self.loss_history)) if len(self.loss_history) > 1 else None,
            'layer_health_scores': {}
        }
        
        for name in self.gradient_monitor.gradient_history.keys():
            diagnostics['layer_health_scores'][name] = self.gradient_monitor.get_layer_health_score(name)
        
        return diagnostics


class ImbalancedDatasetSampler(torch.utils.data.Sampler):
    """
    Adaptive sampling strategy for imbalanced datasets that adjusts
    sampling probabilities based on class performance during training.
    """
    
    def __init__(self, 
                 dataset_labels: torch.Tensor,
                 initial_weights: Optional[torch.Tensor] = None,
                 adaptation_rate: float = 0.01,
                 min_sample_rate: float = 0.1):
        
        self.labels = dataset_labels
        self.num_classes = len(torch.unique(dataset_labels))
        self.num_samples = len(dataset_labels)
        self.adaptation_rate = adaptation_rate
        self.min_sample_rate = min_sample_rate
        
        # Initialize class weights
        if initial_weights is None:
            class_counts = torch.bincount(dataset_labels)
            self.class_weights = 1.0 / (class_counts.float() + 1)
            self.class_weights /= self.class_weights.sum()
        else:
            self.class_weights = initial_weights
        
        # Track class performance
        self.class_errors = torch.zeros(self.num_classes)
        self.class_samples_seen = torch.zeros(self.num_classes)
        
        # Precompute sample weights
        self.sample_weights = self.class_weights[self.labels]
        
    def update_weights(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update sampling weights based on class-wise performance."""
        
        with torch.no_grad():
            for cls in range(self.num_classes):
                mask = targets == cls
                if mask.any():
                    class_error = (predictions[mask].argmax(dim=1) != targets[mask]).float().mean()
                    self.class_errors[cls] = 0.9 * self.class_errors[cls] + 0.1 * class_error
                    self.class_samples_seen[cls] += mask.sum()
            
            # Adjust weights based on errors
            if self.class_samples_seen.min() > 100:
                error_weights = self.class_errors / (self.class_errors.sum() + 1e-10)
                self.class_weights = (1 - self.adaptation_rate) * self.class_weights + self.adaptation_rate * error_weights
                
                # Ensure minimum sampling rate
                self.class_weights = torch.maximum(self.class_weights, torch.tensor(self.min_sample_rate / self.num_classes))
                self.class_weights /= self.class_weights.sum()
                
                # Update sample weights
                self.sample_weights = self.class_weights[self.labels]
    
    def __iter__(self):
        indices = torch.multinomial(self.sample_weights, self.num_samples, replacement=True)
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


class FeatureSpaceRegularizer(nn.Module):
    """
    Regularization module that encourages diverse feature representations
    by maximizing entropy in the feature space while maintaining class separation.
    """
    
    def __init__(self, feature_dim: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Class prototype tracking
        self.register_buffer('class_prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Update class prototypes with exponential moving average."""
        
        with torch.no_grad():
            for cls in range(self.num_classes):
                mask = labels == cls
                if mask.any():
                    class_features = features[mask].mean(dim=0)
                    if self.class_counts[cls] == 0:
                        self.class_prototypes[cls] = class_features
                    else:
                        self.class_prototypes[cls] = 0.9 * self.class_prototypes[cls] + 0.1 * class_features
                    self.class_counts[cls] += 1
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss that balances diversity and separation.
        """
        
        batch_size = features.size(0)
        
        # Normalize features
        features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # Diversity loss: maximize entropy of feature correlations
        feature_similarity = torch.mm(features_norm, features_norm.t()) / self.temperature
        feature_probs = torch.nn.functional.softmax(feature_similarity, dim=1)
        diversity_loss = -torch.mean(torch.sum(feature_probs * torch.log(feature_probs + 1e-10), dim=1))
        
        # Separation loss: maximize distance between different classes
        if self.class_counts.sum() > 0:
            separation_loss = 0
            valid_prototypes = self.class_prototypes[self.class_counts > 0]
            
            if len(valid_prototypes) > 1:
                prototype_similarity = torch.mm(valid_prototypes, valid_prototypes.t())
                prototype_similarity.fill_diagonal_(-float('inf'))
                max_similarity = prototype_similarity.max()
                separation_loss = torch.relu(max_similarity + 0.5)
        else:
            separation_loss = torch.tensor(0.0, device=features.device)
        
        # Update prototypes
        self.update_prototypes(features, labels)
        
        # Combined loss
        total_loss = diversity_loss + 0.5 * separation_loss
        
        return total_loss


def create_adaptive_training_system(model: nn.Module, 
                                   train_loader: torch.utils.data.DataLoader,
                                   num_classes: int,
                                   feature_dim: int,
                                   device: str = 'cuda'):
    """
    Initialize complete adaptive training system with all components.
    """
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = AdaptiveGradientScheduler(optimizer, base_lr=0.001)
    regularizer = FeatureSpaceRegularizer(feature_dim, num_classes).to(device)
    
    return optimizer, scheduler, regularizer


# Example usage and testing
if __name__ == "__main__":
    # Simulate a simple model and training scenario
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.classifier = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            features = self.features(x)
            logits = self.classifier(features)
            return logits, features
    
    # Initialize model and components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = AdaptiveGradientScheduler(optimizer, base_lr=0.001)
    regularizer = FeatureSpaceRegularizer(feature_dim=256, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Simulate training step
    print("Testing adaptive gradient scheduler with simulated data...")
    
    for step in range(100):
        # Fake data for testing
        batch_size = 32
        fake_input = torch.randn(batch_size, 784).to(device)
        fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
        
        # Forward pass
        logits, features = model(fake_input)
        ce_loss = criterion(logits, fake_labels)
        reg_loss = regularizer(features, fake_labels)
        total_loss = ce_loss + 0.1 * reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Adaptive learning rate update
        current_lr = scheduler.compute_adaptive_lr(ce_loss.item(), model)
        optimizer.step()
        
        if step % 20 == 0:
            diagnostics = scheduler.get_diagnostics()
            print(f"Step {step}: Loss={ce_loss.item():.4f}, LR={current_lr:.6f}, Health Scores={len(diagnostics['layer_health_scores'])}")
    
    print("\nSystem functional. Ready for deployment.")
