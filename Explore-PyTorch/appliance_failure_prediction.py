import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import math


class ApplianceSignatureEncoder(nn.Module):
    """
    Encodes operational signatures from appliance sensors including
    vibration patterns, power consumption, temperature readings, and
    acoustic signatures to identify pre-failure patterns.
    """
    
    def __init__(self, 
                 sensor_channels: int = 8,
                 signature_length: int = 512,
                 hidden_dim: int = 256,
                 num_appliance_types: int = 15):
        super().__init__()
        
        self.sensor_channels = sensor_channels
        self.signature_length = signature_length
        
        # Multi-scale convolutions for different frequency patterns
        self.conv_short = nn.Conv1d(sensor_channels, hidden_dim // 4, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(sensor_channels, hidden_dim // 4, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(sensor_channels, hidden_dim // 4, kernel_size=15, padding=7)
        self.conv_very_long = nn.Conv1d(sensor_channels, hidden_dim // 4, kernel_size=31, padding=15)
        
        # Batch normalization for each scale
        self.bn_short = nn.BatchNorm1d(hidden_dim // 4)
        self.bn_medium = nn.BatchNorm1d(hidden_dim // 4)
        self.bn_long = nn.BatchNorm1d(hidden_dim // 4)
        self.bn_very_long = nn.BatchNorm1d(hidden_dim // 4)
        
        # Appliance type embedding
        self.appliance_embedding = nn.Embedding(num_appliance_types, hidden_dim)
        
        # Temporal processing
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # Pattern memory bank for known failure signatures
        self.register_buffer('failure_patterns', torch.zeros(100, hidden_dim))
        self.register_buffer('pattern_counts', torch.zeros(100))
        self.pattern_pointer = 0
        
    def extract_multiscale_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features at multiple temporal scales."""
        
        short = F.relu(self.bn_short(self.conv_short(x)))
        medium = F.relu(self.bn_medium(self.conv_medium(x)))
        long = F.relu(self.bn_long(self.conv_long(x)))
        very_long = F.relu(self.bn_very_long(self.conv_very_long(x)))
        
        # Concatenate multi-scale features
        features = torch.cat([short, medium, long, very_long], dim=1)
        return features
    
    def update_failure_patterns(self, features: torch.Tensor, is_failure: torch.Tensor):
        """Update memory bank with confirmed failure patterns."""
        
        with torch.no_grad():
            failure_indices = torch.where(is_failure)[0]
            for idx in failure_indices:
                if self.pattern_counts[self.pattern_pointer] == 0:
                    self.failure_patterns[self.pattern_pointer] = features[idx]
                else:
                    # Exponential moving average
                    self.failure_patterns[self.pattern_pointer] = (
                        0.9 * self.failure_patterns[self.pattern_pointer] + 
                        0.1 * features[idx]
                    )
                self.pattern_counts[self.pattern_pointer] += 1
                self.pattern_pointer = (self.pattern_pointer + 1) % 100
    
    def forward(self, x: torch.Tensor, appliance_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Extract multi-scale features
        features = self.extract_multiscale_features(x)
        
        # Transpose for LSTM processing
        features = features.transpose(1, 2)
        
        # Add appliance type information
        appliance_info = self.appliance_embedding(appliance_type).unsqueeze(1)
        appliance_info = appliance_info.expand(-1, features.size(1), -1)
        features = features + appliance_info
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Global features from final hidden state
        global_features = hidden[-1]
        
        # Compute similarity to known failure patterns
        if self.pattern_counts.sum() > 0:
            valid_patterns = self.failure_patterns[self.pattern_counts > 0]
            pattern_similarity = F.cosine_similarity(
                global_features.unsqueeze(1),
                valid_patterns.unsqueeze(0),
                dim=2
            )
            max_similarity, _ = pattern_similarity.max(dim=1)
        else:
            max_similarity = torch.zeros(batch_size, device=x.device)
        
        return global_features, max_similarity


class ServiceEventPredictor(nn.Module):
    """
    Predicts service events including failure types, required parts,
    and estimated repair time based on appliance history and current state.
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 num_failure_modes: int = 50,
                 num_parts: int = 500,
                 history_length: int = 30):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.history_length = history_length
        
        # Process service history
        self.history_encoder = nn.LSTM(
            feature_dim + 4,  # features + metadata
            feature_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Failure mode prediction
        self.failure_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_failure_modes)
        )
        
        # Parts requirement prediction (multi-label)
        self.parts_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + num_failure_modes, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_parts)
        )
        
        # Repair time estimation
        self.time_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2 + num_failure_modes, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Cost estimation
        self.cost_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2 + num_parts, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )
    
    def forward(self, 
                current_features: torch.Tensor,
                service_history: torch.Tensor,
                history_metadata: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        batch_size = current_features.size(0)
        
        # Encode service history
        history_input = torch.cat([service_history, history_metadata], dim=2)
        history_out, (history_hidden, _) = self.history_encoder(history_input)
        history_features = history_hidden[-1]
        
        # Combine current and historical features
        combined_features = torch.cat([current_features, history_features], dim=1)
        
        # Predict failure modes
        failure_logits = self.failure_classifier(combined_features)
        failure_probs = torch.softmax(failure_logits, dim=1)
        
        # Predict required parts (conditioned on failure mode)
        parts_input = torch.cat([combined_features, failure_probs], dim=1)
        parts_logits = self.parts_predictor(parts_input)
        parts_probs = torch.sigmoid(parts_logits)
        
        # Estimate repair time
        time_input = torch.cat([combined_features, failure_probs], dim=1)
        repair_time = F.softplus(self.time_estimator(time_input))
        
        # Estimate cost
        cost_input = torch.cat([combined_features, parts_probs], dim=1)
        repair_cost = F.softplus(self.cost_estimator(cost_input))
        
        return {
            'failure_logits': failure_logits,
            'failure_probs': failure_probs,
            'parts_probs': parts_probs,
            'repair_time': repair_time,
            'repair_cost': repair_cost
        }


class TechnicianSkillMatcher(nn.Module):
    """
    Matches service calls to technicians based on skill compatibility,
    geographic proximity, and historical performance on similar repairs.
    """
    
    def __init__(self,
                 num_technicians: int = 50,
                 num_skills: int = 100,
                 geographic_zones: int = 20,
                 embedding_dim: int = 128):
        super().__init__()
        
        # Technician embeddings
        self.technician_embeddings = nn.Embedding(num_technicians, embedding_dim)
        self.skill_embeddings = nn.Embedding(num_skills, embedding_dim)
        self.zone_embeddings = nn.Embedding(geographic_zones, embedding_dim // 2)
        
        # Performance history encoder
        self.performance_encoder = nn.LSTM(
            embedding_dim + 3,  # embedding + success_rate, avg_time, customer_rating
            embedding_dim,
            batch_first=True
        )
        
        # Matching network
        self.matcher = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        # Success probability predictor
        self.success_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def compute_skill_match(self, 
                           required_skills: torch.Tensor,
                           technician_skills: torch.Tensor) -> torch.Tensor:
        """Compute skill compatibility score."""
        
        # Embed skills
        req_skill_embed = self.skill_embeddings(required_skills)
        tech_skill_embed = self.skill_embeddings(technician_skills)
        
        # Compute similarity
        req_skill_avg = req_skill_embed.mean(dim=1)
        tech_skill_avg = tech_skill_embed.mean(dim=1)
        
        similarity = F.cosine_similarity(
            req_skill_avg.unsqueeze(1),
            tech_skill_avg.unsqueeze(0),
            dim=2
        )
        
        return similarity
    
    def forward(self,
                job_features: torch.Tensor,
                required_skills: torch.Tensor,
                technician_ids: torch.Tensor,
                technician_skills: torch.Tensor,
                technician_zones: torch.Tensor,
                job_zone: torch.Tensor,
                performance_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        batch_size = job_features.size(0)
        num_techs = technician_ids.size(0)
        
        # Embed technicians
        tech_embeddings = self.technician_embeddings(technician_ids)
        
        # Encode performance history
        perf_out, (perf_hidden, _) = self.performance_encoder(performance_history)
        tech_performance = perf_hidden[-1]
        
        # Compute skill compatibility
        skill_match = self.compute_skill_match(required_skills, technician_skills)
        
        # Zone proximity
        job_zone_embed = self.zone_embeddings(job_zone)
        tech_zone_embed = self.zone_embeddings(technician_zones)
        zone_similarity = F.cosine_similarity(
            job_zone_embed.unsqueeze(1),
            tech_zone_embed.unsqueeze(0),
            dim=2
        )
        
        # Combine all factors for matching score
        match_scores = []
        success_probs = []
        
        for i in range(batch_size):
            job_feat_expanded = job_features[i].unsqueeze(0).expand(num_techs, -1)
            
            match_input = torch.cat([
                job_feat_expanded,
                tech_embeddings,
                tech_performance,
                skill_match[i].unsqueeze(1).expand(-1, tech_embeddings.size(1))
            ], dim=1)
            
            scores = self.matcher(match_input).squeeze()
            match_scores.append(scores)
            
            # Predict success probability
            success_input = torch.cat([
                job_feat_expanded,
                tech_embeddings,
                tech_performance
            ], dim=1)
            
            success = self.success_predictor(success_input).squeeze()
            success_probs.append(success)
        
        match_scores = torch.stack(match_scores)
        success_probs = torch.stack(success_probs)
        
        # Apply zone penalty for distant technicians
        distance_penalty = 1.0 - zone_similarity
        adjusted_scores = match_scores - 0.3 * distance_penalty
        
        return {
            'match_scores': adjusted_scores,
            'success_probability': success_probs,
            'skill_compatibility': skill_match,
            'zone_proximity': zone_similarity
        }


class InventoryOptimizer(nn.Module):
    """
    Optimizes parts inventory based on failure predictions, seasonal patterns,
    and supply chain constraints.
    """
    
    def __init__(self,
                 num_parts: int = 500,
                 num_warehouses: int = 10,
                 feature_dim: int = 128):
        super().__init__()
        
        self.num_parts = num_parts
        self.num_warehouses = num_warehouses
        
        # Part embeddings
        self.part_embeddings = nn.Embedding(num_parts, feature_dim)
        self.warehouse_embeddings = nn.Embedding(num_warehouses, feature_dim // 2)
        
        # Demand forecasting
        self.demand_lstm = nn.LSTM(
            feature_dim + 12,  # features + monthly indicators
            feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.demand_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_parts)
        )
        
        # Reorder point calculator
        self.reorder_calculator = nn.Sequential(
            nn.Linear(feature_dim + 4, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Softplus()
        )
        
        # Cross-warehouse transfer recommender
        self.transfer_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )
        
    def forecast_demand(self,
                       historical_usage: torch.Tensor,
                       seasonal_features: torch.Tensor,
                       failure_predictions: torch.Tensor) -> torch.Tensor:
        """Forecast part demand for next period."""
        
        # Combine historical usage with seasonal patterns
        lstm_input = torch.cat([historical_usage, seasonal_features], dim=2)
        lstm_out, (hidden, _) = self.demand_lstm(lstm_input)
        
        # Add failure prediction influence
        demand_features = torch.cat([hidden[-1], failure_predictions], dim=1)
        demand_forecast = F.softplus(self.demand_predictor(demand_features))
        
        return demand_forecast
    
    def calculate_reorder_points(self,
                                current_stock: torch.Tensor,
                                demand_forecast: torch.Tensor,
                                lead_times: torch.Tensor,
                                service_level: float = 0.95) -> torch.Tensor:
        """Calculate optimal reorder points for each part."""
        
        batch_size, num_parts = current_stock.shape
        reorder_points = torch.zeros_like(current_stock)
        
        for i in range(num_parts):
            part_embed = self.part_embeddings(torch.tensor(i))
            part_features = torch.cat([
                part_embed,
                demand_forecast[:, i:i+1],
                lead_times[:, i:i+1],
                torch.full((batch_size, 1), service_level),
                current_stock[:, i:i+1]
            ], dim=1)
            
            reorder_points[:, i] = self.reorder_calculator(part_features).squeeze()
        
        return reorder_points
    
    def recommend_transfers(self,
                           warehouse_stocks: torch.Tensor,
                           warehouse_demands: torch.Tensor) -> torch.Tensor:
        """Recommend inter-warehouse transfers to balance inventory."""
        
        num_warehouses = warehouse_stocks.size(0)
        transfer_matrix = torch.zeros(num_warehouses, num_warehouses, self.num_parts)
        
        for source in range(num_warehouses):
            for target in range(num_warehouses):
                if source == target:
                    continue
                
                source_embed = self.warehouse_embeddings(torch.tensor(source))
                target_embed = self.warehouse_embeddings(torch.tensor(target))
                
                # Calculate transfer scores for each part
                for part in range(self.num_parts):
                    source_surplus = warehouse_stocks[source, part] - warehouse_demands[source, part]
                    target_deficit = warehouse_demands[target, part] - warehouse_stocks[target, part]
                    
                    if source_surplus > 0 and target_deficit > 0:
                        transfer_features = torch.cat([
                            source_embed,
                            target_embed,
                            torch.tensor([source_surplus]),
                            torch.tensor([target_deficit])
                        ])
                        
                        score = self.transfer_scorer(transfer_features)
                        transfer_qty = min(source_surplus, target_deficit) * torch.sigmoid(score)
                        transfer_matrix[source, target, part] = transfer_qty
        
        return transfer_matrix


class ApplianceRepairSystem(nn.Module):
    """
    Complete system integrating all components for appliance repair operations.
    """
    
    def __init__(self,
                 config: Dict):
        super().__init__()
        
        self.signature_encoder = ApplianceSignatureEncoder(
            sensor_channels=config['sensor_channels'],
            signature_length=config['signature_length'],
            hidden_dim=config['hidden_dim'],
            num_appliance_types=config['num_appliance_types']
        )
        
        self.service_predictor = ServiceEventPredictor(
            feature_dim=config['hidden_dim'],
            num_failure_modes=config['num_failure_modes'],
            num_parts=config['num_parts']
        )
        
        self.technician_matcher = TechnicianSkillMatcher(
            num_technicians=config['num_technicians'],
            num_skills=config['num_skills'],
            geographic_zones=config['geographic_zones']
        )
        
        self.inventory_optimizer = InventoryOptimizer(
            num_parts=config['num_parts'],
            num_warehouses=config['num_warehouses']
        )
        
        # Customer satisfaction predictor
        self.satisfaction_predictor = nn.Sequential(
            nn.Linear(config['hidden_dim'] * 2 + 4, config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], 5)  # 1-5 star rating
        )
        
        # Warranty claim likelihood
        self.warranty_predictor = nn.Sequential(
            nn.Linear(config['hidden_dim'] + config['num_failure_modes'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Encode appliance signatures
        features, failure_similarity = self.signature_encoder(
            batch['sensor_data'],
            batch['appliance_type']
        )
        
        # Predict service events
        service_predictions = self.service_predictor(
            features,
            batch['service_history'],
            batch['history_metadata']
        )
        
        # Match technicians
        technician_matches = self.technician_matcher(
            features,
            batch['required_skills'],
            batch['technician_ids'],
            batch['technician_skills'],
            batch['technician_zones'],
            batch['job_zone'],
            batch['performance_history']
        )
        
        # Forecast inventory needs
        demand_forecast = self.inventory_optimizer.forecast_demand(
            batch['historical_usage'],
            batch['seasonal_features'],
            service_predictions['parts_probs']
        )
        
        # Predict customer satisfaction
        satisfaction_features = torch.cat([
            features,
            technician_matches['success_probability'].max(dim=1)[0].unsqueeze(1),
            service_predictions['repair_time'],
            service_predictions['repair_cost'],
            batch['wait_time']
        ], dim=1)
        
        satisfaction_logits = self.satisfaction_predictor(satisfaction_features)
        
        # Predict warranty claims
        warranty_features = torch.cat([
            features,
            service_predictions['failure_probs']
        ], dim=1)
        
        warranty_probability = self.warranty_predictor(warranty_features)
        
        return {
            'appliance_features': features,
            'failure_similarity': failure_similarity,
            'service_predictions': service_predictions,
            'technician_matches': technician_matches,
            'demand_forecast': demand_forecast,
            'satisfaction_logits': satisfaction_logits,
            'warranty_probability': warranty_probability
        }


class AdaptiveSchedulingLoss(nn.Module):
    """
    Custom loss function that balances multiple objectives in repair scheduling.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 gamma: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Weight for failure prediction
        self.beta = beta    # Weight for parts prediction
        self.gamma = gamma  # Weight for time estimation
        
    def forward(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        # Failure mode prediction loss
        failure_loss = F.cross_entropy(
            predictions['service_predictions']['failure_logits'],
            targets['failure_mode']
        )
        
        # Parts requirement loss (binary cross-entropy for multi-label)
        parts_loss = F.binary_cross_entropy(
            predictions['service_predictions']['parts_probs'],
            targets['required_parts']
        )
        
        # Repair time estimation loss
        time_loss = F.smooth_l1_loss(
            predictions['service_predictions']['repair_time'].squeeze(),
            targets['actual_repair_time']
        )
        
        # Customer satisfaction loss
        satisfaction_loss = F.cross_entropy(
            predictions['satisfaction_logits'],
            targets['satisfaction_rating']
        )
        
        # Warranty prediction loss
        warranty_loss = F.binary_cross_entropy(
            predictions['warranty_probability'].squeeze(),
            targets['warranty_claimed'].float()
        )
        
        # Weighted combination
        total_loss = (
            self.alpha * failure_loss +
            self.beta * parts_loss +
            self.gamma * time_loss +
            0.2 * satisfaction_loss +
            0.1 * warranty_loss
        )
        
        return total_loss


def generate_synthetic_appliance_data(batch_size: int = 32) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic data for testing the appliance repair system.
    """
    
    # Sensor data (vibration, temperature, power, acoustic, pressure, humidity, runtime, cycles)
    sensor_data = torch.randn(batch_size, 8, 512)
    
    # Add realistic patterns
    for i in range(batch_size):
        # Simulate compressor cycling
        cycle_freq = 0.1 + torch.rand(1) * 0.05
        sensor_data[i, 0] += torch.sin(torch.arange(512) * cycle_freq)
        
        # Temperature drift
        sensor_data[i, 1] += torch.linspace(0, 2, 512)
        
        # Power consumption spikes
        spike_positions = torch.randint(0, 512, (5,))
        for pos in spike_positions:
            sensor_data[i, 2, pos] += 3.0
    
    # Appliance types (0: refrigerator, 1: washer, 2: dryer, etc.)
    appliance_type = torch.randint(0, 15, (batch_size,))
    
    # Service history
    history_length = 30
    service_history = torch.randn(batch_size, history_length, 256)
    history_metadata = torch.randn(batch_size, history_length, 4)
    
    # Required skills for current job
    required_skills = torch.randint(0, 100, (batch_size, 5))
    
    # Technician data
    num_technicians = 20
    technician_ids = torch.arange(num_technicians)
    technician_skills = torch.randint(0, 100, (num_technicians, 10))
    technician_zones = torch.randint(0, 20, (num_technicians,))
    job_zone = torch.randint(0, 20, (batch_size,))
    performance_history = torch.randn(num_technicians, 50, 131)
    
    # Inventory data
    historical_usage = torch.randn(batch_size, 12, 128)
    seasonal_features = torch.randn(batch_size, 12, 12)
    
    # Additional features
    wait_time = torch.rand(batch_size, 1) * 48  # Hours
    
    # Targets
    failure_mode = torch.randint(0, 50, (batch_size,))
    required_parts = torch.randint(0, 2, (batch_size, 500)).float()
    actual_repair_time = torch.rand(batch_size) * 8  # Hours
    satisfaction_rating = torch.randint(0, 5, (batch_size,))
    warranty_claimed = torch.randint(0, 2, (batch_size,))
    
    return {
        'sensor_data': sensor_data,
        'appliance_type': appliance_type,
        'service_history': service_history,
        'history_metadata': history_metadata,
        'required_skills': required_skills,
        'technician_ids': technician_ids,
        'technician_skills': technician_skills,
        'technician_zones': technician_zones,
        'job_zone': job_zone,
        'performance_history': performance_history,
        'historical_usage': historical_usage,
        'seasonal_features': seasonal_features,
        'wait_time': wait_time,
        'failure_mode': failure_mode,
        'required_parts': required_parts,
        'actual_repair_time': actual_repair_time,
        'satisfaction_rating': satisfaction_rating,
        'warranty_claimed': warranty_claimed
    }


if __name__ == "__main__":
    print("Initializing appliance repair optimization system...")
    
    config = {
        'sensor_channels': 8,
        'signature_length': 512,
        'hidden_dim': 256,
        'num_appliance_types': 15,
        'num_failure_modes': 50,
        'num_parts': 500,
        'num_technicians': 50,
        'num_skills': 100,
        'geographic_zones': 20,
        'num_warehouses': 10
    }
    
    model = ApplianceRepairSystem(config)
    
    # Test with synthetic data
    print("Generating synthetic appliance data...")
    batch = generate_synthetic_appliance_data(batch_size=16)
    
    print("Running forward pass...")
    outputs = model(batch)
    
    # Initialize loss and optimizer
    criterion = AdaptiveSchedulingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training loop
    print("Starting training simulation...")
    for epoch in range(5):
        optimizer.zero_grad()
        
        predictions = model(batch)
        
        targets = {
            'failure_mode': batch['failure_mode'],
            'required_parts': batch['required_parts'],
            'actual_repair_time': batch['actual_repair_time'],
            'satisfaction_rating': batch['satisfaction_rating'],
            'warranty_claimed': batch['warranty_claimed']
        }
        
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Display some predictions
        if epoch == 4:
            print(f"\nSample predictions:")
            print(f"Failure similarity scores: {predictions['failure_similarity'][:3].detach().numpy()}")
            print(f"Top predicted failure mode: {predictions['service_predictions']['failure_probs'][0].argmax().item()}")
            print(f"Estimated repair time: {predictions['service_predictions']['repair_time'][0].item():.2f} hours")
            print(f"Warranty claim probability: {predictions['warranty_probability'][0].item():.3f}")
    
    print("\nSystem operational and ready for deployment.")
