"""
Neural Architecture Evolution: Self-Organizing Network Discovery System
PyTorch implementation that evolves neural architectures through genetic algorithms,
discovering optimal network topologies for given tasks while providing real-time visualization
of the evolutionary process and network performance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from dataclasses import dataclass
import random
import copy
import hashlib
import json
from collections import OrderedDict
import time


@dataclass
class GeneticOperator:
    """Defines genetic operations for neural architecture evolution"""
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_fraction: float = 0.1
    population_size: int = 50
    tournament_size: int = 5
    
    
class DynamicNeuralModule(nn.Module):
    """Self-constructing neural module that builds itself from genetic encoding"""
    
    def __init__(self, gene_sequence: List[int], input_dim: int, output_dim: int):
        super().__init__()
        self.gene_sequence = gene_sequence
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.connections = []
        self.attention_weights = nn.ParameterList()
        
        self._decode_architecture()
        
    def _decode_architecture(self):
        """Decode genetic sequence into neural architecture"""
        gene_idx = 0
        current_dim = self.input_dim
        
        while gene_idx < len(self.gene_sequence) - 3:
            op_type = self.gene_sequence[gene_idx] % 8
            layer_size = 16 + (self.gene_sequence[gene_idx + 1] % 240)
            activation = self.gene_sequence[gene_idx + 2] % 5
            skip_connection = self.gene_sequence[gene_idx + 3] % 2
            
            if op_type == 0:  # Dense layer
                self.layers.append(nn.Linear(current_dim, layer_size))
                current_dim = layer_size
            elif op_type == 1:  # Convolutional (1D)
                kernel_size = 3 + (self.gene_sequence[gene_idx + 1] % 5)
                if current_dim > 10:
                    conv = nn.Conv1d(1, layer_size // 4, kernel_size, padding=kernel_size//2)
                    self.layers.append(conv)
                    current_dim = layer_size // 4
            elif op_type == 2:  # LSTM cell
                lstm = nn.LSTMCell(current_dim, layer_size)
                self.layers.append(lstm)
                current_dim = layer_size
            elif op_type == 3:  # GRU cell
                gru = nn.GRUCell(current_dim, layer_size)
                self.layers.append(gru)
                current_dim = layer_size
            elif op_type == 4:  # Multi-head attention
                if current_dim >= 8:
                    n_heads = 2 + (self.gene_sequence[gene_idx + 1] % 6)
                    embed_dim = (current_dim // n_heads) * n_heads
                    attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
                    self.layers.append(attn)
                    current_dim = embed_dim
            elif op_type == 5:  # Residual block
                if len(self.layers) > 0:
                    self.connections.append((len(self.layers) - 1, skip_connection))
            elif op_type == 6:  # Dropout
                dropout_rate = 0.1 + (self.gene_sequence[gene_idx + 1] % 40) / 100
                self.layers.append(nn.Dropout(dropout_rate))
            elif op_type == 7:  # Batch normalization
                if current_dim > 0:
                    self.layers.append(nn.BatchNorm1d(current_dim))
                    
            gene_idx += 4
        
        # Output projection
        if current_dim != self.output_dim:
            self.layers.append(nn.Linear(current_dim, self.output_dim))
            
    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        intermediates = []
        hidden_states = {}
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, (nn.LSTMCell, nn.GRUCell)):
                batch_size = x.size(0)
                if x.dim() == 3:
                    x = x.mean(dim=1)
                hidden_dim = layer.hidden_size
                h = torch.zeros(batch_size, hidden_dim, device=x.device)
                if isinstance(layer, nn.LSTMCell):
                    c = torch.zeros(batch_size, hidden_dim, device=x.device)
                    h, c = layer(x, (h, c))
                    x = h
                else:
                    x = layer(x, h)
            elif isinstance(layer, nn.MultiheadAttention):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x, _ = layer(x, x, x)
                x = x.squeeze(1) if x.size(1) == 1 else x.mean(dim=1)
            elif isinstance(layer, nn.Conv1d):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = layer(x)
                x = x.mean(dim=-1)
            else:
                x = layer(x)
                
            # Handle skip connections
            for conn_idx, use_skip in self.connections:
                if conn_idx == idx and use_skip and conn_idx in hidden_states:
                    x = x + hidden_states[conn_idx]
                    
            hidden_states[idx] = x
            
            if return_intermediates:
                intermediates.append(x.detach().clone())
                
        if return_intermediates:
            return x, intermediates
        return x


class EvolutionaryArchitectureSearch:
    """Main evolutionary search system for discovering optimal architectures"""
    
    def __init__(self, task_data: Tuple[torch.Tensor, torch.Tensor], 
                 genetic_params: GeneticOperator = GeneticOperator()):
        self.X_train, self.y_train = task_data
        self.genetic_params = genetic_params
        self.population = []
        self.fitness_history = []
        self.best_architecture = None
        self.generation = 0
        
    def initialize_population(self):
        """Create initial random population of architectures"""
        for _ in range(self.genetic_params.population_size):
            gene_length = random.randint(12, 48)
            genes = [random.randint(0, 255) for _ in range(gene_length)]
            self.population.append(genes)
            
    def evaluate_fitness(self, genes: List[int]) -> float:
        """Evaluate fitness of a genetic architecture encoding"""
        try:
            model = DynamicNeuralModule(genes, self.X_train.size(1), self.y_train.size(1))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Quick training for fitness evaluation
            model.train()
            total_loss = 0
            for epoch in range(10):
                optimizer.zero_grad()
                output = model(self.X_train)
                loss = F.mse_loss(output, self.y_train)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Fitness combines performance and complexity penalty
            complexity_penalty = len(genes) * 0.001 + sum(p.numel() for p in model.parameters()) * 0.00001
            fitness = 1.0 / (total_loss / 10 + complexity_penalty + 0.001)
            
            return fitness
        except Exception:
            return 0.0
            
    def mutate(self, genes: List[int]) -> List[int]:
        """Apply mutation to genetic sequence"""
        mutated = genes.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.genetic_params.mutation_rate:
                mutation_type = random.choice(['flip', 'swap', 'insert', 'delete'])
                
                if mutation_type == 'flip':
                    mutated[i] = random.randint(0, 255)
                elif mutation_type == 'swap' and i > 0:
                    mutated[i], mutated[i-1] = mutated[i-1], mutated[i]
                elif mutation_type == 'insert' and len(mutated) < 64:
                    mutated.insert(i, random.randint(0, 255))
                elif mutation_type == 'delete' and len(mutated) > 8:
                    del mutated[i]
                    break
                    
        return mutated
        
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform crossover between two parent architectures"""
        if random.random() > self.genetic_params.crossover_rate:
            return parent1.copy()
            
        crossover_type = random.choice(['single', 'double', 'uniform'])
        
        if crossover_type == 'single':
            point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:point] + parent2[point:]
        elif crossover_type == 'double':
            p1 = random.randint(1, min(len(parent1), len(parent2)) // 2)
            p2 = random.randint(p1 + 1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        else:  # uniform
            child = []
            for i in range(max(len(parent1), len(parent2))):
                if i < len(parent1) and i < len(parent2):
                    child.append(parent1[i] if random.random() < 0.5 else parent2[i])
                elif i < len(parent1):
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
                    
        return child
        
    def tournament_selection(self, fitness_scores: List[float]) -> int:
        """Tournament selection for choosing parents"""
        tournament = random.sample(range(len(self.population)), 
                                 min(self.genetic_params.tournament_size, len(self.population)))
        winner = max(tournament, key=lambda x: fitness_scores[x])
        return winner
        
    def evolve_generation(self):
        """Evolve one generation of architectures"""
        # Evaluate fitness for all individuals
        fitness_scores = [self.evaluate_fitness(genes) for genes in self.population]
        
        # Record best fitness
        best_idx = np.argmax(fitness_scores)
        self.best_architecture = self.population[best_idx]
        self.fitness_history.append(max(fitness_scores))
        
        # Create new population
        new_population = []
        
        # Elite preservation
        n_elite = int(self.genetic_params.population_size * self.genetic_params.elite_fraction)
        elite_indices = np.argsort(fitness_scores)[-n_elite:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
            
        # Generate offspring
        while len(new_population) < self.genetic_params.population_size:
            parent1_idx = self.tournament_selection(fitness_scores)
            parent2_idx = self.tournament_selection(fitness_scores)
            
            child = self.crossover(self.population[parent1_idx], self.population[parent2_idx])
            child = self.mutate(child)
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1
        
        return fitness_scores


class ArchitectureVisualizer:
    """Visualize evolving neural architectures and performance metrics"""
    
    def __init__(self, evolution_system: EvolutionaryArchitectureSearch):
        self.evolution = evolution_system
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
    def visualize_architecture(self, genes: List[int], ax):
        """Create graph visualization of neural architecture"""
        G = nx.DiGraph()
        
        # Decode genes into graph structure
        layer_count = len(genes) // 4
        for i in range(layer_count):
            G.add_node(f"L{i}", layer_type=genes[i*4] % 8)
            if i > 0:
                G.add_edge(f"L{i-1}", f"L{i}")
                
        # Add skip connections
        for i in range(1, layer_count):
            if genes[i*4 + 3] % 3 == 0 and i > 2:
                G.add_edge(f"L{i-2}", f"L{i}", style='dashed')
                
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color nodes by layer type
        colors = []
        for node in G.nodes():
            layer_type = G.nodes[node].get('layer_type', 0)
            color_map = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 
                        3: '#96CEB4', 4: '#FECA57', 5: '#DDA0DD', 
                        6: '#98D8C8', 7: '#F7DC6F'}
            colors.append(color_map.get(layer_type, '#95A5A6'))
            
        nx.draw(G, pos, ax=ax, node_color=colors, node_size=500, 
                with_labels=True, font_size=8, font_weight='bold',
                arrows=True, edge_color='#34495E', width=1.5)
        
        ax.set_title(f"Architecture Graph (Generation {self.evolution.generation})", 
                    fontsize=10, fontweight='bold')
        
    def plot_fitness_history(self, ax):
        """Plot fitness evolution over generations"""
        if len(self.evolution.fitness_history) > 0:
            generations = range(len(self.evolution.fitness_history))
            ax.plot(generations, self.evolution.fitness_history, 
                   color='#3498DB', linewidth=2)
            ax.fill_between(generations, 0, self.evolution.fitness_history, 
                           alpha=0.3, color='#3498DB')
            ax.set_xlabel('Generation', fontsize=10)
            ax.set_ylabel('Best Fitness', fontsize=10)
            ax.set_title('Fitness Evolution', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
    def plot_population_diversity(self, ax):
        """Visualize genetic diversity in population"""
        if self.evolution.population:
            # Calculate genetic signatures
            signatures = []
            for genes in self.evolution.population:
                sig = hashlib.md5(str(genes).encode()).hexdigest()
                signatures.append(int(sig[:8], 16))
                
            unique_count = len(set(signatures))
            diversity_score = unique_count / len(signatures)
            
            ax.hist(signatures, bins=20, color='#9B59B6', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Genetic Signature', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Population Diversity (Score: {diversity_score:.2f})', 
                        fontsize=10, fontweight='bold')
            
    def plot_layer_distribution(self, ax):
        """Show distribution of layer types in best architecture"""
        if self.evolution.best_architecture:
            layer_types = [g % 8 for g in self.evolution.best_architecture[::4]]
            type_names = ['Dense', 'Conv1D', 'LSTM', 'GRU', 'Attention', 
                         'Residual', 'Dropout', 'BatchNorm']
            type_counts = [layer_types.count(i) for i in range(8)]
            
            colors = plt.cm.Set3(range(8))
            ax.bar(type_names, type_counts, color=colors)
            ax.set_xlabel('Layer Type', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title('Layer Type Distribution', fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
    def update_visualization(self):
        """Update all visualization components"""
        self.fig.clear()
        
        # Architecture graph
        ax1 = self.fig.add_subplot(self.gs[0, :2])
        if self.evolution.best_architecture:
            self.visualize_architecture(self.evolution.best_architecture, ax1)
            
        # Fitness history
        ax2 = self.fig.add_subplot(self.gs[1, 0])
        self.plot_fitness_history(ax2)
        
        # Population diversity
        ax3 = self.fig.add_subplot(self.gs[1, 1])
        self.plot_population_diversity(ax3)
        
        # Layer distribution
        ax4 = self.fig.add_subplot(self.gs[1, 2])
        self.plot_layer_distribution(ax4)
        
        # Performance metrics
        ax5 = self.fig.add_subplot(self.gs[2, :])
        self.plot_performance_comparison(ax5)
        
        plt.tight_layout()
        
    def plot_performance_comparison(self, ax):
        """Compare performance across different architectures"""
        if len(self.evolution.fitness_history) > 5:
            recent_generations = self.evolution.fitness_history[-10:]
            x = range(len(recent_generations))
            
            ax.plot(x, recent_generations, marker='o', color='#E74C3C', 
                   linewidth=2, markersize=6, label='Best Fitness')
            
            # Add moving average
            if len(recent_generations) >= 3:
                moving_avg = np.convolve(recent_generations, np.ones(3)/3, mode='valid')
                ax.plot(range(1, len(moving_avg)+1), moving_avg, 
                       color='#27AE60', linewidth=2, alpha=0.7, label='Moving Avg')
                
            ax.set_xlabel('Recent Generations', fontsize=10)
            ax.set_ylabel('Fitness Score', fontsize=10)
            ax.set_title('Recent Performance Trends', fontsize=10, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)


class AdaptiveMetaLearner:
    """Meta-learning component that adapts search strategy based on problem characteristics"""
    
    def __init__(self):
        self.problem_embeddings = {}
        self.strategy_memory = []
        self.meta_network = self._build_meta_network()
        
    def _build_meta_network(self):
        """Build meta-learning network for strategy adaptation"""
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.Softmax(dim=-1)
        )
        
    def analyze_problem(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Extract problem characteristics for meta-learning"""
        characteristics = {
            'input_dim': X.size(1),
            'output_dim': y.size(1),
            'sample_size': X.size(0),
            'input_variance': X.var().item(),
            'output_variance': y.var().item(),
            'input_sparsity': (X == 0).float().mean().item(),
            'correlation': torch.corrcoef(torch.stack([X.mean(1), y.mean(1)]))[0, 1].item(),
            'nonlinearity': self._estimate_nonlinearity(X, y)
        }
        return characteristics
        
    def _estimate_nonlinearity(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Estimate problem nonlinearity"""
        # Simple linear fit
        X_flat = X.view(X.size(0), -1)
        linear_pred = X_flat @ torch.linalg.pinv(X_flat) @ y
        linear_error = F.mse_loss(linear_pred, y)
        
        # Polynomial features
        X_poly = torch.cat([X_flat, X_flat**2], dim=1)
        poly_pred = X_poly @ torch.linalg.pinv(X_poly) @ y
        poly_error = F.mse_loss(poly_pred, y)
        
        nonlinearity = (linear_error - poly_error) / (linear_error + 1e-6)
        return nonlinearity.item()
        
    def recommend_strategy(self, problem_chars: Dict[str, float]) -> GeneticOperator:
        """Recommend evolutionary strategy based on problem characteristics"""
        # Convert characteristics to tensor
        char_vector = torch.tensor([
            problem_chars.get('input_dim', 10) / 100,
            problem_chars.get('output_dim', 1) / 10,
            problem_chars.get('sample_size', 100) / 1000,
            problem_chars.get('input_variance', 1.0),
            problem_chars.get('output_variance', 1.0),
            problem_chars.get('input_sparsity', 0.0),
            problem_chars.get('correlation', 0.0),
            problem_chars.get('nonlinearity', 0.5)
        ])
        
        # Pad to expected input size
        char_vector = F.pad(char_vector, (0, 12), value=0.0)
        
        # Get strategy recommendation
        strategy_probs = self.meta_network(char_vector)
        
        # Map to genetic parameters
        mutation_rate = 0.05 + strategy_probs[0].item() * 0.3
        crossover_rate = 0.5 + strategy_probs[1].item() * 0.4
        elite_fraction = 0.05 + strategy_probs[2].item() * 0.2
        population_size = int(20 + strategy_probs[3].item() * 80)
        tournament_size = int(3 + strategy_probs[4].item() * 7)
        
        return GeneticOperator(
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_fraction=elite_fraction,
            population_size=population_size,
            tournament_size=tournament_size
        )


def run_evolutionary_search(X: torch.Tensor, y: torch.Tensor, 
                          generations: int = 50, 
                          visualize: bool = True) -> DynamicNeuralModule:
    """Main function to run evolutionary architecture search"""
    
    print("Initializing Neural Architecture Evolution System")
    print("=" * 60)
    
    # Meta-learning analysis
    meta_learner = AdaptiveMetaLearner()
    problem_chars = meta_learner.analyze_problem(X, y)
    print("\nProblem Characteristics:")
    for key, value in problem_chars.items():
        print(f"  {key}: {value:.4f}")
        
    # Get recommended strategy
    genetic_params = meta_learner.recommend_strategy(problem_chars)
    print(f"\nAdapted Genetic Parameters:")
    print(f"  Mutation Rate: {genetic_params.mutation_rate:.3f}")
    print(f"  Crossover Rate: {genetic_params.crossover_rate:.3f}")
    print(f"  Population Size: {genetic_params.population_size}")
    
    # Initialize evolution
    evolution = EvolutionaryArchitectureSearch((X, y), genetic_params)
    evolution.initialize_population()
    
    # Setup visualization
    if visualize:
        visualizer = ArchitectureVisualizer(evolution)
        plt.ion()
    
    # Evolution loop
    print("\nStarting Evolution:")
    print("-" * 40)
    
    for gen in range(generations):
        fitness_scores = evolution.evolve_generation()
        
        if gen % 5 == 0:
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            print(f"Generation {gen:3d} | Best: {best_fitness:8.4f} | Avg: {avg_fitness:8.4f}")
            
            if visualize:
                visualizer.update_visualization()
                plt.pause(0.1)
    
    # Create final model
    print("\nEvolution Complete!")
    print("=" * 60)
    
    final_model = DynamicNeuralModule(
        evolution.best_architecture, 
        X.size(1), 
        y.size(1)
    )
    
    # Train final model properly
    print("Training Final Architecture...")
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    final_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = final_model(X)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f}")
    
    print("\nArchitecture Summary:")
    total_params = sum(p.numel() for p in final_model.parameters())
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Number of Layers: {len(final_model.layers)}")
    print(f"  Gene Sequence Length: {len(evolution.best_architecture)}")
    
    if visualize:
        plt.ioff()
        plt.show()
    
    return final_model


if __name__ == "__main__":
    # Generate synthetic task for demonstration
    torch.manual_seed(42)
    
    # Complex nonlinear problem
    X = torch.randn(200, 20)
    hidden = torch.tanh(X @ torch.randn(20, 50))
    y = torch.sigmoid(hidden @ torch.randn(50, 10))
    
    # Run evolutionary search
    best_model = run_evolutionary_search(X, y, generations=30, visualize=True)
    
    # Test discovered architecture
    print("\nTesting Discovered Architecture:")
    test_X = torch.randn(50, 20)
    with torch.no_grad():
        predictions = best_model(test_X)
        print(f"Output shape: {predictions.shape}")
        print(f"Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")
