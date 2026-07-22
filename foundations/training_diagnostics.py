import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if isinstance(layer,nn.Linear):
                    dead_fraction = (x <= 0).all(dim=0).float().mean().item()
                    stats.append({
                        "mean":round(torch.mean(x).item(),4),
                        "std":round(torch.std(x).item(),4),
                        "dead_fraction":round(dead_fraction,4)
                    })
        print(stats)
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        prediction = model(x)
        criterion = nn.MSELoss()
        loss = criterion(prediction,y)
        loss.backward()
        stats = []
        for layer in model:
            if isinstance(layer,nn.Linear):
                grad = layer.weight.grad
                stats.append({
                    "mean":round(torch.mean(grad).item(),4),
                    "std":round(torch.std(grad).item(),4),
                    "norm":round(torch.norm(grad).item(),4)
                })
        print(stats)
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for stat in activation_stats:
            if stat["dead_fraction"]>0.5:
                print("dead_neurons")
                return "dead_neurons"
        for stat in gradient_stats:
            if stat["norm"]>10:
                print("exploding_gradients")
                return "exploding_gradients"
        for stat in gradient_stats:
            if stat["norm"]<1e-5:
                print("vanishing_gradients")
                return "vanishing_gradients"
        print("healthy")
        return "healthy"
