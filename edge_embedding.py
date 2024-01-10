import numpy as np

import torch
from torch import nn


class GaussBasisExpansion(nn.Module):
    def __init__(self, mu, sigma, **kwargs):
        super().__init__(**kwargs)
        # shape: (1, len(mu))
        mu = torch.unsqueeze(torch.FloatTensor(mu), 0)
        self.register_buffer("mu", mu)
        # shape: (1, len(sigma))
        sigma = torch.unsqueeze(torch.FloatTensor(sigma), 0)
        self.register_buffer("sigma", sigma)

    @classmethod
    def from_bounds(cls, n: int, low: float, high: float, variance: float = 1.0):
        mus = np.linspace(low, high, num=n + 1)
        var = np.diff(mus)
        mus = mus[1:]
        return cls(mus, np.sqrt(var * variance))

    @classmethod
    def from_bounds_log(
        cls, n: int, low: float, high: float, base: float = 32, variance: float = 1
    ):
        mus = (np.logspace(0, 1, num=n + 1, base=base) - 1) / (base - 1) * (high - low) + low
        var = np.diff(mus)
        mus = mus[1:]
        return cls(mus, np.sqrt(var * variance))

    def forward(self, x, **kwargs):
        return torch.exp(-torch.pow(x - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))


class EdgeEmbedding(nn.Module):
    def __init__(
        self,
        bins_distance: int = 32,
        max_distance: float = 5.0,
        distance_log_base: float = 1.0,
        gaussian_width_distance: float = 1.0,
        bins_voronoi_area: int = None,
        max_voronoi_area: float = 32.0,
        voronoi_area_log_base: float = 1.0,
        gaussian_width_voronoi_area: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        if distance_log_base == 1.0:
            self.distance_embedding = GaussBasisExpansion.from_bounds(
                bins_distance, 0.0, max_distance, variance=gaussian_width_distance
            )
        else:
            self.distance_embedding = GaussBasisExpansion.from_bounds_log(
                bins_distance,
                0.0,
                max_distance,
                base=distance_log_base,
                variance=gaussian_width_distance,
            )

        self.distance_log_base = distance_log_base
        self.log_max_distance = np.log(max_distance)
        return

    def forward(self, distance):
        d = torch.unsqueeze(distance, 1)
        distance_embedded = self.distance_embedding(d)
        return distance_embedded
