"""A conditional flow-matching model, which can represent multimodal outputs."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from able.models.neural import NeuralModel, make_mlp


def embed_time(times: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features with shape (N, dim) for times in [0, 1] with shape
    (N,)."""
    frequencies = torch.exp(
        -math.log(1000.0) * torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    )
    angles = 1000.0 * times[:, None] * frequencies[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class _VelocityField(nn.Module):
    """Maps a point on the path, the input, and the time to a velocity."""

    def __init__(self, network: nn.Module, time_dim: int) -> None:
        super().__init__()
        self.network = network
        self.time_dim = time_dim

    def forward(
        self, point: torch.Tensor, x: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        """The velocity at each point."""
        features = torch.cat([point, x, embed_time(times, self.time_dim)], dim=-1)
        output: torch.Tensor = self.network(features)
        return output


@dataclass(frozen=True, eq=False)
class FlowMatchingModel(NeuralModel):
    """A conditional rectified flow from Gaussian noise to the outputs.

    Training regresses the velocity of the straight path from a noise sample to an
    output. Drawing integrates the learned velocity from noise with Euler steps.
    Unlike regression, the draws cover every mode of the outputs for an input,
    rather than their mean.
    """

    time_dim: int = 64
    num_integration_steps: int = 10

    def _build(self, x_dim: int, y_dim: int) -> nn.Module:
        network = make_mlp(y_dim + x_dim + self.time_dim, y_dim, self.width, self.depth)
        return _VelocityField(network, self.time_dim)

    def _get_loss(
        self,
        network: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn(y.shape, generator=generator)
        times = torch.rand(len(y), generator=generator)
        point = (1 - times[:, None]) * noise + times[:, None] * y
        return ((network(point, x, times) - (y - noise)) ** 2).mean()

    def _draw(
        self, network: nn.Module, x: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        point, step_size = noise, 1.0 / self.num_integration_steps
        for step in range(self.num_integration_steps):
            times = torch.full((len(x),), step * step_size)
            point = point + step_size * network(point, x, times)
        return point
