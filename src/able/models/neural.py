"""Neural conditional models, and an MLP that regresses outputs on inputs."""

import copy
import dataclasses
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from able.models.base import ConditionalModel, Dataset, split_groups


def make_mlp(input_dim: int, output_dim: int, width: int, depth: int) -> nn.Module:
    """A multilayer perceptron with depth hidden layers of the given width."""
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(depth):
        layers += [nn.Linear(dim, width), nn.SiLU()]
        dim = width
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True, eq=False)
class Normalizer:
    """Standardizes each column to zero mean and unit standard deviation.

    Columns with (near) zero standard deviation are only shifted.
    """

    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    @classmethod
    def from_data(cls, data: NDArray[np.float64]) -> "Normalizer":
        """The normalizer for the columns of the data."""
        std = data.std(axis=0)
        return cls(data.mean(axis=0), np.where(std < 1e-8, 1.0, std))

    def normalize(self, data: NDArray[np.float64]) -> torch.Tensor:
        """Standardize the data and convert it to a float32 tensor."""
        return torch.as_tensor((data - self.mean) / self.std, dtype=torch.float32)

    def unnormalize(self, data: torch.Tensor) -> NDArray[np.float64]:
        """Undo normalize()."""
        return data.detach().numpy().astype(np.float64) * self.std + self.mean


@dataclass(frozen=True, eq=False)
class FitResult:
    """A trained network, its normalizers, and how training went.

    The validation loss is NaN if nothing was held out.
    """

    network: nn.Module
    x_normalizer: Normalizer
    y_normalizer: Normalizer
    best_step: int
    validation_loss: float


@dataclass(frozen=True, eq=False)
class NeuralModel(ConditionalModel):
    """A conditional model whose network is trained with minibatch gradient descent.

    Inputs and outputs are standardized. A fraction of the groups is held out, and
    training keeps the weights with the lowest held-out loss. Subclasses define the
    network, the loss, and how a network draws outputs from base noise. Training and
    the initial weights depend only on the seed and the dataset.
    """

    width: int = 256
    depth: int = 3
    num_steps: int = 2000
    batch_size: int = 256
    learning_rate: float = 1e-3
    holdout_fraction: float = 1 / 6
    min_groups: int = 6
    validation_interval: int = 100
    seed: int = 0
    fitted: FitResult | None = None

    @abstractmethod
    def _build(self, x_dim: int, y_dim: int) -> nn.Module:
        """A new, untrained network."""

    @abstractmethod
    def _get_loss(
        self,
        network: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """The training loss on standardized inputs and outputs."""

    @abstractmethod
    def _draw(
        self, network: nn.Module, x: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Standardized outputs for standardized inputs, one per row of base noise
        with the shape of the outputs."""

    @property
    def is_deterministic(self) -> bool:
        """Whether every draw for an input is the same, so one draw is enough."""
        return False

    def fit(self, dataset: Dataset) -> "NeuralModel":
        """Train a new network on the dataset."""
        assert len(dataset) > 0, "Need at least one row"
        x_normalizer = Normalizer.from_data(dataset.x)
        y_normalizer = Normalizer.from_data(dataset.y)
        x = x_normalizer.normalize(dataset.x)
        y = y_normalizer.normalize(dataset.y)
        held = split_groups(
            dataset.groups,
            self.holdout_fraction,
            self.min_groups,
            np.random.default_rng(self.seed),
        )
        train_rows = torch.as_tensor(np.flatnonzero(~held))
        held_rows = torch.as_tensor(np.flatnonzero(held))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            network = self._build(x.shape[1], y.shape[1])
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            total_steps=self.num_steps,
            pct_start=max(0.05, 3 / self.num_steps),
        )
        generator = torch.Generator().manual_seed(self.seed)
        best_loss, best_step, best_state = float("inf"), self.num_steps, None
        for step in range(1, self.num_steps + 1):
            network.train()
            batch = train_rows[
                torch.randint(len(train_rows), (self.batch_size,), generator=generator)
            ]
            loss = self._get_loss(network, x[batch], y[batch], generator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            schedule.step()
            if len(held_rows) and (
                step % self.validation_interval == 0 or step == self.num_steps
            ):
                network.eval()
                with torch.no_grad():
                    validation_generator = torch.Generator().manual_seed(self.seed)
                    held_loss = float(
                        self._get_loss(
                            network, x[held_rows], y[held_rows], validation_generator
                        )
                    )
                if held_loss < best_loss:
                    best_loss, best_step = held_loss, step
                    best_state = copy.deepcopy(network.state_dict())
        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        result = FitResult(
            network=network,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            best_step=best_step,
            validation_loss=best_loss if len(held_rows) else float("nan"),
        )
        return dataclasses.replace(self, fitted=result)

    def sample(
        self, x: NDArray[np.float64], num_samples: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Draw outputs for x from base noise sampled with the rng.

        Deterministic models return one output. Untrained models return none.
        """
        if self.fitted is None or num_samples < 1:
            return np.zeros((0, 0))
        num_draws = 1 if self.is_deterministic else num_samples
        fitted = self.fitted
        y_dim = len(fitted.y_normalizer.mean)
        inputs = fitted.x_normalizer.normalize(np.tile(x, (num_draws, 1)))
        noise = torch.as_tensor(
            rng.standard_normal((num_draws, y_dim)), dtype=torch.float32
        )
        with torch.no_grad():
            outputs = self._draw(fitted.network, inputs, noise)
        return fitted.y_normalizer.unnormalize(outputs)


@dataclass(frozen=True, eq=False)
class MLPRegressor(NeuralModel):
    """Predicts the mean output with an MLP trained on squared error.

    Every draw is the prediction, so sample() returns one output.
    """

    def _build(self, x_dim: int, y_dim: int) -> nn.Module:
        return make_mlp(x_dim, y_dim, self.width, self.depth)

    def _get_loss(
        self,
        network: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del generator  # unused because the loss has no noise
        return ((network(x) - y) ** 2).mean()

    def _draw(
        self, network: nn.Module, x: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        del noise  # unused because the prediction is deterministic
        return network(x)

    @property
    def is_deterministic(self) -> bool:
        return True
