"""Base class for beliefs over hidden environment parameters."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from able.structs import HiddenParamsType


class Belief(ABC, Generic[HiddenParamsType]):
    """An immutable distribution over hidden environment parameters.

    Beliefs are never modified in place. Belief updaters return new beliefs, so a
    learner can imagine updates without changing the robot's actual belief.
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> HiddenParamsType:
        """Draw one sample of the hidden parameters."""
