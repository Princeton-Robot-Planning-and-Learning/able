"""Beliefs about several sites at once (see envs/multi_site.py)."""

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief


class SiteBelief(Belief[NDArray[np.float64]]):
    """Independent beliefs about each site's hidden parameters, and optionally
    the site that a unit of sleep should train for.

    A sample stacks one sample from each site's belief, so its first axis is the
    site. Each site's belief is used on its own: planning for a site's task uses
    that site's belief, and what the robot sees at a site changes only that
    site's belief.
    """

    def __init__(
        self, beliefs: Sequence[Belief[Any]], focus: int | None = None
    ) -> None:
        assert len(beliefs) > 0, "Need at least one site"
        assert focus is None or 0 <= focus < len(beliefs), "Need a site to focus on"
        self.beliefs = tuple(beliefs)
        self.focus = focus

    @property
    def num_sites(self) -> int:
        """The number of sites."""
        return len(self.beliefs)

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """One sample from each site's belief, stacked."""
        return np.stack([np.asarray(belief.sample(rng)) for belief in self.beliefs])

    def with_site(self, site: int, belief: Belief[Any]) -> "SiteBelief":
        """The same beliefs and focus, with the site's belief replaced."""
        beliefs = list(self.beliefs)
        beliefs[site] = belief
        return SiteBelief(beliefs, self.focus)

    def focused(self, site: int) -> "SiteBelief":
        """The same beliefs, focused on the site."""
        return SiteBelief(self.beliefs, site)
