"""Gathering information at one of several sites (see envs/multi_site.py)."""

from typing import Any, Callable

import numpy as np

from able.beliefs.base import Belief
from able.beliefs.sites import SiteBelief
from able.envs.base import Environment
from able.envs.multi_site import SiteQuery
from able.info_gatherers.base import InformationGatherer
from able.structs import Budget, QueryResponse, StepResult


class _SiteEnvironment(Environment[Any, Any, Any, Any, Any]):
    """A multi-site environment that takes single-site queries at one site."""

    def __init__(self, env: Environment[Any, Any, Any, Any, Any], site: int) -> None:
        self._env = env
        self._site = site

    def reset(self, task: Any) -> Any:
        raise NotImplementedError("Only queries are asked at a site")

    def step(self, action: Any) -> StepResult[Any]:
        raise NotImplementedError("Only queries are asked at a site")

    def ask(self, query: Any) -> Any:
        return self._env.ask(SiteQuery(self._site, query))

    def check_goal(self, task: Any, state: Any) -> bool:
        raise NotImplementedError("Only queries are asked at a site")


class MultiSiteGatherer(InformationGatherer[Any, SiteQuery[Any], Any]):
    """Asks queries at the day's site, get_site(rng), with get_gatherer(site)
    under the belief about that site, and none on a day with no site."""

    def __init__(
        self,
        get_gatherer: Callable[[int], InformationGatherer[Any, Any, Any]],
        get_site: Callable[[np.random.Generator], int | None],
    ) -> None:
        self._get_gatherer = get_gatherer
        self._get_site = get_site

    def gather(
        self,
        belief: Belief[Any],
        env: Environment[Any, Any, Any, SiteQuery[Any], Any],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[SiteQuery[Any], Any]]:
        site = self._get_site(rng)
        if site is None:
            return []
        assert isinstance(belief, SiteBelief), "Need a SiteBelief"
        data = self._get_gatherer(site).gather(
            belief.beliefs[site], _SiteEnvironment(env, site), budget, rng
        )
        return [QueryResponse(SiteQuery(site, d.query), d.response) for d in data]
