"""Belief updates for a robot that works at several sites (see envs/multi_site.py)."""

from typing import Any, Sequence

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.sites import SiteBelief
from able.envs.multi_site import SiteState
from able.structs import QueryResponse


class MultiSiteUpdater(BeliefUpdater[SiteState[Any], Any, Any, Any, Any]):
    """Updates the belief about the site where a step happened with the
    single-site updater, and leaves the other sites' beliefs alone. Sites take no
    queries."""

    def __init__(self, inner: BeliefUpdater[Any, Any, Any, Any, Any]) -> None:
        self.inner = inner

    def update_from_transition(
        self,
        belief: Belief[Any],
        state: SiteState[Any],
        action: Any,
        next_state: SiteState[Any],
    ) -> SiteBelief:
        assert isinstance(belief, SiteBelief), "Need a SiteBelief"
        site = state.site
        updated = self.inner.update_from_transition(
            belief.beliefs[site], state.inner, action, next_state.inner
        )
        return belief.with_site(site, updated)

    def update_from_queries(
        self, belief: Belief[Any], data: Sequence[QueryResponse[Any, Any]]
    ) -> Belief[Any]:
        assert not data, "Sites take no queries"
        return belief
