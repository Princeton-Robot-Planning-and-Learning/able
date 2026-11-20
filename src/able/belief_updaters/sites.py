"""Belief updates for a robot that works at several sites (see envs/multi_site.py)."""

from typing import Any, Sequence

from able.belief_updaters.base import BeliefUpdater
from able.beliefs.base import Belief
from able.beliefs.sites import SiteBelief
from able.envs.multi_site import SiteQuery, SiteState
from able.structs import QueryResponse


class MultiSiteUpdater(BeliefUpdater[SiteState[Any], Any, Any, Any, Any]):
    """Updates the belief about the site where a step happened or a query was
    asked with the single-site updater, and leaves the other sites' beliefs
    alone."""

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
        assert isinstance(belief, SiteBelief), "Need a SiteBelief"
        for datum in data:
            query = datum.query
            assert isinstance(query, SiteQuery), "Need a SiteQuery"
            updated = self.inner.update_from_queries(
                belief.beliefs[query.site],
                [QueryResponse(query.inner, datum.response)],
            )
            belief = belief.with_site(query.site, updated)
        return belief
