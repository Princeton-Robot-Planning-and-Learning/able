"""Planning for a robot that works at several sites (see envs/multi_site.py)."""

from typing import Any

import numpy as np

from able.beliefs.base import Belief
from able.beliefs.sites import SiteBelief
from able.envs.multi_site import SiteState, SiteTask
from able.models.base import ConditionalModel, Dataset
from able.planners.base import TrainablePlanner
from able.policies.base import Policy
from able.structs import Budget, Trajectory


class SitePolicy(Policy[SiteState[Any], Any]):
    """A single-site policy run on the single-site part of each state."""

    def __init__(self, inner: Policy[Any, Any]) -> None:
        self.inner = inner

    def get_action(self, state: SiteState[Any]) -> Any:
        return self.inner.get_action(state.inner)

    def get_planning_stats(self) -> dict[str, float]:
        return self.inner.get_planning_stats()


class MultiSitePlanner(TrainablePlanner[SiteState[Any], Any, SiteTask, Any, Any]):
    """Plans each site's task with the single-site planner under that site's
    belief. Every site shares the planner's parameters, and so its model, whose
    inputs describe only the site's own belief. Training rows likewise describe
    the site's own belief."""

    def __init__(self, inner: TrainablePlanner[Any, Any, Any, Any, Any]) -> None:
        self.inner = inner

    def get_default_params(self) -> Any:
        return self.inner.get_default_params()

    def plan(
        self,
        task: SiteTask,
        belief: Belief[Any],
        params: Any,
        budget: Budget,
        rng: np.random.Generator,
    ) -> SitePolicy:
        assert isinstance(belief, SiteBelief), "Need a SiteBelief"
        site_belief = belief.beliefs[task.site]
        return SitePolicy(self.inner.plan(task.inner, site_belief, params, budget, rng))

    def get_model(self, params: Any) -> ConditionalModel:
        return self.inner.get_model(params)

    def set_model(self, params: Any, model: ConditionalModel) -> Any:
        return self.inner.set_model(params, model)

    def get_training_params(self, params: Any) -> Any:
        return self.inner.get_training_params(params)

    def get_plan_examples(
        self, task: SiteTask, belief: Belief[Any], policy: Policy[Any, Any]
    ) -> Dataset:
        assert isinstance(belief, SiteBelief) and isinstance(policy, SitePolicy)
        return self.inner.get_plan_examples(
            task.inner, belief.beliefs[task.site], policy.inner
        )

    def get_rollout_examples(
        self,
        task: SiteTask,
        belief: Belief[Any],
        policy: Policy[Any, Any],
        trajectory: Trajectory[Any, Any],
        solved: bool,
    ) -> Dataset:
        assert isinstance(belief, SiteBelief) and isinstance(policy, SitePolicy)
        inner_trajectory = Trajectory(
            tuple(state.inner for state in trajectory.states),
            trajectory.actions,
            trajectory.costs,
        )
        return self.inner.get_rollout_examples(
            task.inner,
            belief.beliefs[task.site],
            policy.inner,
            inner_trajectory,
            solved,
        )
