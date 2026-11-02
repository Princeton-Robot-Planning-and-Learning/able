"""Information gathering about the joint limits of the two-link arm."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from able.beliefs.base import Belief
from able.beliefs.interval import IntervalBelief
from able.envs.base import Environment
from able.envs.two_link_arm import JointQuery
from able.info_gatherers.base import InformationGatherer
from able.structs import Budget, QueryResponse


class WidestGapGatherer(InformationGatherer[NDArray[np.float64], JointQuery, bool]):
    """Asks whether a joint can move to the midpoint of the widest gap in the belief.

    Each interval of the belief over [lo1, hi1, lo2, hi2] starts as one gap. Each
    query asks about the midpoint of the widest gap, which splits that gap in two,
    so the first query bisects the widest interval. The queries depend only on the
    belief, not on the responses, so a dream that starts from the robot's belief
    asks the robot's queries. One iteration of the budget is one query. Gathering
    stops early once every gap is narrower than min_gap_width.
    """

    def __init__(self, min_gap_width: float = 1e-6) -> None:
        self._min_gap_width = min_gap_width

    def gather(
        self,
        belief: Belief[NDArray[np.float64]],
        env: Environment[Any, Any, Any, JointQuery, bool],
        budget: Budget,
        rng: np.random.Generator,
    ) -> list[QueryResponse[JointQuery, bool]]:
        del rng  # unused because the queries are deterministic
        assert (
            budget.max_iterations is not None or budget.max_wall_time is not None
        ), "Information gathering needs a finite budget"
        assert isinstance(belief, IntervalBelief), "Need an IntervalBelief"
        # Each gap is (low, high, index of the limit in [lo1, hi1, lo2, hi2]).
        gaps = [
            (low, high, index) for index, (low, high) in enumerate(belief.intervals)
        ]
        data: list[QueryResponse[JointQuery, bool]] = []
        start_time = time.perf_counter()
        while not budget.is_exhausted(time.perf_counter() - start_time, len(data)):
            widest = max(range(len(gaps)), key=lambda i: gaps[i][1] - gaps[i][0])
            low, high, index = gaps.pop(widest)
            if high - low < self._min_gap_width:
                break
            midpoint = (low + high) / 2
            gaps += [(low, midpoint, index), (midpoint, high, index)]
            query = JointQuery(joint=index // 2, angle=float(midpoint))
            data.append(QueryResponse(query, env.ask(query)))
        return data
