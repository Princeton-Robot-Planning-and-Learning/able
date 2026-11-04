"""Tests for planners/pddlstream.py."""

import os
from collections import Counter

from able.planners.pddlstream import counted, private_cwd


def test_private_cwd(tmp_path, monkeypatch):
    """Code runs in a fresh directory that is removed afterwards."""
    monkeypatch.chdir(tmp_path)
    with private_cwd():
        inside = os.getcwd()
        assert inside != str(tmp_path)
        with open("scratch", "w", encoding="utf-8") as f:
            f.write("x")
    assert os.getcwd() == str(tmp_path) and not os.path.exists(inside)


def test_counted_streams():
    """Each test call and each generated output counts once."""
    calls: Counter = Counter()
    test = counted(lambda x: x > 0, calls, "positive", generator=False)
    assert test(1) and not test(-1)

    def two_outputs(x):
        yield (x,)
        yield (x + 1,)

    gen = counted(two_outputs, calls, "next", generator=True)
    assert list(gen(1)) == [(1,), (2,)]
    assert calls == {"positive": 2, "next": 3}
