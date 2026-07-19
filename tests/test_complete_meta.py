from __future__ import annotations

from rc_irstd.meta import make_episode_windows


def test_causal_windows_are_disjoint_and_ordered() -> None:
    windows = make_episode_windows(20, 4, 6, stride=2, mode="causal")
    assert windows
    for window in windows:
        assert set(window.support_indices).isdisjoint(window.query_indices)
        assert max(window.support_indices) < min(window.query_indices)
