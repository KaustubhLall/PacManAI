import pytest

from ai.environments.sumtree import SumTree


def test_sumtree_len_tracks_inserted_entries_not_capacity():
    tree = SumTree(3)

    assert len(tree) == 0

    tree.add(1.0, "first")
    tree.add(2.0, "second")

    assert len(tree) == 2
    assert tree.total() == 3.0


def test_sumtree_len_caps_at_capacity_when_wrapping():
    tree = SumTree(2)

    tree.add(1.0, "first")
    tree.add(1.0, "second")
    tree.add(1.0, "third")

    assert len(tree) == 2
    assert tree.data.tolist() == ["third", "second"]


def test_sumtree_rejects_empty_sampling():
    tree = SumTree(2)

    with pytest.raises(ValueError):
        tree.get(0.5)
