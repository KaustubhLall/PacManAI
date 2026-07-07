import numpy as np


class SumTree:
    """Binary sum tree for prioritized replay sampling."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.data[:] = None
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        if self.n_entries == 0 or self.total() <= 0:
            raise ValueError("cannot sample from an empty SumTree")

        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self):
        return self.n_entries
