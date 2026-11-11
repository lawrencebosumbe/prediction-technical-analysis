"""Purged expanding-window time-series folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class WalkForward:
    """Generate ordered train/test index arrays with purge and embargo gaps."""

    train_window: int = 500
    test_window: int = 100
    purge: int = 0
    embargo: int = 0
    expanding: bool = False

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if min(self.train_window, self.test_window) < 1 or min(self.purge, self.embargo) < 0:
            raise ValueError("windows must be positive and purge/embargo non-negative")
        test_start = self.train_window + self.purge
        while test_start < n_samples:
            train_end = test_start - self.purge
            train_start = 0 if self.expanding else max(0, train_end - self.train_window)
            test_end = min(n_samples, test_start + self.test_window)
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            if self.embargo:
                test_idx = test_idx[test_idx >= train_end + self.embargo]
            if train_idx.size and test_idx.size:
                yield train_idx, test_idx
            test_start += self.test_window + self.embargo

