from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class MetricStats:
    name: str = "none"
    values: List[float] = field(default_factory=lambda: [])

    def value(self):
        return np.mean(self.values)


@dataclass
class EpochStats:
    stats: List[MetricStats]

    def __iter__(self):
        return iter(self.stats)

    def __len__(self):
        return len(self.stats)
