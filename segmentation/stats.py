from dataclasses import dataclass, field
from typing import List

import numpy as np
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class MetricStats:
    name: str = "none"
    values: List[float] = field(default_factory=lambda: [])

    def value(self):
        if not self.values:
            return 0.0
        return np.mean(self.values).item()


@dataclass
class EpochStats(DataClassJSONMixin):
    stats: List[MetricStats]

    def __iter__(self):
        return iter(self.stats)

    def __len__(self):
        return len(self.stats)

    def to_dict(self ) -> dict:
        return {i.name: float(i.value()) for i in self.stats}

    def __getitem__(self, item):
        return self.stats[item]
