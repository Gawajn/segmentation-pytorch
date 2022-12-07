from dataclasses import dataclass, field
from typing import List

import numpy as np
from mashumaro.mixins.json import DataClassJSONMixin


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

    def to_dict(self ) -> dict:
        return {i.name: i.value() for i in self.stats}
