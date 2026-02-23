"""Fixed-threshold strategy with an explicit inventory cap."""
from .fixed import FixedThresholdStrategy


class FixedThresholdCapped(FixedThresholdStrategy):
    def __init__(self, inventory_cap=100, **kw):
        super().__init__(**kw)
        self.inventory_cap = inventory_cap
