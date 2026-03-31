import time
import math
from typing import Optional

class ExperimentHarness:
    """
    Simple harness to record metrics and provide stop/check utilities.
    """

    def __init__(self, time_budget: float = 60.0):
        self.time_budget = float(time_budget)
        self.start_time = time.monotonic()
        self.metrics = []
        self._stopped = False

    def should_stop(self) -> bool:
        if self._stopped:
            return True
        return (time.monotonic() - self.start_time) >= self.time_budget

    def check_value(self, val: Optional[float], name: str) -> bool:
        if val is None:
            return False
        try:
            if math.isnan(val) or math.isinf(val):
                return False
            return True
        except Exception:
            return False

    def report_metric(self, name: str, value: float) -> None:
        self.metrics.append((name, float(value)))

    def finalize(self) -> None:
        self._stopped = True