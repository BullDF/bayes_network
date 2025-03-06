from abc import ABC
from typing import Any, Callable, Optional


class Distribution(ABC):
    domain: Any
    probability_function: Optional[Callable]

    def __init__(self, domain: Any) -> None:
        self.domain = domain
        self.probability_function = None

    def set_probability_function(self, probability_function: Callable) -> None:
        self.probability_function = probability_function


class DiscreteDistribution(Distribution):
    domain: set

    def __init__(self, domain: set) -> None:
        super().__init__(domain)


class ContinuousDistribution(Distribution):
    domain: tuple[float, float]

    def __init__(self, domain: tuple[float, float]) -> None:
        super().__init__(domain)

