from typing import Any
from abc import ABC, abstractmethod

from .utils import dict_to_frozenset


class Distribution(ABC):
    domain: set

    @abstractmethod
    def __call__(self, *args) -> float:
        pass

    def __repr__(self):
        return self.__str__()


class UnconditionalDistribution(Distribution):
    distribution: dict[Any, float]

    def __init__(self, domain: set, distribution: dict[Any, float]) -> None:
        self.domain = domain

        if set(distribution.keys()) != domain:
            raise ValueError('Keys of distribution must match domain.')

        if sum(distribution.values()) != 1:
            raise ValueError('Probabilities must sum to 1.')
        
        for value, prob in distribution.items():
            if prob < 0 or prob > 1:
                raise ValueError(f'Probability {prob} is invalid for value {value}.')
            
        self.distribution = distribution

    def __call__(self, *args) -> float:
        if len(args) != 1:
            raise ValueError('Unconditional distribution requires exactly one argument.')
        
        value = args[0]
        if value not in self.domain:
            raise ValueError(f'Value {value} not in domain {self.domain}.')
        return self.distribution[value]
    
    def __str__(self) -> str:
        return str(self.distribution)


class ConditionalDistribution(Distribution):
    distributions: dict[frozenset[str], UnconditionalDistribution]

    def __init__(self, domain: set, distributions: dict[frozenset[str], UnconditionalDistribution]) -> None:
        self.domain = domain

        for distribution in distributions.values():
            if distribution.domain != domain:
                raise ValueError('Domain of distribution must match domain of vertex.')

        self.distributions = distributions

    def __call__(self, *args) -> float:
        if len(args) != 2:
            raise ValueError('Conditional distribution requires exactly two arguments.')
        
        value = args[0]
        conditions = args[1]
        if not isinstance(conditions, dict):
            raise ValueError('Second argument must be a dictionary of conditions.')
        
        if value not in self.domain:
            raise ValueError(f'Value {value} not in domain {self.domain}.')
        
        conditions = dict_to_frozenset(conditions)
        if conditions not in self.distributions:
            raise ValueError(f'Conditions {set(conditions)} is invalid.')

        return self.distributions[conditions](value)
    
    def __str__(self) -> str:
        result = ''
        for conditions, distribution in self.distributions.items():
            result += f'{distribution} | '
            for condition in conditions:
                result += f'{condition}, '
            result = result[:-2] + '\n'

        return result.strip()
        