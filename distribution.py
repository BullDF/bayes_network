from typing import Any
from abc import ABC, abstractmethod
from utils import dict_to_frozenset


class Dist(ABC):
    domain: set

    @abstractmethod
    def __call__(self, *args) -> float:
        pass


class UncondDist(Dist):
    dist: dict[Any, float]

    def __init__(self, domain: set, dist: dict[Any, float]) -> None:
        self.domain = domain

        if set(dist.keys()) != domain:
            raise ValueError('Keys of distribution must match domain.')

        if sum(dist.values()) != 1:
            raise ValueError('Probabilities must sum to 1.')
        
        for value, prob in dist.items():
            if prob < 0 or prob > 1:
                raise ValueError(f'Probability {prob} is invalid for value {value}.')
            
        self.dist = dist

    def __call__(self, *args) -> float:
        if len(args) != 1:
            raise ValueError('Unconditional distribution requires exactly one argument.')
        
        value = args[0]
        if value not in self.domain:
            raise ValueError(f'Value {value} not in domain {self.domain}.')
        return self.dist[value]
    
    def __str__(self) -> str:
        return str(self.dist)


class CondDist(Dist):
    dists: dict[frozenset[str], UncondDist]

    def __init__(self, domain: set, dists: dict[frozenset[str], UncondDist]) -> None:
        self.domain = domain

        for dist in dists.values():
            if dist.domain != domain:
                raise ValueError('Domain of distribution must match domain of vertex.')

        self.dists = dists

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
        if conditions not in self.dists:
            raise ValueError(f'Conditions {set(conditions)} is invalid.')

        return self.dists[conditions](value)
    
    def __str__(self) -> str:
        result = ''
        for conditions, dist in self.dists.items():
            result += f'{dist} | '
            for condition in conditions:
                result += f'{condition}, '
            result = result[:-2] + '\n'

        return result
        