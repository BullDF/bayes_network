from typing import Self, Any, Optional

from .distribution import *
from .utils import str_to_value


class Vertex:
    name: str
    domain: set
    parents: dict[str, Self]
    children: dict[str, Self]
    distribution: Optional[Distribution]

    def __init__(self, name: str, domain: set) -> None:
        self.name = name
        self.domain = domain
        self.parents = {}
        self.children = {}
        self.distribution = None

    def add_parent(self, parent: Self) -> None:
        if self.distribution is not None:
            raise ValueError('Cannot add parent to vertex with distribution.')
        if parent.name in self.parents:
            raise ValueError(f'Vertex {parent.name} is already a parent of vertex {self.name}.')
        if parent.name in self.children:
            raise ValueError(f'Vertex {parent.name} is already a child of vertex {self.name}.')
        
        self.parents[parent.name] = parent

    def add_child(self, child: Self) -> None:
        if child.name in self.children:
            raise ValueError(f'Vertex {child.name} is already a child of vertex {self.name}.')
        if child.name in self.parents:
            raise ValueError(f'Vertex {child.name} is already a parent of vertex {self.name}.')
        self.children[child.name] = child

    def in_domain(self, value: Any) -> bool:
        return value in self.domain
    
    def set_distribution(self, distribution: Distribution) -> None:
        if distribution.domain != self.domain:
            raise ValueError('Domain of distribution must match domain of vertex.')
        
        if self.parents:
            if isinstance(distribution, UnconditionalDistribution):
                raise ValueError('Vertex with parents cannot have unconditional distribution.')
            for conditions in distribution.distributions.keys():
                if len(conditions) != len(self.parents):
                    raise ValueError('Length of conditions must match number of parents.')
                for condition in conditions:
                    colon = condition.index(':')
                    parent = condition[:colon]
                    if parent not in self.parents:
                        raise ValueError(f'Variable {parent} is not a parent of vertex {self.name}.')
                    value = str_to_value(condition[colon + 2:])
                    if not self.parents[parent].in_domain(value):
                        raise ValueError(f'Value {value} is not in domain of parent {parent}.')

        elif not self.parents:
            if isinstance(distribution, ConditionalDistribution):
                raise ValueError('Vertex without parents cannot have conditional distribution.')
                
        self.distribution = distribution

    def __str__(self) -> str:
        return f'{self.name}: {self.domain}'
    
    def __call__(self, *args) -> float:
        if self.distribution is None:
            raise ValueError('Vertex has no distribution.')
        return self.distribution(*args)
    
    def __repr__(self) -> str:
        return self.__str__()