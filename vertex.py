from typing import Self, Any, Optional
from distribution import Distribution

class Vertex:
    name: str
    domain: set
    parents: list[Self]
    children: list[Self]
    distribution: Optional[Distribution]

    def __init__(self, name: str, domain: set) -> None:
        self.name = name
        self.domain = domain
        self.parents = []
        self.children = []
        self.distribution = None

    def add_parent(self, parent: Self) -> None:
        for p in self.parents:
            if p.name == parent.name:
                raise ValueError(f'Vertex {parent.name} is already a parent of vertex {self.name}.')
        self.parents.append(parent)

    def add_child(self, child: Self) -> None:
        for c in self.children:
            if c.name == child.name:
                raise ValueError(f'Vertex {child.name} is already a child of vertex {self.name}.')
        self.children.append(child)

    def in_domain(self, value: Any) -> bool:
        return value in self.domain
    
    def set_distribution(self, distribution: Distribution) -> None:
        self.distribution = distribution

    def __str__(self) -> str:
        return f'{self.name}: {self.domain}'