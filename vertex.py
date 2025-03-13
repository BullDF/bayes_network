from typing import Self, Any, Optional
from distribution import Distribution

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
        if parent.name in self.parents:
            raise ValueError(f'Vertex {parent.name} is already a parent of vertex {self.name}.')
        self.parents[parent.name] = parent

    def add_child(self, child: Self) -> None:
        if child.name in self.children:
            raise ValueError(f'Vertex {child.name} is already a child of vertex {self.name}.')
        self.children[child.name] = child

    def in_domain(self, value: Any) -> bool:
        return value in self.domain
    
    def set_distribution(self, distribution: Distribution) -> None:
        self.distribution = distribution

    def __str__(self) -> str:
        return f'{self.name}: {self.domain}'