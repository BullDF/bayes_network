from typing import Self, Any, Optional, Union
from abc import ABC, abstractmethod
from distribution import Distribution, DiscreteDistribution, ContinuousDistribution


class Node(ABC):
    name: str
    parents: list[Self]
    children: list[Self]
    domain: Any
    distributions: Optional[dict[tuple[tuple[str, Any]], Distribution]]

    def __init__(self, name: str) -> None:
        self.name = name
        self.parents = []
        self.children = []
        self.domain = None
        self.distributions = None

    def add_parent(self, parent: Self) -> None:
        for p in self.parents:
            if p.name == parent.name:
                raise ValueError(f'Node {parent.name} is already a parent of node {self.name}.')
        self.parents.append(parent)

    def add_child(self, child: Self) -> None:
        for c in self.children:
            if c.name == child.name:
                raise ValueError(f'Node {child.name} is already a child of node {self.name}.')
        self.children.append(child)

    @abstractmethod
    def is_in_domain(self, value: Any) -> bool:
        pass

    @abstractmethod
    def check_distributions(self, distributions: dict[tuple[tuple[str, Any]], Distribution]) -> None:
        pass

    def set_distributions(self, distributions: dict[tuple[tuple[str, Any]], Distribution]) -> None:
        self.check_distributions(distributions)
        self.distributions = distributions

    def __str__(self) -> str:
        return f'{self.name}: {self.domain}'


class DiscreteNode(Node):
    domain: set

    def __init__(self, name: str, domain: set) -> None:
        super().__init__(name)
        self.domain = domain

    def is_in_domain(self, value: Any) -> bool:
        return value in self.domain

    def check_distributions(self, distributions: dict[tuple[tuple[str, Any]], Distribution]) -> None:
        for conditions, distribution in distributions.items():
            if not isinstance(distribution, DiscreteDistribution):
                raise ValueError(
                    f'DiscreteNode {self.name} can only have DiscreteDistribution.')
            if self.domain != distribution.domain:
                raise ValueError(
                    f'Distribution domain {distribution.domain} does not match node domain {self.domain}.')
            
            for variable, value in conditions:
                parent = None
                for p in self.parents:
                    if p.name == variable:
                        parent = p
                        break
                if parent is None:
                    raise ValueError(
                        f'Variable {variable} is not a parent of node {self.name}.')
                if not parent.is_in_domain(value):
                    raise ValueError(
                        f'Value {value} is not in the domain of parent {variable}.')


class ContinuousNode(Node):
    domain: tuple[float, float]

    def __init__(self, name: str, domain: tuple[float, float]) -> None:
        super().__init__(name)
        self.domain = domain

    def is_in_domain(self, value: Any) -> bool:
        return self.domain[0] < value < self.domain[1]

    def check_distributions(self, distributions: dict[tuple[tuple[str, Any]], Distribution]) -> None:
        for conditions, distribution in distributions.items():
            if not isinstance(distribution, ContinuousDistribution):
                raise ValueError(
                    f'ContinuousNode {self.name} can only have ContinuousDistribution.')
            if self.domain != distribution.domain:
                raise ValueError(
                    f'Distribution domain {distribution.domain} does not match node domain {self.domain}.')
            
            for variable, value in conditions:
                parent = None
                for p in self.parents:
                    if p.name == variable:
                        parent = p
                        break
                if parent is None:
                    raise ValueError(
                        f'Variable {variable} is not a parent of node {self.name}.')
                if not parent.is_in_domain(value):
                    raise ValueError(
                        f'Value {value} is not in the domain of parent {variable}.')


class BayesNetwork:
    nodes: list[Node]

    def __init__(self):
        self.nodes = []

    def __len__(self) -> int:
        return len(self.nodes)

    def add_node(self, node: Node) -> None:
        for n in self.nodes:
            if n.name == node.name:
                raise ValueError(f'Node {node.name} is already in the network.')
        self.nodes.append(node)

    def add_edge(self, parent: Node, child: Node) -> None:
        parent.add_child(child)
        child.add_parent(parent)

    def joint_probability(self):
        pass
