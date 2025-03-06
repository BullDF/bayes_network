from typing import List, Self
from abc import ABC


class Distribution(ABC):
    pass


class DiscreteDistribution(Distribution):
    pass


class ContinuousDistribution(Distribution):
    pass


class Node:
    name: str
    parents: List[Self]
    children: List[Self]
    distribution: Distribution

    def __init__(self, name: str) -> None:
        self.name = name
        self.parents = []
        self.children = []
        self.distribution = None

    def add_parent(self, parent: Self) -> None:
        self.parents.append(parent)

    def add_child(self, child: Self) -> None:
        self.children.append(child)

    def get_parents(self) -> List[Self]:
        return self.parents

    def get_children(self) -> List[Self]:
        return self.children


class BayesNetwork:
    nodes: List[Node]

    def __init__(self):
        self.nodes = []

    def __len__(self) -> int:
        return len(self.nodes)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_edge(self, parent: Node, child: Node) -> None:
        parent.add_child(child)
        child.add_parent(parent)

    def joint_probability(self):
        pass
