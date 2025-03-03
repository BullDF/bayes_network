from typing import List, Self, Union


class Node:
    name: str
    parents: List[Self]
    children: List[Self]

    def __init__(self, name: str) -> None:
        self.name = name

    def add_parent(self, parent: Self) -> None:
        self.parents.append(parent)

    def add_child(self, child: Self) -> None:
        self.children.append(child)

    def get_parents(self) -> List[Self]:
        return self.parents

    def get_children(self) -> List[Self]:
        return self.children


class DAGM:
    nodes: List[Node]

    def __init__(self) -> None:
        self.nodes = []

    def __len__(self) -> int:
        return len(self.nodes)

    def add_node(self, node: Union[Node, str]) -> None:
        if isinstance(node, str):
            node = Node(node)
        self.nodes.append(node)

    def add_edge(self, parent: Union[Node, str], child: Union[Node, str]) -> None:
        if isinstance(parent, str):
            parent_node = None
            for node in self.nodes:
                if node.name == parent:
                    parent_node = node
                    break
            if parent_node is None:
                raise ValueError(f"Node {parent} not found in the graph")
            parent = parent_node

        if isinstance(child, str):
            child_node = None
            for node in self.nodes:
                if node.name == child:
                    child_node = node
                    break
            if child_node is None:
                raise ValueError(f"Node {child} not found in the graph")
            child = child_node

        parent.add_child(child)
        child.add_parent(parent)