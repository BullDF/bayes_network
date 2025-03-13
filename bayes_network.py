from vertex import Vertex


class BayesNetwork:
    nodes: list[Vertex]

    def __init__(self) -> None:
        self.nodes = []

    def __len__(self) -> int:
        return len(self.nodes)

    def add_node(self, vertex: Vertex) -> None:
        for node in self.nodes:
            if node.name == vertex.name:
                raise ValueError(f'Vertex {vertex.name} is already in the network.')
        self.nodes.append(vertex)

    def add_edge(self, parent: Vertex, child: Vertex) -> None:
        parent.add_child(child)
        child.add_parent(parent)

    def joint_probability(self):
        pass
