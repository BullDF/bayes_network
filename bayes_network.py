from vertex import Vertex


class BayesNetwork:
    vertices: dict[str, Vertex]

    def __init__(self) -> None:
        self.vertices = {}

    def __len__(self) -> int:
        return len(self.vertices)

    def add_node(self, vertex: Vertex) -> None:
        if vertex.name in self.vertices:
            raise ValueError(f'Vertex {vertex.name} already exists in the network.')
        self.vertices[vertex.name] = vertex

    def add_edge(self, parent: Vertex, child: Vertex) -> None:
        parent.add_child(child)
        child.add_parent(parent)

    def joint_probability(self):
        pass
