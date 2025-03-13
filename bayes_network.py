from vertex import Vertex
from typing import Any


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

    def find_roots(self) -> list[Vertex]:
        return [vertex for vertex in self.vertices.values() if not vertex.parents]

    def __call__(self, *args) -> float:
        if len(args) != 1:
            raise ValueError('Computing joint probability requires exactly one argument.')
        values = args[0]
        if not isinstance(values, dict):
            raise ValueError('Values must be provided as a dictionary.')
        if len(values) != len(self.vertices):
            raise ValueError('Length of values must match number of vertices.')
        
        vertices = self.find_roots()
        seen = set(vertices)
        prob = 1
        
        while vertices:
            curr = vertices.pop()
            if curr.name not in values:
                raise ValueError(f'No value provided for vertex {curr.name}.')
            
            if not curr.parents:
                prob *= curr(values[curr.name])
            else:
                conditions = {}
                for parent in curr.parents:
                    conditions[parent] = values[parent]
                prob *= curr(values[curr.name], conditions)

            for child in curr.children.values():
                if child not in seen:
                    vertices.append(child)
                    seen.add(child)

        return prob