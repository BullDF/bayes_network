import re
from collections import defaultdict

from .vertex import Vertex
from .utils import strs_to_domain, str_to_value
from .distribution import UnconditionalDistribution, ConditionalDistribution


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
        if parent.name not in self.vertices:
            raise ValueError(f'Parent vertex {parent.name} does not exist in the network.')
        if child.name not in self.vertices:
            raise ValueError(f'Child vertex {child.name} does not exist in the network.')
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
            raise ValueError('Length of values must match number of vertices. For joint probabilities with fewer vertices or conditional probabilities, please use the Variable Elimination algorithm.')
        
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
    

def read_variables(bn: BayesNetwork, data: str) -> None:
    pattern = re.compile(r"(\w+): \{([^}]+)\}")
    variables = {match.group(1): strs_to_domain(match.group(2).split(", ")) for match in pattern.finditer(data)}
    for variable, domain in variables.items():
        bn.add_node(Vertex(variable, domain))


def read_edges(bn: BayesNetwork, data: str) -> None:
    pattern = re.compile(r"(\w+) -> (\w+)")
    edges = [(match.group(1), match.group(2)) for match in pattern.finditer(data)]
    for parent, child in edges:
        bn.add_edge(bn.vertices[parent], bn.vertices[child])


def read_unconditional_probabilities(bn: BayesNetwork, data: str) -> None:
    pattern = re.compile(r"P\((\w+) = (\w+)\) = ([\d.]+)")
    probabilities = {(m.group(1), m.group(2)): float(m.group(3)) for m in pattern.finditer(data)}
    domains = defaultdict(set)
    distributions = defaultdict(dict)
    for (variable, value), prob in probabilities.items():
        value = str_to_value(value)
        domains[variable].add(value)
        distributions[variable][value] = prob

    for variable in domains:
        bn.vertices[variable].set_distribution(UnconditionalDistribution(
            domain=domains[variable],
            distribution=distributions[variable]
        ))


def read_conditional_probabilities(bn: BayesNetwork, data: str) -> None:
    pattern = re.compile(r"P\((\w+) = (\w+) \| (.+?)\) = ([\d.]+)")
    probabilities = {(m.group(1), m.group(2), tuple(re.sub(r"(\w+) = (\w+)", r"\1: \2", m.group(3)).split(", "))): float(m.group(4)) for m in pattern.finditer(data)}
    
    domains = defaultdict(set)
    distributions = defaultdict(lambda: defaultdict(dict))
    for (variable, value, conditions), prob in probabilities.items():
        value = str_to_value(value)
        domains[variable].add(value)
        distributions[variable][frozenset(conditions)][value] = prob
    
    for variable in domains:
        domain = domains[variable]
        variable_distribution = ConditionalDistribution(domain, {conditions: UnconditionalDistribution(domain, probs) for conditions, probs in distributions[variable].items()})
        bn.vertices[variable].set_distribution(variable_distribution)


def read_bayes_network_from_txt(file_name: str) -> BayesNetwork:
    bn = BayesNetwork()
    
    with open(file_name, 'r') as file:
        data = file.read()
        read_variables(bn, data)
        read_edges(bn, data)
        read_unconditional_probabilities(bn, data)
        read_conditional_probabilities(bn, data)

    return bn


if __name__ == '__main__':
    bn = read_bayes_network_from_txt('../bn_ex.txt')
    print(bn.find_roots())
    print(len(bn))
    print(bn({'A': 1, 'B': 0, 'C': 1, 'D': 1, 'E': 0}))
    