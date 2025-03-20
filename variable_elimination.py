from bayes_network import BayesNetwork, read_bayes_network_from_txt
from typing import Any
from utils import str_to_value
from distribution import *


class Factor:
    scope: set[str]
    distributions: dict[frozenset, float]

    def __init__(self, scope: set[str]):
        self.scope = scope
        self.distributions = {}

    def add_distribution(self, conditions: frozenset[str], prob: float) -> None:
        self.distributions[conditions] = prob
    
    def __str__(self) -> str:
        return f'{self.scope}: {self.distributions}'
    

def check_VE_inputs(bn: BayesNetwork, query: set[str], evidence: dict[str, Any]) -> None:
    if len(set(evidence)) != len(evidence):
        raise ValueError('Evidence must be unique.')
    
    if not query:
        raise ValueError('Query cannot be empty.')
    
    for vertex in query:
        if vertex in evidence:
            raise ValueError(f'Query variable {vertex} cannot be in evidence.')
        if vertex not in bn.vertices:
            raise ValueError(f'Query variable {vertex} not in Bayes Network.')
        
    for vertex, value in evidence.items():
        if vertex not in bn.vertices:
            raise ValueError(f'Evidence variable {vertex} not in Bayes Network.')
        if value not in bn.vertices[vertex].domain:
            raise ValueError(f'Evidence value {value} not in domain of variable {vertex}.')
        

def create_initial_factors(bn: BayesNetwork, evidence: dict[str, Any]) -> list[Factor]:
    factors = []
    for curr_name, curr_vertex in bn.vertices.items():
        factor = Factor({curr_name}.union(curr_vertex.parents.keys()))

        if isinstance(curr_vertex.distribution, UnconditionalDistribution):
            for value, prob in curr_vertex.distribution.distribution.items():
                if curr_name in evidence and value != evidence[curr_name]:
                    continue
                
                factor.add_distribution(frozenset([f'{curr_name}: {value}']), prob)

        elif isinstance(curr_vertex.distribution, ConditionalDistribution):
            for conditions, distribution in curr_vertex.distribution.distributions.items():
                excluded = False
                for condition in conditions:
                    colon = condition.index(':')
                    vertex = condition[:colon]
                    value = str_to_value(condition[colon + 2:])
                    if vertex in evidence and value != evidence[vertex]:
                        excluded = True
                        break

                if not excluded:
                    for value, prob in distribution.distribution.items():
                        if curr_name in evidence and value != evidence[curr_name]:
                            continue

                        factor.add_distribution(frozenset([f'{curr_name}: {value}']).union(conditions), prob)

        print(factor)
        factors.append(factor)

    return factors
        

def variable_elimination(bn: BayesNetwork, query: set[str], evidence: dict[str, Any]={}) -> dict[frozenset, float]:
    check_VE_inputs(bn, query, evidence)
    
    hidden = set(bn.vertices.keys()) - query - set(evidence.keys())
    factors = create_initial_factors(bn, evidence)
    

if __name__ == '__main__':
    bn = read_bayes_network_from_txt('ex.txt')
    
    result = variable_elimination(bn, ['A'], {'C': True, 'B': False})
