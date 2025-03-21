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
    
    def __len__(self) -> int:
        return len(self.scope)
    

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
        

def create_initial_factors(bn: BayesNetwork) -> list[Factor]:
    factors = []
    for curr_name, curr_vertex in bn.vertices.items():
        factor = Factor({curr_name}.union(curr_vertex.parents))

        if isinstance(curr_vertex.distribution, UnconditionalDistribution):
            for value, prob in curr_vertex.distribution.distribution.items():
                factor.add_distribution(frozenset([f'{curr_name}: {value}']), prob)

        elif isinstance(curr_vertex.distribution, ConditionalDistribution):
            for conditions, distribution in curr_vertex.distribution.distributions.items():
                for value, prob in distribution.distribution.items():
                    factor.add_distribution(conditions.union(frozenset([f'{curr_name}: {value}'])), prob)

        factors.append(factor)

    return factors


def restrict_evidence(evidence: dict[str, Any], factors: list[Factor]) -> None:
    for i, factor in enumerate(factors):
        if factor.scope.intersection(evidence):
            new_factor = Factor(factor.scope - set(evidence))
            for conditions, prob in factor.distributions.items():
                new_conditions = conditions - dict_to_frozenset(evidence)
                if len(new_conditions) == len(new_factor):
                    new_factor.add_distribution(new_conditions, prob)
            factors[i] = new_factor
        

def variable_elimination(bn: BayesNetwork, query: set[str], evidence: dict[str, Any]={}) -> dict[frozenset, float]:
    check_VE_inputs(bn, query, evidence)
    
    hidden = set(bn.vertices.keys()) - query - set(evidence.keys())
    factors = create_initial_factors(bn)
    restrict_evidence(evidence, factors)

    for factor in factors:
        print(factor)    

if __name__ == '__main__':
    bn = read_bayes_network_from_txt('ex.txt')
    
    result = variable_elimination(bn, {'A'}, {'C': True, 'B': False})
