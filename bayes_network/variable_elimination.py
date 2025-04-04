from typing import Any

from .bayes_network import BayesNetwork, read_bayes_network_from_txt
from .distribution import *


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
    
    def __repr__(self) -> str:
        return self.__str__()
    

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


def restrict_evidence(evidence: dict[str, Any], factors: list[Factor]) -> list[Factor]:
    new_factors = []
    for factor in factors:
        if factor.scope.intersection(evidence):
            new_factor = Factor(factor.scope - set(evidence))
            for conditions, prob in factor.distributions.items():
                new_conditions = conditions - dict_to_frozenset(evidence)
                if len(new_conditions) == len(new_factor):
                    new_factor.add_distribution(new_conditions, prob)
            if new_factor.scope:
                new_factors.append(new_factor)
        else:
            new_factors.append(factor)

    return new_factors


def combine_factors(factors: list[Factor]) -> Factor:
    while len(factors) > 1:
        factor1 = factors.pop()
        factor2 = factors.pop()
        
        common_variables = len(factor1.scope.intersection(factor2.scope))

        new_factor = Factor(factor1.scope.union(factor2.scope))

        for conditions1, prob1 in factor1.distributions.items():
            for conditions2, prob2 in factor2.distributions.items():
                if len(conditions1.intersection(conditions2)) == common_variables:
                    new_conditions = conditions1.union(conditions2)
                    prob = prob1 * prob2
                    new_factor.add_distribution(new_conditions, prob)

        factors.append(new_factor)

    return factors[0]


def eliminate_hidden_variables(bn: BayesNetwork, query: set[str], evidence: dict[str, Any], factors: list[Factor]) -> None:
    hidden = set(bn.vertices.keys()) - query - set(evidence.keys())

    for variable in hidden:
        new_factors = []
        curr_factors = []

        for factor in factors:
            if variable in factor.scope:
                curr_factors.append(factor)
            else:
                new_factors.append(factor)

        curr_factor = combine_factors(curr_factors)
        new_factor = Factor(curr_factor.scope - {variable})

        while curr_factor.distributions:
            new_prob = 0
            conditions, prob = curr_factor.distributions.popitem()
            new_prob += prob

            for condition in conditions:
                colon = condition.index(':')
                if condition[:colon] == variable:
                    new_conditions = conditions - {condition}
                    break
            
            for value in bn.vertices[variable].domain:
                if new_conditions.union({f'{variable}: {value}'}) in curr_factor.distributions:
                    new_prob += curr_factor.distributions.pop(new_conditions.union({f'{variable}: {value}'}))
            
            new_factor.add_distribution(new_conditions, new_prob)

        new_factors.append(new_factor)
        factors = new_factors

    return factors

def normalize_probabilities(factor: Factor) -> dict[frozenset, float]:
    return {conditions: prob / sum(factor.distributions.values()) for conditions, prob in factor.distributions.items()}
        

def variable_elimination(bn: BayesNetwork, query: set[str], evidence: dict[str, Any]={}) -> dict[frozenset, float]:
    check_VE_inputs(bn, query, evidence)
    factors = create_initial_factors(bn)
    factors = restrict_evidence(evidence, factors)
    factors = eliminate_hidden_variables(bn, query, evidence, factors)
    factor = combine_factors(factors)

    return normalize_probabilities(factor)


if __name__ == '__main__':
    bn = read_bayes_network_from_txt('../bn_ex.txt')
    print(variable_elimination(bn, {'A', 'E'}, {'D': 0}))
