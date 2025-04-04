from typing import Optional
import re
from collections import defaultdict

from .bayes_network import BayesNetwork
from .distribution import *
from .vertex import Vertex
from .utils import *


class HiddenMarkovModel(BayesNetwork):
    time_step: Optional[int]
    hidden_domain: Optional[set]
    observation_domain: Optional[set]
    initial_distribution: Optional[UnconditionalDistribution]
    transition_distribution: Optional[ConditionalDistribution]
    emission_distribution: Optional[ConditionalDistribution]

    def __init__(self) -> None:
        super().__init__()
        self.time_step = None
        self.hidden_domain = None
        self.observation_domain = None
        self.initial_distribution = None
        self.transition_distribution = None
        self.emission_distribution = None

    def set_time_step(self, time_step: int) -> None:
        if not isinstance(time_step, int) or time_step < 0:
            raise ValueError('Time step must be a non-negative integer.')
        self.time_step = time_step

    def set_hidden_domain(self, hidden_domain: set) -> None:
        if not isinstance(hidden_domain, set):
            raise ValueError('Hidden domain must be a set.')
        self.hidden_domain = hidden_domain
    
    def set_observation_domain(self, observation_domain: set) -> None:
        if not isinstance(observation_domain, set):
            raise ValueError('Observation domain must be a set.')
        self.observation_domain = observation_domain

    def set_initial_distribution(self, initial_distribution: UnconditionalDistribution) -> None:
        if not isinstance(initial_distribution, UnconditionalDistribution):
            raise ValueError('Initial distribution must be an UnconditionalDistribution.')
        self.initial_distribution = initial_distribution
    
    def set_transition_distribution(self, transition_distribution: ConditionalDistribution) -> None:
        if not isinstance(transition_distribution, ConditionalDistribution):
            raise ValueError('Hidden distribution must be a ConditionalDistribution.')
        self.transition_distribution = transition_distribution

    def set_emission_distribution(self, emission_distribution: ConditionalDistribution) -> None:
        if not isinstance(emission_distribution, ConditionalDistribution):
            raise ValueError('Emission distribution must be a ConditionalDistribution.')
        self.emission_distribution = emission_distribution


def read_time_step(hmm: HiddenMarkovModel, data: str) -> None:
    pattern = re.compile(r"T = (\d+)")
    match = pattern.search(data)
    time_step = int(match.group(1))
    hmm.set_time_step(time_step)

    
def read_variables(hmm: HiddenMarkovModel, data: str) -> None:
    pattern = re.compile(r"(\w+): \{([^}]+)\}")
    variables = {match.group(1): strs_to_domain(match.group(2).split(", ")) for match in pattern.finditer(data)}
    
    hmm.set_hidden_domain(variables['Z'])
    hmm.set_observation_domain(variables['X'])

    hmm.add_node(Vertex('Z0', hmm.hidden_domain))
    hmm.add_node(Vertex('X0', hmm.observation_domain))
    hmm.add_edge(hmm.vertices['Z0'], hmm.vertices['X0'])

    for i in range(1, hmm.time_step + 1):
        hmm.add_node(Vertex(f'Z{i}', hmm.hidden_domain))
        hmm.add_node(Vertex(f'X{i}', hmm.observation_domain))
        hmm.add_edge(hmm.vertices[f'Z{i - 1}'], hmm.vertices[f'Z{i}'])
        hmm.add_edge(hmm.vertices[f'Z{i}'], hmm.vertices[f'X{i}'])


def read_initial_distribution(hmm: HiddenMarkovModel, data: str) -> None:
    pattern = re.compile(r"P\(Z = (\w+)\) = ([\d.]+)")
    probabilities = {m.group(1): float(m.group(2)) for m in pattern.finditer(data)}
    distribution = UnconditionalDistribution(hmm.hidden_domain, {str_to_value(value): prob for value, prob in probabilities.items()})
    hmm.vertices['Z0'].set_distribution(distribution)
    hmm.set_initial_distribution(distribution)


def read_transition_distribution(hmm: HiddenMarkovModel, data: str) -> None:
    pattern = re.compile(r"P\(Z = (\w+) \| Z = (\w+)\) = ([\d.]+)")
    probabilities = {(m.group(1), m.group(2)): float(m.group(3)) for m in pattern.finditer(data)}
    distributions = defaultdict(lambda: defaultdict(dict))
    for (curr, prev), prob in probabilities.items():
        distributions[prev][curr] = prob

    transition_distribution = {}
    for prev, dist in distributions.items():
        key = frozenset([f'Zt-1: {prev}'])
        transition_distribution[key] = UnconditionalDistribution(hmm.hidden_domain, {str_to_value(value): prob for value, prob in dist.items()})
    hmm.set_transition_distribution(ConditionalDistribution(hmm.hidden_domain, transition_distribution))

    for i in range(1, hmm.time_step + 1):
        variable_distribution = {}
        for prev, dist in distributions.items():
            key = frozenset([f'Z{i - 1}: {prev}'])
            variable_distribution[key] = UnconditionalDistribution(hmm.hidden_domain, {str_to_value(value): prob for value, prob in dist.items()})

        hmm.vertices[f'Z{i}'].set_distribution(ConditionalDistribution(hmm.hidden_domain, variable_distribution))


def read_emission_distribution(hmm: HiddenMarkovModel, data: str) -> None:
    pattern = re.compile(r"P\(X = (\w+) \| Z = (\w+)\) = ([\d.]+)")
    probabilities = {(m.group(1), m.group(2)): float(m.group(3)) for m in pattern.finditer(data)}
    distributions = defaultdict(lambda: defaultdict(dict))
    for (obs, state), prob in probabilities.items():
        distributions[state][obs] = prob

    emission_distribution = {}
    for state, dist in distributions.items():
        key = frozenset([f'Zt: {state}'])
        emission_distribution[key] = UnconditionalDistribution(hmm.observation_domain, {str_to_value(value): prob for value, prob in dist.items()})
    hmm.set_emission_distribution(ConditionalDistribution(hmm.observation_domain, emission_distribution))

    for i in range(0, hmm.time_step + 1):
        variable_distribution = {}
        for state, dist in distributions.items():
            key = frozenset([f'Z{i}: {state}'])
            variable_distribution[key] = UnconditionalDistribution(hmm.observation_domain, {str_to_value(value): prob for value, prob in dist.items()})

        hmm.vertices[f'X{i}'].set_distribution(ConditionalDistribution(hmm.observation_domain, variable_distribution))


def read_hmm_from_txt(file_name: str) -> HiddenMarkovModel:
    hmm = HiddenMarkovModel()
    with open(file_name, 'r') as file:
        data = file.read()
        read_time_step(hmm, data)
        read_variables(hmm, data)
        read_initial_distribution(hmm, data)
        read_transition_distribution(hmm, data)
        read_emission_distribution(hmm, data)

    return hmm
        

if __name__ == "__main__":
    hmm = read_hmm_from_txt('../hmm_ex.txt')
    print(hmm.find_roots())
    print(hmm.time_step)
    print(hmm.hidden_domain)
    print(hmm.observation_domain)
    print(len(hmm))
    print(hmm({'Z0': 'c', 'X0': 0, 'Z1': 'h', 'X1': 1, 'Z2': 'h', 'X2': 2}))
