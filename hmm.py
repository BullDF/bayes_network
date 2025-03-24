from bayes_network import BayesNetwork
from distribution import *
from typing import Optional


class HiddenMarkovModel(BayesNetwork):
    hidden_domain: set
    observation_domain: set
    initial_distribution: Optional[UnconditionalDistribution]
    transition_distribution: Optional[ConditionalDistribution]
    emission_distribution: Optional[ConditionalDistribution]

    def __init__(self, hidden_domain: set, observation_domain: set):
        self.hidden_domain = hidden_domain
        self.observation_domain = observation_domain
        self.initial_distribution = None
        self.transition_distribution = None
        self.emission_distribution = None

    def set_initial_distribution(self, initial_distribution: UnconditionalDistribution) -> None:
        if initial_distribution.domain != self.hidden_domain:
            raise ValueError('Initial distribution domain must match hidden domain.')
        self.initial_distribution = initial_distribution

    def set_transition_distribution(self, transition_distribution: ConditionalDistribution) -> None:
        if transition_distribution.domain != self.hidden_domain:
            raise ValueError('Transition distribution domain must match hidden domain.')
        self.transition_distribution = transition_distribution

    def set_emission_distribution(self, emission_distribution: ConditionalDistribution) -> None:
        if emission_distribution.domain != self.observation_domain:
            raise ValueError('Emission distribution domain must match observation domain.')
        self.emission_distribution = emission_distribution

    

    