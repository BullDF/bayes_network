from hmm import HiddenMarkovModel, read_hmm_from_txt
from typing import Any


def normalize(probs: dict) -> dict:
    return {value: prob / sum(probs.values()) for value, prob in probs.items()}


def viterbi_base(hmm: HiddenMarkovModel, initial_obs: Any) -> tuple[Any, float]:
    if initial_obs not in hmm.observation_domain:
        raise ValueError(f'Observation {initial_obs} not in observation domain.')

    probs = {}
    for initial_hidden in hmm.hidden_domain:
        probs[initial_hidden] = hmm.initial_distribution.distribution[initial_hidden] * hmm.emission_distribution.distributions[frozenset([f'Zt: {initial_hidden}'])].distribution[initial_obs]

    max_value = max(probs, key=probs.get)
    normalized_probs = normalize(probs)
    return max_value, normalized_probs[max_value]


def viterbi_step(hmm: HiddenMarkovModel, prev_hidden: Any, prev_prob: float, curr_obs: Any) -> tuple[Any, float]:
    if curr_obs not in hmm.observation_domain:
        raise ValueError(f'Observation {curr_obs} not in observation domain.')
    
    probs = {}
    for curr_hidden in hmm.hidden_domain:
        transition_prob = hmm.transition_distribution.distributions[frozenset([f'Zt-1: {prev_hidden}'])].distribution[curr_hidden]
        emission_prob = hmm.emission_distribution.distributions[frozenset([f'Zt: {curr_hidden}'])].distribution[curr_obs]
        probs[curr_hidden] = prev_prob * transition_prob * emission_prob
    max_value = max(probs, key=probs.get)
    normalized_probs = normalize(probs)
    return max_value, normalized_probs[max_value]


def viterbi(hmm: HiddenMarkovModel, observations: list) -> list:
    if len(observations) < 1:
        raise ValueError('At least one observation is required.')
    
    prev_hidden, prev_prob = viterbi_base(hmm, observations[0])
    seq = [prev_hidden]

    for obs in observations[1:]:
        prev_hidden, prev_prob = viterbi_step(hmm, prev_hidden, prev_prob, obs)
        seq.append(prev_hidden)

    return seq


if __name__ == "__main__":
    pass