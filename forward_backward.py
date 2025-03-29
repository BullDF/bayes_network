from hmm import HiddenMarkovModel, read_hmm_from_txt
from typing import Any


def normalize(probs: dict[Any, float]) -> dict[Any, float]:
    return {value: prob / sum(probs.values()) for value, prob in probs.items()}


def forward_base(hmm: HiddenMarkovModel, initial_obs: Any) -> dict[Any, float]:
    alpha = {}
    for value in hmm.hidden_domain:
        alpha[value] = hmm.initial_distribution(value) * hmm.emission_distribution(initial_obs, {'Zt': value})

    normalized_alpha = normalize(alpha)
    return normalized_alpha


def prediction_step(hmm: HiddenMarkovModel, prev_alpha: dict[Any, float]) -> dict[Any, float]:
    probs = {}
    for j in hmm.hidden_domain:
        sum_j = 0
        for i in hmm.hidden_domain:
            transition_prob = hmm.transition_distribution(j, {'Zt-1': i})
            sum_j += prev_alpha[i] * transition_prob
        probs[j] = sum_j

    normalized_probs = normalize(probs)
    return normalized_probs


def forward_step(hmm: HiddenMarkovModel, probs: dict[Any, float], curr_obs: Any) -> dict[Any, float]:
    alpha = {}
    for value in hmm.hidden_domain:
        alpha[value] = probs[value] * hmm.emission_distribution(curr_obs, {'Zt': value})

    normalized_alpha = normalize(alpha)
    return normalized_alpha


def forward(hmm: HiddenMarkovModel, observations: list, t: int=0) -> dict:
    if t < 0:
        raise ValueError('Time step t must be non-negative.')
    
    alpha = forward_base(hmm, observations[0])
    for i in range(1, t + 1):
        probs = prediction_step(hmm, alpha)
        alpha = forward_step(hmm, probs, observations[i])
    
    return alpha


if __name__ == "__main__":
    hmm = read_hmm_from_txt('hmm_ex.txt')
    print(forward(hmm, [1, 0, 1, 0, 0], t=3))