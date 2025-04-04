from typing import Any

from .hmm import HiddenMarkovModel, read_hmm_from_txt
from .variable_elimination import variable_elimination


def normalize(probs: dict[Any, float]) -> dict[Any, float]:
    return {value: prob / sum(probs.values()) for value, prob in probs.items()}


def forward_base(hmm: HiddenMarkovModel, initial_obs: Any) -> dict[Any, float]:
    if initial_obs not in hmm.observation_domain:
        raise ValueError(f'Observation {initial_obs} not in observation domain.')
    
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
    if curr_obs not in hmm.observation_domain:
        raise ValueError(f'Observation {curr_obs} not in observation domain.')
    
    alpha = {}
    for value in hmm.hidden_domain:
        alpha[value] = probs[value] * hmm.emission_distribution(curr_obs, {'Zt': value})

    normalized_alpha = normalize(alpha)
    return normalized_alpha


def forward(hmm: HiddenMarkovModel, observations: list, t: int=0) -> dict[Any, float]:
    if t < 0:
        raise ValueError('Time step t must be non-negative.')
    
    alpha = forward_base(hmm, observations[0])
    for i in range(1, t + 1):
        probs = prediction_step(hmm, alpha)
        alpha = forward_step(hmm, probs, observations[i])
    
    return alpha


def filtering(hmm: HiddenMarkovModel, observations: list) -> dict[Any, float]:
    return forward(hmm, observations, t=len(observations) - 1)


def backward_step(hmm: HiddenMarkovModel, prev_beta: dict[Any, float], curr_obs: Any) -> dict[Any, float]:
    if curr_obs not in hmm.observation_domain:
        raise ValueError(f'Observation {curr_obs} not in observation domain.')
    
    beta = {}
    for i in hmm.hidden_domain:
        sum_i = 0
        for j in hmm.hidden_domain:
            transition_prob = hmm.transition_distribution(j, {'Zt-1': i})
            emission_prob = hmm.emission_distribution(curr_obs, {'Zt': j})
            sum_i += prev_beta[j] * transition_prob * emission_prob
        beta[i] = sum_i

    normalized_beta = normalize(beta)
    return normalized_beta


def backward(hmm: HiddenMarkovModel, observations: list, t: int=0) -> dict[Any, float]:
    if t < 0:
        raise ValueError('Time step t must be non-negative.')
    if t >= len(observations) - 1:
        raise ValueError('Time step t must be less than the length of observations minus 1.')
    
    beta = {value: 1 / len(hmm.hidden_domain) for value in hmm.hidden_domain}

    for i in range(len(observations) - 2, t - 1, -1):
        beta = backward_step(hmm, beta, observations[i + 1])

    return beta


def forward_backward(hmm: HiddenMarkovModel, observations: list, t: int=0) -> dict[Any, float]:
    alpha = forward(hmm, observations, t)
    beta = backward(hmm, observations, t)
    
    return normalize({value: alpha[value] * beta[value] for value in hmm.hidden_domain})


def smoothing(hmm: HiddenMarkovModel, observations: list, t: int) -> dict[Any, float]:
    return forward_backward(hmm, observations, t)


if __name__ == "__main__":
    hmm = read_hmm_from_txt('../hmm_ex_t7.txt')
    print(filtering(hmm, [0, 1, 2, 1, 1, 2, 0]))
    print(smoothing(hmm, [0, 1, 2, 1, 1, 2, 0], 2))
    print(smoothing(hmm, [0, 1, 2, 1, 1, 2, 0], 5))

    print(variable_elimination(hmm, {'Z6'}, {'X0': 0, 'X1': 1, 'X2': 2, 'X3': 1, 'X4': 1, 'X5': 2, 'X6': 0}))
    print(variable_elimination(hmm, {'Z2'}, {'X0': 0, 'X1': 1, 'X2': 2, 'X3': 1, 'X4': 1, 'X5': 2, 'X6': 0}))
    print(variable_elimination(hmm, {'Z5'}, {'X0': 0, 'X1': 1, 'X2': 2, 'X3': 1, 'X4': 1, 'X5': 2, 'X6': 0}))

    import time
    trials = 100
    
    fb_total_time = 0
    for _ in range(trials):
        start_time = time.time()
        filtering(hmm, [0, 1, 2, 1, 1, 2, 0])
        fb_total_time += time.time() - start_time
    print(fb_total_time / trials)

    vea_total_time = 0
    for _ in range(trials):
        start_time = time.time()
        variable_elimination(hmm, {'Z6'}, {'X0': 0, 'X1': 1, 'X2': 2, 'X3': 1, 'X4': 1, 'X5': 2, 'X6': 0})
        vea_total_time += time.time() - start_time
    print(vea_total_time / trials)
