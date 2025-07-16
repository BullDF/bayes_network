# STA410 Bayes Network

A Python package for Bayesian Networks and Hidden Markov Models with exact and approximate inference algorithms.

## Documentation

Please click [here](https://github.com/BullDF/bayes_network/blob/main/documentation.pdf) for the complete documentation of this project.

## Installation

```bash
pip install -e .
```

## Quick Start

### Bayesian Networks

```python
from bayes_network_bulldf import BayesNetwork, read_bayes_network_from_txt
from bayes_network_bulldf import variable_elimination, ancestral_sampling

# Load network from file
bn = read_bayes_network_from_txt('bn_ex.txt')

# Exact inference
result = variable_elimination(bn, query={'A', 'E'}, evidence={'D': 0})

# Approximate inference
samples = ancestral_sampling(bn, n=1000)

# Joint probability
prob = bn({'A': 1, 'B': 0, 'C': 1, 'D': 1, 'E': 0})
```

### Hidden Markov Models

```python
from bayes_network_bulldf import HiddenMarkovModel, read_hmm_from_txt
from bayes_network_bulldf import viterbi, filtering, smoothing

# Load HMM from file
hmm = read_hmm_from_txt('hmm_ex.txt')

# Most likely sequence (Viterbi)
sequence = viterbi(hmm, observations=[0, 1, 2, 1])

# Current state estimation (filtering)
current_state = filtering(hmm, observations=[0, 1, 2])

# Past state estimation (smoothing)
past_state = smoothing(hmm, observations=[0, 1, 2, 1], t=1)
```

## Key APIs

### Core Classes
- `BayesNetwork`: Main class for Bayesian networks
- `HiddenMarkovModel`: Extends BayesNetwork for temporal models
- `Vertex`: Represents nodes with domains and distributions

### Inference Algorithms
- `variable_elimination(bn, query, evidence)`: Exact inference
- `ancestral_sampling(bn, n)`: Generate samples from joint distribution
- `viterbi(hmm, observations)`: Most likely hidden state sequence
- `filtering(hmm, observations)`: Current state probabilities
- `smoothing(hmm, observations, t)`: Past state probabilities

### File I/O
- `read_bayes_network_from_txt(filename)`: Load Bayesian network
- `read_hmm_from_txt(filename)`: Load Hidden Markov Model

## Requirements

- Python 3.7+
- No external dependencies