from bayes_network import BayesNetwork
from typing import Any, Union
from distribution import *
import random
from utils import dict_to_frozenset
from collections import deque


def ancestral_sampling(bn: BayesNetwork, n: int=1) -> Union[dict[str, Any], list[dict[str, Any]]]:
    if n == 1:
        vertices = deque(bn.find_roots())
        seen = {root.name for root in vertices}
        sample = {}

        while vertices:
            curr = vertices.popleft()
            dist = curr.distribution
            if dist is None:
                raise ValueError(f'Vertex {curr.name} has no distribution.')
            
            if isinstance(dist, UncondDist):
                domain = []
                probs = []
                for value, prob in dist.dist.items():
                    domain.append(value)
                    probs.append(prob)
                
                sample[curr.name] = random.choices(domain, probs)[0]
            
            elif isinstance(dist, CondDist):
                conditions = dict_to_frozenset({parent: sample[parent] for parent in curr.parents})

                if conditions not in dist.dists:
                    vertices.append(curr)
                    continue

                domain = []
                probs = []
                for value, prob in dist.dists[conditions].dist.items():
                    domain.append(value)
                    probs.append(prob)

                sample[curr.name] = random.choices(domain, probs)[0]

            for child in curr.children.values():
                if child.name not in seen:
                    vertices.append(child)
                    seen.add(child.name)

        return sample
    
    else:
        return [ancestral_sampling(bn) for _ in range(n)]


if __name__ == '__main__':
    pass