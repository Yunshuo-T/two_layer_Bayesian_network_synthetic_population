import numpy as np


def random_seeds(random_seed):

    rng = np.random.default_rng(random_seed)

    if random_seed is None:
        return None
    else:
        return int(rng.integers(0, np.iinfo(np.uint32).max))
