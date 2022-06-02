def set_random_seed():
    # Set an arbitrary seed value
    seed_value = 21

    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    import random
    random.seed(seed_value)

    import numpy as np
    np.random.seed(seed_value)

    import torch
    torch.manual_seed(seed_value)

    print('Random seed set')

set_random_seed()

