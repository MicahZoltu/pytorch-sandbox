from functools import partial

import torch

RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
random_tensor = partial(torch.randn, device=device)
empty_tensor = partial(torch.empty, device=device)
zero_tensor = partial(torch.zeros, device=device)
tensor = partial(torch.tensor, device=device)
