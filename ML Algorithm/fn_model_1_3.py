from torch import nn as nn
from fn_try_gpu import *


def init_model():
    return nn.Sequential(nn.Linear(42, 32),
                         nn.ReLU(),
                         nn.Linear(32, 32),
                         nn.ReLU(),
                         nn.Linear(32, 16),
                         nn.ReLU(),
                         nn.Linear(16, 12),).to(device=try_gpu())
