from torch import nn as nn
from fn_try_gpu import *


def init_model():
    return nn.Sequential(nn.Linear(42, 512),
                         nn.ReLU(),
                         nn.Dropout(p=0.4),
                         nn.Linear(512, 256),
                         nn.ReLU(),
                         nn.Dropout(p=0.2),
                         nn.Linear(256, 64),
                         nn.ReLU(),
                         nn.Dropout(p=0.1),
                         nn.Linear(64, 12),).to(device=try_gpu())
