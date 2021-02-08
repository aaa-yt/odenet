import sys
sys.path.append("../")
from config import Config
import numpy as np

class Optimizer(object):
    def __init__(self, config: Config):
        self.config = config


class SGD(Optimizer):
    def __init__(self, config: Config):
        super(SGD, self).__init__(config)
        self.rate = self.config.trainer.learning_rate
    
    def __call__(self, params, g_params):
        return tuple(param - self.rate * g_param for param, g_param in zip(params, g_params))


class Momentum(Optimizer):
    def __init__(self, config: Config, params):
        super(Momentum, self).__init__(config)
        self.rate = self.config.trainer.learning_rate
        self.momentum = self.config.trainer.momentum
        self.v = tuple(np.zeros_like(param) for param in params)
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(param + self.momentum * v - self.rate * g_param, self.momentum * v - self.rate * g_param) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


def get(config: Config, params=None):
    optimizer_type = config.trainer.optimizer_type.lower()
    if params is not None:
        if optimizer_type == "momentum":
            return Momentum(config, params)
    return SGD(config)