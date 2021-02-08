import sys
sys.path.append("../")
from config import Config
import numpy as np

class Regularizer(object):
    def __init__(self, config: Config):
        self.config = config
        self.rate = config.trainer.regularizer_rate
        self.max_time = config.model.maximum_time
    
    def __call__(self, params):
        return 0.
    
    def gradient(self, params):
        return tuple(np.zeros_like(param) for param in params)


class L1Regularizer(Regularizer):
    def __init__(self, config):
        super(L1Regularizer, self).__init__(config)
    
    def __call__(self, params):
        norm = 0.
        for param in params:
            if param.ndim == 2:
                norm += np.mean(np.abs(np.linalg.norm(param, axis=1)))
            elif param.ndim == 3:
                norm += np.mean(np.abs(np.linalg.norm(param, axis=(1, 2))))
        return self.rate * self.max_time * norm
    
    def gradient(self, params):
        return tuple(self.rate * param / np.linalg.norm(param, axis=1).reshape(-1, 1) if param.ndim==2 else self.rate * param / np.linalg.norm(param, axis=(1, 2)).reshape(-1, 1, 1) for param in params)


class L2Regularizer(Regularizer):
    def __init__(self, config: Config):
        super(L2Regularizer, self).__init__(config)
    
    def __call__(self, params):
        norm = 0.
        for param in params:
            if param.ndim == 2:
                norm += np.mean(np.square(np.linalg.norm(param, axis=1)))
            elif param.ndim == 3:
                norm += np.mean(np.square(np.linalg.norm(param, axis=(1, 2))))
        return self.rate * self.max_time * norm
    
    def gradient(self, params):
        return tuple(self.rate * param for param in params)


def get(config: Config):
    regularizer_type = config.trainer.regularizer_type
    if regularizer_type.lower() == "l1":
        return L1Regularizer(config)
    elif regularizer_type.lower() == "l2":
        return L2Regularizer(config)
    else:
        return Regularizer(config)