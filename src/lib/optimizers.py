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


class Nesterov(Optimizer):
    def __init__(self, config: Config, params):
        super(Nesterov, self).__init__(config)
        self.rate = self.config.trainer.learning_rate
        self.momentum = self.config.trainer.momentum
        self.v = tuple(np.zeros_like(param) for param in params)
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(param + self.momentum * (self.momentum * v - self.rate * g_param) - self.rate * g_param, self.momentum * v - self.rate * g_param) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


class AdaGrad(Optimizer):
    def __init__(self, config: Config, params):
        super(AdaGrad, self).__init__(config)
        self.rate = self.config.trainer.learning_rate
        self.v = tuple(np.zeros_like(param) for param in params)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(param - self.rate * np.divide(g_param, np.sqrt(v + np.square(g_param) + self.eps).astype(np.float32)), v + np.square(g_param)) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


class RMSprop(Optimizer):
    def __init__(self, config: Config, params):
        super(RMSprop, self).__init__(config)
        self.rate = self.config.trainer.learning_rate
        self.decay = self.config.trainer.decay
        self.v = tuple(np.zeros_like(param) for param in params)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(param - self.rate * np.divide(g_param, np.sqrt(self.decay * v + (1. - self.decay) * np.square(g_param) + self.eps).astype(np.float32)), self.decay * v + (1. - self.decay) * np.square(g_param)) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


class AdaDelta(Optimizer):
    def __init__(self, config: Config, params):
        super(AdaDelta, self).__init__(config)
        self.decay = self.config.trainer.decay
        self.v = tuple(np.zeros_like(param) for param in params)
        self.u = tuple(np.zeros_like(param) for param in params)
        self.params_prev = tuple(np.zeros_like(param) for param in params)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v, new_u = zip(*[(param - np.multiply(np.divide(np.sqrt(u + self.eps).astype(np.float32), np.sqrt(self.decay * v + (1. - self.decay) * np.square(g_param) + self.eps).astype(np.float32)), g_param), self.decay * v + (1. - self.decay) * np.square(g_param), self.decay * u + (1. - self.decay) * (param - param_prev)) for param, g_param, v, u, param_prev in zip(params, g_params, self.v, self.u, self.params_prev)])
        self.params_prev = params
        self.v = new_v
        self.u = new_u
        return new_params

class Adam(Optimizer):
    def __init__(self, config: Config, params):
        super(Adam, self).__init__(config)
        self.rate = self.config.trainer.learning_rate
        self.decay = self.config.trainer.decay
        self.decay2 = self.config.trainer.decay2
        self.v = tuple(np.zeros_like(param) for param in params)
        self.m = tuple(np.zeros_like(param) for param in params)
        self.t = 1
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_m, new_v = zip(*[(param - self.rate * np.divide((self.decay * m + (1. - self.decay) * g_param) / (1. - self.decay ** self.t), np.sqrt(((self.decay2 * v + (1. - self.decay2) * np.square(g_param)) / (1. - self.decay2))+ self.eps).astype(np.float32)), self.decay * m + (1. - self.decay) * g_param, self.decay2 * v + (1. - self.decay2) * np.square(g_param)) for param, g_param, m, v in zip(params, g_params, self.m, self.v)])
        self.m = new_m
        self.v = new_v
        return new_params


def get(config: Config, params=None):
    optimizer_type = config.trainer.optimizer_type.lower()
    if params is not None:
        if optimizer_type == "momentum":
            return Momentum(config, params)
        elif optimizer_type == "nesterov":
            return Nesterov(config, params)
        elif optimizer_type == "adagrad":
            return AdaGrad(config, params)
        elif optimizer_type == "rmsprop":
            return RMSprop(config, params)
        elif optimizer_type == "adadelta":
            return AdaDelta(config, params)
        elif optimizer_type == "adam":
            return Adam(config, params)
    return SGD(config)