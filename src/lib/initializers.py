import sys
sys.path.append("../")
from config import Config
import numpy as np

class Initializer(object):
    def __init__(self, config: Config):
        self.config = config
        self.parameter = config.model.initializer_parameter


class ZeroInitializer(Initializer):
    def __init__(self, config: Config):
        super(ZeroInitializer, self).__init__(config)
    
    def __call__(self, shape):
        return np.zeros(shape=shape, dtype=np.float32)


class UniformInitializer(Initializer):
    def __init__(self, config: Config):
        super(UniformInitializer, self).__init__(config)
    
    def __call__(self, shape):
        return np.random.uniform(low=-self.parameter, high=self.parameter, size=shape).astype(np.float32)


class NormalInitializer(Initializer):
    def __init__(self, config: Config):
        super().__init__(config)
    
    def __call__(self, shape):
        return np.random.normal(scale=self.parameter, size=shape).astype(np.float32)
    

class ConstantInitializer(Initializer):
    def __init__(self, config: Config):
        super(ConstantInitializer, self).__init__(config)
    
    def __call__(self, shape):
        return self.parameter * np.ones(shape=shape, dtype=np.float32)


class GlorotUniformInitializer(Initializer):
    def __init__(self, config: Config):
        super(GlorotUniformInitializer, self).__init__(config)
        self.parameter = np.sqrt(2. / (config.model.input_dimension + config.model.output_dimension))
    
    def __call__(self, shape):
        return np.random.uniform(low=-self.parameter, high=self.parameter, size=shape).astype(np.float32)


def get(config: Config):
    initializer_type = config.model.initializer_type.lower()
    if initializer_type == "uniform":
        return UniformInitializer(config)
    elif initializer_type == "normal":
        return NormalInitializer(config)
    elif initializer_type == "constant":
        return ConstantInitializer(config)
    elif initializer_type == "glorot_uniform":
        return GlorotUniformInitializer(config)
    else:
        return ZeroInitializer(config)
    