import numpy as np

class Sigmoid:
    def __init__(self, gain=1.):
        self.gain = gain
        self.function = np.vectorize(self._sigmoid, otypes=[np.float32])
        self.d_function = np.vectorize(self._derivative_sigmoid, otypes=[np.float32])
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return self.d_function(x)
    
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-self.gain * x))
    
    def _derivative_sigmoid(self, x):
        return self.gain * self._sigmoid(x) * (1. - self._sigmoid(x))


class ReLU:
    def __init__(self, gain=1.):
        self.gain = gain
        self.function = np.vectorize(self._relu, otypes=[np.float32])
        self.d_function = np.vectorize(self._derivative_relu, otypes=[np.float32])
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return self.d_function(x)
    
    def _relu(self, x):
        if x < 0:
            return 0.
        else:
            return self.gain * x
    
    def _derivative_relu(self, x):
        if x < 0:
            return 0.
        else:
            return self.gain


class Tanh:
    def __init__(self, gain=1.):
        self.gain = gain
        self.function = np.vectorize(self._tanh, otypes=[np.float32])
        self.d_function = np.vectorize(self._derivative_tanh, otypes=[np.float32])
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return self.d_function(x)
    
    def _tanh(self, x):
        return np.tanh(self.gain * x)
    
    def _derivative_tanh(self, x):
        return self.gain / np.cosh(self.gain * x)


def get(function_type):
    if function_type.lower() == "relu":
        relu = ReLU()
        return relu, relu.derivative
    elif function_type.lower() == "tanh":
        tanh = Tanh()
        return tanh, tanh.derivative
    else:
        sigmoid = Sigmoid()
        return sigmoid, sigmoid.derivative