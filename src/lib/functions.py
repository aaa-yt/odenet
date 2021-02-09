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


class Softplus:
    def __init__(self):
        self.function = np.vectorize(self._softplus, otypes=[np.float32])
        self.d_function = np.vectorize(self._derivative_softplus, otypes=[np.float32])
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return self.d_function(x)
    
    def _softplus(self, x):
        return np.log(1. + np.exp(x))
    
    def _derivative_softplus(self, x):
        return 1. / (1. + np.exp(-x))

class Polynomial:
    def __init__(self, n=2):
        self.n = n
        self.function = np.vectorize(self._polynomial, otypes=[np.float32])
        self.d_function = np.vectorize(self._derivative_polynomial, otypes=[np.float32])
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return self.d_function(x)
    
    def _polynomial(self, x):
        return x**self.n
    
    def _derivative_polynomial(self, x):
        return self.n * x**(self.n-1.)

class Gaussian:
    def __init__(self):
        self.function = np.vectorize(self._gaussian, otypes=[np.float32])
        self.d_function = np.vectorize(self._derivative_gaussian, otypes=[np.float32])
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return self.d_function(x)
    
    def _gaussian(self, x):
        return np.exp(-x * x / 2.) / np.sqrt(2. * np.pi)
    
    def _derivative_gaussian(self, x):
        return -x * np.sqrt(-x * x / 2.) / np.sqrt(2. * np.pi)


def get(function_type):
    if function_type.lower() == "relu":
        relu = ReLU()
        return relu, relu.derivative
    elif function_type.lower() == "tanh":
        tanh = Tanh()
        return tanh, tanh.derivative
    elif function_type.lower() == "softplus":
        softplus = Softplus()
        return softplus, softplus.derivative
    elif function_type.lower() == "polynomial":
        polynomial = Polynomial()
        return polynomial, polynomial.derivative
    elif function_type.lower() == "gaussian":
        gaussian = Gaussian()
        return gaussian, gaussian.derivative
    else:
        sigmoid = Sigmoid()
        return sigmoid, sigmoid.derivative