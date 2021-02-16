import os
import json
from logging import getLogger
import numpy as np

from config import Config
from lib import initializers
from lib import functions
from lib import solvers

logger = getLogger(__name__)

class ODENetModel:
    def __init__(self, config: Config):
        self.config = config
        self.dim_in = config.model.input_dimension
        self.dim_out = config.model.output_dimension
        self.max_time = config.model.maximum_time
        self.division = config.model.weights_division
        self.params = None
        self.P = None
        self.Q = None
        self.function = None
        self.d_function = None
        self.t = np.linspace(0., self.max_time, self.division)
        self.solver = None
    
    def __call__(self, xi):
        def func(t, x, params, function, division):
            index = int(t * (division - 1))
            return params[0][index] * function(np.dot(x, params[1][index].T) + params[2][index])
        
        self.x = self.solver(func, self.t, np.dot(xi, self.Q.T), args=(self.params, self.function, self.division))
        return np.dot(self.x[-1], self.P.T)
    
    def gradient(self, loss_grad, params_reg):
        def func(t, lam, params, function, x, division):
            index = int(t * (division - 1))
            return -np.dot(lam * params[0][index] * function(np.dot(x[index], params[1][index].T) + params[2][index]), params[1][index])

        lam = self.solver(func, self.t[::-1], np.dot(loss_grad, self.P), args=(self.params, self.d_function, self.x, self.division))
        z = np.einsum('ikl,ijl->ijk', self.params[1], self.x) + self.params[2][:, np.newaxis, :]
        gradient_a = np.sum(lam[::-1] * self.function(z), 1) + params_reg[0]
        gradient_W = np.einsum('ilj,ilk->ijk', lam[::-1] * self.params[0][:, np.newaxis, :] * self.d_function(z), self.x) + params_reg[1]
        gradient_b = np.sum(lam[::-1] * self.params[0][:, np.newaxis, :] * self.d_function(z), 1) + params_reg[2]
        return (gradient_a, gradient_W, gradient_b)

    def load(self, model_path):
        flag = True
        if os.path.exists(model_path):
            logger.debug("loding model from {}".format(model_path))
            with open(model_path, "rt") as f:
                model = json.load(f)
            a = np.array(model.get("a"), dtype=np.float32)
            W = np.array(model.get("W"), dtype=np.float32)
            b = np.array(model.get("b"), dtype=np.float32)
            P = np.array(model.get("P"), dtype=np.float32)
            Q = np.array(model.get("Q"), dtype=np.float32)
            if (a.shape == (self.division, self.dim_in+self.dim_out)) and (W.shape == (self.division, self.dim_in+self.dim_out, self.dim_in+self.dim_out)) and (b.shape == (self.division, self.dim_in+self.dim_out)) and (P.shape == (self.dim_out, self.dim_in+self.dim_out)) and (Q.shape == (self.dim_in+self.dim_out, self.dim_in)):
                flag = False
                self.params = (a, W, b)
                self.P = P
                self.Q = Q
        if flag:
            logger.info("initialize parameter with {}".format(self.config.model.initializer_type))
            initializer = initializers.get(self.config)
            a = initializer((self.division, self.dim_in+self.dim_out))
            W = initializer((self.division, self.dim_in+self.dim_out, self.dim_in+self.dim_out))
            b = initializer((self.division, self.dim_in+self.dim_out))
            self.params = (a, W, b)
            self.P = np.eye(self.dim_out, self.dim_in+self.dim_out, k=self.dim_in, dtype=np.float32)
            self.Q = np.eye(self.dim_in+self.dim_out, self.dim_in, dtype=np.float32)
        self.function, self.d_function = functions.get(self.config.model.function_type)
        self.solver = solvers.get(self.config.model.solver_type)

    def save(self, config_path, model_path):
        logger.debug("save model config to {}".format(config_path))
        self.config.save_parameter(config_path)
        logger.debug("save model to {}".format(model_path))
        model = {
            "a": self.params[0].tolist(),
            "W": self.params[1].tolist(),
            "b": self.params[2].tolist(),
            "P": self.P.tolist(),
            "Q": self.Q.tolist()
        }
        with open(model_path, "wt") as f:
            json.dump(model, f, indent=4)
