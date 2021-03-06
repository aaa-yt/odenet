import sys
sys.path.append("../")
from config import Config
import numpy as np

class MeanSquareError:
    def __call__(self, y_pred, y_true):
        return np.mean(np.sum(np.square(y_pred - y_true), 1)) * 0.5
    
    def gradient(self, y_pred, y_true):
        return (y_pred - y_true) / len(y_true)


class BinaryCrossEntropy:
    def __init__(self):
        self.eps = 1e-8

    def __call__(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + self.eps) + (1 - y_true) * np.log((1. - y_pred + self.eps)))
    
    def gradient(self, y_pred, y_true):
        return (np.divide(1 - y_true, 1. - y_pred + self.eps) - np.divide(y_true, y_pred+self.eps)) / len(y_true)


class CrossEntropy:
    def __init__(self):
        self.eps = 1e-8

    def __call__(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred + self.eps), 1))
    
    def gradient(self, y_pred, y_true):
        return -np.divide(y_true, y_pred+self.eps) / len(y_true)

def get(loss_type):
    if loss_type.lower() == "crossentropy":
        return CrossEntropy()
    elif loss_type.lower() == "binarycrossentropy":
        return BinaryCrossEntropy()
    else:
        return MeanSquareError()

class Accuracy:
    def __init__(self, config: Config):
        self.config = config
        self.dim_out = config.model.output_dimension
        if self.dim_out == 1:
            self.accuracy = self._binary_accuracy
        else:
            self.accuracy = self._categorical_accuracy
    
    def __call__(self, y_pred, y_true):
        return self.accuracy(y_pred, y_true)

    def _binary_accuracy(self, y_pred, y_true):
        return np.mean(np.equal(np.where(y_pred<0.5, 0, 1), y_true).astype(np.float32))
    
    def _categorical_accuracy(self, y_pred, y_true):
        return np.mean(np.equal(np.argmax(y_pred, 1), np.argmax(y_true, 1)).astype(np.float32))