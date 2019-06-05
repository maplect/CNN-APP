import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
class network:
    def __init__(self,params):
        self.params = params
    def predict(self, x):
        w1 = self.params['w1']
        w2 = self.params['w2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        z1 = np.dot(x, w1) + b1
        z2 = sigmoid(z1)
        z3 = np.dot(z2,w2) + b2
        y = softmax(z3)
        return y
