import numpy as np
import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self,input_size,output_size,hide_size,weight_std =0.01):
        self.params = {}
        self.params['w1'] = weight_std * \
                            np.random.randn(input_size, hide_size)
        self.params['b1'] = np.zeros(hide_size)
        self.params['w2'] = weight_std * \
                            np.random.randn(hide_size, output_size)
        self.params['b2'] = np.zeros(output_size)
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
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    def numerical_gradient(self, x, t):
        loss_w = lambda w:self.loss(x, t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads
    def print_result(self,y):
        print(y)

