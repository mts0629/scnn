"""
reference implementation of NN with numpy
for verification

reference:
https://www2.kaiyodai.ac.jp/~takenawa/learning/
"""
import numpy as np

class Affine:
    """
    Affine layer
    (Inner product/Fully connected)
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out
        
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        # sum for batch processing
        self.db = np.sum(dout, axis=0)
        return dx

def sigmoid(x):
    # W/A for exp overflow (around x=709)
    x = -709 * (x <= -709) + x * (x > -709)
    return 1. / (1. + np.exp(-x))

class Sigmoid:
    """
    Sigmoid layer
    """
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        self.y = sigmoid(x)
        return self.y
    
    def backward(self, dout):
        dx = dout * (1 - self.y) * self.y
        return dx

def softmax(a):
    # if a is 1-d vector
    if a.ndim == 1:
        c = np.max(a)
        x = a - c
        x = 709 * (x >= 709) + x * (x < 709)
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x

    # if a is 2-d array
    a = a.T
    # W/A for overflow
    c = np.max(a, axis=0)
    x = a - c
    x = 709 * (x >= 709) + x * (x < 709)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=0)
    return (exp_x / sum_exp_x).T

class SoftmaxWithLoss:
    """
    Softmax layer with cross entropy loss
    (to simplify delta of network output)
    """
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        y = softmax(x)
        self.y = y
        self.t = t

        # averaging for batch
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y+1e-7)) / batch_size
    
    def backward(self):
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

class TestNet:
    def __init__(self, W, b):
        self.l1 = Affine(W, b)
        self.l2 = Sigmoid()
        self.l3 = SoftmaxWithLoss()

    def forward(self, x, t):
        a = self.l1.forward(x)
        z = self.l2.forward(a)
        y = self.l3.forward(z, t)
        return y

    def backward(self):
        dx3 = self.l3.backward()
        dx2 = self.l2.backward(dx3)
        dx1 = self.l1.backward(dx2)
        return dx1

# test

x = np.array([[0.1, 0.2]])

t = np.array([[0, 1]])

W = np.array([[1., 2.],
              [3., 4.]])

b = np.array([[0., 1]])

net = TestNet(W, b)

y = net.forward(x, t)

dx = net.backward()
