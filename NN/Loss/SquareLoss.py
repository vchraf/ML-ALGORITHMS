import numpy as np
from .Loss import Loss

class SquareLoss(Loss):
    def __init__(this): pass
    def loss(this, y, y_hat): return 0.5*np.power((y-y_hat),2)
    def grad(this, y,y_hat): return -(y - y_hat)
    def acc(this, y, y_hat):return np.sum(y == y_hat, axis=0) / len(y)