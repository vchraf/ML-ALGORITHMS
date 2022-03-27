import copy
import numpy as np

from .Layer import Layer

class Dense(Layer):
    def __init__(this, nbrUnits, inShape=None):
        this.inShape = inShape
        this.nbrUnits = nbrUnits
        this.inLayer = None
        this.trainable = True
        this.W = None
        this.b = None

    def initialize(this, optimizer):
        inf = np.sqrt(6/(this.inShape[0]))
        this.W = np.random.uniform(low=-inf, high=inf, size=(this.inShape[0], this.nbrUnits))
        this.b = np.zeros(1, this.nbrUnits)

        this.W_optim = copy.copy(optimizer)
        this.b_optim = copy.copy(optimizer)

    def parameters(this):
        return np.prod(this.W.shape) + np.prod(this.b.shape)
    
    def forwardPass(this, X, Training = True):
        this.inLayer = X 
        return X.dot(this.W) + this.b 
    
    def backwordPass(this, grad):
        W = this.W
        if this.trainable:
            W_grad = this.inLayer.T.dot(grad)
            b_grad = np.sum(grad, axis=0,keepdims=True)

            this.W = this.W_optim.update(this.W, W_grad)
            this.b = this.b_optim.update(this.b, b_grad)

        return grad.dot(W.T)
    
    def getOutputShape(this):
        return (this.nbrUnits,)
