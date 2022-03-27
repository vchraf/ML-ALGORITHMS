from .Layer import Layer
from ..Activation import ReLU, Sigmoid

class Activation(Layer):
    def __init__(this, funName):
        activation_functions = {'relu': ReLU, 'sigmoid': Sigmoid}
        this.actFun =  activation_functions[funName]
        this.trainable = True

    def layerName(this):
        return f'Activation : {this.actFun.__class__.__name__}'
    
    def forwardPass(this, X, training=True):
        this.inLayer = X
        return this.actFun(X)
    
    def backwordPass(this, grad):
        return grad * this.actFun.grad(this.inLayer)

    def getOutputShape(this):
        return this.inShape