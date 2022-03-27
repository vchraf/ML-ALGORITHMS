import numpy as np

class Sigmoid:
    def __call__(this,X):
        return 1/(1+np.exp(-X))
    
    def grad(this, X):
        return this.__call__(X) * (1 - this.__call__(X))