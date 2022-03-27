import numpy as np

class ReLU:
    def __call__(this, X):
        return np.where(X >=0, X, 0)

    def grad(this, X):
        return np.where(X >=0, 1, 0)