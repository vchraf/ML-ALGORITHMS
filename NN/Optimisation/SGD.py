import numpy as np

class SGD:
    def __init__(this, lr=1e-3, momentum=0) -> None:
        this.lr = lr
        this.momentum = momentum
        this.velocity = None
    
    def update(this, W, grad):
        if this.velocity is None:
            this.velocity = np.zeros(np.shape(W))
        
        this.velocity = this.momentum * this.velocity - this.lr * grad
        return W + this.velocity
