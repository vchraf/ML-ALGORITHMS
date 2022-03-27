import numpy as np

class SGD:
    def __init__(this, lr=1e-3, momentum=0) -> None:
        this.lr = lr
        this.momentum = momentum
        this.velocity = None
    
    def update(this, w, delW):
        if this.velocity is None:
            this.velocity = np.zeros(np.shape(w))
        
        this.velocity = this.momentum * this.velocity - this.lr * delW
        return w + this.velocity
