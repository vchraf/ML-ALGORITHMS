from tkinter.messagebox import RETRY
import numpy as np

class Adam:
    def __init__(this, lr=1e-4, p1=9e-1, p2=999e-3, epsilon=1e-7):
        this.lr = lr
        this.p1 = p1
        this.p2 = p2
        this.epsilon = epsilon
        this.s = None
        this.r = None

    def update(this, W, grad):
        if this.m is None:
            this.m = np.zeros(np.shape(grad))
            this.v = np.zeros(np.shape(grad))

        this.s = this.p1 * (1 - this.p1) * grad
        this.r = this.p2 * (1 - this.p2) * (grad * grad)

        s_hat = this.s / (1 - this.p1)
        r_hat = this.r / (1 - this.p2)

        delta = this.lr * (s_hat/np.sqrt(r_hat + this.epsilon))

        return W - delta 

