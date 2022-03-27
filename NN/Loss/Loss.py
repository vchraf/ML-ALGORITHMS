class Loss(object):
    def loss(this, y, y_hat):raise NotImplementedError()
    def grad(this, y, y_hat):raise NotImplementedError()
    def acc(this, y, y_hat):raise NotImplementedError()