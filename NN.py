import numpy as np

class NN:
    def __init__(this, optimizer, lossFun, validationData = None):
        this.layers      = []
        this.optimizer  = optimizer
        this.lossFun    = lossFun()
        this.errors     = {"training": [], "validation":[]}
        this.valSet     = None
    
        if validationData:
            X, y = validationData
            this.valSet = {"X": X, "y": y}
    
    def setTrainable(this, trainable):
        for layer in this.layers:
            layer.trainable = trainable

    def add(this, layer):
        if this.layers:
            layer.setInputShape(shape = this.layers[-1].getOutputShape())
        
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=this.optimizer)
        
        this.layers.append(layer)
    
    def _frowardPass(this, X, training = True):
        layerOutput = X
        for layer in this.layers:
            layerOutput = layer.forwardPass(layerOutput, training)
        return layerOutput

    def _backwardPass(this, grad):
        for layer in this.layers:
            grad = layer.backwordPass(grad)
    
    def trainOnBatch(this, X, y):
        y_hat = this._frowardPass(X)
        loss = np.mean(this.lossFun.loss(y,y_hat))
        grad = this.lossFun.grad(y,y_hat)
        this._backwardPass(grad= grad)
        return loss

    def testOnBatch(this, X, y):
        y_hat = this._frowardPass(X)
        loss = np.mean(this.lossFun.loss(y,y_hat))
        grad = this.lossFun.grad(y,y_hat)
        return loss

    def fit(this, X, y, n_epochs, batch_size):
        def batch_iterator(X, y = None, batch_size = 64):
            n_samples = X.shape[0]
            for i in np.arange(0, n_samples, batch_size):
                begin, end = i , min(i+batch_size, n_samples)
                if y is not None:
                    yield X[begin:end], y[begin:end]
                else:
                    yield X[begin:end]
    
        for _ in range(n_epochs):
            batch_error = []
            for X_batch, y_batch in batch_iterator(X, y, batch_size):
                loss = this.trainOnBatch(X_batch, y_batch)
                batch_error.append(loss)
            this.errors['training'].append(np.mean(batch_error)) 

            if this.valSet is not None:
                valLoss = this.testOnBatch(this.valSet["X"], this.valSet["y"])
                this.errors["validation"].append(valLoss)
        print("training Dane!!!")
        return this.errors["training"], this.errors["validation"]
    
    def predict(this, X):
        return this._frowardPass(X, training = False)