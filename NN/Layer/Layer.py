class Layer(object):
    def setInputShape(this, inShape):this.inputShape = inShape
    def layerName(this):return this.__class__.__name__
    def parameters(this):return 0
    def forwardPass(this, X, training = True):raise NotImplementedError()
    def backwordPass(this, grad):raise NotImplementedError()
    def getOutputShape(this):raise NotImplementedError()