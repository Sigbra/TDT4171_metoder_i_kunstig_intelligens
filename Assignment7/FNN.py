import numpy as np


class LayerDense:
    def init(self, n_neurons, n_inputs):
        self.biases = np.zeros(1,n_neurons)
        self.weights = np.random(n_inputs, n_neurons)

    def forwardFeed(self,inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

class ActivationSigmoid:
    def forwardFeed(self, inputs):
        exp = np.e
        self.output = 1/(1+exp**(-inputs))

class Loss:
    def calculate_loss(self, output, y):
        loss = 0.5*(output*y)**2
        return loss