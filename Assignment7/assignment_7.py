import numpy as np


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


class LayerDense:
    def __init__(self, n_neurons, n_inputs):
        self.biases = np.zeros((1,n_neurons))
        self.weights = np.random.randn(n_inputs, n_neurons)

    def forwardFeed(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationSigmoid:
    def forwardFeed(self, inputs):
        self.output = 1/(1+np.exp(-inputs))

    def backward(self, dvalues):
        #Gradient values from the next layer 
        self.dinputs = dvalues * (1-self.output)*self.output

class LossMSE:
    def calculate(self, pred, target):
        loss = np.mean((pred-target)**2, axis=1)
        return loss
    
    def backward(self, dvalues, y_true):
        samples=len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

def update_parameters(layer, learning_rate):
    layer.weights -= learning_rate * layer.dweights
    layer.biases -= learning_rate * layer.dbiases

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    #print("X_train.shape: ", X_train.shape[0])
    #print(type(X_train))
    # init
    n_inputs = X_train.shape[0]
    layer1 = LayerDense(2, 2) #Hidden layer
    activation1 = ActivationSigmoid() #Sigmoid activation func for hidden layer output
    layer2 = LayerDense(1,2) #Output unit
    loss_func = LossMSE()

    for epoch in range(1000):
        #forward
        layer1.forwardFeed(X_train)
        activation1.forwardFeed(layer1.output)
        layer2.forwardFeed(activation1.output)
        loss = loss_func.calculate(layer2.output, y_train)

        #backward
        loss_func.backward(layer2.output, y_train)
        layer2.backward(loss_func.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        update_parameters(layer1, 0.1)
        update_parameters(layer2, 0.1)

        print(f'Epoch {epoch}, Loss: {loss}')

    

