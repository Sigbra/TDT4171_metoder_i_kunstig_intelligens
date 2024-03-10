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
    def __init__(self, n_inputs, n_neurons,):
        self.biases = np.zeros((1,n_neurons))
        self.weights = np.random.randn(n_inputs, n_neurons)

    def forwardFeed(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
        #print("self.output LD :", self.output)
        self.inputs = inputs

    def backwardFeed(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationSigmoid:
    def forwardFeed(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        self.inputs = inputs

    def backwardFeed(self, dvalues):
        # Gradient on values from the next layer
        self.dinputs = dvalues * (1 - self.output) * self.output

class LossMSE:
    def calculate(self, y_pred, y_target):
        y_target= np.reshape(y_target, (len(y_target), 1))
        #print(f"y_pred: {y_pred.shape}, y_target: {y_target.shape}")
        sum_squared_error = 0
        for pred, target in zip(y_pred, y_target):
            sum_squared_error += (target - pred) ** 2
        mse = sum_squared_error / (2*len(y_pred))
        return mse

    def backwardFeed(self, dvalues, y_target):
        y_target= np.reshape(y_target, (len(y_target), 1))
        #print(f"dvalues: {dvalues.shape}, y_target: {y_target.shape}")
        samples = len(dvalues)
        # Gradient wrt the loss (mse)
        self.dinputs = -(y_target - dvalues) / samples
        #print("dinputs from backwardFeed in loss: ", self.dinputs.shape)


def update_parameters(layer, learning_rate):
    layer.weights -= learning_rate * layer.dweights
    layer.biases -= learning_rate * layer.dbiases


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    #print("X_train.shape: ", X_train.shape[0])
    #print(type(X_train))

    # init
    n_inputs = X_train.shape[1] #=2 for given dataset
    n_neurons = 2
    n_outputs = 1
    layer1 = LayerDense(n_inputs, n_neurons) #Hidden layer (n_input,n_neurons)
    activation1 = ActivationSigmoid() #Sigmoid activation func for hidden layer output
    layer2 = LayerDense(n_neurons, n_outputs) #Output unit
    #No activation function for last layer was specified
    loss_func = LossMSE()

    print("Training:")
    for epoch in range(10000):
        #forward pass
        layer1.forwardFeed(X_train)
        activation1.forwardFeed(layer1.output)
        layer2.forwardFeed(activation1.output)

        #current loss
        loss = loss_func.calculate(layer2.output, y_train)

        #backwardFeed pass
        loss_func.backwardFeed(layer2.output, y_train)
        layer2.backwardFeed(loss_func.dinputs)
        activation1.backwardFeed(layer2.dinputs)
        layer1.backwardFeed(activation1.dinputs)

        update_parameters(layer1, 0.6)
        update_parameters(layer2, 0.6)

        if epoch == 0:
            print(f'Epoch    {epoch + 1}, Loss: {loss}')
        elif(epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss}')
    
    print("\nTesting:")
    layer1.forwardFeed(X_test)
    activation1.forwardFeed(layer1.output)
    layer2.forwardFeed(activation1.output)
    predictions = layer2.output
    loss = loss_func.calculate(predictions, y_test)
    #print(f'Test Predictions: {predictions}')
    print(f'Test Loss: {loss}')
        


    

    

