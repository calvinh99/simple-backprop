import math
import numpy as np

class Layer():
    def __init__(self):
        pass
    def forward(self, input):
        pass
    def backward(self, input_gradient):
        pass

class MultiplyLayer(Layer):
    """
    This class implements the a fully-connected hidden layer in a neural network with weights and bias.
    """

    def __init__(self, n_features, n_units):
        self.weights = np.random.normal(size=(n_features, n_units))
        self.bias = np.random.normal(size=n_units) # add bias to each row
        
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias # numpy broadcast bias to each row
        self.local_weights_gradient = self.input
        self.local_bias_gradient = 1 # technically (batch_size, n_units, n_units) with batch_size identity matrices
        self.local_input_gradient = self.weights
        return self.output # output is a matrix of shape (batch_size, n_units)
    
    def backward(self, upstream_gradient):
        self.weights_gradient = np.einsum('ij,ik->ijk', self.local_weights_gradient, upstream_gradient) # broadcast np.outer(row of input, row of upstream)
        self.bias_gradient = self.local_bias_gradient * upstream_gradient # broadcast np.dot(identity, row of upstream)
        self.input_gradient = np.einsum('jk,ik->ij', self.local_input_gradient, upstream_gradient) # broadcast np.dot(weights, row of upstream)
        return self.input_gradient

class ReLU(Layer):
    """
    This class implements the ReLU function as a layer in a neural network.
    """

    def __init__(self):
        pass
    
    def forward(self, input):
        self.input = input
        map_grad = np.vectorize(lambda x: 1 if x > 0 else 0) # elementwise, so works for 2d numpy arrays as well
        self.local_gradient = map_grad(self.input).astype(float)
        self.output = np.maximum(np.zeros(input.shape), input)
        return self.output
    
    def backward(self, input_gradient):
        self.gradient = self.local_gradient * input_gradient # elementwise multiplication
        return self.gradient

class Softmax(Layer):
    """
    This class implements the softmax function as a layer in a neural network.
    """

    def __init__(self):
        pass
    
    def forward(self, input):
        self.input = input # n by n_units (in our example 120 by 3)
        self.output = np.exp(self.input) / np.sum(np.exp(self.input), axis=1, keepdims=True)
        self.local_gradient = np.zeros((self.input.shape[0], self.input.shape[1], self.input.shape[1]))
        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):
                for k in range(self.input.shape[1]):
                    if j == k:
                        self.local_gradient[i][j][k] = self.output[i][j] * (1 - self.output[i][j])
                    else:
                        self.local_gradient[i][j][k] = -1 * self.output[i][j] * self.output[i][k]
        return self.output
        
    def backward(self, input_gradient):
        self.gradient = np.einsum('ijk,ik->ij', self.local_gradient, input_gradient) # local grad is (n, n_units, n_units) and input grad is (n, n_units), do row-wise dot product
        return self.gradient

class CrossEntropyLoss(Layer):
    """
    This class implements the cross entropy loss function as a layer in a neural network.
    """

    def __init__(self):
        pass
    
    def forward(self, input, label):
        self.input = input # (batch_size, n_classes) softmax output
        self.label = label # (batch_size, n_classes) one-hot encoded 
        self.loss = -1 * np.einsum('ij,ij->i', self.label, np.log(self.input)) # row wise dot product
        self.local_gradient = -1 * (self.label / self.input)
        return self.loss
    
    def backward(self):
        return self.local_gradient
    
class NeuralNetwork():
    def __init__(self, layers: list[Layer], loss_func: Layer):
        self.layers = layers
        self.loss_func = loss_func
    
    def forward(self, X, Y=None, logging=False):
        Z = X
        if logging:
            print("Input Shape: ", X.shape(), " Label Shape: ", Y.shape())
            
        for layer in self.layers:
            Z = layer.forward(Z)
            if logging:
                print("\n", type(layer), "\nOutput Shape: ", Z.shape)
                
        if Y is not None:
            self.loss = self.loss_func.forward(Z, Y)
            if logging:
                print("\nLoss Shape: ", self.loss.shape)
                
        return Z
    
    def backward(self, logging=False):
        dZ = self.loss_func.backward()
        if logging:
            print("dL/dpred shape: ", dZ.shape)
            
        for layer in self.layers[::-1]:
            dZ = layer.backward(dZ)
            if logging:
                print("\n", type(layer), "\nGradient Shape: ", dZ.shape)
    
    def batch_gd(self, X, Y, epochs, batch_size=None, logging=False, lr=0.01):
        n = len(X)
        if batch_size is None:
            batch_size = n
        for e in range(epochs):
            if logging:
                print("Epoch ", e, "\n", "-"*50, "\n")

            for i in range(0, len(X), batch_size):
                if i + batch_size > n:
                    j = n
                else:
                    j = i + batch_size
                self.forward(X[i:j], Y[i:j])
                self.backward()
                for layer in self.layers:
                    if isinstance(layer, MultiplyLayer): # this part is still hardcoded
                        layer.weights -= lr * np.sum(layer.weights_gradient, axis=0) / batch_size
                        layer.bias -= lr * np.sum(layer.bias_gradient, axis=0) / batch_size

            if logging:
                preds = np.argmax(self.forward(X), axis=1)
                labels = np.argmax(Y, axis=1)
                accuracy = np.mean(np.equal(preds, labels))
                print("Accuracy: ", accuracy)