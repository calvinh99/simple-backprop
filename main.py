import math
import numpy as np
import pandas as pd

from nn import MultiplyLayer, ReLU, Softmax, CrossEntropyLoss, NeuralNetwork

if __name__ == "__main__":
    # load iris dataset
    iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_df = pd.read_csv(iris_url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    iris_np = iris_df.values

    # map class names to integers
    map_dict = {'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2}
    map_func = np.vectorize(lambda x: map_dict[x])
    iris_np[:, -1] = map_func(iris_np[:, -1])

    # split train test randomly, but with same number of samples of each class
    test_indices = np.concatenate((np.random.choice(np.arange(0, 50), size=10, replace=False), 
                                   np.random.choice(np.arange(50, 100), size=10, replace=False), 
                                   np.random.choice(np.arange(100, 150), size=10, replace=False)))
    train_indices = np.array(list(set(np.arange(0, 150)) - set(test_indices)))
    train, test = iris_np[train_indices,:], iris_np[test_indices,:]

    # process data into X, Y (normalize X and one-hot encode Y)
    def process(dataset):
        X = dataset[:, :-1].astype(float)
        Y = dataset[:, -1].astype(int)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y_one_hot = np.zeros((len(Y), max(Y)+1))
        Y_one_hot[np.arange(len(Y)), Y] = 1
        return X, Y_one_hot
    X, Y = process(train)

    # initialize neural network
    nn = NeuralNetwork([
        MultiplyLayer(4, 6),
        ReLU(),
        MultiplyLayer(6, 4),
        ReLU(),
        MultiplyLayer(4, 3),
        Softmax()
    ], CrossEntropyLoss())

    # train neural network
    nn.batch_gd(X, Y, 50, batch_size=32, logging=True, lr=0.3) # these are fine-tuned manually, can create function for this

    # test neural network
    X_test, Y_test = process(test)
    preds = np.argmax(nn.forward(X_test), axis=1)
    labels = np.argmax(Y_test, axis=1)
    accuracy = np.mean(np.equal(preds, labels))
    print("Test Accuracy: ", accuracy)