from __future__ import print_function, division
from builtins import range, input


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from utils import get_class_data, plot_decision_boundary

N = 500
D = 2

class BaggedTreeClassifier:
    def __init__(self, B):
        self.B = B

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]
            model = DecisionTreeClassifier(max_depth=2)
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        # no need to keep a dictionary since we are doing binary classification
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return np.round(predictions / self.B)

        # Use this and comment the print decision boundary function
        # predictions = np.zeros((N,D))
        # print("len",len(model.predict(X)))
        # for model in self.models:
        #     predictions[np.arange(N),model.predict(X)]+=1
        # print("2",predictions.argmax(axis=1).shape)
        # return predictions.argmax(axis=1)


    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

if __name__ == '__main__':

    np.random.seed(10)
    X,Y = get_class_data(N,D)
    # plot the data
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    # lone decision tree
    model = DecisionTreeClassifier()
    model.fit(X, Y)
    print("score for 1 tree:", model.score(X, Y))

    # plot data with boundary
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plot_decision_boundary(X, model)
    plt.show()

    model = BaggedTreeClassifier(200)
    model.fit(X, Y)

    print("score for bagged model:", model.score(X, Y))

    # plot data with boundary
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plot_decision_boundary(X, model)
    plt.show()
