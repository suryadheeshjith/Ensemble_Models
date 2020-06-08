from __future__ import print_function, division
from builtins import range, input


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from utils import get_regress_data, plot_decision_boundary

T = 100
N = 30

class BaggedTreeRegressor:
  def __init__(self, B):
    self.B = B

  def fit(self, X, Y):
    N = len(X)
    self.models = []
    for b in range(self.B):
      idx = np.random.choice(N, size=N, replace=True)
      Xb = X[idx]
      Yb = Y[idx]

      model = DecisionTreeRegressor()
      model.fit(Xb, Yb)
      self.models.append(model)

  def predict(self, X):
    predictions = np.zeros(len(X))
    for model in self.models:
      predictions += model.predict(X)
    return predictions / self.B

  def score(self, X, Y):
    d1 = Y - self.predict(X)
    d2 = Y - Y.mean()
    return 1 - d1.dot(d1) / d2.dot(d2)



if __name__ == '__main__':

    np.random.seed(10)

    # get the training data
    Xtrain,Ytrain,x_axis,y_axis = get_regress_data(N,T)

    # try a lone decision tree
    model = DecisionTreeRegressor()
    model.fit(Xtrain, Ytrain)
    prediction = model.predict(x_axis.reshape(T, 1))
    print("score for 1 tree:", model.score(x_axis.reshape(T, 1), y_axis))

    # plot the lone decision tree's predictions
    plt.plot(x_axis, prediction)
    plt.plot(x_axis, y_axis)
    plt.show()

    model = BaggedTreeRegressor(200)
    model.fit(Xtrain, Ytrain)
    print("score for bagged tree:", model.score(x_axis.reshape(T, 1), y_axis))
    prediction = model.predict(x_axis.reshape(T, 1))

    # plot the bagged regressor's predictions
    plt.plot(x_axis, prediction)
    plt.plot(x_axis, y_axis)
    plt.show()
