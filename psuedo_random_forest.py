##########################################################################################

# Random forests selects randomly sampled features at each node of the tree. Here, random selection of features
# happens only once at the start of building the decision tree.

##########################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import get_class_data, BaggedTreeClassifier

N = 500
D = 2


class NotAsRandomForest:
  def __init__(self, n_estimators):
    self.B = n_estimators

  def fit(self, X, Y, M=None):
    N, D = X.shape
    if M is None:
      M = int(np.sqrt(D))

    self.models = []
    self.features = []
    for b in range(self.B):
      tree = DecisionTreeClassifier()

      # sample features
      features = np.random.choice(D, size=M, replace=False)

      # sample training samples
      idx = np.random.choice(N, size=N, replace=True)
      Xb = X[idx]
      Yb = Y[idx]

      tree.fit(Xb[:, features], Yb)
      self.features.append(features)
      self.models.append(tree)

  def predict(self, X):
    N = len(X)
    P = np.zeros(N)
    for features, tree in zip(self.features, self.models):
      P += tree.predict(X[:, features])
    return np.round(P / self.B)

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y)




if __name__ == '__main__':
    np.random.seed(10)

    X,Y = get_class_data(N,D)
    Ntrain = int(0.8*len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    T = 500
    test_error_prf = np.empty(T)
    test_error_rf = np.empty(T)
    test_error_bag = np.empty(T)
    for num_trees in range(T):
        if num_trees == 0:
            test_error_prf[num_trees] = None
            test_error_rf[num_trees] = None
            test_error_bag[num_trees] = None
        else:
            rf = RandomForestClassifier(n_estimators=num_trees)
            rf.fit(Xtrain, Ytrain)
            test_error_rf[num_trees] = rf.score(Xtest, Ytest)

            bg = BaggedTreeClassifier(n_estimators=num_trees)
            bg.fit(Xtrain, Ytrain)
            test_error_bag[num_trees] = bg.score(Xtest, Ytest)

            prf = NotAsRandomForest(n_estimators=num_trees)
            prf.fit(Xtrain, Ytrain)
            test_error_prf[num_trees] = prf.score(Xtest, Ytest)

        if num_trees % 10 == 0:
            print("num_trees:", num_trees)

    plt.plot(test_error_rf, label='rf')
    plt.plot(test_error_prf, label='pseudo rf')
    plt.plot(test_error_bag, label='bag')
    plt.legend()
    plt.show()
