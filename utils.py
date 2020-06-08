import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

def plot_decision_boundary(X, model):
  h = .02  # step size in the mesh
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, m_max]x[y_min, y_max].
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def get_class_data(N,D):
    # create the data

    X = np.random.randn(N, D)

    # 2 gaussians
    # sep = 1.5
    # X[:N/2] += np.array([sep, sep])
    # X[N/2:] += np.array([-sep, -sep])
    # Y = np.array([0]*(N/2) + [1]*(N/2))

    # noisy XOR
    sep = 2
    X[:125] += np.array([sep, sep])
    X[125:250] += np.array([sep, -sep])
    X[250:375] += np.array([-sep, -sep])
    X[375:] += np.array([-sep, sep])
    Y = np.array([0]*125 + [1]*125 + [0]*125 + [1]*125)

    return X,Y


def get_regress_data(N,T):

    # create the data
    x_axis = np.linspace(0, 2*np.pi, T) # 100 points between 0 and 2pi
    y_axis = np.sin(x_axis)

    idx = np.random.choice(T, size=N, replace=False)
    Xtrain = x_axis[idx].reshape(N, 1)
    Ytrain = y_axis[idx]

    return Xtrain,Ytrain,x_axis,y_axis

class BaggedTreeRegressor:
    def __init__(self, n_estimators, max_depth=None):
        self.B = n_estimators
        self.max_depth = max_depth

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]

            model = DecisionTreeRegressor(max_depth=self.max_depth)
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


class BaggedTreeClassifier:
    def __init__(self, n_estimators, max_depth=None):
        self.B = n_estimators
        self.max_depth = max_depth

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]

            model = DecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        # no need to keep a dictionary since we are doing binary classification
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return np.round(predictions / self.B)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)
