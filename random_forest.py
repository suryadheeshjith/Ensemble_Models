import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics


if __name__=='__main__':
    iris = load_iris()
    X = pd.DataFrame(iris.data[:, :], columns = iris.feature_names[:])
    y = pd.DataFrame(iris.target, columns =["Species"])

    # Splitting Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 20, random_state = 100)

    # Defining the stump
    stump = DecisionTreeClassifier(max_depth = 1)

    # Create Random Forest
    stump2 = DecisionTreeClassifier(splitter = "best", max_features = "sqrt")
    ensemble = BaggingClassifier(base_estimator = stump2, n_estimators = 1000,bootstrap = True)

    # Training classifiers
    stump.fit(X_train, np.ravel(y_train))
    ensemble.fit(X_train, np.ravel(y_train))

    # Making predictions
    y_pred_stump = stump.predict(X_test)
    y_pred_ensemble = ensemble.predict(X_test)

    # Determine performance
    stump_accuracy = metrics.accuracy_score(y_test, y_pred_stump)
    ensemble_accuracy = metrics.accuracy_score(y_test, y_pred_ensemble)

    # Print message to user
    print(f"The accuracy of the stump is {stump_accuracy*100:.1f} %")
    print(f"The accuracy of the Random Forest is {ensemble_accuracy*100:.1f} %")
