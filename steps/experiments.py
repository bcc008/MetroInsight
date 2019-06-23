import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import xgboost as xgb
import keras


"""
Modify these functions to experiment with different machine learning models.
Input = Train and test data and (if classification) their labels 
Output = Predictions after fit

Cluster experiment: Clustering algorithm used in cluster class.
Binary_clf_experiment: Classification algorithm used in binary classification stage.
Multi_clf_experiment: Classification algorithm used in multi classification stage.
"""

def version():
    version = 0
    return version

def cluster(X_train, X_test):
    clst = KMeans(n_clusters=2, init='k-means++', n_jobs=30, random_state=1337)
    clst.fit(X_train)
    pred_train = clst.predict(X_train)
    pred_test = clst.predict(X_test)
    return pred_train, pred_test

# use function to provide LIST of parameters to try
def binary_clf_parameters():
    n_estimators = [128]
    criterion  = ['gini']
    iter_params = list(itertools.product(n_estimators, criterion))
    parameters = [{'n_estimators': iter_params[i][0], 'criterion': iter_params[i][1], 'n_jobs': 30, 'random_state': 1337} for i in range(len(iter_params))]
    return parameters

def binary_clf(X_train, X_test, y_train, y_test, params):
    clf = RandomForestClassifier()
    model = str(clf)
    clf.set_params(**params)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions, model, clf

# use function to provide LIST of parameters to try
def multi_clf_parameters():
    n_estimators = [128]
    criterion  = ['gini']
    iter_params = list(itertools.product(n_estimators, criterion))
    parameters = [{'n_estimators': iter_params[i][0], 'criterion': iter_params[i][1], 'n_jobs': 30, 'random_state': 1337} for i in range(len(iter_params))]
    return parameters

def multi_clf(X_train, X_test, y_train, y_test, params):
    clf = RandomForestClassifier()
    model = str(clf)
    clf.set_params(**params)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions, model, clf