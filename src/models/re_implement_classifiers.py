from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class LogisticRegression(LogisticRegression):
    def __init__(self, C=1.0, penalty='l2', class_weight=None, l1_ratio=None, solver='lbfgs'):
        # initialize the LogisticRegression model with the specified parameters
        super().__init__(C=C, penalty=penalty, class_weight=class_weight, l1_ratio=l1_ratio, solver=solver)
    
    def fit(self, X, y):
        # fit the model to the data as usual
        super().fit(X, y)
    
    def predict(self, X):
        # make predictions using the model as usual
        return super().predict(X)
    
    def score(self, X, y):
        # redefine the score method to return the ROC AUC score instead of the default accuracy score
        return roc_auc_score(y, self.predict_proba(X)[:,1])

class RandomForestClassifier(RandomForestClassifier):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', criterion='gini', class_weight=None):
        # initialize the RandomForestClassifier model with the specified parameters
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, criterion=criterion, class_weight=class_weight)
    
    def fit(self, X, y):
        # fit the model to the data as usual
        super().fit(X, y)
    
    def predict(self, X):
        # make predictions using the model as usual
        return super().predict(X)
    
    def score(self, X, y):
        # return the ROC AUC score as the score for the model
        return roc_auc_score(y, self.predict_proba(X)[:,1])

from sklearn.svm import SVC

class SVC(SVC):
    def __init__(self, C=1.0, gamma='scale', degree=3, kernel='rbf', class_weight=None):
        # initialize the SVC model with the specified parameters
        super().__init__(C=C, gamma=gamma, degree=degree, kernel=kernel, class_weight=class_weight)
    
    def fit(self, X, y):
        # fit the model to the data as usual
        super().fit(X, y)
    
    def predict(self, X):
        # make predictions using the model as usual
        return super().predict(X)
    
    def score(self, X, y):
        # return the ROC AUC score as the score for the model
        return roc_auc_score(y, self.predict_proba(X)[:,1])