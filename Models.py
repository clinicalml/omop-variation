import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor

class IterativeRegionEstimator(BaseEstimator):
    '''
    Identifies a region of disagreement using the IterativeAlg method.
    '''

    def __init__(self, region_modelclass=RandomForestRegressor(), beta=0.25, n_iter=10):
        '''
        Initializes the iterative region estimator.

        Arguments: 
        region_modelclass -- an unfitted model class that has a .fit(X, y) method and a .predict(X) method
                             (default RandomForestRegressor)
        beta -- a real number between 0 and 1 representing the size of the desired region (default 0.25)
        n_iter -- maximum number of iterations of the algorithm (default 10)
        '''
        self.region_modelclass = region_modelclass
        self.beta = beta
        self.n_iter = n_iter

    def _best_grouping(self, S, X, y, a, preds):
        '''
        Identifies the best grouping given a region.

        Arguments:
        S -- a boolean array of the same length as the number of rows in X, indicating membership in the region
        X, y, a -- data inherited from .fit()
        preds -- array-like of the same length as the number of rows in X

        Returns:
        G -- a dictionary mapping each unique element of a to a binary grouping
        q_score -- a float hat{Q}(S, G), a measure of the variation on S under grouping G
        '''

        # Put everyone in group 0
        G = {}
        for agent in np.unique(a):
            G[agent] = 0

        # Put agents with positive total residual on S into group 1
        q_score = 0.0
        for agent in np.unique(a):
            ixs = (a[S] == agent)
            if np.sum(ixs) > 0:
                term = (1/np.sum(S)) * np.sum(y[S][ixs] - preds[S][ixs])
                if term >= 0:
                    G[agent] = 1
                    q_score += term

        return G, q_score

    def _best_region(self, G, X, y, a, preds):
        '''
        Identifies the best region given a grouping.

        Arguments:
        G -- a dictionary mapping each unique element of a to a binary grouping
        X, y, a -- data inherited from .fit()
        preds -- array-like of the same length as the number of rows in X

        Returns:
        region_model -- a fitted estimator of the same class as self.region_modelclass
        '''
        
        # Get the groupings for agents of each data point
        g = np.zeros(len(a))
        for i in range(len(a)):
            g[i] = G[a[i]]

        # Train model to predict residuals in group 1
        res = (y - preds) * g
        region_model = self.region_modelclass.fit(X, res)
        
        return region_model

    def fit(self, X, y, a, outcome_model):
        '''
        Fits the estimator to data.

        Arguments:
        X -- array-like of shape (n_samples, n_features)
        y -- array-like of shape (n_samples,)
        a -- array-like of shape (n_samples,), representing the decision-maker
        outcome_model -- a fitted predictor that has a .predict_proba(X) method that accepts 
                         an input matrix and outputs an array of as many rows, and 2 columns, such that 
                         .predict_proba(X)[:, 1] consists of real numbers between 0 and 1, representing 
                         the probabilities P[Y=1 | X=x]. 

        Returns:
        self -- this object
        '''

        # Get predictions from outcome_model
        preds = outcome_model.predict_proba(X)[:, 1]

        # Initialize S to the entire space
        S = np.array([True] * X.shape[0])
        G = None
        G_prev = None
        region_model = None
        threshold = None
        
        for it in range(self.n_iter):
            # Find the best grouping for the current region
            G, q_score = self._best_grouping(S, X, y, a, preds)
            if G_prev is not None and G_prev == G:
                break
            G_prev = G

            # Find the best region for the current grouping
            region_model = self._best_region(G, X, y, a, preds)
            region_scores = region_model.predict(X)
            threshold = np.quantile(region_scores, 1-self.beta)
            S = region_scores >= threshold
        G, q_score = self._best_grouping(S, X, y, a, preds)

        # Store fitted model attributes
        self.grouping_ = G
        self.region_model_ = region_model
        self.threshold_ = threshold

        return self

    def predict(self, X):
        '''
        Classifies data points inside/outside the region.

        Arguments:
        X -- array-like of shape(n_samples, n_features)

        Returns:
        y_pred -- array-like of shape (n_samples,)
        '''
        region_scores = self.region_model_.predict(X)
        return region_scores >= self.threshold_

