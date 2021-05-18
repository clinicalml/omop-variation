import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
import gurobipy as grb

class IterativeRegionEstimator(BaseEstimator):
    '''
    Identifies a region of disagreement using the IterativeAlg method.

    Parameters
    ----------
    region_modelclass : BaseEstimator, default=RandomForestRegressor()
        An unfitted model class that has a .fit(X, y) method and a .predict(X) method.

    beta : float, default=0.25
        A real number between 0 and 1 representing the size of the desired region.

    n_iter : int, default=10
        Maximum number of iterations of the algorithm.

    Attributes
    ----------
    grouping_ : dictionary
        A dictionary mapping each agent to a binary grouping.

    region_model_ : BaseEstimator
        A fitted model of the same class as self.region_modelclass.

    threshold_ : float
        Defines the identified region of variation as the inputs x such that 
        region_model.predict(X) >= threshold.

    '''

    def __init__(self, region_modelclass=RandomForestRegressor(), beta=0.25, n_iter=10):
        self.region_modelclass = region_modelclass
        self.beta = beta
        self.n_iter = n_iter

    def _best_grouping(self, S, X, y, a, preds):
        '''
        Identifies the best grouping given a region, based on positive /
        negative estimates of bias.

        Parameters
        ----------
        S : array-like of shape (n_samples,)
            A list of booleans indicating membership in the current region.

        X, y, a : data inherited from .fit().

        preds : array-like of shape (n_samples,)
            A list of floats representing predictions of the outcome_model passed into .fit().

        Returns
        -------
        G : dictionary
            A dictionary mapping each unique element of a to a binary grouping.

        q_score : float
            The value hat{Q}(S, G), a measure of the variation on S under grouping G.
        '''

        # Put everyone in group 0 to start, but should be replaced by +1 / -1
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
                else:
                    G[agent] = -1
                    q_score += (term * -1)

        assert np.all(np.abs(G) == 1)

        return G, q_score

    def _best_region(self, G, X, y, a, preds):
        '''
        Identifies the best region given a grouping.

        Parameters
        ----------
        G : dictionary
            A dictionary mapping each unique element of a to a binary grouping.

        X, y, a : data inherited from .fit().

        preds : array-like of shape (n_samples,)
            A list of floats representing predictions of the outcome_model passed into .fit().

        Returns
        -------
        region_model : BaseEstimator
            A fitted estimator of the same class as self.region_modelclass.
        '''

        # Get the groupings for agents of each data point
        g = np.zeros(len(a))
        for i in range(len(a)):
            g[i] = G[a[i]]

        # Train model to predict absolute value of residuals
        res = (y - preds) * g
        region_model = self.region_modelclass.fit(X, res)

        return region_model

    def fit(self, X, y, a, outcome_model):
        '''
        Fits the estimator to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        a : array-like of shape (n_samples,)
            Agent labels relative to X.

        outcome_model
            A fitted predictor with a .predict_proba(X) method such that 
            .predict_proba(X)[:, 1] consists of real numbers between 0 and 1.

        Returns
        -------
        self
            Fitted estimator.
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

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            A list of booleans indicating membership in the identified region.

        '''
        region_scores = self.region_model_.predict(X)
        return region_scores >= self.threshold_

class HyperboxILPRegionEstimator(BaseEstimator):
    '''
    Identifies a region of disagreement as a hyperbox, using the Hyperbox integer linear program.

    Parameters
    ----------
    beta : float
        A real number between 0 and 1 representing the size of the desired region.

    Attributes
    ----------
    lb_ : array-like of shape (n_features,)
        A list of lower bounds of the fitted hyperbox, one for each feature.
    
    ub_ : array-like of shape (n_features,)
        A list of upper bounds of the fitted hyperbox, one for each feature.

    '''

    def __init__(self, beta=0.25):
        self.beta = beta

    def fit(self, X, y, a, outcome_model, grb_params={"MIPGap":.05, "Threads": 12, "TimeLimit": 600}):
        '''
        Fits the estimator to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        a : array-like of shape (n_samples,)
            Agent labels relative to X.

        outcome_model
            A fitted predictor with a .predict_proba(X) method such that 
            .predict_proba(X)[:, 1] consists of real numbers between 0 and 1.

        grb_params : dictionary, default {"MIPGap": .05, "Threads": 12, "TimeLimit": 600}
            Parameters for grb.Model('model').

        Returns
        -------
        self
            Fitted estimator.
        '''

        # Get predictions from outcome_model
        preds = outcome_model.predict_proba(X)[:, 1]

        # Compute ILP coefficients
        n_samples, n_features = X.shape
        n_agents = len(np.unique(a))
        data_terms = np.zeros((n_samples, n_agents))
        for i in range(n_samples):
            data_terms[i, a[i]] = y[i]-preds[i]
            
        # Optimizer settings
        model = grb.Model('model')
        for param in grb_params:
            model.setParam(param, grb_params[param])

        # Region indicators
        svars = []
        for i in range(n_samples):
            si = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
            svars.append(si)
            
        # Absolute value terms
        tvars = []
        bvars = []
        for a in range(n_agents):
            ta = model.addVar(vtype=grb.GRB.CONTINUOUS)
            ba = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
            model.addConstr(ta >= grb.quicksum(svars[i]*data_terms[i,a] for i in range(n_samples)))
            model.addConstr(ta >= -grb.quicksum(svars[i]*data_terms[i,a] for i in range(n_samples)))
            model.addGenConstrIndicator(ba, True, ta + grb.quicksum(svars[i]*data_terms[i,a] for i in range(n_samples)) <= 0.0)
            model.addGenConstrIndicator(ba, False, ta - grb.quicksum(svars[i]*data_terms[i,a] for i in range(n_samples)) <= 0.0)
            tvars.append(ta)
            bvars.append(ba)
            
        # Hyper box constraints
        lvars = []
        uvars = []
        for j in range(n_features):
            lj = model.addVar(lb=-50, ub=50, vtype=grb.GRB.CONTINUOUS)
            uj = model.addVar(lb=-50, ub=50, vtype=grb.GRB.CONTINUOUS)
            model.addConstr(lj <= uj)
            lvars.append(lj)
            uvars.append(uj)
        vvars = {}
        wvars = {}
        for i in range(n_samples):
            for j in range(n_features):
                vij = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
                wij = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
                eps = 1e-8     # Necessary for strict inequality.
                model.addGenConstrIndicator(vij, True, lvars[j] - X[i,j] <= 0.0)
                model.addGenConstrIndicator(vij, False, -lvars[j] + X[i,j] + eps <= 0.0)
                model.addGenConstrIndicator(wij, True, X[i,j] - uvars[j] <= 0.0)
                model.addGenConstrIndicator(wij, False, -X[i,j] + uvars[j] + eps <= 0.0)            
                vvars[(i, j)] = vij
                wvars[(i, j)] = wij
            model.addGenConstrIndicator(svars[i], True, grb.quicksum(vvars[(i, j)]+wvars[(i, j)] for j in range(n_features)) == 2*n_features)
            model.addGenConstrIndicator(svars[i], False, grb.quicksum(vvars[(i, j)]+wvars[(i, j)] for j in range(n_features)) <= 2*n_features-1)
        
        # Region size constraint
        region_size = int(n_samples*self.beta)
        model.addConstr(grb.quicksum(svars[i] for i in range(n_samples)) <= region_size)
        
        # Objective and optimization
        objective = (1/n_samples) * grb.quicksum(ti for ti in tvars)
        model.ModelSense = grb.GRB.MAXIMIZE
        model.setObjective(objective)
        model.optimize()

        # Store solutions
        self.lb_ = np.empty(n_features)
        self.ub_ = np.empty(n_features)
        for i in range(n_features):
            self.lb_[i] = lvars[i].X
            self.ub_[i] = uvars[i].X

        return self


    def predict(self, X):
        '''
        Classifies data points inside/outside the region.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            A list of booleans indicating membership in the identified region.

        '''
        n_samples, n_features = X.shape

        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            in_box = True
            for j in range(n_features):
                if X[i, j] < self.lb_[j] or X[i, j] > self.ub_[j]:
                    in_box = False
            y_pred[i] = in_box

        return y_pred



class HyperboxILPComplementRegionEstimator(BaseEstimator):
    '''
    Identifies a region of disagreement as a hyperbox, using the Hyperbox Comp integer linear program.

    Parameters
    ----------
    beta : float
        A real number between 0 and 1 representing the size of the desired region.

    Attributes
    ----------
    lb_ : array-like of shape (n_features,)
        A list of lower bounds of the fitted hyperbox, one for each feature.
    
    ub_ : array-like of shape (n_features,)
        A list of upper bounds of the fitted hyperbox, one for each feature.

    '''

    def __init__(self, beta=0.25):
        self.beta = beta

    def fit(self, X, y, a, outcome_model, grb_params={"MIPGap":.05, "Threads": 12, "TimeLimit": 600}):
        '''
        Fits the estimator to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        a : array-like of shape (n_samples,)
            Agent labels relative to X.

        outcome_model
            A fitted predictor with a .predict_proba(X) method such that 
            .predict_proba(X)[:, 1] consists of real numbers between 0 and 1.

        grb_params : dictionary, default {"MIPGap": .05, "Threads": 12, "TimeLimit": 600}
            Parameters for grb.Model('model').

        Returns
        -------
        self
            Fitted estimator.
        '''

        # Get predictions from outcome_model
        preds = outcome_model.predict_proba(X)[:, 1]

        # Compute ILP coefficients
        n_samples, n_features = X.shape
        n_agents = len(np.unique(a))
        data_terms = np.zeros((n_samples, n_agents))
        for i in range(n_samples):
            data_terms[i, a[i]] = y[i]-preds[i]
            
        # Optimizer settings
        model = grb.Model('model')
        for param in grb_params:
            model.setParam(param, grb_params[param])

        # Region indicators
        svars = []
        for i in range(n_samples):
            si = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
            svars.append(si)
            
        # Absolute value terms
        tvars = []
        for a in range(n_agents):
            ta = model.addVar(vtype=grb.GRB.CONTINUOUS)
            model.addConstr(ta >= grb.quicksum((1-svars[i])*data_terms[i,a] for i in range(n_samples)))
            model.addConstr(ta >= -grb.quicksum((1-svars[i])*data_terms[i,a] for i in range(n_samples)))
            tvars.append(ta)
            
        # Hyper box constraints
        lvars = []
        uvars = []
        for j in range(n_features):
            lj = model.addVar(lb=-50, ub=50, vtype=grb.GRB.CONTINUOUS)
            uj = model.addVar(lb=-50, ub=50, vtype=grb.GRB.CONTINUOUS)
            model.addConstr(lj <= uj)
            lvars.append(lj)
            uvars.append(uj)
        vvars = {}
        wvars = {}
        for i in range(n_samples):
            for j in range(n_features):
                vij = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
                wij = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
                eps = 1e-8     # Necessary for strict inequality.
                model.addGenConstrIndicator(vij, True, lvars[j] - X[i,j] <= 0.0)
                model.addGenConstrIndicator(vij, False, -lvars[j] + X[i,j] + eps <= 0.0)
                model.addGenConstrIndicator(wij, True, X[i,j] - uvars[j] <= 0.0)
                model.addGenConstrIndicator(wij, False, -X[i,j] + uvars[j] + eps <= 0.0)            
                vvars[(i, j)] = vij
                wvars[(i, j)] = wij
            model.addGenConstrIndicator(svars[i], True, grb.quicksum(vvars[(i, j)]+wvars[(i, j)] for j in range(n_features)) == 2*n_features)
            model.addGenConstrIndicator(svars[i], False, grb.quicksum(vvars[(i, j)]+wvars[(i, j)] for j in range(n_features)) <= 2*n_features-1)
        
        # Region size constraint
        region_size = int(n_samples*self.beta)
        model.addConstr(grb.quicksum(svars[i] for i in range(n_samples)) <= region_size)
        
        # Objective and optimization
        objective = (1/n_samples) * grb.quicksum(ti for ti in tvars)
        model.ModelSense = grb.GRB.MINIMIZE
        model.setObjective(objective)
        model.optimize()

        # Store solutions
        self.lb_ = np.empty(n_features)
        self.ub_ = np.empty(n_features)
        for i in range(n_features):
            self.lb_[i] = lvars[i].X
            self.ub_[i] = uvars[i].X

        return self


    def predict(self, X):
        '''
        Classifies data points inside/outside the region.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            A list of booleans indicating membership in the identified region.

        '''
        n_samples, n_features = X.shape

        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            in_box = True
            for j in range(n_features):
                if X[i, j] < self.lb_[j] or X[i, j] > self.ub_[j]:
                    in_box = False
            y_pred[i] = in_box

        return y_pred


class HyperboxILPGroupRegionEstimator(BaseEstimator):
    '''
    Identifies a region of disagreement as a hyperbox, using the Hyperbox Group integer linear program.

    Parameters
    ----------
    beta : float
        A real number between 0 and 1 representing the size of the desired region.

    Attributes
    ----------
    lb_ : array-like of shape (n_features,)
        A list of lower bounds of the fitted hyperbox, one for each feature.
    
    ub_ : array-like of shape (n_features,)
        A list of upper bounds of the fitted hyperbox, one for each feature.
    '''

    def __init__(self, beta=0.25):
        self.beta = beta

    def fit(self, X, y, a, outcome_model, grb_params={"MIPGap":.05, "Threads": 12, "TimeLimit": 600}):
        '''
        Fits the estimator to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        a : array-like of shape (n_samples,)
            Agent labels relative to X.

        outcome_model
            A fitted predictor with a .predict_proba(X) method such that 
            .predict_proba(X)[:, 1] consists of real numbers between 0 and 1.

        grb_params : dictionary, default {"MIPGap": .05, "Threads": 12, "TimeLimit": 600}
            Parameters for grb.Model('model').

        Returns
        -------
        self
            Fitted estimator.
        '''

        # Get predictions from outcome_model
        preds = outcome_model.predict_proba(X)[:, 1]

        # Compute ILP coefficients
        n_samples, n_features = X.shape
        n_agents = len(np.unique(a))
        data_terms = np.zeros((n_samples, n_agents))
        for i in range(n_samples):
            data_terms[i, a[i]] = y[i]-preds[i]
            
        # Optimizer settings
        model = grb.Model('model')
        for param in grb_params:
            model.setParam(param, grb_params[param])

        # Region indicators
        svars = []
        for i in range(n_samples):
            si = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
            svars.append(si)

        # Group variables
        gvars = []
        for a in range(n_agents):
            ga = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
            gvars.append(ga)
            
        # Group terms
        tvars = []
        for a in range(n_agents):
            ta = model.addVar(vtype=grb.GRB.CONTINUOUS)
            model.addConstr(ta == grb.quicksum(gvars[a]*svars[i]*data_terms[i,a] for i in range(n_samples)))
            tvars.append(ta)
            
        # Hyper box constraints
        lvars = []
        uvars = []
        for j in range(n_features):
            lj = model.addVar(lb=-50, ub=50, vtype=grb.GRB.CONTINUOUS)
            uj = model.addVar(lb=-50, ub=50, vtype=grb.GRB.CONTINUOUS)
            model.addConstr(lj <= uj)
            lvars.append(lj)
            uvars.append(uj)
        vvars = {}
        wvars = {}
        for i in range(n_samples):
            for j in range(n_features):
                vij = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
                wij = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY)
                eps = 1e-8     # Necessary for strict inequality.
                model.addGenConstrIndicator(vij, True, lvars[j] - X[i,j] <= 0.0)
                model.addGenConstrIndicator(vij, False, -lvars[j] + X[i,j] + eps <= 0.0)
                model.addGenConstrIndicator(wij, True, X[i,j] - uvars[j] <= 0.0)
                model.addGenConstrIndicator(wij, False, -X[i,j] + uvars[j] + eps <= 0.0)            
                vvars[(i, j)] = vij
                wvars[(i, j)] = wij
            model.addGenConstrIndicator(svars[i], True, grb.quicksum(vvars[(i, j)]+wvars[(i, j)] for j in range(n_features)) == 2*n_features)
            model.addGenConstrIndicator(svars[i], False, grb.quicksum(vvars[(i, j)]+wvars[(i, j)] for j in range(n_features)) <= 2*n_features-1)
        
        # Region size constraint
        region_size = int(n_samples*self.beta)
        model.addConstr(grb.quicksum(svars[i] for i in range(n_samples)) <= region_size)
        
        # Objective and optimization
        objective = (1/n_samples) * grb.quicksum(ti for ti in tvars)
        model.ModelSense = grb.GRB.MAXIMIZE
        model.setObjective(objective)
        model.optimize()

        # Store solutions
        self.lb_ = np.empty(n_features)
        self.ub_ = np.empty(n_features)
        for i in range(n_features):
            self.lb_[i] = lvars[i].X
            self.ub_[i] = uvars[i].X

        return self


    def predict(self, X):
        '''
        Classifies data points inside/outside the region.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            A list of booleans indicating membership in the identified region.

        '''
        n_samples, n_features = X.shape

        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            in_box = True
            for j in range(n_features):
                if X[i, j] < self.lb_[j] or X[i, j] > self.ub_[j]:
                    in_box = False
            y_pred[i] = in_box

        return y_pred


