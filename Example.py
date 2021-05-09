import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from Models import IterativeRegionEstimator, HyperboxILPRegionEstimator
from sklearn.metrics import f1_score

def generate_synthetic_data(n_points, n_providers):
    '''
    Generate synthetic data such that the region of variation is the lower left quadrant.
    '''
    X = np.hstack([np.random.normal(size=n_points).reshape(-1, 1) for i in range(2)])
    y = np.zeros(n_points)
    a = np.random.choice([i for i in range(n_providers)], size=n_points, p=[1./n_providers for i in range(n_providers)])
    for i in range(n_points):
        if X[i,0] < 0 and X[i,1] < 0:
            if a[i] <= int((n_providers-1)/2):
                y[i] = np.random.choice([0,1], p=[0.5, 0.5])
            else:
                y[i] = np.random.choice([0,1], p=[0.9, 0.1])
        else:
            y[i] = np.random.choice([0,1], p=[0.1, 0.9])
    y = y.astype(int)
    region_true = np.logical_and(X[:,0]<0, X[:,1]<0)
    return X, y, a, region_true

# Generate synthetic data
X, y, a, region_true = generate_synthetic_data(1000, 10)

# Train outcome model
outcome_model = RandomForestClassifier(min_samples_leaf=10).fit(X, y)

'''
Example 1: Iterative algorithm
'''
model = IterativeRegionEstimator(region_modelclass=RandomForestRegressor(min_samples_leaf=10), beta=0.25)
model.fit(X, y, a, outcome_model)
region = model.predict(X)
region_scores = model.region_model_.predict(X)
print("IterativeRegionEstimator")
print("F1 score: %.4f" % f1_score(region_true, region))

'''
Example 2: Hyperbox using integer programming
'''
model = HyperboxILPRegionEstimator(beta=0.25)
model.fit(X, y, a, outcome_model)
region = model.predict(X)
print("HyperboxILPRegionEstimator")
print("F1 score: %.4f" % f1_score(region_true, region))
print("Hyperbox parameters:")
for j in range(X.shape[1]):
    print("Dimension %d: [%.4f, %.4f]" % (j, model.lb_[j], model.ub_[j]))

