import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from Models import IterativeRegionEstimator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Generate synthetic data

def generate_synthetic_data(n_points, n_providers, seed=0):
    '''
    First half of providers in group 0. Rest in group 1.
    For group 0, Y ~ Ber(0.5) in lower left quadrant, and Y ~ Ber(0.9) o.w.
    For group 1, Y ~ Ber(0.1) in lower left quadrant, and Y ~ Ber(0.9) o.w.
    '''
    np.random.seed(seed)
    X = np.hstack([np.random.normal(size=n_points).reshape(-1, 1) for i in range(2)])
    y = np.zeros(n_points)
    a = np.random.choice([i for i in range(n_providers)], size=n_points, p=[1./n_providers for i in range(n_providers)])
    for i in range(len(y)):
        if a[i] <= int((n_providers-1)/2):
            if X[i,0] < 0 and X[i,1] < 0:
                y[i] = np.random.choice([0,1], p=[0.5, 0.5])
            else:
                y[i] = np.random.choice([0,1], p=[0.1, 0.9])
        else:
            if X[i,0] < 0 and X[i,1] < 0:
                y[i] = np.random.choice([0,1], p=[0.9, 0.1])
            else:
                y[i] = np.random.choice([0,1], p=[0.1, 0.9])
    y = y.astype(int)
    is_in_region = np.logical_and(X[:,0]<0, X[:,1]<0)
    return X, y, a, is_in_region

X, y, a, is_in_region = generate_synthetic_data(10000, 100)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("ExampleData.pdf", dpi=100)

# Train outcome model
outcome_model = DecisionTreeClassifier(min_samples_leaf=100).fit(X, y)

# Train region model
model = IterativeRegionEstimator(region_modelclass=DecisionTreeRegressor(min_samples_leaf=100), beta=0.25)
model.fit(X, y, a, outcome_model)
region = model.predict(X)
region_scores = model.region_model_.predict(X)
print("Region precision: %.4f" % precision_score(is_in_region, region))
print("Region recall: %.4f" % recall_score(is_in_region, region))
print("Region F1 score: %.4f" % f1_score(is_in_region, region))
print("Region AUC: %.4f" % roc_auc_score(is_in_region, region_scores))

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=region)
plt.savefig("ExampleRegion.pdf", dpi=100)

