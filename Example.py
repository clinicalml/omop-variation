import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from Models import IterativeRegionEstimator, HyperboxILPRegionEstimator
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

X, y, a, region_true = generate_synthetic_data(1000, 10)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("ExampleData.pdf", dpi=100)

# Train outcome model
outcome_model = RandomForestClassifier(min_samples_leaf=10).fit(X, y)

# Train region model using IterativeRegionEstimator
model = IterativeRegionEstimator(region_modelclass=RandomForestRegressor(min_samples_leaf=10), beta=0.25)
model.fit(X, y, a, outcome_model)
region = model.predict(X)
region_scores = model.region_model_.predict(X)
print("IterativeRegionEstimator")
print("Precision: %.4f" % precision_score(region_true, region))
print("Recall: %.4f" % recall_score(region_true, region))
print("F1 score: %.4f" % f1_score(region_true, region))
print("AUC: %.4f" % roc_auc_score(region_true, region_scores))

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=region)
plt.savefig("IterativeRegionEstimator.pdf", dpi=100)

# Train region model using HyperboxILPRegionEstimator
model = HyperboxILPRegionEstimator(beta=0.25)
model.fit(X, y, a, outcome_model)
region = model.predict(X)
print("HyperboxILPRegionEstimator")
print("Precision: %.4f" % precision_score(region_true, region))
print("Recall: %.4f" % recall_score(region_true, region))
print("F1 score: %.4f" % f1_score(region_true, region))
print("AUC: %.4f" % roc_auc_score(region_true, region))

print("Hyperbox parameters:")
for j in range(X.shape[1]):
    print("Dimension %d: [%.4f, %.4f]" % (j, model.lb_[j], model.ub_[j]))

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=region)
plt.savefig("HyperboxILPRegionEstimator.pdf", dpi=100)


