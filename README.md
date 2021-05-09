# omop-variation

omop-variation is a tool to identify and evaluate heterogeneity in decision-making processes.

- [Documentation](https://clinicalml.github.io/omop-variation/)
- [Placeholder for thesis]()

## Background

Different individuals can make different decisions, even when faced with the same context. This is because decisions are influenced by two types of factors: _contextual_ factors common to a given decision-making setup, and _agent-specific_ factors. For instance, in medicine, for a given patient, clinicians may choose different treatment options due to differences in training, opinion, guidelines, and so on. Heterogeneity in clinical decisions occur due to the agent-specific factors, even when the contextual factors (the patient) are the same.

This repository contains algorithms to identify the contexts on which this heterogeneity occurs. For instance, using these algorithms, we can answer questions like:
- For which types of patients do doctors consistently make different decisions on?
- For which types of product categories do consumers have consistent preferences on?

For more background on this topic, and for a complete exposition of the algorithms contained in this repository, refer to [placeholder for thesis]().

## Usage

The algorithms can be applied to any general decision-making process with contextual factors `X`, a binary decision `y`, and the agent / decision-maker `a`. There are four algorithms, implemented in [Models.py](https://github.com/clinicalml/omop-variation/blob/main/Models.py). 

Each estimator follows the scikit-learn [API specification](https://scikit-learn.org/stable/developers/develop.html). Usage is straightforward, as shown in the code block below:

```python
# Train outcome model
outcome_model = RandomForestClassifier(min_samples_leaf=10).fit(X, y)

# Identify regions of X with heterogeneity
model = IterativeRegionEstimator(region_modelclass=RandomForestRegressor(min_samples_leaf=10), beta=0.25)
model.fit(X, y, a, outcome_model)
region = model.predict(X)
```

For more examples, see [Example.py](https://github.com/clinicalml/omop-variation/blob/main/Example.py). A complete documentation of each estimator and their parameters can be found [here](https://clinicalml.github.io/omop-variation/).

## License

[MIT](https://github.com/clinicalml/omop-variation/blob/main/LICENSE)
