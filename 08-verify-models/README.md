# 08 · Verify models

*Goal: **verify** a model — prove it actually works, beyond a single nice number.*
*(Source plan "what to learn first": step 10, "evaluation beyond simple metrics.")*

Building a model is easy. Trusting it is the hard part. This module is about honest
verification — of both the **model** and the **code**.

## Verifying the model

### Evaluate without cheating
- [ ] Final numbers come from the **test set**, touched once (module 03)
- [ ] No **data leakage**: nothing from test/validation influenced training or
      feature engineering
- [ ] Use the **metric that matches the real-world cost** (module 05), not just accuracy

### Read the learning curves
Plot training vs validation loss over epochs:

```text
Train loss falling, validation loss rising  → overfitting
Both high and flat                          → underfitting
Both low and close                          → good generalization
```

- [ ] Plot train vs validation loss and diagnose the fit

### Error analysis (the highest-value habit)
- [ ] Pull the examples the model got **wrong** and look at them
- [ ] Build a **confusion matrix** (module 05) — which classes get confused?
- [ ] Look for patterns: a weak class, a data-quality issue, a mislabeled set

### Beyond a single metric (source step 10)
> A model is not "good" because one metric is high.

- [ ] **Robustness**: does it hold up on noisy / shifted / edge-case inputs?
- [ ] **Calibration**: when it says "90% sure," is it right ~90% of the time?
- [ ] **Slice analysis / fairness**: does performance hold across subgroups, not just
      on average?
- [ ] **Baselines**: does it beat a trivial baseline and the simple model from module 03?

From the source plan — the failure modes a single metric hides:

```text
A fraud model with high accuracy may still miss most fraud.
A medical model with high precision may still miss too many sick patients.
A chatbot with low cross entropy may still produce bad answers.
A search model with high cosine similarity may still retrieve irrelevant documents.
```

## Verifying the code

A model can be "right" and the code around it still be broken. Test it like software.

- [ ] **Data tests**: shapes, ranges, no NaNs, no train/test overlap
- [ ] **Metric tests**: feed known inputs, assert known outputs (e.g. precision of a
      hand-built confusion matrix)
- [ ] **Training-step test**: one step on a tiny batch should reduce the loss
- [ ] **Overfit-a-tiny-batch test**: a correct model should reach ~0 loss on a handful
      of examples — if it can't, something is wrong

```python
# tests/test_metrics.py
from sklearn.metrics import precision_score

def test_precision_known_case():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]   # TP=1, FP=1 → precision = 1/2
    assert precision_score(y_true, y_pred) == 0.5
```

Run with `pytest`.

## Reproducibility

If you can't reproduce a result, you can't trust it.

- [ ] Set random **seeds** (NumPy, PyTorch)
- [ ] Pin dependencies (`requirements.txt`)
- [ ] Save the **config** (hyperparameters) and metrics with each run
- [ ] Version the data or record exactly how to fetch it

## Checklist before you call a model "done"
- [ ] Test-set metric matches the real-world cost
- [ ] No leakage; results reproduce with a fixed seed
- [ ] Learning curves look healthy
- [ ] Error analysis done; beats the baseline
- [ ] Code has tests that pass

## Resources
- *Made With ML* — evaluation & testing for ML
- Jeremy Howard / fast.ai — practical model evaluation

Next: [`../09-deploy-models/`](../09-deploy-models/)
