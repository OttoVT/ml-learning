# 03 · ML foundations

*Goal: the machine-learning mindset, before any neural network.*
*(Source plan "what to learn first": steps 1, 2, 5.)*

These ideas apply to **every** model — linear regression, a random forest, or a giant
transformer. Get them right and everything later is easier.

## 1. Train/test split (source step 1)

The single most important habit: **never measure a model on the data it learned from.**

```text
All data
   ├── Training set    → the model learns from this
   ├── Validation set  → you tune choices on this
   └── Test set        → you measure final performance ONCE on this
```

If you test on training data, you're measuring memorization, not learning. A model
that scores 100% on data it has seen can be useless on new data.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

- [ ] Understand why holding data back is non-negotiable
- [ ] Know train vs validation vs test
- [ ] Know what **k-fold cross-validation** is and when to use it
- [ ] Understand **data leakage** (the silent killer of ML projects)

## 2. Classification vs regression (source step 2)

The two core supervised problems:

```text
Classification → predict a category   ("spam" / "not spam", "cat" / "dog")
Regression     → predict a number     (house price, temperature tomorrow)
```

This choice drives everything downstream:

| | Classification | Regression |
| --- | --- | --- |
| Output | a class / probability | a number |
| Typical loss | cross-entropy | mean squared error |
| Typical metrics | accuracy, precision, recall, F1 | MAE, RMSE, R² |

- [ ] Classify 5 real problems as classification or regression
- [ ] Know which loss and metrics pair with each

## 3. Overfitting and generalization (source step 5)

The central tension in all of ML:

```text
Underfitting:  model too simple   → bad on train AND test
Good fit:      model generalizes  → good on train, good on test
Overfitting:   model memorizes    → great on train, BAD on test
```

**Generalization** = doing well on data you've never seen. That's the whole goal.

Tools that fight overfitting (intuition for now, detail later):
- More data
- Simpler model / fewer parameters
- Regularization (L2, dropout)
- Early stopping

- [ ] Explain overfitting with the train-vs-test gap
- [ ] Know bias–variance trade-off at an intuitive level

## Hands-on

Build a **baseline** with scikit-learn before you ever touch a neural network — it's
faster, and a neural net that can't beat a baseline isn't worth deploying.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression().fit(X_train, y_train)
preds = model.predict(X_test)
print(accuracy_score(y_test, preds))
```

- [ ] Train a logistic regression classifier and a linear regression model
- [ ] Report test (not train) performance
- [ ] Deliberately overfit a model and watch the train/test gap open up

## Resources
- Google's *Machine Learning Crash Course* (generalization, train/test)
- scikit-learn user guide: model selection

Next: [`../04-loss-and-training/`](../04-loss-and-training/)
