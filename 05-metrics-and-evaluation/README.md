# 05 · Metrics & evaluation

*Goal: measure whether a model is actually good — and pick the right yardstick.*
*(Source plan sections 2, 4, 6; "what to learn first" steps 4 & 7-metrics.)*

This module is the heart of the source conversation. Run the graph generator first:

```bash
python ../src/concept_graphs.py   # writes to ../assets/generated/
```

## The confusion matrix — where the metrics come from (source section 6)

For binary classification, every prediction lands in one of four cells:

```text
                         Actual positive      Actual negative
Predicted positive       True positive (TP)   False positive (FP)
Predicted negative       False negative (FN)  True negative (TN)
```

Everything below is computed from these four numbers:

```text
accuracy  = (TP + TN) / everything
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 · precision · recall / (precision + recall)
```

- [ ] Draw the confusion matrix from memory
- [ ] Derive accuracy, precision, recall from it

## Accuracy — and why it lies

```text
accuracy = correct predictions / total predictions
```

Great when classes are balanced. **Dangerous** when they're not:

```text
Disease dataset: 99% healthy, 1% sick.
A model that always says "healthy" → 99% accuracy → completely useless.
```

- [ ] Explain the imbalanced-data trap in one sentence

## Precision vs recall

```text
precision = "when the model says YES, how often is it right?"   (few false alarms)
recall    = "of all real YES cases, how many did it find?"      (few misses)
```

Use **precision** when false positives are costly (spam: don't trash real email).
Use **recall** when false negatives are costly (cancer screening: don't miss a case).

There's a **trade-off**, controlled by the decision threshold. Look at
`precision_recall_f1_tradeoff.png`: as the threshold rises, precision goes up but
recall goes down.

- [ ] State a problem where precision matters more, and one where recall does
- [ ] Explain how moving the threshold trades one for the other

## F1 score

```text
F1 = 2 · precision · recall / (precision + recall)
```

One number balancing precision and recall — useful for **imbalanced** problems.
Caveat from the source plan:

> F1 ignores true negatives, so it is not always the best metric.

- [ ] Compute F1 for precision 0.80, recall 0.60 (≈ 0.686)

## Cosine similarity — comparing embeddings

```text
Same direction      → cosine ≈ 1
Unrelated           → cosine ≈ 0
Opposite direction  → cosine ≈ -1
```

Used to compare **embeddings** (vector representations) for semantic search,
recommendations, RAG, clustering. Look at `cosine_similarity_curve.png`: it's just
`cos(angle)` — it measures **direction, not magnitude**.

```python
import numpy as np
def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

- [ ] Compute cosine similarity for a few vector pairs
- [ ] Connect it to the embeddings idea in module 06

## Choosing the right metric (the deepest lesson)

From the source plan — which metric for which job:

```text
Balanced classification     → accuracy, cross-entropy
Imbalanced classification   → precision, recall, F1, confusion matrix
Medical screening           → recall (don't miss sick patients)
Spam detection              → precision (don't trash real mail)
Fraud detection             → both precision and recall
Search / embeddings         → cosine similarity, recall@k, MRR, top-k accuracy
Language models             → cross-entropy, perplexity, human eval
```

> A model is not "good" because one metric is high. It is good only if the metric
> matches the real-world cost of mistakes.

- [ ] For 3 real problems, name the cost of each mistake type and pick the metric

## Hands-on

```python
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred),
      precision_score(y_true, y_pred),
      recall_score(y_true, y_pred),
      f1_score(y_true, y_pred))
```

- [ ] Compute all of these on a real model's predictions
- [ ] Recompute precision/recall after moving the threshold; watch the trade-off

Next: [`../06-architectures/`](../06-architectures/)
