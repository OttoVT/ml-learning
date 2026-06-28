# ROADMAP — the step-by-step path

This is the spine of the whole repo. It expands the source plan's *"What to learn
first"* list (see [`../plan/chatgpt-source-plan.md`](../plan/chatgpt-source-plan.md),
section 7) into a complete journey from math → Python → building → verifying →
deploying neural networks.

Work top to bottom. Check a box when you can *explain it to someone else* and *write
the code without copying*. That's the bar.

> The source plan's order:
> `1. Train/test split → 2. Classification vs regression → 3. Loss functions →
> 4. Accuracy/precision/recall/F1 → 5. Overfitting & generalization →
> 6. Feedforward networks → 7. Embeddings & cosine similarity → 8. Attention →
> 9. Transformers → 10. Evaluation beyond simple metrics`
>
> The phases below keep that order and add the math/Python you need *before* it and
> the build/verify/deploy skills you need *after* it.

---

## Phase 0 — Setup (module `00-setup`)
*Goal: a working environment you can run code in.*

- [ ] Install Python 3.11+ and create a virtual environment
- [ ] `pip install -r requirements.txt`
- [ ] Run `python src/concept_graphs.py` and view the three PNGs
- [ ] Launch Jupyter and run a cell

## Phase 1 — Math foundations (module `01-math`)
*Goal: enough math to read and reason about neural networks.*

- [ ] Vectors, matrices, dot products, matrix multiplication
- [ ] Why a neural layer is just `Wx + b` (a matrix multiply plus a bias)
- [ ] Derivatives & gradients — the intuition (slope = direction to improve)
- [ ] The chain rule (this *is* backpropagation)
- [ ] Probability basics: distributions, log-probabilities, `log` and why we use it
- [ ] Cosine of an angle / vector geometry (sets up cosine similarity)

## Phase 2 — Python & NumPy (module `02-python`)
*Goal: write clean numerical Python fluently.*

- [ ] Python essentials: functions, lists/dicts, comprehensions, classes
- [ ] NumPy arrays, broadcasting, vectorized operations
- [ ] pandas: load a CSV, inspect, select, filter
- [ ] matplotlib: plot a line and a scatter
- [ ] Re-implement the source plan's graphs yourself

## Phase 3 — ML foundations (module `03-foundations`)
*Goal: the ML mindset, before any neural network. (Source steps 1, 2, 5.)*

- [ ] **Train/test split** — why you must hold data back
- [ ] Train / validation / test, and cross-validation
- [ ] **Classification vs regression** — the two core problem types
- [ ] **Overfitting and generalization** — the central tension in ML
- [ ] Bias–variance, underfitting, regularization (intuition)
- [ ] Build a baseline with scikit-learn (logistic regression / linear regression)

## Phase 4 — Loss & training (module `04-loss-and-training`)
*Goal: how a model actually learns. (Source step 3.)*

- [ ] What a loss function is (the thing the model minimizes)
- [ ] MSE for regression
- [ ] **Cross-entropy** for classification — `-log(p_correct)` and why it punishes confident wrong answers
- [ ] Gradient descent: step downhill on the loss
- [ ] Learning rate, epochs, batches
- [ ] Implement gradient descent by hand on a tiny problem

## Phase 5 — Metrics & evaluation (module `05-metrics-and-evaluation`)
*Goal: measure whether a model is good. (Source steps 4 & 7 metrics, sections 2/4/6.)*

- [ ] The **confusion matrix** (TP / FP / FN / TN) — the source of the metrics
- [ ] **Accuracy** — and why it lies on imbalanced data
- [ ] **Precision** — "when it says yes, how often is it right?"
- [ ] **Recall** — "of all real yes cases, how many did it find?"
- [ ] **Precision/recall trade-off** and the decision threshold
- [ ] **F1 score** — and what it ignores (true negatives)
- [ ] **Cosine similarity** — comparing embeddings by direction
- [ ] **Choosing the right metric for the real-world cost** (the deepest lesson)

## Phase 6 — Architectures (module `06-architectures`)
*Goal: understand the model families. (Source steps 6, 7, 8, 9; section 1.)*

- [ ] **Feedforward networks (MLPs)** — fixed input → prediction
- [ ] **Recurrent networks (RNN/LSTM/GRU)** — sequences and memory
- [ ] **Embeddings** — turning things into vectors (pairs with cosine similarity)
- [ ] **Attention** — each token decides which others matter
- [ ] **Transformers** — the architecture behind modern LLMs

## Phase 7 — Build neural networks (module `07-build-neural-networks`)
*Goal: **create** them yourself.*

- [ ] A neuron and a layer from scratch in NumPy
- [ ] Forward pass for an MLP
- [ ] Backpropagation from scratch (apply the chain rule from Phase 1)
- [ ] Train an MLP on a real dataset with NumPy
- [ ] Rebuild the same MLP in PyTorch (tensors, autograd, `nn.Module`)
- [ ] Train a small RNN and a small attention block in PyTorch

## Phase 8 — Verify models (module `08-verify-models`)
*Goal: **verify** them. (Source step 10: evaluation beyond simple metrics.)*

- [ ] Proper evaluation: held-out test set, no leakage
- [ ] Reading learning curves (train vs validation loss)
- [ ] Error analysis: look at what the model gets wrong
- [ ] Robustness checks, calibration, fairness/imbalance checks
- [ ] **Testing ML code** with `pytest` (data, training step, metrics)
- [ ] Reproducibility: seeds, pinned deps, saved configs

## Phase 9 — Deploy models (module `09-deploy-models`)
*Goal: **deploy** them.*

- [ ] Save and load a trained model
- [ ] Wrap a model in a FastAPI endpoint
- [ ] Input validation and pre/post-processing at the boundary
- [ ] Batch vs real-time inference
- [ ] Monitoring in production: latency, drift, and the metrics from Phase 5
- [ ] (Stretch) Containerize and deploy somewhere

---

## Capstone

When the phases are done, prove it with one end-to-end project:

- [ ] Pick a real dataset and a real-world cost of mistakes
- [ ] Choose the architecture, loss, and **metric that matches the cost**
- [ ] Build it (Phase 7), verify it honestly (Phase 8), deploy it (Phase 9)
- [ ] Write up what you'd do differently

## Progress tracking

This roadmap is mirrored as **epics** (one per phase) and **stories** (one per
lesson) for review-friendly tracking. See
[`../_bmad-output/planning-artifacts/epics.md`](../_bmad-output/planning-artifacts/epics.md)
— 11 epics, 60 stories, each with acceptance criteria.
