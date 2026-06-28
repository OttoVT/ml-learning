---
stepsCompleted: ["step-01-validate-prerequisites", "step-02-design-epics", "step-03-create-stories", "step-04-final-validation"]
inputDocuments:
  - "{project-root}/roadmap/ROADMAP.md"
  - "{project-root}/plan/chatgpt-source-plan.md"
---

# ml-real - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for **ml-real**, a
step-by-step machine-learning learning curriculum. It decomposes the learning path in
[`roadmap/ROADMAP.md`](../../roadmap/ROADMAP.md) — itself sourced from the original
ChatGPT plan in [`plan/chatgpt-source-plan.md`](../../plan/chatgpt-source-plan.md) —
into implementable learning stories.

> **Adaptation note.** The BMAD `create-epics-and-stories` workflow is built for
> software products (PRD → FRs/NFRs → epics/stories). This project is a *learning
> curriculum*, so the mapping is:
> - **Functional Requirements → Learning Requirements (LR):** the competencies a
>   learner must acquire (the roadmap's checkboxes).
> - **Epic → Phase:** each roadmap phase (0–9) plus the capstone.
> - **Story → Lesson:** each checkbox becomes one learning story.
> - **Acceptance Criteria → the mastery bar:** "I can explain it to someone else
>   *and* write the code without copying" (from the roadmap), expressed as
>   Given/When/Then.
> - **`As a {user_type}`** is always **`As a learner`**.

## Requirements Inventory

### Functional Requirements (Learning Requirements)

These are the competencies the curriculum must teach. Numbered `LR-<phase>.<item>`
so coverage is self-evident.

**Phase 0 — Setup**
- LR-0.1: Create and activate a Python virtual environment
- LR-0.2: Install all project dependencies from `requirements.txt`
- LR-0.3: Run `src/concept_graphs.py` and view the three generated graphs
- LR-0.4: Launch Jupyter and execute a cell

**Phase 1 — Math foundations**
- LR-1.1: Vectors, matrices, dot products, matrix multiplication
- LR-1.2: Understand a neural layer as `Wx + b`
- LR-1.3: Derivatives and gradients (slope = direction to improve)
- LR-1.4: The chain rule (the basis of backpropagation)
- LR-1.5: Probability basics, `log`, and log-probabilities
- LR-1.6: Vector geometry / cosine of an angle (sets up cosine similarity)

**Phase 2 — Python & NumPy**
- LR-2.1: Python essentials (functions, collections, comprehensions, classes)
- LR-2.2: NumPy arrays, broadcasting, vectorized operations
- LR-2.3: pandas: load/inspect/select/filter tabular data
- LR-2.4: matplotlib: line and scatter plots
- LR-2.5: Re-implement the source plan's three graphs from scratch

**Phase 3 — ML foundations**
- LR-3.1: Train/test split and why data must be held back
- LR-3.2: Train/validation/test and cross-validation; data leakage
- LR-3.3: Classification vs regression
- LR-3.4: Overfitting and generalization
- LR-3.5: Bias–variance, underfitting, regularization (intuition)
- LR-3.6: Build a scikit-learn baseline (logistic/linear regression)

**Phase 4 — Loss & training**
- LR-4.1: What a loss function is
- LR-4.2: Mean squared error for regression
- LR-4.3: Cross-entropy for classification
- LR-4.4: Gradient descent
- LR-4.5: Learning rate, epochs, batches
- LR-4.6: Implement gradient descent by hand

**Phase 5 — Metrics & evaluation**
- LR-5.1: The confusion matrix (TP/FP/FN/TN)
- LR-5.2: Accuracy and its failure on imbalanced data
- LR-5.3: Precision
- LR-5.4: Recall
- LR-5.5: Precision/recall trade-off and the decision threshold
- LR-5.6: F1 score (and what it ignores)
- LR-5.7: Cosine similarity for embeddings
- LR-5.8: Choosing the metric that matches the real-world cost

**Phase 6 — Architectures**
- LR-6.1: Feedforward networks (MLPs)
- LR-6.2: Recurrent networks (RNN/LSTM/GRU)
- LR-6.3: Embeddings
- LR-6.4: Attention
- LR-6.5: Transformers

**Phase 7 — Build neural networks**
- LR-7.1: A neuron and a layer from scratch in NumPy
- LR-7.2: Forward pass for an MLP
- LR-7.3: Backpropagation from scratch
- LR-7.4: Train a NumPy MLP on a real dataset
- LR-7.5: Rebuild the MLP in PyTorch (autograd, `nn.Module`)
- LR-7.6: Train a small RNN and an attention block in PyTorch

**Phase 8 — Verify models**
- LR-8.1: Proper evaluation with no leakage
- LR-8.2: Reading learning curves (train vs validation)
- LR-8.3: Error analysis
- LR-8.4: Robustness, calibration, fairness/slice checks
- LR-8.5: Testing ML code with `pytest`
- LR-8.6: Reproducibility (seeds, pinned deps, saved configs)

**Phase 9 — Deploy models**
- LR-9.1: Save and load a trained model
- LR-9.2: Serve a model via a FastAPI endpoint
- LR-9.3: Input validation and matching preprocessing at the boundary
- LR-9.4: Batch vs real-time inference
- LR-9.5: Production monitoring (latency, drift, live metrics)
- LR-9.6: Containerize and deploy (stretch)

**Capstone**
- LR-C.1: Choose a dataset and define the real-world cost of mistakes
- LR-C.2: Choose architecture, loss, and the metric that matches the cost
- LR-C.3: Build, verify, and deploy end to end
- LR-C.4: Write up results and what you'd do differently

### NonFunctional Requirements (Learning Quality Requirements)

- NFR1 (Mastery bar): A story is "done" only when the learner can explain the concept
  to someone else *and* write the relevant code without copying.
- NFR2 (Honest evaluation): All reported model performance comes from a held-out test
  set with no data leakage.
- NFR3 (Reproducibility): Every code result is reproducible — fixed random seeds,
  pinned dependencies, saved configs.
- NFR4 (Runnable): Every code example in the repo runs from the repo root after
  `pip install -r requirements.txt`.
- NFR5 (Source fidelity): Concepts trace back to the verbatim source plan; the
  original is never edited, only expanded.
- NFR6 (Sequencing): Each epic stands alone and builds only on earlier epics, never
  on later ones.

### Additional Requirements (Environment & Tooling)

- Python 3.11+ with a virtual environment.
- Dependencies pinned in `requirements.txt` (NumPy, matplotlib, pandas, scikit-learn,
  PyTorch, transformers, FastAPI, pytest, Jupyter).
- `src/concept_graphs.py` generates the three explanatory graphs into
  `assets/generated/` (verified working).
- Generated artifacts (datasets, model weights, graphs) are git-ignored and
  regenerable from code.

### UX Design Requirements

Not applicable — this is a code/curriculum repository with no user interface. (The
only "interface" is the repo's README navigation and the optional FastAPI demo in
Epic 10, whose requirements are captured as LR-9.x.)

### FR Coverage Map

Learning Requirements map to epics by their phase number — every LR is covered:

- LR-0.* → Epic 1 (Setup)
- LR-1.* → Epic 2 (Math foundations)
- LR-2.* → Epic 3 (Python & NumPy)
- LR-3.* → Epic 4 (ML foundations)
- LR-4.* → Epic 5 (Loss & training)
- LR-5.* → Epic 6 (Metrics & evaluation)
- LR-6.* → Epic 7 (Architectures)
- LR-7.* → Epic 8 (Build neural networks)
- LR-8.* → Epic 9 (Verify models)
- LR-9.* → Epic 10 (Deploy models)
- LR-C.* → Epic 11 (Capstone)

NFR1–NFR6 are cross-cutting and apply to the acceptance criteria of every story.

## Epic List

### Epic 1 — Phase 0: Working environment
Get a Python environment where every example in the repo runs.
**LRs covered:** LR-0.1, LR-0.2, LR-0.3, LR-0.4

### Epic 2 — Phase 1: Math foundations
Build the math intuition that neural networks are made of.
**LRs covered:** LR-1.1 … LR-1.6

### Epic 3 — Phase 2: Python & NumPy
Write clean numerical Python fluently.
**LRs covered:** LR-2.1 … LR-2.5

### Epic 4 — Phase 3: ML foundations
Adopt the ML mindset before any neural network.
**LRs covered:** LR-3.1 … LR-3.6

### Epic 5 — Phase 4: Loss & training
Understand how a model actually learns.
**LRs covered:** LR-4.1 … LR-4.6

### Epic 6 — Phase 5: Metrics & evaluation
Measure whether a model is good and pick the right yardstick.
**LRs covered:** LR-5.1 … LR-5.8

### Epic 7 — Phase 6: Architectures
Understand feedforward, recurrent, and transformer model families.
**LRs covered:** LR-6.1 … LR-6.5

### Epic 8 — Phase 7: Build neural networks
Create neural networks from scratch, then with PyTorch.
**LRs covered:** LR-7.1 … LR-7.6

### Epic 9 — Phase 8: Verify models
Prove a model works — beyond a single nice number.
**LRs covered:** LR-8.1 … LR-8.6

### Epic 10 — Phase 9: Deploy models
Serve a model and keep it healthy in production.
**LRs covered:** LR-9.1 … LR-9.6

### Epic 11 — Capstone
Build, verify, and deploy one end-to-end project that proves mastery.
**LRs covered:** LR-C.1 … LR-C.4

---

## Epic 1 — Phase 0: Working environment

Get a Python environment where every example in the repo runs. Standalone: produces a
machine that can execute all later epics; depends on nothing.

### Story 1.1: Create and activate a virtual environment

As a learner,
I want an isolated Python environment for this project,
So that its packages don't collide with anything else on my machine.

**Acceptance Criteria:**

**Given** Python 3.11+ is installed
**When** I run `python3 -m venv .venv` and activate it
**Then** `which python` points inside `.venv`
**And** `python --version` reports 3.11 or higher.

### Story 1.2: Install project dependencies

As a learner,
I want all required libraries installed,
So that every example in the repo can run.

**Acceptance Criteria:**

**Given** the virtual environment is active
**When** I run `pip install -r requirements.txt`
**Then** the install completes without errors
**And** `python -c "import numpy, torch, sklearn, fastapi"` succeeds.

### Story 1.3: Generate the source-plan graphs

As a learner,
I want to run the concept-graph generator,
So that I see the cross-entropy, precision/recall/F1, and cosine graphs from the source plan.

**Acceptance Criteria:**

**Given** dependencies are installed
**When** I run `python src/concept_graphs.py`
**Then** three PNGs appear in `assets/generated/`
**And** I can open and describe what each graph shows.

### Story 1.4: Run a Jupyter cell

As a learner,
I want a working notebook environment,
So that I can do interactive lessons later.

**Acceptance Criteria:**

**Given** dependencies are installed
**When** I launch `jupyter notebook` and run `import torch; print(torch.__version__)`
**Then** the cell executes and prints a version without error.

---

## Epic 2 — Phase 1: Math foundations

Build the math intuition neural networks are made of. Depends on Epic 1 only.

### Story 2.1: Vectors, matrices, and matrix multiplication

As a learner,
I want to compute dot products and matrix multiplications,
So that I can reason about what a layer does to its input.

**Acceptance Criteria:**

**Given** a 2×3 matrix and a 3-vector
**When** I multiply them by hand and then with NumPy
**Then** both results match
**And** I can state the output shape rule (m×n times n → m).

### Story 2.2: A layer is `Wx + b`

As a learner,
I want to understand a neural layer as a matrix multiply plus a bias,
So that the architecture chapters aren't magic.

**Acceptance Criteria:**

**Given** weights `W`, input `x`, bias `b`
**When** I compute `W @ x + b` in NumPy
**Then** I can explain what each term represents and why the shapes line up.

### Story 2.3: Derivatives and gradients

As a learner,
I want to understand a derivative as a slope and a gradient as a direction,
So that gradient descent makes sense later.

**Acceptance Criteria:**

**Given** `y = x²`
**When** I plot it and its derivative `2x`
**Then** I can identify where the slope is zero
**And** explain why that point is the minimum gradient descent seeks.

### Story 2.4: The chain rule

As a learner,
I want to apply the chain rule to a composed function,
So that I understand backpropagation when I build it.

**Acceptance Criteria:**

**Given** a composed function like `f(g(x))`
**When** I compute its derivative
**Then** I can show the chain-rule steps and explain how this generalizes to layers.

### Story 2.5: Probability, log, and log-probabilities

As a learner,
I want to understand probability distributions and the `log` function,
So that cross-entropy is intuitive.

**Acceptance Criteria:**

**Given** a model output like `P(cat)=0.9, P(dog)=0.1`
**When** I compute `-log(p)` for each
**Then** I can explain why `log` turns products into sums and why small `p` gives large loss.

### Story 2.6: Vector geometry and cosine

As a learner,
I want to compute the cosine of the angle between two vectors,
So that cosine similarity is grounded in geometry.

**Acceptance Criteria:**

**Given** vectors `[1,0]` and `[1,1]`
**When** I compute their cosine similarity by hand
**Then** I get ≈ 0.707
**And** I can explain that it measures direction, not magnitude.

---

## Epic 3 — Phase 2: Python & NumPy

Write clean numerical Python fluently. Depends on Epics 1–2.

### Story 3.1: Python essentials

As a learner,
I want fluency with functions, collections, comprehensions, and classes,
So that I can read and write ML code.

**Acceptance Criteria:**

**Given** a small data task
**When** I solve it with functions, a comprehension, and a class
**Then** the code runs and I can explain each construct.

### Story 3.2: NumPy arrays and broadcasting

As a learner,
I want to use vectorized NumPy operations and broadcasting,
So that I can express a layer without Python loops.

**Acceptance Criteria:**

**Given** `W` (2×3), `x` (3,), `b` (2,)
**When** I compute `W @ x + b`
**Then** I get a shape-(2,) result with no explicit loop
**And** I can explain how broadcasting added the bias.

### Story 3.3: pandas basics

As a learner,
I want to load and inspect a CSV with pandas,
So that I can work with real tabular datasets.

**Acceptance Criteria:**

**Given** a CSV file
**When** I `read_csv`, then `.head()`, `.info()`, filter rows, and select columns
**Then** I can describe the dataset's shape, columns, and a filtered subset.

### Story 3.4: matplotlib basics

As a learner,
I want to make a line plot and a scatter plot,
So that I can visualize data and results.

**Acceptance Criteria:**

**Given** some x/y data
**When** I plot a line and a scatter with labels, title, and legend
**Then** the figure renders correctly and is saved to a file.

### Story 3.5: Re-implement the source-plan graphs

As a learner,
I want to recreate the three concept graphs from scratch,
So that I prove I understand both the math and the plotting.

**Acceptance Criteria:**

**Given** only the formulas (`-log(p)`, the P/R/F1 curves, `cos(θ)`)
**When** I write the plots without looking at `src/concept_graphs.py`
**Then** my graphs match the originals in `assets/generated/`.

---

## Epic 4 — Phase 3: ML foundations

Adopt the ML mindset before any neural network. Depends on Epics 1–3.

### Story 4.1: Train/test split

As a learner,
I want to split data into training and test sets,
So that I never measure a model on data it learned from.

**Acceptance Criteria:**

**Given** a dataset
**When** I use `train_test_split` and train on train, evaluate on test
**Then** I can explain why testing on training data is meaningless.

### Story 4.2: Validation, cross-validation, and leakage

As a learner,
I want to understand train/validation/test and k-fold cross-validation,
So that I tune choices honestly and avoid leakage.

**Acceptance Criteria:**

**Given** a modeling task
**When** I separate a validation set (or use k-fold) and keep test untouched
**Then** I can define data leakage and name one way it sneaks in.

### Story 4.3: Classification vs regression

As a learner,
I want to tell classification and regression problems apart,
So that I choose the right loss and metrics.

**Acceptance Criteria:**

**Given** five real-world problems
**When** I classify each as classification or regression
**Then** I name the typical loss and metric for each correctly.

### Story 4.4: Overfitting and generalization

As a learner,
I want to recognize overfitting via the train/test gap,
So that I can judge whether a model will generalize.

**Acceptance Criteria:**

**Given** a deliberately overfit model
**When** I compare training and test performance
**Then** the gap is visible and I can explain underfit vs good-fit vs overfit.

### Story 4.5: Bias–variance and regularization

As a learner,
I want intuition for bias–variance and regularization,
So that I know the levers that fight overfitting.

**Acceptance Criteria:**

**Given** an overfit model
**When** I apply a regularization technique (e.g. simpler model or L2)
**Then** the train/test gap narrows and I can explain why.

### Story 4.6: Build a scikit-learn baseline

As a learner,
I want a simple baseline model,
So that I have something a neural network must beat.

**Acceptance Criteria:**

**Given** a split dataset
**When** I train logistic/linear regression and evaluate on the test set
**Then** I report a test metric (not a training metric) as the baseline.

---

## Epic 5 — Phase 4: Loss & training

Understand how a model actually learns. Depends on Epics 1–4.

### Story 5.1: What a loss function is

As a learner,
I want to understand loss as the thing a model minimizes,
So that the training loop has a purpose.

**Acceptance Criteria:**

**Given** predictions and targets
**When** I compute a loss value
**Then** I can explain that training adjusts weights to lower this number.

### Story 5.2: Mean squared error

As a learner,
I want to compute MSE for a regression problem,
So that I can train regression models.

**Acceptance Criteria:**

**Given** predictions and numeric targets
**When** I compute `mean((pred - target)²)`
**Then** I can explain why squaring punishes large errors more.

### Story 5.3: Cross-entropy

As a learner,
I want to compute cross-entropy for classification,
So that I understand the source plan's central loss.

**Acceptance Criteria:**

**Given** the source plan's A/B/C cat example
**When** I compute `-log(p_correct)` for each model
**Then** A < B < C in loss
**And** I can explain why it punishes confident wrong predictions.

### Story 5.4: Gradient descent

As a learner,
I want to understand stepping the weights against the gradient,
So that I know how a model improves.

**Acceptance Criteria:**

**Given** a loss and its gradient
**When** I apply `w = w - lr * grad`
**Then** I can explain why this reduces the loss.

### Story 5.5: Learning rate, epochs, batches

As a learner,
I want to know what learning rate, epoch, and batch mean,
So that I can configure training.

**Acceptance Criteria:**

**Given** a training run
**When** I change the learning rate too high and too low
**Then** I can describe the symptom of each (diverge vs crawl).

### Story 5.6: Implement gradient descent by hand

As a learner,
I want to fit `y = 2x` with hand-written gradient descent,
So that the training loop holds no mystery.

**Acceptance Criteria:**

**Given** data following `y = 2x`
**When** I run the NumPy gradient-descent loop from the module
**Then** `w` approaches 2 and the printed loss shrinks over steps.

---

## Epic 6 — Phase 5: Metrics & evaluation

Measure whether a model is good and pick the right yardstick. Depends on Epics 1–5.

### Story 6.1: The confusion matrix

As a learner,
I want to build a confusion matrix,
So that I can derive the other metrics from it.

**Acceptance Criteria:**

**Given** true labels and predictions
**When** I compute TP, FP, FN, TN
**Then** I can derive accuracy, precision, and recall from those four numbers.

### Story 6.2: Accuracy and its limits

As a learner,
I want to compute accuracy and see where it lies,
So that I don't trust it blindly on imbalanced data.

**Acceptance Criteria:**

**Given** a 99%/1% imbalanced dataset
**When** I evaluate an "always predict majority" model
**Then** accuracy is ~99% yet the model is useless, and I can explain why.

### Story 6.3: Precision

As a learner,
I want to compute precision,
So that I can reason about false alarms.

**Acceptance Criteria:**

**Given** a confusion matrix
**When** I compute `TP / (TP + FP)`
**Then** I can state a problem (e.g. spam) where precision matters most and why.

### Story 6.4: Recall

As a learner,
I want to compute recall,
So that I can reason about missed positives.

**Acceptance Criteria:**

**Given** a confusion matrix
**When** I compute `TP / (TP + FN)`
**Then** I can state a problem (e.g. cancer screening) where recall matters most and why.

### Story 6.5: Precision/recall trade-off

As a learner,
I want to move the decision threshold,
So that I can see precision and recall trade against each other.

**Acceptance Criteria:**

**Given** model scores and a threshold
**When** I raise and lower the threshold
**Then** precision and recall move in opposite directions
**And** my observation matches `precision_recall_f1_tradeoff.png`.

### Story 6.6: F1 score

As a learner,
I want to compute F1,
So that I can balance precision and recall in one number.

**Acceptance Criteria:**

**Given** precision 0.80 and recall 0.60
**When** I compute F1
**Then** I get ≈ 0.686
**And** I can state what F1 ignores (true negatives).

### Story 6.7: Cosine similarity

As a learner,
I want to compute cosine similarity between embeddings,
So that I can compare items by meaning.

**Acceptance Criteria:**

**Given** two vectors
**When** I compute `(a·b)/(||a||·||b||)`
**Then** identical-direction vectors give ≈ 1 and orthogonal give ≈ 0
**And** I can connect this to semantic search.

### Story 6.8: Choosing the right metric

As a learner,
I want to choose a metric based on the cost of mistakes,
So that "good" reflects the real goal, not a high number.

**Acceptance Criteria:**

**Given** three real problems with different costs of error
**When** I name the cost of each mistake type
**Then** I select an appropriate metric for each and justify it using the source plan's guidance.

---

## Epic 7 — Phase 6: Architectures

Understand the model families from the source plan. Depends on Epics 1–6.

### Story 7.1: Feedforward networks (MLPs)

As a learner,
I want to understand feedforward networks,
So that I know the simplest standard architecture.

**Acceptance Criteria:**

**Given** the MLP diagram
**When** I describe the data flow input → hidden → output
**Then** I can explain why a nonlinear activation is essential.

### Story 7.2: Recurrent networks

As a learner,
I want to understand RNNs and their variants,
So that I know how sequences with memory are modeled.

**Acceptance Criteria:**

**Given** a sequence task
**When** I describe the hidden-state update `output = f(input + previous memory)`
**Then** I can explain the long-range dependency weakness and what LSTM/GRU improve.

### Story 7.3: Embeddings

As a learner,
I want to understand embeddings as vectors of meaning,
So that I can connect them to cosine similarity.

**Acceptance Criteria:**

**Given** a few sentences
**When** I generate embeddings and rank them by cosine similarity to a query
**Then** semantically closer sentences rank higher.

### Story 7.4: Attention

As a learner,
I want to understand self-attention,
So that I grasp the transformer's core mechanism.

**Acceptance Criteria:**

**Given** the sentence "the animal didn't cross the road because it was tired"
**When** I explain attention for the word "it"
**Then** I can say why attention handles long-range links better than an RNN.

### Story 7.5: Transformers

As a learner,
I want to trace data through a transformer,
So that I understand the architecture behind modern LLMs.

**Acceptance Criteria:**

**Given** the transformer diagram
**When** I trace tokens → embeddings → self-attention → feedforward → output
**Then** I can name each building block and what it does.

---

## Epic 8 — Phase 7: Build neural networks

Create neural networks yourself, from scratch then with PyTorch. Depends on Epics 1–7.

### Story 8.1: A neuron and a layer in NumPy

As a learner,
I want to implement a neuron and a layer from scratch,
So that I remove all the magic from a forward pass.

**Acceptance Criteria:**

**Given** weights, a bias, and an activation
**When** I implement `activation(W @ x + b)` in NumPy
**Then** it produces correct output shapes for a small input.

### Story 8.2: Forward pass for an MLP

As a learner,
I want to chain layers into a forward pass,
So that I can produce predictions.

**Acceptance Criteria:**

**Given** the MLP class skeleton
**When** I implement `forward`
**Then** it returns a valid probability distribution (softmax sums to 1).

### Story 8.3: Backpropagation from scratch

As a learner,
I want to implement backprop using the chain rule,
So that my network can learn.

**Acceptance Criteria:**

**Given** my forward pass and a loss
**When** I implement `backward` and update weights
**Then** the loss decreases over training iterations.

### Story 8.4: Train a NumPy MLP on real data

As a learner,
I want to train my from-scratch MLP on a real dataset,
So that I prove it actually learns.

**Acceptance Criteria:**

**Given** a dataset like Iris or an MNIST subset, split into train/test
**When** I train my NumPy MLP
**Then** test accuracy is clearly better than random
**And** I report the test (not train) accuracy.

### Story 8.5: Rebuild the MLP in PyTorch

As a learner,
I want to reproduce the MLP in PyTorch with autograd,
So that I see how a framework automates what I did by hand.

**Acceptance Criteria:**

**Given** the same dataset
**When** I build an `nn.Module`, pick a loss and optimizer, and run the training loop
**Then** I reach comparable test accuracy with far less code
**And** I can map each loop line to the by-hand version.

### Story 8.6: Train an RNN and an attention block

As a learner,
I want to train a small RNN and a minimal attention block in PyTorch,
So that I have hands-on experience with sequence models.

**Acceptance Criteria:**

**Given** a small sequence task
**When** I train an RNN/LSTM and a self-attention block
**Then** both train without error and I can observe attention weights over tokens.

---

## Epic 9 — Phase 8: Verify models

Prove a model works, beyond a single nice number. Depends on Epics 1–8.

### Story 9.1: Evaluate without cheating

As a learner,
I want to evaluate strictly on a held-out test set,
So that my numbers are trustworthy.

**Acceptance Criteria:**

**Given** a trained model
**When** I compute the final metric on the untouched test set
**Then** I can confirm no test data influenced training or feature engineering.

### Story 9.2: Read learning curves

As a learner,
I want to plot train vs validation loss,
So that I can diagnose under/overfitting.

**Acceptance Criteria:**

**Given** a training run that logged losses
**When** I plot train and validation loss over epochs
**Then** I correctly label the curves as underfit, good fit, or overfit.

### Story 9.3: Error analysis

As a learner,
I want to inspect the examples a model gets wrong,
So that I can improve it deliberately.

**Acceptance Criteria:**

**Given** model predictions on a test set
**When** I pull the misclassified examples and build a confusion matrix
**Then** I can identify at least one pattern (weak class, bad labels, etc.).

### Story 9.4: Robustness, calibration, and slices

As a learner,
I want to check robustness, calibration, and per-slice performance,
So that I trust the model beyond its average score.

**Acceptance Criteria:**

**Given** a trained model
**When** I test noisy/edge inputs, check confidence calibration, and evaluate on subgroups
**Then** I can report where it is and isn't reliable.

### Story 9.5: Test ML code with pytest

As a learner,
I want automated tests for my ML code,
So that broken data or training logic is caught early.

**Acceptance Criteria:**

**Given** metric, data, and training-step code
**When** I write pytest tests (including a known-precision case and an overfit-tiny-batch test)
**Then** `pytest` passes and the tests fail if I introduce a bug.

### Story 9.6: Reproducibility

As a learner,
I want my results to reproduce,
So that they can be trusted and compared.

**Acceptance Criteria:**

**Given** a training script
**When** I fix seeds, pin dependencies, and save the config and metrics
**Then** re-running produces the same result.

---

## Epic 10 — Phase 9: Deploy models

Serve a model and keep it healthy. Depends on Epics 1–9.

### Story 10.1: Save and load a model

As a learner,
I want to persist and reload a trained model,
So that inference doesn't require retraining.

**Acceptance Criteria:**

**Given** a trained model
**When** I save its weights and load them into the same architecture
**Then** the reloaded model reproduces the original predictions.

### Story 10.2: Serve the model via FastAPI

As a learner,
I want a prediction endpoint,
So that other programs can use my model.

**Acceptance Criteria:**

**Given** a saved model
**When** I run a FastAPI `/predict` endpoint and POST an input
**Then** I get back a prediction and confidence
**And** the model is loaded once at startup, not per request.

### Story 10.3: Validate input and match preprocessing

As a learner,
I want input validation and training-matched preprocessing at the boundary,
So that the service is safe and accurate.

**Acceptance Criteria:**

**Given** the prediction endpoint
**When** I send malformed input
**Then** it is rejected with a clear error
**And** valid input is preprocessed exactly as during training.

### Story 10.4: Batch vs real-time inference

As a learner,
I want to understand batch and real-time inference,
So that I can match serving to the use case.

**Acceptance Criteria:**

**Given** the served model
**When** I run a single request and a batched request
**Then** I can describe the latency/throughput trade-off between them.

### Story 10.5: Monitor in production

As a learner,
I want to monitor a deployed model,
So that I notice degradation before users do.

**Acceptance Criteria:**

**Given** a running service
**When** I track latency, error rate, and live prediction metrics
**Then** I can define data drift vs concept drift and name a response to each.

### Story 10.6: Containerize and deploy (stretch)

As a learner,
I want to package the service in a container,
So that it runs the same anywhere.

**Acceptance Criteria:**

**Given** the FastAPI service
**When** I write a Dockerfile and build/run the image
**Then** the endpoint responds from inside the container
**And** the pytest suite from Epic 9 runs before deploy.

---

## Epic 11 — Capstone

Build, verify, and deploy one end-to-end project that proves mastery. Depends on all
previous epics.

### Story 11.1: Choose dataset and define the cost of mistakes

As a learner,
I want a real dataset and a clear cost of errors,
So that my metric choice is grounded in reality.

**Acceptance Criteria:**

**Given** a candidate problem
**When** I describe the dataset and the real-world cost of each mistake type
**Then** I can justify why one error type is worse than the other.

### Story 11.2: Choose architecture, loss, and metric

As a learner,
I want to select the architecture, loss, and the metric that matches the cost,
So that my design fits the goal.

**Acceptance Criteria:**

**Given** the defined cost of mistakes
**When** I pick architecture, loss, and primary metric
**Then** each choice is justified against the problem (per the source plan's mental model).

### Story 11.3: Build, verify, and deploy end to end

As a learner,
I want to implement the full pipeline,
So that I demonstrate every phase together.

**Acceptance Criteria:**

**Given** my design
**When** I build (Epic 8), verify honestly (Epic 9), and deploy (Epic 10)
**Then** a served endpoint returns predictions and my reported metric comes from a clean test set.

### Story 11.4: Write up results and reflection

As a learner,
I want to document outcomes and lessons,
So that I consolidate what I learned.

**Acceptance Criteria:**

**Given** the finished project
**When** I write up the metric results, failure modes, and what I'd do differently
**Then** the write-up ties the result back to the cost of mistakes from Story 11.1.

---

## Validation summary

- **Coverage:** All Learning Requirements LR-0.1 … LR-C.4 are covered by exactly one
  story (see FR Coverage Map). 11 epics, 60 stories.
- **Sequencing (NFR6):** Each epic depends only on earlier epics; no forward
  dependencies. Within each epic, stories are ordered and self-contained.
- **Template compliance:** Every story uses the `As a / I want / So that` form and at
  least one `Given/When/Then` acceptance criterion.
- **Cross-cutting NFRs:** The mastery bar (NFR1), honest evaluation (NFR2),
  reproducibility (NFR3), runnable code (NFR4), and source fidelity (NFR5) are baked
  into the acceptance criteria throughout.
