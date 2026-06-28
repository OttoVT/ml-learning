# Source plan (verbatim from ChatGPT)

> This file is the **original conversation** that seeded this repository, captured
> verbatim so we always have the source of truth.
>
> Source: https://chatgpt.com/share/6a418f3c-15e8-83ed-9977-3fcbcf311348
> Conversation title: *"Neural Networks and Metrics"*
>
> The rest of this repository expands this concept map into a full, step-by-step
> learning path (math → Python → build → verify → deploy). See
> [`../roadmap/ROADMAP.md`](../roadmap/ROADMAP.md).

---

## The question that started this (user)

> Various neural network architectures, including feedforward, recurrent,
> and transformer.
> Metrics such as accuracy, F1, precision, recall, cosine similarity, and
> cross entropy.
>
> I want you to teach me all those concepts. What are the most important things
> in those related to machine learning? I would like to see diagrams and graphs.

## The approach (assistant)

> I'll build this as a guided "map" of the concepts: architectures first, then
> metrics, then how to know which things matter most in practice. I'll also
> generate a few simple visual graphs you can inspect alongside the explanation.

---

# The concept map (assistant answer, verbatim)

Below is the practical map of the concepts you listed.

## 1. Neural network architectures

A **neural network** is a function that learns to map inputs to outputs.

```text
Input data  →  Neural network  →  Prediction
image       →  model           →  "cat"
text        →  model           →  sentiment
numbers     →  model           →  price estimate
```

The most important architectures you mentioned are:

## Feedforward neural networks

A **feedforward network** is the simplest standard neural network.

Information moves in one direction:

```text
Input layer        Hidden layers             Output layer

x1 ─────┐
x2 ─────┼──► [ neurons ] ─► [ neurons ] ─► prediction
x3 ─────┘
```

It is called **feedforward** because there is no memory or looping. The model sees one input and produces one output.

Use it for:

```text
tabular data
basic classification
basic regression
feature-based prediction
```

Example:

```text
Input:
age, income, country, previous purchases

Output:
probability customer buys product
```

Important idea:

> Feedforward networks learn nonlinear patterns from fixed-size input data.

They are often called **MLPs**, or **multilayer perceptrons**.

---

## Recurrent neural networks

A **recurrent neural network**, or **RNN**, is designed for sequences.

Unlike feedforward networks, it has memory.

```text
Word 1 ─► RNN ─► hidden state ─┐
Word 2 ─► RNN ─► hidden state ─┤
Word 3 ─► RNN ─► hidden state ─┤──► output
Word 4 ─► RNN ─► hidden state ─┘
```

The key idea:

```text
current output = function(current input + previous memory)
```

Use it for:

```text
text
time series
speech
music
sequential events
```

Example:

```text
"I love this movie" → positive sentiment
```

The RNN reads words one by one and updates its internal memory.

Important variants:

```text
Vanilla RNN  → simple but weak for long sequences
LSTM         → better long-term memory
GRU          → simpler alternative to LSTM
```

Main weakness:

> RNNs process sequences step by step, so they are slower and often struggle with long-range dependencies.

For example, in a long paragraph, a word near the beginning may affect the meaning of a word near the end. RNNs can have trouble preserving that information.

---

## Transformers

Transformers are the architecture behind modern large language models.

Their key mechanism is **attention**.

Instead of reading one token at a time like an RNN, a transformer compares tokens with other tokens.

```text
"The animal didn't cross the road because it was tired."

Transformer asks:
What does "it" refer to?
animal? road?
```

Simplified transformer diagram:

```text
Input tokens
    │
    ▼
Embeddings
    │
    ▼
Self-attention
    │
    ▼
Feedforward layers
    │
    ▼
Output predictions
```

The core idea:

```text
Each token looks at other tokens and decides which ones matter.
```

Self-attention diagram:

```text
        attention
"cat" ─────────────► "sat"
  │                  ▲
  ├───────────────► "mat"
  │
  └───────────────► "the"
```

Transformers are powerful because they can model relationships across long distances.

Use them for:

```text
language models
translation
summarization
image generation
code generation
speech
vision models
multimodal AI
```

The most important transformer concept:

> Attention lets the model decide which parts of the input are relevant to each other.

---

# 2. Metrics

Metrics tell you how good your model is.

Different problems need different metrics.

## Accuracy

**Accuracy** measures how often the model is correct.

```text
accuracy = correct predictions / total predictions
```

Example:

```text
100 examples
90 correct
accuracy = 90%
```

Good when classes are balanced.

Bad when classes are imbalanced.

Example:

```text
Disease detection dataset:
99% healthy
1% sick
```

A dumb model that always predicts "healthy" gets 99% accuracy, but it is useless.

So accuracy can be misleading.

---

## Precision

**Precision** answers:

> When the model says "positive," how often is it right?

```text
precision = true positives / predicted positives
```

Example:

```text
Model flags 100 emails as spam.
80 are actually spam.

precision = 80 / 100 = 80%
```

High precision means few false alarms.

Use precision when false positives are costly.

Examples:

```text
spam detection
fraud investigation
medical diagnosis alerts
moderation systems
```

---

## Recall

**Recall** answers:

> Of all real positives, how many did the model find?

```text
recall = true positives / actual positives
```

Example:

```text
There are 100 actual spam emails.
Model finds 70.

recall = 70 / 100 = 70%
```

High recall means the model misses fewer positives.

Use recall when false negatives are costly.

Examples:

```text
cancer screening
security threat detection
fraud detection
search systems
```

---

## Precision vs recall

There is often a trade-off.

```text
High precision:
Model is careful.
Fewer false positives.
May miss real positives.

High recall:
Model is aggressive.
Finds more positives.
May create more false positives.
```

Example:

```text
Security scanner

High precision:
Only alerts when very sure.
Misses some attacks.

High recall:
Alerts on anything suspicious.
Catches more attacks but creates noise.
```

Generated graph: [precision/recall/F1 tradeoff](sandbox:/mnt/data/ml_concept_graphs/precision_recall_f1_tradeoff.png)

---

## F1 score

**F1** combines precision and recall into one number.

```text
F1 = 2 × precision × recall / (precision + recall)
```

It is useful when you care about both precision and recall.

Example:

```text
precision = 0.80
recall = 0.60

F1 = 0.686
```

F1 is especially useful for imbalanced classification problems.

Important:

> F1 ignores true negatives, so it is not always the best metric.

For example, in some medical or risk systems, you may care deeply about both false positives and false negatives in different ways.

---

## Cross entropy

**Cross entropy** is usually a training loss, not just an evaluation metric.

It measures how wrong a probability prediction is.

Example:

```text
Correct class: cat

Model A:
cat: 0.90
dog: 0.10

Model B:
cat: 0.51
dog: 0.49

Model C:
cat: 0.01
dog: 0.99
```

Model A has low loss.
Model B has medium loss.
Model C has huge loss.

Cross entropy punishes confident wrong predictions very strongly.

Generated graph: [cross-entropy curve](sandbox:/mnt/data/ml_concept_graphs/cross_entropy_curve.png)

Core formula:

```text
cross entropy = -log(probability assigned to correct answer)
```

So if the model gives the correct class probability 1.0, loss is near 0.

If it gives the correct class probability near 0, loss becomes very large.

---

## Cosine similarity

**Cosine similarity** measures how similar two vectors are by direction.

It is commonly used with embeddings.

An embedding is a vector representation of something:

```text
"king"      → [0.21, -0.44, 0.88, ...]
"queen"     → [0.25, -0.40, 0.84, ...]
"airplane"  → [-0.71, 0.12, 0.03, ...]
```

Cosine similarity asks:

```text
Do these vectors point in similar directions?
```

```text
Same direction:       cosine similarity ≈ 1
Unrelated direction:  cosine similarity ≈ 0
Opposite direction:   cosine similarity ≈ -1
```

Diagram:

```text
similar vectors:

       A
      /
     /
    /
   / B

different vectors:

A ─────────►

B
│
│
▼
```

Generated graph: [cosine similarity curve](sandbox:/mnt/data/ml_concept_graphs/cosine_similarity_curve.png)

Use cosine similarity for:

```text
semantic search
recommendation systems
text similarity
image similarity
clustering
retrieval-augmented generation
```

Example:

```text
Query:
"best laptop for programming"

Document A:
"Top developer laptops in 2026"

Document B:
"How to bake sourdough bread"

Document A should have higher cosine similarity to the query.
```

---

# 3. The most important things to understand

## The big architecture picture

```text
Feedforward network:
fixed input → prediction

RNN:
sequence → memory → prediction

Transformer:
sequence → attention between all parts → prediction
```

In modern machine learning:

```text
Feedforward networks are still useful.
RNNs are historically important and still useful in some sequence tasks.
Transformers dominate modern language, vision, and multimodal AI.
```

The most important architecture to study deeply today is the **transformer**.

---

## The big metric picture

```text
Accuracy:
How often correct overall?

Precision:
When model says yes, how often is it right?

Recall:
Of all real yes cases, how many did it find?

F1:
Balance between precision and recall.

Cross entropy:
How good are the model's probability predictions?

Cosine similarity:
How similar are two embeddings?
```

---

# 4. Which metric should you use?

## Classification

For normal balanced classification:

```text
accuracy
cross entropy
```

For imbalanced classification:

```text
precision
recall
F1
confusion matrix
```

For medical screening:

```text
recall is often very important
```

Because missing a disease can be worse than sending someone for extra testing.

For spam detection:

```text
precision is very important
```

Because marking real email as spam is bad.

For fraud detection:

```text
recall and precision both matter
```

You want to catch fraud, but not overwhelm investigators with false alarms.

---

## Embedding/search systems

Use:

```text
cosine similarity
top-k accuracy
mean reciprocal rank
recall@k
```

For example, in semantic search:

```text
User query → embedding
Documents → embeddings
Rank documents by cosine similarity
```

---

## Language models

Use:

```text
cross entropy
perplexity
human evaluation
task-specific accuracy
```

Cross entropy is central during training.

---

# 5. Simple mental model

The most useful way to think about machine learning is:

```text
Architecture = what kind of machine learns the pattern

Loss = what the machine tries to minimize while learning

Metric = how humans judge whether the result is good
```

Example:

```text
Architecture:
Transformer

Loss:
cross entropy

Metric:
accuracy, F1, human preference, retrieval quality
```

---

# 6. Confusion matrix: the source of many metrics

For binary classification:

```text
                         Actual positive      Actual negative

Predicted positive       True positive        False positive

Predicted negative       False negative       True negative
```

From this table:

```text
accuracy  = (TP + TN) / all examples
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = balance of precision and recall
```

This is one of the most important things to understand in ML evaluation.

---

# 7. What to learn first

Study in this order:

```text
1. Train/test split
2. Classification vs regression
3. Loss functions
4. Accuracy, precision, recall, F1
5. Overfitting and generalization
6. Feedforward networks
7. Embeddings and cosine similarity
8. Attention
9. Transformers
10. Evaluation beyond simple metrics
```

The deepest practical lesson:

> A model is not "good" because one metric is high. It is good only if the metric matches the real-world cost of mistakes.

For example:

```text
A fraud model with high accuracy may still miss most fraud.
A medical model with high precision may still miss too many sick patients.
A chatbot with low cross entropy may still produce bad answers.
A search model with high cosine similarity may still retrieve irrelevant documents.
```

So the most important ML skill is not just knowing the formulas. It is knowing **which metric matches the real goal**.

---

# Appendix: the graph-generation code (verbatim)

The assistant generated three explanatory graphs with this script. A cleaned,
runnable version lives at [`../src/concept_graphs.py`](../src/concept_graphs.py).

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("/mnt/data/ml_concept_graphs")
out_dir.mkdir(exist_ok=True)

# 1. Cross-entropy loss vs predicted probability for true class
p = np.linspace(0.001, 0.999, 500)
ce = -np.log(p)
plt.figure(figsize=(7, 4.5))
plt.plot(p, ce)
plt.xlabel("Predicted probability assigned to the true class")
plt.ylabel("Cross-entropy loss")
plt.title("Cross-entropy strongly punishes confident wrong predictions")
plt.grid(True, alpha=0.3)
ce_path = out_dir / "cross_entropy_curve.png"
plt.savefig(ce_path, bbox_inches="tight", dpi=160)
plt.show()

# 2. Precision/recall trade-off as threshold changes on synthetic scores
thresholds = np.linspace(0.01, 0.99, 99)
# illustrative smooth curves, not from a real model
precision = 0.45 + 0.5 * thresholds**0.7
recall = 0.95 - 0.75 * thresholds**1.2
f1 = 2 * precision * recall / (precision + recall)
plt.figure(figsize=(7, 4.5))
plt.plot(thresholds, precision, label="Precision")
plt.plot(thresholds, recall, label="Recall")
plt.plot(thresholds, f1, label="F1")
plt.xlabel("Decision threshold")
plt.ylabel("Metric value")
plt.title("Precision, recall, and F1 usually trade off")
plt.legend()
plt.grid(True, alpha=0.3)
pr_path = out_dir / "precision_recall_f1_tradeoff.png"
plt.savefig(pr_path, bbox_inches="tight", dpi=160)
plt.show()

# 3. Cosine similarity as angle between vectors
angles = np.linspace(0, 180, 361)
cos_sim = np.cos(np.deg2rad(angles))
plt.figure(figsize=(7, 4.5))
plt.plot(angles, cos_sim)
plt.xlabel("Angle between two vectors, degrees")
plt.ylabel("Cosine similarity")
plt.title("Cosine similarity measures direction, not magnitude")
plt.grid(True, alpha=0.3)
cos_path = out_dir / "cosine_similarity_curve.png"
plt.savefig(cos_path, bbox_inches="tight", dpi=160)
plt.show()

print("Generated graphs:")
print(ce_path)
print(pr_path)
print(cos_path)
```
