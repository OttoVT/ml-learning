# 04 · Loss & training

*Goal: understand how a model actually learns.*
*(Source plan "what to learn first": step 3. Source section 5: "Loss = what the
machine tries to minimize while learning.")*

## The big picture

```text
Architecture = what kind of machine learns the pattern
Loss         = what the machine tries to minimize while learning   ← this module
Metric       = how humans judge whether the result is good
```

Training is a loop:

```text
1. Make a prediction with current weights
2. Measure how wrong it is  → the LOSS
3. Compute the gradient of the loss w.r.t. the weights
4. Nudge the weights to reduce the loss   → gradient descent
5. Repeat
```

## Loss functions

A **loss function** turns "how wrong was the model" into a single number to minimize.

### Mean squared error (regression)
```text
MSE = average( (prediction - target)² )
```
Big errors are punished more (because of the square).

### Cross-entropy (classification) — the star of the source plan
```text
cross-entropy = -log(probability assigned to the correct answer)
```

From the source plan:

```text
Correct class: cat
Model A: cat 0.90 → low loss
Model B: cat 0.51 → medium loss
Model C: cat 0.01 → HUGE loss
```

> Cross-entropy punishes confident wrong predictions very strongly.

Run `python ../src/concept_graphs.py` and look at `cross_entropy_curve.png`: as the
probability of the correct class goes to 0, the loss shoots to infinity. That's the
model being heavily penalized for being confidently wrong.

- [ ] Explain why we use `-log` (low prob → big loss; prob 1 → zero loss)
- [ ] Compute cross-entropy for the A/B/C example above by hand
- [ ] Know that cross-entropy pairs with a softmax output for multi-class

## Gradient descent

To reduce the loss, step the weights *against* the gradient:

```text
new_weight = old_weight - learning_rate × gradient
```

- **Gradient** — the direction the loss increases fastest (from Phase 1 calculus)
- **Learning rate** — how big a step you take. Too big → diverge. Too small → crawl.
- **Epoch** — one full pass over the training data
- **Batch** — a chunk of examples processed at once (mini-batch gradient descent)

- [ ] Understand the update rule above
- [ ] Know what learning rate, epoch, and batch mean
- [ ] Recognize the symptoms of a too-high / too-low learning rate

## Hands-on

Implement gradient descent by hand to fit a line `y = w·x + b`:

```python
import numpy as np

x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)   # true relationship: y = 2x
w, b, lr = 0.0, 0.0, 0.01

for step in range(1000):
    pred = w * x + b
    error = pred - y
    loss = (error ** 2).mean()             # MSE
    grad_w = (2 * error * x).mean()        # d(loss)/dw
    grad_b = (2 * error).mean()            # d(loss)/db
    w -= lr * grad_w
    b -= lr * grad_b

print(w, b)   # should approach 2, 0
```

- [ ] Run it; watch `w` approach 2
- [ ] Print the loss every 100 steps and watch it shrink
- [ ] Break it on purpose with `lr = 1.0` and see it explode

This loop is *exactly* what PyTorch automates later — autograd just computes the
gradients for you.

## Resources
- 3Blue1Brown — *Gradient descent, how neural networks learn*
- Andrej Karpathy — *micrograd* (backprop from scratch)

Next: [`../05-metrics-and-evaluation/`](../05-metrics-and-evaluation/)
