# 01 · Math foundations

*Goal: enough math intuition to read and reason about neural networks — not a degree.*

You do **not** need to be a mathematician. You need to be comfortable with a small
set of ideas, because neural networks are literally built from them.

## What a neural network is, mathematically

A single layer is:

```text
output = activation( W · x + b )
```

- `x` — the input vector (your data as numbers)
- `W` — a matrix of weights (what the model learns)
- `b` — a bias vector
- `·` — matrix multiplication
- `activation` — a nonlinear function (ReLU, sigmoid, …) so the model can learn curves

Stack a few of these and you have the feedforward network from the source plan.

## The minimum that matters

### Linear algebra
- [ ] Vectors and what a dot product means (similarity / projection)
- [ ] Matrices and matrix multiplication (a layer = a matrix multiply)
- [ ] Shapes: if `W` is `(m, n)` and `x` is `(n,)`, then `Wx` is `(m,)` — shape errors are the #1 beginner bug
- [ ] Vector length (norm) and the **angle between vectors** → sets up cosine similarity

### Calculus
- [ ] Derivative = slope = "which way and how fast does the output change?"
- [ ] Gradient = the derivative for many inputs at once = direction of steepest increase
- [ ] **Gradient descent**: step *against* the gradient to reduce the loss
- [ ] The **chain rule** — composing derivatives. Backpropagation is just the chain rule applied through the layers.

### Probability
- [ ] Probability distributions over classes (a model outputs `P(cat)=0.9, P(dog)=0.1`)
- [ ] `log` and log-probabilities, and why `log` turns products into sums
- [ ] This sets up **cross-entropy** = `-log(probability of the correct class)`

## Connect it to the source plan

- The **cosine similarity** graph in `src/concept_graphs.py` is pure vector geometry:
  `cos(θ)` between two vectors. Run it and match the curve to the angle idea.
- The **cross-entropy** graph is `-log(p)`. Run it and see why `p → 0` blows up.

## Exercises

1. By hand, multiply a `(2×3)` matrix by a `(3,)` vector. Then verify with NumPy.
2. Compute the cosine similarity of `[1, 0]` and `[1, 1]` by hand. (Answer ≈ 0.707.)
3. Plot `y = x²` and its derivative `2x`. Where is the slope zero? That's the minimum
   gradient descent is looking for.

## Resources
- 3Blue1Brown — *Essence of Linear Algebra* and *Essence of Calculus* (YouTube)
- Khan Academy — linear algebra, derivatives, probability

Next: [`../02-python/`](../02-python/)
