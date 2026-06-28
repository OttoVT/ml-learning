# 02 · Python & NumPy

*Goal: write clean numerical Python fluently — the language all ML is written in.*

## What you need (and what you don't)

You need *practical* Python and the numerical stack. You do **not** need advanced
software-engineering Python (metaclasses, async, etc.) to start.

### Python essentials
- [ ] Variables, types, f-strings
- [ ] `if` / `for` / `while`
- [ ] Functions, default arguments, return values
- [ ] Lists, dicts, tuples, sets
- [ ] List/dict comprehensions
- [ ] Classes (you'll subclass `nn.Module` in PyTorch later)
- [ ] Imports, virtual environments, `pip`

### NumPy — the core of numerical Python
- [ ] Creating arrays, `dtype`, `shape`
- [ ] Indexing and slicing
- [ ] **Broadcasting** (operations between different-shaped arrays) — the key mental model
- [ ] Vectorized math (no Python loops) — `a * b`, `np.dot`, `a.sum(axis=0)`
- [ ] `np.linspace`, `np.exp`, `np.log` (you'll use these constantly)

### pandas — tabular data
- [ ] Load a CSV with `pd.read_csv`
- [ ] Inspect: `.head()`, `.info()`, `.describe()`
- [ ] Select columns, filter rows, `.groupby`

### matplotlib — see your data
- [ ] `plt.plot` (line) and `plt.scatter` (points)
- [ ] Labels, title, legend, `plt.savefig`

## Why NumPy matters so much

A neural network layer `W · x + b` is one NumPy line:

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])          # input, shape (3,)
W = np.random.randn(2, 3)              # weights, shape (2, 3)
b = np.array([0.1, -0.2])             # bias,    shape (2,)

out = W @ x + b                        # shape (2,) — one layer, no loops
```

If you understand that line, you understand the computational core of a feedforward net.

## Exercises

1. Re-create the three graphs from `src/concept_graphs.py` yourself from scratch,
   without looking. Compare to the originals in `assets/generated/`.
2. Implement cosine similarity in NumPy:
   ```python
   def cosine_similarity(a, b):
       return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
   ```
   Test it on `[1, 0]` vs `[0, 1]` (≈ 0) and `[1, 1]` vs `[2, 2]` (≈ 1).
3. Implement cross-entropy for one example: `-np.log(p_correct)`. Confirm it matches
   the curve in the cross-entropy graph.

## Resources
- The official NumPy "Absolute Beginner's Guide"
- *Python Data Science Handbook* (Jake VanderPlas) — free online

Next: [`../03-foundations/`](../03-foundations/)
