# ml-real — learning machine learning, end to end, in one place

This repository is a **step-by-step learning environment** for understanding neural
networks: the math behind them, the Python to build them, how to **create**,
**verify**, and **deploy** them — and how to judge whether a model is actually good.

It started from one ChatGPT conversation about neural network architectures and
metrics. That conversation is preserved verbatim in
[`plan/chatgpt-source-plan.md`](plan/chatgpt-source-plan.md), and the whole repo
expands it into a real curriculum.

## How this repo is organized

The folders are numbered in the order you should work through them. Each module is
self-contained: a `README.md` that teaches the concept, plus code/notebooks/exercises.

| Module | What you learn | Why it's here |
| --- | --- | --- |
| [`00-setup`](00-setup/) | Install Python, set up the environment, run your first script | You can't learn ML without a working toolchain |
| [`01-math`](01-math/) | Linear algebra, calculus, probability — the minimum that matters | Neural nets *are* math; you need the intuition |
| [`02-python`](02-python/) | Python + NumPy + pandas + plotting | The language all ML is written in |
| [`03-foundations`](03-foundations/) | Train/test split, classification vs regression, overfitting | The ML mindset before any neural network |
| [`04-loss-and-training`](04-loss-and-training/) | Loss functions, gradient descent, cross-entropy | How a model actually *learns* |
| [`05-metrics-and-evaluation`](05-metrics-and-evaluation/) | Accuracy, precision, recall, F1, confusion matrix, cosine similarity | How you measure if it worked |
| [`06-architectures`](06-architectures/) | Feedforward, RNN, embeddings, attention, transformers | The model families from the source plan |
| [`07-build-neural-networks`](07-build-neural-networks/) | Build nets from scratch (NumPy) then with PyTorch | **Creating** them |
| [`08-verify-models`](08-verify-models/) | Validation, testing ML code, evaluation beyond one metric | **Verifying** them |
| [`09-deploy-models`](09-deploy-models/) | Save/load models, serve as an API, monitor | **Deploying** them |

Supporting folders:

- [`plan/`](plan/) — the original ChatGPT plan, verbatim (the source of truth).
- [`roadmap/`](roadmap/) — [`ROADMAP.md`](roadmap/ROADMAP.md), the full step-by-step path with checkboxes.
- [`src/`](src/) — shared, runnable code (e.g. the concept-graph generator).
- [`notebooks/`](notebooks/) — interactive Jupyter lessons.
- [`datasets/`](datasets/) — local data (git-ignored; each lesson says how to fetch).
- [`assets/`](assets/) — diagrams and generated graphs.

## Quick start

```bash
# 1. From the repo root, create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install everything
pip install -r requirements.txt

# 3. Generate the three concept graphs from the source plan
python src/concept_graphs.py
#    -> writes PNGs to assets/generated/

# 4. Start the roadmap
open roadmap/ROADMAP.md          # or just read it in your editor
```

## The one idea to keep in mind

From the source plan, the mental model that ties everything together:

```text
Architecture = what kind of machine learns the pattern
Loss         = what the machine tries to minimize while learning
Metric       = how humans judge whether the result is good
```

And the deepest practical lesson:

> A model is not "good" because one metric is high. It is good only if the metric
> matches the real-world cost of mistakes.

## How to use this repo day to day

1. Open [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md) and find your current step.
2. Go to that module's folder and read its `README.md`.
3. Do the exercises / run the code in that module.
4. Check the box in the roadmap. Move on.

The work is also tracked as **epics and stories** (one epic per phase, one story per
lesson) so progress is visible and reviewable — see
[`_bmad-output/planning-artifacts/epics.md`](_bmad-output/planning-artifacts/epics.md)
(11 epics, 60 stories, generated with BMAD).
