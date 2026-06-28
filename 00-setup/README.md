# 00 · Setup

*Goal: a working environment where you can run every example in this repo.*

## Why this comes first

You learn ML by running code and looking at the results — not by reading alone. So
before anything else, get a Python environment that works.

## Steps

1. **Install Python 3.11+.** Check with `python3 --version`.
2. **Create a virtual environment** (keeps this project's packages isolated):
   ```bash
   cd ml-real
   python3 -m venv .venv
   source .venv/bin/activate     # Windows: .venv\Scripts\activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (This is a big install because it includes PyTorch. That's expected.)
4. **Run your first script** — the three concept graphs from the source plan:
   ```bash
   python src/concept_graphs.py
   ```
   Open the PNGs in [`../assets/generated/`](../assets/generated/). You should see a
   cross-entropy curve, a precision/recall/F1 trade-off, and a cosine-similarity curve.
5. **Launch a notebook:**
   ```bash
   jupyter notebook
   ```
   Make a new notebook, run `import torch; print(torch.__version__)`.

## Checklist

- [ ] `python3 --version` shows 3.11 or higher
- [ ] Virtual environment activates
- [ ] `pip install -r requirements.txt` finishes without errors
- [ ] `python src/concept_graphs.py` produces three PNGs
- [ ] Jupyter runs a cell

## Troubleshooting

- **PyTorch install is slow/large** — normal. If you don't have a GPU, the CPU build
  is fine for everything here.
- **`command not found: python3`** — install from python.org or via your OS package
  manager (`brew install python` on macOS).
- **Matplotlib "building the font cache"** — harmless first-run message.

Next: [`../01-math/`](../01-math/)
