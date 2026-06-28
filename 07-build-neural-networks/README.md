# 07 · Build neural networks

*Goal: **create** neural networks yourself — first from scratch, then with PyTorch.*

You learned the math (01), the Python (02), how training works (04), and the
architectures (06). Now you build.

## Stage 1 — From scratch in NumPy

The point isn't to write production code; it's to remove all magic. If you can build
an MLP and its backprop by hand, you'll never be confused by a framework again.

- [ ] **A neuron**: `output = activation(w · x + b)`
- [ ] **A layer**: stack neurons → `activation(W·x + b)` (one matrix multiply)
- [ ] **Forward pass**: chain layers to get a prediction
- [ ] **A loss**: cross-entropy or MSE (module 04)
- [ ] **Backpropagation**: apply the chain rule (module 01) to get gradients
- [ ] **Training loop**: forward → loss → backward → gradient-descent update (module 04)
- [ ] Train your NumPy MLP on a small real dataset (e.g. Iris / MNIST subset) and
      reach decent test accuracy

Skeleton:

```python
import numpy as np

def relu(z):       return np.maximum(0, z)
def softmax(z):    e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, n_in, n_hidden, n_out):
        self.W1 = np.random.randn(n_in, n_hidden) * 0.01
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_out) * 0.01
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return softmax(self.z2)
    # TODO: backward() — derive the gradients and update the weights
```

## Stage 2 — The same thing in PyTorch

Now let the framework compute gradients for you (autograd) and see how much less code
it takes.

- [ ] Tensors and `requires_grad` / autograd
- [ ] Define a model by subclassing `nn.Module`
- [ ] Pick a loss (`nn.CrossEntropyLoss`, `nn.MSELoss`)
- [ ] Pick an optimizer (`torch.optim.SGD`, `torch.optim.Adam`)
- [ ] Write the canonical training loop
- [ ] Reproduce your NumPy MLP's results with far less code

Canonical PyTorch loop:

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_out))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    opt.zero_grad()
    logits = model(X_train)            # forward
    loss = loss_fn(logits, y_train)    # measure
    loss.backward()                    # autograd computes gradients
    opt.step()                         # gradient-descent update
```

Compare this to the by-hand loop in module 04 — same five steps, gradients automated.

## Stage 3 — Beyond the MLP

- [ ] Build and train a small **RNN/LSTM** on a sequence task
- [ ] Build a minimal **self-attention** block and see how tokens attend to each other
- [ ] (Stretch) Assemble a tiny transformer block (embedding → attention → feedforward)

## Verify as you go

Always report **test** performance (module 03) using the **right metric** (module
05), not training accuracy. A net that nails training data but fails the test set is
overfitting — that's the bridge to module 08.

## Resources
- Andrej Karpathy — *Neural Networks: Zero to Hero* (micrograd → makemore → GPT)
- PyTorch official "Learn the Basics" tutorial

Next: [`../08-verify-models/`](../08-verify-models/)
