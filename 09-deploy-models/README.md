# 09 · Deploy models

*Goal: **deploy** a model so something other than your laptop can use it.*

A model only creates value once it serves real predictions. This module turns your
trained model into a service — and keeps an eye on it afterward.

## 1. Save and load the model

- [ ] Save trained weights (`torch.save(model.state_dict(), "model.pt")`)
- [ ] Load them into the same architecture and reproduce a prediction
- [ ] Save everything needed to reproduce inference: weights, preprocessing,
      class labels, and the config
- [ ] (Optional) Export to a portable format like **ONNX** for cross-framework serving

```python
import torch
torch.save(model.state_dict(), "model.pt")
# later:
model.load_state_dict(torch.load("model.pt"))
model.eval()   # turn off dropout/batchnorm training behavior
```

## 2. Serve it as an API

Wrap the model in a small web service (this repo includes FastAPI for exactly this):

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = ...            # load once at startup
model.eval()

class Request(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(req: Request):
    x = torch.tensor([req.features])
    with torch.no_grad():
        logits = model(x)
        prob = logits.softmax(dim=1)[0]
    return {"prediction": int(prob.argmax()), "confidence": float(prob.max())}
```

```bash
uvicorn app:app --reload      # then POST to http://127.0.0.1:8000/predict
```

- [ ] Load the model **once** at startup, not per request
- [ ] Use `model.eval()` and `torch.no_grad()` for inference
- [ ] **Validate inputs** at the boundary (Pydantic) — never trust raw input
- [ ] Apply the **same preprocessing** as training (a mismatch silently wrecks accuracy)
- [ ] Return calibrated confidence, not just a class

## 3. Inference patterns

- [ ] **Real-time** (one request at a time, low latency) vs **batch** (many at once,
      high throughput)
- [ ] Batch requests together when you can — it's far more efficient on GPUs
- [ ] Know your latency budget and measure it

## 4. Monitor in production

Deployment is not the finish line — models decay as the world changes.

- [ ] **Operational metrics**: latency, throughput, error rate, resource usage
- [ ] **Model metrics**: track the same metrics from module 05 on live predictions
      (where you can get labels)
- [ ] **Data drift**: input distribution moving away from training data
- [ ] **Concept drift**: the relationship between inputs and labels changing
- [ ] Alerting + a plan to retrain/roll back

## 5. Package it (stretch)

- [ ] Write a `Dockerfile` and run the API in a container
- [ ] Deploy to a host (a VM, a container platform, or a serverless runtime)
- [ ] Add CI: run the `pytest` suite from module 08 before every deploy

## The full loop

```text
Build (07) → Verify (08) → Deploy (09) → Monitor → (drift?) → Retrain → Verify → ...
```

Deployment closes the loop back to evaluation: the **metrics** you chose in module 05
are exactly what you watch in production.

## Resources
- FastAPI official docs (first-steps + request bodies)
- *Designing Machine Learning Systems* (Chip Huyen) — deployment & monitoring
- *Made With ML* — serving, monitoring, CI/CD for ML

Back to the start: [`../roadmap/ROADMAP.md`](../roadmap/ROADMAP.md) → the **Capstone**.
