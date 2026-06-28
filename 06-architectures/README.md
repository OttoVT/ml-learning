# 06 · Architectures

*Goal: understand the neural network families from the source plan.*
*(Source plan section 1; "what to learn first" steps 6, 7, 8, 9.)*

A neural network is **a function that learns to map inputs to outputs.**

```text
Input data  →  Neural network  →  Prediction
image       →  model           →  "cat"
text        →  model           →  sentiment
numbers     →  model           →  price estimate
```

The three families, in one picture (source section 3):

```text
Feedforward:  fixed input → prediction
RNN:          sequence → memory → prediction
Transformer:  sequence → attention between all parts → prediction
```

## Feedforward networks (MLPs) — source step 6

The simplest standard net. Information flows one direction, no memory:

```text
x1 ─┐
x2 ─┼──► [ neurons ] ─► [ neurons ] ─► prediction
x3 ─┘
   input      hidden layers       output
```

- Each layer is `activation(W·x + b)` (the math from module 01)
- Good for **tabular data**, basic classification/regression
- Also called an **MLP** (multilayer perceptron)

> Feedforward networks learn nonlinear patterns from fixed-size input data.

- [ ] Draw an MLP and label the layers
- [ ] Explain why the nonlinear activation is essential (without it, many layers
      collapse into one linear layer)

## Recurrent networks (RNNs) — source step 8 precursor

Designed for **sequences**. Unlike feedforward nets, an RNN has **memory**:

```text
Word 1 ─► RNN ─► hidden state ─┐
Word 2 ─► RNN ─► hidden state ─┤
Word 3 ─► RNN ─► hidden state ─┤──► output
Word 4 ─► RNN ─► hidden state ─┘

current output = function(current input + previous memory)
```

- Good for text, time series, speech, music
- Variants: **vanilla RNN** (weak on long sequences), **LSTM** (better long-term
  memory), **GRU** (simpler LSTM)

> RNNs process sequences step by step, so they are slower and often struggle with
> long-range dependencies.

- [ ] Explain the hidden-state / memory idea
- [ ] Explain the long-range dependency weakness (a word at the start affecting a
      word at the end)

## Embeddings — source step 7 (pairs with cosine similarity)

An **embedding** turns a thing (word, sentence, image) into a vector so the model can
do math on meaning:

```text
"king"     → [0.21, -0.44, 0.88, ...]
"queen"    → [0.25, -0.40, 0.84, ...]
"airplane" → [-0.71, 0.12, 0.03, ...]
```

Similar things get vectors pointing in similar directions → measure with **cosine
similarity** (module 05). This powers semantic search, recommendations, and RAG.

- [ ] Generate sentence embeddings (e.g. `sentence-transformers`) and rank by cosine
      similarity to a query

## Attention — source step 8

The key mechanism in transformers. Instead of reading one token at a time, the model
lets **each token look at the others and decide which matter**:

```text
"The animal didn't cross the road because it was tired."
What does "it" refer to?  animal? road?  ← attention resolves this

        attention
"cat" ─────────────► "sat"
  ├───────────────► "mat"
  └───────────────► "the"
```

> Attention lets the model decide which parts of the input are relevant to each other.

- [ ] Explain self-attention in one sentence
- [ ] Understand why this beats RNNs for long-range dependencies (every token reaches
      every other directly, not step-by-step)

## Transformers — source step 9

The architecture behind modern LLMs, vision models, and multimodal AI:

```text
Input tokens → Embeddings → Self-attention → Feedforward layers → Output predictions
```

> The most important architecture to study deeply today is the transformer.

Used for: language models, translation, summarization, code/image/speech generation,
vision, multimodal AI.

- [ ] Trace data through the transformer diagram above
- [ ] Know that embeddings + attention + feedforward are the building blocks you
      already learned

## Hands-on (leads into module 07)
- [ ] Run a pretrained transformer with `transformers` (e.g. sentiment analysis)
- [ ] Compare: solve a tiny sequence task with an RNN vs. with attention

Next: [`../07-build-neural-networks/`](../07-build-neural-networks/)
