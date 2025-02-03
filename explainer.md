# Mechanistic Interpretability in a Tiny Transformer: A “Reverse + Increment” Toy Task

## 1. Overview 

We will work with a tiny vocabulary consisting of the tokens `{0, 1, 2, 3}` plus a “BOS” (beginning-of-sequence) token labeled `4`. 

### The Task

- **Inputs**: Four tokens `[x0, x1, x2, x3]`, each in `{0, 1, 2, 3}`.
- **Outputs**: Four tokens `[y0, y1, y2, y3]`, where 
  - `y0 = (x3 + 1) mod 4`, 
  - `y1 = (x2 + 1) mod 4`, 
  - `y2 = (x1 + 1) mod 4`, 
  - `y3 = (x0 + 1) mod 4`.

In other words, we **reverse** the order of the input tokens and **increment** each by 1 modulo 4.

### Full Sequence for Next-Token Prediction

We prepend a `BOS` token (value = 4) at the start, so each example’s full sequence is:

\[
[\text{BOS},\, x0,\, x1,\, x2,\, x3,\, y0,\, y1,\, y2,\, y3].
\]

When we do *next-token prediction*, we feed the first 8 tokens (`positions 0..7`) and ask the model to predict the next 8 tokens (`positions 1..8`). 

However, we only calculate the training loss on the 4 tokens that matter (the output half of the sequence). This focuses the model’s capacity on learning the transformation from inputs to outputs, instead of also trying to “predict” the input tokens themselves.

---

## 2. Why Use a Holdout Set?

In any machine learning experiment, it’s common to separate our data into a **training** set—on which the model is directly optimized—and a **holdout** or *test* set—used purely for checking that the model has truly learned to *generalize*, rather than simply memorize or “overfit” the training examples.

In many educational or toy tasks, it’s easy to inadvertently memorize a small dataset. But by *withholding* some examples and only checking them *after* training, we can confirm whether the model’s discovered “circuit” (i.e. how it processes inputs internally) is genuinely robust to unseen data.

Here, our dataset is quite small—just 256 possible 4-token sequences (since each of the 4 positions can be `0,1,2,3`, we have `4^4 = 256` possible inputs). We use 200 of these for training, and 56 for holdout.

## 3. Building the Dataset

Below is the code snippet that creates the entire dataset of size 256. Then, we split it into **train** and **holdout** sets.

```python
import torch
import torch.nn.functional as F
import numpy as np
import itertools
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "colab"
import tqdm.auto as tqdm
import einops

from transformer_lens import HookedTransformer, HookedTransformerConfig

# Generate all possible 4-token sequences from tokens 0,1,2,3.
all_sequences = list(itertools.product(range(4), repeat=4))
all_sequences = torch.tensor(all_sequences, dtype=torch.long)  # shape: [256, 4]

# Compute targets: add 1 mod 4 to each token then reverse the sequence.
targets = torch.flip((all_sequences + 1) % 4, dims=[1])  # shape: [256, 4]

# Define BOS token as 4 (so our vocabulary is {0,1,2,3,4})
BOS_TOKEN = 4
BOS_column = torch.full((all_sequences.size(0), 1), BOS_TOKEN, dtype=torch.long)

full_dataset = torch.cat([BOS_column, all_sequences, targets], dim=1)
print("Full dataset shape:", full_dataset.shape)
print("First 5 examples:\n", full_dataset[:5])

# 2. Split the Data into Training and Holdout Sets
train_dataset = full_dataset[:200]
holdout_dataset = full_dataset[200:]
print("Training set shape:", train_dataset.shape, "Holdout set shape:", holdout_dataset.shape)
```

**Notes on the Data Construction**

1. We create all possible 4-token sequences from `{0,1,2,3}`.
2. We define the *target* sequence by reversing `x0..x3` and incrementing each by 1 (mod 4).
3. We add a special BOS token = `4` to the front. This means the vocabulary is now `{0,1,2,3,4}`.
4. Each final sequence has length 9 (`[BOS, x0, x1, x2, x3, y0, y1, y2, y3]`).

---

<a name="architecture"></a>
## 4. Defining Our Mini-Transformer Architecture

Transformers typically have many layers and large hidden dimensions. To make our example interpretable, we’ll use a *very small* configuration:

- **2 layers**
- **Hidden dimension = 4** (`d_model = 4`)
- **Single attention head** (with `d_head=2`)
- **MLP dimension = 4** 
- **Vocabulary size = 5** (tokens `{0,1,2,3,4}`)
- **Maximum context length = 9**

This is intentionally tiny, so that (1) it can learn the task quickly, and (2) we can attempt to interpret the learned parameters.

```python
#########################################
# 3. Define a Tiny Transformer Model
#########################################
cfg = HookedTransformerConfig(
    n_layers=2,       # Use 2 layers to help with capacity.
    d_model=4,        # Hidden dimension = 4.
    d_head=2,         # Head dimension = 2.
    n_heads=1,        # Single attention head.
    d_mlp=4,          # MLP hidden dimension = 4.
    d_vocab=5,        # Vocabulary: tokens 0,1,2,3 and BOS=4.
    n_ctx=9,          # Sequence length is 9.
    act_fn='relu',
)
model = HookedTransformer(cfg).cuda()
print("Model configuration:\n", model)
```

You’ll see in the printed `model` description that we have exactly two `TransformerBlock`s, each containing a single-head attention mechanism and a small feed-forward MLP (`d_mlp=4`). The final linear layer (`unembed`) maps hidden states back to our 5-token vocabulary.

---

<a name="training"></a>
## 5. Training the Model

### Loss Function  
We *only* compute loss on the output tokens, i.e., positions `[5,6,7,8]` in the full sequence. (Positions `[0..4]` are either the BOS or input tokens that we do *not* need to predict.)

```python
#########################################
# 4. Define the Loss Function (Output Tokens Only)
#########################################
def loss_fn(logits, tokens):
    # logits: shape [batch, seq_len-1, vocab_size] 
    # (we feed in 8 tokens to predict next 8)
    # tokens: shape [batch, 9]
    return F.cross_entropy(
        logits[:, 4:].reshape(-1, logits.size(-1)),  
        tokens[:, 5:].reshape(-1)
    )
```

### Evaluation Function  
To measure *accuracy*, we’ll check how often the model’s *argmax* matches the target tokens for the 4 outputs (positions `[5..8]`).

```python
#########################################
# 5. Define the Evaluation Function
#########################################
def evaluate_model(model, dataset):
    model.eval()
    with torch.no_grad():
        logits = model(dataset[:, :-1].cuda())  # feed first 8 tokens
        preds = logits.argmax(dim=-1)
        # Compare predictions for positions 4..7 vs. tokens at positions 5..8
        correct = (preds[:, 4:] == dataset[:, 5:].cuda()).sum().item()
        total = dataset.size(0) * 4  # each example has 4 output tokens
        return correct / total
```

### Hyperparameters and Training Loop

```python
#########################################
# 6. Training Hyperparameters & Loop
#########################################
num_epochs = 4000
batch_size = 16   # small batch size
lr = 1e-2         # relatively high learning rate

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
losses = []

for epoch in range(num_epochs):
    model.train()
    indices = torch.randperm(train_dataset.size(0))[:batch_size]
    batch = train_dataset[indices].cuda()
    
    logits = model(batch[:, :-1])
    loss = loss_fn(logits, batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # Print periodic updates
    if epoch % 100 == 0:
        train_acc = evaluate_model(model, train_dataset)
        holdout_acc = evaluate_model(model, holdout_dataset)
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc*100:5.2f}% | "
              f"Holdout Acc: {holdout_acc*100:5.2f}%")

# Plot training loss
px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Training Loss").show()
```

**Key Observations:**
- By ~4000 epochs, the model *should* reach close to 100% accuracy on both training and holdout sets.  
- We confirm it’s not overfitting by verifying that holdout accuracy also goes to 100%.  
- The final model reliably performs the “reverse + increment mod 4” task on all 256 possible sequences.

---

## 6. Evaluating Accuracy on Training vs. Holdout

You should see printouts during training:

```
Epoch    0 | Loss: 2.4985 | Train Acc: 27.12% | Holdout Acc: 15.18%
...
Epoch 1000 | Loss: 0.1240 | Train Acc: 86.75% | Holdout Acc: 66.96%
...
Epoch 2000 | Loss: 0.0015 | Train Acc: 100.00% | Holdout Acc: 100.00%
...
Epoch 3900 | Loss: 0.0007 | Train Acc: 100.00% | Holdout Acc: 100.00%
```

In many runs, the model eventually achieves **perfect** accuracy on both sets. This is great news: we can be confident that the model’s solution *generalizes* to all 256 sequences.

At this point, we’ve shown that:
- The model can perfectly learn the function “reverse + increment” on a toy problem.
- We have separate training vs. holdout sets to ensure we’re not just memorizing the data.

Next, let’s try to *interpret* what the model is doing internally.

---

## 7. Inspecting the Attention Patterns

**Goal**: We hypothesize that the self-attention mechanism is responsible for “reversing” the order of the input tokens, i.e. each output position “looks” at the input token that will become `(x_i + 1) mod 4` after the MLP.

### Code: Extracting and Printing Attention

```python
#########################################
# 7. Inspecting the Attention Patterns
#########################################
model.eval()
with torch.no_grad():
    # Pick a few samples from the holdout set for analysis
    sample = holdout_dataset[:4].cuda()  # shape [4, 9]
    logits, cache = model.run_with_cache(sample[:, :-1])
    
    # The attention patterns are stored in the cache under the key "pattern".
    attn = cache["pattern", 0]  # from layer 0, shape [batch, n_heads, 8, 8]
    
    for i in range(sample.size(0)):
        attn_matrix = attn[i, 0].cpu().numpy()
        print(f"\nAttention pattern for holdout sample {i}:")
        print(attn_matrix)
        # Plot as a heatmap
        fig = px.imshow(
            attn_matrix,
            title=f"Attention Pattern (Sample {i}, Layer 0, Head 0)",
            labels={"x": "Source Position", "y": "Destination Position"},
            x=[str(j) for j in range(attn_matrix.shape[1])],
            y=[str(j) for j in range(attn_matrix.shape[0])],
            color_continuous_scale="RdBu",
        )
        fig.show()
```

### Interpreting the Results

When you run the above code, you’ll see printed matrices of shape (8×8), plus a heatmap (if you’re in a notebook environment).

- **Rows** correspond to the “destination” position (the query in self-attention).
- **Columns** correspond to the “source” position (the keys/values that the row “looks at”).

We fed in tokens `[0..7]` (the last token in the 9-length sequence is left out, because we do next-token prediction). So these 8 positions correspond to:

1. Position 0: BOS  
2. Position 1: x0  
3. Position 2: x1  
4. Position 3: x2  
5. Position 4: x3  
6. Position 5: y0  
7. Position 6: y1  
8. Position 7: y2  

*(Note that the model is predicting positions 1..8, but we only *calculate loss* on positions 5..8. You can see how the indexing can get a bit tricky!)*

### Where Do We Expect “Reversal”?

- For `y0` (which sits at position 5 in the input), we ideally want it to attend strongly to `x3` (which sits at position 4).
- For `y1` (position 6 in the input) to attend strongly to `x2` (position 3), and so on.

In an ideal “perfectly interpretable” scenario, row 5’s attention would be *all* on column 4, row 6’s attention *all* on column 3, etc. In practice, the attention might be distributed.  

Here’s a placeholder snippet of an attention matrix for holdout sample 0 (e.g., row 5 might have a large value on column 4):

```
Attention pattern for holdout sample 0:
[[1.00000000e+00 0.00000000e+00 ... ]
 ...
 [1.08846769e-01 2.76224018e-05 ... ]
 ...
 [4.68673646e-01 2.25587864e-04 ... ]]
```

**[Insert Heatmap Visualization for holdout sample 0 here]**

You might see a row (like row 5) that has a ~0.89 attention on column 3, which aligns with the intuition that `y1` is derived primarily from `x2`.

**Important**: Because the model is quite tiny, the patterns are not always “purely diagonal reversed.” Some attention is spread out, or some rows do unexpected things. Still, we see a strong *tendency* for the target positions to attend to the correct reversed input positions. 

Thus, *attention is capturing the “where to read from” for the reversal operation*.

---

<a name="mlp"></a>
## 8. Inspecting the MLP Circuit

### Why Look at the MLP?
Attention alone only *selects* which token’s hidden state to focus on. We still need to *increment* that token by 1 modulo 4. That’s presumably done by the feed-forward layer(s), i.e. the MLP. Mechanistically, we suspect the MLP transforms the hidden representation of “token k” into “token (k+1) mod 4.”

### Post-Activation Outputs

By hooking into the “cache” from `transformer_lens`, we can extract `mlp_out` from, say, layer 0 (the output after the first MLP). For example:

```python
mlp_out = cache["mlp_out", 0]  # [batch, seq_len, d_model]
print("\nMLP outputs (Layer 0) for the first sample (all positions):")
print(mlp_out[0].cpu().numpy())
```

You’ll see something like:

```
[[ 0.08655317  0.17682916 -0.40267545  0.05291257]
 [ 0.39043748  0.00528924 -0.37168258  0.12626119]
 ...
 [-0.21111171 -3.6924958   2.0988955   1.8454499 ]
 ... ]
```

At different positions (the input vs. output tokens), we get distinct 4-dimensional vectors. In principle, if we correlated these vectors with the embedding vectors for each token, we might see a pattern such that the MLP transforms `[embedding of x_i]` into something close to `[embedding of (x_i+1) mod 4]`.

### Printing the MLP Weights

We can also directly look at the MLP’s weight matrices `W_in` and `W_out`:

```python
layer = 0
mlp_in_weight = model.blocks[layer].mlp.W_in.data.cpu().numpy()
mlp_out_weight = model.blocks[layer].mlp.W_out.data.cpu().numpy()

print("\nLayer 0 MLP W_in weight matrix:")
print(mlp_in_weight)
print("\nLayer 0 MLP W_out weight matrix:")
print(mlp_out_weight)
```

You’ll get two small matrices of shape `(4,4)` and `(4,4)`. They don’t look like a perfect permutation or identity, but with some effort you might interpret them as forming a “shift by 1 mod 4” operation in the learned representation space. Because of the *tiny* dimension, the model will do its best to *pack* the needed transformations into just a few parameters, so it may not appear “clean.”

---

## 9. Full Code Listing

For completeness, here’s the combined code (the same as we’ve been walking through), ready to run in a single cell if desired:

```python
import torch
import torch.nn.functional as F
import numpy as np
import itertools
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "colab"
import tqdm.auto as tqdm
import einops

from transformer_lens import HookedTransformer, HookedTransformerConfig

#########################################
# 1. Build the STATIC Dataset of 256 Examples
#########################################
all_sequences = list(itertools.product(range(4), repeat=4))
all_sequences = torch.tensor(all_sequences, dtype=torch.long)  # [256, 4]
targets = torch.flip((all_sequences + 1) % 4, dims=[1])        # [256, 4]
BOS_TOKEN = 4
BOS_column = torch.full((all_sequences.size(0), 1), BOS_TOKEN, dtype=torch.long)
full_dataset = torch.cat([BOS_column, all_sequences, targets], dim=1)
train_dataset = full_dataset[:200]
holdout_dataset = full_dataset[200:]

#########################################
# 2. Define a Tiny Transformer Model
#########################################
cfg = HookedTransformerConfig(
    n_layers=2,
    d_model=4,
    d_head=2,
    n_heads=1,
    d_mlp=4,
    d_vocab=5,
    n_ctx=9,
    act_fn='relu',
)
model = HookedTransformer(cfg).cuda()
print("Model configuration:\n", model)

#########################################
# 3. Define the Loss Function (Output Tokens Only)
#########################################
def loss_fn(logits, tokens):
    return F.cross_entropy(
        logits[:, 4:].reshape(-1, logits.size(-1)),
        tokens[:, 5:].reshape(-1)
    )

#########################################
# 4. Define Evaluation Function
#########################################
def evaluate_model(model, dataset):
    model.eval()
    with torch.no_grad():
        logits = model(dataset[:, :-1].cuda())
        preds = logits.argmax(dim=-1)
        correct = (preds[:, 4:] == dataset[:, 5:].cuda()).sum().item()
        total = dataset.size(0) * 4
        return correct / total

#########################################
# 5. Train the Model
#########################################
num_epochs = 4000
batch_size = 16
lr = 1e-2

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
losses = []

for epoch in range(num_epochs):
    model.train()
    indices = torch.randperm(train_dataset.size(0))[:batch_size]
    batch = train_dataset[indices].cuda()
    
    logits = model(batch[:, :-1])
    loss = loss_fn(logits, batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        train_acc = evaluate_model(model, train_dataset)
        holdout_acc = evaluate_model(model, holdout_dataset)
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc*100:5.2f}% | "
              f"Holdout Acc: {holdout_acc*100:5.2f}%")

px.line(y=losses, labels={"x": "Epoch", "y": "Loss"}, title="Training Loss").show()

#########################################
# 6. Inspecting the Attention Patterns
#########################################
model.eval()
with torch.no_grad():
    sample = holdout_dataset[:4].cuda()
    logits, cache = model.run_with_cache(sample[:, :-1])
    attn = cache["pattern", 0]  # [batch=4, n_heads=1, seq=8, seq=8]
    for i in range(sample.size(0)):
        attn_matrix = attn[i, 0].cpu().numpy()
        print(f"\nAttention pattern for holdout sample {i}:")
        print(attn_matrix)
        fig = px.imshow(
            attn_matrix,
            title=f"Attention Pattern (Sample {i}, Layer 0, Head 0)",
            labels={"x": "Source Pos", "y": "Dest Pos"},
            color_continuous_scale="RdBu",
        )
        fig.show()

#########################################
# 7. Inspect MLP (FFN) Outputs and Weights
#########################################
    mlp_out = cache["mlp_out", 0]  # [4, 8, 4]
    print("\nMLP outputs (Layer 0) for the first sample (all positions):")
    print(mlp_out[0].cpu().numpy())

layer = 0
mlp_in_weight = model.blocks[layer].mlp.W_in.data.cpu().numpy()
mlp_out_weight = model.blocks[layer].mlp.W_out.data.cpu().numpy()
print("\nLayer 0 MLP W_in weight matrix:")
print(mlp_in_weight)
print("\nLayer 0 MLP W_out weight matrix:")
print(mlp_out_weight)
```

---

## 10. Conclusion



