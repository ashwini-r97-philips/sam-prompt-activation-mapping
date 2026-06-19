# True Head Importance Analysis

## Objective

The original head importance experiment used the output of the attention module after head mixing and output projection:

```text
[B, T, C]
=
[1, 5184, 256]
```

and then artificially reshaped the feature dimension into:

```text
[B, T, H, D]
=
[1, 5184, 8, 32]
```

to approximate per-head importance.

However, at this stage the individual attention heads have already been mixed together by the output projection layer (`out_proj`). Therefore the resulting scores do not correspond to true attention heads.

To address this, a new experiment was implemented which captures the output of:

```python
torch.nn.functional.scaled_dot_product_attention()
```

directly before head flattening and output projection.

This produces the true attention output tensor:

```text
[B, H, T, D]
=
[1, 8, 5184, 32]
```

where:

```text
B = batch size
H = number of attention heads
T = number of visual tokens
D = per-head feature dimension
```

Gradients were retained on these tensors and true per-head importance was computed using:

```text
grad_x_activation = mean(|gradient × activation|)
```

This allows attribution to actual attention heads rather than projected feature channels.

---

# Verification

The captured tensors had the expected shapes:

## Self-Attention

```text
Q = (1, 8, 5184, 32)
K = (1, 8, 5184, 32)
V = (1, 8, 5184, 32)
OUT = (1, 8, 5184, 32)
```

## Cross-Attention

```text
Q = (1, 8, 5184, 32)
K = (1, 8, 33, 32)
V = (1, 8, 33, 32)
OUT = (1, 8, 5184, 32)
```

This confirms that the attribution is being performed before head mixing and output projection.

---

# School Bus

## Top Heads

| Rank | Head                     | Importance |
| ---- | ------------------------ | ---------- |
| 1    | Layer 5 self_attn Head 6 | 2.02e-07   |
| 2    | Layer 3 self_attn Head 1 | 1.82e-07   |
| 3    | Layer 1 self_attn Head 1 | 1.77e-07   |
| 4    | Layer 4 self_attn Head 0 | 1.74e-07   |
| 5    | Layer 3 self_attn Head 7 | 1.52e-07   |

### Layer-Level Importance

#### Self-Attention

```text
Layer 0 = 1.48e-07
Layer 1 = 7.42e-07
Layer 2 = 5.68e-07
Layer 3 = 8.33e-07
Layer 4 = 6.22e-07
Layer 5 = 5.56e-07
```

#### Cross-Attention

```text
Layer 0 = 5.43e-07
Layer 1 = 3.25e-07
Layer 2 = 1.57e-07
Layer 3 = 9.34e-08
Layer 4 = 5.41e-08
Layer 5 = 4.49e-08
```

### Interpretation

Cross-attention importance is strongest at Layer 0 and steadily decreases through the network.

Self-attention importance becomes dominant after prompt information enters the visual stream.

The most important head is:

```text
Layer 5 self_attn Head 6
```

suggesting that final mask formation depends heavily on late self-attention processing rather than direct prompt injection.

---

# Cat

## Top Heads

| Rank | Head                     | Importance |
| ---- | ------------------------ | ---------- |
| 1    | Layer 5 self_attn Head 6 | 4.33e-08   |
| 2    | Layer 4 self_attn Head 0 | 3.37e-08   |
| 3    | Layer 3 self_attn Head 1 | 3.09e-08   |
| 4    | Layer 1 self_attn Head 1 | 3.04e-08   |
| 5    | Layer 1 self_attn Head 6 | 2.69e-08   |

### Interpretation

The cat example shows the same pattern observed in the school bus example.

The strongest importance occurs in self-attention heads rather than cross-attention heads.

The dominant head is again:

```text
Layer 5 self_attn Head 6
```

indicating that late-layer self-attention plays a major role in forming the final mask-relevant representation.

---

# Dog

## Top Heads

| Rank | Head                     | Importance |
| ---- | ------------------------ | ---------- |
| 1    | Layer 5 self_attn Head 6 | 5.65e-08   |
| 2    | Layer 4 self_attn Head 0 | 4.37e-08   |
| 3    | Layer 3 self_attn Head 1 | 4.20e-08   |
| 4    | Layer 1 self_attn Head 1 | 3.57e-08   |
| 5    | Layer 1 self_attn Head 2 | 3.44e-08   |

### Interpretation

The dog example is highly consistent with the cat example.

The same set of self-attention heads dominate the ranking.

The strongest head remains:

```text
Layer 5 self_attn Head 6
```

suggesting that this head may be consistently involved in generating the final segmentation-relevant representation.

---

# Cross-Example Analysis

A strong pattern emerges across all three examples:

| Example    | Strongest Head           |
| ---------- | ------------------------ |
| School Bus | Layer 5 self_attn Head 6 |
| Cat        | Layer 5 self_attn Head 6 |
| Dog        | Layer 5 self_attn Head 6 |

Additional recurring heads include:

```text
Layer 4 self_attn Head 0
Layer 3 self_attn Head 1
Layer 1 self_attn Head 1
```

The consistency across multiple prompts and images suggests that these heads may play specialized roles in transforming prompt-conditioned visual features into mask-relevant representations.

---

# Relationship to Previous Experiments

The true head attribution results align closely with the previous DeltaF and Gradient-weighted DeltaF experiments.

### DeltaF

Earlier experiments showed:

```text
Prompt-induced feature changes increase through the Multimodal Decoder.
```

with the largest feature changes occurring in:

```text
Layers 4–5
```

### Gradient-weighted DeltaF

Earlier experiments also showed:

```text
Layers 4–5 contain the strongest prompt-conditioned features that influence the final mask.
```

### True Head Attribution

The strongest heads are now found in:

```text
Layer 5 self-attention
Layer 4 self-attention
Layer 3 self-attention
```

These findings are mutually consistent.

---

# Final Interpretation

The results support the following mechanism inside the Multimodal Decoder:

```text
Early Layers
    ↓
Cross-attention injects prompt information into image tokens
    ↓
Middle Layers
    ↓
Self-attention propagates and refines prompt-conditioned features
    ↓
Late Layers
    ↓
Self-attention heads form mask-relevant representations
    ↓
Final Segmentation Mask
```

The strongest overall conclusion is:

> Prompt information enters through early cross-attention layers, but the final segmentation mask depends most heavily on a small set of self-attention heads in the middle and late Multimodal Decoder layers. The most consistently important head across all tested examples is Layer 5 self_attn Head 6.
