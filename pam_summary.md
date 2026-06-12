# Prompt Activation Mapping (PAM) – Multimodal Decoder Analysis

## Objective

The goal of this work was to understand how text prompts influence image representations inside the SAM3 Multimodal Decoder and determine which prompt-induced changes are most relevant to the final segmentation mask.

The analysis was restricted to:

```text
transformer.encoder.layers.0 – transformer.encoder.layers.5
```

which correspond to the Multimodal Decoder block in SAM3.

Experiments were performed on three examples:

* Dog
* Cat
* Yellow School Bus

---

# 1. DeltaF Analysis (Prompt-Induced Feature Change)

## Method

For each image, SAM3 was run twice:

**Run A**

```text
Image + Real Prompt
```

**Run B**

```text
Image + Empty Prompt
```

For every Multimodal Decoder layer, the image-token representation was captured.

Layer output shape:

```text
(1, 5184, 256)
```

which corresponds to:

```text
72 × 72 visual token grid
256-dimensional feature vector per token
```

For each layer:

```text
DeltaF = F_prompt − F_baseline
```

and

```text
DeltaF map = ||DeltaF||
```

was computed.

DeltaF therefore measures:

> How much the text prompt changes the image representation.

---

## Results

### Dog

| Layer | Mean DeltaF |
| ----- | ----------- |
| 0     | 3.12        |
| 1     | 5.97        |
| 2     | 7.55        |
| 3     | 11.02       |
| 4     | 16.47       |
| 5     | 24.33       |

### Cat

| Layer | Mean DeltaF |
| ----- | ----------- |
| 0     | 2.70        |
| 1     | 5.98        |
| 2     | 8.41        |
| 3     | 12.54       |
| 4     | 19.96       |
| 5     | 29.41       |

### School Bus

| Layer | Mean DeltaF |
| ----- | ----------- |
| 0     | 3.46        |
| 1     | 7.03        |
| 2     | 8.79        |
| 3     | 13.58       |
| 4     | 19.56       |
| 5     | 27.46       |

---

## Interpretation

All three examples show the same trend:

```text
Layer 0 < Layer 1 < Layer 2 < Layer 3 < Layer 4 < Layer 5
```

This indicates that prompt-induced feature changes accumulate progressively throughout the Multimodal Decoder.

The strongest prompt-conditioned image representations consistently occur in the deepest layers.

This suggests that SAM3 does not simply use the prompt as a retrieval signal. Instead, prompt information is repeatedly integrated into the visual representation as it passes through the Multimodal Decoder.

---

# 2. Q/K/V Trace Analysis

## Method

For every Multimodal Decoder layer, the following attention modules were inspected:

```text
self_attn
cross_attn_image
```

The following rule was used:

```text
Q = stream being updated
K/V = stream being read from
Output length = Q length
```

---

## Results

### Self-Attention

For all layers:

```text
Q = (1, 5184, 256)
K = (1, 5184, 256)
V = (1, 5184, 256)
Output = (1, 5184, 256)
```

Interpretation:

```text
Image tokens read from image tokens.
```

---

### Cross-Attention

For all layers:

```text
Q = (1, 5184, 256)
K = (1, 33, 256)
V = (1, 33, 256)
Output = (1, 5184, 256)
```

Interpretation:

```text
Image tokens read from prompt tokens.
```

---

## Interpretation

The visual grid is directly updated using prompt information at every Multimodal Decoder layer.

This confirms that the DeltaF experiment is genuinely measuring prompt-driven modifications to image features.

---

# 3. Gradient-Weighted DeltaF

## Method

DeltaF answers:

```text
Where did the prompt change image features?
```

Gradients answer:

```text
Which image features affected the final mask?
```

These two signals were combined:

```text
PAM = DeltaF × Gradient
```

The gradient target was:

```text
Mean of the selected mask logits
```

This experiment therefore measures:

> Which prompt-induced feature changes actually contribute to the final segmentation mask.

---

## Results

### Dog

| Layer | Mean DeltaF × Grad |
| ----- | ------------------ |
| 0     | 8.18e-06           |
| 1     | 1.42e-05           |
| 2     | 1.61e-05           |
| 3     | 1.66e-05           |
| 4     | 1.90e-05           |
| 5     | 2.12e-05           |

Peak:

```text
Layer 5
```

---

### Cat

| Layer | Mean DeltaF × Grad |
| ----- | ------------------ |
| 0     | 5.85e-06           |
| 1     | 1.20e-05           |
| 2     | 1.48e-05           |
| 3     | 1.54e-05           |
| 4     | 1.77e-05           |
| 5     | 1.78e-05           |

Peak:

```text
Layer 5
```

---

### School Bus

| Layer | Mean DeltaF × Grad |
| ----- | ------------------ |
| 0     | 4.65e-05           |
| 1     | 8.02e-05           |
| 2     | 8.83e-05           |
| 3     | 9.02e-05           |
| 4     | 9.40e-05           |
| 5     | 8.66e-05           |

Peak:

```text
Layer 4
```

---

## Interpretation

Across all examples, the strongest prompt-conditioned features that also influence the final mask appear in:

```text
Layers 4–5
```

This suggests that while prompt information is injected early, the final segmentation decision depends most strongly on prompt-conditioned representations formed in the deeper Multimodal Decoder layers.

---

# 4. Head Importance Analysis

## Method

Gradient-based head importance was computed for:

```text
transformer.encoder.layers.0–5.self_attn
transformer.encoder.layers.0–5.cross_attn_image
```

The importance score used was:

```text
grad_x_activation
```

Important note:

This is currently a gradient-based **head-slot importance proxy**, not exact pre-output-projection head attribution.

---

## School Bus

### Top Heads

| Rank | Head                            |
| ---- | ------------------------------- |
| 1    | Layer 0 cross_attn_image Head 7 |
| 2    | Layer 1 self_attn Head 0        |
| 3    | Layer 0 cross_attn_image Head 2 |
| 4    | Layer 1 self_attn Head 7        |
| 5    | Layer 0 cross_attn_image Head 5 |

### Layer Importance

| Module                   | Importance |
| ------------------------ | ---------- |
| Layer 0 cross_attn_image | 8.97e-07   |
| Layer 1 self_attn        | 8.25e-07   |
| Layer 3 self_attn        | 7.51e-07   |

### Interpretation

For the school bus example, prompt-to-image fusion is extremely important.

Several Layer 0 cross-attention heads dominate the ranking, suggesting that prompt information is injected strongly into the visual grid very early.

---

## Cat

### Top Heads

| Rank | Head                     |
| ---- | ------------------------ |
| 1    | Layer 1 self_attn Head 7 |
| 2    | Layer 1 self_attn Head 0 |
| 3    | Layer 1 self_attn Head 4 |
| 4    | Layer 1 self_attn Head 1 |
| 5    | Layer 3 self_attn Head 6 |

### Layer Importance

| Module            | Importance |
| ----------------- | ---------- |
| Layer 1 self_attn | 1.65e-07   |
| Layer 3 self_attn | 1.43e-07   |
| Layer 2 self_attn | 1.36e-07   |

### Interpretation

For the cat example, Layer 1 self-attention is the dominant mechanism.

Prompt information appears to be injected first and then propagated through the visual grid using self-attention.

---

## Dog

### Top Heads

| Rank | Head                            |
| ---- | ------------------------------- |
| 1    | Layer 1 self_attn Head 7        |
| 2    | Layer 1 self_attn Head 0        |
| 3    | Layer 1 self_attn Head 4        |
| 4    | Layer 3 self_attn Head 6        |
| 5    | Layer 0 cross_attn_image Head 7 |

### Layer Importance

| Module            | Importance |
| ----------------- | ---------- |
| Layer 1 self_attn | 2.25e-07   |
| Layer 3 self_attn | 1.99e-07   |
| Layer 5 self_attn | 1.88e-07   |

### Interpretation

The dog result is very similar to the cat result.

The same Layer 1 self-attention heads dominate the ranking, suggesting that animal-related prompts may rely on a common set of self-attention heads after prompt information has entered the visual stream.

---

# Overall Interpretation

Taken together, the experiments suggest the following processing pipeline inside the SAM3 Multimodal Decoder:

```text
Layer 0 cross_attn_image
        ↓
Prompt information enters image tokens
        ↓
Layers 1–3 self_attn
        ↓
Prompt information is propagated and refined
        ↓
Layers 4–5
        ↓
Prompt-conditioned representation becomes most relevant to final mask prediction
```

The strongest findings are:

1. Prompt information directly modifies image-token representations.
2. Prompt-induced feature changes accumulate throughout the Multimodal Decoder.
3. The most mask-relevant prompt-conditioned representations occur in Layers 4–5.
4. Layer 0 cross-attention appears responsible for prompt injection.
5. Layer 1 self-attention appears responsible for propagating prompt information through the visual grid.
6. Similar head patterns appear across dog and cat examples, while school bus relies more heavily on direct prompt-to-image fusion.

Overall, the results support a two-stage mechanism:

```text
Prompt Injection
        ↓
Prompt Propagation
        ↓
Mask-Relevant Prompt-Conditioned Representation
```

inside the SAM3 Multimodal Decoder.
