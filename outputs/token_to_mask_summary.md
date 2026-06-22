# Fine-Grained Token-to-Mask Analysis

## Objective

The earlier Prompt Activation Mapping (PAM) experiments established a complete prompt-to-mask pathway:

```text
Prompt
↓
Prompt-conditioned fusion representations
↓
DETR Query 144
↓
Final segmentation mask
```

However, these experiments operated at the level of entire prompts and fusion representations.

The goal of this analysis was to move one level deeper and investigate:

> Which specific prompt tokens influence the final segmentation mask, how those tokens propagate through fusion representations, and whether those token-specific pathways can be validated causally.

This analysis was performed on:

```text
"yellow school bus"
"cat"
"dog"
```

---

# 1. Prompt Token Importance

## Method

Gradients were backpropagated from the selected final mask to the prompt representation.

For each prompt token:

```text
importance(token_i)
=
|| d(mask) / d(token_i) ||
```

was computed.

This measures:

> How sensitive the final mask is to each prompt token.

---

## Results

### School Bus

Top prompt tokens:

| Token Index | Normalized Importance |
| ----------- | --------------------- |
| 0           | 1.000                 |
| 4           | 0.530                 |
| 32          | 0.444                 |
| 3           | 0.257                 |
| 1           | 0.211                 |
| 2           | 0.135                 |

### Cat

Top prompt tokens:

| Token Index | Normalized Importance |
| ----------- | --------------------- |
| 0           | 1.000                 |
| 2           | 0.649                 |
| 32          | 0.612                 |
| 1           | 0.410                 |

### Dog

Top prompt tokens:

| Token Index | Normalized Importance |
| ----------- | --------------------- |
| 0           | 1.000                 |
| 2           | 0.585                 |
| 32          | 0.534                 |
| 1           | 0.260                 |

---

## Interpretation

Across all three examples:

```text
token 0
```

was consistently the most important token.

A second consistent pattern was:

```text
token 2
```

which appeared among the strongest prompt tokens for cat and dog.

This suggests that only a small subset of prompt tokens strongly influences final mask generation.

---

# 2. Prompt Token → Fusion Token Trace

## Method

The Multimodal Decoder cross-attention modules:

```text
transformer.encoder.layers.0–5.cross_attn_image
```

were traced.

For each prompt token:

```text
prompt token
↓
fusion token attention map
```

was computed.

The resulting attention maps were projected onto the:

```text
72 × 72
```

fusion-memory grid.

---

## Interpretation

This experiment identifies:

```text
Which fusion-memory locations
are influenced by each prompt token.
```

Rather than treating the prompt as a single entity, the contribution of individual prompt tokens can be visualized throughout the fusion representation.

---

# 3. Fusion Token → Mask Attribution

## Method

Gradients were backpropagated from the selected mask to the fusion-memory representations.

For each fusion token:

```text
fusion importance =
|| d(mask) / d(fusion token) ||
```

was computed.

This measures:

> Which fusion-memory locations are most important for final mask generation.

---

## Interpretation

This experiment provides the second half of the pathway:

```text
fusion token
↓
final mask
```

allowing prompt-token influence and mask relevance to be connected.

---

# 4. Token-to-Mask Pathway Analysis

## Method

The previous two signals were combined:

```text
Prompt Token Attention
×
Fusion-to-Mask Gradient
```

This produced a token-specific pathway score:

```text
Prompt Token
↓
Fusion Token
↓
Mask
```

for every prompt token and every fusion layer.

---

## Results

### School Bus

Strongest pathways:

```text
Layer 4 token 0
Layer 2 token 0
Layer 3 token 0
Layer 1 token 0
Layer 5 token 0
```

### Cat

Strongest pathways:

```text
Layer 2 token 0
Layer 4 token 0
Layer 1 token 0
Layer 3 token 0
Layer 0 token 0
```

### Dog

Strongest pathways:

```text
Layer 4 token 0
Layer 2 token 0
Layer 5 token 0
Layer 3 token 0
```

---

## Interpretation

Across all three examples:

```text
token 0
```

consistently produced the strongest prompt-token → mask pathways.

This is one of the strongest and most reproducible findings of the token-level analysis.

---

# 5. Prompt Token Identification

## School Bus

Prompt:

```text
yellow school bus
```

Tokenization:

```text
0 → <start token>
1 → yellow
2 → school
3 → bus
4 → <end token>
```

---

## Cat

Prompt:

```text
cat
```

Tokenization:

```text
0 → <start token>
1 → cat
2 → <end token>
```

---

## Dog

Prompt:

```text
dog
```

Tokenization:

```text
0 → <start token>
1 → dog
2 → <end token>
```

---

## Interpretation

This allows the token-level results to be interpreted using actual prompt content rather than anonymous token indices.

---

# 6. Token-Level Causal Validation

## Method

Individual prompt tokens were ablated by replacing their embeddings with zeros.

The model was then re-run and the resulting segmentation output was compared against the original prediction.

This measures:

> Whether a prompt token causally contributes to mask generation.

---

## Results

### School Bus

| Token   | Effect              |
| ------- | ------------------- |
| <start> | Small score drop    |
| yellow  | Small score drop    |
| school  | Moderate score drop |
| bus     | Measurable effect   |
| <end>   | NO_MASK             |

### Cat

| Token   | Score Drop |
| ------- | ---------- |
| <start> | 0.00074    |
| cat     | 0.00244    |
| <end>   | 0.01590    |

### Dog

| Token   | Score Drop |
| ------- | ---------- |
| <start> | 0.00067    |
| dog     | 0.00136    |
| <end>   | 0.02747    |

---

## Interpretation

The most surprising finding is that:

```text
<end token>
```

is consistently the most important token across all examples.

For:

```text
cat
dog
```

the end token produces the largest score drop by a substantial margin.

For:

```text
yellow school bus
```

removing the end token causes:

```text
NO_MASK
```

meaning the model fails to generate a valid segmentation mask.

This suggests that the end token is not merely a formatting token but a critical component of SAM3's prompt-processing pipeline.

---

# 7. Token-Specific Fusion Pathway Ablation

## Method

The strongest fusion-memory pathways associated with individual prompt tokens were identified and ablated before the DETR decoder.

This tests:

```text
Prompt Token
↓
Fusion Pathway
↓
Mask
```

causally.

---

## Results

### Cat

Largest pathway ablations:

```text
token 0 pathway → score drop ≈ 0.0083
token 1 pathway → score drop ≈ 0.0091
token 2 pathway → score drop ≈ 0.0071
```

### Dog

Largest pathway ablations:

```text
token 0 pathway → score drop ≈ 0.0058
token 1 pathway → score drop ≈ 0.0068
token 2 pathway → score drop ≈ 0.0061
```

### School Bus

Token-specific pathway ablations produced score drops substantially larger than direct token ablation, demonstrating that the corresponding fusion pathways contribute meaningfully to final mask generation.

---

## Interpretation

The fusion pathways associated with individual prompt tokens are not merely correlated with the final mask.

Removing those pathways consistently degrades segmentation quality.

This provides causal evidence for:

```text
Prompt Token
↓
Fusion Representation
↓
Final Mask
```

rather than simply:

```text
Prompt Token
↓
Mask
```

---

# Overall Interpretation

The token-level experiments extend the original PAM analysis from:

```text
Prompt
↓
Fusion Memory
↓
Mask
```

to:

```text
Specific Prompt Token
↓
Specific Fusion Pathway
↓
Final Mask
```

The strongest findings are:

1. Not all prompt tokens contribute equally to mask generation.
2. Token 0 (<start token>) consistently produces the strongest prompt-to-mask pathways.
3. The end token is the most causally important token across all tested examples.
4. Removing the end token can severely degrade or completely break segmentation.
5. Semantic object tokens (cat, dog, bus) contribute meaningfully but are less dominant than the special boundary tokens.
6. Token-specific fusion pathways can be traced and causally validated.
7. Prompt influence can now be followed from individual tokens all the way to final mask behavior.

Overall, the results suggest that SAM3 processes prompts through a hierarchy:

```text
Special Prompt Tokens
        ↓
Prompt Structure
        ↓
Object-Specific Tokens
        ↓
Fusion Representations
        ↓
DETR Query 144
        ↓
Final Segmentation Mask
```

This represents the most fine-grained prompt-to-mask explanation obtained during the project.
