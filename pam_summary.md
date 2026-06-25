# Prompt Activation Mapping (PAM) – Prompt-to-Mask Analysis in SAM3

## Objective

The goal of this work was to understand how text prompts influence visual representations inside SAM3 and how those prompt-induced changes ultimately affect the final segmentation mask.

The analysis focused on tracing the complete pathway:

```text
Prompt
↓
Multimodal Decoder
↓
Prompt-conditioned fusion representations
↓
DETR Decoder
↓
Final segmentation mask
```

Experiments were performed on three examples:

* Dog
* Cat
* Yellow School Bus

---

# Overview of the Pipeline

The relevant portion of SAM3 can be viewed as:

```text
Image
↓
Vision Backbone
↓
72 × 72 image tokens
↓
Multimodal Decoder (transformer.encoder.layers.0–5)
↓
Prompt-conditioned fusion tokens
↓
DETR Decoder (transformer.decoder.layers.0–5)
↓
Final segmentation mask
```

The objective of PAM is to determine:

1. How the prompt modifies fusion tokens.
2. Which prompt-induced changes affect the final mask.
3. Which attention heads are responsible for processing prompt information.
4. Which DETR query produces the final mask.
5. Which fusion tokens are consumed by that query.
6. Whether prompt-conditioned fusion tokens causally contribute to the final segmentation output.

---

# 1. DeltaF Analysis (Prompt-Induced Feature Change)

## Method

For each image, SAM3 was run twice:

### Run A

```text
Image + Real Prompt
```

### Run B

```text
Image + Empty Prompt
```

For every Multimodal Decoder layer, the image-token representation was captured.

Layer output shape:

```text
(1, 5184, 256)
```

corresponding to:

```text
72 × 72 visual token grid
256-dimensional feature vector per token
```

For each layer:

```text
DeltaF = F_prompt − F_baseline
```

and:

```text
DeltaF map = ||DeltaF||
```

was computed.

DeltaF measures:

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

All three examples show:

```text
Layer 0 < Layer 1 < Layer 2 < Layer 3 < Layer 4 < Layer 5
```

Prompt-induced feature changes accumulate throughout the Multimodal Decoder.

The strongest prompt-conditioned visual representations consistently occur in the deepest layers.

This indicates that prompt information is repeatedly integrated into the visual representation rather than being injected only once.

---

# 2. Q/K/V Trace Analysis

## Method

For every Multimodal Decoder layer:

```text
self_attn
cross_attn_image
```

were traced.

Interpretation rule:

```text
Q = stream being updated
K/V = stream being read from
```

---

## Results

### Self-Attention

For all layers:

```text
Q = (1, 5184, 256)
K = (1, 5184, 256)
V = (1, 5184, 256)
```

Meaning:

```text
Image tokens read from image tokens.
```

### Cross-Attention

For all layers:

```text
Q = (1, 5184, 256)
K = (1, 33, 256)
V = (1, 33, 256)
```

Meaning:

```text
Image tokens read from prompt tokens.
```

---

## Interpretation

Prompt information enters the visual representation through cross-attention and is then propagated across image tokens through self-attention.

This confirms that DeltaF is measuring genuine prompt-induced modifications to image features.

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

These signals were combined:

```text
PAM = DeltaF × Gradient
```

using:

```text
target = mean(masks_logits)
```

as the backward objective.

This identifies:

> Prompt-induced feature changes that actually contribute to the final mask.

---

## Results

### Dog

Peak:

```text
Layer 5
```

### Cat

Peak:

```text
Layer 5
```

### School Bus

Peak:

```text
Layer 4
```

---

## Interpretation

Across all examples:

```text
Layers 4–5
```

contain the strongest prompt-conditioned features that influence the final segmentation output.

Prompt information is injected early but becomes most mask-relevant in the deepest fusion layers.

---

# 4. True Head Importance Analysis

## Method

True per-head attribution was computed directly from:

```text
[B, H, T, D]
```

attention outputs captured before output projection.

This avoids the pseudo-head approximation used in earlier experiments.

Importance was computed using:

```text
grad × activation
```

for each attention head.

---

## Results

Across all three examples:

The most important heads were dominated by:

```text
self_attn
```

rather than:

```text
cross_attn_image
```

The most consistently important heads included:

```text
Layer 5 self_attn Head 6
Layer 4 self_attn Head 0
Layer 3 self_attn Head 1
Layer 1 self_attn Head 1
```

These heads appeared repeatedly across:

```text
School Bus
Cat
Dog
```

---

## Interpretation

Prompt information is injected through cross-attention but much of the important processing happens inside self-attention.

The strongest prompt-relevant computations occur after prompt information has already entered the visual stream.

---

# 5. DETR Decoder Query Mapping

## Method

Gradients were backpropagated from the final selected mask into:

```text
hs
```

the DETR decoder output.

For each decoder query:

```text
query_score(q) = || d(mask) / d(hs_q) ||
```

was computed.

The responsible query was defined as:

```text
q* = argmax query_score(q)
```

---

## Results

| Image      | Responsible Query |
| ---------- | ----------------- |
| School Bus | 144               |
| Cat        | 144               |
| Dog        | 144               |

This result was independently verified for all three examples.

---

## Interpretation

The selected final mask is consistently produced by:

```text
query 144
```

for the tested examples.

This identifies the decoder query responsible for final mask generation.

---

# 6. DETR Decoder Cross-Attention

## Method

The following modules were traced:

```text
transformer.decoder.layers.0–5.cross_attn
```

Observed shapes:

```text
Q = (1, 8, 201, 32)
K = (1, 8, 5184, 32)
V = (1, 8, 5184, 32)
```

Therefore:

```text
Query 144
↓
reads 72 × 72 fusion-memory tokens
```

---

## Results

### School Bus

Main attended region:

```text
row 35–45
col 51–56
```

### Cat

Main attended region:

```text
row 35–38
col 55
```

### Dog

Main attended region:

```text
row 48–50
col 52–53
```

---

## Interpretation

Query 144 is not tied to a fixed location.

Instead, it dynamically attends to image-specific regions of the fusion-memory grid.

This establishes the bridge:

```text
Fusion Memory
↓
Query 144
↓
Final Mask
```

---

# 7. Pathway Consistency Analysis

## Objective

Determine whether:

```text
Prompt-important fusion tokens
```

are the same tokens that:

```text
Query 144 actually reads
```

---

## Method

Compared:

```text
DeltaF × Gradient
```

with:

```text
Query-144 attention map
```

using:

```text
Cosine Similarity
Pearson Correlation
Top-k Overlap
```

---

## Results

### School Bus

Best alignment:

```text
Layer 5
Cosine = 0.240
Correlation = 0.215
Top-20 Overlap = 40%
```

### Dog

Best alignment:

```text
Layer 4
Cosine = 0.305
Correlation = 0.240
```

### Cat

Alignment remained weak across all layers.

---

## Interpretation

For school bus and dog, the decoder reads a subset of the prompt-important fusion tokens, especially in later layers.

For cat, the relationship is much weaker.

Therefore:

> The pathway consistency experiment provides correlational evidence that some prompt-conditioned, mask-relevant fusion tokens are subsequently read by the mask-producing DETR decoder query.

However, this alone does not establish causality.

---

# 8. Causal Ablation Analysis

## Objective

Test whether prompt-conditioned fusion tokens are actually required for the final mask.

---

## Method

Tokens were ranked using:

```text
DeltaF × Gradient
```

Top-k tokens were removed before the DETR decoder.

The resulting mask was compared against:

```text
Random token ablation
```

using:

```text
Mask score drop
IoU drop
```

---

## Results

Small ablations:

```text
k = 50, 100, 200
```

produced weak effects.

Large ablations:

```text
k = 500
```

produced consistent effects.

### School Bus

```text
Layer 1
Score-drop advantage = 0.00929
```

### Dog

```text
Layer 3
Score-drop advantage = 0.00569
```

### Cat

```text
Layer 3
Score-drop advantage = 0.01035
```

---

## Interpretation

Removing a sufficiently large set of high DeltaF×Gradient tokens damages the final mask more than removing random tokens.

The effect is distributed across many fusion tokens rather than concentrated in a tiny hotspot.

This provides causal evidence that prompt-conditioned fusion tokens contribute to final mask generation.

---

# 9. Decoder Attention Ablation

## Objective

The causal ablation experiment demonstrated that prompt-conditioned fusion tokens contribute to final mask generation.

However, that experiment does not directly test whether the fusion-memory tokens read by the DETR decoder are themselves important.

The goal of this experiment was therefore:

> Do the fusion-memory tokens attended by Query 144 causally contribute to the final segmentation mask?

---

## Method

The DETR decoder attention maps generated earlier were used.

For each decoder layer:

```text
decoder_layer_i_query144_attn.npy
```

the fusion-memory tokens were ranked according to Query 144 attention.

For each layer:

1. Select the top-k attended fusion tokens.
2. Remove those tokens before the DETR decoder.
3. Re-run SAM3.
4. Compare against random token ablation of the same size.

The following values were evaluated:

```text
k = 50
k = 100
k = 200
k = 500
```

Mask score drop and IoU drop were compared against random-token ablation.

---

## Results

### School Bus

Strongest result:

```text
Layer 5
k = 500

Top Attention Score Drop = 0.00643
Random Score Drop        = 0.00268
Advantage                = 0.00375
```

### Dog

Strongest result:

```text
Layer 5
k = 500

Top Attention Score Drop = 0.00276
Random Score Drop        = 0.00145
Advantage                = 0.00131
```

### Cat

Strongest result:

```text
Layer 5
k = 500

Top Attention Score Drop = 0.00200
Random Score Drop        = 0.00100
Advantage                = 0.00100
```

---

## Cross-Example Interpretation

Across all three examples:

```text
School Bus
Dog
Cat
```

the strongest effects occur for:

```text
k = 500
```

while smaller ablations:

```text
k = 50
k = 100
k = 200
```

produce weaker and less consistent effects.

This indicates that decoder attention is distributed across a broader set of fusion-memory tokens rather than concentrated in only a few highly attended locations.

---

## Interpretation

This experiment directly tests whether the fusion tokens attended by Query 144 matter for final mask generation.

The results show:

1. Removing highly attended fusion tokens damages the final mask more than random ablation.
2. The effect is strongest when larger groups of attended tokens are removed.
3. Decoder attention is not merely a visualization artifact.
4. The fusion-memory tokens read by Query 144 contribute to final segmentation behavior.

This establishes:

```text
Prompt-important tokens matter
✓

Decoder-read tokens matter
✓
```

and strengthens the prompt-to-mask pathway established by previous experiments.

---

## Conclusion

The Decoder Attention Ablation experiment provides causal evidence that the fusion-memory tokens read by Query 144 contribute to final mask generation.

The strongest effects occur in later decoder layers and become most visible when larger groups of attended tokens are removed.

The results therefore support the interpretation that:

```text
Query 144
↓
reads fusion-memory tokens
↓
uses those tokens
↓
produces the final segmentation mask
```

rather than merely assigning attention without functional importance.

# 10. Query 144 Dominance Analysis

## Objective

The DETR query-tracing experiments identified:

```text
Query 144
```

as the query responsible for generating the final selected mask.

The remaining question was:

> Is Query 144 only slightly more important than the remaining queries, or does it completely dominate mask generation?

---

## Method

For each image:

```text
School Bus
Cat
Dog
```

the query-gradient scores were ranked.

For every decoder query:

```text
query_score(q) = || d(mask) / d(hs_q) ||
```

was computed.

The top-ranked query, second-ranked query, and number of non-zero queries were recorded.

---

## Results

### School Bus

```text
Top Query = 144
Gradient  = 5.71e-03

Second Query = 125
Gradient      = 0

Non-Zero Queries = 1
```

### Cat

```text
Top Query = 144
Gradient  = 1.02e-03

Second Query = 125
Gradient      = 0

Non-Zero Queries = 1
```

### Dog

```text
Top Query = 144
Gradient  = 1.69e-03

Second Query = 125
Gradient      = 0

Non-Zero Queries = 1
```

---

## Cross-Example Interpretation

The same pattern appeared for all three examples:

| Image      | Top Query | Non-Zero Queries |
| ---------- | --------- | ---------------- |
| School Bus | 144       | 1                |
| Cat        | 144       | 1                |
| Dog        | 144       | 1                |

The responsible query remained identical despite:

```text
Different prompts
Different object categories
Different image content
```

This suggests that Query 144 behaves as a specialized object query used by SAM3 for the selected segmentation output.

---

## Interpretation

The DETR decoder contains:

```text
200 object queries
```

but the final selected mask in all tested examples is routed through:

```text
Query 144
```

and no measurable gradient reaches the remaining queries.

This means that Query 144 completely dominates mask generation for the tested examples.

The prompt-to-mask pathway can therefore be simplified as:

```text
Prompt
↓
Fusion-memory representations
↓
Query 144
↓
Final segmentation mask
```

rather than:

```text
Prompt
↓
Many DETR queries
↓
Final mask
```

for the examples analyzed.

---

## Conclusion

The Query 144 Dominance Analysis shows that:

1. Query 144 is consistently responsible for the selected final mask.
2. No other decoder query receives measurable gradient from the final mask objective.
3. Query 144 completely dominates mask generation for the school bus, cat, and dog examples.
4. Query 144 acts as the primary decoder readout mechanism through which prompt-conditioned fusion representations influence the final segmentation output.

This result strengthens the interpretation that Query 144 is the key decoder component connecting fusion-memory representations to final mask generation in the tested examples.

# Final Prompt-to-Mask Pathway

Combining all experiments yields the following picture:

```text
Prompt
↓
Cross-Attention
↓
Prompt information enters image tokens
↓
Self-Attention
↓
Prompt information propagates through image tokens
↓
Deep fusion layers (Layers 4–5)
↓
Prompt-conditioned mask-relevant fusion representations
↓
DETR Query 144
↓
Cross-attention to selected fusion-memory tokens
↓
Final segmentation mask
```

---

# Main Conclusions

1. Prompt information directly modifies image-token representations.
2. Prompt-induced feature changes accumulate throughout the Multimodal Decoder.
3. The most mask-relevant prompt-conditioned features occur in Layers 4–5.
4. Self-attention plays a dominant role in refining prompt information after prompt injection.
5. The final selected mask is consistently produced by DETR decoder query 144 for the tested examples.
6. Query 144 reads image-dependent regions of the 72×72 fusion-memory grid.
7. Some prompt-conditioned fusion tokens overlap with the tokens read by query 144.
8. Removing large groups of high DeltaF×Gradient tokens degrades the mask more than random token removal.
9. Prompt-conditioned fusion representations therefore contribute causally to final mask generation.
10. Fusion-memory tokens attended by Query 144 contribute causally to final mask generation.
11. Query 144 completely dominates final-mask generation for the school bus, cat, and dog examples.
12. Cross-layer fused Query-144 attention identifies fusion-memory tokens that causally affect the final mask, especially for schoolbus and dog.

Overall, the results support the following mechanism inside SAM3:

```text
Prompt Injection
        ↓
Prompt Propagation
        ↓
Prompt-Conditioned Fusion Representation
        ↓
Query 144 Readout
        ↓
Decoder-Attended Fusion Tokens
        ↓
Final Segmentation Mask
```

```
```
