# Pathway Consistency Analysis

## Objective

The Multimodal Decoder experiments established that text prompts modify fusion-memory representations and that some of these prompt-induced changes are relevant to the final segmentation mask.

The DETR decoder experiments established that the final selected mask is produced by decoder query 144 and that query 144 reads from specific regions of the 72×72 fusion-memory grid.

The remaining question was:

> Are the prompt-changed, mask-relevant fusion tokens the same tokens that are actually read by the DETR decoder query that produces the final mask?

This experiment was designed to connect the Multimodal Decoder and DETR Decoder analyses into a single prompt-to-mask pathway.

---

# Method

Two spatial maps were compared.

## Prompt-to-Mask Importance Map

From the earlier Gradient-Weighted DeltaF experiment:

```text
P = DeltaF × Gradient
```

where:

```text
DeltaF
=
Prompt-induced change in fusion features

Gradient
=
Sensitivity of the final mask to those features
```

This produces a 72×72 map identifying fusion tokens that:

```text
Changed because of the prompt
and
Influenced the final mask
```

---

## DETR Decoder Readout Map

From the DETR decoder cross-attention experiment:

```text
A = Attention(query 144 → fusion memory)
```

This produces a 72×72 map identifying fusion tokens that are read by the mask-producing decoder query.

---

## Comparison Metrics

The following metrics were computed for every layer:

### Cosine Similarity

Measures overall similarity between the two spatial maps.

Higher values indicate stronger agreement.

### Pearson Correlation

Measures whether high values in one map correspond to high values in the other map.

Higher values indicate stronger spatial correspondence.

### Top-k Overlap

Measures overlap between the most important spatial locations in both maps.

For example:

```text
top20 overlap = 0.40
```

means:

```text
40% of the top 20 tokens are shared
```

between the two maps.

---

# Results

## School Bus

### Layer-Level Results

| Layer | Cosine | Correlation | Top-20 Overlap | Top-100 Overlap |
| ----- | ------ | ----------- | -------------- | --------------- |
| 0     | 0.014  | 0.001       | 0.00           | 0.00            |
| 1     | 0.074  | 0.032       | 0.00           | 0.07            |
| 2     | 0.067  | 0.025       | 0.00           | 0.05            |
| 3     | 0.117  | 0.084       | 0.00           | 0.09            |
| 4     | 0.170  | 0.142       | 0.10           | 0.21            |
| 5     | 0.240  | 0.215       | 0.40           | 0.18            |

### Interpretation

The strongest alignment occurs in the deepest layers.

Layer 5 shows:

```text
Cosine Similarity = 0.240
Correlation = 0.215
Top-20 Overlap = 40%
```

This suggests that a meaningful subset of the prompt-important fusion tokens are also being read by the mask-producing decoder query.

---

## Dog

### Layer-Level Results

| Layer | Cosine | Correlation | Top-20 Overlap | Top-100 Overlap |
| ----- | ------ | ----------- | -------------- | --------------- |
| 0     | 0.021  | -0.019      | 0.00           | 0.00            |
| 1     | 0.136  | 0.055       | 0.05           | 0.03            |
| 2     | 0.170  | 0.067       | 0.05           | 0.05            |
| 3     | 0.241  | 0.158       | 0.10           | 0.10            |
| 4     | 0.305  | 0.240       | 0.10           | 0.12            |
| 5     | 0.242  | 0.189       | 0.05           | 0.11            |

### Interpretation

The strongest alignment occurs in Layer 4:

```text
Cosine Similarity = 0.305
Correlation = 0.240
```

This is the strongest pathway-consistency result obtained across all examples.

The results suggest that prompt-important fusion tokens and decoder-read tokens overlap most strongly in the middle-to-late fusion layers.

---

## Cat

### Layer-Level Results

| Layer | Cosine | Correlation | Top-20 Overlap | Top-100 Overlap |
| ----- | ------ | ----------- | -------------- | --------------- |
| 0     | 0.011  | -0.001      | 0.00           | 0.00            |
| 1     | 0.015  | -0.007      | 0.00           | 0.00            |
| 2     | 0.011  | -0.016      | 0.00           | 0.00            |
| 3     | 0.007  | -0.011      | 0.00           | 0.00            |
| 4     | 0.008  | -0.011      | 0.00           | 0.00            |
| 5     | 0.018  | -0.002      | 0.00           | 0.01            |

### Interpretation

Unlike the dog and school bus examples, the cat example shows very weak alignment.

This suggests that the relationship between prompt-conditioned importance and decoder attention is not identical across all examples.

---

# Overall Interpretation

This experiment was intended to answer:

> Are the fusion tokens that are modified by the prompt and important to the mask the same tokens that are read by the DETR decoder?

The results provide moderate evidence that this is true for some examples.

The strongest evidence comes from:

```text
Dog
School Bus
```

where alignment increases in later layers.

The results are weaker for:

```text
Cat
```

indicating that the relationship is not universally strong.

Therefore, the correct interpretation is:

> The pathway consistency experiment provides correlational evidence that some prompt-conditioned, mask-relevant fusion tokens are subsequently read by the mask-producing DETR decoder query, particularly in later fusion layers. However, the relationship is not strong enough across all examples to claim that decoder attention and prompt-conditioned importance are identical signals.

This experiment provides supportive evidence for a prompt-to-mask pathway, but does not by itself establish causality.
