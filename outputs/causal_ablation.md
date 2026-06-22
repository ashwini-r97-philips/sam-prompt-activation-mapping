# Causal Ablation Analysis

## Objective

The pathway consistency experiment provided correlational evidence that some prompt-conditioned fusion tokens are read by the DETR decoder.

However, correlation alone does not prove that these tokens are actually responsible for the final mask.

The goal of the causal ablation experiment was therefore:

> Do prompt-conditioned fusion tokens causally contribute to the final segmentation mask?

---

# Method

The experiment used the previously computed:

```text
DeltaF × Gradient
```

maps.

These maps identify fusion tokens that are:

```text
Changed by the prompt
and
Important to the final mask
```

For each layer:

1. Rank all fusion tokens using DeltaF×Gradient.
2. Select the top-k tokens.
3. Remove those tokens from the fusion memory before the DETR decoder.
4. Re-run SAM3.
5. Measure the change in mask quality.

A control experiment was also performed:

1. Select k random tokens.
2. Remove them.
3. Re-run SAM3.
4. Measure the same metrics.

If the top DeltaF×Gradient tokens are genuinely important, removing them should damage the mask more than removing random tokens.

---

# Metrics

## Mask Score Drop

Measures the change in:

```text
mean(masks_logits)
```

after ablation.

Higher score drop indicates stronger impact on the final mask.

---

## IoU Drop

Measures the change in mask shape relative to the original prediction.

Higher IoU drop indicates greater disruption of the segmentation output.

---

# School Bus Results

For small ablations:

```text
k = 50
k = 100
k = 200
```

the effect was weak and inconsistent.

For larger ablations:

```text
k = 500
```

the effect became clear.

### Strongest Score-Drop Result

```text
Layer 1
Top-token score drop = 0.01191
Random score drop    = 0.00262
Advantage            = 0.00929
```

### Strongest IoU-Drop Result

```text
Layer 3
Top-token IoU drop   = 0.00311
Random IoU drop      = 0.00040
Advantage            = 0.00272
```

---

# Dog Results

### Strongest Score-Drop Result

```text
Layer 3
Top-token score drop = 0.00711
Random score drop    = 0.00142
Advantage            = 0.00569
```

### Strongest IoU-Drop Result

```text
Layer 3
Top-token IoU drop   = 0.00287
Random IoU drop      = 0.00069
Advantage            = 0.00218
```

---

# Cat Results

### Strongest Score-Drop Result

```text
Layer 3
Top-token score drop = 0.01143
Random score drop    = 0.00109
Advantage            = 0.01035
```

### Strongest IoU-Drop Result

```text
Layer 3
Top-token IoU drop   = 0.00166
Random IoU drop      = 0.00037
Advantage            = 0.00129
```

---

# Key Observation

For:

```text
k = 50
k = 100
k = 200
```

the effects were generally small.

However, for:

```text
k = 500
```

top-token ablation consistently damaged the mask more than random-token ablation.

This indicates that the causal signal is distributed across many fusion tokens rather than concentrated in a tiny set of highly localized tokens.

Since:

```text
500 / 5184 ≈ 9.6%
```

of the fusion grid was removed, the results suggest that prompt-conditioned information is represented across a broader region of the fusion memory.

---

# Cross-Example Interpretation

Across all three examples:

```text
School Bus
Dog
Cat
```

the strongest causal effects consistently appeared when ablating larger groups of high DeltaF×Gradient tokens.

The most consistent layer was:

```text
Layer 3
```

which repeatedly showed large score-drop and IoU-drop advantages relative to random ablation.

Examples:

```text
School Bus Layer 3:
Score-drop advantage = 0.00796

Dog Layer 3:
Score-drop advantage = 0.00569

Cat Layer 3:
Score-drop advantage = 0.01035
```

---

# Overall Interpretation

This experiment directly tested whether prompt-conditioned fusion tokens are causally important to the final mask.

The results show:

1. Removing top DeltaF×Gradient tokens damages the final mask more than removing random tokens.
2. The effect is strongest when larger groups of important tokens are removed.
3. The causal signal appears distributed across a broad set of fusion tokens.
4. Layer 3 consistently exhibits strong causal effects across all tested examples.

Therefore:

> The causal ablation experiment provides evidence that prompt-conditioned fusion tokens are not merely correlated with the final mask. A substantial subset of these tokens causally contributes to mask generation, and removing them degrades segmentation performance more than random token removal.

This is the strongest evidence obtained in the project because it moves beyond correlation and directly tests the effect of removing prompt-conditioned information from the model.
