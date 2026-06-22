# Query 144 Dominance Analysis

## Objective

The DETR query-tracing experiments identified:

```text
Query 144
```

as the query responsible for generating the final selected mask.

However, an important question remained:

> Is Query 144 only slightly more important than the other decoder queries, or is it uniquely responsible for the final mask?

This experiment was designed to quantify the relative importance of all DETR decoder queries.

---

# Method

The query-gradient traces generated earlier were analyzed.

For every decoder query:

```text
query_score(q) = || d(mask) / d(hs_q) ||
```

was computed.

The following files were analyzed:

```text
query_gradient_scores.csv
```

for:

```text
School Bus
Cat
Dog
```

The top 20 queries were ranked according to gradient magnitude.

The following statistics were reported:

1. Top query.
2. Second-ranked query.
3. Dominance ratio:

```text
Top Query Gradient
-------------------
Second Query Gradient
```

4. Number of queries with non-zero gradient.

---

# Results

## School Bus

### Top Query

```text
Query 144
```

Gradient magnitude:

```text
5.71e-03
```

### Second Query

```text
Query 125
```

Gradient magnitude:

```text
0.0
```

### Non-Zero Queries

```text
1
```

---

## Cat

### Top Query

```text
Query 144
```

Gradient magnitude:

```text
1.02e-03
```

### Second Query

```text
Query 125
```

Gradient magnitude:

```text
0.0
```

### Non-Zero Queries

```text
1
```

---

## Dog

### Top Query

```text
Query 144
```

Gradient magnitude:

```text
1.69e-03
```

### Second Query

```text
Query 125
```

Gradient magnitude:

```text
0.0
```

### Non-Zero Queries

```text
1
```

---

# Interpretation

The result is unusually strong.

For all three examples:

```text
School Bus
Cat
Dog
```

only:

```text
Query 144
```

had non-zero gradient with respect to the selected final mask.

This means:

```text
Query 144
```

completely dominates final-mask generation for the tested examples.

No evidence was found that other DETR queries contribute to the selected mask.

---

# Cross-Example Analysis

The same pattern appeared for all three images:

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

# Interpretation

The DETR decoder contains:

```text
200 object queries
```

but the final selected mask in all tested examples is routed through:

```text
Query 144
```

and no measurable gradient reaches the remaining queries.

This means that the prompt-to-mask pathway discovered throughout the project can be simplified as:

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

for the examples tested.

---

# Overall Conclusion

The Query 144 Dominance Analysis shows that:

1. Query 144 is consistently responsible for the selected final mask.
2. No other decoder query receives measurable gradient from the final mask objective.
3. Query 144 completely dominates mask generation for the school bus, cat, and dog examples.
4. Query 144 acts as the primary readout mechanism through which prompt-conditioned fusion representations influence the final segmentation output.

This result strengthens the interpretation that Query 144 is the key decoder component connecting fusion-memory representations to final mask generation in the tested examples.
