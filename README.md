# Prompt Activation Mapping for SAM 3

> Gradient-weighted edge attribution + cross-attention heatmaps for SAM 3 grounded segmentation.

Given an image and a text prompt, this tool explains **which image regions and prompt tokens** drove the model's detection, mask, and semantic segmentation outputs. It does this by hooking into cross-attention layers across the encoder, decoder, and geometry encoder, then computing gradient-weighted edge attributions that decompose the model's output into per-layer, per-head, per-token spatial contributions.

---

## Table of contents

- [Capabilities](#capabilities)
- [Architecture overview](#architecture-overview)
- [Attribution formula](#attribution-formula)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Modes of operation](#modes-of-operation)
  - [1. List attention modules](#1-list-attention-modules)
  - [2. Raw / effective attention heatmaps](#2-raw--effective-attention-heatmaps)
  - [3. Attribution flow](#3-attribution-flow)
  - [4. Compare target objectives](#4-compare-target-objectives)
  - [5. Compare instance vs semantic attribution](#5-compare-instance-vs-semantic-attribution)
- [Target objectives](#target-objectives)
  - [Detection score objectives](#detection-score-objectives)
  - [Instance mask objectives](#instance-mask-objectives)
  - [Semantic segmentation objectives](#semantic-segmentation-objectives)
- [CLI reference](#cli-reference)
- [Output files](#output-files)
- [How to interpret outputs](#how-to-interpret-outputs)
- [Architecture deep dive](#architecture-deep-dive)
- [Development](#development)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Open questions](#-open-question-how-should-we-aggregate-attention)

---

## Capabilities

| Capability | Status |
|---|---|
| Cross-attention heatmaps per prompt token (raw + effective) | ✅ |
| Gradient-weighted edge attribution (attribution flow) | ✅ |
| Group A encoder maps — where prompt tokens write into image features | ✅ |
| Group A geometry maps — how geometry queries read from image | ✅ |
| Group B decoder maps — where a specific object query reads from image | ✅ |
| Joint product & sum heatmaps combining Groups A and B | ✅ |
| 11 target objectives (score, mask, semantic) | ✅ |
| Multi-objective comparison panels | ✅ |
| Instance vs semantic attribution comparison tool | ✅ |
| Foreground/background contrastive attribution | ✅ |
| Per-layer per-head per-token contribution CSVs | ✅ |
| MHA reconstruction validation | ✅ |
| Local conservation verification | ✅ |
| Debug artefacts (JSON + NPZ + tensor inventory) | ✅ |

---

## Architecture overview

SAM 3 uses a DETR-style architecture with three groups of cross-attention modules. The attribution system hooks all three:

```
                                 ┌──────────────────────────────┐
                                 │  Text prompt → tokeniser     │
                                 │  "yellow school bus"          │
                                 │  → 3 text + 30 geometry/     │
                                 │    visual prompt tokens       │
                                 │    (33 total)                 │
                                 └──────────┬───────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Geometry Encoder (3 layers)  — Group A geometry                     │
│  geometry_encoder.encode.{0-2}.cross_attn_image                      │
│  1 geometry query × 5184 image tokens                                │
│  "How does the geometry query read from image features?"             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Transformer Encoder (6 layers)  — Group A encoder                   │
│  transformer.encoder.layers.{0-5}.cross_attn_image                   │
│  5184 image queries × 33 prompt sources                              │
│  "Where does each prompt token write into the image representation?" │
└───────────┬───────────────────────────────┬──────────────────────────┘
            │                               │
            │  encoder_hidden_states        │  encoder_hidden_states
            ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────────────┐
│  Transformer Decoder      │   │  Pixel Embedding                  │
│  (6 layers) — Group B     │   │  pixel_embed = _embed_pixels(     │
│  decoder                  │   │    backbone_feats, img_ids,       │
│  transformer.decoder.     │   │    encoder_hidden_states)         │
│  layers.{0-5}.cross_attn  │   │                                   │
│  201 queries × 5184       │   └───────────┬───────────────────────┘
│  image tokens             │               │
│                           │               ▼
│  (1 presence + 200        │   ┌───────────────────────────────────┐
│   object queries)         │   │  Semantic Seg Head                │
│                           │   │  Conv2d(pixel_dim, 1, 1)          │
└───────────┬───────────────┘   │  → semantic_seg [1,1,288,288]     │
            │                   └───────────────────────────────────┘
            ▼
┌───────────────────────────┐
│  Mask Head + Box Head      │
│  pred_masks [1,200,288,288]│
│  pred_logits [1,200,1]     │
│  presence_logit [1,1]      │
└────────────────────────────┘
```

**Key insight**: Instance mask outputs (`pred_masks`, `pred_logits`) flow through both encoder (Group A) and decoder (Group B). Semantic segmentation outputs (`semantic_seg`) flow through encoder (Group A) only — they bypass the decoder entirely. This means:

- For instance mask objectives: both Group A and Group B have significant attribution mass.
- For semantic segmentation objectives: Group A has significant mass, Group B is near-zero (expected, not a bug).

---

## Attribution formula

Each cross-attention module produces an edge attribution matrix:

$$
R_{ij}^{h} = A_{ij}^{h} \cdot \langle V_j^h, \frac{\partial S}{\partial O_i^h} \rangle
$$

where:
- $A_{ij}^h$ — attention probability from query $i$ to source $j$ at head $h$
- $V_j^h$ — value vector at source position $j$, head $h$
- $\frac{\partial S}{\partial O_i^h}$ — gradient of the target scalar $S$ w.r.t. the pre-output-projection head output at query $i$, head $h$
- $\langle \cdot, \cdot \rangle$ — dot product

Positive attribution means "this edge contributed positively to the target scalar." Negative attribution means "this edge suppressed the target."

**Local conservation**: for each query position $i$ and head $h$:

$$
\sum_j R_{ij}^h = \langle O_i^h, \frac{\partial S}{\partial O_i^h} \rangle
$$

This is verified automatically and reported in `local_conservation_report.csv`.

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU with CUDA 12.6+
- PyTorch 2.7+

### 1. Install SAM 3

```bash
git clone https://github.com/facebookresearch/sam3.git external/sam3
cd external/sam3
pip install -e ".[notebooks]"
cd ../..
```

> ⚠️ You need to [request checkpoint access](https://huggingface.co/facebook/sam3) on Hugging Face and authenticate via `hf auth login` before running inference.

### 2. Install this package

```bash
pip install -e ".[dev]"
```

---

## Quick start

```bash
# Run attribution flow with the recommended contrastive mask objective
python -m pam.cli \
    --image inputs/2.jpg \
    --prompt "yellow school bus" \
    --target-mask-index 0 \
    --aggregation-mode attribution-flow \
    --target-objective mask_contrastive_logit \
    --background-penalty 0.25 \
    --output-dir outputs/bus_attribution \
    --save-debug \
    --validate-mha-reconstruction
```

This produces overlays, CSVs, debug JSON, and a 6-panel summary in `outputs/bus_attribution/`.

---

## Modes of operation

### 1. List attention modules

Discover which modules can be hooked:

```bash
python -m pam.cli --list-attention-modules --device cuda
```

### 2. Raw / effective attention heatmaps

Simple forward-pass attention visualisation (no gradients needed):

```bash
python -m pam.cli \
    --image inputs/2.jpg \
    --prompt "yellow school bus" \
    --output-dir outputs/bus_attention \
    --device cuda \
    --save-debug
```

**Raw attention** shows where each prompt token looks in the image. **Effective attention** reweights by value-vector magnitude:

$$
\text{EffAttn}[t, i] = \frac{A[t, i] \cdot \|V[i]\|_2}{\sum_j A[t, j] \cdot \|V[j]\|_2}
$$

### 3. Attribution flow

Gradient-based edge attribution across all 15 cross-attention modules:

```bash
python -m pam.cli \
    --image inputs/2.jpg \
    --prompt "yellow school bus" \
    --target-mask-index 0 \
    --aggregation-mode attribution-flow \
    --target-objective mask_contrastive_logit \
    --output-dir outputs/bus_mask_attribution \
    --save-debug \
    --validate-mha-reconstruction
```

The pipeline:
1. Runs inference **with gradients** (bypassing `@torch.inference_mode()`)
2. Hooks 15 cross-attention modules (6 encoder + 3 geometry + 6 decoder)
3. Resolves the target objective into a differentiable scalar
4. Calls `backward()` on the target scalar
5. For each hooked module, reconstructs MHA internals (Q, K, V, attention probs) and computes per-head gradient w.r.t. pre-output-projection outputs
6. Computes edge attributions $R_{ij}^h = A_{ij}^h \cdot \langle V_j^h, \nabla_{O_i^h} S \rangle$
7. Aggregates into spatial heatmaps per group
8. Produces joint heatmaps combining Group A and Group B

### 4. Compare target objectives

Run attribution for multiple objectives on the same image and produce a side-by-side comparison panel:

```bash
python -m pam.cli \
    --image inputs/2.jpg \
    --prompt "yellow school bus" \
    --target-mask-index 0 \
    --aggregation-mode attribution-flow \
    --compare-target-objectives \
        score \
        mask_contrastive_logit \
        mask_foreground_logit_mean \
        semantic_contrastive_logit \
    --output-dir outputs/bus_comparison
```

### 5. Compare instance vs semantic attribution

After running both an instance-mask and semantic-seg attribution, use the comparison tool:

```bash
python -m pam.compare_outputs \
    --instance-dir outputs/bus_mask_attribution \
    --semantic-dir outputs/bus_semantic_attribution \
    --output outputs/comparison_panel.png
```

Produces a 2×3 panel: Original, Selected mask, Instance joint_sum, Instance joint_product, Semantic joint_sum, Semantic joint_product.

---

## Target objectives

The `--target-objective` flag selects which differentiable scalar to backpropagate through. Different objectives answer different questions.

### Detection score objectives

These target the detection confidence scores. Gradients flow through `pred_logits` and `presence_logit_dec`.

| Objective | Formula | Question answered |
|---|---|---|
| `score` | `sigmoid(pred_logits[0,q,0]) × sigmoid(presence)` | What drove the model's combined detection confidence? |
| `object_query_score` | `sigmoid(pred_logits[0,q,0])` | What drove this query's individual logit (ignoring presence)? |
| `raw_query_score` | `pred_logits[0,q,0]` | What drove the raw logit before sigmoid? |

### Instance mask objectives

These target the predicted mask logits. Gradients flow through `pred_masks`, which depends on both encoder and decoder. Uses a foreground mask (from postprocessed detection masks, resized to logit resolution) to separate foreground and background pixels.

| Objective | Formula | Question answered |
|---|---|---|
| `mask_logit_mean` | `mean(pred_masks[0,q])` | What drove the overall mask output (all pixels)? |
| `mask_foreground_logit_mean` | `mean(pred_masks[0,q][fg])` | What drove high logits specifically in the foreground? |
| `mask_probability_foreground_mean` | `mean(sigmoid(pred_masks[0,q][fg]))` | Same, but through sigmoid (saturates for high logits) |
| `mask_contrastive_logit` | `mean(fg) − α·mean(bg)` | What maximises foreground logits while suppressing background? **Recommended.** |
| `combined_query_and_mask` | `sigmoid(logit) + λ·mean(fg_logits)` | Joint detection confidence + mask quality |

The `--background-penalty` flag controls α (default 0.25). The `--lambda-mask` flag controls λ (default 1.0).

### Semantic segmentation objectives

These target the semantic segmentation output (`_raw_outputs.semantic_seg`), which is computed as `Conv2d(pixel_embed)`. The `pixel_embed` depends on encoder hidden states but **not** on decoder object queries. Therefore:
- **Group A** (encoder + geometry) attribution mass will be significant.
- **Group B** (decoder) attribution mass will be near-zero. This is expected.

| Objective | Formula | Question answered |
|---|---|---|
| `semantic_logit_mean` | `mean(semantic_seg[0,0])` | What drove overall semantic segmentation activation? |
| `semantic_foreground_logit_mean` | `mean(semantic_seg[0,0][fg])` | What drove semantic activation in the foreground? |
| `semantic_contrastive_logit` | `mean(sem_fg) − α·mean(sem_bg)` | What maximises semantic foreground while suppressing background? |

The foreground mask is the same as for instance mask objectives — derived from the postprocessed detection mask, resized to the semantic map resolution (288×288).

### Choosing an objective

| Use case | Recommended objective |
|---|---|
| General-purpose "why did the model detect this?" | `mask_contrastive_logit` |
| Investigating detection confidence only | `score` or `object_query_score` |
| Comparing instance vs semantic pathways | `mask_contrastive_logit` + `semantic_contrastive_logit` |
| Understanding which prompt tokens matter most | Any objective; check `group_a_prompt_token_contributions.csv` |
| Checking which decoder layers/heads matter | Instance objectives; check `group_b_query_layer_head_contributions.csv` |

---

## CLI reference

### Core arguments

| Flag | Default | Description |
|---|---|---|
| `--image` | *required* | Path to input image |
| `--prompt` | *required* | Text prompt, e.g. `"yellow school bus"` |
| `--output-dir` | `outputs/pam_run` | Output directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--aggregation-mode` | `attention` | `attention` (raw/effective) or `attribution-flow` (gradient-based) |

### Attribution-flow arguments

| Flag | Default | Description |
|---|---|---|
| `--target-mask-index` | `None` | Index into filtered detections (0-based) |
| `--target-objective` | `score` | Target scalar for backward pass (see [Target objectives](#target-objectives)) |
| `--query-index` | `None` | Override decoder query index (0-based object query) |
| `--background-penalty` | `0.25` | Background penalty α for contrastive objectives |
| `--lambda-mask` | `1.0` | Mask weight λ for `combined_query_and_mask` |
| `--compare-target-objectives` | `None` | Run multiple objectives and save comparison panel |
| `--validate-mha-reconstruction` | `false` | Verify Q·K^T reconstruction matches actual attention |
| `--save-debug` | `false` | Save full debug JSON, tensor inventory, and NPZ |
| `--top-k-contributors` | `24` | Number of top contributors in grid/CSV |
| `--save-negative` | `false` | Also save negative attribution overlays |
| `--attribution-capture-device` | `cpu` | Device for attribution tensors (`cpu` saves GPU memory) |

### Attention-mode arguments

| Flag | Default | Description |
|---|---|---|
| `--module-filter` | auto-detect | Substring filter for module names |
| `--module-name` | — | Exact module name override |
| `--layer-index` | `first` | `first`, `last`, or integer |
| `--head-reduction` | `mean` | `mean` or `max` across heads |
| `--alpha` | `0.5` | Overlay transparency |
| `--no-overlay` | `false` | Save heatmaps only |
| `--max-tokens` | — | Cap on tokens visualised |
| `--skip-per-head` | `false` | Skip per-head PNGs |
| `--skip-per-layer` | `false` | Skip per-layer PNGs |

---

## Output files

### Attribution flow mode outputs

After an attribution-flow run, the output directory contains:

#### Overlays (PNG)

| File | Description |
|---|---|
| `group_a_prompt_write_positive_overlay.png` | Where prompt tokens write positively into the image (sum over all encoder layers, heads, tokens) |
| `group_a_prompt_write_negative_overlay.png` | Negative attribution (suppression) in encoder |
| `group_b_query_read_positive_overlay.png` | Where the selected decoder query reads from image *(instance objectives only)* |
| `group_b_query_read_negative_overlay.png` | Negative attribution in decoder *(instance objectives only)* |
| `joint_product_positive_overlay.png` | `sqrt(norm(GroupA) × norm(GroupB))` — highlights regions important in both groups *(instance objectives only)* |
| `joint_sum_positive_overlay.png` | `0.5·norm(GroupA) + 0.5·norm(GroupB)` — balanced combination *(instance objectives only)* |
| `full_mask_attribution_summary.png` | 6-panel summary: Original, Mask, Group A, Group B, Joint product, Joint sum |
| `top_group_a_prompt_token_grid.png` | Grid showing top-K prompt token individual spatial maps |
| `top_group_b_query_head_grid.png` | Grid showing top-K decoder layer×head spatial maps *(instance objectives only)* |
| `decoder_token_comparison.png` | Comparison of selected query vs other top queries *(instance objectives only)* |
| `detection_boxes.png` | Input image with bounding boxes |
| `mask_00.png` | Selected detection mask |

For semantic objectives, Group B files are absent (Group B mass is near-zero so no overlays are generated).

#### CSVs

| File | Description |
|---|---|
| `group_a_prompt_token_contributions.csv` | Per-prompt-token positive/negative/signed contribution and normalised weight |
| `group_a_layer_head_token_contributions.csv` | Per-layer × per-head × per-token breakdown |
| `group_b_query_layer_head_contributions.csv` | Per-layer × per-head contribution for the selected query *(instance objectives)* |
| `group_b_contribution_per_query.csv` | All 201 decoder tokens ranked by positive contribution *(instance objectives)* |
| `attribution_flow_top_contributors.csv` | Top-K contributors across both groups, ranked |
| `query_index_validation.csv` | Validation table mapping object query indices ↔ decoder token indices *(instance objectives)* |
| `mha_reconstruction_report.csv` | Per-module max/mean reconstruction error |
| `local_conservation_report.csv` | Per-module local conservation error |

#### JSONs

| File | Description |
|---|---|
| `debug_attribution_flow.json` | Full run metadata — target objective, scalar value, query indices, spatial grid, module counts, reconstruction errors, conservation errors, Group B mass, warnings |
| `target_objective_details.json` | Selected objective formula, foreground/background stats, target value |
| `query_index_mapping.json` | `object_query_index`, `decoder_token_index`, `presence_token_index`, resolution method |
| `query_resolution.json` | How the query index was resolved (kept_indices, gradient, manual, etc.) |
| `output_tensor_inventory.json` | Every tensor in the model outputs — shape, dtype, requires_grad, grad_fn |
| `detections.json` | Detection results — scores, boxes, labels |

#### NPY files

| File | Description |
|---|---|
| `group_a_prompt_image_positive.npy` | Raw Group A positive spatial map `[Gh, Gw]` |
| `group_a_prompt_image_negative.npy` | Raw Group A negative spatial map |
| `group_b_query_image_positive.npy` | Raw Group B positive spatial map *(instance objectives)* |
| `group_b_query_image_negative.npy` | Raw Group B negative spatial map *(instance objectives)* |

---

## How to interpret outputs

### Reading the summary panel

The `full_mask_attribution_summary.png` is the primary output. It shows 6 panels:

1. **Original** — the input image
2. **Selected mask** — the detection mask for `--target-mask-index`
3. **Group A prompt-write** — where the text prompt conditions the image. Hot regions are where the prompt has the most influence on the image representation. This is the **encoder** perspective.
4. **Group B query-read** — where the winning object query reads from the image. Hot regions are where the decoder "looks" to produce this specific detection. *(Near-zero for semantic objectives.)*
5. **Joint product** — `sqrt(norm(A) × norm(B))`. Highlights regions important in **both** groups. Regions must be both prompt-influenced AND query-attended to show up.
6. **Joint sum** — `0.5·norm(A) + 0.5·norm(B)`. More forgiving; a region can show up if it's important in either group.

### Reading the CSVs

**`group_a_prompt_token_contributions.csv`** — answers "which prompt token mattered most?"

```
token_index, token_label, positive_contribution, ..., normalized_positive_fraction
0,          txt_yellow,  0.042,                 ..., 0.12
1,          txt_school,  0.038,                 ..., 0.11
2,          txt_bus,     0.051,                 ..., 0.15
3,          geo_00,      0.008,                 ..., 0.02
...
```

- `positive_contribution`: total positive attribution mass for this token
- `normalized_positive_fraction`: what fraction of the total positive mass this token accounts for

**`attribution_flow_top_contributors.csv`** — answers "which specific layer×head×token combination contributed the most?"

```
rank, group,    module_name,                              layer, head, positive, normalized_weight
1,    A_encoder, transformer.encoder.layers.5.cross_attn, 5,     3,    0.012,   0.08
2,    B_decoder, transformer.decoder.layers.4.cross_attn, 4,     7,    0.010,   0.07
...
```

### Reading the debug JSON

Key fields in `debug_attribution_flow.json`:

| Field | Meaning |
|---|---|
| `target_objective` | Which objective was used |
| `target_source` | `"pred_logits"`, `"pred_masks"`, or `"semantic_seg"` |
| `target_scalar_value` | The scalar that was backpropagated |
| `target_requires_grad` | Must be `true` for valid attribution |
| `object_query_index` | Index into pred_logits (0–199) |
| `decoder_token_index` | Index into decoder attention (0–200); = object_query_index + 1 |
| `group_b_positive_mass` | Total Group B attribution mass. Near-zero for semantic objectives. |
| `group_b_near_zero_expected` | `true` for semantic objectives |
| `foreground_area_fraction` | Fraction of pixels classified as foreground |
| `max_reconstruction_error` | Largest MHA reconstruction error across all modules |
| `max_conservation_error` | Largest local conservation error |

### Understanding Group B near-zero mass

When using semantic objectives (`semantic_*`), you'll see:

```
Group B positive attribution mass: 0.0000e+00
NOTE: Group B attribution mass is near-zero. This is expected for semantic_seg
targets because semantic_seg depends on encoder (Group A) not decoder (Group B).
```

This is **correct and expected**. The semantic segmentation head (`semantic_seg = Conv2d(pixel_embed)`) only depends on encoder hidden states via `pixel_embed`. It does not pass through the decoder's object queries. Therefore, gradients from `semantic_seg` do not flow through decoder cross-attention, and Group B attributions are zero.

This is itself informative: it confirms that the semantic branch is architecturally independent of the decoder. Compare with instance mask objectives where Group B mass is significant.

### Query index mapping

SAM 3's decoder prepends a **presence token** at index 0. This means:
- **Object query index** (0–199): index into `pred_logits`, `pred_masks`, `pred_boxes`
- **Decoder token index** (0–200): index into decoder cross-attention query dimension. `decoder_token_index = object_query_index + 1`
- The presence token at index 0 has its own attribution and is tracked separately

The mapping is saved in `query_index_mapping.json` and resolved automatically from `_kept_indices` (which maps filtered detection index → object query index).

---

## Architecture deep dive

### Cross-attention groups

| Group | Modules | Dimensions | What it captures |
|---|---|---|---|
| Group A encoder | `transformer.encoder.layers.{0-5}.cross_attn_image` (6 layers) | 5184 image queries × 33 prompt sources, 8 heads | How the text prompt conditions image features at each spatial position |
| Group A geometry | `geometry_encoder.encode.{0-2}.cross_attn_image` (3 layers) | 1 geometry query × 5184 image sources, 8 heads | How the geometry query aggregates over the image |
| Group B decoder | `transformer.decoder.layers.{0-5}.cross_attn` (6 layers) | 201 decoder queries × 5184 image sources, 8 heads | How each object query (and presence token) reads from image features |

### MHA reconstruction

SAM 3 uses `F.scaled_dot_product_attention` (SDPA) internally, which doesn't expose attention weights. We reconstruct them:

1. Hook the forward pass to capture `query`, `key`, `value` inputs
2. Recompute `A = softmax(Q·K^T / √d_k)` from captured Q, K
3. Validate reconstruction: `max_error = max|A_reconstructed - A_actual|` (when available)
4. Compute per-head outputs: `O_h = A·V_h`

Typical reconstruction errors: <1e-2 for most layers (layers 0–1 of decoder can reach ~3.8e-2 due to numerical precision with bfloat16).

### Spatial grid

The 5184 image tokens correspond to a 72×72 spatial grid. All spatial heatmaps are produced at 72×72 resolution and then overlaid onto the original image via bilinear interpolation.

### Prompt tokens

The 33 prompt sources include:
- 3 text tokens (from the text prompt, e.g. "yellow", "school", "bus")
- 30 geometry/visual prompt tokens

By default, all 33 tokens are included in Group A attribution. The text tokens typically dominate the positive attribution mass.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (no SAM 3 checkpoint required)
python -m pytest tests/ -v

# Currently: 34 tests passing
```

### Tests

| Test file | What it tests |
|---|---|
| `test_attribution_flow_shapes.py` | Edge attribution shapes, group map reshaping |
| `test_effective_attention.py` | Effective attention computation, row sums |
| `test_heatmap_shapes.py` | Spatial grid inference, filename sanitisation |
| `test_local_conservation.py` | Local conservation property |
| `test_mha_reconstruction.py` | MHA reconstruction accuracy, batch_first handling |

---

## Project structure

```
src/pam/
  cli.py                      # CLI entry point: attention + attribution-flow modes
  sam3_loader.py               # Model loading + inference (no_grad and with_grad paths)
  attribution_capture.py       # Hook system — registers on 15 cross-attention modules
  mha_reconstruction.py        # Reconstruct Q/K/V/A from hooked inputs
  attribution_flow.py          # Edge attribution computation + group spatial maps + CSVs
  attribution_visualization.py # Overlay rendering, summary panels, comparison grids
  target_resolver.py           # 11 target objectives → differentiable scalar
  query_resolver.py            # mask_index → object_query_index ↔ decoder_token_index
  compare_outputs.py           # Instance vs semantic attribution comparison tool
  effective_attention.py       # Effective attention computation (attention × value norms)
  attention_capture.py         # Attention capture hooks (for raw attention mode)
  tokenization.py              # Token labelling
  visualization.py             # Heatmap rendering & overlay utilities
  debug_utils.py               # JSON/NPZ debug artefacts
tests/
  test_attribution_flow_shapes.py
  test_effective_attention.py
  test_heatmap_shapes.py
  test_local_conservation.py
  test_mha_reconstruction.py
```

---

## Limitations

- **Module naming**: Cross-attention module names may change across SAM 3 versions. Use `--list-attention-modules` to discover current names.
- **Fused attention kernels**: SAM 3 uses `F.scaled_dot_product_attention` which doesn't return attention weights. We reconstruct from Q, K — mathematically equivalent but reconstruction error can reach ~3.8e-2 on decoder layer 0.
- **Token labels**: May be approximate if the SAM 3 tokeniser cannot be accessed (whitespace-split fallback).
- **`@torch.inference_mode()`**: The SAM 3 processor uses `@torch.inference_mode()` decorators, which block gradient computation. The tool bypasses this by calling `model.backbone.forward_image()`, `model.backbone.forward_text()`, `model.forward_grounding()` directly.
- **Interpretability caveat**: Heatmaps are interpretability aids, not causal proof. High attribution ≠ causal necessity. The attribution formula decomposes the output into per-edge contributions but does not prove counterfactual importance.
- **bfloat16 precision**: SAM 3 runs in bfloat16 by default. Reconstruction errors are higher than with float32 but acceptable for attribution purposes.

---

## ⚠️ Open question: how should we aggregate attention?

> *Multi-layer attention capture — hook all 6 encoder `cross_attn_image` layers, average the attention across layers — should we be averaging the attention?? I feel like that causes the heatmap to diffuse more. Averaging across layers and across heads feels like something that would cause the heatmap focus to diffuse more. Also there are multiple masks getting generated right, wouldn't we want to get the attention for each of the mask? Or each of the prompt? What should we focus on?*

### Analysis

The concern is correct. There are three distinct aggregation problems, each requiring a different approach.

**1. Layer aggregation (6 encoder layers)**

Each layer refines the image features with prompt information. Early layers attend broadly, later layers attend sharply. Averaging dilutes the sharp late-layer signals with the diffuse early-layer ones. The current approach saves **per-layer heatmaps** so each layer's contribution is visible individually, plus a **layer-max** aggregate (taking the sharpest signal at each spatial position across layers) instead of mean.

**2. Head aggregation (8 heads per layer)**

Different heads specialise on different aspects (colour, shape, position, etc). Averaging across heads merges unrelated spatial patterns. The current approach saves **per-head heatmaps** for each layer, alongside the head-mean and head-max reductions.

**3. Per-detection vs shared attention**

This is an architectural constraint. The 6 encoder `cross_attn_image` layers produce a **shared** prompt-conditioned image representation — all detections emerge from the same conditioned features. The per-detection specialisation happens in the **decoder**:

| Module | What attends to what | Per-detection? |
|---|---|---|
| `transformer.encoder.layers.*.cross_attn_image` | Image positions → prompt tokens (text + geometry) | No — shared |
| `transformer.decoder.layers.*.cross_attn` | Object queries → image features | **Yes** — each query becomes one detection |
| `transformer.decoder.layers.*.ca_text` | Object queries → text features | **Yes** — each query attends to text differently |

For per-detection attention maps (e.g. "where did detection #3 look in the image?"), the decoder `cross_attn` is the right target. For "how did the text prompt condition the image features globally?", the encoder `cross_attn_image` is the right target.

**4. Per-token heatmaps vs combined prompt attention**

> *Why are we doing this per prompt token? When we say "yellow school bus" aren't we expecting all the tokens to contribute to the end result? Should we be aggregating across all prompt tokens (including the ones that are not text)?*

This is a valid concern, but the naive fix — summing attention across all prompt tokens — doesn't work, and the reason is instructive.

In the encoder cross-attention, each image position distributes its attention across all 33 prompt tokens via softmax. **The attention weights at every image position already sum to 1.0.** Summing across all prompt tokens gives a uniform map of ones — no spatial information at all.

What IS meaningful is to sum across **only the text tokens** (3 out of 33). Since the softmax distributes attention across all 33 tokens, the fraction going to the 3 text tokens varies by image position. This gives a **text influence fraction** map:

$$
\text{TextFraction}[i] = \sum_{j \in \text{text}} A[i, j]
$$

where $A[i,j]$ is the softmax attention from image position $i$ to prompt token $j$.

- Image positions where `TextFraction ≈ 3/33 ≈ 0.09` attend to text tokens about as much as to geometry tokens (uniform baseline).
- Image positions where `TextFraction >> 0.09` are disproportionately influenced by the text prompt.
- Image positions where `TextFraction << 0.09` are dominated by geometry/visual prompt tokens.

This "text vs geometry influence" map is a different and complementary view to per-token heatmaps. Per-token maps show **which text token** matters where; the combined text fraction map shows **whether text matters at all** at each position.

There is also a richer aggregation: instead of summing raw attention weights, compute the **actual information flow** through text tokens — the L2 norm of the text tokens' contribution to the cross-attention output:

$$
\text{TextFlow}[i] = \left\| \sum_{j \in \text{text}} A[i,j] \cdot V_j \right\|_2
$$

This captures not just how much attention text gets, but how much information it actually injects. A token can receive high attention but have a near-zero value vector (no information flows). `TextFlow` accounts for both.

Neither of these invalidates per-token heatmaps — they answer different questions:

| View | Question |
|---|---|
| Per-token heatmap | "Where does the word *yellow* specifically attend?" |
| Text fraction map | "Which image regions are text-influenced vs geometry-influenced?" |
| Text flow map | "Where does text actually inject information into the features?" |
| Per-head heatmap | "What does each attention head specialise on?" |

### How attribution flow resolves this

The attribution-flow mode (`--aggregation-mode attribution-flow`) sidesteps the aggregation dilemma entirely. Instead of choosing how to average/max/sum attention across layers and heads, it uses **gradient-weighted edge attribution**: each layer×head×token combination gets a contribution weight determined by how much it actually influenced the target output. The gradient automatically handles which layers and heads matter — no manual aggregation choice needed.

---

## Future methods

The following methods are **planned but not yet implemented**.

### A. PCA on ΔF = F_after_prompt − F_before_prompt

> *How did the prompt change the representation, and what structure does that change have?*

- Capture features before and after prompt conditioning.
- Compute ΔF.
- Run PCA across spatial tokens.
- Map top 3 PCs to RGB channels.
- Inspect explained variance: is the prompt-induced shift low-rank or diffuse?

### B. CKA before vs after

> *How much did the prompt change the representation overall?*

- Compute CKA(F_before, F_after).
- Repeat per DETR encoder layer.
- Compare distributions across text, point, box, and exemplar prompts.
- Use as a quantitative benchmark of prompt impact.

### C. Integrated Gradients (prompt → mask)

> *Which prompt embedding dimensions contribute to the final mask?*

- Baseline: zero or neutral prompt embedding.
- Interpolate to actual prompt embedding.
- Run 50–100 gradient steps.
- Integrate gradients w.r.t. selected mask score or logits.
- Requires gradients and multiple forward/backward passes — intentionally excluded from the MVP.

### D. Contrastive two-prompt comparison

> *What changes when prompt A is replaced with prompt B on the same image?*

- Run two prompts on the same image.
- Compare conditioned features: visualise F_A − F_B spatially.
- Optionally PCA on the contrastive difference.
- Useful for controlled prompt-specific variation analysis.

### E. Probing classifiers

> *What information did the prompt inject into the features?*

- Train simple linear probes on frozen features.
- Classify inside-mask vs outside-mask.
- Compare probe accuracy on: before-prompt features, after-prompt features, and ΔF.
- High ΔF probe accuracy ⇒ prompt-induced changes are directly decodable for segmentation.

---

## License

MIT
