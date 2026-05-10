# Prompt Activation Mapping for SAM 3

> **Method 1 (MVP):** Cross-attention maps + Effective attention for SAM 3 image inference.

This repository visualises, for a given image and text prompt, *where each prompt token attends spatially inside the image* at the prompt → vision cross-attention layer of SAM 3's DETR-style detector.  It also computes **effective attention** — raw attention corrected by value-vector norms — so that high-attention-but-low-impact connections are down-weighted.

---

## What this repo does

| Capability | Status |
|---|---|
| Cross-attention heatmaps per prompt token | ✅ Implemented |
| Effective-attention heatmaps per prompt token | ✅ Implemented |
| Side-by-side raw vs effective comparisons | ✅ Implemented |
| Debug artefacts (JSON + NPZ) | ✅ Implemented |
| PCA on ΔF | 📋 Planned |
| CKA before vs after | 📋 Planned |
| Integrated Gradients (prompt → mask) | 📋 Planned |
| Contrastive two-prompt comparison | 📋 Planned |
| Probing classifiers | 📋 Planned |

---

## What is raw cross-attention?

Raw cross-attention answers: **"Where does this prompt token look in the image?"**

For each prompt token the model produces a distribution of attention weights over all image tokens (spatial positions in the feature map).  Reshaping that 1-D distribution back to the 2-D spatial grid and overlaying it on the original image yields a heatmap.

## What is effective attention?

Raw attention can be misleading because a token may assign **high attention** to an image location whose **value vector has low magnitude**.  The information actually flowing through the network is `Attention × Value`, not `Attention` alone.

Effective attention multiplies raw attention by the L2 norm of the corresponding value vector and then re-normalises:

$$
\text{EffAttn}[t, i] = \frac{A[t, i] \cdot \|V[i]\|_2}{\sum_j A[t, j] \cdot \|V[j]\|_2}
$$

Where:
- $A[t, i]$ — raw attention from prompt token $t$ to image token $i$
- $V[i]$ — value vector at image position $i$

## Why this is the first method

- **Gradient-free**: needs only one forward pass, no backward pass.
- **Intuitive**: directly visualises the prompt ↔ vision interaction mechanism.
- **Fast**: single `torch.no_grad()` pass.
- **Diagnostic**: exposes exactly what the cross-attention layer is doing.

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU with CUDA 12.6+ (or CPU for testing)
- PyTorch 2.7+

### 1. Install SAM 3

SAM 3 is an **external dependency** and must be installed separately:

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

## Usage

### List available attention modules

Discover which modules can be hooked:

```bash
python -m pam.cli --list-attention-modules --device cuda
```

### Generate attention heatmaps

```bash
python -m pam.cli \
    --image examples/bus.jpg \
    --prompt "yellow school bus" \
    --output-dir outputs/bus \
    --device cuda \
    --save-debug
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--image` | *required* | Path to input image |
| `--prompt` | *required* | Text prompt |
| `--output-dir` | `outputs/pam_run` | Output directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--module-filter` | auto-detect | Substring filter for module names |
| `--module-name` | — | Exact module name override |
| `--layer-index` | `first` | `first`, `last`, or integer index |
| `--head-reduction` | `mean` | `mean` or `max` across heads |
| `--alpha` | `0.5` | Overlay transparency |
| `--no-overlay` | `false` | Save heatmaps only (no blending) |
| `--max-tokens` | — | Cap on number of tokens visualised |
| `--resize-long-side` | — | Resize image long side for visualisation |
| `--save-debug` | `false` | Save `debug.json` + `debug_tensors.npz` |
| `--list-attention-modules` | — | Print candidate modules and exit |

---

## Output files

After a run, the output directory contains:

```
outputs/bus/
  raw_token_00_CLS.png                    # Raw attention overlay for [CLS]
  effective_token_00_CLS.png              # Effective attention overlay for [CLS]
  comparison_token_00_CLS.png             # Side-by-side: original | raw | effective
  raw_token_01_yellow.png
  effective_token_01_yellow.png
  comparison_token_01_yellow.png
  ...
  debug.json                              # Metadata, shapes, module info
  debug_tensors.npz                       # Raw numpy tensors
```

### debug.json contents

- `prompt`, `image_path`
- `selected_module_name` and short `repr`
- `all_candidate_module_names`
- `raw_attention_shape`, `value_tensor_shape`
- `num_prompt_tokens`, `num_image_tokens`
- `inferred_spatial_grid`
- `token_labels`
- `output_file_paths`
- `warnings`

---

## Limitations

- **Module naming**: Cross-attention module names may change across SAM 3 versions.  The `--list-attention-modules` command helps discover the correct module.
- **Fused attention kernels**: SAM 3 uses `F.scaled_dot_product_attention` (and optionally Flash Attention 3) which do **not** return attention weight matrices.  This tool reconstructs weights from Q and K via hooks — this is mathematically equivalent to what SDPA computes, but bypasses any kernel-specific optimisations (e.g. dropout patterns).
- **Token labels**: May be approximate if the exact SAM 3 tokeniser cannot be accessed.
- **Interpretability caveat**: Heatmaps are interpretability aids, **not** causal proof by themselves.  High attention ≠ causal importance.

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

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (no SAM 3 checkpoint required)
python -m pytest tests/ -v
```

### Philosophy

- Start with one image and one text prompt.
- Prefer transparent tensor capture over complicated abstractions.
- Save debug artefacts for every run.
- **Fail loudly** rather than silently producing wrong maps.
- Keep SAM 3 as an external dependency.

---

## Project structure

```
prompt-activation-mapping/
  README.md
  pyproject.toml
  src/
    pam/
      __init__.py
      __main__.py
      cli.py                    # CLI entry point
      sam3_loader.py            # SAM 3 model loading
      attention_capture.py      # Hook-based attention capture
      effective_attention.py    # Effective attention computation
      tokenization.py           # Token labelling
      visualization.py          # Heatmap rendering & saving
      debug_utils.py            # JSON/NPZ debug artefacts
  tests/
    test_effective_attention.py
    test_heatmap_shapes.py
  examples/
    run_sam3_pam.sh
  outputs/
    .gitkeep
```

---

## License

MIT
