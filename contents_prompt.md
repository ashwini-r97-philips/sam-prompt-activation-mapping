You are building the first MVP of a research/interpretability project called “Prompt Activation Mapping” for SAM 3.

Goal:
Build only Method 1 for now: Cross-Attention Maps + Effective Attention for SAM 3 image inference.

The project should visualize, for a given image and text prompt, where each prompt token attends spatially inside the image at the first prompt/vision cross-attention interaction in the SAM 3 DETR-style detector. It should also compute “effective attention” by correcting raw attention weights with value-vector norms, so high-attention but low-impact connections are downweighted.

Do not build PCA, CKA, Integrated Gradients, contrastive comparison, or probing classifiers yet. Only document them in the README as planned future methods.

Context:
SAM 3 is available at:
https://github.com/facebookresearch/sam3.git

The official SAM 3 README says SAM 3 is a unified promptable segmentation model for images and videos, with a detector and tracker sharing a vision encoder. The detector is DETR-based and conditioned on text, geometry, and image exemplars. SAM 3 currently requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+, and checkpoint access from the Hugging Face model repo.

Use the official SAM 3 implementation. Do not vendor or rewrite SAM 3 internals unless absolutely necessary. Prefer importing from a local editable install or from a git submodule.

Primary deliverable:
A working CLI and small Python package that:
1. Loads a SAM 3 image model.
2. Runs one no-gradient forward pass on an image and a text prompt.
3. Captures cross-attention weights and value vectors from the relevant prompt/vision cross-attention module.
4. Computes:
   - raw attention heatmaps per prompt token
   - effective attention heatmaps per prompt token
5. Saves visualizations to disk:
   - one overlay per token for raw attention
   - one overlay per token for effective attention
   - optionally a side-by-side raw vs effective image per token
   - a JSON/NPZ debug artifact containing tensor shapes, selected layer/module names, token labels, and saved heatmap paths
6. Includes a README that explains:
   - project purpose
   - installation
   - how to clone/install SAM 3
   - how to run the CLI
   - how raw attention is computed
   - how effective attention is computed
   - known limitations
   - future methods: PCA on feature differences, CKA, Integrated Gradients, contrastive two-prompt comparison, probing classifiers

Repository expectations:
Create a clean Python project structure, for example:

prompt-activation-mapping/
  README.md
  pyproject.toml
  src/
    pam/
      __init__.py
      cli.py
      sam3_loader.py
      attention_capture.py
      effective_attention.py
      tokenization.py
      visualization.py
      debug_utils.py
  examples/
    run_sam3_pam.sh
  outputs/
    .gitkeep
  tests/
    test_effective_attention.py
    test_heatmap_shapes.py

Use clear, heavily documented code. Add docstrings and comments wherever tensor shapes are manipulated.

Important technical goal:
The code must be robust to SAM 3 internal naming differences. Do not hardcode a single fragile module path unless unavoidable. Instead:
1. Print/discover candidate attention modules using model.named_modules().
2. Provide a CLI option to list candidate modules.
3. Provide a CLI option to manually select a module by substring or exact module name.
4. Provide a reasonable automatic heuristic to select likely cross-attention modules.

CLI requirements:
Implement a command like:

python -m pam.cli \
  --image path/to/image.jpg \
  --prompt "yellow school bus" \
  --output-dir outputs/bus_pam \
  --device cuda \
  --module-filter cross \
  --layer-index first \
  --save-debug

Also implement:

python -m pam.cli --list-attention-modules --device cuda

Useful options:
--image: input image path
--prompt: text prompt
--output-dir: output directory
--device: cuda/cpu
--module-filter: substring used to filter attention modules, default should search for cross-attention-like names
--module-name: exact or full module name override
--layer-index: first, last, or integer index among matching modules
--head-reduction: mean or max, default mean
--save-debug: save NPZ/JSON debug outputs
--no-overlay: save heatmaps only
--alpha: overlay transparency
--max-tokens: optional cap on number of tokens visualized
--resize-long-side: optional visualization resize only

SAM 3 setup behavior:
The project should assume SAM 3 is installed separately, ideally via:

git clone https://github.com/facebookresearch/sam3.git external/sam3
cd external/sam3
pip install -e ".[notebooks]"

Then this project imports:

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

Use the official image model and processor path where possible.

If the user has not installed SAM 3, fail with a friendly error explaining how to clone and install it. Do not silently mock SAM 3.

Forward-pass behavior:
Run inference under torch.no_grad() and model.eval().
The goal for this MVP is one forward pass, no gradients.

The inference path should be close to:

model = build_sam3_image_model()
model.eval().to(device)
processor = Sam3Processor(model)
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt=prompt)

Adapt this if the official SAM 3 API differs in the installed repo version.

Attention capture requirements:
Implement an AttentionCapture class that can:
1. Register forward hooks on candidate attention modules.
2. Store:
   - module name
   - raw attention weights if exposed by the module output
   - query/key/value tensors if available from inputs or internal attributes
   - output tensor shapes
   - input tensor shapes
3. Remove hooks cleanly after inference.

Because SAM 3 internals may use custom attention implementations, support multiple capture strategies:

Strategy A: Standard PyTorch MultiheadAttention-like modules
- Try to capture attn_output_weights if the module returns it.
- If the module is called with need_weights=False, document that this module may need monkey-patching or a wrapper.
- Do not globally break model behavior.

Strategy B: Custom attention modules with q/k/v projections
- Detect common attributes such as q_proj, k_proj, v_proj, in_proj_weight, qkv, proj, num_heads, n_heads, embed_dim, head_dim.
- Capture module input tensors.
- If possible, reconstruct Q, K, V and attention weights manually:
  scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)
  attention = softmax(scores, dim=-1)
  values = V
- Be very explicit in comments about assumed tensor shapes.

Strategy C: Fallback
- If attention weights and values cannot be recovered, save a useful debug report listing candidate modules, input/output shapes, and module repr.
- Raise a clear error telling the user to run --list-attention-modules and/or select --module-name.

Important:
Do not fake attention maps. If attention tensors cannot be captured correctly, fail loudly with diagnostics.

Effective attention formula:
Given raw attention A and value vectors V:

A shape should be normalized to:
[B, H, T_prompt, T_image]

V shape should be normalized to:
[B, H, T_image, D_head]

Compute:

value_norm = ||V||_2 over D_head
effective = A * value_norm[:, :, None, :]
effective = effective / (effective.sum(dim=-1, keepdim=True) + eps)

Then reduce heads:

if head_reduction == "mean":
    token_heatmap = effective[0, :, token_index, :].mean(dim=0)

if head_reduction == "max":
    token_heatmap = effective[0, :, token_index, :].max(dim=0).values

Do the same for raw attention.

Simpler explanation for comments:
Raw attention shows where the prompt token looked. Effective attention shows where it looked and the value vector carried a strong signal.

Prompt token mapping:
Implement best-effort token labels.
Try to access SAM 3’s actual tokenizer through the processor/model if possible.
If exact tokenizer access is difficult, provide a fallback simple token split and clearly mark token labels as approximate.
The heatmap tensor token dimension and displayed token labels must be aligned as much as possible.

If the actual cross-attention token dimension includes special tokens, keep them and label them clearly, for example:
[CLS], [EOS], token_03, etc.

Image-token to spatial heatmap mapping:
Implement robust logic:
1. Infer the number of image tokens N_image from attention shape.
2. If N_image is a perfect square, reshape to sqrt(N_image) x sqrt(N_image).
3. If not a perfect square, attempt to infer H x W from captured image feature metadata if available.
4. If H x W cannot be inferred, save a 1D attention plot and raise a warning that spatial reshape was not possible.

For typical ViT-like image tokens, expect a spatial grid. Do not hardcode 72x72 unless verified from tensor shapes.

Visualization:
Use PIL/matplotlib/numpy/torch only.
For every token:
- Normalize heatmap to [0, 1].
- Resize heatmap to original image size.
- Save:
  raw_token_{idx}_{safe_label}.png
  effective_token_{idx}_{safe_label}.png
  comparison_token_{idx}_{safe_label}.png

The comparison image should place original image, raw overlay, and effective overlay side by side if feasible.

Avoid requiring OpenCV unless necessary.

Debug artifacts:
Save debug.json with:
- prompt
- image path
- selected module name
- selected module repr short string
- all candidate module names
- raw attention shape
- value tensor shape
- normalized attention shape
- normalized value shape
- number of prompt tokens
- number of image tokens
- inferred spatial grid size
- token labels
- output file paths
- warnings

Optionally save tensors:
debug_tensors.npz containing raw attention, effective attention, token heatmaps, etc., moved to CPU and converted to float32 numpy arrays.

Tests:
Add unit tests that do not require SAM 3 checkpoints:
1. test_effective_attention_shape:
   - create random A with shape [1, 4, 5, 64]
   - create random V with shape [1, 4, 64, 32]
   - verify effective shape is [1, 4, 5, 64]
   - verify sums over image tokens are approximately 1
2. test_heatmap_square_inference:
   - N=64 gives 8x8
   - N=72*72 gives 72x72
   - non-square raises/warns cleanly
3. test_safe_filename_token_labels:
   - spaces and special characters are sanitized

README content:
Write a comprehensive README with the following sections:

1. Title:
Prompt Activation Mapping for SAM 3

2. What this repo does:
This repo currently implements cross-attention maps and effective attention maps for SAM 3 text-prompted image segmentation. It captures prompt-to-image attention from the DETR-style detector and turns each prompt token’s spatial attention into a heatmap.

3. What is raw cross-attention?
Explain:
Raw cross-attention answers: “Where does this prompt token look in the image?”
For each prompt token, attention weights over image tokens are reshaped into a spatial heatmap.

4. What is effective attention?
Explain:
Raw attention can be misleading because a token may assign high attention to an image location whose value vector has low magnitude. Effective attention multiplies raw attention by the L2 norm of the corresponding value vector, then renormalizes.

Formula:
EffectiveAttention[token, image_token] =
Attention[token, image_token] * ||Value[image_token]||_2

Then normalize over image tokens.

5. Why this is the first method:
It is gradient-free, needs one forward pass, is intuitive, and directly visualizes the prompt/vision interaction mechanism.

6. Installation:
Include:
- Python 3.12+
- CUDA-compatible PyTorch
- clone/install SAM 3
- request checkpoint access on Hugging Face if needed
- install this repo with pip install -e .

Mention that SAM 3 should be installed separately:
git clone https://github.com/facebookresearch/sam3.git external/sam3
cd external/sam3
pip install -e ".[notebooks]"

7. Usage:
Show:
python -m pam.cli --list-attention-modules --device cuda

Then:
python -m pam.cli --image examples/bus.jpg --prompt "yellow school bus" --output-dir outputs/bus --device cuda --save-debug

8. Output files:
Explain raw overlays, effective overlays, comparisons, debug JSON, NPZ.

9. Limitations:
- Cross-attention module names may change across SAM 3 versions.
- Some optimized attention kernels may not expose attention weights.
- If attention is computed through fused kernels, the code may need to reconstruct attention from Q/K/V or disable fused attention.
- Token labels may be approximate if the exact tokenizer is not exposed.
- Heatmaps are interpretability aids, not causal proof by themselves.

10. Future methods:
Document but do not implement:

A. PCA on ΔF = F_after_prompt - F_before_prompt
Answer: How did the prompt change the representation, and what structure does that change have?
Plan:
- capture features before prompt conditioning and after prompt conditioning
- compute ΔF
- run PCA across spatial tokens
- map top 3 PCs to RGB
- inspect explained variance to see whether the prompt-induced shift is low-rank or diffuse

B. CKA before vs after
Answer: How much did the prompt change the representation overall?
Plan:
- compute CKA(F_before, F_after)
- repeat per DETR encoder layer
- compare distributions across text, point, box, and exemplar prompts
- use as a quantitative benchmark

C. Integrated Gradients prompt to mask
Answer: Which prompt embedding dimensions contribute to the final mask?
Plan:
- baseline prompt embedding = zero or neutral prompt embedding
- interpolate to actual prompt embedding
- run 50-100 gradient steps
- integrate gradients with respect to selected mask score or mask logits
- use completeness property as a rigor advantage
Note:
This requires gradients and multiple forward/backward passes, so it is intentionally excluded from the MVP.

D. Contrastive two-prompt comparison
Answer: What changes when prompt A is replaced with prompt B on the same image?
Plan:
- run two prompts on the same image
- compare conditioned features
- visualize F_A - F_B spatially
- optionally PCA on the contrastive difference
- useful for controlled prompt-specific variation

E. Probing classifiers
Answer: What information did the prompt inject into the features?
Plan:
- train simple linear probes on frozen features
- classify inside-mask vs outside-mask
- compare probe accuracy on before-prompt features, after-prompt features, and ΔF
- high ΔF probe accuracy means prompt-induced changes are directly decodable for segmentation

11. Development philosophy:
- Start with one image and one text prompt.
- Prefer transparent tensor capture over complicated abstractions.
- Save debug artifacts for every run.
- Fail loudly rather than silently producing wrong maps.
- Keep SAM 3 as an external dependency.

Implementation details:
Use type hints.
Use dataclasses where helpful, for example:

@dataclass
class CapturedAttention:
    module_name: str
    raw_attention: torch.Tensor | None
    values: torch.Tensor | None
    query: torch.Tensor | None = None
    key: torch.Tensor | None = None
    input_shapes: list | None = None
    output_shapes: list | None = None
    warnings: list[str] = field(default_factory=list)

Core functions to implement:

discover_attention_modules(model, filter_text=None) -> list[tuple[str, nn.Module]]

select_attention_module(candidates, module_name=None, layer_index="first") -> tuple[str, nn.Module]

compute_effective_attention(attn, values, eps=1e-6) -> torch.Tensor

reduce_heads(attn_or_effective, mode="mean") -> torch.Tensor

infer_spatial_grid(num_image_tokens, metadata=None) -> tuple[int, int] | None

make_attention_overlay(image, heatmap, alpha=0.5) -> PIL.Image

save_token_heatmaps(...)

Potential issue:
Some SAM 3 modules may use fused scaled dot-product attention and not expose attention matrices. If so:
- Try to hook before the fused call and reconstruct attention from Q and K.
- If reconstruction is impossible, write diagnostics and stop.
- Do not invent or approximate attention from outputs alone.

Acceptance criteria:
1. Running --list-attention-modules prints candidate module names and types.
2. Running the CLI on a valid SAM 3 install and image creates output files.
3. debug.json contains tensor shapes and selected module information.
4. effective attention is mathematically tested with synthetic tensors.
5. README clearly states that only Method 1 is implemented and all other methods are future work.
6. The code is readable enough for a researcher to inspect and modify hooks manually.

Do not:
- Implement the other five methods yet.
- Train anything.
- Require gradients.
- Require a dataset.
- Rewrite SAM 3.
- Produce fake heatmaps if attention capture fails.
- Hide shape mismatches.