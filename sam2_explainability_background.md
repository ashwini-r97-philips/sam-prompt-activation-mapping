# Explainability & Interpretability Research on SAM 2 — Background Review

> Prepared as background for the SAM 3 Prompt Activation Mapping (PAM) project.
> Scope: papers that **analyze, probe, or explain** SAM 2 (and closely related SAM
> variants), and the kinds of **outputs / artefacts** each produced.

---

## TL;DR — the state of the field

> **Scope note:** This list is restricted to genuine **explainability / interpretability
> (XAI)** work that opens up SAM 2 itself — i.e. it analyses the model's *internal* behaviour
> (stage-wise representations, token / component attribution, what each part of the
> architecture contributes to the decision). It deliberately **excludes**: (a) papers that
> merely *use* SAM 2 as a tool to explain other models, and (b) pure benchmark / robustness /
> comparison / adaptation-method papers — even when they call themselves "interpretable" —
> because those *evaluate* or *improve* SAM, they do not explain its internal decision process.

- **Dedicated XAI work that opens the SAM 2 black box is extremely thin — effectively two
  papers.** One does stage-wise internal analysis of the architecture (Bromley et al.); the
  other does token-level component attribution (E-BayesSAM).
- **No published work analyses how the text / concept prompt reshapes the internal
  visual features of SAM 2 or SAM 3.** This is the gap the PAM project targets.

The two papers below are ordered from stage-wise internal analysis to token-level
component attribution.

---

## 1. An Analysis of Data Transformation Effects on Segment Anything 2

- **Link:** https://arxiv.org/abs/2503.00042 (Bromley, Moore, Saini, Poland, Carrano — LLNL, 2025)
- **What it is about:** The closest existing analogue to our work. A stage-by-stage
  internal analysis of the SAM 2 architecture to understand *how* it achieves
  high-quality video object segmentation.
- **What was done / techniques:**
  - Constructed datasets of **complex video transformations** (noise, distortions, etc.).
  - Passed these transformed videos through SAM 2 and **measured the impact at each
    stage** of the architecture (encoder → memory → decoder pipeline).
  - Produced **visualisations of the segmented object as it propagates through each stage**.
- **Goals:** Understand which architectural stage is responsible for robustness, and how
  the model filters noise and isolates the target object.
- **Results / interpretation:** Each **progressive stage filters transformation noise and
  increasingly emphasises the object of interest** — i.e. robustness is built up
  cumulatively across the pipeline rather than at a single point.
- **Why it matters to us:** This is structurally the same *genre* of claim as our
  layer-wise ΔF finding ("prompt-induced change accumulates monotonically and peaks in
  deep decoder layers"). Different probe (input perturbation vs prompt-on/off ΔF), same
  story shape. Best paper to cite as prior art and to contrast against.

---

## 2. E-BayesSAM — Token-wise Bayesian Interpretation for Uncertainty-Aware Segmentation

- **Link:** https://arxiv.org/abs/2508.17408 (Huang, Liu, Wen, et al. — MICCAI 2025)
- **What it is about:** An efficient Bayesian adaptation of SAM that adds **token-level
  interpretability** and uncertainty estimation. Built on the SAM family; relevant to
  SAM 2's token-based decoder design.
- **What was done / techniques:**
  - **Token-wise Variational Bayesian Inference (T-VBI):** reinterprets SAM's output
    tokens as dynamic probabilistic weights / latent variables (training-free).
  - **Self-Optimizing Kolmogorov-Arnold Network (SO-KAN):** learnable spline activations
    to improve and *interpret* token prediction, enabling pruning of redundant tokens.
- **Goals:** Make SAM's decisions interpretable at the token level and quantify uncertainty,
  while staying efficient for clinical use.
- **Results / interpretation:** Real-time inference (0.03 s/image), competitive Dice
  (~89%), and crucially **identified four critical tokens that govern SAM's decisions** —
  a concrete interpretability artefact (which internal tokens matter).
- **Why it matters to us:** Closest example of *internal-component attribution* on a SAM
  model — analogous in spirit to our head-importance analysis (which heads/tokens drive
  the output).

---

## How these map onto output / artefact types

| Output type | Papers | Example artefact |
|---|---|---|
| Stage/layer-wise visual progression | Bromley et al. (#1) | Object mask shown after each architectural stage |
| Internal token/component attribution | E-BayesSAM (#2) | "4 critical tokens govern the decision" |

---

## Takeaways for the PAM project

1. **The closest precedent (Bromley et al.) is a perturbation-based, stage-wise *qualitative*
   analysis.** Our prompt-on/off ΔF + gradient-gated ΔF + head-importance battery is a
   more *quantitative and mechanistic* probe — a clear differentiator.
2. **Dominant output types in this space are qualitative maps + a few quantitative curves /
   component-attribution claims.** Our per-layer ΔF tables and gradient-weighted scores are
   already richer than most.
3. **No prior work studies prompt → visual-feature conditioning inside SAM 2 or SAM 3.**
   The natural framing: *the only genuine SAM 2 XAI is (a) stage-wise internal analysis
   (Bromley) and (b) token-level component attribution (E-BayesSAM); neither analyses how
   the text / concept prompt reshapes the internal visual feature stream.*

---

## New section: Explainability papers on multimodal encoders and DETR decoders (SAM3-adjacent)

### Status for SAM 3 specifically

- **I did not find a dedicated paper that directly explains SAM 3's multimodal encoder or
  DETR-style decoder internals** in the same sense as XAI/probing work.
- The strongest available prior art is therefore **adjacent transformer-XAI work** that can
  be transferred to SAM 3's architecture (cross-attention explainability, query-level
  decoding analysis, head/token importance).

### A) Multimodal encoder / cross-attention explainability

1. **Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers**
   (Chefer, Gur, Wolf, ICCV 2021)
   - Link: https://arxiv.org/abs/2103.15679
   - Why relevant: Directly explains bi-modal and encoder-decoder transformers via
     cross-attention-aware relevance propagation; this is a strong methodological fit for
     SAM3 prompt-vision interactions.

2. **Transformer Interpretability Beyond Attention Visualization**
   (Chefer, Gur, Wolf, CVPR 2021)
   - Link: https://arxiv.org/abs/2012.09838
   - Why relevant: Shows why raw attention alone is insufficient and introduces more
     faithful transformer attribution, useful for SAM3 internal analyses beyond heatmaps.

3. **Generic Attention-model Explainability by Weighted Relevance Accumulation**
   (Huang, Jia, Zhang, Zhang, arXiv 2023)
   - Link: https://arxiv.org/abs/2308.10240
   - Why relevant: Extends cross-attention explainability with weighted relevance
     accumulation, a useful candidate baseline for multimodal-token attribution in SAM3.

4. **Quantifying Attention Flow in Transformers**
   (Abnar, Zuidema, arXiv 2020)
   - Link: https://arxiv.org/abs/2005.00928
   - Why relevant: Introduces attention rollout/flow for layerwise signal tracing; useful
     for studying where prompt influence accumulates across SAM3 layers.

5. **HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models**
   (Zhu, Zhang, Wang, et al., CVPR 2026)
   - Link: https://arxiv.org/abs/2604.07812
   - Why relevant: Explicitly quantifies head importance in multimodal attention, aligning
     closely with SAM3 head-importance style analyses.

### B) DETR decoder / object-query explainability context

1. **End-to-End Object Detection with Transformers (DETR)**
   (Carion et al., ECCV 2020)
   - Link: https://arxiv.org/abs/2005.12872
   - Why relevant: Foundational DETR object-query decoder that motivates how to interpret
     query-slot behaviour in SAM3-like detector components.

2. **Conditional DETR for Fast Training Convergence**
   (Meng et al., ICCV 2021)
   - Link: https://arxiv.org/abs/2108.06152
   - Why relevant: Adds spatially-conditional cross-attention and improves interpretability
     of query-to-region associations.

3. **Anchor DETR: Query Design for Transformer-Based Object Detection**
   (Wang, Zhang, Yang, Sun, AAAI 2022)
   - Link: https://arxiv.org/abs/2109.07107
   - Why relevant: Makes decoder queries more spatially grounded, which is useful for
     interpretable query semantics in DETR-style decoders.

4. **DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR**
   (Liu et al., ICLR 2022)
   - Link: https://arxiv.org/abs/2201.12329
   - Why relevant: Interprets queries as dynamic box priors updated per layer, offering a
     concrete framework for tracking query evolution in decoder internals.

5. **EIVE: End-to-End Instance-Specific Visual Explanations for Detection Transformers**
   (Xiang, Li, Dai, arXiv 2026)
   - Link: https://arxiv.org/abs/2606.01601
   - Why relevant: Direct DETR-specific explainability method that produces instance-level
     explanations from decoder mechanisms and can inform SAM3 decoder XAI.

### How to position these in PAM writing

- **Direct SAM-family XAI**: Bromley et al. (#1 above) and E-BayesSAM (#2 above).
- **Transferable methods for SAM3 internals**: multimodal transformer XAI and DETR-query
  explainability papers in this new section.
- **Honest claim**: There is still no established paper that directly opens SAM 3's
  multimodal encoder/decoder internals with the same granularity PAM is targeting.

---

### Notes / caveats

- **Excluded by scope:** the following were deliberately dropped because they are *not*
  XAI of SAM 2's internals:
  - Papers that *use* SAM 2 to explain other models (Explain Any Concept / EAC,
    saliency-fusion methods, LLM region-captioning like Perceive Anything).
  - **Benchmark / comparison studies** (e.g. *Comparing SAM 2 and SAM 3 for 3D Medical
    Data*, arXiv:2511.21926) — these evaluate behaviour, they don't explain internals.
  - **Robustness / corruption studies** (e.g. *Robustness of SAM Under Corruptions*,
    arXiv:2306.07713) — input-perturbation evaluation, and on SAM v1.
  - **Limitation / failure-characterisation studies** (e.g. *Quantifying the Limits of
    Segmentation Foundation Models*, arXiv:2412.04243) — "interpretable metrics" about
    *object properties*, not attribution of the model's decision.
  - **Adaptation / feature-enrichment methods** that self-describe as "interpretable"
    (e.g. *SAMwave*, arXiv:2507.20186) — they *improve* SAM, they don't explain it.
- SAM 3 capability extensions (e.g. **SAM3-I**, https://arxiv.org/abs/2512.04585) are
  *not* interpretability papers — they extend instruction-following. Treat as related-work
  context only.
- Many "interpretable SAM 2" hits in search are **downstream medical pipelines** that add
  an interpretable head for a clinical task; they do not analyse SAM 2's internals and are
  omitted here except where the interpretability contribution is substantive (#2).
