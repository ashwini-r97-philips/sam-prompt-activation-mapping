# Multimodal Decoder DeltaF Summary

## Method

For each image, I ran SAM3 twice:

1. real prompt
2. empty baseline prompt

I hooked:

- transformer.encoder.layers.0
- transformer.encoder.layers.1
- transformer.encoder.layers.2
- transformer.encoder.layers.3
- transformer.encoder.layers.4
- transformer.encoder.layers.5

For each layer:

DeltaF = F_prompt - F_baseline

Then I computed the L2 norm over the 256 feature dimensions and reshaped 5184 tokens into a 72x72 heatmap.

## Results

Dog:
3.11 → 5.95 → 7.54 → 11.00 → 16.45 → 24.30

Cat:
2.68 → 5.97 → 8.40 → 12.54 → 19.95 → 29.40

Schoolbus:
3.44 → 7.02 → 8.78 → 13.57 → 19.54 → 27.43

## Observation

Mean DeltaF increases monotonically from layer 0 to layer 5 for all three examples.

## Interpretation

Prompt-induced feature change accumulates through the Multimodal Decoder.

## Limitation

This shows that the prompt changes image-token representations, but it does not yet prove that those changes affect the final mask.

## Next step

Gradient-weighted DeltaF.
