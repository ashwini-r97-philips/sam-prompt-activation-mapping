# Cross-Layer Fused Decoder Attention

## Objective

Earlier decoder attention analysis traced Query 144 attention separately for each DETR decoder layer. This experiment fused the Query 144 attention maps across all decoder layers to create one cleaner query-to-memory saliency map.

The goal was to test whether fusion-memory tokens that are consistently attended across decoder layers are causally important for the final mask.

## Method

For each example, the six decoder attention maps were averaged:

```text
A_fused = mean(A_layer_0 ... A_layer_5)

The top-k tokens from the fused attention map were ablated before the DETR decoder and compared against random token ablation.

Tested values:

k = 50, 100, 200, 500
Results
School Bus

Fused attention ablation outperformed random ablation at all k values.

Strongest result:

k = 500
top fused attention drop = 0.00441
random drop mean         = 0.00255
advantage                = 0.00186
Dog

Fused attention ablation also outperformed random ablation at all k values.

Strongest result:

k = 500
top fused attention drop = 0.00247
random drop mean         = 0.00142
advantage                = 0.00106
Cat

Cat showed weaker results. Fused attention ablation was below random for k=50, 100, and 200, but slightly above random for k=500.

k = 500
top fused attention drop = 0.00108
random drop mean         = 0.00098
advantage                = 0.00011
Interpretation

The fused Query-144 attention map identifies causally important fusion-memory tokens for schoolbus and dog. This supports the claim that Query 144 is not merely attending visually, but is reading memory locations that contribute to the final mask.

The cat result is weaker, suggesting that either the cat mask depends on more distributed fusion-memory evidence or that the fused attention map is less selective for this example.

Overall, this experiment strengthens the decoder-side pathway:

Fusion memory
↓
Cross-layer Query 144 attention
↓
Final mask

and provides a cleaner query-to-memory explanation than inspecting each decoder layer independently.