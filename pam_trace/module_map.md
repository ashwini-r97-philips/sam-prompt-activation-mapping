# SAM 3 Module Map (PAM Trace)

Total named modules in model: **1166**  Hooked: **71**

Each row carries two names:
- **`code_name`** -- the dotted module path in the SAM 3 source code
  (e.g. `transformer.encoder.layers.0`).
- **`diagram_name`** -- the architecture-block label from the SAM 3
  diagram (e.g. *Multimodal Decoder*). These names sometimes differ:
  notably the central fusion block is called `transformer.encoder`
  in code but appears as the **Multimodal Decoder** in the diagram.
  The class docstring of `TransformerEncoderLayer` explicitly notes
  that the layer was "previously called TransformerDecoderLayer".

## Hooked modules by group

### `image_encoder`

| Layer | code_name (path) | code_name (class) | diagram_name | Source tag | Source file |
|---:|---|---|---|---|---|
|  | `backbone.vision_backbone` | `Sam3DualViTDetNeck` | Image Encoder | `image` | `sam3/model/necks.py` |
|  | `backbone.vision_backbone.trunk` | `ViT` | Image Encoder (ViT trunk) | `image` | `sam3/model/vitdet.py` |
|  | `backbone.vision_backbone.position_encoding` | `PositionEmbeddingSine` | Image Encoder (Pos Enc) | `image` | `sam3/model/position_encoding.py` |
|  | `backbone.vision_backbone.convs` | `ModuleList` | Image Encoder (Multi-scale Neck) | `image` | `nn/modules/container.py` |

### `text_encoder`

| Layer | code_name (path) | code_name (class) | diagram_name | Source tag | Source file |
|---:|---|---|---|---|---|
|  | `backbone.language_backbone` | `VETextEncoder` | Text Encoder | `text` | `sam3/model/text_encoder_ve.py` |
|  | `backbone.language_backbone.encoder` | `TextTransformer` | Text Encoder | `text` | `sam3/model/text_encoder_ve.py` |
|  | `backbone.language_backbone.resizer` | `Linear` | Text Encoder (Resizer) | `text` | `nn/modules/linear.py` |

### `prompt_encoder`

| Layer | code_name (path) | code_name (class) | diagram_name | Source tag | Source file |
|---:|---|---|---|---|---|
|  | `geometry_encoder` | `SequenceGeometryEncoder` | Geometry Prompt Encoder | `prompt_geom` | `sam3/model/geometry_encoders.py` |
|  | `geometry_encoder.pos_enc` | `PositionEmbeddingSine` | Geometry Prompt Encoder | `prompt_geom` | `sam3/model/position_encoding.py` |
|  | `geometry_encoder.label_embed` | `Embedding` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/sparse.py` |
|  | `geometry_encoder.cls_embed` | `Embedding` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/sparse.py` |
|  | `geometry_encoder.points_direct_project` | `Linear` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/linear.py` |
|  | `geometry_encoder.points_pool_project` | `Linear` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/linear.py` |
|  | `geometry_encoder.points_pos_enc_project` | `Linear` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/linear.py` |
|  | `geometry_encoder.boxes_direct_project` | `Linear` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/linear.py` |
|  | `geometry_encoder.boxes_pool_project` | `Conv2d` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/conv.py` |
|  | `geometry_encoder.boxes_pos_enc_project` | `Linear` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/linear.py` |
|  | `geometry_encoder.final_proj` | `Linear` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/linear.py` |
|  | `geometry_encoder.norm` | `LayerNorm` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/normalization.py` |
|  | `geometry_encoder.img_pre_norm` | `LayerNorm` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/normalization.py` |
|  | `geometry_encoder.encode` | `ModuleList` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/container.py` |
|  | `geometry_encoder.encode_norm` | `LayerNorm` | Geometry Prompt Encoder | `prompt_geom` | `nn/modules/normalization.py` |

### `multimodal_decoder`

| Layer | code_name (path) | code_name (class) | diagram_name | Source tag | Source file |
|---:|---|---|---|---|---|
|  | `transformer.encoder` | `TransformerEncoderFusion` | Multimodal Decoder | `fused` | `sam3/model/encoder.py` |
| 0 | `transformer.encoder.layers.0` | `TransformerEncoderLayer` | Multimodal Decoder Layer | `fused` | `sam3/model/encoder.py` |
| 0 | `transformer.encoder.layers.0.self_attn` | `MultiheadAttentionWrapper` | Multimodal Decoder Self-Attention | `fused` | `sam3/model/model_misc.py` |
| 0 | `transformer.encoder.layers.0.cross_attn_image` | `MultiheadAttentionWrapper` | Multimodal Decoder Cross-Attention to Image | `fused` | `sam3/model/model_misc.py` |
| 1 | `transformer.encoder.layers.1` | `TransformerEncoderLayer` | Multimodal Decoder Layer | `fused` | `sam3/model/encoder.py` |
| 1 | `transformer.encoder.layers.1.self_attn` | `MultiheadAttentionWrapper` | Multimodal Decoder Self-Attention | `fused` | `sam3/model/model_misc.py` |
| 1 | `transformer.encoder.layers.1.cross_attn_image` | `MultiheadAttentionWrapper` | Multimodal Decoder Cross-Attention to Image | `fused` | `sam3/model/model_misc.py` |
| 2 | `transformer.encoder.layers.2` | `TransformerEncoderLayer` | Multimodal Decoder Layer | `fused` | `sam3/model/encoder.py` |
| 2 | `transformer.encoder.layers.2.self_attn` | `MultiheadAttentionWrapper` | Multimodal Decoder Self-Attention | `fused` | `sam3/model/model_misc.py` |
| 2 | `transformer.encoder.layers.2.cross_attn_image` | `MultiheadAttentionWrapper` | Multimodal Decoder Cross-Attention to Image | `fused` | `sam3/model/model_misc.py` |
| 3 | `transformer.encoder.layers.3` | `TransformerEncoderLayer` | Multimodal Decoder Layer | `fused` | `sam3/model/encoder.py` |
| 3 | `transformer.encoder.layers.3.self_attn` | `MultiheadAttentionWrapper` | Multimodal Decoder Self-Attention | `fused` | `sam3/model/model_misc.py` |
| 3 | `transformer.encoder.layers.3.cross_attn_image` | `MultiheadAttentionWrapper` | Multimodal Decoder Cross-Attention to Image | `fused` | `sam3/model/model_misc.py` |
| 4 | `transformer.encoder.layers.4` | `TransformerEncoderLayer` | Multimodal Decoder Layer | `fused` | `sam3/model/encoder.py` |
| 4 | `transformer.encoder.layers.4.self_attn` | `MultiheadAttentionWrapper` | Multimodal Decoder Self-Attention | `fused` | `sam3/model/model_misc.py` |
| 4 | `transformer.encoder.layers.4.cross_attn_image` | `MultiheadAttentionWrapper` | Multimodal Decoder Cross-Attention to Image | `fused` | `sam3/model/model_misc.py` |
| 5 | `transformer.encoder.layers.5` | `TransformerEncoderLayer` | Multimodal Decoder Layer | `fused` | `sam3/model/encoder.py` |
| 5 | `transformer.encoder.layers.5.self_attn` | `MultiheadAttentionWrapper` | Multimodal Decoder Self-Attention | `fused` | `sam3/model/model_misc.py` |
| 5 | `transformer.encoder.layers.5.cross_attn_image` | `MultiheadAttentionWrapper` | Multimodal Decoder Cross-Attention to Image | `fused` | `sam3/model/model_misc.py` |

### `detector_decoder`

| Layer | code_name (path) | code_name (class) | diagram_name | Source tag | Source file |
|---:|---|---|---|---|---|
|  | `transformer.decoder` | `TransformerDecoder` | Detector Decoder | `detector_queries` | `sam3/model/decoder.py` |
| 0 | `transformer.decoder.layers.0` | `TransformerDecoderLayer` | Detector Decoder Layer | `detector_queries` | `sam3/model/decoder.py` |
| 0 | `transformer.decoder.layers.0.cross_attn` | `MultiheadAttentionWrapper` | Detector Decoder Cross-Attention to Image | `detector_queries` | `sam3/model/model_misc.py` |
| 0 | `transformer.decoder.layers.0.ca_text` | `MultiheadAttention` | Detector Decoder Cross-Attention to Text | `detector_queries` | `nn/modules/activation.py` |
| 0 | `transformer.decoder.layers.0.self_attn` | `MultiheadAttention` | Detector Decoder Self-Attention | `detector_queries` | `nn/modules/activation.py` |
| 1 | `transformer.decoder.layers.1` | `TransformerDecoderLayer` | Detector Decoder Layer | `detector_queries` | `sam3/model/decoder.py` |
| 1 | `transformer.decoder.layers.1.cross_attn` | `MultiheadAttentionWrapper` | Detector Decoder Cross-Attention to Image | `detector_queries` | `sam3/model/model_misc.py` |
| 1 | `transformer.decoder.layers.1.ca_text` | `MultiheadAttention` | Detector Decoder Cross-Attention to Text | `detector_queries` | `nn/modules/activation.py` |
| 1 | `transformer.decoder.layers.1.self_attn` | `MultiheadAttention` | Detector Decoder Self-Attention | `detector_queries` | `nn/modules/activation.py` |
| 2 | `transformer.decoder.layers.2` | `TransformerDecoderLayer` | Detector Decoder Layer | `detector_queries` | `sam3/model/decoder.py` |
| 2 | `transformer.decoder.layers.2.cross_attn` | `MultiheadAttentionWrapper` | Detector Decoder Cross-Attention to Image | `detector_queries` | `sam3/model/model_misc.py` |
| 2 | `transformer.decoder.layers.2.ca_text` | `MultiheadAttention` | Detector Decoder Cross-Attention to Text | `detector_queries` | `nn/modules/activation.py` |
| 2 | `transformer.decoder.layers.2.self_attn` | `MultiheadAttention` | Detector Decoder Self-Attention | `detector_queries` | `nn/modules/activation.py` |
| 3 | `transformer.decoder.layers.3` | `TransformerDecoderLayer` | Detector Decoder Layer | `detector_queries` | `sam3/model/decoder.py` |
| 3 | `transformer.decoder.layers.3.cross_attn` | `MultiheadAttentionWrapper` | Detector Decoder Cross-Attention to Image | `detector_queries` | `sam3/model/model_misc.py` |
| 3 | `transformer.decoder.layers.3.ca_text` | `MultiheadAttention` | Detector Decoder Cross-Attention to Text | `detector_queries` | `nn/modules/activation.py` |
| 3 | `transformer.decoder.layers.3.self_attn` | `MultiheadAttention` | Detector Decoder Self-Attention | `detector_queries` | `nn/modules/activation.py` |
| 4 | `transformer.decoder.layers.4` | `TransformerDecoderLayer` | Detector Decoder Layer | `detector_queries` | `sam3/model/decoder.py` |
| 4 | `transformer.decoder.layers.4.cross_attn` | `MultiheadAttentionWrapper` | Detector Decoder Cross-Attention to Image | `detector_queries` | `sam3/model/model_misc.py` |
| 4 | `transformer.decoder.layers.4.ca_text` | `MultiheadAttention` | Detector Decoder Cross-Attention to Text | `detector_queries` | `nn/modules/activation.py` |
| 4 | `transformer.decoder.layers.4.self_attn` | `MultiheadAttention` | Detector Decoder Self-Attention | `detector_queries` | `nn/modules/activation.py` |
| 5 | `transformer.decoder.layers.5` | `TransformerDecoderLayer` | Detector Decoder Layer | `detector_queries` | `sam3/model/decoder.py` |
| 5 | `transformer.decoder.layers.5.cross_attn` | `MultiheadAttentionWrapper` | Detector Decoder Cross-Attention to Image | `detector_queries` | `sam3/model/model_misc.py` |
| 5 | `transformer.decoder.layers.5.ca_text` | `MultiheadAttention` | Detector Decoder Cross-Attention to Text | `detector_queries` | `nn/modules/activation.py` |
| 5 | `transformer.decoder.layers.5.self_attn` | `MultiheadAttention` | Detector Decoder Self-Attention | `detector_queries` | `nn/modules/activation.py` |

### `heads`

| Layer | code_name (path) | code_name (class) | diagram_name | Source tag | Source file |
|---:|---|---|---|---|---|
|  | `transformer.decoder.bbox_embed` | `MLP` | Box Head | `final_outputs` | `sam3/model/model_misc.py` |

### `segmentation`

| Layer | code_name (path) | code_name (class) | diagram_name | Source tag | Source file |
|---:|---|---|---|---|---|
|  | `segmentation_head` | `UniversalSegmentationHead` | Segmentation Head | `mask_head` | `sam3/model/maskformer_segmentation.py` |
|  | `segmentation_head.pixel_decoder` | `PixelDecoder` | Pixel Decoder | `pixel_features` | `sam3/model/maskformer_segmentation.py` |
|  | `segmentation_head.semantic_seg_head` | `Conv2d` | Semantic Head | `semantic` | `nn/modules/conv.py` |
|  | `segmentation_head.instance_seg_head` | `Conv2d` | Mask Predictor | `mask_head` | `nn/modules/conv.py` |

## All named modules (full inventory)

| code_name (path) | code_name (class) | Hooked | diagram_name | Source tag |
|---|---|:---:|---|---|
| `backbone` | `SAM3VLBackbone` |  |  | `` |
| `backbone.vision_backbone` | `Sam3DualViTDetNeck` | yes | Image Encoder | `image` |
| `backbone.vision_backbone.trunk` | `ViT` | yes | Image Encoder (ViT trunk) | `image` |
| `backbone.vision_backbone.trunk.patch_embed` | `PatchEmbed` |  |  | `` |
| `backbone.vision_backbone.trunk.patch_embed.proj` | `Conv2d` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks` | `ModuleList` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.drop_path` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.0.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.1.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.2.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.3.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.4.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.5.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.6.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.7.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.8.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.9.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.10.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.11.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.12.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.13.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.14.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.15.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.16.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.17.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.18.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.19.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.20.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.21.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.22.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.23.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.24.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.25.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.26.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.27.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.28.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.29.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.30.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31` | `Block` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.norm1` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.attn` | `Attention` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.attn.qkv` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.attn.proj` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.ls1` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.drop_path` | `DropPath` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.norm2` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.mlp` | `Mlp` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.mlp.fc1` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.mlp.act` | `GELU` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.mlp.drop1` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.mlp.norm` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.mlp.fc2` | `Linear` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.mlp.drop2` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.ls2` | `Identity` |  |  | `` |
| `backbone.vision_backbone.trunk.blocks.31.dropout` | `Dropout` |  |  | `` |
| `backbone.vision_backbone.trunk.ln_pre` | `LayerNorm` |  |  | `` |
| `backbone.vision_backbone.trunk.ln_post` | `Identity` |  |  | `` |
| `backbone.vision_backbone.position_encoding` | `PositionEmbeddingSine` | yes | Image Encoder (Pos Enc) | `image` |
| `backbone.vision_backbone.convs` | `ModuleList` | yes | Image Encoder (Multi-scale Neck) | `image` |
| `backbone.vision_backbone.convs.0` | `Sequential` |  |  | `` |
| `backbone.vision_backbone.convs.0.dconv_2x2_0` | `ConvTranspose2d` |  |  | `` |
| `backbone.vision_backbone.convs.0.gelu` | `GELU` |  |  | `` |
| `backbone.vision_backbone.convs.0.dconv_2x2_1` | `ConvTranspose2d` |  |  | `` |
| `backbone.vision_backbone.convs.0.conv_1x1` | `Conv2d` |  |  | `` |
| `backbone.vision_backbone.convs.0.conv_3x3` | `Conv2d` |  |  | `` |
| `backbone.vision_backbone.convs.1` | `Sequential` |  |  | `` |
| `backbone.vision_backbone.convs.1.dconv_2x2` | `ConvTranspose2d` |  |  | `` |
| `backbone.vision_backbone.convs.1.conv_1x1` | `Conv2d` |  |  | `` |
| `backbone.vision_backbone.convs.1.conv_3x3` | `Conv2d` |  |  | `` |
| `backbone.vision_backbone.convs.2` | `Sequential` |  |  | `` |
| `backbone.vision_backbone.convs.2.conv_1x1` | `Conv2d` |  |  | `` |
| `backbone.vision_backbone.convs.2.conv_3x3` | `Conv2d` |  |  | `` |
| `backbone.vision_backbone.convs.3` | `Sequential` |  |  | `` |
| `backbone.vision_backbone.convs.3.maxpool_2x2` | `MaxPool2d` |  |  | `` |
| `backbone.vision_backbone.convs.3.conv_1x1` | `Conv2d` |  |  | `` |
| `backbone.vision_backbone.convs.3.conv_3x3` | `Conv2d` |  |  | `` |
| `backbone.language_backbone` | `VETextEncoder` | yes | Text Encoder | `text` |
| `backbone.language_backbone.encoder` | `TextTransformer` | yes | Text Encoder | `text` |
| `backbone.language_backbone.encoder.token_embedding` | `Embedding` |  |  | `` |
| `backbone.language_backbone.encoder.transformer` | `Transformer` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks` | `ModuleList` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.0.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.1.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.2.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.3.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.4.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.5.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.6.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.7.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.8.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.9.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.10.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.11.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.12.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.13.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.14.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.15.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.16.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.17.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.18.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.19.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.20.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.21.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.22.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23` | `ResidualAttentionBlock` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.attn` | `MultiheadAttention` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.ln_1` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.ln_2` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.ls_1` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.ls_2` | `Identity` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.mlp` | `Sequential` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.mlp.c_fc` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.mlp.gelu` | `GELU` |  |  | `` |
| `backbone.language_backbone.encoder.transformer.resblocks.23.mlp.c_proj` | `Linear` |  |  | `` |
| `backbone.language_backbone.encoder.ln_final` | `LayerNorm` |  |  | `` |
| `backbone.language_backbone.resizer` | `Linear` | yes | Text Encoder (Resizer) | `text` |
| `geometry_encoder` | `SequenceGeometryEncoder` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.pos_enc` | `PositionEmbeddingSine` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.label_embed` | `Embedding` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.cls_embed` | `Embedding` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.points_direct_project` | `Linear` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.points_pool_project` | `Linear` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.points_pos_enc_project` | `Linear` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.boxes_direct_project` | `Linear` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.boxes_pool_project` | `Conv2d` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.boxes_pos_enc_project` | `Linear` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.final_proj` | `Linear` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.norm` | `LayerNorm` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.img_pre_norm` | `LayerNorm` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.encode` | `ModuleList` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `geometry_encoder.encode.0` | `TransformerEncoderLayer` |  |  | `` |
| `geometry_encoder.encode.0.self_attn` | `MultiheadAttentionWrapper` |  |  | `` |
| `geometry_encoder.encode.0.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `geometry_encoder.encode.0.cross_attn_image` | `MultiheadAttentionWrapper` |  |  | `` |
| `geometry_encoder.encode.0.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `geometry_encoder.encode.0.linear1` | `Linear` |  |  | `` |
| `geometry_encoder.encode.0.dropout` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.0.linear2` | `Linear` |  |  | `` |
| `geometry_encoder.encode.0.norm1` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.0.norm2` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.0.norm3` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.0.dropout1` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.0.dropout2` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.0.dropout3` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.1` | `TransformerEncoderLayer` |  |  | `` |
| `geometry_encoder.encode.1.self_attn` | `MultiheadAttentionWrapper` |  |  | `` |
| `geometry_encoder.encode.1.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `geometry_encoder.encode.1.cross_attn_image` | `MultiheadAttentionWrapper` |  |  | `` |
| `geometry_encoder.encode.1.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `geometry_encoder.encode.1.linear1` | `Linear` |  |  | `` |
| `geometry_encoder.encode.1.dropout` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.1.linear2` | `Linear` |  |  | `` |
| `geometry_encoder.encode.1.norm1` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.1.norm2` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.1.norm3` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.1.dropout1` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.1.dropout2` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.1.dropout3` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.2` | `TransformerEncoderLayer` |  |  | `` |
| `geometry_encoder.encode.2.self_attn` | `MultiheadAttentionWrapper` |  |  | `` |
| `geometry_encoder.encode.2.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `geometry_encoder.encode.2.cross_attn_image` | `MultiheadAttentionWrapper` |  |  | `` |
| `geometry_encoder.encode.2.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `geometry_encoder.encode.2.linear1` | `Linear` |  |  | `` |
| `geometry_encoder.encode.2.dropout` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.2.linear2` | `Linear` |  |  | `` |
| `geometry_encoder.encode.2.norm1` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.2.norm2` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.2.norm3` | `LayerNorm` |  |  | `` |
| `geometry_encoder.encode.2.dropout1` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.2.dropout2` | `Dropout` |  |  | `` |
| `geometry_encoder.encode.2.dropout3` | `Dropout` |  |  | `` |
| `geometry_encoder.encode_norm` | `LayerNorm` | yes | Geometry Prompt Encoder | `prompt_geom` |
| `transformer` | `TransformerWrapper` |  |  | `` |
| `transformer.encoder` | `TransformerEncoderFusion` | yes | Multimodal Decoder | `fused` |
| `transformer.encoder.layers` | `ModuleList` |  |  | `` |
| `transformer.encoder.layers.0` | `TransformerEncoderLayer` | yes | Multimodal Decoder Layer | `fused` |
| `transformer.encoder.layers.0.self_attn` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Self-Attention | `fused` |
| `transformer.encoder.layers.0.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.0.cross_attn_image` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Cross-Attention to Image | `fused` |
| `transformer.encoder.layers.0.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.0.linear1` | `Linear` |  |  | `` |
| `transformer.encoder.layers.0.dropout` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.0.linear2` | `Linear` |  |  | `` |
| `transformer.encoder.layers.0.norm1` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.0.norm2` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.0.norm3` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.0.dropout1` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.0.dropout2` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.0.dropout3` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.1` | `TransformerEncoderLayer` | yes | Multimodal Decoder Layer | `fused` |
| `transformer.encoder.layers.1.self_attn` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Self-Attention | `fused` |
| `transformer.encoder.layers.1.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.1.cross_attn_image` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Cross-Attention to Image | `fused` |
| `transformer.encoder.layers.1.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.1.linear1` | `Linear` |  |  | `` |
| `transformer.encoder.layers.1.dropout` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.1.linear2` | `Linear` |  |  | `` |
| `transformer.encoder.layers.1.norm1` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.1.norm2` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.1.norm3` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.1.dropout1` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.1.dropout2` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.1.dropout3` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.2` | `TransformerEncoderLayer` | yes | Multimodal Decoder Layer | `fused` |
| `transformer.encoder.layers.2.self_attn` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Self-Attention | `fused` |
| `transformer.encoder.layers.2.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.2.cross_attn_image` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Cross-Attention to Image | `fused` |
| `transformer.encoder.layers.2.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.2.linear1` | `Linear` |  |  | `` |
| `transformer.encoder.layers.2.dropout` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.2.linear2` | `Linear` |  |  | `` |
| `transformer.encoder.layers.2.norm1` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.2.norm2` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.2.norm3` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.2.dropout1` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.2.dropout2` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.2.dropout3` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.3` | `TransformerEncoderLayer` | yes | Multimodal Decoder Layer | `fused` |
| `transformer.encoder.layers.3.self_attn` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Self-Attention | `fused` |
| `transformer.encoder.layers.3.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.3.cross_attn_image` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Cross-Attention to Image | `fused` |
| `transformer.encoder.layers.3.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.3.linear1` | `Linear` |  |  | `` |
| `transformer.encoder.layers.3.dropout` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.3.linear2` | `Linear` |  |  | `` |
| `transformer.encoder.layers.3.norm1` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.3.norm2` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.3.norm3` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.3.dropout1` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.3.dropout2` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.3.dropout3` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.4` | `TransformerEncoderLayer` | yes | Multimodal Decoder Layer | `fused` |
| `transformer.encoder.layers.4.self_attn` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Self-Attention | `fused` |
| `transformer.encoder.layers.4.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.4.cross_attn_image` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Cross-Attention to Image | `fused` |
| `transformer.encoder.layers.4.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.4.linear1` | `Linear` |  |  | `` |
| `transformer.encoder.layers.4.dropout` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.4.linear2` | `Linear` |  |  | `` |
| `transformer.encoder.layers.4.norm1` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.4.norm2` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.4.norm3` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.4.dropout1` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.4.dropout2` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.4.dropout3` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.5` | `TransformerEncoderLayer` | yes | Multimodal Decoder Layer | `fused` |
| `transformer.encoder.layers.5.self_attn` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Self-Attention | `fused` |
| `transformer.encoder.layers.5.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.5.cross_attn_image` | `MultiheadAttentionWrapper` | yes | Multimodal Decoder Cross-Attention to Image | `fused` |
| `transformer.encoder.layers.5.cross_attn_image.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.encoder.layers.5.linear1` | `Linear` |  |  | `` |
| `transformer.encoder.layers.5.dropout` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.5.linear2` | `Linear` |  |  | `` |
| `transformer.encoder.layers.5.norm1` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.5.norm2` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.5.norm3` | `LayerNorm` |  |  | `` |
| `transformer.encoder.layers.5.dropout1` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.5.dropout2` | `Dropout` |  |  | `` |
| `transformer.encoder.layers.5.dropout3` | `Dropout` |  |  | `` |
| `transformer.decoder` | `TransformerDecoder` | yes | Detector Decoder | `detector_queries` |
| `transformer.decoder.layers` | `ModuleList` |  |  | `` |
| `transformer.decoder.layers.0` | `TransformerDecoderLayer` | yes | Detector Decoder Layer | `detector_queries` |
| `transformer.decoder.layers.0.cross_attn` | `MultiheadAttentionWrapper` | yes | Detector Decoder Cross-Attention to Image | `detector_queries` |
| `transformer.decoder.layers.0.cross_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.0.dropout1` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.0.norm1` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.0.ca_text` | `MultiheadAttention` | yes | Detector Decoder Cross-Attention to Text | `detector_queries` |
| `transformer.decoder.layers.0.ca_text.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.0.catext_dropout` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.0.catext_norm` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.0.self_attn` | `MultiheadAttention` | yes | Detector Decoder Self-Attention | `detector_queries` |
| `transformer.decoder.layers.0.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.0.dropout2` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.0.norm2` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.0.linear1` | `Linear` |  |  | `` |
| `transformer.decoder.layers.0.dropout3` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.0.linear2` | `Linear` |  |  | `` |
| `transformer.decoder.layers.0.dropout4` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.0.norm3` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.1` | `TransformerDecoderLayer` | yes | Detector Decoder Layer | `detector_queries` |
| `transformer.decoder.layers.1.cross_attn` | `MultiheadAttentionWrapper` | yes | Detector Decoder Cross-Attention to Image | `detector_queries` |
| `transformer.decoder.layers.1.cross_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.1.dropout1` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.1.norm1` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.1.ca_text` | `MultiheadAttention` | yes | Detector Decoder Cross-Attention to Text | `detector_queries` |
| `transformer.decoder.layers.1.ca_text.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.1.catext_dropout` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.1.catext_norm` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.1.self_attn` | `MultiheadAttention` | yes | Detector Decoder Self-Attention | `detector_queries` |
| `transformer.decoder.layers.1.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.1.dropout2` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.1.norm2` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.1.linear1` | `Linear` |  |  | `` |
| `transformer.decoder.layers.1.dropout3` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.1.linear2` | `Linear` |  |  | `` |
| `transformer.decoder.layers.1.dropout4` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.1.norm3` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.2` | `TransformerDecoderLayer` | yes | Detector Decoder Layer | `detector_queries` |
| `transformer.decoder.layers.2.cross_attn` | `MultiheadAttentionWrapper` | yes | Detector Decoder Cross-Attention to Image | `detector_queries` |
| `transformer.decoder.layers.2.cross_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.2.dropout1` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.2.norm1` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.2.ca_text` | `MultiheadAttention` | yes | Detector Decoder Cross-Attention to Text | `detector_queries` |
| `transformer.decoder.layers.2.ca_text.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.2.catext_dropout` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.2.catext_norm` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.2.self_attn` | `MultiheadAttention` | yes | Detector Decoder Self-Attention | `detector_queries` |
| `transformer.decoder.layers.2.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.2.dropout2` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.2.norm2` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.2.linear1` | `Linear` |  |  | `` |
| `transformer.decoder.layers.2.dropout3` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.2.linear2` | `Linear` |  |  | `` |
| `transformer.decoder.layers.2.dropout4` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.2.norm3` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.3` | `TransformerDecoderLayer` | yes | Detector Decoder Layer | `detector_queries` |
| `transformer.decoder.layers.3.cross_attn` | `MultiheadAttentionWrapper` | yes | Detector Decoder Cross-Attention to Image | `detector_queries` |
| `transformer.decoder.layers.3.cross_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.3.dropout1` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.3.norm1` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.3.ca_text` | `MultiheadAttention` | yes | Detector Decoder Cross-Attention to Text | `detector_queries` |
| `transformer.decoder.layers.3.ca_text.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.3.catext_dropout` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.3.catext_norm` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.3.self_attn` | `MultiheadAttention` | yes | Detector Decoder Self-Attention | `detector_queries` |
| `transformer.decoder.layers.3.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.3.dropout2` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.3.norm2` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.3.linear1` | `Linear` |  |  | `` |
| `transformer.decoder.layers.3.dropout3` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.3.linear2` | `Linear` |  |  | `` |
| `transformer.decoder.layers.3.dropout4` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.3.norm3` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.4` | `TransformerDecoderLayer` | yes | Detector Decoder Layer | `detector_queries` |
| `transformer.decoder.layers.4.cross_attn` | `MultiheadAttentionWrapper` | yes | Detector Decoder Cross-Attention to Image | `detector_queries` |
| `transformer.decoder.layers.4.cross_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.4.dropout1` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.4.norm1` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.4.ca_text` | `MultiheadAttention` | yes | Detector Decoder Cross-Attention to Text | `detector_queries` |
| `transformer.decoder.layers.4.ca_text.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.4.catext_dropout` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.4.catext_norm` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.4.self_attn` | `MultiheadAttention` | yes | Detector Decoder Self-Attention | `detector_queries` |
| `transformer.decoder.layers.4.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.4.dropout2` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.4.norm2` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.4.linear1` | `Linear` |  |  | `` |
| `transformer.decoder.layers.4.dropout3` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.4.linear2` | `Linear` |  |  | `` |
| `transformer.decoder.layers.4.dropout4` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.4.norm3` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.5` | `TransformerDecoderLayer` | yes | Detector Decoder Layer | `detector_queries` |
| `transformer.decoder.layers.5.cross_attn` | `MultiheadAttentionWrapper` | yes | Detector Decoder Cross-Attention to Image | `detector_queries` |
| `transformer.decoder.layers.5.cross_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.5.dropout1` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.5.norm1` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.5.ca_text` | `MultiheadAttention` | yes | Detector Decoder Cross-Attention to Text | `detector_queries` |
| `transformer.decoder.layers.5.ca_text.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.5.catext_dropout` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.5.catext_norm` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.5.self_attn` | `MultiheadAttention` | yes | Detector Decoder Self-Attention | `detector_queries` |
| `transformer.decoder.layers.5.self_attn.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `transformer.decoder.layers.5.dropout2` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.5.norm2` | `LayerNorm` |  |  | `` |
| `transformer.decoder.layers.5.linear1` | `Linear` |  |  | `` |
| `transformer.decoder.layers.5.dropout3` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.5.linear2` | `Linear` |  |  | `` |
| `transformer.decoder.layers.5.dropout4` | `Dropout` |  |  | `` |
| `transformer.decoder.layers.5.norm3` | `LayerNorm` |  |  | `` |
| `transformer.decoder.norm` | `LayerNorm` |  |  | `` |
| `transformer.decoder.bbox_embed` | `MLP` | yes | Box Head | `final_outputs` |
| `transformer.decoder.bbox_embed.layers` | `ModuleList` |  |  | `` |
| `transformer.decoder.bbox_embed.layers.0` | `Linear` |  |  | `` |
| `transformer.decoder.bbox_embed.layers.1` | `Linear` |  |  | `` |
| `transformer.decoder.bbox_embed.layers.2` | `Linear` |  |  | `` |
| `transformer.decoder.bbox_embed.drop` | `Identity` |  |  | `` |
| `transformer.decoder.bbox_embed.out_norm` | `Identity` |  |  | `` |
| `transformer.decoder.query_embed` | `Embedding` |  |  | `` |
| `transformer.decoder.reference_points` | `Embedding` |  |  | `` |
| `transformer.decoder.boxRPB_embed_x` | `MLP` |  |  | `` |
| `transformer.decoder.boxRPB_embed_x.layers` | `ModuleList` |  |  | `` |
| `transformer.decoder.boxRPB_embed_x.layers.0` | `Linear` |  |  | `` |
| `transformer.decoder.boxRPB_embed_x.layers.1` | `Linear` |  |  | `` |
| `transformer.decoder.boxRPB_embed_x.drop` | `Identity` |  |  | `` |
| `transformer.decoder.boxRPB_embed_x.out_norm` | `Identity` |  |  | `` |
| `transformer.decoder.boxRPB_embed_y` | `MLP` |  |  | `` |
| `transformer.decoder.boxRPB_embed_y.layers` | `ModuleList` |  |  | `` |
| `transformer.decoder.boxRPB_embed_y.layers.0` | `Linear` |  |  | `` |
| `transformer.decoder.boxRPB_embed_y.layers.1` | `Linear` |  |  | `` |
| `transformer.decoder.boxRPB_embed_y.drop` | `Identity` |  |  | `` |
| `transformer.decoder.boxRPB_embed_y.out_norm` | `Identity` |  |  | `` |
| `transformer.decoder.presence_token` | `Embedding` |  |  | `` |
| `transformer.decoder.presence_token_head` | `MLP` |  |  | `` |
| `transformer.decoder.presence_token_head.layers` | `ModuleList` |  |  | `` |
| `transformer.decoder.presence_token_head.layers.0` | `Linear` |  |  | `` |
| `transformer.decoder.presence_token_head.layers.1` | `Linear` |  |  | `` |
| `transformer.decoder.presence_token_head.layers.2` | `Linear` |  |  | `` |
| `transformer.decoder.presence_token_head.drop` | `Identity` |  |  | `` |
| `transformer.decoder.presence_token_head.out_norm` | `Identity` |  |  | `` |
| `transformer.decoder.presence_token_out_norm` | `LayerNorm` |  |  | `` |
| `transformer.decoder.ref_point_head` | `MLP` |  |  | `` |
| `transformer.decoder.ref_point_head.layers` | `ModuleList` |  |  | `` |
| `transformer.decoder.ref_point_head.layers.0` | `Linear` |  |  | `` |
| `transformer.decoder.ref_point_head.layers.1` | `Linear` |  |  | `` |
| `transformer.decoder.ref_point_head.drop` | `Identity` |  |  | `` |
| `transformer.decoder.ref_point_head.out_norm` | `Identity` |  |  | `` |
| `segmentation_head` | `UniversalSegmentationHead` | yes | Segmentation Head | `mask_head` |
| `segmentation_head.pixel_decoder` | `PixelDecoder` | yes | Pixel Decoder | `pixel_features` |
| `segmentation_head.pixel_decoder.conv_layers` | `ModuleList` |  |  | `` |
| `segmentation_head.pixel_decoder.conv_layers.0` | `Conv2d` |  |  | `` |
| `segmentation_head.pixel_decoder.conv_layers.1` | `Conv2d` |  |  | `` |
| `segmentation_head.pixel_decoder.conv_layers.2` | `Conv2d` |  |  | `` |
| `segmentation_head.pixel_decoder.norms` | `ModuleList` |  |  | `` |
| `segmentation_head.pixel_decoder.norms.0` | `GroupNorm` |  |  | `` |
| `segmentation_head.pixel_decoder.norms.1` | `GroupNorm` |  |  | `` |
| `segmentation_head.pixel_decoder.norms.2` | `GroupNorm` |  |  | `` |
| `segmentation_head.mask_predictor` | `MaskPredictor` |  |  | `` |
| `segmentation_head.mask_predictor.mask_embed` | `MLP` |  |  | `` |
| `segmentation_head.mask_predictor.mask_embed.layers` | `ModuleList` |  |  | `` |
| `segmentation_head.mask_predictor.mask_embed.layers.0` | `Linear` |  |  | `` |
| `segmentation_head.mask_predictor.mask_embed.layers.1` | `Linear` |  |  | `` |
| `segmentation_head.mask_predictor.mask_embed.layers.2` | `Linear` |  |  | `` |
| `segmentation_head.mask_predictor.mask_embed.drop` | `Identity` |  |  | `` |
| `segmentation_head.mask_predictor.mask_embed.out_norm` | `Identity` |  |  | `` |
| `segmentation_head.cross_attend_prompt` | `MultiheadAttentionWrapper` |  |  | `` |
| `segmentation_head.cross_attend_prompt.out_proj` | `NonDynamicallyQuantizableLinear` |  |  | `` |
| `segmentation_head.cross_attn_norm` | `LayerNorm` |  |  | `` |
| `segmentation_head.semantic_seg_head` | `Conv2d` | yes | Semantic Head | `semantic` |
| `segmentation_head.instance_seg_head` | `Conv2d` | yes | Mask Predictor | `mask_head` |
| `dot_prod_scoring` | `DotProductScoring` |  |  | `` |
| `dot_prod_scoring.prompt_mlp` | `MLP` |  |  | `` |
| `dot_prod_scoring.prompt_mlp.layers` | `ModuleList` |  |  | `` |
| `dot_prod_scoring.prompt_mlp.layers.0` | `Linear` |  |  | `` |
| `dot_prod_scoring.prompt_mlp.layers.1` | `Linear` |  |  | `` |
| `dot_prod_scoring.prompt_mlp.drop` | `Dropout` |  |  | `` |
| `dot_prod_scoring.prompt_mlp.out_norm` | `LayerNorm` |  |  | `` |
| `dot_prod_scoring.prompt_proj` | `Linear` |  |  | `` |
| `dot_prod_scoring.hs_proj` | `Linear` |  |  | `` |
