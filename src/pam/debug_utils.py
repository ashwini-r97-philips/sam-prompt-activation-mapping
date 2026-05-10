"""Debug-artifact serialisation (JSON + NPZ).

Every run can optionally save:
* ``debug.json`` — human-readable metadata.
* ``debug_tensors.npz`` — raw tensors moved to CPU as float32 numpy arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def save_debug_json(
    output_dir: str | Path,
    *,
    prompt: str,
    image_path: str,
    selected_module_name: str,
    selected_module_repr: str,
    all_candidate_names: list[str],
    raw_attention_shape: list[int],
    value_tensor_shape: list[int],
    normalised_attention_shape: list[int],
    normalised_value_shape: list[int],
    num_prompt_tokens: int,
    num_image_tokens: int,
    inferred_spatial_grid: tuple[int, int] | None,
    token_labels: list[str],
    output_file_paths: list[dict],
    warnings: list[str],
) -> Path:
    """Write ``debug.json`` to *output_dir*.

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "debug.json"

    data: Dict[str, Any] = {
        "prompt": prompt,
        "image_path": str(image_path),
        "selected_module_name": selected_module_name,
        "selected_module_repr": selected_module_repr[:500],
        "all_candidate_module_names": all_candidate_names,
        "raw_attention_shape": raw_attention_shape,
        "value_tensor_shape": value_tensor_shape,
        "normalised_attention_shape": normalised_attention_shape,
        "normalised_value_shape": normalised_value_shape,
        "num_prompt_tokens": num_prompt_tokens,
        "num_image_tokens": num_image_tokens,
        "inferred_spatial_grid": list(inferred_spatial_grid) if inferred_spatial_grid else None,
        "token_labels": token_labels,
        "output_file_paths": output_file_paths,
        "warnings": warnings,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def save_debug_tensors(
    output_dir: str | Path,
    *,
    raw_attention: torch.Tensor | None = None,
    effective_attention: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    query: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
) -> Path:
    """Write ``debug_tensors.npz`` with CPU float32 numpy arrays.

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "debug_tensors.npz"

    arrays: Dict[str, np.ndarray] = {}
    for name, tensor in [
        ("raw_attention", raw_attention),
        ("effective_attention", effective_attention),
        ("values", values),
        ("query", query),
        ("key", key),
    ]:
        if tensor is not None:
            arrays[name] = tensor.detach().cpu().float().numpy()

    np.savez(str(path), **arrays)
    return path
