"""PAM Trace: non-invasive forward-hook tracer for SAM 3.

Records the major tensors flowing through the model during a single
``image + text prompt`` inference call without modifying SAM 3 source.

Public entry points:
    PamTracer                       -- coarse-grained module-level tracer
    write_outputs                   -- dump the 4 module-level trace files
    AttentionTracer                 -- attention-level tracer (10 patterns)
    write_attention_outputs         -- dump attention_trace.jsonl + summary.csv
    SelectedQueryCollector          -- per-query activation collector
    write_selected_query_outputs    -- write the 4 selected-query files
"""

from .tracer import PamTracer, write_outputs
from .attention_tracer import (
    AttentionTracer,
    force_attention_weights,
    write_attention_outputs,
)
from .selected_query import (
    SelectedQueryCollector,
    select_query,
    write_selected_query_outputs,
)

__all__ = [
    "PamTracer",
    "write_outputs",
    "AttentionTracer",
    "force_attention_weights",
    "write_attention_outputs",
    "SelectedQueryCollector",
    "select_query",
    "write_selected_query_outputs",
]
