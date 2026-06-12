"""Prompt Activation Mapping (PAM) for SAM 3."""

# Install the torch.inference_mode bypass before anything else can import sam3.
# Must stay at the top of this file.
from . import _grad_bypass  # noqa: F401

__version__ = "0.1.0"
