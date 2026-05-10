#!/usr/bin/env bash
# Example: run Prompt Activation Mapping on a single image.
#
# Prerequisites:
#   1. SAM 3 installed (see README.md)
#   2. This package installed: pip install -e ".[dev]"
#   3. HuggingFace authentication: hf auth login
#   4. An image at examples/bus.jpg (or change the path below)

set -euo pipefail

IMAGE="${1:-examples/bus.jpg}"
PROMPT="${2:-yellow school bus}"
OUTPUT_DIR="${3:-outputs/bus_pam}"
DEVICE="${4:-cuda}"

echo "=== Prompt Activation Mapping for SAM 3 ==="
echo "Image : $IMAGE"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Step 1: list available attention modules (optional, for debugging)
echo "--- Available attention modules ---"
python -m pam.cli --list-attention-modules --device "$DEVICE"
echo ""

# Step 2: generate heatmaps
echo "--- Generating heatmaps ---"
python -m pam.cli \
    --image "$IMAGE" \
    --prompt "$PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --save-debug

echo ""
echo "Done. Check $OUTPUT_DIR for output files."
