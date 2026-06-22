from __future__ import annotations

import argparse
import json
from pathlib import Path

from pam.sam3_loader import load_sam3_image_model


def safe(obj):
    try:
        return str(obj)
    except Exception as e:
        return repr(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / "inspect_language_tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_sam3_image_model(device=args.device)

    tokenizer = model.backbone.language_backbone.tokenizer

    print("TOKENIZER TYPE:", type(tokenizer))
    print("\nTOKENIZER DICT KEYS:")
    print(getattr(tokenizer, "__dict__", {}).keys())

    print("\nTOKENIZER ATTRS:")
    for name in dir(tokenizer):
        low = name.lower()
        if (
            "token" in low
            or "encode" in low
            or "decode" in low
            or "bpe" in low
            or "vocab" in low
            or "encoder" in low
            or "decoder" in low
        ):
            print(name)

    outputs = {
        "prompt": args.prompt,
        "tokenizer_type": str(type(tokenizer)),
    }

    for method_name in ["encode", "decode", "tokenize"]:
        if hasattr(tokenizer, method_name):
            print(f"\nTrying tokenizer.{method_name}...")
            method = getattr(tokenizer, method_name)
            try:
                if method_name == "decode":
                    continue
                result = method(args.prompt)
                print(result)
                outputs[method_name] = safe(result)
            except Exception as e:
                print("FAILED:", repr(e))
                outputs[f"{method_name}_error"] = repr(e)

    # Try common CLIP-style call
    try:
        result = tokenizer(args.prompt)
        print("\ntokenizer(prompt):")
        print(result)
        outputs["call"] = safe(result)
    except Exception as e:
        print("\ntokenizer(prompt) FAILED:", repr(e))
        outputs["call_error"] = repr(e)

    # If encode returns ids, decode each id if possible
    try:
        ids = tokenizer.encode(args.prompt)
        outputs["ids"] = [int(x) for x in ids]
        print("\nIDS:", ids)

        if hasattr(tokenizer, "decode"):
            decoded = []
            for i in ids:
                try:
                    decoded.append(tokenizer.decode([int(i)]))
                except Exception as e:
                    decoded.append(f"ERR:{repr(e)}")
            outputs["decoded_each_id"] = decoded
            print("DECODED EACH ID:", decoded)
    except Exception as e:
        outputs["ids_error"] = repr(e)

    with open(out_dir / "language_tokenizer_inspection.json", "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"\nSaved: {out_dir / 'language_tokenizer_inspection.json'}")


if __name__ == "__main__":
    main()