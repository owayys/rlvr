"""Download Qwen3 models from HF."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3 models")
    parser.add_argument(
        "--model",
        required=True,
        choices=["0.6b", "1.7b"],
        help="Model size to download",
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Output directory for models",
    )

    args = parser.parse_args()

    model_map = {
        "0.6b": "Qwen/Qwen3-0.6B-Base",
        "1.7b": "Qwen/Qwen3-1.7B-Base",
    }

    model_id = model_map[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_output_dir = output_dir / model_id.replace("/", "_")

    print(f"Downloading {model_id} to {model_output_dir}")
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=str(model_output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Done! Model saved to {model_output_dir}")


if __name__ == "__main__":
    main()
