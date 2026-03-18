"""Preprocess dataset for countdown task."""

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def make_prefix(dp, template_type):
    """Generate prompt prefix for countdown task."""
    target = dp["target"]
    numbers = dp["nums"]
    if template_type == "base":
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    else:
        raise ValueError(f"Unknown template type: {template_type}")
    return prefix


def main():
    parser = argparse.ArgumentParser(description="Preprocess countdown dataset")
    parser.add_argument("--local_dir", default="data/countdown", help="Local output directory")
    parser.add_argument("--train_size", type=int, default=327680, help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=1024, help="Number of test samples")
    parser.add_argument("--template_type", type=str, default="base", help="Prompt template type")

    args = parser.parse_args()

    # create output dir
    output_dir = Path(args.local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from HuggingFace: Jiayi-Pan/Countdown-Tasks-3to4")
    raw_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    print(f"Dataset size: {len(raw_dataset)}")
    assert len(raw_dataset) > args.train_size + args.test_size, "Dataset too small"

    train_dataset = raw_dataset.select(range(args.train_size))
    test_dataset = raw_dataset.select(range(args.train_size, args.train_size + args.test_size))

    print(f"Train split: {len(train_dataset)}, Test split: {len(test_dataset)}")

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {"target": example["target"], "numbers": example["nums"]}
            data = {
                "data_source": "countdown",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    print("Processing train dataset...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    print("Processing test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # save to parquet
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    print(f"Saving to {train_path}")
    train_dataset.to_parquet(str(train_path))

    print(f"Saving to {test_path}")
    test_dataset.to_parquet(str(test_path))

    print("Done!")


if __name__ == "__main__":
    main()
