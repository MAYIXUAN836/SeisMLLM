#!/usr/bin/env python3
"""Prepare public SeisMLLM conversations and launch Qwen2.5-VL LoRA SFT."""

import argparse
import json
import subprocess
from pathlib import Path


def prepare_data(dataset_dir, output_dir):
    dataset_dir = Path(dataset_dir).resolve()
    records = json.loads((dataset_dir / "train.json").read_text(encoding="utf-8"))
    for item in records:
        paths = []
        for value in item["images"]:
            path = Path(value)
            if not path.is_absolute():
                relative = value.replace("\\", "/")
                if relative.startswith("Dataset/"):
                    relative = relative[len("Dataset/"):]
                path = dataset_dir / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(str(path.resolve()))
        user_text = "\n".join(m["content"] for m in item["messages"] if m["role"] == "user")
        if user_text.count("<image>") != len(paths):
            raise ValueError("Image placeholder count does not match the case's images")
        item["images"] = paths
    prepared_dir = Path(output_dir).resolve() / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / "train.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    registration = {
        "seismllm_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    }
    (prepared_dir / "dataset_info.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    return prepared_dir, len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="Dataset")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", default="saves/seismllm")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepared_dir, count = prepare_data(args.dataset_dir, args.output_dir)
    command = [
        "llamafactory-cli", "train", "--stage", "sft", "--do_train",
        "--model_name_or_path", args.model,
        "--dataset", "seismllm_train", "--dataset_dir", str(prepared_dir),
        "--overwrite_cache", "--template", "qwen2_vl", "--finetuning_type", "lora",
        "--lora_target", "all", "--output_dir", str(Path(args.output_dir).resolve()),
        "--warmup_ratio", "0.1", "--weight_decay", "0.1",
        "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "32", "--learning_rate", "2e-5",
        "--num_train_epochs", "4.0", "--logging_steps", "10", "--save_steps", "20",
        "--save_total_limit", "3", "--val_size", "0.1", "--eval_strategy", "steps",
        "--eval_steps", "20", "--load_best_model_at_end", "--plot_loss", "--bf16",
        "--cutoff_len", "10240", "--dataloader_num_workers", "0",
        "--preprocessing_num_workers", "1", "--ddp_timeout", "180000000",
        "--lora_rank", "32", "--lora_alpha", "64",
    ]
    print(f"Prepared {count} training cases in {prepared_dir}")
    if args.dry_run:
        print(json.dumps(command, indent=2))
    else:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
