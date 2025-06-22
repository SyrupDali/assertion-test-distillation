#!/usr/bin/env python
"""
Benchmark a student model's inference speed and (CPU) memory usage on macOS.
This variant forces CPU-only execution for more stable peak-RSS measurements.
"""
import argparse
import json
import os
import re
import sys
import torch
import time
import resource
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# --- Simplified Dataset Loading ---
class AssertionDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, max_src=1024):
        self.data = data
        self.tok = tokenizer
        self.max_src = max_src

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        e = self.data[i]
        src = f"FOCAL METHOD:\n{e['focal_method']}\n\nTEST METHOD:\n{e['test_method_masked']}"
        tokens = self.tok(
            src,
            padding="max_length",
            truncation=True,
            max_length=self.max_src,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
        }

def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }

# --- Core Performance Measurement ---
def measure_performance(model, tok, loader, device):
    model.to(device).eval()

    total_inference_time = 0.0
    total_methods = 0
    total_assertions = 0

    print("\nRunning inference (CPU) to measure performance...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Benchmarking"):
            inp = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)

            start = time.perf_counter()
            gen_ids = model.generate(
                inp,
                attention_mask=msk,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            end = time.perf_counter()

            total_inference_time += (end - start)
            total_methods += len(inp)

            gen_lists = [
                [s.strip() for s in re.split(r";|\n", tok.decode(g, skip_special_tokens=True)) if s.strip()]
                for g in gen_ids
            ]
            total_assertions += sum(len(gl) for gl in gen_lists)

    avg_method = total_inference_time / total_methods if total_methods else 0.0
    avg_assert = total_inference_time / total_assertions if total_assertions else 0.0

    peak_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes
    peak_mb = peak_raw / (1024 ** 2)

    return {
        "avg_speed_per_method_sec": avg_method,
        "avg_speed_per_assertion_sec": avg_assert,
        "peak_memory_mb": peak_mb,
        "total_methods": total_methods
    }

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_name", default='Salesforce/codet5p-770m')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Force CPU
    device = torch.device("cpu")
    print(f"Device: {device} (CPU-only for stable memory measure)")

    # Load data
    with open(args.data_path) as f:
        all_data = [json.loads(l) for l in f if l.strip() and not l.strip().startswith('{"header"')]
    data = all_data[:args.max_samples] if args.max_samples else all_data
    print(f"Loaded {len(data)} examples (max_samples={args.max_samples})")

    print(f"Loading tokenizer from '{args.tokenizer_name}'...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print(f"Loading model from '{args.model_dir}'...")
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    ds = AssertionDataset(data, tok, max_src=1024)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    metrics = measure_performance(model, tok, dl, device)

    print("\n--- Performance Benchmark Results ---")
    print(f"Model: {args.model_dir}")
    print(f"Total methods benchmarked: {metrics['total_methods']}")
    print("-----------------------------------")
    print(f"Average inference speed per method: {metrics['avg_speed_per_method_sec']:.4f} sec")
    print(f"Average inference speed per assertion: {metrics['avg_speed_per_assertion_sec']:.4f} sec")
    print(f"Peak memory usage (RSS): {metrics['peak_memory_mb']:.2f} MB")
    print("-----------------------------------")

if __name__ == "__main__":
    main()
