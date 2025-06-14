#!/usr/bin/env python
"""
A lightweight script to benchmark a student model's performance on a Mac M1.
It measures only inference speed and peak memory usage, minimizing the
overhead from other quality evaluation metrics.
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

# --- Simplified Dataset Loading (from the original evaluation script) ---

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

# --- Core Performance Measurement Function ---

def measure_performance(model, tok, loader, device):
    """
    Performs inference and measures speed and memory usage.
    """
    model.to(device).eval()

    # Performance metric accumulators
    total_inference_time = 0.0
    total_methods_processed = 0
    total_assertions_generated = 0

    print("\nRunning inference to measure performance...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Benchmarking"):
            inp = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)

            # Time the inference call
            start_time = time.perf_counter()
            gen_ids = model.generate(
                inp,
                attention_mask=msk,
                max_length=512, # A reasonable max length for generation
                num_beams=4,
                early_stopping=True
            )
            end_time = time.perf_counter()

            # Accumulate performance data
            total_inference_time += (end_time - start_time)
            total_methods_processed += len(inp)

            # We still need to decode to count the number of assertions generated
            gen_lists = [
                [s.strip() for s in re.split(r";|\n", tok.decode(g, skip_special_tokens=True)) if s.strip()]
                for g in gen_ids
            ]
            total_assertions_generated += sum(len(gl) for gl in gen_lists)

    # --- Calculate Final Performance Metrics ---

    avg_speed_per_method = total_inference_time / total_methods_processed if total_methods_processed > 0 else 0.0
    avg_speed_per_assertion = total_inference_time / total_assertions_generated if total_assertions_generated > 0 else 0.0

    # Get peak memory usage (platform-aware)
    peak_memory_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS
        # On macOS, ru_maxrss is in bytes. Divide by 1024*1024 to get MB.
        peak_memory_mb = peak_memory_raw / (1024 ** 2)
    else:  # Assuming Linux
        # On Linux, ru_maxrss is in kilobytes. Divide by 1024 to get MB.
        peak_memory_mb = peak_memory_raw / 1024

    return {
        "avg_speed_per_method_sec": avg_speed_per_method,
        "avg_speed_per_assertion_sec": avg_speed_per_assertion,
        "peak_memory_mb": peak_memory_mb,
        "total_methods_processed": total_methods_processed,
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark a student model for inference speed and memory.")
    parser.add_argument("--data_path", required=True, help="Path to the JSONL data file for input.")
    parser.add_argument("--model_dir", required=True, help="Path to the directory containing the student model.")
    parser.add_argument(
        "--tokenizer_name",
        default='Salesforce/codet5p-770m',
        help="The name of the tokenizer to use, typically the teacher's."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference. Use 1 for realistic single-example latency.")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to run for benchmarking.")
    args = parser.parse_args()

    # --- Setup Device (with M1 Pro support) ---
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")

    # --- Load Data and Model ---
    with open(args.data_path) as f:
        all_data = [json.loads(l) for l in f if l.strip() and not l.strip().startswith('{"header"')]

    data = all_data[:args.max_samples] if args.max_samples else all_data
    print(f"Loaded {len(data)} examples for benchmarking (max_samples={args.max_samples})")

    print(f"Loading tokenizer from '{args.tokenizer_name}'...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"Loading model from '{args.model_dir}'...")
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    ds = AssertionDataset(data, tok)
    # Use a batch size of 1 to measure single-example latency, but you can increase it
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Run Benchmark ---
    performance_metrics = measure_performance(model, tok, dl, device)

    # --- Print Results ---
    print("\n--- Performance Benchmark Results ---")
    print(f"Model: {args.model_dir}")
    print(f"Total methods benchmarked: {performance_metrics['total_methods_processed']}")
    print("-" * 35)
    print(f"Average inference speed per method: {performance_metrics['avg_speed_per_method_sec']:.4f} seconds")
    print(f"Average inference speed per assertion: {performance_metrics['avg_speed_per_assertion_sec']:.4f} seconds")
    print(f"Peak memory usage of the script: {performance_metrics['peak_memory_mb']:.2f} MB")
    print("-" * 35)


if __name__ == "__main__":
    main()