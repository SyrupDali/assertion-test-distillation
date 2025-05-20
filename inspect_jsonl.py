#!/usr/bin/env python3
"""
Quick script to inspect the structure of a JSONL dataset.
Prints the keys and sample values from the first few examples.
"""

import json
import argparse

def inspect_jsonl(path, num_examples=3):
    print(f"Inspecting: {path}\n")
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {i+1} is not valid JSON: {e}")
                continue

            print(f"--- Example {i+1} ---")
            for key, value in obj.items():
                preview = str(value)
                preview = preview.replace("\n", "\\n")
                if len(preview) > 120:
                    preview = preview[:120] + "..."
                print(f"{key}: {preview}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the JSONL file to inspect")
    parser.add_argument("--n", type=int, default=3, help="Number of examples to print")
    args = parser.parse_args()

    inspect_jsonl(args.file, args.n)
