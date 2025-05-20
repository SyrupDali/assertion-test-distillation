#!/usr/bin/env python
import json

def preview_predictions(file_path, num_examples=10):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            try:
                data = json.loads(line)
                print(f"\n--- Example {i+1} ---")
                print("FOCAL CODE snippet:",
                      data['focal_file'][:200].replace("\n", " ") + "...")
                print("Test method:",
                      data['test_method_masked'].replace("\n", " "))
                print("Ground truth assertions:", data['original_target'])
                print("Teacher prediction:", data.get('teacher_prediction', 'N/A'))
                print("Student prediction:", data.get('student_prediction', 'N/A'))
                print("Prediction metrics:", data.get('prediction_metrics', {}))
            except Exception as e:
                print(f"Error on line {i+1}: {e}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True,
                   help="Path to your JSONL prediction file")
    p.add_argument("--n",    type=int, default=10,
                   help="Number of examples to preview")
    args = p.parse_args()

    preview_predictions(args.file, args.n)
