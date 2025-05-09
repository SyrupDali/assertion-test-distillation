import json

def preview_predictions(file_path, num_examples=10):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            try:
                data = json.loads(line)
                print(f"\n--- Example {i+1} ---")
                print("FOCAL CODE snippet:", data['focal_file'][:200].strip().replace("\n", " ") + "...")
                print("Test method:", data['test_method_masked'].strip().replace("\n", " "))
                print("Ground truth assertions:", data['assertions'])
                print("Model prediction:", data.get('model_prediction', 'N/A'))
                print("Prediction metrics:", data.get('prediction_metrics', {}))
            except Exception as e:
                print(f"Error reading line {i+1}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to your JSONL prediction file")
    parser.add_argument("--n", type=int, default=10, help="Number of examples to preview")
    args = parser.parse_args()

    preview_predictions(args.file, args.n)
