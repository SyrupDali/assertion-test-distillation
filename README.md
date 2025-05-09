Created decompression test script. Run it with:
python ./teacher_output/decompression_test.py

Final Metrics:
  precision: 0.2238
  recall: 0.3173
  f1: 0.2624
  accuracy: 0.2624
  avg_per_sample_accuracy: 0.2837
  avg_similarity: 0.7154
  avg_per_sample_f1: 0.3056
  total_exact_matches: 16940
  total_generated: 75702
  total_reference: 53391
  compression_stats:
    raw_size_mb: 626953.1250
    compressed_size_mb: 17416.0494
    compression_ratio: 35.9986
    precision_bits: 4

Results saved to ./teacher_output
  - Predictions and metrics with compressed logits: ./teacher_output/dataset_with_predictions.jsonl
  - Overall metrics: ./teacher_output/prediction_metrics.json
  - Per-example metrics: ./teacher_output/teacher_predictions.csv
  - Visualizations: ./teacher_output/visualizations
  - Decompression test script: ./teacher_output/decompression_test.py