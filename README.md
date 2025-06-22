# Test Assertion Generation via Knowledge Distillation

This repository contains the replication package for our paper on test assertion generation using knowledge distillation. It includes tools for training smaller, more efficient models while maintaining the performance of larger teacher models.

## Repository Structure

- `knowledge_distillation.py`: Main training script for distilling knowledge from a teacher model to a student model
- `benchmark_performance.py`: Benchmarking script to measure inference speed and memory usage
- `evaluate_student.py`: Evaluation script for student and teacher models, reporting precision/recall/F1, CodeBLEU, AST validity, token accuracy, and generating summary plots
- `requirements.txt`: Python dependencies

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your data in JSONL format with the following fields:
   - `focal_method`: The source code method being tested
   - `test_method_masked`: The test method with assertions masked out
   - `assertions`: Ground truth assertions
   - `predicted_assertions`: Teacher model predictions
   - `compressed_logits`: Compressed logits from teacher model

## Usage

### Training (Knowledge Distillation)

```bash
python knowledge_distillation.py \
    --data_path path/to/your/data.jsonl \
    --model_output_dir my_students \
    --teacher_name Salesforce/codet5p-770m \
    --student_name Salesforce/codet5p-220m \
    --batch_size 4 \
    --epochs 4
```

Key parameters:

- `temperature`: Controls softness of teacher distribution (default: 1.0)
- `weight_teacher_ce`: Weight for teacher's CE loss (default: 0.3)
- `weight_gold_ce`: Weight for gold CE loss (default: 0.4)

### Benchmarking

To evaluate model performance:

```bash
python benchmark_performance.py \
    --data_path path/to/test/data.jsonl \
    --model_dir path/to/trained/model \
    --batch_size 1
```

The benchmark script will report:

- Average inference speed per method
- Average inference speed per assertion
- Peak memory usage

### Evaluation

To evaluate a distilled student model and compare with teacher predictions:

```bash
python evaluate_student.py \
    --data_path path/to/test/data.jsonl \
    --model_dir path/to/trained/model \
    --output_dir results/ \
    --batch_size 8
```

This script computes:

- Micro-averaged precision, recall, F1
- Per-example accuracy (macro)
- Per-assertion similarity (micro)
- Per-example CodeBLEU (macro)
- Per-assertion AST validity (micro)
- Per-assertion token accuracy (micro)
- Summary visualizations (histograms and bar plots)

### Notes

- For CodeBLEU support, install the `codebleu` package and its dependencies.
- For Java AST parsing, install `javalang`.

## Hardware Requirements

- Training: GPU recommended (CUDA-capable) with at least 16GB VRAM
- Inference: Can run on CPU, but GPU recommended for faster inference
- RAM: Minimum 16GB recommended

## Citation

If you use this code in your research, please cite our paper:

```bibtex
[Citation will be added upon paper publication]
```

## Model Metrics

Evaluation metrics for teacher and student models are saved in the `metrics` folder after running `evaluate_student.py`. You will find:

- `metrics/<experiment_name>/student_metrics.json`: Overall metrics for the student model for each experiment
- (If available) `metrics/<experiment_name>/teacher_metrics.json`: Overall metrics for the teacher predictions for each experiment

Each `student_metrics.json` contains fields such as:

```json
{
  "precision": 0.3,
  "recall": 0.29,
  "f1": 0.29,
  "accuracy": 0.32,
  "avg_similarity": 0.77,
  "avg_codebleu": 0.45,
  "avg_ast_validity": 0.87,
  "avg_token_accuracy": 0.42
}
```

You can compare different experiments (e.g., different distillation weights or model variants) by inspecting the corresponding subfolders and their metrics files in `metrics/`.

## License

_No license specified. Please contact the authors for usage permissions._

## Contact
For questions or issues, please open an issue on this repository or contact the authors via email.

## Acknowledgements
This work builds upon the CodeT5/CodeT5+ models and the Hugging Face Transformers library. We thank the authors of these projects for their contributions to the NLP community.