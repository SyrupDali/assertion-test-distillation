#!/usr/bin/env python
"""
Script to generate predictions from a distilled student T5 model on Java test assertion generation,
evaluate them against ground truth, and produce summary metrics and visualizations.
Supports evaluating on a subset by specifying --max_samples.
"""
import argparse
import json
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(jsonl_path):
    """Load data from JSONL file"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def normalize_assertion(assertion):
    """Normalize assertion text for comparison"""
    assertion = re.sub(r"\s+", " ", assertion).strip()
    assertion = re.sub(
        r"assertEquals\(\s*[^,]+,\s*([^)]+)\)",
        r"assertEquals(VALUE, \1)", assertion
    )
    assertion = re.sub(
        r"assert(Equals|That|True|False)",
        lambda m: 'assert' + m.group(1),
        assertion, flags=re.IGNORECASE
    )
    return assertion


def calculate_similarity(reference, candidate):
    """Calculate string similarity using SequenceMatcher"""
    return SequenceMatcher(None, reference, candidate).ratio()


def classify_assertion_type(assertion):
    """Classify the type of assertion"""
    a = assertion.lower()
    if 'assertequals' in a or ('assertthat' in a and '.isequalto' in a):
        return 'equality'
    if 'asserttrue' in a:
        return 'truth'
    if 'assertfalse' in a:
        return 'falsity'
    if 'assertnull' in a:
        return 'null'
    if 'assertnotnull' in a:
        return 'not_null'
    if 'assertthrows' in a:
        return 'exception'
    if 'assertsame' in a and 'assertnotsame' not in a:
        return 'same'
    if 'assertnotsame' in a:
        return 'not_same'
    if 'assertarrayequals' in a:
        return 'array_equality'
    return 'other'


def evaluate_assertions(generated, reference):
    """Evaluate generated assertions against reference assertions"""
    if isinstance(generated, str):
        gen_list = [s.strip() + ';' for s in re.split(r';|\n', generated) if s.strip()]
    else:
        gen_list = generated
    if isinstance(reference, str):
        ref_list = [s.strip() + ';' for s in re.split(r';|\n', reference) if s.strip()]
    else:
        ref_list = reference

    gen_norm = [normalize_assertion(a) for a in gen_list]
    ref_norm = [normalize_assertion(a) for a in ref_list]

    exact = sum(1 for g in gen_norm if g in ref_norm)
    sim_scores = []
    gen_types = []
    ref_types = []
    for g in gen_list:
        best = 0.0
        for r in ref_list:
            best = max(best, calculate_similarity(r, g))
        sim_scores.append(best)
        gen_types.append(classify_assertion_type(g))
    for r in ref_list:
        ref_types.append(classify_assertion_type(r))

    precision = exact / len(gen_norm) if gen_norm else 0.0
    recall = exact / len(ref_norm) if ref_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = exact / max(len(gen_norm), len(ref_norm)) if max(len(gen_norm), len(ref_norm)) > 0 else 0.0
    avg_similarity = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0

    gen_counts = {t: gen_types.count(t) for t in set(gen_types)}
    ref_counts = {t: ref_types.count(t) for t in set(ref_types)}

    return {
        'exact_matches': exact,
        'generated_count': len(gen_norm),
        'reference_count': len(ref_norm),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'similarity_score_avg': avg_similarity,
        'similarity_scores': sim_scores,
        'generated_type_counts': gen_counts,
        'reference_type_counts': ref_counts
    }


class AssertionDataset(Dataset):
    """Dataset for assertion generation"""
    def __init__(self, data, tokenizer, max_src_length=1024, max_tgt_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"FOCAL CODE:\n{item['focal_file']}\n\nTEST METHOD:\n{item['test_method_masked']}"
        target_text = "\n".join(item['assertions'])

        src = self.tokenizer(
            input_text,
            max_length=self.max_src_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        tgt = self.tokenizer(
            target_text,
            max_length=self.max_tgt_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        input_ids = src['input_ids'].squeeze(0)
        attention_mask = src['attention_mask'].squeeze(0)
        labels = tgt['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'original_entry': item
        }


def collate_fn(batch):
    """Custom collate to batch tensors and preserve list of originals"""
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    originals = [b['original_entry'] for b in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'original_entry': originals
    }


def generate_and_evaluate(model, tokenizer, dataloader, device, output_file):
    """Generate predictions, evaluate metrics, and save results."""
    model.to(device)
    model.eval()

    all_entries = []
    all_metrics = {
        'exact_matches': 0,
        'generated_count': 0,
        'reference_count': 0,
        'similarity_scores': [],
        'accuracy_scores': [],
        'f1_scores': []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating'):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            originals = batch['original_entry']

            generated_ids = model.generate(
                input_ids=inputs,
                attention_mask=masks,
                max_length=dataloader.dataset.max_tgt_length,
                num_beams=4,
                early_stopping=True
            )

            for i in range(len(inputs)):
                entry = originals[i].copy()
                gen_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                metrics = evaluate_assertions(gen_text, entry['assertions'])

                all_metrics['exact_matches'] += metrics['exact_matches']
                all_metrics['generated_count'] += metrics['generated_count']
                all_metrics['reference_count'] += metrics['reference_count']
                all_metrics['similarity_scores'].extend(metrics['similarity_scores'])
                all_metrics['accuracy_scores'].append(metrics['accuracy'])
                all_metrics['f1_scores'].append(metrics['f1'])

                record = entry.copy()
                record['model_prediction'] = gen_text
                record['prediction_metrics'] = {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy'],
                    'avg_similarity': metrics['similarity_score_avg']
                }
                all_entries.append(record)

    with open(output_file, 'w') as f:
        for r in all_entries:
            f.write(json.dumps(r) + '\n')

    if all_metrics['generated_count'] > 0 and all_metrics['reference_count'] > 0:
        precision = all_metrics['exact_matches'] / all_metrics['generated_count']
        recall = all_metrics['exact_matches'] / all_metrics['reference_count']
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = all_metrics['exact_matches'] * 2 / (all_metrics['generated_count'] + all_metrics['reference_count'])
    else:
        precision = recall = f1 = accuracy = 0.0
    sim_avg = np.mean(all_metrics['similarity_scores']) if all_metrics['similarity_scores'] else 0.0

    final_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'avg_similarity': sim_avg
    }
    metrics_file = os.path.join(os.path.dirname(output_file), 'student_prediction_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    vis_dir = os.path.join(os.path.dirname(output_file), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(final_metrics.keys()), y=list(final_metrics.values()))
    plt.ylim(0, 1)
    plt.title('Student Assertion Metrics')
    plt.savefig(os.path.join(vis_dir, 'student_overall_metrics.png'))
    plt.close()

    for metric_name, data in [
        ('similarity', all_metrics['similarity_scores']),
        ('accuracy', all_metrics['accuracy_scores']),
        ('f1', all_metrics['f1_scores'])
    ]:
        plt.figure(figsize=(6, 4))
        sns.histplot(data, bins=20, kde=True)
        plt.title(f'Distribution of {metric_name.capitalize()} Scores')
        plt.xlabel(metric_name.capitalize())
        plt.ylabel('Count')
        plt.savefig(os.path.join(vis_dir, f'student_{metric_name}_distribution.png'))
        plt.close()

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Generate and evaluate student model assertions')
    parser.add_argument('--data_path', type=str, required=True, help='Path to JSONL dataset')
    parser.add_argument('--model_dir', type=str, default='distilled_codet5_student', help='Student model dir')
    parser.add_argument('--output_dir', type=str, default='student_output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_src_length', type=int, default=1024, help='Max source length')
    parser.add_argument('--max_tgt_length', type=int, default=512, help='Max target length')
    parser.add_argument('--max_samples', type=int, default=None, help='Max number of examples to evaluate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_file = os.path.join(args.output_dir, 'student_predictions.jsonl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data = load_dataset(args.data_path)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f'Loaded {len(data)} examples')

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)

    dataset = AssertionDataset(data, tokenizer,
                               max_src_length=args.max_src_length,
                               max_tgt_length=args.max_tgt_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    metrics = generate_and_evaluate(model, tokenizer, loader, device, predictions_file)
    print('Final student metrics:')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
    
# python generate_student_predictions.py \
#   --data_path dataset_with_predictions.jsonl \
#   --model_dir distilled_codet5_student \
#   --output_dir student_output \
#   --max_samples 100
