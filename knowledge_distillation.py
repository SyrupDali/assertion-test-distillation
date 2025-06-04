#!/usr/bin/env python
"""
Script to distill a student T5 model on Java test assertion generation,
combining teacher‐prediction CE, gold CE, and soft KL losses with tunable weights.

Expects a JSONL where:
 - The first line may be a header (with key "header").
 - Each example has:
     "focal_method"                (str),
     "test_method_masked"          (str),
     "assertions"                  (list of str, newline/semicolon‐separated GT),
     "predicted_assertions"        (list of str, newline/semicolon‐separated teacher preds),
     "compressed_logits"           (dict, as written by compress_logits)
"""
import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW

from decompress_tensor import decompress_logits  # your utility for de‐compressing logits

LOG_STEPS = 10

class DistillationDataset(Dataset):
    def __init__(self, path, tokenizer, decoder_max_length, max_samples=None):
        self.tokenizer = tokenizer
        self.decoder_max_length = decoder_max_length
        self.entries = []
        with open(path, 'r') as f:
            for line in f:
                if max_samples and len(self.entries) >= max_samples:
                    break
                obj = json.loads(line)
                # skip header or malformed
                if 'header' in obj or 'focal_method' not in obj or 'test_method_masked' not in obj:
                    continue

                # Join the list of assertions into one newline‐separated string
                gold_list = obj.get('assertions', [])
                gold_str = "\n".join(gold_list) if isinstance(gold_list, list) else ""

                # Join teacher predictions likewise
                teacher_list = obj.get('predicted_assertions', [])
                teacher_str = "\n".join(teacher_list) if isinstance(teacher_list, list) else ""

                inp = (
                    f"FOCAL METHOD:\n{obj['focal_method']}\n\n"
                    f"TEST METHOD:\n{obj['test_method_masked']}"
                )

                self.entries.append({
                    'input':          inp,
                    'gold':           gold_str,
                    'teacher_pred':   teacher_str,
                    'teacher_logits': obj.get('compressed_logits')
                })
        print(f"> Loaded {len(self.entries)} examples")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rec = self.entries[idx]

        # Tokenize the input prompt ("FOCAL METHOD..." + "TEST METHOD...")
        enc = self.tokenizer(
            rec['input'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize the ground‐truth assertions (as one newline‐joined string)
        gold_enc = self.tokenizer(
            rec['gold'],
            max_length=self.decoder_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize the teacher predictions (as one newline‐joined string)
        tp_enc = self.tokenizer(
            rec['teacher_pred'],
            max_length=self.decoder_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Decompress teacher logits (dictionary as written by compress_tensor_optimized)
        t_logits = decompress_logits(rec['teacher_logits'])
        if t_logits is None:
            raise ValueError(f"Could not decompress logits for example {idx}")
        t_logits = t_logits.float()
        # If there is an extra batch dimension ([1, T, V]), squeeze it
        if t_logits.dim() == 3:
            t_logits = t_logits[0]  # shape → (T, V)

        # Align teacher logits to the decoder length L
        L = gold_enc['input_ids'].size(1)   # L = decoder_max_length
        V = t_logits.size(-1)               # vocabulary size
        T = t_logits.size(0)                # actual length of teacher logit sequence
        if T >= L:
            t_logits = t_logits[:L]
        else:
            pad_tensor = torch.full((L - T, V), -1e9, dtype=t_logits.dtype)
            t_logits = torch.cat([t_logits, pad_tensor], dim=0)  # now (L, V)

        labels_gold = gold_enc['input_ids'].squeeze(0)   # shape → (L,)
        labels_tp   = tp_enc['input_ids'].squeeze(0)     # shape → (L,)

        # Replace pad_token_id with -100 so that CrossEntropy ignores padding
        labels_gold[labels_gold == self.tokenizer.pad_token_id] = -100
        labels_tp  [labels_tp   == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids':      enc['input_ids'].squeeze(0),       # (512,)
            'attention_mask': enc['attention_mask'].squeeze(0),  # (512,)
            'raw_input':      rec['input'],                      # for debugging
            'raw_gold':       rec['gold'],                       # for debugging
            'raw_teacher':    rec['teacher_pred'],                # for debugging
            'gold_labels':    labels_gold,                        # (L,)
            'teacher_labels': labels_tp,                          # (L,)
            'teacher_logits': t_logits                            # (L, V)
        }

def collate_fn(batch):
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'raw_input':      [b['raw_input']      for b in batch],
        'raw_gold':       [b['raw_gold']       for b in batch],
        'raw_teacher':    [b['raw_teacher']    for b in batch],
        'gold_labels':    torch.stack([b['gold_labels']    for b in batch]),
        'teacher_labels': torch.stack([b['teacher_labels'] for b in batch]),
        'teacher_logits': torch.stack([b['teacher_logits'] for b in batch]),
    }

def distillation_loss(
    student_logits, teacher_logits,
    gold_labels, teacher_labels,
    temperature, w_teacher_ce, w_gold_ce
):
    """
    Computes combined loss = w_teacher_ce * CE(student_logits, teacher_labels)
                         + w_gold_ce    * CE(student_logits, gold_labels)
                         + (1 - w_teacher_ce - w_gold_ce) * KL(student_probs || teacher_probs)

    Args:
      student_logits: (B, L, V)
      teacher_logits: (B, L, V)
      gold_labels:    (B, L) with -100 for pad
      teacher_labels:(B, L) with -100 beyond teacher EOS
      temperature:     float (for scaling logits before KL)
      w_teacher_ce:    float in [0,1]
      w_gold_ce:       float in [0,1]

    Returns:
      A single scalar loss.
    """
    B, L, V = student_logits.shape

    # Flatten label masks → (B*L,)
    mask = (teacher_labels != -100).view(-1)  # True where teacher had a real token

    # Soft KL‐divergence on masked positions
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)   # (B, L, V)
    s_logp  = F.log_softmax(student_logits / temperature, dim=-1)
    t_flat = t_probs.view(-1, V)
    s_flat = s_logp.view(-1, V)
    if mask.sum() > 0:
        kl = F.kl_div(s_flat[mask], t_flat[mask], reduction='batchmean') * (temperature**2)
    else:
        kl = torch.tensor(0.0, device=student_logits.device)

    # Hard CE vs. teacher (using teacher_labels as “true” indices)
    bt = F.cross_entropy(
        student_logits.view(-1, V),
        teacher_labels.view(-1),
        ignore_index=-100
    )
    # Hard CE vs. gold (using gold_labels as “true” indices)
    bg = F.cross_entropy(
        student_logits.view(-1, V),
        gold_labels.view(-1),
        ignore_index=-100
    )

    w_soft = 1.0 - w_teacher_ce - w_gold_ce
    return w_teacher_ce * bt + w_gold_ce * bg + w_soft * kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',          required=True,
                        help="Path to the new JSONL (with keys: focal_method, test_method_masked, assertions, etc.)")
    parser.add_argument('--model_output_dir',   default='my_students',
                        help="Folder under which the distilled student will be saved")
    parser.add_argument('--teacher_name',       default='Salesforce/codet5p-770m',
                        help="HuggingFace name or local path of the teacher model (now codet5p-770m)")
    parser.add_argument('--student_name',       default='Salesforce/codet5p-220m',
                        help="HuggingFace name or local path of the student model to initialize/distill (codet5p-220m is a reasonable choice)")
    parser.add_argument('--batch_size',         type=int, default=4)
    parser.add_argument('--max_samples',        type=int, default=None,
                        help="If set, only use up to this many training examples")
    parser.add_argument('--decoder_max_length', type=int, default=128)
    parser.add_argument('--epochs',             type=int, default=4)
    parser.add_argument('--lr',                 type=float, default=1e-4)
    parser.add_argument('--temperature',        type=float, default=1.0,
                        help="Temperature for soft‐KL (higher=softer teacher distribution)")
    parser.add_argument('--weight_teacher_ce',  type=float, default=0.3,
                        help="Weight for teacher’s hard CE loss")
    parser.add_argument('--weight_gold_ce',     type=float, default=0.4,
                        help="Weight for gold‐CE loss")
    args = parser.parse_args()

    if args.weight_teacher_ce + args.weight_gold_ce > 1.0:
        raise ValueError("Sum of weight_teacher_ce + weight_gold_ce must be ≤ 1.0")

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        # torch.device('mps')  if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print(f"> Device = {device}")

    # Load tokenizer from teacher; load student model from HF or local
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_name)
    student   = AutoModelForSeq2SeqLM.from_pretrained(args.student_name)
    # Make sure student and tokenizer share the same vocab size
    student.resize_token_embeddings(len(tokenizer))
    student.to(device)

    # Build the distillation dataset
    ds = DistillationDataset(
        path=args.data_path,
        tokenizer=tokenizer,
        decoder_max_length=args.decoder_max_length,
        max_samples=args.max_samples
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"> {len(loader)} batches, {len(ds)} examples loaded")

    optimizer = AdamW(student.parameters(), lr=args.lr)
    global_step = 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = len(loader)
        # Handy checkpoints every 10%
        checkpoints = {
            max(1, int(n_batches * i / LOG_STEPS)): i * 10
            for i in range(1, LOG_STEPS + 1)
        }
        print(f"> Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(loader, start=1):
            global_step += 1
            inp = batch['input_ids'].to(device)          # (B, 512)
            msk = batch['attention_mask'].to(device)     # (B, 512)
            gl  = batch['gold_labels'].to(device)        # (B, L)
            tl  = batch['teacher_labels'].to(device)     # (B, L)
            tlog= batch['teacher_logits'].to(device)     # (B, L, V)

            # Forward once on the student, using gold labels so that out.logits is (B, L, V)
            outputs = student(input_ids=inp, attention_mask=msk, labels=gl)
            slog = outputs.logits  # shape: (B, L, V)

            loss = distillation_loss(
                student_logits=slog,
                teacher_logits=tlog,
                gold_labels=gl,
                teacher_labels=tl,
                temperature=args.temperature,
                w_teacher_ce=args.weight_teacher_ce,
                w_gold_ce=args.weight_gold_ce
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Every 1000 steps, print a sample for sanity check
            if global_step % 1000 == 0:
                raw_in  = batch['raw_input'][0]
                raw_gt  = batch['raw_gold'][0]
                raw_tch = batch['raw_teacher'][0]

                with torch.no_grad():
                    pred_ids = student.generate(
                        inp[:1],
                        attention_mask=msk[:1],
                        max_length=ds.decoder_max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                student_txt = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

                print("\n―――― Sample check ――――")
                print(f"[Step {global_step}] INPUT PROMPT:\n{raw_in}\n")
                print(f"GROUND‐TRUTH ASSERTIONS:\n{raw_gt}\n")
                print(f"TEACHER PREDICTIONS    :\n{raw_tch}\n")
                print(f"STUDENT PREDICTION     :\n{student_txt}\n")
                print("―――― End sample ――――\n")

            # Occasionally print progress
            if step in checkpoints:
                pct = checkpoints[step]
                avg_loss = total_loss / step
                print(f"  → {pct}% done, avg loss {avg_loss:.4f}")

        avg_epoch_loss = total_loss / n_batches
        print(f"> Epoch {epoch} done, avg loss {avg_epoch_loss:.4f}\n")

    # Save the final student model + tokenizer under a timestamped subfolder
    base = args.model_output_dir
    os.makedirs(base, exist_ok=True)
    suffix = args.student_name.rstrip('/').split('/')[-1]
    outdir = os.path.join(base, suffix)
    if os.path.exists(outdir):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"{outdir}_{ts}"
    os.makedirs(outdir, exist_ok=True)

    student.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"✅ Saved distilled student to {outdir}")

if __name__ == '__main__':
    main()
