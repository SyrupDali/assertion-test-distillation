#!/usr/bin/env python
"""
Script to distill a student T5 model on Java test assertion generation,
combining teacher‐prediction CE, gold CE, and soft KL losses with tunable weights.
Expects a JSONL where:
 - The first line may be a header (with key "header").
 - Each example has:
     "focal_file" (str),
     "test_method_masked" (str),
     "original_target" (newline/semicolon‐separated GT),
     "predicted_assertions" (newline/semicolon‐separated teacher preds),
     "compressed_logits" (dict, as written by compress_logits).
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

from decompress_tensor import decompress_logits  # your new util

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
                if 'header' in obj or 'focal_file' not in obj or 'test_method_masked' not in obj:
                    continue
                inp = (
                    f"FOCAL CODE:\n{obj['focal_file']}\n\n"
                    f"TEST METHOD:\n{obj['test_method_masked']}"
                )
                self.entries.append({
                    'input':          inp,
                    'gold':           obj.get('original_target', ''),
                    'teacher_pred':   obj.get('predicted_assertions', ''),
                    'teacher_logits': obj.get('compressed_logits')
                })
        print(f"> Loaded {len(self.entries)} examples")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rec = self.entries[idx]
        enc = self.tokenizer(
            rec['input'],
            max_length=512, padding='max_length', truncation=True, return_tensors='pt'
        )
        gold_enc = self.tokenizer(
            rec['gold'],
            max_length=self.decoder_max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        tp_enc = self.tokenizer(
            rec['teacher_pred'],
            max_length=self.decoder_max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        # decompress teacher logits (may come back as [1, T, V] or [T, V])
        t_logits = decompress_logits(rec['teacher_logits'])
        if t_logits is None:
            raise ValueError(f"Could not decompress logits for example {idx}")
        t_logits = t_logits.float()
        # if we still have a batch dimension, just take the first item
        if t_logits.dim() == 3:
            t_logits = t_logits[0]

        # align to decoder length L
        L = gold_enc['input_ids'].size(1)
        V = t_logits.size(-1)
        T = t_logits.size(0)
        if T >= L:
            t_logits = t_logits[:L]
        else:
            pad = torch.full((L - T, V), -1e9, dtype=t_logits.dtype)
            t_logits = torch.cat([t_logits, pad], dim=0)

        labels_gold = gold_enc['input_ids'].squeeze(0)
        labels_tp   = tp_enc  ['input_ids'].squeeze(0)
        # mark pads as -100
        labels_gold[labels_gold == self.tokenizer.pad_token_id] = -100
        labels_tp  [labels_tp   == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'raw_input':      rec['input'],            # for debugging
            'raw_gold':       rec['gold'],             # for debugging
            'raw_teacher':    rec['teacher_pred'],     # for debugging
            'gold_labels':    labels_gold,
            'teacher_labels': labels_tp,
            'teacher_logits': t_logits
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
    student_logits: (B, L, V)
    teacher_logits: (B, L, V)
    gold_labels:    (B, L) with -100 for pad
    teacher_labels:(B, L) with -100 beyond teacher EOS
    """
    B, L, V = student_logits.shape

    # Create mask from teacher_labels != -100 → up to teacher EOS
    mask = (teacher_labels != -100).view(-1)  # (B*L,)

    # Soft KL on masked positions
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    s_logp  = F.log_softmax(student_logits / temperature, dim=-1)
    t_flat = t_probs.view(-1, V)
    s_flat = s_logp.view(-1, V)
    if mask.sum() > 0:
        kl = F.kl_div(s_flat[mask], t_flat[mask], reduction='batchmean') * (temperature**2)
    else:
        kl = torch.tensor(0.0, device=student_logits.device)

    # CE vs. teacher prediction (hard labels)
    bt = F.cross_entropy(
        student_logits.view(-1, V),
        teacher_labels.view(-1),
        ignore_index=-100
    )
    # CE vs. gold target
    bg = F.cross_entropy(
        student_logits.view(-1, V),
        gold_labels.view(-1),
        ignore_index=-100
    )

    w_soft = 1.0 - w_teacher_ce - w_gold_ce
    return w_teacher_ce * bt + w_gold_ce * bg + w_soft * kl

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',          required=True)
    p.add_argument('--model_output_dir',   default='student_output')
    p.add_argument('--teacher_name',       default='Salesforce/codet5-base')
    p.add_argument('--student_name',       default='Salesforce/codet5-small')
    p.add_argument('--batch_size',         type=int, default=4)
    p.add_argument('--max_samples',        type=int, default=None)
    p.add_argument('--decoder_max_length', type=int, default=128)
    p.add_argument('--epochs',             type=int, default=4)
    p.add_argument('--lr',                 type=float, default=1e-4)
    p.add_argument('--temperature',        type=float, default=1.0)
    p.add_argument('--weight_teacher_ce',  type=float, default=0.3)
    p.add_argument('--weight_gold_ce',     type=float, default=0.4)
    args = p.parse_args()

    if args.weight_teacher_ce + args.weight_gold_ce > 1.0:
        raise ValueError("Sum of teacher_ce + gold_ce must ≤ 1")

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps')  if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print(f"> Device = {device}")

    # Load tokenizer from teacher; student from HF or local
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_name)
    student   = AutoModelForSeq2SeqLM.from_pretrained(args.student_name)
    student.resize_token_embeddings(len(tokenizer))
    student.to(device)

    ds = DistillationDataset(
        path=args.data_path,
        tokenizer=tokenizer,
        decoder_max_length=args.decoder_max_length,
        max_samples=args.max_samples
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    print(f"> {len(loader)} batches, {len(ds)} examples loaded")

    optim = AdamW(student.parameters(), lr=args.lr)
    global_step = 0

    for ep in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = len(loader)
        checkpoints = { max(1, int(n_batches*i/LOG_STEPS)): i*10 for i in range(1, LOG_STEPS+1) }
        print(f"> Epoch {ep}/{args.epochs}")

        for i, batch in enumerate(loader, 1):
            global_step += 1
            inp = batch['input_ids'].to(device)
            msk = batch['attention_mask'].to(device)
            gl  = batch['gold_labels'].to(device)
            tl  = batch['teacher_labels'].to(device)
            tlog= batch['teacher_logits'].to(device)

            out = student(input_ids=inp, attention_mask=msk, labels=gl)
            slog = out.logits  # (B,L,V)

            loss = distillation_loss(
                slog, tlog, gl, tl,
                args.temperature, args.weight_teacher_ce, args.weight_gold_ce
            )
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

            # Every 100 steps, print one example for verification
            if global_step % 1000 == 0:
                raw_in  = batch['raw_input'][0]
                raw_gt  = batch['raw_gold'][0]
                raw_tch = batch['raw_teacher'][0]

                with torch.no_grad():
                    pred_ids = student.generate(
                        inp[:1], attention_mask=msk[:1],
                        max_length=ds.decoder_max_length,
                        num_beams=4, early_stopping=True
                    )
                student_txt = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

                print("\n―――― Sample check ――――")
                print(f"[Step {global_step}] INPUT PROMPT:\n{raw_in}\n")
                print(f"GROUND-TRUTH ASSERTIONS:\n{raw_gt}\n")
                print(f"TEACHER PREDICTIONS :\n{raw_tch}\n")
                print(f"STUDENT PREDICTION  :\n{student_txt}\n")
                print("―――― End sample ――――\n")

            if i in checkpoints:
                print(f"  → {checkpoints[i]:3d}% done, avg loss {total_loss/i:.4f}")

        print(f"> Epoch {ep} done, avg loss {total_loss/n_batches:.4f}\n")

    # Save final student
    base = args.model_output_dir
    os.makedirs(base, exist_ok=True)
    suffix = args.student_name.split('/')[-1]
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
