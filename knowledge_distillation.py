#!/usr/bin/env python
"""
Script to distill a student T5 model on Java test assertion generation,
combining teacher‐prediction CE, gold CE, and soft KL losses with tunable weights.
Includes AST‐based penalty, weight decay, learning‐rate scheduling based on validation,
and dropout configuration to reduce overfitting.

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
import gc
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import javalang

from decompress_tensor import decompress_logits  # your utility for de‐compressing logits

LOG_STEPS = 10


def get_dynamic_hyperparams(epoch: int, total_epochs: int) -> Tuple[float, float]:
    """
    Exponential temperature decay: start at 4.0, end at 1.0 over total_epochs.
    Returns (temperature, alpha) with alpha fixed at 0.5.
    """
    start_temp, end_temp = 4.0, 1.0
    temperature = start_temp * ((end_temp / start_temp) ** (epoch / max(total_epochs - 1, 1)))
    alpha = 0.5
    return temperature, alpha


def compute_ast_penalty(predictions: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    AST-aware penalty for Java code predictions.
    Penalizes predictions that are syntactically invalid Java.
    """
    device = predictions.device
    total_penalty = 0.0
    batch_size = predictions.size(0)

    for i in range(batch_size):
        try:
            pred_ids = predictions[i]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            java_code = f"""
            public class TestClass {{
                public void testMethod() {{
                    {pred_text}
                }}
            }}
            """
            try:
                javalang.parse.parse(java_code)
                penalty = 0.0
            except javalang.parser.JavaSyntaxError:
                penalty = 1.0
            except:
                penalty = 0.5
        except:
            penalty = 1.0
        total_penalty += penalty

    avg_penalty = total_penalty / batch_size if batch_size > 0 else 0.0
    return torch.tensor(avg_penalty, device=device)


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
                # Skip header or malformed
                if 'header' in obj or 'focal_method' not in obj or 'test_method_masked' not in obj:
                    continue

                gold_list = obj.get('assertions', [])
                gold_str = "\n".join(gold_list) if isinstance(gold_list, list) else ""
                teacher_list = obj.get('predicted_assertions', [])
                teacher_str = "\n".join(teacher_list) if isinstance(teacher_list, list) else ""
                inp = (
                    f"FOCAL METHOD:\n{obj['focal_method']}\n\n"
                    f"TEST METHOD:\n{obj['test_method_masked']}"
                )

                self.entries.append({
                    'input':            inp,
                    'gold':             gold_str,
                    'teacher_pred':     teacher_str,
                    'compressed_logits': obj.get('compressed_logits')
                })

        print(f"> Loaded {len(self.entries)} examples (metadata only)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rec = self.entries[idx]

        # 1) Tokenize input
        enc = self.tokenizer(
            rec['input'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 2) Tokenize ground-truth assertions
        gold_enc = self.tokenizer(
            rec['gold'],
            max_length=self.decoder_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 3) Tokenize teacher predictions
        tp_enc = self.tokenizer(
            rec['teacher_pred'],
            max_length=self.decoder_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 4) Decompress teacher logits on the fly
        t_logits = decompress_logits(rec['compressed_logits'])
        if t_logits is None:
            raise ValueError(f"Could not decompress logits for example {idx}")
        t_logits = t_logits.half()  # cast to float16

        if t_logits.dim() == 3:
            t_logits = t_logits.squeeze(0)  # (T, V)

        # 5) Pad/truncate teacher_logits to length L = decoder_max_length
        L = self.decoder_max_length
        V = t_logits.size(-1)
        T = t_logits.size(0)
        if T >= L:
            t_logits = t_logits[:L]
        else:
            pad_tensor = torch.full((L - T, V), -1e9, dtype=torch.float16)
            t_logits = torch.cat([t_logits, pad_tensor], dim=0)  # (L, V)

        # 6) Build label tensors (replace pad_token_id with -100)
        labels_gold = gold_enc['input_ids'].squeeze(0)   # (L,)
        labels_tp   = tp_enc['input_ids'].squeeze(0)     # (L,)
        labels_gold[labels_gold == self.tokenizer.pad_token_id] = -100
        labels_tp[labels_tp == self.tokenizer.pad_token_id]       = -100

        return {
            'input_ids':      enc['input_ids'].squeeze(0),       # (512,)
            'attention_mask': enc['attention_mask'].squeeze(0),  # (512,)
            'raw_input':      rec['input'],
            'raw_gold':       rec['gold'],
            'raw_teacher':    rec['teacher_pred'],
            'gold_labels':    labels_gold,                        # (L,)
            'teacher_labels': labels_tp,                          # (L,)
            'teacher_logits': t_logits                            # (L, V) float16
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
    temperature, w_teacher_ce, w_gold_ce,
    ast_weight: float = 0.1,
    tokenizer=None
):
    """
    Computes combined loss = w_teacher_ce * CE(student_logits, teacher_labels)
                         + w_gold_ce    * CE(student_logits, gold_labels)
                         + (1 - w_teacher_ce - w_gold_ce) * KL(student_probs || teacher_probs)
                         + ast_weight   * AST_Penalty
    """
    B, L, V = student_logits.shape
    mask = (teacher_labels != -100).view(-1)

    # Soft KL (cast teacher to float32 for softmax stability)
    t_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    s_logp  = F.log_softmax(student_logits / temperature, dim=-1)
    t_flat = t_probs.view(-1, V)
    s_flat = s_logp.view(-1, V)
    kl = F.kl_div(s_flat[mask], t_flat[mask], reduction='batchmean') * (temperature ** 2) if mask.sum() > 0 else torch.tensor(0.0, device=student_logits.device)

    # Hard CE vs. teacher and gold
    bt = F.cross_entropy(
        student_logits.view(-1, V),
        teacher_labels.view(-1),
        ignore_index=-100
    )
    bg = F.cross_entropy(
        student_logits.view(-1, V),
        gold_labels.view(-1),
        ignore_index=-100
    )

    w_soft = 1.0 - w_teacher_ce - w_gold_ce
    base_loss = w_teacher_ce * bt + w_gold_ce * bg + w_soft * kl

    # AST penalty
    if tokenizer is not None:
        preds = torch.argmax(student_logits, dim=-1)  # (B, L)
        ast_pen = compute_ast_penalty(preds, tokenizer)  # scalar
    else:
        ast_pen = torch.tensor(0.0, device=student_logits.device)

    total_loss = base_loss + ast_weight * ast_pen
    return total_loss


def evaluate_validation(model, tokenizer, val_loader, device, decoder_max_length):
    """
    Run one pass on validation data and return average loss.
    """
    model.eval()
    total_val_loss = 0.0
    n_val_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            inp  = batch['input_ids'].to(device,   non_blocking=True)
            msk  = batch['attention_mask'].to(device, non_blocking=True)
            gl   = batch['gold_labels'].to(device,   non_blocking=True)
            tl   = batch['teacher_labels'].to(device, non_blocking=True)
            tlog = batch['teacher_logits'].to(device, non_blocking=True)

            outputs = model(input_ids=inp, attention_mask=msk, labels=gl)
            slog = outputs.logits  # (B, L, V)

            # Use fixed hyperparams for validation
            val_loss = distillation_loss(
                slog, tlog, gl, tl,
                temperature=1.0,
                w_teacher_ce=0.3,
                w_gold_ce=0.4,
                ast_weight=0.1,
                tokenizer=tokenizer
            )
            total_val_loss += val_loss.item()

    return total_val_loss / n_val_batches if n_val_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',    required=True)
    parser.add_argument('--val_data_path',      required=True)
    parser.add_argument('--model_output_dir',   required=True)
    parser.add_argument('--teacher_name',       default='Salesforce/codet5p-770m')
    parser.add_argument('--student_name',       default='Salesforce/codet5p-220m')
    parser.add_argument('--batch_size',         type=int, default=4)
    parser.add_argument('--max_train_samples',  type=int, default=None)
    parser.add_argument('--max_val_samples',    type=int, default=None)
    parser.add_argument('--decoder_max_length', type=int, default=128)
    parser.add_argument('--epochs',             type=int, default=10)
    parser.add_argument('--lr',                 type=float, default=3e-5)
    parser.add_argument('--weight_decay',       type=float, default=0.01)
    parser.add_argument('--temperature',        type=float, default=4.0,
                        help="Initial temperature for soft-KL (will decay)")
    parser.add_argument('--weight_teacher_ce',  type=float, default=0.3)
    parser.add_argument('--weight_gold_ce',     type=float, default=0.4)
    parser.add_argument('--ast_weight',         type=float, default=0.1,
                        help="Weight for AST penalty")
    parser.add_argument('--dropout_rate',       type=float, default=0.1,
                        help="Dropout rate to apply to student model")
    args = parser.parse_args()

    # Ensure CE+CE ≤ 1
    if args.weight_teacher_ce + args.weight_gold_ce > 1.0:
        raise ValueError("Sum of weight_teacher_ce + weight_gold_ce must be ≤ 1.0")

    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"> Device = {device}")

    # Load tokenizer & student model, set dropout
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_name)
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student_name)
    student.config.dropout_rate = args.dropout_rate
    student.config.attention_dropout = args.dropout_rate
    student.resize_token_embeddings(len(tokenizer))
    student.to(device)

    # Build datasets
    train_ds = DistillationDataset(
        path=args.train_data_path,
        tokenizer=tokenizer,
        decoder_max_length=args.decoder_max_length,
        max_samples=args.max_train_samples
    )
    val_ds = DistillationDataset(
        path=args.val_data_path,
        tokenizer=tokenizer,
        decoder_max_length=args.decoder_max_length,
        max_samples=args.max_val_samples
    )

    # DataLoader settings (force single‐worker, no pin_memory on Colab)
    num_workers = 0
    pin_memory = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"> {len(train_loader)} training batches, {len(train_ds)} training examples")
    print(f"> {len(val_loader)} validation batches, {len(val_ds)} validation examples")

    # Setup optimizer with weight decay and LR scheduler on validation loss
    optimizer = AdamW(
        [
            {"params": [p for n, p in student.named_parameters() if 'bias' not in n and 'LayerNorm.weight' not in n],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in student.named_parameters() if 'bias' in n or 'LayerNorm.weight' in n],
             "weight_decay": 0.0},
        ],
        lr=args.lr
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    scaler = GradScaler()

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_train_loss = 0.0
        n_train_batches = len(train_loader)

        # Compute dynamic hyperparams for this epoch
        temperature, alpha = get_dynamic_hyperparams(epoch - 1, args.epochs)

        print(f"> Epoch {epoch}/{args.epochs} | Temperature={temperature:.3f} | α={alpha:.3f}")

        checkpoints = {
            max(1, int(n_train_batches * i / LOG_STEPS)): i * 10
            for i in range(1, LOG_STEPS + 1)
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for step, batch in enumerate(progress_bar, start=1):
            global_step += 1

            inp  = batch['input_ids'].to(device,   non_blocking=True)
            msk  = batch['attention_mask'].to(device, non_blocking=True)
            gl   = batch['gold_labels'].to(device,   non_blocking=True)
            tl   = batch['teacher_labels'].to(device, non_blocking=True)
            tlog = batch['teacher_logits'].to(device, non_blocking=True)

            with autocast():
                outputs = student(input_ids=inp, attention_mask=msk, labels=gl)
                slog = outputs.logits  # (B, L, V) in mixed precision

                loss = distillation_loss(
                    student_logits=slog,
                    teacher_logits=tlog,
                    gold_labels=gl,
                    teacher_labels=tl,
                    temperature=temperature,
                    w_teacher_ce=args.weight_teacher_ce,
                    w_gold_ce=args.weight_gold_ce,
                    ast_weight=args.ast_weight,
                    tokenizer=tokenizer
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            avg_loss = total_train_loss / step
            progress_bar.set_postfix({'avg_train_loss': f"{avg_loss:.4f}"})

            # Sample check every 1000 steps
            if global_step % 1000 == 0:
                raw_in  = batch['raw_input'][0]
                raw_gt  = batch['raw_gold'][0]
                raw_tch = batch['raw_teacher'][0]

                with torch.no_grad():
                    pred_ids = student.generate(
                        inp[:1],
                        attention_mask=msk[:1],
                        max_length=args.decoder_max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                student_txt = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

                print("\n―――― Sample check ――――")
                print(f"[Step {global_step}] INPUT PROMPT:\n{raw_in}\n")
                print(f"GROUND-TRUTH ASSERTIONS:\n{raw_gt}\n")
                print(f"TEACHER PREDICTIONS    :\n{raw_tch}\n")
                print(f"STUDENT PREDICTION     :\n{student_txt}\n")
                print("―――― End sample ――――\n")

            # Percent completion
            if step in checkpoints:
                pct = checkpoints[step]
                print(f"  → {pct}% complete")

        avg_epoch_loss = total_train_loss / n_train_batches
        print(f"> Epoch {epoch} training done, avg loss {avg_epoch_loss:.4f}")

        # Validation
        val_loss = evaluate_validation(student, tokenizer, val_loader, device, args.decoder_max_length)
        print(f"> Epoch {epoch} validation loss: {val_loss:.4f}")

        # Step LR scheduler on validation loss
        scheduler.step(val_loss)

        # Free memory once per epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        else:
            del batch
            gc.collect()

    # Save final student model + tokenizer
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
