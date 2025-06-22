#!/usr/bin/env python
"""
Script to distill a student T5 model on Java test assertion generation.
VERSION 2.4: Fixes RNG state loading issue when resuming on a GPU.
"""
import argparse
import json
import os
import gc
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import javalang
import sacrebleu
import matplotlib.pyplot as plt

from decompress_tensor import decompress_logits

LOG_STEPS = 10


def get_dynamic_hyperparams(epoch: int, total_epochs: int) -> Tuple[float, float]:
    start_temp, end_temp = 4.0, 1.0
    temperature = start_temp * ((end_temp / start_temp) ** (epoch / max(total_epochs - 1, 1)))
    alpha = 0.5
    return temperature, alpha

def compute_ast_penalty(predictions: torch.Tensor, tokenizer) -> torch.Tensor:
    device = predictions.device
    total_penalty = 0.0
    batch_size = predictions.size(0)
    for i in range(batch_size):
        try:
            pred_ids, pred_text = predictions[i], tokenizer.decode(predictions[i], skip_special_tokens=True)
            java_code = f"public class TestClass {{ public void testMethod() {{ {pred_text} }} }}"
            try:
                javalang.parse.parse(java_code)
                penalty = 0.0
            except javalang.parser.JavaSyntaxError: penalty = 1.0
            except: penalty = 0.5
        except: penalty = 1.0
        total_penalty += penalty
    avg_penalty = total_penalty / batch_size if batch_size > 0 else 0.0
    return torch.tensor(avg_penalty, device=device)

class DistillationDataset(Dataset):
    def __init__(self, path, tokenizer, decoder_max_length, max_samples=None):
        self.tokenizer, self.decoder_max_length, self.entries = tokenizer, decoder_max_length, []
        with open(path, 'r') as f:
            for line in f:
                if max_samples and len(self.entries) >= max_samples: break
                obj = json.loads(line)
                if 'header' in obj or 'focal_method' not in obj or 'test_method_masked' not in obj: continue
                gold_list, teacher_list = obj.get('assertions', []), obj.get('predicted_assertions', [])
                gold_str, teacher_str = "\n".join(gold_list), "\n".join(teacher_list)
                inp = (f"FOCAL METHOD:\n{obj['focal_method']}\n\n" f"TEST METHOD:\n{obj['test_method_masked']}")
                self.entries.append({'input': inp, 'gold': gold_str, 'teacher_pred': teacher_str, 'compressed_logits': obj.get('compressed_logits')})
        print(f"> Loaded {len(self.entries)} examples (metadata only)")
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        rec = self.entries[idx]
        enc = self.tokenizer(rec['input'], max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        gold_enc = self.tokenizer(rec['gold'], max_length=self.decoder_max_length, padding='max_length', truncation=True, return_tensors='pt')
        tp_enc = self.tokenizer(rec['teacher_pred'], max_length=self.decoder_max_length, padding='max_length', truncation=True, return_tensors='pt')
        t_logits = decompress_logits(rec['compressed_logits'])
        if t_logits is None: raise ValueError(f"Could not decompress logits for example {idx}")
        t_logits = t_logits.half()
        if t_logits.dim() == 3: t_logits = t_logits.squeeze(0)
        L, V, T = self.decoder_max_length, t_logits.size(-1), t_logits.size(0)
        if T < L: t_logits = torch.cat([t_logits, torch.full((L - T, V), -1e9, dtype=torch.float16)], dim=0)
        else: t_logits = t_logits[:L]
        labels_gold, labels_tp = gold_enc['input_ids'].squeeze(0), tp_enc['input_ids'].squeeze(0)
        labels_gold[labels_gold == self.tokenizer.pad_token_id] = -100
        labels_tp[labels_tp == self.tokenizer.pad_token_id] = -100
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'raw_input': rec['input'], 'raw_gold': rec['gold'], 'raw_teacher': rec['teacher_pred'], 'gold_labels': labels_gold, 'teacher_labels': labels_tp, 'teacher_logits': t_logits}

def collate_fn(batch):
    return {'input_ids': torch.stack([b['input_ids'] for b in batch]), 'attention_mask': torch.stack([b['attention_mask'] for b in batch]), 'raw_input': [b['raw_input'] for b in batch], 'raw_gold': [b['raw_gold'] for b in batch], 'raw_teacher': [b['raw_teacher'] for b in batch], 'gold_labels': torch.stack([b['gold_labels'] for b in batch]), 'teacher_labels': torch.stack([b['teacher_labels'] for b in batch]), 'teacher_logits': torch.stack([b['teacher_logits'] for b in batch])}

def distillation_loss(student_logits, teacher_logits, gold_labels, teacher_labels, temperature, w_teacher_ce, w_gold_ce, ast_weight: float = 0.1, tokenizer=None):
    B, L, V = student_logits.shape; mask = (teacher_labels != -100).view(-1)
    t_probs, s_logp = F.softmax(teacher_logits.float()/temperature,dim=-1), F.log_softmax(student_logits/temperature,dim=-1)
    t_flat, s_flat = t_probs.view(-1, V), s_logp.view(-1, V)
    kl = F.kl_div(s_flat[mask],t_flat[mask],reduction='batchmean')*(temperature**2) if mask.sum()>0 else torch.tensor(0.0,device=student_logits.device)
    bt, bg = F.cross_entropy(student_logits.view(-1,V),teacher_labels.view(-1),ignore_index=-100), F.cross_entropy(student_logits.view(-1,V),gold_labels.view(-1),ignore_index=-100)
    w_soft = 1.0 - w_teacher_ce - w_gold_ce
    base_loss = w_teacher_ce*bt + w_gold_ce*bg + w_soft*kl
    preds = torch.argmax(student_logits, dim=-1)
    ast_pen = compute_ast_penalty(preds, tokenizer) if tokenizer is not None else torch.tensor(0.0, device=student_logits.device)
    return base_loss + ast_weight*ast_pen

def evaluate_validation(model, tokenizer, val_loader, device):
    model.eval()
    total_val_loss, exact_matches, total_samples = 0.0, 0, 0
    all_preds_text, all_golds_text = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            inp, msk, gl = batch['input_ids'].to(device,non_blocking=True), batch['attention_mask'].to(device,non_blocking=True), batch['gold_labels'].to(device,non_blocking=True)
            with autocast(device_type=device.type):
                outputs = model(input_ids=inp, attention_mask=msk, labels=gl)
                total_val_loss += outputs.loss.item()
            generated_ids = model.generate(inp, attention_mask=msk, max_length=128, num_beams=4, early_stopping=True)
            preds_text, golds_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True), batch['raw_gold']
            for pred, gold in zip(preds_text, golds_text):
                pred_clean, gold_clean = pred.strip(), gold.strip()
                all_preds_text.append(pred_clean)
                all_golds_text.append(gold_clean)
                if pred_clean == gold_clean: exact_matches += 1
            total_samples += len(golds_text)
    avg_loss = total_val_loss/len(val_loader)
    em = exact_matches/total_samples if total_samples>0 else 0.0
    bleu_result = sacrebleu.corpus_bleu(all_preds_text, [all_golds_text])
    bleu = bleu_result.score / 100.0
    return {'val_loss': avg_loss, 'val_bleu': bleu, 'val_em': em}

def plot_metrics(history, output_dir):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss'); plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png')); plt.close()
    print(f"✅ Saved loss history plot to {output_dir}")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'; ax1.set_xlabel('Epochs'); ax1.set_ylabel('BLEU Score', color=color)
    ax1.plot(epochs, history['val_bleu'], color=color, marker='o', label='Validation BLEU'); ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx(); color = 'tab:green'; ax2.set_ylabel('Exact Match', color=color)
    ax2.plot(epochs, history['val_em'], color=color, marker='s', linestyle='--', label='Validation EM'); ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout(); plt.title('Validation BLEU and Exact Match'); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_history.png')); plt.close()
    print(f"✅ Saved metrics history plot to {output_dir}")

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, output_dir, is_best=False):
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    state = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'history': history,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),
    }
    filename = 'best_model_checkpoint.pt' if is_best else 'latest_checkpoint.pt'
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    print(f"✅ Checkpoint saved to {filepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--val_data_path', required=True)
    parser.add_argument('--model_output_dir', required=True)
    parser.add_argument('--teacher_name', default='Salesforce/codet5p-770m')
    parser.add_argument('--student_name', default='Salesforce/codet5p-220m')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--decoder_max_length', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--weight_teacher_ce', type=float, default=0.3)
    parser.add_argument('--weight_gold_ce', type=float, default=0.4)
    parser.add_argument('--ast_weight', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to checkpoint file to resume training.")
    args = parser.parse_args()

    if args.weight_teacher_ce + args.weight_gold_ce > 1.0:
        raise ValueError("Sum of weight_teacher_ce + weight_gold_ce must be ≤ 1.0")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"> Device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_name)
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student_name)
    student.config.dropout_rate, student.config.attention_dropout = args.dropout_rate, args.dropout_rate
    student.resize_token_embeddings(len(tokenizer))
    
    try:
        student = torch.compile(student)
        print("> Model compiled for optimized performance (PyTorch 2.0+).")
    except Exception:
        print("> Could not compile model. Using standard eager mode.")
    
    student.to(device)

    train_ds = DistillationDataset(path=args.train_data_path, tokenizer=tokenizer, decoder_max_length=args.decoder_max_length, max_samples=args.max_train_samples)
    val_ds = DistillationDataset(path=args.val_data_path, tokenizer=tokenizer, decoder_max_length=args.decoder_max_length, max_samples=args.max_val_samples)
    num_workers = 2
    pin_memory = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    print(f"> {len(train_loader)} training batches, {len(train_ds)} training examples")
    print(f"> {len(val_loader)} validation batches, {len(val_ds)} validation examples")

    optimizer = AdamW([{"params": [p for n, p in student.named_parameters() if 'bias' not in n and 'LayerNorm.weight' not in n], "weight_decay": args.weight_decay},
                       {"params": [p for n, p in student.named_parameters() if 'bias' in n or 'LayerNorm.weight' in n], "weight_decay": 0.0}], lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scaler = GradScaler()

    start_epoch = 1
    global_step = 0
    history = {'train_loss': [], 'val_loss': [], 'val_bleu': [], 'val_em': []}
    best_val_loss = float('inf')
    run_output_dir = args.model_output_dir

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"> Resuming training from checkpoint: {args.resume_from_checkpoint}")
        # --- FIXED: Load to CPU first to prevent RNG state error on GPU ---
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu', weights_only=False)
        
        cleaned_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('_orig_mod.'):
                cleaned_state_dict[k[10:]] = v
            else:
                cleaned_state_dict[k] = v
        
        model_to_load = student._orig_mod if hasattr(student, '_orig_mod') else student
        model_to_load.load_state_dict(cleaned_state_dict)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        run_output_dir = os.path.dirname(args.resume_from_checkpoint)
        
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])
        
        print(f"> Resumed successfully. Starting from epoch {start_epoch}. Output dir: {run_output_dir}")
    else:
        base = args.model_output_dir
        suffix = args.student_name.rstrip('/').split('/')[-1]
        run_output_dir = os.path.join(base, f"{suffix}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_output_dir, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        total_train_loss = 0.0
        n_train_batches = len(train_loader)
        temperature, alpha = get_dynamic_hyperparams(epoch - 1, args.epochs)
        print(f"> Epoch {epoch}/{args.epochs} | Temperature={temperature:.3f}")

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for step, batch in enumerate(progress_bar, start=1):
            with autocast(device_type=device.type):
                inp, msk, gl, tl, tlog = (batch['input_ids'].to(device), batch['attention_mask'].to(device),
                                          batch['gold_labels'].to(device), batch['teacher_labels'].to(device),
                                          batch['teacher_logits'].to(device))
                outputs = student(input_ids=inp, attention_mask=msk, labels=gl)
                slog = outputs.logits
                loss = distillation_loss(slog, tlog, gl, tl, temperature, args.weight_teacher_ce, args.weight_gold_ce, args.ast_weight, tokenizer)
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            total_train_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({'avg_train_loss': f"{total_train_loss / step:.4f}"})
        
        avg_epoch_loss = total_train_loss / n_train_batches
        print(f"> Epoch {epoch} training done, avg loss {avg_epoch_loss:.4f}")

        val_metrics = evaluate_validation(student, tokenizer, val_loader, device)
        val_loss = val_metrics['val_loss']
        history['train_loss'].append(avg_epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_bleu'].append(val_metrics['val_bleu'])
        history['val_em'].append(val_metrics['val_em'])
        print(f"> Epoch {epoch} | Val Loss: {val_loss:.4f} | Val BLEU: {val_metrics['val_bleu']:.4f} | Val EM: {val_metrics['val_em']:.4f}")

        scheduler.step(val_loss)

        save_checkpoint(epoch, student, optimizer, scheduler, scaler, history, run_output_dir)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, student, optimizer, scheduler, scaler, history, run_output_dir, is_best=True)
            print(f"> New best validation loss: {best_val_loss:.4f}")

        if device.type == 'cuda': torch.cuda.empty_cache()
        else: gc.collect()

    print("✅ Training finished.")
    model_to_save = student._orig_mod if hasattr(student, '_orig_mod') else student
    model_to_save.save_pretrained(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)
    print(f"✅ Saved final student model to {run_output_dir}")
    plot_metrics(history, run_output_dir)
                                                            
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()