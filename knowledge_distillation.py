#!/usr/bin/env python
import json
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from decompress_tensor import decompress_tensor_optimized

LOG_STEPS = 10  # log at every 10% of an epoch

class DistillationDataset(Dataset):
    def __init__(self, path, tokenizer, decoder_max_length, max_samples=None):
        self.tokenizer = tokenizer
        self.decoder_max_length = decoder_max_length
        self.entries = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line)
                self.entries.append({
                    "input": obj["test_method_masked"],
                    "output": obj["assertions"][0],
                    "teacher_logits_compressed": obj["teacher_logits"]
                })
        print(f"> Loaded {len(self.entries)} examples", flush=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rec = self.entries[idx]
        # tokenize encoder inputs
        enc = self.tokenizer(
            rec["input"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # tokenize decoder inputs (labels) to fixed decoder_max_length
        dec = self.tokenizer(
            rec["output"],
            padding="max_length",
            truncation=True,
            max_length=self.decoder_max_length,
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels         = dec["input_ids"].squeeze(0)  # shape = (decoder_max_length,)

        # decompress teacher logits
        t_logits = decompress_tensor_optimized(rec["teacher_logits_compressed"])
        # dynamically crop or pad teacher logits to labels length
        teacher_len, vocab_size = t_logits.shape
        student_len = labels.size(0)
        if teacher_len >= student_len:
            t_logits = t_logits[:student_len, :]
        else:
            pad_len = student_len - teacher_len
            pad = torch.full((pad_len, vocab_size), -1e9)
            t_logits = torch.cat([t_logits, pad], dim=0)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "teacher_logits": t_logits
        }

def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "teacher_logits": torch.stack([b["teacher_logits"] for b in batch]),
    }

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7):
    # soft (KL) loss
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    s_logp  = F.log_softmax(student_logits  / temperature, dim=-1)
    kl_loss = F.kl_div(s_logp, t_probs, reduction="batchmean") * (temperature ** 2)
    # hard (CE) loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    return alpha * ce_loss + (1 - alpha) * kl_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",         type=int, default=4,
                        help="Batch size (reduce if you OOM)")
    parser.add_argument("--max_samples",        type=int, default=1000,
                        help="Max number of examples to load (None for all)")
    parser.add_argument("--decoder_max_length", type=int, default=128,
                        help="Max output length for student decoder")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Running on device: {device}", flush=True)
    print(f"> batch_size={args.batch_size}, max_samples={args.max_samples}, decoder_max_length={args.decoder_max_length}\n", flush=True)

    # 1) load tokenizer & student model (use CodeT5 vocab)
    teacher_tok = "Salesforce/codet5-small"
    student_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(teacher_tok)
    student = AutoModelForSeq2SeqLM.from_pretrained(student_name)
    student.resize_token_embeddings(len(tokenizer))
    student.to(device)

    # 2) prepare dataset & loader
    ds = DistillationDataset(
        path="dataset_with_predictions.jsonl",
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
    print(f"> DataLoader ready: {len(loader)} batches\n", flush=True)

    optimizer = AdamW(student.parameters(), lr=1e-4)
    epochs = 3

    # 3) training loop with 10% logging
    for ep in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = len(loader)
        checkpoints = {max(1, int(n_batches * i / LOG_STEPS)): i * 10 for i in range(1, LOG_STEPS + 1)}
        print(f"> Epoch {ep}/{epochs} — {n_batches} batches", flush=True)

        for i, batch in enumerate(loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            t_logits       = batch["teacher_logits"].to(device)

            outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            s_logits = outputs.logits
            loss = distillation_loss(s_logits, t_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i in checkpoints:
                pct = checkpoints[i]
                print(f"  → {pct}% done, avg loss: {total_loss/i:.4f}", flush=True)

        epoch_avg = total_loss / n_batches
        print(f"> Epoch {ep} done, avg loss: {epoch_avg:.4f}\n", flush=True)

    # 4) save distilled student
    out_dir = "distilled_codet5_student"
    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ Done. Model & tokenizer saved to `{out_dir}`", flush=True)

if __name__ == "__main__":
    main()
