#!/usr/bin/env python
"""
Evaluate a distilled student T5 model and also analyze the teacher predictions,
computing per-sample precision/recall/F1/accuracy/similarity/BLEU/CodeBLEU and producing
predictions+metrics, plus summary visuals.

This version expects the JSONL to have:
 - "focal_method" (str)
 - "test_method_masked" (str)
 - "assertions" (list of strings, each ending in ";")
 - "predicted_assertions" (list of strings, each ending in ";")
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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import seaborn as sns

import javalang
from codebleu import calc_code_bleu

smooth_fn = SmoothingFunction().method1


def normalize_assertion(a: str) -> str:
    a = re.sub(r"\s+", " ", a).strip()
    # Attempt to parse with javalang to normalize formatting if possible
    try:
        tree = javalang.parse.parse_member_declaration(a.rstrip(";"))
        # Reconstruct code from AST node for a normalized form
        # javalang does not provide a direct unparser, so we leave the original if parsing fails
        normalized = a
    except (javalang.parser.JavaSyntaxError, IndexError, TypeError):
        normalized = a
    normalized = re.sub(
        r"assertEquals\(\s*[^,]+,\s*([^)]+)\)",
        r"assertEquals(VALUE, \1)",
        normalized
    )
    return re.sub(
        r"assert(Equals|That|True|False)",
        lambda m: "assert" + m.group(1),
        normalized,
        flags=re.IGNORECASE
    )


def evaluate_assertions(
    generated_list: list[str],
    reference_list: list[str]
) -> dict:
    """
    Compare two Python lists of single-assertion strings (each ending in ";").
    Returns precision, recall, f1, accuracy, avg_similarity, bleu, codebleu, etc.
    """
    # Normalize each assertion
    gen_norm = [normalize_assertion(x.strip()) for x in generated_list if x.strip()]
    ref_norm = [normalize_assertion(x.strip()) for x in reference_list if x.strip()]

    # Count true positives (exact matches on normalized lines)
    tp = sum(1 for g in gen_norm if g in ref_norm)

    # Similarity: for each generated, best match in reference
    sim_scores = []
    for g in gen_norm:
        best = 0.0
        for r in ref_norm:
            best = max(best, SequenceMatcher(None, g, r).ratio())
        sim_scores.append(best)

    # Precision / Recall / F1 / Accuracy
    P = tp / len(gen_norm) if gen_norm else 0.0
    R = tp / len(ref_norm) if ref_norm else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    ACC = tp / max(len(gen_norm), len(ref_norm)) if max(len(gen_norm), len(ref_norm)) > 0 else 0.0

    # BLEU: join all normalized lines with "\n", tokenize on whitespace
    joined_gen = " ".join(gen_norm)
    joined_ref = " ".join(ref_norm)
    cand_tokens = joined_gen.split()
    ref_tokens = [joined_ref.split()]
    try:
        BLEU = sentence_bleu(
            ref_tokens,
            cand_tokens,
            smoothing_function=smooth_fn
        )
    except Exception:
        BLEU = 0.0

    # CodeBLEU: compute on the joined strings
    try:
        codebleu_scores = calc_code_bleu(
            refs=[joined_ref],
            preds=[joined_gen],
            lang="java",
            ngram=4,
            no_comment=True,
            no_doc=True,
            vocab_path=None
        )
        CODEBLEU = codebleu_scores["CodeBLEU"]
    except Exception:
        CODEBLEU = 0.0

    return {
        "precision": P,
        "recall": R,
        "f1": F1,
        "accuracy": ACC,
        "avg_similarity": (sum(sim_scores) / len(sim_scores)) if sim_scores else 0.0,
        "bleu": BLEU,
        "codebleu": CODEBLEU,
        "exact_matches": tp,
        "generated_count": len(gen_norm),
        "reference_count": len(ref_norm),
        "similarity_scores": sim_scores
    }


class AssertionDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, max_src=1024, max_tgt=512):
        self.data = data
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        e = self.data[i]
        focal = e["focal_method"]
        test_masked = e["test_method_masked"]

        # Construct source string
        src = f"FOCAL METHOD:\n{focal}\n\nTEST METHOD:\n{test_masked}"
        tokens = self.tok(
            src,
            padding="max_length",
            truncation=True,
            max_length=self.max_src,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
            "original": e
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "originals": [b["original"] for b in batch]
    }


def generate_and_evaluate(model, tok, loader, device, kind, out_dir):
    """
    kind: "student" or "teacher"
    """
    model.to(device).eval()
    ts = torch.tensor(0)  # dummy to avoid unused import
    preds_file = os.path.join(out_dir, f"{kind}_preds.jsonl")
    metrics_file = os.path.join(out_dir, f"{kind}_metrics.json")
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    writer = open(preds_file, "w")
    P_list, R_list, F1_list, ACC_list, BL_list, CB_list, SIM_list = [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Gen {kind}"):
            inp = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            origs = batch["originals"]

            if kind == "student":
                gen_ids = model.generate(
                    inp,
                    attention_mask=msk,
                    max_length=loader.dataset.max_tgt,
                    num_beams=4,
                    early_stopping=True
                )
                outs = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
                # Split each student output into a list of assertions
                gen_lists = [
                    [s.strip() + ";" for s in re.split(r";|\n", out_str) if s.strip()]
                    for out_str in outs
                ]
            else:
                # Teacher: JSON field "predicted_assertions" is already a list
                gen_lists = [
                    o.get("predicted_assertions", [])
                    for o in origs
                ]

            # Reference lists: JSON field "assertions" is already a list
            ref_lists = [
                o.get("assertions", [])
                for o in origs
            ]

            for o, gen_list, ref_list in zip(origs, gen_lists, ref_lists):
                m = evaluate_assertions(gen_list, ref_list)
                P_list.append(m["precision"])
                R_list.append(m["recall"])
                F1_list.append(m["f1"])
                ACC_list.append(m["accuracy"])
                BL_list.append(m["bleu"])
                CB_list.append(m["codebleu"])
                SIM_list.extend(m["similarity_scores"])

                rec = {
                    **o,
                    f"{kind}_prediction": gen_list,
                    "metrics": {
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1": m["f1"],
                        "accuracy": m["accuracy"],
                        "avg_similarity": m["avg_similarity"],
                        "bleu": m["bleu"],
                        "codebleu": m["codebleu"]
                    }
                }
                writer.write(json.dumps(rec) + "\n")

    writer.close()

    final = {
        "precision": float(np.mean(P_list)),
        "recall": float(np.mean(R_list)),
        "f1": float(np.mean(F1_list)),
        "accuracy": float(np.mean(ACC_list)),
        "avg_similarity": float(np.mean(SIM_list)),
        "avg_bleu": float(np.mean(BL_list)),
        "avg_codebleu": float(np.mean(CB_list))
    }
    with open(metrics_file, "w") as f:
        json.dump(final, f, indent=2)

    # visuals
    for name, data in [
        ("precision", P_list),
        ("recall", R_list),
        ("f1", F1_list),
        ("accuracy", ACC_list),
        ("bleu", BL_list),
        ("codebleu", CB_list)
    ]:
        plt.figure(figsize=(6, 4))
        sns.histplot(data, bins=20, kde=True)
        plt.title(f"{kind.capitalize()} {name.capitalize()} Dist.")
        plt.savefig(os.path.join(vis_dir, f"{kind}_{name}_hist.png"))
        plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(final.keys()), y=list(final.values()))
    plt.ylim(0, 1)
    plt.title(f"{kind.capitalize()} Overall")
    plt.savefig(os.path.join(vis_dir, f"{kind}_overall.png"))
    plt.close()

    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_dir", default="student_output")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_src_length", type=int, default=1024)
    p.add_argument("--max_tgt_length", type=int, default=512)
    args = p.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Device:", device)

    # Load JSONL data
    with open(args.data_path) as f:
        data = [
            json.loads(l) for l in f
            if l.strip() and not l.strip().startswith('{"header"')
        ]
    print(f"Loaded {len(data)} examples")

    # Tokenizer & Model
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    ds = AssertionDataset(data, tok, max_src=args.max_src_length, max_tgt=args.max_tgt_length)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Prepare output dirs (timestamped)
    name = os.path.basename(args.model_dir.rstrip("/"))
    ts = torch.tensor(0)  # dummy to avoid unused import
    base = os.path.join(args.output_dir, f"{name}_{torch.tensor(0)}")
    stu_dir = os.path.join(base, "student")
    tea_dir = os.path.join(base, "teacher")
    os.makedirs(stu_dir, exist_ok=True)
    os.makedirs(tea_dir, exist_ok=True)

    print("\n==> Student evaluation")
    stu_m = generate_and_evaluate(model, tok, dl, device, "student", stu_dir)
    print("Student metrics:", stu_m)

    print("\n==> Teacher evaluation")
    tea_m = generate_and_evaluate(model, tok, dl, device, "teacher", tea_dir)
    print("Teacher metrics:", tea_m)


if __name__ == "__main__":
    main()
