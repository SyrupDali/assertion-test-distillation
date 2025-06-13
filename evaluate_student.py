#!/usr/bin/env python
"""
Evaluate a distilled student T5 model and also analyze the teacher predictions,
computing micro‐averaged precision/recall/F1, per‐example accuracy (macro),
per‐assertion similarity (micro), per‐example CodeBLEU (macro),
per‐assertion AST validity (micro), per‐assertion token‐accuracy (micro),
plus summary visuals.

This version supports limiting the number of samples via --max_samples.
"""
import argparse
import json
import os
import re
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns

import javalang

# Try importing CodeBLEU; if unavailable, fall back to 0.0
try:
    from codebleu import calc_codebleu
    _CODEBLEU_AVAILABLE = True
except ImportError:
    print("Warning: could not import calc_codebleu; all CodeBLEU scores will be set to 0.0")
    _CODEBLEU_AVAILABLE = False


def normalize_assertion(a: str) -> str:
    a = re.sub(r"\s+", " ", a).strip()
    try:
        # wrap in dummy class so javalang can parse
        stmt = a if a.endswith(";") else a + ";"
        snippet = f"public class Dummy {{ public void m() {{ {stmt} }} }}"
        javalang.parse.parse(snippet)
        normalized = a
    except:
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
    gen_norm = [normalize_assertion(x.strip()) for x in generated_list if x.strip()]
    ref_norm = [normalize_assertion(x.strip()) for x in reference_list if x.strip()]

    tp = sum(1 for g in gen_norm if g in ref_norm)
    gen_count = len(gen_norm)
    ref_count = len(ref_norm)

    P = tp / gen_count if gen_count else 0.0
    R = tp / ref_count if ref_count else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    ACC = tp / max(gen_count, ref_count) if max(gen_count, ref_count) > 0 else 0.0

    sim_scores = []
    for g in gen_norm:
        best = 0.0
        for r in ref_norm:
            best = max(best, SequenceMatcher(None, g, r).ratio())
        sim_scores.append(best)
    sim_avg = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0

    joined_gen = "\n".join(gen_norm)
    joined_ref = "\n".join(ref_norm)
    
    if _CODEBLEU_AVAILABLE:
        if not joined_gen.strip() or not joined_ref.strip():
            CODEBLEU = 0.0
        else:
            try:
                cb_result_dict = calc_codebleu(
                    references=[[joined_ref]],
                    predictions=[joined_gen],
                    lang="java"
                )
                CODEBLEU = cb_result_dict.get("codebleu", 0.0)
            except Exception as e:
                # Errors will still be printed, which is good practice
                print(f"!!! CodeBLEU calculation FAILED. Error: {e} !!!")
                CODEBLEU = 0.0
    else:
        CODEBLEU = 0.0

    ast_valid_flags = []
    for g in gen_norm:
        stmt = g if g.endswith(";") else g + ";"
        snippet = f"public class Dummy {{ public void m() {{ {stmt} }} }}"
        try:
            javalang.parse.parse(snippet)
            ast_valid_flags.append(1)
        except:
            ast_valid_flags.append(0)

    token_acc_scores = []
    for g in gen_norm:
        g_tokens = g.split()
        best_acc = 0.0
        for r in ref_norm:
            r_tokens = r.split()
            if not r_tokens:
                acc = 1.0 if not g_tokens else 0.0
            else:
                matches = sum(1 for i in range(min(len(g_tokens), len(r_tokens))) if g_tokens[i] == r_tokens[i])
                acc = matches / len(r_tokens)
            best_acc = max(best_acc, acc)
        token_acc_scores.append(best_acc)

    return {
        "exact_matches":       tp,
        "generated_count":     gen_count,
        "reference_count":     ref_count,
        "precision":           P,
        "recall":              R,
        "f1":                  F1,
        "accuracy":            ACC,
        "similarity_scores":   sim_scores,
        "similarity_score_avg": sim_avg,
        "codebleu":            CODEBLEU,
        "ast_valid_flags":     ast_valid_flags,
        "token_acc_scores":    token_acc_scores
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
        src = f"FOCAL METHOD:\n{e['focal_method']}\n\nTEST METHOD:\n{e['test_method_masked']}"
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
    model.to(device).eval()
    os.makedirs(out_dir, exist_ok=True)
    total_tp = total_gen = total_ref = 0
    ACC_list, CB_list, SIM_all, AST_all, TOKACC_all = [], [], [], [], []

    with open(os.path.join(out_dir, f"{kind}_preds.jsonl"), "w") as writer:
        for batch in tqdm(loader, desc=f"Gen {kind}"):
            inp = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            origs = batch["originals"]

            if kind == "student":
                gen_ids = model.generate(
                    inp, attention_mask=msk,
                    max_length=loader.dataset.max_tgt,
                    num_beams=4, early_stopping=True
                )
                gen_lists = [
                    [s.strip() + ";" for s in re.split(r";|\n", tok.decode(g, skip_special_tokens=True)) if s.strip()]
                    for g in gen_ids
                ]
            else: # kind == "teacher"
                raw_teacher_preds = [o.get("predicted_assertions", []) for o in origs]
                gen_lists = []
                for pred in raw_teacher_preds:
                    if isinstance(pred, str):
                        processed_list = [s.strip() + ";" for s in re.split(r";|\n", pred) if s.strip()]
                        gen_lists.append(processed_list)
                    else:
                        gen_lists.append(pred)

            ref_lists = [o.get("assertions", []) for o in origs]

            for o, gen_list, ref_list in zip(origs, gen_lists, ref_lists):
                m = evaluate_assertions(gen_list, ref_list)
                total_tp   += m["exact_matches"]
                total_gen  += m["generated_count"]
                total_ref  += m["reference_count"]
                ACC_list.append(m["accuracy"])
                CB_list.append(m["codebleu"])
                SIM_all.extend(m["similarity_scores"])
                AST_all.extend(m["ast_valid_flags"])
                TOKACC_all.extend(m["token_acc_scores"])

                rec = {
                    **o,
                    f"{kind}_prediction": gen_list,
                    "metrics": {
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1": m["f1"],
                        "accuracy": m["accuracy"],
                        "avg_similarity": m["similarity_score_avg"],
                        "codebleu": m["codebleu"],
                        "ast_validity_avg": (
                            sum(m["ast_valid_flags"]) / len(m["ast_valid_flags"])
                            if m["ast_valid_flags"] else 0.0
                        ),
                        "token_accuracy_avg": (
                            sum(m["token_acc_scores"]) / len(m["token_acc_scores"])
                            if m["token_acc_scores"] else 0.0
                        )
                    }
                }
                writer.write(json.dumps(rec) + "\n")

    micro_P = total_tp/total_gen if total_gen>0 else 0.0
    micro_R = total_tp/total_ref if total_ref>0 else 0.0
    micro_F1 = (
        2*micro_P*micro_R/(micro_P+micro_R)
        if (micro_P+micro_R)>0 else 0.0
    )
    avg_accuracy        = float(np.mean(ACC_list))    if ACC_list    else 0.0
    avg_codebleu        = float(np.mean(CB_list))     if CB_list     else 0.0
    avg_similarity      = float(np.mean(SIM_all))    if SIM_all     else 0.0
    avg_ast_validity    = float(np.mean(AST_all))    if AST_all     else 0.0
    avg_token_accuracy  = float(np.mean(TOKACC_all)) if TOKACC_all  else 0.0

    final = {
        "precision":         micro_P,
        "recall":            micro_R,
        "f1":                micro_F1,
        "accuracy":          avg_accuracy,
        "avg_similarity":    avg_similarity,
        "avg_codebleu":      avg_codebleu,
        "avg_ast_validity":  avg_ast_validity,
        "avg_token_accuracy":avg_token_accuracy
    }

    with open(os.path.join(out_dir, f"{kind}_metrics.json"), "w") as f:
        json.dump(final, f, indent=2)

    # visualizations
    for name, data in [
        ("accuracy", ACC_list),
        ("codebleu", CB_list),
        ("similarity", SIM_all),
        ("ast_validity", AST_all),
        ("token_accuracy", TOKACC_all)
    ]:
        if data:
            plt.figure(figsize=(6,4))
            sns.histplot(data, bins=20, kde=True)
            plt.title(f"{kind.capitalize()} {name.capitalize()} Dist.")
            plt.savefig(os.path.join(out_dir, f"{kind}_{name}_hist.png"))
            plt.close()

    labels = [
        "precision","recall","f1","accuracy",
        "avg_similarity","avg_codebleu",
        "avg_ast_validity","avg_token_accuracy"
    ]
    values = [final[k] for k in labels]
    plt.figure(figsize=(10,4))
    sns.barplot(x=labels,y=values)
    plt.ylim(0,1)
    plt.xticks(rotation=30)
    plt.title(f"{kind.capitalize()} Overall Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,f"{kind}_overall.png"))
    plt.close()

    return final


def main():
    # Suppress informational warnings from the codebleu library
    logging.getLogger().setLevel(logging.ERROR)
    
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",     required=True)
    p.add_argument("--model_dir",     required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--max_src_length",type=int, default=1024)
    p.add_argument("--max_tgt_length",type=int, default=512)
    p.add_argument("--max_samples",   type=int, default=None,
                   help="If set, only evaluate this many examples")
    args = p.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    with open(args.data_path) as f:
        all_data = [
            json.loads(l) for l in f
            if l.strip() and not l.strip().startswith('{"header"')
        ]

    if args.max_samples:
        data = all_data[: args.max_samples]
    else:
        data = all_data

    print(f"Loaded {len(data)} examples (max_samples={args.max_samples})")

    tok   = AutoTokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    ds = AssertionDataset(data, tok,
                          max_src=args.max_src_length,
                          max_tgt=args.max_tgt_length)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate_fn)

    base    = os.path.join(args.output_dir,
                           os.path.basename(args.model_dir.rstrip("/")))
    stu_dir = os.path.join(base, "student")
    tea_dir = os.path.join(base, "teacher")

    print("\n==> Student evaluation")
    stu_m = generate_and_evaluate(model, tok, dl, device, "student", stu_dir)
    print("Student metrics:", stu_m)

    print("\n==> Teacher evaluation")
    tea_m = generate_and_evaluate(model, tok, dl, device, "teacher", tea_dir)
    print("Teacher metrics:", tea_m)


if __name__ == "__main__":
    main()