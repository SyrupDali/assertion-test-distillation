#!/usr/bin/env python
"""
Evaluate a distilled student T5 model and also analyze the teacher predictions,
computing per-sample precision/recall/F1/accuracy/similarity/BLEU and producing
predictions+metrics, plus summary visuals.
"""
import argparse, json, os, re
import torch, numpy as np
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import seaborn as sns
from codebleu import calc_codebleu

smooth_fn = SmoothingFunction().method1

def normalize_assertion(a):
    a = re.sub(r"\s+", " ", a).strip()
    a = re.sub(r"assertEquals\(\s*[^,]+,\s*([^)]+)\)",
               r"assertEquals(VALUE, \1)", a)
    return re.sub(r"assert(Equals|That|True|False)",
                  lambda m: "assert"+m.group(1),
                  a, flags=re.IGNORECASE)

def evaluate_assertions(generated, reference):
    gen = [s.strip()+";" for s in re.split(r";|\n", generated) if s.strip()]
    ref = [s.strip()+";" for s in re.split(r";|\n", reference) if s.strip()]
    gen_n = [normalize_assertion(x) for x in gen]
    ref_n = [normalize_assertion(x) for x in ref]
    tp = sum(1 for x in gen_n if x in ref_n)
    # similarity
    sim = []
    for x in gen:
        best = 0
        for r in ref:
            best = max(best, SequenceMatcher(None, x, r).ratio())
        sim.append(best)
    P = tp/len(gen_n) if gen_n else 0.0
    R = tp/len(ref_n) if ref_n else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    ACC = tp/max(len(gen_n),len(ref_n)) if max(len(gen_n),len(ref_n))>0 else 0.0

    # # BLEU: token‐level over the whole generated string against all refs
    # # strip trailing “;” and split by whitespace
    # cand_tokens = generated.strip().rstrip(";").split()
    # ref_tokens  = [r.strip().rstrip(";").split() for r in ref]
    # try:
    #     BLEU = sentence_bleu(ref_tokens, cand_tokens,
    #                          smoothing_function=smooth_fn)
    # except Exception:
    #     BLEU = 0.0
    
    # join all assertions with newlines
    joined_gen = "\n".join(gen_n)
    joined_ref = "\n".join(ref_n)

    # tokenize on whitespace
    cand_tokens = joined_gen.split()
    ref_tokens  = [joined_ref.split()]

    try:
        BLEU = sentence_bleu(
            ref_tokens,
            cand_tokens,
            smoothing_function=smooth_fn,
            # e.g. weights=(0.25,0.25,0.25,0.25)
        )
    except Exception:
        BLEU = 0.0


    return {
      "precision":P, "recall":R, "f1":F1, "accuracy":ACC,
      "avg_similarity":(sum(sim)/len(sim) if sim else 0.0),
      "bleu":BLEU,
      "exact_matches":tp,
      "generated_count":len(gen_n),
      "reference_count":len(ref_n),
      "similarity_scores":sim
    }

class AssertionDataset(Dataset):
    def __init__(self, data, tokenizer, max_src=1024, max_tgt=512):
        self.data,self.tok,self.max_src,self.max_tgt = data,tokenizer,max_src,max_tgt
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        e = self.data[i]
        src = f"FOCAL CODE:\n{e['focal_file']}\n\nTEST METHOD:\n{e['test_method_masked']}"
        tokens = self.tok(src, padding="max_length", truncation=True,
                          max_length=self.max_src, return_tensors="pt")
        return {
          "input_ids": tokens.input_ids.squeeze(0),
          "attention_mask": tokens.attention_mask.squeeze(0),
          "original": e
        }

def collate_fn(batch):
    return {
      "input_ids":    torch.stack([b["input_ids"] for b in batch]),
      "attention_mask":torch.stack([b["attention_mask"] for b in batch]),
      "originals":    [b["original"] for b in batch]
    }

def generate_and_evaluate(model, tok, loader, device, kind, out_dir):
    """
    kind: "student" or "teacher"
    """
    model.to(device).eval()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preds_file  = os.path.join(out_dir, f"{kind}_preds.jsonl")
    metrics_file= os.path.join(out_dir, f"{kind}_metrics.json")
    vis_dir     = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    writer = open(preds_file, "w")
    P_list,R_list,F1_list,ACC_list,BL_list,SIM_list = [],[],[],[],[],[]

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Gen {kind}"):
            inp = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            origs = batch["originals"]

            if kind=="student":
                gen_ids = model.generate(inp, attention_mask=msk,
                                         max_length=loader.dataset.max_tgt,
                                         num_beams=4, early_stopping=True)
                outs = [tok.decode(g,skip_special_tokens=True) for g in gen_ids]
            else:
                outs = [o.get("predicted_assertions","") for o in origs]

            for o, out_str in zip(origs, outs):
                m = evaluate_assertions(out_str, o.get("original_target",""))
                P_list.append(m["precision"])
                R_list.append(m["recall"])
                F1_list.append(m["f1"])
                ACC_list.append(m["accuracy"])
                SIM_list.extend(m["similarity_scores"])
                BL_list.append(m["bleu"])

                rec = { **o,
                        f"{kind}_prediction": out_str,
                        "metrics": {
                          "precision":m["precision"],
                          "recall":m["recall"],
                          "f1":m["f1"],
                          "accuracy":m["accuracy"],
                          "avg_similarity":m["avg_similarity"],
                          "bleu": m["bleu"],
                        }}
                writer.write(json.dumps(rec)+"\n")

    writer.close()

    final = {
      "precision": float(np.mean(P_list)),
      "recall":    float(np.mean(R_list)),
      "f1":        float(np.mean(F1_list)),
      "accuracy":  float(np.mean(ACC_list)),
      "avg_similarity": float(np.mean(SIM_list)),
      "avg_bleu":  float(np.mean(BL_list))
    }
    with open(metrics_file,"w") as f:
        json.dump(final, f, indent=2)

    # visuals
    for name, data in [
        ("precision",P_list),
        ("recall",R_list),
        ("f1",F1_list),
        ("accuracy",ACC_list),
        ("bleu",BL_list)
    ]:
        plt.figure(figsize=(6,4))
        sns.histplot(data, bins=20, kde=True)
        plt.title(f"{kind.capitalize()} {name.capitalize()} Dist.")
        plt.savefig(os.path.join(vis_dir, f"{kind}_{name}_hist.png"))
        plt.close()

    # bar
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(final.keys()), y=list(final.values()))
    plt.ylim(0,1); plt.title(f"{kind.capitalize()} Overall")
    plt.savefig(os.path.join(vis_dir, f"{kind}_overall.png"))
    plt.close()

    return final

from datetime import datetime
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_dir", default="student_output")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--max_src_length", type=int, default=1024)
    p.add_argument("--max_tgt_length", type=int, default=512)
    args = p.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Device:",device)

    # load data
    with open(args.data_path) as f:
        data = [json.loads(l) for l in f
                if l.strip() and not l.strip().startswith('{"header"')]
    print(f"Loaded {len(data)} examples")

    # tokenizer & model
    tok   = AutoTokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    ds = AssertionDataset(data, tok,
                          max_src=args.max_src_length,
                          max_tgt=args.max_tgt_length)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate_fn)

    # prepare output (with timestamp)
    name = os.path.basename(args.model_dir.rstrip("/"))
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.output_dir, f"{name}_{ts}")
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

if __name__=="__main__":
    main()
