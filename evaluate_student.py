#!/usr/bin/env python
"""
Evaluate a distilled student T5 model and also analyze the teacher predictions,
computing per-sample precision/recall/F1/accuracy/similarity and producing
predictions+metrics, plus summary visuals, under:

  <output_dir>/<model_name>/
      student/
        student_predictions.jsonl
        student_prediction_metrics.json
        visualizations/
      teacher/
        teacher_prediction_metrics.json
        visualizations/
"""
import argparse, json, os, re
import torch, numpy as np
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns

# ---- utility functions ----

def normalize_assertion(a):
    a = re.sub(r"\s+", " ", a).strip()
    a = re.sub(r"assertEquals\(\s*[^,]+,\s*([^)]+)\)", r"assertEquals(VALUE, \1)", a)
    return re.sub(r"assert(Equals|That|True|False)",
                  lambda m: "assert"+m.group(1),
                  a, flags=re.IGNORECASE)

def evaluate_assertions(generated, reference):
    # split on semicolon/newline, re-append ';'
    gen = [s.strip()+";" for s in re.split(r";|\n", generated) if s.strip()]
    ref = [s.strip()+";" for s in re.split(r";|\n", reference) if s.strip()]
    gen_n = [normalize_assertion(x) for x in gen]
    ref_n = [normalize_assertion(x) for x in ref]
    # exact matches
    tp = sum(1 for x in gen_n if x in ref_n)
    # similarity per gen
    sim = []
    for x in gen:
        best = 0
        for r in ref:
            best = max(best, SequenceMatcher(None,x,r).ratio())
        sim.append(best)
    P = tp/len(gen_n) if gen_n else 0.0
    R = tp/len(ref_n) if ref_n else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    ACC = tp/max(len(gen_n),len(ref_n)) if max(len(gen_n),len(ref_n))>0 else 0.0
    return {"precision":P, "recall":R, "f1":F1, "accuracy":ACC,
            "avg_similarity": (sum(sim)/len(sim) if sim else 0.0),
            "exact_matches":tp, "generated_count":len(gen_n),
            "reference_count":len(ref_n), "similarity_scores":sim}

# ---- dataset ----

class AssertionDataset(Dataset):
    def __init__(self, data, tokenizer, max_src=1024, max_tgt=512):
        self.data,self.tok,self.max_src,self.max_tgt = data,tokenizer,max_src,max_tgt
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        e = self.data[i]
        src = f"FOCAL CODE:\n{e['focal_file']}\n\nTEST METHOD:\n{e['test_method_masked']}"
        return {
          "input_ids": self.tok(src, padding="max_length", truncation=True,
                                max_length=self.max_src, return_tensors="pt").input_ids.squeeze(0),
          "attention_mask": self.tok(src, padding="max_length", truncation=True,
                                     max_length=self.max_src, return_tensors="pt").attention_mask.squeeze(0),
          "original": e
        }

def collate_fn(batch):
    return {
      "input_ids":    torch.stack([b["input_ids"] for b in batch]),
      "attention_mask":torch.stack([b["attention_mask"] for b in batch]),
      "originals":    [b["original"] for b in batch]
    }

# ---- generation & evaluation ----

def generate_and_evaluate(model, tok, loader, device, kind, out_dir):
    """
    kind: "student" or "teacher"
    for student: generate model outputs; for teacher: use e["predicted_assertions"]
    """
    model.to(device).eval()
    preds_file = os.path.join(out_dir, f"{kind}_predictions.jsonl")
    metrics_file = os.path.join(out_dir, f"{kind}_prediction_metrics.json")
    vis_dir      = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    writer = open(preds_file, "w")
    # per-example lists
    precisions, recalls, f1s, accs, sims = [],[],[],[],[]

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Generating {kind}"):
            inp = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            origs = batch["originals"]

            if kind=="student":
                gen_ids = model.generate(inp, attention_mask=msk,
                                         max_length=loader.dataset.max_tgt,
                                         num_beams=4, early_stopping=True)
                outputs = [tok.decode(g,skip_special_tokens=True) for g in gen_ids]
            else:
                # teacher uses provided string
                outputs = [o.get("predicted_assertions","") for o in origs]

            for o, out_str in zip(origs, outputs):
                ref_str = o.get("original_target","")
                m = evaluate_assertions(out_str, ref_str)
                # record per-sample
                precisions.append(m["precision"])
                recalls.append(m["recall"])
                f1s.append(m["f1"])
                accs.append(m["accuracy"])
                sims.extend(m["similarity_scores"])
                # write record
                rec = {
                  **o,
                  f"{kind}_prediction": out_str,
                  "prediction_metrics": {
                    "precision":m["precision"],
                    "recall":m["recall"],
                    "f1":m["f1"],
                    "accuracy":m["accuracy"],
                    "avg_similarity":m["avg_similarity"]
                  }
                }
                writer.write(json.dumps(rec)+"\n")

    writer.close()

    # overall = mean of per-sample scores
    final = {
      "precision": float(np.mean(precisions) if precisions else 0.0),
      "recall":    float(np.mean(recalls)    if recalls else 0.0),
      "f1":        float(np.mean(f1s)        if f1s else 0.0),
      "accuracy":  float(np.mean(accs)       if accs else 0.0),
      "avg_similarity": float(np.mean(sims)  if sims else 0.0)
    }
    # save metrics
    with open(metrics_file,"w") as f:
        json.dump(final, f, indent=2)

    # visuals
    # bar chart
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(final.keys()), y=list(final.values()))
    plt.ylim(0,1); plt.title(f"{kind.capitalize()} Overall Metrics")
    plt.savefig(os.path.join(vis_dir, f"{kind}_overall_metrics.png"))
    plt.close()

    # histograms
    for name, data in [("precision",precisions),
                       ("recall",   recalls),
                       ("f1",       f1s),
                       ("accuracy", accs)]:
        plt.figure(figsize=(6,4))
        sns.histplot(data, bins=20, kde=True)
        plt.title(f"{kind.capitalize()} {name.capitalize()} Distribution")
        plt.xlabel(name.capitalize()); plt.ylabel("Count")
        plt.savefig(os.path.join(vis_dir, f"{kind}_{name}_distribution.png"))
        plt.close()

    return final

# ---- main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  required=True)
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--output_dir", default="student_output")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_src_length", type=int, default=1024)
    p.add_argument("--max_tgt_length", type=int, default=512)
    args = p.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print("Device:",device)

    # load data (skip header)
    with open(args.data_path) as f:
        data = [json.loads(l) for l in f if l.strip() and not l.strip().startswith('{"header"')]
    print(f"Loaded {len(data)} examples")

    # model + tokenizer
    tok   = AutoTokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    # dataloader
    ds = AssertionDataset(data, tok,
                           max_src=args.max_src_length,
                           max_tgt=args.max_tgt_length)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate_fn)

    # prepare folders
    name = os.path.basename(args.model_dir.rstrip("/"))
    base = os.path.join(args.output_dir, name)
    stu_dir = os.path.join(base, "student")
    tea_dir = os.path.join(base, "teacher")
    os.makedirs(stu_dir, exist_ok=True)
    os.makedirs(tea_dir, exist_ok=True)

    # student
    print("\n==> Student evaluation")
    stu_metrics = generate_and_evaluate(model, tok, dl, device, kind="student", out_dir=stu_dir)
    print("Student metrics:",stu_metrics)

    # teacher
    print("\n==> Teacher evaluation")
    tea_metrics = generate_and_evaluate(model, tok, dl, device, kind="teacher", out_dir=tea_dir)
    print("Teacher metrics:",tea_metrics)

if __name__=="__main__":
    main()
