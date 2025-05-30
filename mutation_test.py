#!/usr/bin/env python
"""
For each example in your validation JSONL:
 - (preview mode) for the first N examples print out the filled test code
   instead of running PIT
 - otherwise for ground-truth / teacher / student fill in the placeholders,
   run PIT, and collect mutation kill‐rate.

Usage:
  python mutation_test.py \
    --data_path validation.jsonl \
    --student_preds student_output/codet5-small_*/student/student_preds.jsonl \
    --pom_template pom_template \
    --output mutation_scores.json \
    [--preview_count 5]
"""
import argparse, json, os, re, shutil, subprocess, tempfile
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

PLACEHOLDER_RE = re.compile(r"//\s*ASSERTION PLACEHOLDER")

def insert_assertions(masked_method: str, assertions: list[str]) -> str:
    lines, out, idx = masked_method.splitlines(), [], 0
    for ln in lines:
        if PLACEHOLDER_RE.search(ln):
            indent = ln[:ln.find('//')]
            if idx < len(assertions):
                out.append(indent + assertions[idx])
                idx += 1
        else:
            out.append(ln)
    if idx < len(assertions):
        # any leftover assertions: insert before final '}'
        for i in range(len(out)-1, -1, -1):
            if out[i].strip() == "}":
                insert_at = i
                break
        else:
            insert_at = len(out)
        # detect method indent
        method_indent = ""
        for ln in out:
            if ln.strip().startswith("public"):
                method_indent = ln[:ln.find("public")] + "    "
                break
        if not method_indent:
            method_indent = "    "
        for j in range(idx, len(assertions)):
            out.insert(insert_at, method_indent + assertions[j])
            insert_at += 1
    return "\n".join(out)

def run_pit(project_dir: str, target_fqn: str) -> float:
    """
    1) mvn test-compile → if that fails, return -1.0
    2) mvn pitest → if that fails (e.g. no mutations), return 0.0
    3) parse mutations.xml and return killed/total
    """
    # 1) ensure compile
    try:
        subprocess.run(
            ["mvn", "clean", "test-compile", "-q"],
            cwd=project_dir, check=True
        )
    except subprocess.CalledProcessError:
        return -1.0

    # 2) run pitest
    try:
        subprocess.run(
            ["mvn", "org.pitest:pitest-maven:mutationCoverage",
             f"-DtargetClasses={target_fqn}", "-q"],
            cwd=project_dir, check=True
        )
    except subprocess.CalledProcessError:
        # pitest error (often no mutations found)
        return 0.0

    # 3) parse report
    rpt = os.path.join(project_dir, "target", "pit-reports", "mutations.xml")
    try:
        tree = ET.parse(rpt)
        cov  = tree.find(".//mutationCoverage")
        killed = int(cov.get("killed", "0"))
        total  = int(cov.get("total",  "0"))
        return (killed/total) if total>0 else 0.0
    except Exception:
        return 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",     required=True)
    p.add_argument("--student_preds", required=True)
    p.add_argument("--pom_template",  required=True)
    p.add_argument("--output",        default="mutation_scores.json")
    p.add_argument("--preview_count", type=int, default=0,
                   help="If >0, just print the first N filled tests and exit")
    args = p.parse_args()

    # load validation + student predictions in order
    with open(args.data_path) as f:
        val = [json.loads(l) for l in f
               if l.strip() and not l.strip().startswith('{"header"')]
    with open(args.student_preds) as f:
        stu_preds = [json.loads(l) for l in f if l.strip()]

    assert len(val) == len(stu_preds), "validation and student_preds must have same length"

    # preview?
    if args.preview_count > 0:
        for i, ex in enumerate(val[:args.preview_count]):
            print(f"\n=== Example {i} preview ===")
            print("Focal code:\n", ex["focal_file"])
            for kind, src_field in [
                ("ground",  "original_target"),
                ("teacher", "predicted_assertions"),
                ("student", "student_prediction")
            ]:
                raw = (ex[src_field] if kind!="student"
                       else stu_preds[i]["student_prediction"])
                asserts = [s.strip()+";"
                           for s in re.split(r";|\n", raw) if s.strip()]
                print(f"\n--- {kind} filled test ---")
                print(insert_assertions(ex["test_method_masked"], asserts))
        return

    # full mutation run
    scores = {"ground":[], "teacher":[], "student":[]}
    for i, ex in enumerate(val):
        cls_code = ex["focal_file"]
        pkg_m    = re.search(r"package\s+([^;]+);", cls_code)
        pkg      = pkg_m.group(1) if pkg_m else ""
        name_m   = re.search(r"class\s+(\w+)", cls_code)
        cn       = name_m.group(1) if name_m else f"Cl{i}"

        for kind, src_field in [
            ("ground",  "original_target"),
            ("teacher", "predicted_assertions"),
            ("student", "student_prediction")
        ]:
            raw = (ex[src_field] if kind!="student"
                   else stu_preds[i]["student_prediction"])
            asserts = [s.strip()+";" for s in re.split(r";|\n", raw) if s.strip()]
            filled = insert_assertions(ex["test_method_masked"], asserts)

            with tempfile.TemporaryDirectory() as tmp:
                # copy POM template
                shutil.copytree(args.pom_template, tmp, dirs_exist_ok=True)
                main_dir = os.path.join(tmp, "src", "main", "java", *pkg.split("."))
                test_dir = os.path.join(tmp, "src", "test", "java", *pkg.split("."))
                os.makedirs(main_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)

                # write class under test
                with open(os.path.join(main_dir, f"{cn}.java"), "w") as fw:
                    fw.write(cls_code)

                # wrap into Test class
                test_code = f"""\
package {pkg};

import org.junit.Test;
import static org.junit.Assert.*;

public class {cn}Test {{
{filled}
}}
"""
                with open(os.path.join(test_dir, f"{cn}Test.java"), "w") as fw:
                    fw.write(test_code)

                fqcn = pkg + "." + cn
                score = run_pit(tmp, fqcn)

            scores[kind].append(score)

    # aggregate + dump
    out = { k: float(np.mean(v)) for k,v in scores.items() }
    out["per_example"] = scores
    with open(args.output, "w") as fw:
        json.dump(out, fw, indent=2)

    print("✔ Saved mutation scores →", args.output)
    for k in ["ground","teacher","student"]:
        print(f"  {k}: {out[k]:.3f}")

if __name__=="__main__":
    main()
