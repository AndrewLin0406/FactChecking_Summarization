"""
Evaluation Step:
    EVALUATION: Give the collected information, into a 6 class classification task. Into 
        the following class:
            True = accurate and not missing important context
            Mostly true = mostly accurate but needs minor clarification
            Half true = partially accurate but leaves out important details
            Mostly false = contains an element of truth but ignores critical facts
            False = inaccurate
            Pants on fire = false and absurdly misleading
"""

import time, torch, json, re, ast, os, argparse
from urllib.parse import urljoin
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

hf_logging.set_verbosity_error()

# ====================================================================================
# HELPERS
# ====================================================================================

df = None

LABELS = [
    "True",
    "Mostly true",
    "Half true",
    "Mostly false",
    "False",
    "Pants on fire"
]

# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

def map2(label):
    if label in ["True", "Mostly true", "Half true"]:
        return "True"
    elif label in ["False", "Mostly false", "Pants on fire"]:
        return "False"
    else:
        return "False"


def map4(label):
    if label in ["True", "Mostly true"]:
        return "True"
    elif label == "Half true":
        return "Half true"
    elif label in ["Mostly false", "False"]:
        return "False"
    elif label == "Pants on fire":
        return "Pants on fire"
    else:
        return "UNKNOWN"

def map3(label):
    if label in ["True", "Mostly True"]:
        return "True"
    elif label in ["Half true", "Mostly false"]:
        return "Mixed"
    elif label in ["False", "Pants on fire"]:
        return "False"
    else:
        return "UNKNOWN"
    
def format_fact_item(item):
    if isinstance(item, str):
        return item

    fact = item.get("fact", "")
    importance = item.get("importance", "")
    support_type = item.get("support_type", None)

    if support_type:
        return f"[importance={importance}, support_type={support_type}] {fact}"
    return f"[importance={importance}] {fact}"


def build_structured_evidence(extracted):
    return (
        "Supporting facts:\n- " + "\n- ".join(
            format_fact_item(x) for x in extracted.get("supporting_facts", [])
        ) + "\n\n"
        "Contradicting facts:\n- " + "\n- ".join(
            format_fact_item(x) for x in extracted.get("weakening_facts", [])
        ) + "\n\n"
        "Missing context:\n- " + "\n- ".join(
            format_fact_item(x) for x in extracted.get("missing_context", [])
        )
    )

# ====================================================================================
# LOAD DATA
# ====================================================================================

def load_data(dataSize):
    return pd.read_json("Summary/METHOD1_Results_2.json")

# ====================================================================================
# MODELS
# ====================================================================================

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

eval_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token=HF_TOKEN
)

eval_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token=HF_TOKEN,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    attn_implementation="sdpa",
)

eval_model = eval_model.to(device)

if eval_tokenizer.pad_token is None:
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

eval_model.config.pad_token_id = eval_tokenizer.pad_token_id

# ====================================================================================
# PROMPTS
# ====================================================================================

def build_eval_messages(statement: str, summary: str):
    return [
        {
            "role": "system",
            "content": (
                "You are a professional fact-checker.\n\n"

                "Your task is to evaluate the accuracy of a statement "
                "using ONLY the provided article summary.\n\n"

                "Use the following Truth Scale definitions:\n\n"

                "Truth Scale:\n"
                "- True: The statement is accurate and nothing significant is missing.\n"
                "- Mostly true: The statement is accurate but needs clarification or additional information.\n"
                "- Half true: The statement is partially accurate but leaves out important details or takes things out of context.\n"
                "- Mostly false: The statement contains an element of truth but ignores critical facts that would give a different impression.\n"
                "- False: The statement is not accurate.\n"
                "- Pants on fire: The statement is not accurate and makes a ridiculous or wildly inaccurate claim.\n\n"

                "Important Rules:\n"
                "- Use ONLY the information in the article summary.\n"
                "- Do NOT use outside knowledge.\n"
                "- Choose the single best label from the Truth Scale.\n"
                "- Choose Half true only when the support and contradiction are genuinely balanced.\n"
                "- Choose Mostly false when there is a small element of truth but the main impression is wrong.\n"
                "- Choose False when the main factual claim is directly contradicted.\n"
                "- Choose Pants on fire when the claim is fabricated, absurd, impossible, or based on satire.\n"
                "- Return ONLY the label exactly as written.\n"
                "- Do not explain your reasoning.\n\n"

                "Valid labels:\n"
                "True\n"
                "Mostly true\n"
                "Half true\n"
                "Mostly false\n"
                "False\n"
                "Pants on fire"
            )
        },
        {
            "role": "user",
            "content": (
                f"Statement:\n{statement}\n\n"
                f"Article summary:\n{summary}\n\n"
                "Answer:"
            )
        }
    ]

# ====================================================================================
# STEPS
# ====================================================================================

# EVALUATION
def classify_statement(statement: str, evidence: str) -> str:
    messages = build_eval_messages(statement, evidence)

    inputs = eval_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(eval_model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = eval_model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=eval_tokenizer.pad_token_id,
            eos_token_id=eval_tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0, input_len:]

    out = eval_tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    first_line = out.splitlines()[0].strip()

    for label in LABELS:
        if first_line.lower() == label.lower():
            return label

    raw_lower = out.lower()

    for label in LABELS:
        if label.lower() in raw_lower:
            return label

    print("INVALID OUTPUT:", out)

    return "INVALID"

# ====================================================================================
# RUN
# ====================================================================================

def run(eval_df, columnName, columnName2, mode=0, output_file="results.json"):
    # mode 0 -> summary only
    # mode 1 -> formatted extraction only
    # mode 2 -> both
    y_true_so_far = []
    y_pred_so_far = []
    results = []

    # store per-example outputs
    count = 0

    for _, row in eval_df.iterrows():
        count += 1

        statement = row["statement"]
        summary = row[columnName]
        extracted = row[columnName2]
        gold = row["verdict"]

        match mode:
            case 0:
                evidence = summary
            case 1: 
                evidence = build_structured_evidence(extracted)
            case 2:
                evidence = summary + "\n\n" + build_structured_evidence(extracted)
        
        pred = classify_statement(statement, evidence)

        if pred == "INVALID":
            print(f"INVALID at index {count}")
            continue

        y_true_so_far.append(gold)
        y_pred_so_far.append(pred)

        # save per-example result
        results.append({
            "statement": statement,
            "gold": gold,
            "prediction": pred,
            "evidence": evidence
        })

        # periodic metrics print
        if count % 100 == 0 or count == len(eval_df):

            acc = accuracy_score(
                y_true_so_far,
                y_pred_so_far
            )

            macro_f1 = f1_score(
                y_true_so_far,
                y_pred_so_far,
                labels=LABELS,
                average="macro"
            )

            # binary
            y_true_bin = [map2(i) for i in y_true_so_far]
            y_pred_bin = [map2(i) for i in y_pred_so_far]

            accBin = accuracy_score(
                y_true_bin,
                y_pred_bin
            )

            macro_f1Bin = f1_score(
                y_true_bin,
                y_pred_bin,
                labels=["True", "False"],
                average="macro"
            )

            # 4-class
            y_true_4 = [map4(i) for i in y_true_so_far]
            y_pred_4 = [map4(i) for i in y_pred_so_far]

            acc4 = accuracy_score(
                y_true_4,
                y_pred_4
            )

            macro_f14 = f1_score(
                y_true_4,
                y_pred_4,
                labels=["True", "Half true", "False", "Pants on fire"],
                average="macro"
            )

            acc3 = accuracy_score(
                [map3(i) for i in y_true_so_far],
                [map3(i) for i in y_pred_so_far]
            )

            macro_f13 = f1_score(
                [map3(i) for i in y_true_so_far],
                [map3(i) for i in y_pred_so_far],
                labels=["True", "Mixed", "Pants on fire"],
                average="macro"
            )

            print(f"After {count} examples:")
            print(f" Accuracy (Binary): {accBin:.4f}")
            print(f" Macro F1 (Binary): {macro_f1Bin:.4f}")
            print(f" Accuracy (3 Class): {acc3:.4f}")
            print(f" Macro F1 (3 Class): {macro_f13:.4f}")
            print(f" Accuracy (4 Class): {acc4:.4f}")
            print(f" Macro F1 (4 Class): {macro_f14:.4f}")
            print(f" Accuracy (6 Class): {acc:.4f}")
            print(f" Macro F1 (6 Class): {macro_f1:.4f}")
            print("-" * 40)

    # =========================
    # FINAL METRICS
    # =========================

    final_metrics = {
        "accuracy_binary": accuracy_score(
            [map2(i) for i in y_true_so_far],
            [map2(i) for i in y_pred_so_far]
        ),

        "macro_f1_binary": f1_score(
            [map2(i) for i in y_true_so_far],
            [map2(i) for i in y_pred_so_far],
            labels=["True", "False"],
            average="macro"
        ),

        "accuracy_3_class": accuracy_score(
            [map3(i) for i in y_true_so_far],
            [map3(i) for i in y_pred_so_far]
        ),

        "macro_f1_3_class": f1_score(
            [map3(i) for i in y_true_so_far],
            [map3(i) for i in y_pred_so_far],
            labels=["True", "Mixed", "Pants on fire"],
            average="macro"
        ),

        "accuracy_4_class": accuracy_score(
            [map4(i) for i in y_true_so_far],
            [map4(i) for i in y_pred_so_far]
        ),

        "macro_f1_4_class": f1_score(
            [map4(i) for i in y_true_so_far],
            [map4(i) for i in y_pred_so_far],
            labels=["True", "Half true", "False", "Pants on fire"],
            average="macro"
        ),

        "accuracy_6_class": accuracy_score(
            y_true_so_far,
            y_pred_so_far
        ),

        "macro_f1_6_class": f1_score(
            y_true_so_far,
            y_pred_so_far,
            labels=LABELS,
            average="macro"
        ),
    }

    # =========================
    # SAVE TO JSON
    # =========================

    output = {
        "metrics": final_metrics,
        "num_examples": len(results),
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\n Results saved to {output_file}")

def experiment(k, outDir):

    versionNum = str(k)

    extractedName = "Extracted_Info_" + versionNum
    summaryName = "summary_" + versionNum

    Output = outDir / str("METHOD1_Results_" + versionNum + ".json")

    jsonOutput = outDir / str("METHOD1_Results_" + versionNum + "_SUMMARY.json")
    jsonOutput1 = outDir / str("METHOD1_Results_" + versionNum + "_EXTRACTED.json")
    jsonOutput2 = outDir / str("METHOD1_Results_" + versionNum + "_BOTH.json")

    # =========================================================
    # SAVE CSV
    # =========================================================
    df.to_json(Output, orient="records", indent=2)

    # =========================================================
    # EVALUATION: SUMMARY
    # =========================================================
    start = time.perf_counter()

    run(
        df,
        summaryName,
        extractedName,
        mode=0,
        output_file=jsonOutput
    )

    eval_summary_time = time.perf_counter() - start

    print(f"\nSummary Eval Time: {eval_summary_time:.2f} seconds")

    # # # =========================================================
    # # # EVALUATION: EXTRACTED
    # # # =========================================================
    start = time.perf_counter()

    run(
        df,
        summaryName,
        extractedName,
        mode=1,
        output_file=jsonOutput1
    )

    eval_extracted_time = time.perf_counter() - start

    print(f"\nExtracted Eval Time: {eval_extracted_time:.2f} seconds")

    # # =========================================================
    # # EVALUATION: BOTH
    # # =========================================================
    start = time.perf_counter()

    run(
        df,
        summaryName,
        extractedName,
        mode=2,
        output_file=jsonOutput2
    )

    eval_both_time = time.perf_counter() - start

    print(f"\nBoth Eval Time: {eval_both_time:.2f} seconds")

    # # # =========================================================
    # # # TOTAL
    # # # =========================================================
    # total = (
    #     extraction_time
    #     + parsing_time
    #     + summary_time
    #     + eval_summary_time
    #     + eval_extracted_time
    #     + eval_both_time
    # )

    # print("\n" + "=" * 50)
    # print(f"TOTAL PIPELINE TIME: {total:.2f} seconds")
    # print("=" * 50)
# ====================================================================================
# EXPERIMENT
# ====================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarization pipeline to fact-check articles and claims."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("RESULTS"),
        help="Root directory containing extracted datasets.",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="ALL",
        help="k value : 2, 4, 6, or ALL"
    )
    parser.add_argument(
        "--datasize",
        type=int,
        default=300,
        help="Size of the data"
    )
    return parser.parse_args()

def main():
    global df

    args = parse_args()

    output_root = args.output_dir
    k = args.k
    dataSize = args.datasize

    df = load_data(dataSize)

    experimentNumber = []
    if k == "ALL":
        experimentNumber.extend([2, 4, 6])
    else:
        experimentNumber.append(k)

    for i in experimentNumber:
        print(i)
        experiment(i, output_root)

if __name__ == "__main__":
    main()