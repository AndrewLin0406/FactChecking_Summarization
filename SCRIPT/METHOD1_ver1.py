"""
METHOD 1: retrieval -> structured evidence -> evaluation
    Retrieve relevant evidence related to the given claim and categorize the information into three groups:
        Supporting — evidence that supports the claim
        Weakening — evidence that contradicts or challenges the claim
        Missing Context — important contextual information omitted from the claim
    The retrieved information is structured in JSON format.

    SUMMARIZATION: Given the claim and retrieved information, summarize the information into
        a quick summary, targetting the signal / direction of verdict.

    EVALUATION: Give the collected information, into a 6 class classification task. Into 
        the following class:
            True = accurate and not missing important context
            Mostly true = mostly accurate but needs minor clarification
            Half true = partially accurate but leaves out important details
            Mostly false = contains an element of truth but ignores critical facts
            False = inaccurate
            Pants on fire = false and absurdly misleading
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json, re, ast, os
from transformers import logging as hf_logging
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

hf_logging.set_verbosity_error()

# ====================================================================================
# HELPERS
# ====================================================================================

dataSize = 50

STOP_PATTERNS = [
    r"\bour ruling\b",
    r"\bour rating\b",
    r"\bwe rate\b",
    r"\bpolitifact rating\b",
    r"\bshare the facts\b",
    r"\btruth-o-meter\b",
]

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

def getText(text):
    text = text.lower()

    cut_idx = len(text)

    for pattern in STOP_PATTERNS:
        m = re.search(pattern, text)
        if m:
            cut_idx = min(cut_idx, m.start())

    text = text[:cut_idx]

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if len(p.strip().split()) >= 5
    ]

    return paragraphs

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
    
def build_structured_evidence(extracted):
    return (
        f"Main subject: {extracted.get('main_subject', '')}\n\n"
        f"Supporting facts:\n- " + "\n- ".join(extracted.get("supporting_facts", [])) + "\n\n"
        f"Weakening facts:\n- " + "\n- ".join(extracted.get("weakening_facts", [])) + "\n\n"
        f"Missing context:\n- " + "\n- ".join(extracted.get("missing_context", []))
    )

def robust_parse(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return empty_json()
    # Try strict JSON
    try:
        return json.loads(x)
    except:
        pass
    # Fix bad escaping
    try:
        x_fixed = x.replace('\\"', '"').replace("\\'", "'")
        return json.loads(x_fixed)
    except:
        pass
    # Extract JSON substring
    try:
        match = re.search(r"\{[\s\S]*?\}", x)
        if match:
            return json.loads(match.group())
    except:
        pass
    # Python dict fallback
    try:
        return ast.literal_eval(x)
    except:
        pass
    # Final fallback
    return empty_json()

def empty_json():
    return {
        "main_subject": "",
        "supporting_facts": [],
        "weakening_facts": [],
        "missing_context": [],
    }

# ====================================================================================
# LOAD DATA
# ====================================================================================

cleaned = pd.read_parquet("data/cleaned/2024-10-10_factchecks_cleaned_nans_flipometer_removed.parquet")
rawClaim = pd.read_parquet("data/raw/2024-10-10_factchecks.parquet")
rawText = pd.read_parquet("data/raw/2024-10-19_fc_analysis_text.parquet")
summary = pd.read_parquet("data/cleaned/factcheck_summaries.parquet")

rawText = rawText[rawText["factcheck_analysis_text"] != ""].dropna().reset_index()
df = pd.merge(rawText.drop(columns=["index"]), rawClaim[["statement", "factcheck_analysis_link"]], on="factcheck_analysis_link", how="inner")
df = pd.merge(df, cleaned[["verdict", "factcheck_analysis_link"]], on="factcheck_analysis_link", how="inner")
df = df.sample(n=dataSize, random_state=42).reset_index(drop=True)
print(df["verdict"].value_counts())

df["noRuling"] = df["factcheck_analysis_text"].apply(getText)

# ====================================================================================
# MODELS
# ====================================================================================

MODEL_NAME = "Qwen/Qwen3.5-4B"
device = "mps"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float32,
    # try float32 if this is unstable
    attn_implementation="sdpa",
)

model = model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

eval_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token=HF_TOKEN
)

eval_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype="auto",
    token=HF_TOKEN
)

eval_model = eval_model.to(device)

if eval_tokenizer.pad_token is None:
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

eval_model.config.pad_token_id = eval_tokenizer.pad_token_id

# ====================================================================================
# PROMPTS
# ====================================================================================

def build_extraction_messages(statement: str, article: str, k=4):
    return [
        {
            "role": "system",
            "content": (
                "You extract short factual evidence from an article for fact-checking.\n"
                "Use only the article.\n"
                "Do not invent facts.\n"
                "Do not give a verdict.\n"
                "Return valid JSON only.\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Statement:\n{statement}\n\n"
                f"Article:\n{article}\n\n"
                "Return this JSON object:\n"
                "{\n"
                '  "main_subject": "",\n'
                '  "supporting_facts": [],\n'
                '  "weakening_facts": [],\n'
                '  "missing_context": []\n'
                "}\n\n"
                "Meaning of each key:\n"
                "- main_subject: the main person, group, or thing the statement is about\n"
                f"- supporting_facts: the {k - 1} or {k} most important article facts that support the statement\n"
                f"- weakening_facts: the {k - 1} or {k} most important article facts that weaken, contradict, or make the statement misleading\n"
                f"- missing_context: the {k - 1} or {k} most important article facts the statement leaves out\n\n"
                "Rules:\n"
                "- main_subject: short phrase\n"
                f"- supporting_facts: up to {k} short facts from the article\n"
                f"- weakening_facts: up to {k} short facts from the article\n"
                f"- missing_context: up to {k} short facts from the article\n"
                "- Do not treat the speaker repeating the claim as supporting evidence\n"
                "- If nothing is found for a list, return []\n"
                "- Output JSON only\n"
            )
        }
    ]


def build_summary_messages(statement: str, extracted_json: dict, k=3):
    return [
        {
            "role": "system",
            "content": (
                "You write short fact-check summaries.\n"
                "Use only the provided evidence.\n"
                "Do not invent facts.\n"
                "Preserve the factual direction and degree of support/contradiction.\n"
                f"Write exactly {k} short sentences.\n"
                "Focus on the most important facts for judging the statement.\n"
                "Prioritize contradictions or limitations when they exist.\n"
                "Include important missing context if it changes how the claim is interpreted.\n"
                "Keep the writing natural, concise, and specific.\n"
                "Do not repeat the statement verbatim unless necessary.\n"
                "Return plain text only.\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Statement:\n{statement}\n\n"
                f"Extracted Evidence:\n{json.dumps(extracted_json, ensure_ascii=False)}\n\n"
                f"Write exactly {k} short sentences."
            )
        }
    ]


def build_eval_messages(statement: str, evidence: str):
    return [
        {
            "role": "system",
            "content": (
                "You are a fact-checker using these labels:\n"
                "True = accurate and not missing important context\n"
                "Mostly true = mostly accurate but needs minor clarification\n"
                "Half true = partially accurate but leaves out important details\n"
                "Mostly false = contains an element of truth but ignores critical facts\n"
                "False = inaccurate\n"
                "Pants on fire = false and absurdly misleading\n\n"
                "Output exactly one of these labels only:\n"
                "True\n"
                "Mostly true\n"
                "Half true\n"
                "Mostly false\n"
                "False\n"
                "Pants on fire\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Statement:\n{statement}\n\n"
                f"Evidence:\n{evidence}\n"
            )
        }
    ]

# ====================================================================================
# STEPS
# ====================================================================================

# EXTRACTION
def generate_extraction(statement, article, max_new_tokens=600, k=4):
    if isinstance(article, list):
        article = "\n\n".join(article)

    messages = build_extraction_messages(statement, article, k=k)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0, input_len:]

    text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return text.strip()

# SUMMARIZATION
def generate_summary(statement, evidence, k=3):
    inputs = tokenizer.apply_chat_template(
        build_summary_messages(statement, evidence, k=k),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    summary = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return summary

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
        if count % 500 == 0 or count == len(eval_df):

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

            print(f"After {count} examples:")
            print(f" Accuracy (6 Class): {acc:.4f}")
            print(f" Macro F1 (6 Class): {macro_f1:.4f}")
            print(f" Accuracy (Binary): {accBin:.4f}")
            print(f" Macro F1 (Binary): {macro_f1Bin:.4f}")
            print(f" Accuracy (4 Class): {acc4:.4f}")
            print(f" Macro F1 (4 Class): {macro_f14:.4f}")
            print("-" * 40)

    # =========================
    # FINAL METRICS
    # =========================

    final_metrics = {
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

def experiment(k):
    versionNum = str(k)
    extractedName = "Extracted_Info_" + versionNum
    summaryName = "summary_" + versionNum
    jsonOutput = "RESULTS/METHOD1_Results_" + versionNum + ".json"
    csvOutput = "RESULTS/METHOD1_Results_" + versionNum + ".csv"

    df[extractedName] = df.apply(lambda x: generate_extraction(x["statement"], x["noRuling"], k=k), axis=1)
    df[extractedName] = df[extractedName].apply(robust_parse)
    df[summaryName] = df.apply(lambda x: generate_summary(x["statement"], x[extractedName]), axis=1)
    df.to_csv(csvOutput, index=False)
    run(df, summaryName, extractedName, mode=0, output_file=jsonOutput)
    #run(df, summaryName, extractedName, mode=1, output_file=jsonOutput)
    #run(df, summaryName, extractedName, mode=2, output_file=jsonOutput)
# ====================================================================================
# EXPERIMENT
# ====================================================================================

experimentNumber = [4]

for i in experimentNumber:
    experiment(i)