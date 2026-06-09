"""
METHOD 1: retrieval -> structured evidence -> evaluation
    Retrieve relevant evidence related to the given claim and categorize the information into three groups:
        Supporting — evidence that supports the claim
        Weakening — evidence that contradicts or challenges the claim
        Missing Context — important contextual information omitted from the claim
    PROMPT SEPERATION TO RETRIEVE THE THREE CATEGORY SEPERATETLY
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

def build_supporting_messages(statement: str, article: str, k=4):
    return [
        {
            "role": "system",
            "content": (
                "You extract factual evidence that supports a claim.\n"
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
                "Return ONLY a JSON list of supporting facts.\n\n"
                "Rules:\n"
                f"- Include up to {k} short factual statements from the article\n"
                "- Only include facts that support the statement\n"
                "- Prioritize direct factual confirmation of the claim\n"
                "- Include explicit quotes, numbers, dates, or events when available\n"
                "- If the core event truly happened, include it even if context later weakens it\n"
                "- If nothing is found, return []\n"
                "- Output JSON list only\n\n"
                "Example:\n"
                "[\n"
                '  "The bill increased education funding by $20 billion.",\n'
                '  "State records confirm the spending increase."\n'
                "]"
            )
        }
    ]


def build_weakening_messages(statement: str, article: str, k=4):
    return [
        {
            "role": "system",
            "content": (
                "You extract factual evidence that weakens or contradicts a claim.\n"
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
                "Return ONLY a JSON list of weakening facts.\n\n"
                "Rules:\n"
                f"- Include up to {k} short factual statements from the article\n"
                "- Only include facts that directly contradict, refute, or substantially weaken the claim\n"
                "- Do not include minor clarifications or background details\n"
                "- A statement being incomplete does not make it false\n"
                "- Distinguish contradiction from contextual nuance\n"
                "- If nothing is found, return []\n"
                "- Output JSON list only\n\n"
                "Example:\n"
                "[\n"
                '  "The investigation had already ended before the statement was made.",\n'
                '  "Experts said there was no evidence supporting the broader claim."\n'
                "]"
            )
        }
    ]


def build_context_messages(statement: str, article: str, k=4):
    return [
        {
            "role": "system",
            "content": (
                "You extract important missing context related to a claim.\n"
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
                "Return ONLY a JSON list of missing context facts.\n\n"
                "Rules:\n"
                f"- Include up to {k} short factual statements from the article\n"
                "- Missing context should clarify timing, scope, conditions, limitations, or important nuance\n"
                "- Missing context should clarify interpretation, not directly refute the claim\n"
                "- Do not include facts that directly contradict the statement\n"
                "- Do not include irrelevant background information\n"
                "- If nothing is found, return []\n"
                "- Output JSON list only\n\n"
                "Example:\n"
                "[\n"
                '  "The comments were made during a campaign event, not an official briefing.",\n'
                '  "The White House later clarified the remarks did not reflect official policy."\n'
                "]"
            )
        }
    ]

def build_summary_messages(statement: str, extracted_json: dict, k=3):
    return [
        {
            "role": "system",
            "content": (
                "You summarize fact-check evidence for classification.\n"
                "Use only the provided extracted evidence.\n"
                "Do not use outside knowledge.\n"
                "Do not invent facts.\n"
                "Do not give a verdict label.\n\n"
                f"Write exactly {k} short sentences.\n"
                "Each sentence must be precise and factual.\n"
                "Return plain text only.\n\n"
                "Important:\n"
                "- If the core event or statement factually occurred, acknowledge this clearly before discussing limitations\n"
                "- Missing context alone does not invalidate a claim\n"
                "- Distinguish between inaccurate, misleading, and false claims\n"
                "- Clearly reflect whether evidence supports, contradicts, or is mixed.\n"
                "- Indicate strength (strong, partial, weak) when appropriate.\n"
                "- Indicate whether missing context is minor or important.\n"
                "- Avoid vague wording.\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Statement:\n{statement}\n\n"
                f"Extracted Evidence:\n{json.dumps(extracted_json, ensure_ascii=False)}\n\n"
                "Write exactly 3 short sentences:\n"
                "1. Restate the claim clearly.\n"
                "2. Describe the balance between supporting and contradicting evidence, and indicate its strength (strong, partial, weak).\n"
                "3. Describe the most important missing context and whether it is minor or significant.\n"
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
def run_prompt(messages, max_new_tokens=300):
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


def safe_json_load(text, default):
    try:
        return json.loads(text)
    except:
        return default


def generate_extraction(statement, article, k=4):
    if isinstance(article, list):
        article = "\n\n".join(article)

    # SUPPORTING FACTS
    supporting_messages = build_supporting_messages(
        statement,
        article,
        k=k
    )

    supporting_text = run_prompt(
        supporting_messages,
        max_new_tokens=300
    )

    supporting_facts = safe_json_load(
        supporting_text,
        []
    )


    # WEAKENING FACTS
    weakening_messages = build_weakening_messages(
        statement,
        article,
        k=k
    )

    weakening_text = run_prompt(
        weakening_messages,
        max_new_tokens=300
    )

    weakening_facts = safe_json_load(
        weakening_text,
        []
    )


    # MISSING CONTEXT
    context_messages = build_context_messages(
        statement,
        article,
        k=k
    )

    context_text = run_prompt(
        context_messages,
        max_new_tokens=300
    )

    missing_context = safe_json_load(
        context_text,
        []
    )

    # FINAL MERGED OUTPUT
    return {
        "supporting_facts": supporting_facts,
        "weakening_facts": weakening_facts,
        "missing_context": missing_context
    }

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
    jsonOutput = "RESULTS/METHOD1_ver3_Results_" + versionNum + "SUMMARY.json"
    csvOutput = "RESULTS/METHOD1_ver3_Results_" + versionNum + "SUMMARY.csv"

    jsonOutput1 = "RESULTS/METHOD1_ver3_Results_" + versionNum + "EXTRACTED.json"
    csvOutput1 = "RESULTS/METHOD1_ver3_Results_" + versionNum + "EXTRACTED.csv"

    jsonOutput2 = "RESULTS/METHOD1_ver3_Results_" + versionNum + "BOTH.json"
    csvOutput2 = "RESULTS/METHOD1_ver3_Results_" + versionNum + "BOTH.csv"

    df[extractedName] = df.apply(lambda x: generate_extraction(x["statement"], x["noRuling"], k=k), axis=1)
    df[extractedName] = df[extractedName].apply(robust_parse)
    df[summaryName] = df.apply(lambda x: generate_summary(x["statement"], x[extractedName]), axis=1)
    df.to_csv(csvOutput, index=False)
    run(df, summaryName, extractedName, mode=0, output_file=jsonOutput)
    run(df, summaryName, extractedName, mode=1, output_file=jsonOutput1)
    run(df, summaryName, extractedName, mode=2, output_file=jsonOutput2)
# ====================================================================================
# EXPERIMENT
# ====================================================================================

experimentNumber = [4]

for i in experimentNumber:
    experiment(i)