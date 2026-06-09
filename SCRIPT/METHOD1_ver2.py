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

# STOP_PATTERNS = [
#     r"\bour ruling\b",
#     r"\bour rating\b",
#     r"\bwe rate\b",
#     r"\bpolitifact rating\b",
#     r"\bshare the facts\b",
#     r"\btruth-o-meter\b",
# ]

# STOP_PATTERNS = [
#     r"\bour ruling\b",
#     r"\bour rating\b",
# ]

PRIMARY_STOP_PATTERNS = [
    r"\bour ruling\b",
    r"\bour rating\b",
]

FALLBACK_STOP_PATTERNS = [
    r"\bwe rate\b",
    r"\bwe rule\b",
    r"\bwe find\b",
    r"\bso we find\b",
    r"\bour conclusion\b",
]

# WHEN FILTERING STOP PATTERNS, CHECK IF IT DELETES IMPORTANT FACTS OR CONTAINS GROUND TRUTH

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

# def getText(text):
#     text = text.lower()

#     cut_idx = len(text)

#     for pattern in STOP_PATTERNS:
#         m = re.search(pattern, text)
#         if m:
#             cut_idx = min(cut_idx, m.start())

#     text = text[:cut_idx]

#     paragraphs = [
#         p.strip()
#         for p in text.split("\n\n")
#         if len(p.strip().split()) >= 5
#     ]

#     return paragraphs

def getText(text):
    text = text.lower()

    # Step 1: prefer official ruling-section markers
    cut_idx = None
    for pattern in PRIMARY_STOP_PATTERNS:
        m = re.search(pattern, text)
        if m:
            cut_idx = m.start()
            break

    # Step 2: only if no official marker exists, use fallback verdict-like phrases
    if cut_idx is None:
        for pattern in FALLBACK_STOP_PATTERNS:
            m = re.search(pattern, text)
            if m:
                cut_idx = m.start()
                break

    # Step 3: no marker found, keep full text
    if cut_idx is not None:
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

def map3(label):
    if label in ["True", "Mostly True"]:
        return "True"
    elif label in ["Half true", "Mostly false"]:
        return "Mixed"
    elif label in ["False", "Pants on fire"]:
        return "False"
    else:
        return "UNKNOWN"
    
# def build_structured_evidence(extracted):
#     return (
#         f"Supporting facts:\n- " + "\n- ".join(extracted.get("supporting_facts", [])) + "\n\n"
#         f"Weakening facts:\n- " + "\n- ".join(extracted.get("weakening_facts", [])) + "\n\n"
#         f"Missing context:\n- " + "\n- ".join(extracted.get("missing_context", []))
#     )

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

def load_data(dataSize):
    cleaned = pd.read_parquet("data/cleaned/2024-10-10_factchecks_cleaned_nans_flipometer_removed.parquet")
    rawClaim = pd.read_parquet("data/raw/2024-10-10_factchecks.parquet")
    rawText = pd.read_parquet("data/raw/2024-10-19_fc_analysis_text.parquet")
    summary = pd.read_parquet("data/cleaned/factcheck_summaries.parquet")

    rawText = rawText[rawText["factcheck_analysis_text"] != ""].dropna().reset_index()
    df = pd.merge(rawText.drop(columns=["index"]), rawClaim[["statement", "factcheck_analysis_link"]], on="factcheck_analysis_link", how="inner")
    df = pd.merge(df, cleaned[["verdict", "factcheck_analysis_link"]], on="factcheck_analysis_link", how="inner")
    df = (df.groupby("verdict", group_keys=False).sample(n= int(dataSize / 6), random_state=42).reset_index(drop=True))
    print(df["verdict"].value_counts())

    df["noRuling"] = df["factcheck_analysis_text"].apply(getText)

    return df

# ====================================================================================
# MODELS
# ====================================================================================

MODEL_NAME = "Qwen/Qwen3.5-4B"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
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

# def build_extraction_messages(statement: str, article: str, k=4):
#     return [
#         {
#             "role": "system",
#             "content": (
#                 "You extract factual evidence from an article for fact-checking.\n"
#                 "Use only the article.\n"
#                 "Do not invent facts.\n"
#                 "Do not give a verdict, rating, or truth label.\n"
#                 "Return valid JSON only.\n"
#             )
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Statement:\n{statement}\n\n"
#                 f"Article:\n{article}\n\n"
#                 "Return valid JSON with exactly these keys:\n"
#                 "{\n"
#                 '  "supporting_facts": [],\n'
#                 '  "weakening_facts": [],\n'
#                 '  "missing_context": []\n'
#                 "}\n\n"
#                 "Definitions:\n"
#                 f"- supporting_facts: 0 to {k} article facts that support the statement.\n"
#                 f"- weakening_facts: 0 to {k} article facts that contradict, qualify, weaken, or make the statement misleading.\n"
#                 f"- missing_context: 0 to {k} article facts that are important context but omitted by the statement.\n\n"
#                 "Rules:\n"
#                 "- Each item should be one short sentence.\n"
#                 "- Use only information explicitly stated in the article.\n"
#                 "- Do not treat a quoted repetition of the statement as evidence.\n"
#                 "- Do not include the article's final rating or verdict.\n"
#                 "- Do not mention Truth-O-Meter labels.\n"
#                 "- If a category has no relevant facts, return an empty list.\n"
#                 "- Output JSON only, with no markdown.\n"
#             )
#         }
#     ]

def build_extraction_messages(statement: str, article: str, k=4):
    return [
        {
            "role": "system",
            "content": (
                "You extract short factual evidence from an article for fact-checking.\n"
                "Use only the article.\n"
                "Do not invent facts.\n"
                "Do not give a verdict label.\n"
                "Return valid JSON only.\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Statement:\n{statement}\n\n"
                f"Article:\n{article}\n\n"
                "Return valid JSON with exactly these keys:\n"
                "{\n"
                '  "supporting_facts": [],\n'
                '  "contradicting_facts": [],\n'
                '  "missing_context": []\n'
                "}\n\n"
                "For supporting_facts, each item must be:\n"
                '{ "fact": "", "importance": 1, "support_type": "full" }\n\n'
                "For contradicting_facts and missing_context, each item must be:\n"
                '{ "fact": "", "importance": 1 }\n\n'
                "Definitions:\n"
                f"- supporting_facts: 0 to {k} article facts that support the statement.\n"
                f"- contradicting_facts: 0 to {k} article facts that contradict, disprove, or seriously weaken the statement.\n"
                f"- missing_context: 0 to {k} article facts omitted by the statement that change interpretation.\n\n"
                "Importance scale:\n"
                "- 1 = minor or background detail\n"
                "- 2 = useful but not decisive\n"
                "- 3 = important but not central\n"
                "- 4 = very important\n"
                "- 5 = decisive or central to judging the statement\n\n"
                "Rules:\n"
                "- For supporting_facts, support_type must be either 'full' or 'partial'.\n"
                "- Use 'full' only if the fact supports the entire statement, including the same main relationship.\n"
                "- Use 'partial' if the fact supports only a number, quote, date, person, premise, or subclaim.\n"
                "- A fact supports the statement only if it supports the same relationship expressed in the statement.\n"
                "- Example: evidence about long-term troop presence does not support a claim about long-term war unless the article says the presence is combat or war.\n"
                "- Do not treat someone repeating, alleging, suggesting, or speculating the claim as supporting evidence.\n"
                "- Put quoted repetitions of the claim in no category unless they are needed as background.\n"
                "- If a category has no valid facts, return [].\n"
                "- Do not fill categories just to reach a number.\n"
                "- Prefer direct evidence over background or meta-evidence.\n"
                "- Background about who made the claim, social media spread, or fact-check flags should usually be omitted.\n"
                "- Sort each list from highest importance to lowest importance.\n"
                "- Output JSON only.\n"
            )
        }
    ]

# def build_summary_messages(statement: str, extracted_json: dict, k=3):
#     return [
#         {
#             "role": "system",
#             "content": (
#                 "You summarize fact-check evidence for classification.\n"
#                 "Use only the provided extracted evidence.\n"
#                 "Do not use outside knowledge.\n"
#                 "Do not invent facts.\n"
#                 "Do not give a verdict label.\n\n"
#                 f"Write exactly {k} short sentences.\n"
#                 "Each sentence must be precise and factual.\n"
#                 "Return plain text only.\n\n"
#                 "Important:\n"
#                 "- Clearly reflect whether evidence supports, contradicts, or is mixed.\n"
#                 "- Indicate strength (strong, partial, weak) when appropriate.\n"
#                 "- Indicate whether missing context is minor or important.\n"
#                 "- Avoid vague wording.\n"
#             )
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Statement:\n{statement}\n\n"
#                 f"Extracted Evidence:\n{json.dumps(extracted_json, ensure_ascii=False)}\n\n"
#                 "Write exactly 3 short sentences:\n"
#                 "1. Restate the claim clearly.\n"
#                 "2. Describe the strongest supporting or contradicting evidence, and indicate its strength (strong, partial, weak).\n"
#                 "3. Describe the most important missing context and whether it is minor or significant.\n"
#             )
#         }
#     ]

def build_summary_messages(statement: str, extracted_json: dict):
    return [
        {
            "role": "system",
            "content": (
                "You summarize extracted fact-check evidence for classification.\n"
                "Use only the provided extracted evidence.\n"
                "Do not use outside knowledge.\n"
                "Do not invent facts.\n"
                "Do not give a verdict label.\n\n"
                "Write exactly 3 short sentences.\n"
                "Return plain text only.\n\n"
                "Important:\n"
                "- Do not restate the claim; the classifier will already receive the claim separately.\n"
                "- Use the highest-importance facts first.\n"
                "- Distinguish full support from partial support.\n"
                "- Do not let partial support outweigh direct contradiction.\n"
                "- If a fact contradicts the main relationship in the claim, emphasize that contradiction.\n"
                "- If no evidence is provided, say the extraction contains no usable evidence; do not infer whether the claim is true or false."
            )
        },
        {
            "role": "user",
            "content": (
                f"Statement:\n{statement}\n\n"
                f"Extracted Evidence:\n{json.dumps(extracted_json, ensure_ascii=False)}\n\n"
                "Write exactly 3 short sentences:\n"
                "1. Summarize the strongest supporting evidence, if any, and whether it is full or partial support.\n"
                "2. Summarize the strongest contradicting evidence, if any.\n"
                "3. Summarize the most important missing context and whether it is minor, significant, or absent."
            )
        }
    ]

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
        build_summary_messages(statement, evidence),
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
            max_new_tokens=220,
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
    # EXTRACTION
    # =========================================================
    start = time.perf_counter()

    df[extractedName] = df.apply(
        lambda x: generate_extraction(
            x["statement"],
            x["noRuling"],
            k=k
        ),
        axis=1
    )

    extraction_time = time.perf_counter() - start

    print(f"\nExtraction Time: {extraction_time:.2f} seconds")
    print(f"Average per example: {extraction_time / len(df):.2f} seconds")

    # =========================================================
    # PARSING
    # =========================================================
    start = time.perf_counter()

    df[extractedName] = df[extractedName].apply(robust_parse)

    parsing_time = time.perf_counter() - start

    print(f"\nParsing Time: {parsing_time:.2f} seconds")

    # =========================================================
    # SUMMARIZATION
    # =========================================================
    start = time.perf_counter()

    df[summaryName] = df.apply(
        lambda x: generate_summary(
            x["statement"],
            x[extractedName]
        ),
        axis=1
    )

    summary_time = time.perf_counter() - start

    print(f"\nSummary Time: {summary_time:.2f} seconds")
    print(f"Average per example: {summary_time / len(df):.2f} seconds")

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
    # start = time.perf_counter()

    # run(
    #     df,
    #     summaryName,
    #     extractedName,
    #     mode=1,
    #     output_file=jsonOutput1
    # )

    # eval_extracted_time = time.perf_counter() - start

    # print(f"\nExtracted Eval Time: {eval_extracted_time:.2f} seconds")

    # # # =========================================================
    # # # EVALUATION: BOTH
    # # # =========================================================
    # start = time.perf_counter()

    # run(
    #     df,
    #     summaryName,
    #     extractedName,
    #     mode=2,
    #     output_file=jsonOutput2
    # )

    # eval_both_time = time.perf_counter() - start

    # print(f"\nBoth Eval Time: {eval_both_time:.2f} seconds")

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