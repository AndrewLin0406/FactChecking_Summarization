"""
Ver2 : Half True in False
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json, re, ast, os
from transformers import logging as hf_logging
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from sklearn.metrics import confusion_matrix
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

hf_logging.set_verbosity_error()

# ====================================================================================
# HELPERS
# ====================================================================================

dataSize = 300

STOP_PATTERNS = [
    r"\bour ruling\b",
    r"\bour rating\b",
]

LEAKAGE_REPLACEMENTS = [
    (r"\bmostly true\b", ""),
    (r"\bmostly false\b", ""),
    (r"\bhalf true\b", ""),
    (r"\bpants on fire\b", ""),
]

FINAL_LEAKAGE_PATTERNS = [
    r"\bwe rate\b.*",
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

    return "\n\n".join(paragraphs)

def getText_2(text):

    # =========================================================
    # STEP 1: Lowercase
    # =========================================================
    text = text.lower()

    # =========================================================
    # STEP 2: Split into paragraphs
    # =========================================================
    paragraphs = text.split("\n\n")

    cleaned_paragraphs = []

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        para_lower = para.lower()

        # =====================================================
        # STEP 3: Stop at verdict section
        # =====================================================
        stop = False

        for pattern in STOP_PATTERNS:
            if re.search(pattern, para_lower):
                stop = True
                break

        if stop:
            break

        # =====================================================
        # STEP 4: Remove explicit leakage phrases
        # =====================================================
        for pattern, replacement in LEAKAGE_REPLACEMENTS:
            para = re.sub(
                pattern,
                replacement,
                para,
                flags=re.IGNORECASE
            )

        # =====================================================
        # STEP 5: Sentence tokenization
        # =====================================================
        sentences = sent_tokenize(para)

        cleaned_sentences = []

        for sentence in sentences:

            # ================================================
            # Remove explicit verdict supervision
            # ================================================
            for pattern in FINAL_LEAKAGE_PATTERNS:
                sentence = re.sub(
                    pattern,
                    "",
                    sentence,
                    flags=re.IGNORECASE
                )

            # ================================================
            # Cleanup whitespace only
            # ================================================
            sentence = re.sub(r"\s+", " ", sentence)
            sentence = re.sub(r"\s+\.", ".", sentence)
            sentence = sentence.strip()

            if sentence:
                cleaned_sentences.append(sentence)

        # =====================================================
        # STEP 6: Rebuild paragraph
        # =====================================================
        para = " ".join(cleaned_sentences).strip()

        if para:
            cleaned_paragraphs.append(para)

    # =========================================================
    # STEP 7: Recombine paragraphs
    # =========================================================
    cleaned_text = "\n\n".join(cleaned_paragraphs)

    return cleaned_text

def map2(label):
    if label in ["True", "Half true"]:
        return "True"
    elif label in ["False", "Mostly false", "Pants on fire", "Mostly true"]:
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
    if label in ["True", "Mostly true"]:
        return "True"
    elif label in ["Half true", "Mostly false"]:
        return "Mixed"
    elif label in ["False", "Pants on fire"]:
        return "False"
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
df = pd.merge(df, summary[["summary", "factcheck_analysis_link"]], on="factcheck_analysis_link", how="inner")
df2 = df[df["verdict"].isin(["True", "Pants on fire"])]

df2 = (df[df["verdict"].isin(["True", "Pants on fire"])].groupby("verdict", group_keys=False).sample(n= int(dataSize / 2), random_state=42).reset_index(drop=True))
print(df2["verdict"].value_counts())

df = (df.groupby("verdict", group_keys=False).sample(n= int(dataSize / 6), random_state=42).reset_index(drop=True))
print(df["verdict"].value_counts())

# ====================================================================================
# MODELS
# ====================================================================================

MODEL_NAME = "Qwen/Qwen3.5-4B"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

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
            temperature=0.0,
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
        gold = row["verdict"]

        match mode:
            case 0:
                evidence = summary
        
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

            # 3-class
            y_true_3 = [map3(i) for i in y_true_so_far]
            y_pred_3 = [map3(i) for i in y_pred_so_far]

            acc3 = accuracy_score(
                y_true_3,
                y_pred_3
            )

            macro_f13 = f1_score(
                y_true_3,
                y_pred_3,
                labels=["True", "Mixed", "Pants on fire"],
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
            print(f" Accuracy (3 Class): {acc3:.4f}")
            print(f" Macro F1 (3 Class): {macro_f13:.4f}")
            print("-" * 40)

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
        gold = row["verdict"]

        match mode:
            case 0:
                evidence = summary
        
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

            # 3-class
            y_true_3 = [map3(i) for i in y_true_so_far]
            y_pred_3 = [map3(i) for i in y_pred_so_far]

            acc3 = accuracy_score(
                y_true_3,
                y_pred_3
            )

            macro_f13 = f1_score(
                y_true_3,
                y_pred_3,
                labels=["True", "False", "Mixed"],
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
            print(f" Accuracy (3 Class): {acc3:.4f}")
            print(f" Macro F1 (3 Class): {macro_f13:.4f}")
            print("-" * 40)

            cm = confusion_matrix(
                y_true_so_far,
                y_pred_so_far,
                labels=LABELS
            )

            print("Confusion Matrix (6 Class):")
            print(pd.DataFrame(
                cm,
                index=[f"gold_{label}" for label in LABELS],
                columns=[f"pred_{label}" for label in LABELS]
            ))

# ====================================================================================
# EXPERIMENT
# ====================================================================================


df["noRuling"] = df["factcheck_analysis_text"].apply(getText)
df["noRuling_2"] = df["factcheck_analysis_text"].apply(getText_2)
df2["noRuling"] = df2["factcheck_analysis_text"].apply(getText)
df2["noRuling_2"] = df2["factcheck_analysis_text"].apply(getText_2)

print("\n\n", "=" * 20, "SUMMARY", "=" * 20)
run(df, "summary", "", mode=0, output_file="")
print("\n\n", "=" * 20, "ORIGINAL TEXT", "=" * 20)
run(df, "factcheck_analysis_text", "", mode=0, output_file="")
print("\n\n", "=" * 20, "OLD FILTER", "=" * 20)
run(df, "noRuling", "", mode=0, output_file="")
print("\n\n", "=" * 20, "NEW FILTER", "=" * 20)
run(df, "noRuling_2", "", mode=0, output_file="")

print("\n\n", "=" * 20, "SUMMARY", "=" * 20)
run(df2, "summary", "", mode=0, output_file="")
print("\n\n", "=" * 20, "ORIGINAL TEXT", "=" * 20)
run(df2, "factcheck_analysis_text", "", mode=0, output_file="")
print("\n\n", "=" * 20, "OLD FILTER", "=" * 20)
run(df2, "noRuling", "", mode=0, output_file="")
print("\n\n", "=" * 20, "NEW FILTER", "=" * 20)
run(df2, "noRuling_2", "", mode=0, output_file="")
