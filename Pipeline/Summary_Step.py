"""
Summarization Step:
    SUMMARIZATION: Given the claim and retrieved information, summarize the information into
        a quick summary, targetting the signal / direction of verdict.
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

# ====================================================================================
# LOAD DATA
# ====================================================================================

def load_data():
    return pd.read_json("Extraction/METHOD1_Results_2.json")

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

# ====================================================================================
# PROMPTS
# ====================================================================================

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
                "- Do not let partial support outweigh direct weakening evidence.\n"
                "- Do not describe a fact as support if it only repeats the speaker's claim.\n"
                "- If the central claim is contradicted, say 'the main claim is contradicted.'\n"
                "- If the claim has only a minor true detail but the main impression is wrong, say 'this is mostly contradicted.'\n"
                "- If the claim is fabricated, satirical, impossible, or absurd, explicitly say that.\n"
                "- If a fact weakens the main relationship in the claim, emphasize that weakness.\n"
                "- If a fact appears placed in the wrong category, summarize its actual effect on the claim.\n"
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
                "2. Summarize the strongest weakening evidence, if any.\n"
                "3. Summarize the most important missing context and whether it is minor, significant, or absent."
            )
        }
    ]

# ====================================================================================
# STEPS
# ====================================================================================

# SUMMARIZATION
def generate_summary(statement, evidence, k=3):
    max_new_tokens = 160
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    summary = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    finish_token = new_tokens[-1].item() if len(new_tokens) > 0 else None
    hit_limit = len(new_tokens) >= max_new_tokens

    if hit_limit:
        print("\nWARNING: hit max_new_tokens")
        print("Statement:", statement[:150])
        print("Output tail:", summary[-300:])

    return summary

def experiment(k, outDir):

    versionNum = str(k)

    extractedName = "Extracted_Info_" + versionNum
    summaryName = "summary_" + versionNum

    Output = outDir / str("METHOD1_Results_" + versionNum + ".json")

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
    return parser.parse_args()

def main():
    global df

    args = parse_args()

    output_root = args.output_dir

    df = load_data()

    experimentNumber = [2]

    for i in experimentNumber:
        experiment(i, output_root)

if __name__ == "__main__":
    main()