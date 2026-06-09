"""
Extraction Step"
    Retrieve relevant evidence related to the given claim and categorize the information into three groups:
        Supporting — evidence that supports the claim labeled with level of importance
            - also has a label of supporting type
                "- Use 'full' only if the fact independently supports the entire statement's main relationship.\n"
                "- Use 'partial' only for support of a number, quote, date, person, premise, or subclaim.\n"
        Weakening — evidence that contradicts or challenges the claim labeled with level of importance
        Missing Context — important contextual information omitted from the claim labeled with level of importance
    The retrieved information is structured in JSON format.
"""

import time, torch, json, re, ast, os, argparse
from pathlib import Path
import pandas as pd
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

def empty_json():
    return {
        "supporting_facts": [],
        "weakening_facts": [],
        "missing_context": [],
    }


def normalize_schema(x):
    if not isinstance(x, dict):
        return empty_json()

    # tolerate old/new key names
    if "weakening_facts" not in x and "contradicting_facts" in x:
        x["weakening_facts"] = x["contradicting_facts"]

    normalized = {
        "supporting_facts": x.get("supporting_facts", []),
        "weakening_facts": x.get("weakening_facts", []),
        "missing_context": x.get("missing_context", []),
    }

    # remove accidental support_type from non-support categories
    for key in ["weakening_facts", "missing_context"]:
        for item in normalized.get(key, []):
            if isinstance(item, dict):
                item.pop("support_type", None)

    return normalized


def extract_json_block(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end+1]


def robust_parse(x):
    if isinstance(x, dict):
        return normalize_schema(x)

    if not isinstance(x, str):
        return empty_json()

    candidates = [x]
    block = extract_json_block(x)
    if block is not None:
        candidates.append(block)

    for candidate in candidates:
        try:
            return normalize_schema(json.loads(candidate))
        except Exception:
            pass

        try:
            return normalize_schema(ast.literal_eval(candidate))
        except Exception:
            pass

    print("\nPARSE FAILED")
    print(x[:1000])
    return empty_json()

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

# ====================================================================================
# PROMPTS
# ====================================================================================

def build_extraction_messages(statement: str, article: str, k=2):
    return [
        {
            "role": "system",
            "content": (
                "Extract concise factual evidence from the article for fact-checking.\n"
                "Use only the article. Do not invent facts. Do not give a verdict label.\n"
                "Return valid JSON only."
            )
        },
        {
            "role": "user",
            "content": (
                f"Statement:\n{statement}\n\n"
                f"Article:\n{article}\n\n"
                "Return JSON with exactly these keys:\n"
                "{\n"
                '  "supporting_facts": [],\n'
                '  "weakening_facts": [],\n'
                '  "missing_context": []\n'
                "}\n\n"
                "Each supporting_facts item must be:\n"
                '{ "fact": "", "importance": 1, "support_type": "full" }\n'
                "Each weakening_facts and missing_context item must be:\n"
                '{ "fact": "", "importance": 1 }\n\n'
                f"Return at most {k} facts per category. Prefer fewer decisive facts.\n"
                "Each fact must be one short sentence, maximum 25 words.\n\n"
                "Definitions:\n"
                "- supporting_facts: independent article facts that show the statement is accurate, not merely that the statement was said.\n"
                "- weakening_facts: article facts that contradict, disprove, or seriously weaken the statement.\n"
                "- missing_context: omitted article facts that change interpretation.\n\n"
                "Importance: 1=minor, 3=important, 5=decisive.\n\n"
                "Rules:\n"
                "- support_type is only for supporting_facts and must be 'full' or 'partial'.\n"
                "- Use 'full' only if the fact independently supports the entire statement's main relationship.\n"
                "- Use 'partial' only for support of a number, quote, date, person, premise, or subclaim.\n"
                "- Never include quoted repetitions of the claim in supporting_facts.\n"
                "- If the article says the speaker made the exact statement being checked, that is not supporting evidence. Omit it unless the claim is only about whether the person said those words."
                "- A fact is support only if it verifies the claim independently of the speaker's own statement."
                "- If a fact says the article, evidence, pledge, data, or record does NOT mention or does NOT show the relationship claimed in the statement, put it in weakening_facts, not supporting_facts."
                "- Do not include 'X said/claimed/stated...' in supporting_facts unless the checked statement is only about whether X said those words."
                "- If the speaker merely states, repeats, defends, alleges, or explains their own claim, omit it.\n"
                "- Only include a quote as support if the statement is specifically about whether that quote was said.\n"
                "- Evidence about long-term troop presence does not support a claim about long-term war unless the article says it is combat or war.\n"
                "- Do not fill categories just to reach the limit.\n"
                "- Omit background about who made the claim, social media spread, or fact-check flags unless essential.\n"
                "- Sort each list from highest importance to lowest.\n"
                "- Output JSON only."
            )
        }
    ]

# ====================================================================================
# STEPS
# ====================================================================================

# EXTRACTION
def generate_extraction(statement, article, k=2):
    max_new_tokens = min(1400, max(400, 250 + 180 * int(k)))

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

    finish_token = new_tokens[-1].item() if len(new_tokens) > 0 else None
    hit_limit = len(new_tokens) >= max_new_tokens

    if hit_limit:
        print("\nWARNING: hit max_new_tokens")
        print("Statement:", statement[:150])
        print("Output tail:", text[-300:])

    return text.strip()


def experiment(k, outDir):

    versionNum = str(k)

    extractedName = "Extracted_Info_" + versionNum
    summaryName = "summary_" + versionNum

    Output = outDir / str("METHOD1_Results_" + versionNum + ".json")

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
    # SAVE CSV
    # =========================================================
    df.to_json(Output, orient="records", indent=2)
    print("Saved Extraciton File To", Output)

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
        experimentNumber.extend([2, 3, 4, 6])
    else:
        experimentNumber.append(int(k))

    for i in experimentNumber:
        experiment(i, output_root)

if __name__ == "__main__":
    main()