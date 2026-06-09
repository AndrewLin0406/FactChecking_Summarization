# Fact-Checking Pipeline

This project contains experiments for a PolitiFact-style fact-checking pipeline. The goal is to classify factual claims into Truth-O-Meter labels using cleaned article evidence, structured evidence extraction, optional summarization, and model-based evaluation.

## Project Structure

```text
.
├── Pipeline/
│   └── All_STEPS.py
│
├── SCRIPTS/
│   └── older methods, experimental versions, and ablation scripts
│
├── data/
│   ├── raw/
│   └── cleaned/
│
├── RESULTS/
│   └── saved outputs, metrics, predictions, and debugging runs
│
└── README.md
```

## Main Directories

### `Pipeline/`

The `Pipeline/` directory contains the current version of the pipeline.

The main current script is:

```text
Pipeline/All_STEPS.py
```

This script runs the full pipeline:

```text
clean article text
        ↓
structured evidence extraction
        ↓
evidence summarization
        ↓
evaluation using summary, extracted evidence, and both
        ↓
metrics + saved JSON outputs
```

Use this directory when running the latest working version of the project.

### `SCRIPTS/`

The `SCRIPTS/` directory contains older methods, previous versions, and experimental variants.

This includes earlier prompt versions, ablation scripts, different method implementations, and debugging experiments. These files are useful for tracking the development history of the project, but they should not be treated as the main current pipeline.

Examples of what may belong in `SCRIPTS/`:

* older extraction → summary → evaluation pipelines
* direct classification baselines
* different prompt versions
* experiments with different `k` values
* ablations using summary only, extracted evidence only, or both
* metric comparison scripts
* debugging scripts

## Current Pipeline

The current implementation is located in:

```bash
Pipeline/All_STEPS.py
```

The pipeline performs three main stages:

### 1. Extraction

The model extracts structured evidence from the article into three categories:

```text
supporting_facts
weakening_facts
missing_context
```

Each extracted fact may include an `importance` score from 1 to 5.

Supporting facts may also include:

```text
support_type: full | partial
```

### 2. Summarization

The extracted evidence is summarized into a short evidence summary for classification.

### 3. Evaluation

The evaluator predicts one of the six Truth-O-Meter labels:

```text
True
Mostly true
Half true
Mostly false
False
Pants on fire
```

The script evaluates the claim using three evidence modes:

```text
summary only
structured extracted evidence only
summary + structured extracted evidence
```

## Use Case

Use `Pipeline/All_STEPS.py` when you want to run the current full pipeline on a sampled dataset and save the results.

Basic command:

```bash
python3 Pipeline/All_STEPS.py --output-dir RESULTS --k 2 --datasize 60
```

This runs the current pipeline with:

```text
--output-dir RESULTS
```

Directory where output JSON files will be saved.

```text
--k 2
```

Maximum number of extracted facts per evidence category.

```text
--datasize 60
```

Total number of examples sampled from the dataset.

## Arguments

### `--output-dir`

Output directory for saved result files.

Default:

```bash
RESULTS
```

Example:

```bash
python3 Pipeline/All_STEPS.py --output-dir testResults --k 2 --datasize 60
```

### `--k`

Controls the maximum number of extracted facts per category.

Supported values:

```text
2
4
6
ALL
```

If `--k ALL` is used, the script runs experiments for:

```text
k = 2
k = 4
k = 6
```

Example:

```bash
python3 Pipeline/All_STEPS.py --output-dir RESULTS --k ALL --datasize 300
```

### `--datasize`

Controls how many examples are sampled from the dataset.

The script samples examples evenly across the six Truth-O-Meter labels.

Example:

```bash
python3 Pipeline/All_STEPS.py --output-dir RESULTS --k 2 --datasize 300
```

## Example Runs

Run a small debugging experiment:

```bash
python3 Pipeline/All_STEPS.py --output-dir testResults --k 2 --datasize 30
```

Run the current preferred setup:

```bash
python3 Pipeline/All_STEPS.py --output-dir RESULTS --k 2 --datasize 300
```

Run multiple `k` values:

```bash
python3 Pipeline/All_STEPS.py --output-dir RESULTS --k ALL --datasize 300
```

## Outputs

For each value of `k`, the script saves several JSON files.

Example for `k=2`:

```text
METHOD1_Results_2.json
METHOD1_Results_2_SUMMARY.json
METHOD1_Results_2_EXTRACTED.json
METHOD1_Results_2_BOTH.json
```

These correspond to:

```text
METHOD1_Results_2.json
```

The dataframe-style output containing extracted evidence and summaries.

```text
METHOD1_Results_2_SUMMARY.json
```

Evaluation results using the summary only.

```text
METHOD1_Results_2_EXTRACTED.json
```

Evaluation results using the formatted extracted evidence only.

```text
METHOD1_Results_2_BOTH.json
```

Evaluation results using both the summary and extracted evidence.

## Metrics

The script reports and saves the following metrics:

```text
accuracy_binary
macro_f1_binary
accuracy_3_class
macro_f1_3_class
accuracy_4_class
macro_f1_4_class
accuracy_6_class
macro_f1_6_class
```

The six-class metrics evaluate the original Truth-O-Meter labels.

The binary, 3-class, and 4-class metrics evaluate collapsed label mappings.

## Recommended Workflow

1. Make current pipeline changes inside `Pipeline/`.
2. Run a small debug experiment first.

```bash
python3 Pipeline/All_STEPS.py --output-dir testResults --k 2 --datasize 30
```

3. Inspect extracted evidence, summaries, and predictions.
4. If the run looks stable, increase the dataset size.

```bash
python3 Pipeline/All_STEPS.py --output-dir RESULTS --k 2 --datasize 300
```

5. Save older experimental versions in `SCRIPTS/`.
6. Compare metrics before replacing the current pipeline.

## Notes

The current best practice is to keep `Pipeline/` focused on the latest working version and use `SCRIPTS/` as an archive for previous methods and experiments.

When saving results, include enough information to identify the run:

* method name
* script version
* dataset size
* `k` value
* model names
* date or run identifier
* output directory
