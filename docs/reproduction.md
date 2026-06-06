# Reproduction Notes

This repository contains source code, configuration templates, and minimal examples. It does not contain full audio data, base models, LoRA adapters, full prediction dumps, or paper-result JSON files.

## 1. Prepare Paths

Use environment variables to avoid hard-coded local paths:

```bash
export DATA_ROOT=/path/to/kyzh_data
export MODEL_ROOT=/path/to/models
export OUT_ROOT=/path/to/earl_outputs
```

## 2. Prepare Data

Prepare these files:

```text
$DATA_ROOT/train.jsonl
$DATA_ROOT/train.ner.jsonl
$DATA_ROOT/dev.jsonl
$DATA_ROOT/dev.ner.jsonl
$DATA_ROOT/test.jsonl
$DATA_ROOT/test.ner.jsonl
```

See `docs/data.md` for schemas.

## 3. Run SFT

```bash
python scripts/train_gemma4_sft_qlora.py \
  --base-model-path $MODEL_ROOT/gemma-4-e4b-it \
  --train-data-path $DATA_ROOT/train.jsonl \
  --val-data-path $DATA_ROOT/dev.jsonl \
  --test-data-path $DATA_ROOT/test.jsonl \
  --output-root $OUT_ROOT/sft \
  --experiment-name gemma4_e4b_sft \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --mixed-precision bf16 \
  --gradient-checkpointing
```

## 4. Run GRPO

The paper setting uses a reward mixture of BLEU, chrF, and entity recall. The corresponding config family is under `configs/fca_grpo/`.
The repository provides two KL-regularized group-relative objectives, `group_relative_risk_kl` and `clipped_grpo`. Choose one explicitly with `--grpo-objective`; the reward, sampling, LoRA, and evaluation hyperparameters below are shared so the objective choice is the only intended difference.

```bash
export GRPO_OBJECTIVE=group_relative_risk_kl  # or clipped_grpo

python scripts/train_gemma4_grpo_lora.py \
  --grpo-objective $GRPO_OBJECTIVE \
  --base-model-path $MODEL_ROOT/gemma-4-e4b-it \
  --init-adapter-path $OUT_ROOT/sft/gemma4_e4b_sft/adapter_best \
  --train-data-path $DATA_ROOT/train.jsonl \
  --train-entity-path $DATA_ROOT/train.ner.jsonl \
  --val-data-path $DATA_ROOT/dev.jsonl \
  --val-entity-path $DATA_ROOT/dev.ner.jsonl \
  --output-root $OUT_ROOT/grpo \
  --experiment-name gemma4_e4b_sft_grpo \
  --bleu-weight 0.3 \
  --chrf-weight 0.5 \
  --entity-weight 0.2 \
  --entity-reward-mode entity_substring \
  --kl-coef 0.02 \
  --group-policy-scale 0.1 \
  --num-candidates 4 \
  --temperature 1.0 \
  --top-k 50 \
  --top-p 0.9
```

## 5. Evaluate

```bash
python scripts/run_eval_suite.py --config configs/eval_suite/offline_predictions_example.yaml
```

For strict paper-style entity recall, use `entity_key_recall` and provide `dataset.reference_entity_path`.

## Result Artifacts

Generated files such as `metrics.summary.json`, `metrics.by_sample.jsonl`, and `predictions.jsonl` should stay in your experiment output directory or be published as separate artifacts. They are intentionally omitted from this source-only release.
