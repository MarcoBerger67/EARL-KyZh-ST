# Evaluation Suite

The evaluation suite provides a shared runner for offline predictions and model-based generation.

Main entry point:

```bash
python scripts/run_eval_suite.py --config <yaml-config>
```

Supported modes:

- `score_predictions`: read an existing prediction JSONL and compute metrics.
- `generate_and_score`: load a model adapter, generate predictions, then compute metrics.

Core metrics:

- `bleu`: SacreBLEU with Chinese tokenization.
- `chrf`: SacreBLEU chrF.
- `entity_key_recall`: strict reference-side entity recall.
- `entity_lcs`: soft LCS-based reference-side entity recall.
- `entity_f1` and `entity_soft`: HanLP-based prediction/reference entity matching.

For paper-style entity recall, use `entity_key_recall` and set:

```yaml
dataset:
  reference_entity_path: path/to/entities.jsonl
evaluation:
  metrics: [bleu, chrf, entity_key_recall]
```

