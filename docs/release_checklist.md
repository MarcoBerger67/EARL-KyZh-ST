# Release Checklist

Before publishing:

1. Confirm the repository contains no model weights, full data, full prediction dumps, or local caches.
2. Confirm `rg -n "api_key|apikey|secret|token|Bearer|sk-" .` returns no real credential.
3. Replace placeholder repository URLs in `CITATION.cff`.
4. Confirm the selected license is acceptable for your institution and collaborators.
5. Install the development dependencies (`pip install -r requirements-dev.txt`), then run:

```bash
pytest
python scripts/run_eval_suite.py --config configs/eval_suite/offline_mini.yaml
```

