from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from eval_suite.runner import run_eval_spec
from run_gemma4_base_testt_eval import validate_testt_sidecar


DETAIL_ENTITY_PROMPT = (
    "You are a strict Kyrgyz-to-Chinese speech translation system.\n\n"
    "Task: translate the input Kyrgyz speech segment into fluent Simplified Chinese.\n\n"
    "Output rules:\n"
    "1. Output only the final Simplified Chinese translation.\n"
    "2. Output one single line only.\n"
    "3. Do not output Kyrgyz, English, transliteration, explanations, notes, labels, speaker tags, or markdown.\n"
    "4. Do not output prefixes or suffixes such as \"Translation:\", \"Answer:\", or quotation marks around the answer.\n"
    "5. Translate the complete speech segment; do not summarize or omit content.\n\n"
    "Entity fidelity rules. Preserve and translate entities as accurately as possible:\n"
    "- PER: person names, speakers, authors, historical figures, and transliterated names. Preserve the full name when audible; do not drop given names or family names.\n"
    "- LOC: countries, cities, regions, landmarks, buildings, roads, facilities, and natural geographic names. Translate or transliterate them into the most natural Chinese form.\n"
    "- ORG: organizations, institutions, companies, schools, government departments, media outlets, hospitals, teams, and international bodies. Preserve the organization as a named unit.\n"
    "- TERM: domain terms, technical concepts, event names, methods, diseases, products, policies, and important specialized expressions. Use precise Chinese terminology when possible.\n"
    "- NUM: numbers, quantities, percentages, money, rankings, temperatures, measurements, and identifiers. Write digits using Arabic numerals, e.g. 1.7, 3, 80%.\n"
    "- TIME: dates, years, periods, durations, festivals, and time expressions. Preserve the time meaning accurately.\n\n"
    "Faithfulness rules:\n"
    "1. Do not drop names, places, organizations, terms, numbers, or time expressions.\n"
    "2. Do not replace one entity with another.\n"
    "3. Do not invent entities that are not present in the speech.\n"
    "4. If some audio is unclear, translate conservatively based on the audio and context.\n"
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma4 SFT with a detailed entity-focused prompt only; no sampling, MBR, or entity list input."
    )
    parser.add_argument("--test-data-path", type=Path, default=root_dir / "data" / "converted_testt_format" / "testt.jsonl")
    parser.add_argument("--test-entity-path", type=Path, default=root_dir / "data" / "converted_testt_format" / "testt.ner.jsonl")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="none")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--entity-lcs-labels", type=str, default="LOC,PER,TERM,NUM,ORG,TIME")
    parser.add_argument(
        "--entity-lcs-skip-labels",
        type=str,
        default="",
        help="Comma-separated labels skipped in entity_lcs. Empty means include all labels.",
    )
    parser.add_argument("--config-out", type=Path, default=None)
    return parser.parse_args()


def build_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    entity_lcs_labels = [label.strip().upper() for label in args.entity_lcs_labels.split(",") if label.strip()]
    entity_lcs_skip_labels = [label.strip().upper() for label in args.entity_lcs_skip_labels.split(",") if label.strip()]
    model_name = args.model_name or f"{Path(args.base_model_path.rstrip('/')).name}-sft-detail-entity-prompt"
    model_config: dict[str, Any] = {
        "kind": "gemma4_audio_lora",
        "name": model_name,
        "base_model_path": args.base_model_path,
        "processor_path": args.processor_path or args.base_model_path,
        "adapter_path": args.adapter_path,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "device_map": args.device_map,
        "sampling_rate": args.sampling_rate,
        "prompt_mode": "fixed",
        "prompt_text": DETAIL_ENTITY_PROMPT,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    if args.attn_implementation:
        model_config["attn_implementation"] = args.attn_implementation
    return {
        "dataset": {
            "path": str(args.test_data_path),
            "format": "converted_translation_jsonl",
            "reference_entity_path": str(args.test_entity_path),
            "audio_prefix_from": args.audio_prefix_from,
            "audio_prefix_to": args.audio_prefix_to,
            "limit": args.limit,
        },
        "model": model_config,
        "evaluation": {
            "mode": "generate_and_score",
            "output_dir": str(args.output_dir),
            "metrics": ["bleu", "chrf", "entity_lcs"],
            "entity_lcs_labels": entity_lcs_labels,
            "entity_lcs_skip_labels": entity_lcs_skip_labels,
            "progress_every": args.progress_every,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_report = validate_testt_sidecar(args.test_data_path, args.test_entity_path, args.limit)
    (args.output_dir / "sidecar.validation.json").write_text(
        json.dumps(sidecar_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = build_eval_config(args)
    config_path = args.config_out or (args.output_dir / "gemma4_entity_prompt_eval.resolved.yaml")
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = run_eval_spec(config_path)
    summary["entity_recall"] = summary.get("entity_recall", summary.get("entity_lcs_recall", 0.0))
    summary["prompt_variant"] = "detail_entity_prompt"
    summary_path = args.output_dir / "metrics.entity_prompt.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
