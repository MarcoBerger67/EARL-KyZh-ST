from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


VALID_LABELS = {"PER", "LOC", "ORG", "TIME", "NUM", "TERM", "TITLE"}

DEFAULT_PROMPT = """你是一名严格、细致的中文命名实体识别标注员。你的任务是从给定中文译文中抽取所有明确出现的实体，并输出边界准确、类型一致的 JSON 数组。

输出要求：
1. 只输出合法 JSON 数组，不要输出 Markdown、解释、注释、思考过程或额外文本。
2. 每个元素必须且只能包含两个字段：{"text": "实体原文", "label": "实体类型"}。
3. "text" 必须逐字复制输入文本中的连续片段，不要改写、翻译、补全、合并或标准化。
4. "label" 只能是 PER、LOC、ORG、TIME、NUM、TERM、TITLE 之一。
5. 如果没有任何实体，输出 []。

通用标注规则：
1. 标注要足够细致：文本中明确出现的姓名、地点、组织、时间、数字、专有术语、正式名称都应尽量抽取。
2. 只标注文本中明确出现的实体，不要根据常识、上下文或外部知识补充不存在的实体。
3. 实体边界要精确，通常保留最小但完整的可指称片段，不要把普通修饰语、虚词、标点放入实体。
4. 不做嵌套标注；若短语内部有多个可标实体，优先选择语义最完整且最自然的实体边界。
5. 同一个实体在文本中出现多次时，每次出现都要按出现顺序单独输出。
6. 保持实体在原文中的首次出现顺序。

类型定义与注意事项：
- PER：人名、译名、昵称、历史人物、作者、说话人姓名。例如“德米特里·克雷洛夫”“李白”。不要把“总统”“教授”“医生”等职位本身标为 PER。
- LOC：国家、城市、地区、地点、建筑物、自然地理实体、道路、场馆。例如“中国”“比什凯克”“天山”“人民广场”。政府机关如果强调机构职能，标 ORG；如果只是地点，标 LOC。
- ORG：组织、机构、公司、学校、政府部门、媒体、医院、球队、国际组织。例如“联合国”“教育部”“北京大学”“新华社”“苹果公司”。
- TIME：具体或泛化的时间表达，包括日期、年份、年代、时段、节日、持续时间。例如“2024年5月”“昨天上午”“21世纪”“三年内”。纯序号不要标 TIME。
- NUM：数字、数量、比例、金额、年龄、温度、编号、排名、尺寸、度量值。例如“3”“80%”“1.7”“20美元”“第5名”“16公里”。如果数字是时间的一部分，整体优先标 TIME。
- TERM：专业术语、领域概念、技术名词、疾病名、算法名、学科名、产品类别、事件/制度/抽象专名。例如“最小风险训练”“语音翻译”“新冠肺炎”“三分法”。普通常用词不要过度标 TERM。
- TITLE：书名、作品名、法律法规、政策文件、会议、项目、课程、计划、正式活动名称。例如“巴黎协定”“人工智能基础课程”“十四五规划”。职位头衔如“总统”“主任”不要标 TITLE，除非它是正式名称的一部分。

示例：
输入：德米特里·克雷洛夫在联合国气候变化大会上表示，2024年全球平均气温上升了1.5摄氏度。
输出：[{"text":"德米特里·克雷洛夫","label":"PER"},{"text":"联合国","label":"ORG"},{"text":"气候变化大会","label":"TITLE"},{"text":"2024年","label":"TIME"},{"text":"全球平均气温","label":"TERM"},{"text":"1.5摄氏度","label":"NUM"}]

输入：其中说明了放置主体对象的最佳位置是垂直和水平将图像三等分的线条交点。
输出：[{"text":"三等分","label":"TERM"}]
"""


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_input = root_dir / "data" / "converted_testt_format" / "testt.jsonl"
    parser = argparse.ArgumentParser(
        description="Build testt.ner.jsonl with Gemma-4 text NER over reference translations."
    )
    parser.add_argument("--base-model-path", type=str, default="gemma-4-E4B-it")
    parser.add_argument("--input-path", type=Path, default=default_input)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Reuse rows already present in the output file.")
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text", "") or "").strip()
                if text:
                    texts.append(text)
        return " ".join(texts).strip()
    return ""


def extract_reference_translation(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = extract_text_from_content(message.get("content"))
        if text:
            return text
    return ""


def load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("id") or record.get("key") or f"line_{line_number}")
            rows.append(
                {
                    "id": record_id,
                    "text": extract_reference_translation(record),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("id")
            entities = row.get("entities")
            if isinstance(record_id, str) and isinstance(entities, list):
                existing[record_id] = {"id": record_id, "entities": sanitize_entities(entities)}
    return existing


def build_messages(text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{DEFAULT_PROMPT}\n\nChinese translation:\n{text}",
                }
            ],
        }
    ]


def normalize_generated_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    if text.startswith("```"):
        text = text[len("```") :].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text.strip()


def extract_json_array(text: str) -> str:
    text = normalize_generated_text(text)
    if text.startswith("[") and text.endswith("]"):
        return text
    match = re.search(r"\[[\s\S]*\]", text)
    return match.group(0) if match else "[]"


def sanitize_entities(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        label = str(item.get("label", "") or item.get("type", "") or "").strip().upper()
        if not text or label not in VALID_LABELS:
            continue
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"text": text, "label": label})
    return entities


def parse_entities(raw_output: str) -> list[dict[str, str]]:
    try:
        return sanitize_entities(json.loads(extract_json_array(raw_output)))
    except json.JSONDecodeError:
        return []


def load_model_and_processor(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {"torch_dtype": resolve_torch_dtype(args.torch_dtype)}
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(args.base_model_path, trust_remote_code=True, **model_kwargs)
    if args.device_map.lower() == "none":
        model.to(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return processor, model


def infer_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def get_pad_token_id(processor: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def generate_entities(
    rows: list[dict[str, str]],
    existing: dict[str, dict[str, Any]],
    processor: Any,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    outputs = dict(existing)
    pending_rows = [row for row in rows if row["id"] not in outputs]
    device = infer_device(model)
    pad_token_id = get_pad_token_id(processor)

    for start in range(0, len(pending_rows), args.batch_size):
        batch_rows = pending_rows[start : start + args.batch_size]
        non_empty = [row for row in batch_rows if row["text"].strip()]
        for row in batch_rows:
            if not row["text"].strip():
                outputs[row["id"]] = {"id": row["id"], "entities": []}
        if not non_empty:
            continue

        model_inputs = processor.apply_chat_template(
            [build_messages(row["text"]) for row in non_empty],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = move_batch_to_device(model_inputs, device)
        prompt_length = model_inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        decoded = processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for row, raw_output in zip(non_empty, decoded):
            outputs[row["id"]] = {"id": row["id"], "entities": parse_entities(raw_output)}

        batch_index = start // args.batch_size + 1
        if args.progress_every > 0 and batch_index % args.progress_every == 0:
            print(f"Processed {min(start + len(batch_rows), len(pending_rows))}/{len(pending_rows)} pending rows")

    return outputs


def write_outputs(path: Path, rows: list[dict[str, str]], outputs: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            output = outputs.get(row["id"], {"id": row["id"], "entities": []})
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_report(path: Path, input_path: Path, output_path: Path, rows: list[dict[str, str]], outputs: dict[str, dict[str, Any]]) -> None:
    entity_count = sum(len(outputs.get(row["id"], {}).get("entities", [])) for row in rows)
    non_empty_count = sum(bool(outputs.get(row["id"], {}).get("entities", [])) for row in rows)
    label_counts: dict[str, int] = {}
    for row in rows:
        for entity in outputs.get(row["id"], {}).get("entities", []):
            label = entity["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": len(rows),
        "non_empty_entity_rows": non_empty_count,
        "entity_count": entity_count,
        "label_counts": dict(sorted(label_counts.items())),
        "format": {"id": "string", "entities": [{"text": "string", "label": "PER|LOC|ORG|TIME|NUM|TERM|TITLE"}]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.input_path.with_suffix(".ner.jsonl")
    report_path = args.report_path or output_path.with_suffix(".report.json")
    rows = load_rows(args.input_path, args.limit)
    existing = load_existing(output_path) if args.resume else {}
    processor, model = load_model_and_processor(args)
    outputs = generate_entities(rows, existing, processor, model, args)
    write_outputs(output_path, rows, outputs)
    write_report(report_path, args.input_path, output_path, rows, outputs)
    print(json.dumps({"input_path": str(args.input_path), "output_path": str(output_path), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
