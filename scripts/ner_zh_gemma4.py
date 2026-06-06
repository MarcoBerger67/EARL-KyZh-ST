from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_BASE_MODEL_PATH = "gemma-4-E4B-it"
DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "converted_testt_format"
    / "train_ky2zh_full285h_stage1.cleaned.jsonl"
)
DEFAULT_PROMPT = """你是一名严格、细致的中文命名实体识别标注员。你的任务是对输入中文文本进行实体识别，并输出高质量、边界准确、标签一致的标注结果。

输出要求：
1. 只输出合法 JSON 数组，不要输出 Markdown、解释、注释、思考过程或额外文本。
2. 每个元素必须且只能包含两个字段：{"text": "实体原文", "label": "实体类型"}。
3. "text" 必须逐字复制输入文本中的连续片段，不要改写、翻译、补全、合并或标准化。
4. "label" 只能是 PER、LOC、ORG、TIME、NUM、TERM、TITLE 之一。
5. 若文本中没有实体，返回 []。

通用标注规则：
1. 标注要足够细致：文本中明确出现的姓名、地点、组织、时间、数字、专有术语、正式名称都应尽量抽取。
2. 只标注文本中“明确出现”的实体，不允许根据常识补充、猜测或改写。
3. 实体边界必须精确，通常保留最小但完整的可指称片段，不要把普通修饰语、虚词、标点放入实体。
4. 不做嵌套标注；若短语内部有多个可标实体，优先选择语义最完整且最自然的实体边界。
5. 同一句中相同实体出现多次时，每次出现都要按出现顺序单独标注。
6. 若类别有冲突，优先判断实体在当前文本中的语义功能，而不是只凭词面。

实体标签定义与注意事项：
- PER：人名、译名、昵称、历史人物、作者、说话人姓名。例如“德米特里·克雷洛夫”“李白”。不要把“总统”“教授”“医生”等职位本身标为 PER。
- LOC：地名、地点、国家、城市、区域、建筑物、自然地理实体、道路、场馆。例如“中国”“比什凯克”“天山”“人民广场”。政府机关如果强调机构职能，标 ORG；如果只是地点，标 LOC。
- ORG：组织、机构、学校、公司、政府部门、媒体、医院、球队、国际组织。例如“联合国”“教育部”“北京大学”“新华社”“苹果公司”。
- TIME：时间表达，包括日期、年份、年代、时段、节日、持续时间。例如“2024年5月”“昨天上午”“21世纪”“三年内”。纯序号不要标 TIME。
- NUM：数字、金额、比例、数量、年龄、温度、编号、排名、尺寸、度量值。例如“3”“80%”“1.7”“20美元”“第5名”“16公里”。如果数字是时间的一部分，整体优先标 TIME。
- TERM：专业术语、领域概念、技术名词、疾病名、算法名、学科名、产品类别、事件/制度/抽象专名。例如“最小风险训练”“语音翻译”“新冠肺炎”“三分法”。普通常用词不要过度标 TERM。
- TITLE：书名、作品名、法律法规、政策文件、会议、项目、课程、计划、正式活动名称。例如“巴黎协定”“人工智能基础课程”“十四五规划”。职位头衔如“总统”“主任”不要标 TITLE，除非它是正式名称的一部分。

示例：
输入：德米特里·克雷洛夫在联合国气候变化大会上表示，2024年全球平均气温上升了1.5摄氏度。
输出：[{"text":"德米特里·克雷洛夫","label":"PER"},{"text":"联合国","label":"ORG"},{"text":"气候变化大会","label":"TITLE"},{"text":"2024年","label":"TIME"},{"text":"全球平均气温","label":"TERM"},{"text":"1.5摄氏度","label":"NUM"}]

输入：其中说明了放置主体对象的最佳位置是垂直和水平将图像三等分的线条交点。
输出：[{"text":"三等分","label":"TERM"}]"""
VALID_LABELS = {"PER", "LOC", "ORG", "TIME", "NUM", "TERM", "TITLE"}

warnings.filterwarnings(
    "ignore",
    message=r"Kwargs passed to `processor\.__call__` have to be in `processor_kwargs` dict, not in `\*\*kwargs`",
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run line-aligned Chinese named entity recognition with Gemma 4 and "
            "save one JSON row per input line."
        )
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Local or remote Hugging Face model path.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input text or jsonl file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output jsonl path. Defaults to <input_stem>.ner.jsonl next to the input file.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "txt", "jsonl", "converted_translation_jsonl"],
        default="converted_translation_jsonl",
        help="Interpret the input as plain text lines or jsonl records.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name used to extract text from each generic jsonl record.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="NER instruction prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens per line.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate placeholder passed through processor_kwargs for API compatibility.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Device map for from_pretrained. Use "none" to disable.',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Explicit device when --device-map none, for example "cuda:0".',
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help='Optional attention implementation, for example "flash_attention_2".',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of input lines for smoke tests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N batches.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def infer_input_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "txt"


def load_lines(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_format = infer_input_format(args.input_path, args.input_format)
    rows: list[dict[str, Any]] = []
    with args.input_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            raw_text = raw_line.rstrip("\n")
            if input_format == "jsonl":
                if not raw_text.strip():
                    rows.append(
                        {
                            "line_number": line_number,
                            "source": raw_text,
                            "text": "",
                        }
                    )
                    continue
                record = json.loads(raw_text)
                text = record.get(args.text_field, "")
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    text = str(text)
                rows.append(
                    {
                        "line_number": line_number,
                        "id": record.get("id", f"line_{line_number}"),
                        "source": raw_text,
                        "text": text.strip(),
                    }
                )
            elif input_format == "converted_translation_jsonl":
                if not raw_text.strip():
                    rows.append(
                        {
                            "line_number": line_number,
                            "id": f"line_{line_number}",
                            "source": raw_text,
                            "text": "",
                        }
                    )
                    continue
                record = json.loads(raw_text)
                text = extract_translation_from_converted_record(record)
                rows.append(
                    {
                        "line_number": line_number,
                        "id": record.get("id", f"line_{line_number}"),
                        "source": raw_text,
                        "text": text,
                    }
                )
            else:
                rows.append(
                    {
                        "line_number": line_number,
                        "id": f"line_{line_number}",
                        "source": raw_text,
                        "text": raw_text.strip(),
                    }
                )
            if args.limit is not None and len(rows) >= args.limit:
                break
    return rows


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        merged = " ".join(text for text in texts if text).strip()
        if merged:
            return merged
    return ""


def extract_translation_from_converted_record(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = extract_text_from_content(message.get("content"))
        if text:
            return text
    return ""


def build_messages(text: str, prompt: str) -> list[dict[str, Any]]:
    payload = f"{prompt}\n\n待标注文本：\n{text}"
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": payload}],
        }
    ]


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def infer_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def get_pad_token_id(processor) -> int | None:
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def normalize_generated_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    if text.startswith("```"):
        text = text[len("```") :].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def extract_json_payload(text: str) -> str:
    stripped = normalize_generated_text(text)
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    match = re.search(r"\[[\s\S]*\]", stripped)
    if match:
        return match.group(0)
    return stripped


def sanitize_entities(parsed: Any) -> list[dict[str, str]]:
    if not isinstance(parsed, list):
        return []
    entities: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        label = item.get("label", "")
        if text is None or label is None:
            continue
        text = str(text).strip()
        label = str(label).strip().upper()
        if not text or label not in VALID_LABELS:
            continue
        entities.append({"text": text, "label": label})
    return entities


def parse_entities_from_output(raw_output: str) -> list[dict[str, str]]:
    payload = extract_json_payload(raw_output)
    try:
        return sanitize_entities(json.loads(payload))
    except json.JSONDecodeError:
        return []


def load_model_and_processor(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_path,
        **model_kwargs,
    )
    if args.device_map.lower() == "none":
        model.to(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return processor, model


def generate_entities(
    rows: list[dict[str, Any]],
    processor,
    model,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    device = infer_device(model)
    pad_token_id = get_pad_token_id(processor)
    outputs: list[dict[str, Any]] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        non_empty_rows = [row for row in batch_rows if row["text"]]
        empty_rows = [row for row in batch_rows if not row["text"]]

        for row in empty_rows:
            outputs.append(
                {
                    "id": row["id"],
                    "entities": [],
                }
            )

        if not non_empty_rows:
            continue

        conversations = [build_messages(row["text"], args.prompt) for row in non_empty_rows]

        model_inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"sampling_rate": args.sampling_rate},
        )
        model_inputs = move_batch_to_device(model_inputs, device)

        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[1]
        generated_only = generated[:, prompt_length:]
        decoded = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for row, raw_output in zip(non_empty_rows, decoded):
            outputs.append(
                {
                    "id": row["id"],
                    "entities": parse_entities_from_output(raw_output),
                }
            )

        batch_index = start // args.batch_size + 1
        if args.progress_every > 0 and batch_index % args.progress_every == 0:
            print(f"Processed {min(start + len(batch_rows), len(rows))}/{len(rows)} lines")

    return outputs


def save_outputs(output_path: Path, outputs: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.input_path.with_suffix(".ner.jsonl")

    rows = load_lines(args)
    processor, model = load_model_and_processor(args)
    outputs = generate_entities(rows, processor, model, args)
    save_outputs(output_path, outputs)

    print(
        json.dumps(
            {
                "input_path": str(args.input_path),
                "output_path": str(output_path),
                "model": args.base_model_path,
                "lines": len(outputs),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
