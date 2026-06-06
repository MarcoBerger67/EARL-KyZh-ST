from __future__ import annotations

import contextlib
import math
from dataclasses import asdict
from typing import Any

import torch

from .types import EmbeddingConfig, Entity, NERConfig


@contextlib.contextmanager
def _nullcontext():
    yield


VALID_LABELS = {"PER", "LOC", "ORG", "TIME", "NUM", "TERM", "TITLE"}


# HanLP NER (MSRA / OntoNotes-style) returns abbreviated Chinese tags that do
# not match the gold sidecar's label set. Without this mapping all predicted
# entities get filtered out by ``sanitize_entities`` and the soft-matching
# reward collapses to 0.
#
#   NR  / PERSON   → PER (人名)
#   NS  / LOCATION → LOC (地名)
#   NT  / ORG.     → ORG (机构名)
#   NZ            → TERM (其他专名)
#   T / DATE / TIME → TIME (时间)
#   M / NUMBER     → NUM (数字)
_HANLP_LABEL_ALIASES: dict[str, str] = {
    "NR": "PER",
    "NRF": "PER",
    "NRJ": "PER",
    "PERSON": "PER",
    "NS": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "NT": "ORG",
    "ORGANIZATION": "ORG",
    "NZ": "TERM",
    "T": "TIME",
    "DATE": "TIME",
    "M": "NUM",
    "NUMBER": "NUM",
    "QUANTITY": "NUM",
    "PERCENT": "NUM",
    "MONEY": "NUM",
    "ORDINAL": "NUM",
    "CARDINAL": "NUM",
}


def _canonicalize_label(label: str) -> str:
    cleaned = label.strip().upper()
    return _HANLP_LABEL_ALIASES.get(cleaned, cleaned)


def install_transformers_tokenizer_compat() -> None:
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return

    if not hasattr(PreTrainedTokenizerBase, "encode_plus"):

        def encode_plus(
            self,
            text: Any,
            text_pair: Any = None,
            add_special_tokens: bool = True,
            padding: bool | str = False,
            truncation: bool | str | None = None,
            max_length: int | None = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: int | None = None,
            padding_side: str | None = None,
            return_tensors: str | None = None,
            return_token_type_ids: bool | None = None,
            return_attention_mask: bool | None = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs: Any,
        ):
            return self(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

        PreTrainedTokenizerBase.encode_plus = encode_plus  # type: ignore[attr-defined]

    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):

        def batch_encode_plus(
            self,
            batch_text_or_text_pairs: list[Any],
            add_special_tokens: bool = True,
            padding: bool | str = False,
            truncation: bool | str | None = None,
            max_length: int | None = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: int | None = None,
            padding_side: str | None = None,
            return_tensors: str | None = None,
            return_token_type_ids: bool | None = None,
            return_attention_mask: bool | None = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs: Any,
        ):
            encoded_inputs = []
            for item in batch_text_or_text_pairs:
                if isinstance(item, tuple) and len(item) == 2:
                    text, text_pair = item
                else:
                    text, text_pair = item, None
                encoded_inputs.append(
                    self(
                        text=text,
                        text_pair=text_pair,
                        add_special_tokens=add_special_tokens,
                        padding=False,
                        truncation=truncation,
                        max_length=max_length,
                        stride=stride,
                        is_split_into_words=is_split_into_words,
                        pad_to_multiple_of=None,
                        padding_side=padding_side,
                        return_tensors=None,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        return_overflowing_tokens=return_overflowing_tokens,
                        return_special_tokens_mask=return_special_tokens_mask,
                        return_offsets_mapping=return_offsets_mapping,
                        return_length=return_length,
                        verbose=verbose,
                        **kwargs,
                    )
                )
            return self.pad(
                encoded_inputs,
                padding=padding,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors,
            )

        PreTrainedTokenizerBase.batch_encode_plus = batch_encode_plus  # type: ignore[attr-defined]


def install_transformers_backend_compat() -> None:
    """Patch newer Transformers import metadata for older HanLP imports.

    Some Transformers builds expose objects guarded by a ``tensorflow_text``
    backend, while the runtime BACKENDS_MAPPING does not define that optional
    backend. HanLP imports BertTokenizer through the lazy Transformers module,
    so make the missing optional backend explicit instead of failing the import.
    """
    try:
        from transformers.utils import import_utils
    except Exception:
        return

    mapping = getattr(import_utils, "BACKENDS_MAPPING", None)
    if not isinstance(mapping, dict) or "tensorflow_text" in mapping:
        return

    def is_tensorflow_text_available() -> bool:
        try:
            import tensorflow_text  # type: ignore  # noqa: F401
        except Exception:
            return False
        return True

    mapping["tensorflow_text"] = (
        is_tensorflow_text_available,
        "tensorflow-text is required for this object but is not installed.",
    )


def patch_transformers_encode_plus_compat(component: Any) -> None:
    """HanLP releases before recent Transformers changes may call encode_plus."""
    install_transformers_backend_compat()
    install_transformers_tokenizer_compat()
    visited: set[int] = set()

    def safe_getattr(obj: Any, name: str) -> Any:
        try:
            return getattr(obj, name)
        except Exception:
            return None

    def patch_object(obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 6:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        encode_plus = safe_getattr(obj, "encode_plus")
        private_encode_plus = safe_getattr(obj, "_encode_plus")
        if encode_plus is None and private_encode_plus is not None:
            try:
                setattr(obj, "encode_plus", private_encode_plus)
            except Exception:
                pass
        batch_encode_plus = safe_getattr(obj, "batch_encode_plus")
        private_batch_encode_plus = safe_getattr(obj, "_batch_encode_plus")
        if batch_encode_plus is None and private_batch_encode_plus is not None:
            try:
                setattr(obj, "batch_encode_plus", private_batch_encode_plus)
            except Exception:
                pass

        if isinstance(obj, dict):
            values = obj.values()
        elif isinstance(obj, (list, tuple, set)):
            values = obj
        else:
            try:
                values = vars(obj).values()
            except Exception:
                return

        for value in values:
            if isinstance(value, (str, bytes, int, float, bool)):
                continue
            patch_object(value, depth + 1)

    patch_object(component)


def _normalize_entity(item: Any) -> Entity | None:
    if isinstance(item, Entity):
        return item
    if isinstance(item, dict):
        text = str(item.get("text", "") or "").strip()
        label = _canonicalize_label(str(item.get("label", "") or ""))
        if text and label:
            return Entity(text=text, label=label)
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        text = str(item[0]).strip()
        label = _canonicalize_label(str(item[1]))
        if text and label:
            return Entity(text=text, label=label)
    return None


def sanitize_entities(items: list[Any]) -> list[Entity]:
    entities: list[Entity] = []
    for item in items:
        entity = _normalize_entity(item)
        if entity is None:
            continue
        if entity.label in VALID_LABELS:
            entities.append(entity)
    return entities


def _extract_entities_from_hanlp_output(output: Any, tokens: list[str] | None = None) -> list[Entity]:
    if isinstance(output, dict):
        if "ner" in output:
            return _extract_entities_from_hanlp_output(output["ner"], tokens=tokens)
        collected: list[Entity] = []
        for value in output.values():
            if isinstance(value, (list, tuple)):
                collected.extend(_extract_entities_from_hanlp_output(value, tokens=tokens))
        return sanitize_entities(collected)

    if isinstance(output, list):
        entities: list[Entity] = []
        for item in output:
            if isinstance(item, list):
                entities.extend(_extract_entities_from_hanlp_output(item, tokens=tokens))
                continue
            entity = _normalize_entity(item)
            if entity is not None:
                entities.append(entity)
                continue
            if isinstance(item, tuple) and len(item) >= 4 and tokens is not None:
                label = _canonicalize_label(str(item[1]))
                start = max(int(item[2]), 0)
                end = min(int(item[3]), len(tokens))
                text = "".join(tokens[start:end]).strip()
                if text and label:
                    entities.append(Entity(text=text, label=label))
        return sanitize_entities(entities)

    return []


class HanLPEntityExtractor:
    def __init__(self, config: NERConfig, device: str | None = None) -> None:
        self.config = config
        self._tokenizer = None
        self._ner = None
        self._target_device = device

    def _resolve_hanlp_resource(self, hanlp: Any, path: str | None, pretrained_name: str) -> Any:
        if path:
            return path
        for group_name in ("tok", "ner"):
            pretrained_group = getattr(getattr(hanlp, "pretrained", None), group_name, None)
            if pretrained_group and hasattr(pretrained_group, pretrained_name):
                return getattr(pretrained_group, pretrained_name)
        return pretrained_name

    def _load(self) -> None:
        if self._tokenizer is not None and self._ner is not None:
            return
        install_transformers_backend_compat()
        try:
            import hanlp
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "HanLP is required for entity metrics. Install it with `pip install hanlp`."
            ) from exc

        tok_resource = self._resolve_hanlp_resource(
            hanlp, self.config.tokenizer_path, self.config.tokenizer_model
        )
        ner_resource = self._resolve_hanlp_resource(
            hanlp, self.config.ner_path, self.config.ner_model
        )
        self._tokenizer = hanlp.load(tok_resource)
        self._ner = hanlp.load(ner_resource)
        patch_transformers_encode_plus_compat(self._tokenizer)
        patch_transformers_encode_plus_compat(self._ner)
        if self._target_device:
            for component in (self._tokenizer, self._ner):
                mover = getattr(component, "to", None)
                if callable(mover):
                    try:
                        mover(self._target_device)
                    except Exception:
                        pass

    def extract(self, text: str) -> list[Entity]:
        text = text.strip()
        if not text:
            return []
        self._load()
        tokens = self._tokenizer(text)
        if isinstance(tokens, str):
            tokens = list(tokens)
        ner_output = self._ner(tokens)
        return sanitize_entities(_extract_entities_from_hanlp_output(ner_output, tokens=tokens))


class MBertEmbedder:
    def __init__(self, config: EmbeddingConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(self.config.model_name)
        self._model.to(self.device)
        self._model.eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 1), dtype=torch.float32)
        self._load()
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            if self.config.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp_min(1)
                pooled = summed / denom
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.float().cpu()


_GEMMA_NER_PROMPT = (
    "你是一名严格、细致的中文命名实体识别标注员。请从给定中文文本中抽取所有明确出现的实体，"
    "并只输出合法 JSON 数组。\n\n"
    "输出要求：\n"
    '1. 每项格式必须为 {{"text": "实体原文", "label": "实体类型"}}。\n'
    "2. text 必须逐字复制输入文本中的连续片段，不要改写、翻译、补全、合并或标准化。\n"
    "3. label 只能是 PER、LOC、ORG、TIME、NUM、TERM、TITLE 之一。\n"
    "4. 无实体时输出 []；不要输出 Markdown、解释、注释或额外文本。\n\n"
    "通用规则：\n"
    "1. 标注要足够细致：明确出现的姓名、地点、组织、时间、数字、专有术语、正式名称都应尽量抽取。\n"
    "2. 只标注文本中明确出现的实体，不要根据常识补充或猜测。\n"
    "3. 边界要精确，保留最小但完整的可指称片段，不要把普通修饰语、虚词、标点放入实体。\n"
    "4. 不做嵌套标注；若短语内部有多个可标实体，优先选择语义最完整且最自然的实体边界。\n"
    "5. 同一实体出现多次时，每次出现都要按出现顺序单独输出。\n\n"
    "类型说明：\n"
    "- PER：人名、译名、昵称、历史人物、作者、说话人姓名。例如“德米特里·克雷洛夫”“李白”。职位本身不标 PER。\n"
    "- LOC：国家、城市、地区、地点、建筑物、自然地理实体、道路、场馆。例如“中国”“比什凯克”“天山”。\n"
    "- ORG：组织、机构、公司、学校、政府部门、媒体、医院、球队、国际组织。例如“联合国”“教育部”“新华社”。\n"
    "- TIME：日期、年份、年代、时段、节日、持续时间。例如“2024年5月”“昨天上午”“21世纪”“三年内”。\n"
    "- NUM：数字、数量、比例、金额、年龄、温度、编号、排名、尺寸、度量值。例如“3”“80%”“1.7”“20美元”。\n"
    "- TERM：专业术语、领域概念、技术名词、疾病名、算法名、学科名、事件/制度/抽象专名。例如“最小风险训练”“语音翻译”“三分法”。\n"
    "- TITLE：书名、作品名、法律法规、政策文件、会议、项目、课程、计划、正式活动名称。例如“巴黎协定”“十四五规划”。\n\n"
    "示例：\n"
    "输入：德米特里·克雷洛夫在联合国气候变化大会上表示，2024年全球平均气温上升了1.5摄氏度。\n"
    '输出：[{{"text":"德米特里·克雷洛夫","label":"PER"}},{{"text":"联合国","label":"ORG"}},'
    '{{"text":"气候变化大会","label":"TITLE"}},{{"text":"2024年","label":"TIME"}},'
    '{{"text":"全球平均气温","label":"TERM"}},{{"text":"1.5摄氏度","label":"NUM"}}]\n\n'
    "文本：{text}"
)


class GemmaEntityExtractor:
    """Use a Gemma instruction-tuned model to extract named entities via prompting.

    Two construction modes:
      * ``GemmaEntityExtractor(model_path, device)`` — loads a standalone copy.
      * ``GemmaEntityExtractor.from_shared_model(model, tokenizer)`` — reuses
        the training model so we don't duplicate the weights in VRAM.

    Inference always runs under ``torch.inference_mode()`` and (when available)
    with PEFT adapters disabled, so it never interferes with training gradients
    or pollute the NER call with translation-LoRA biases.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        self._model_path = model_path
        self._device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._shared: bool = False  # True when reusing an externally-owned model

    @classmethod
    def from_shared_model(cls, model: Any, tokenizer: Any) -> "GemmaEntityExtractor":
        instance = cls.__new__(cls)
        instance._model_path = None
        instance._device = None
        instance._model = model
        instance._tokenizer = tokenizer
        instance._shared = True
        return instance

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        self._model.eval()

    def _resolve_device(self) -> Any:
        if self._shared:
            try:
                return next(self._model.parameters()).device
            except StopIteration:
                return "cpu"
        return self._device

    def extract(self, text: str) -> list[Entity]:
        if self._model is None:
            self._load()
        prompt = _GEMMA_NER_PROMPT.format(text=text.strip())
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted: str = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
        device = self._resolve_device()
        inputs = self._tokenizer(formatted, return_tensors="pt").to(device)
        pad_id = getattr(self._tokenizer, "eos_token_id", None)

        # Disable PEFT adapters during NER so the translation-LoRA does not
        # bias the entity extraction.  No-op on non-PEFT models.
        disable_adapter_cm = getattr(self._model, "disable_adapter", None)
        was_training = getattr(self._model, "training", False)
        if was_training:
            self._model.eval()
        try:
            ctx = disable_adapter_cm() if callable(disable_adapter_cm) else _nullcontext()
            with ctx, torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
        except Exception as _exc:
            import sys
            print(f"[GemmaEntityExtractor] generate() failed: {_exc}", file=sys.stderr, flush=True)
            return []
        finally:
            if was_training:
                self._model.train()

        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        generated = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        result = self._parse(generated)
        import os as _os
        if _os.environ.get("DEBUG_NER_EXTRACT"):
            import sys, json as _json
            print(f"[GemmaNER] input={text[:60]!r} → raw={generated[:120]!r} → entities={_json.dumps([{'text':e.text,'label':e.label} for e in result], ensure_ascii=False)}", file=sys.stderr, flush=True)
        return result

    def _parse(self, text: str) -> list[Entity]:
        import json
        import re

        m = re.search(r"\[[\s\S]*?\]", text)
        if not m:
            return []
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return []
        entities: list[Entity] = []
        seen: set[tuple[str, str]] = set()
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            entity_text = str(item.get("text", "") or "").strip()
            label = _canonicalize_label(str(item.get("label", "") or ""))
            if not entity_text or label not in VALID_LABELS:
                continue
            key = (entity_text, label)
            if key not in seen:
                seen.add(key)
                entities.append(Entity(text=entity_text, label=label))
        return entities


def cosine_similarity_matrix(left: torch.Tensor, right: torch.Tensor) -> list[list[float]]:
    if left.numel() == 0 or right.numel() == 0:
        return []
    sims = left @ right.T
    return [[float(value) for value in row] for row in sims.tolist()]


def _min_cost_max_flow(cost_matrix: list[list[int]]) -> list[tuple[int, int]]:
    left_size = len(cost_matrix)
    right_size = len(cost_matrix[0]) if cost_matrix else 0
    node_count = 2 + left_size + right_size
    source = 0
    sink = node_count - 1
    graph: list[list[dict[str, int]]] = [[] for _ in range(node_count)]

    def add_edge(u: int, v: int, capacity: int, cost: int) -> None:
        graph[u].append({"to": v, "rev": len(graph[v]), "cap": capacity, "cost": cost})
        graph[v].append({"to": u, "rev": len(graph[u]) - 1, "cap": 0, "cost": -cost})

    for i in range(left_size):
        add_edge(source, 1 + i, 1, 0)
    for j in range(right_size):
        add_edge(1 + left_size + j, sink, 1, 0)
    for i in range(left_size):
        for j in range(right_size):
            if cost_matrix[i][j] >= 0:
                add_edge(1 + i, 1 + left_size + j, 1, cost_matrix[i][j])

    while True:
        dist = [math.inf] * node_count
        in_queue = [False] * node_count
        prev_node = [-1] * node_count
        prev_edge = [-1] * node_count
        dist[source] = 0
        queue = [source]
        in_queue[source] = True

        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            for edge_index, edge in enumerate(graph[u]):
                if edge["cap"] <= 0:
                    continue
                v = edge["to"]
                new_dist = dist[u] + edge["cost"]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev_node[v] = u
                    prev_edge[v] = edge_index
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        if prev_node[sink] == -1 or dist[sink] >= 0:
            break

        v = sink
        while v != source:
            u = prev_node[v]
            edge_index = prev_edge[v]
            edge = graph[u][edge_index]
            edge["cap"] -= 1
            graph[v][edge["rev"]]["cap"] += 1
            v = u

    matches: list[tuple[int, int]] = []
    for i in range(left_size):
        u = 1 + i
        for edge in graph[u]:
            if 1 + left_size <= edge["to"] < sink and edge["cap"] == 0:
                matches.append((i, edge["to"] - 1 - left_size))
    return matches


def soft_entity_matching(
    predicted: list[Entity],
    reference: list[Entity],
    similarity_matrix: list[list[float]],
    tau: float,
) -> list[tuple[int, int, float]]:
    """Match predicted and reference entities by semantic similarity only.

    Label-agnostic by design: ``sanitize_entities`` upstream still drops
    garbage tags (O/MISC/X/etc) via the ``VALID_LABELS`` whitelist, but here
    we do a single global min-cost / max-flow assignment that ignores the
    label field. Reasons:

    * NER models disagree on label vocabularies (HanLP MSRA uses NR/NS/NT,
      OntoNotes uses PERSON/LOC/ORG, gold sidecar uses PER/LOC/ORG/…).
      Bucketing by label makes the reward extremely fragile to the choice of
      NER backend.
    * Same surface entity routinely shifts category across sentences (e.g.
      "中国" LOC vs ORG, "三年" NUM vs TIME). A strict same-label rule turns
      these into false negatives.
    * The ``tau`` cosine threshold already provides a semantic precision
      filter — "北京" and "约翰" will not cross it just because their labels
      happen to match.
    """
    left_size = len(predicted)
    right_size = len(reference)
    if left_size == 0 or right_size == 0:
        return []

    # _min_cost_max_flow treats cost_matrix[i][j] < 0 as "no edge" and
    # cost_matrix[i][j] >= 0 as a valid edge with that cost.  We want to
    # *maximise* cosine similarity, so we translate:
    #
    #   cost = 10**6 - scaled   where scaled = round(similarity * 10**6)
    #
    # Higher similarity → smaller cost → preferred by the minimiser.
    # tau-filtered pairs always satisfy  0 <= cost <= (1 - tau) * 10**6.
    # Pairs below tau keep the sentinel value -1 (no edge).
    base_bonus = 10**6
    cost_matrix = [[-1 for _ in range(right_size)] for _ in range(left_size)]
    for i in range(left_size):
        row_sims = similarity_matrix[i] if i < len(similarity_matrix) else []
        for j in range(right_size):
            if j >= len(row_sims):
                continue
            score = row_sims[j]
            if score >= tau:
                scaled = int(round(score * 10**6))
                cost_matrix[i][j] = base_bonus - scaled  # >= 0

    raw_matches = _min_cost_max_flow(cost_matrix)
    matches: list[tuple[int, int, float]] = []
    for left_id, right_id in raw_matches:
        matches.append((left_id, right_id, similarity_matrix[left_id][right_id]))
    return matches


def entity_records_to_json(rows: list[tuple[str, list[Entity], str, str]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for sample_id, entities, text, source in rows:
        serialized.append(
            {
                "id": sample_id,
                "text": text,
                "source": source,
                "entities": [asdict(entity) for entity in entities],
            }
        )
    return serialized
