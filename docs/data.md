# Data Format

The training and evaluation scripts expect JSONL records in a chat-style speech translation format.

## Translation JSONL

Each row has an `id` and a `messages` list. The user message contains an audio path and a prompt. The assistant message contains the Chinese reference translation.

```json
{"id":"sample_001","messages":[{"role":"user","content":[{"type":"audio","path":"examples/audio/sample_001.wav"},{"type":"text","text":"Translate the speech into Simplified Chinese."}]},{"role":"assistant","content":[{"type":"text","text":"北京大学位于北京。"}]}]}
```

Audio paths can be local paths, cluster paths, or rewritten through `audio_prefix_from` and `audio_prefix_to` in the YAML config.

## Entity Sidecar

Reference entities are stored separately and joined by `id`.

```json
{"id":"sample_001","entities":[{"text":"北京大学","label":"ORG"},{"text":"北京","label":"LOC"}]}
```

Supported labels used in the paper include:

```text
PER, LOC, ORG, TERM, NUM, TIME, TITLE
```

The strict paper metric is `entity_key_recall`: each normalized reference entity receives credit only when its full normalized text appears in the prediction.

## Full Corpus

The full corpus is not included in this repository. The paper corpus contains approximately 90k samples and 266 hours, built from FLEURS-train, Common Voice, Kyrgyz speech data, web-crawled speech, and TTS-synthesized samples.

