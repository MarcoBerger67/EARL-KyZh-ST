# 技术指南

本文件说明本仓库的**数据位置、代码结构、配置体系、运行流程与输出产物**，作为 README 的补充。数据 schema 见 [data.md](data.md)，端到端复现见 [reproduction.md](reproduction.md)，发布前检查见 [release_checklist.md](release_checklist.md)。

---

## 1. 总览

EARL 是两阶段训练流程：

1. **SFT（第一阶段）** —— `train_gemma4_sft_qlora.py`，对 Gemma-4 做 LoRA/QLoRA 监督微调，得到基础翻译策略。
2. **GRPO（第二阶段）** —— 以 SFT 适配器为初始化与参考模型，用 BLEU + chrF + 参考侧实体召回的联合奖励继续优化。统一入口 `train_gemma4_grpo_lora.py`，通常通过 `run_fca_grpo_experiments.py` 按配置驱动。

评测统一在 Google FLEURS ky→zh 上进行（dev 选超参，test 报结果）。

---

## 2. 目录结构

```text
configs/
  eval_suite/        run_eval_suite.py 的评测配置
  fca_grpo/          GRPO 实验配置（common.yaml + 各实验 yaml）
docs/                本指南、数据格式、复现、发布说明
examples/            极小离线评测样例（随仓库附带）
scripts/             训练 / GRPO / 基线 / 评测 / 数据准备代码
tests/               最小化指标测试
```

---

## 3. 数据位置与准备

仓库**只含源码与极小样例**，真实数据不再分发（被 `.gitignore` 排除）。脚本默认相对仓库根解析路径，也可用命令行参数或环境变量覆盖。

### 3.1 推荐用环境变量（见 reproduction.md）

```bash
export DATA_ROOT=/path/to/kyzh_data
export MODEL_ROOT=/path/to/models
export OUT_ROOT=/path/to/earl_outputs
```

### 3.2 默认数据布局

`configs/fca_grpo/common.yaml` 中 `data:` 段引用的相对路径（位于仓库根 `data/converted_testt_format/` 下）：

| 配置键 | 默认文件 | 用途 |
|---|---|---|
| `sft_train_data_path` | `train_ky2zh_full285h_stage3.cleaned.jsonl` | SFT 训练集 |
| `grpo_train_data_path` | `train_ky2zh_full285h_stage3.cleaned.final.evalfilter.jsonl` | GRPO 训练集 |
| `train_entity_path` | `train_ky2zh_full285h_stage3.cleaned.final.ner.jsonl` | 训练参考侧实体 sidecar |
| `val_data_path` | `val_ky2zh_full285h.jsonl` | 验证集（FLEURS-dev） |
| `test_data_path` | `testt.jsonl` | 测试集（FLEURS-test） |

- 翻译数据为 chat 风格 JSONL（`id` + `messages`，user 含音频路径与提示，assistant 含中文参考）。
- 实体 sidecar 为 `*.ner.jsonl`，按 `id` 与翻译数据对齐，每行 `{"id":..., "entities":[{"text":..., "label":...}]}`，七类标签 `PER / LOC / ORG / TERM / NUM / TIME / TITLE`。
- 音频路径可用 `audio_prefix_from` / `audio_prefix_to` 在 YAML 中重写到本地。

详细 schema 见 [data.md](data.md)。极小可跑样例见 `examples/`（`mini_test.jsonl`、`mini_test.ner.jsonl`、`mini_predictions.jsonl`）。

### 3.3 模型与输出根

| 配置键 | 默认 | 说明 |
|---|---|---|
| `runtime.base_model_path` | `model/gemma-4-e2b-it` | 基座；主实验用 E4B，经 `--base-model-path` 覆盖为 `gemma-4-e4b-it` |
| `runtime.sft_output_root` | `model/sft` | SFT 输出根 |
| `runtime.output_root` | `model/grpo` | GRPO 输出根 |

---

## 4. 代码导览（scripts/）

### 4.1 核心训练

| 文件 | 作用 |
|---|---|
| `train_gemma4_sft_qlora.py` | 第一阶段：Gemma-4 LoRA/QLoRA SFT（基于 HF `Trainer`） |
| `train_gemma4_grpo_lora.py` | 第二阶段入口（薄封装，调用 `fca_grpo_runtime.train()`） |
| `fca_grpo_runtime.py` | 统一 GRPO 入口；按 `--grpo-objective` 分发到 risk 或 clipped 运行时 |
| `fca_grpo_group_risk_impl.py` | `group_relative_risk_kl` 目标的完整实现：数据加载、候选生成、奖励计算、组内 z-score 归一化、group-relative risk 损失 + KL、训练循环、断点续训、验证。**奖励与实体度量的公共函数也在此**，被多处复用 |
| `fca_grpo_risk_runtime.py` | risk 运行时（`from fca_grpo_group_risk_impl import *` 的薄再导出） |
| `fca_grpo_clipped_runtime.py` | `clipped_grpo` 目标：PPO 式截断重要性比 + KL |
| `run_fca_grpo_experiments.py` | 配置驱动的实验启动器：合并 `common.yaml` + 实验 yaml，构建训练与评测命令，支持单卡/多卡（torchrun / accelerate）与 dry-run |

### 4.2 评测

| 文件 | 作用 |
|---|---|
| `eval_testt_bleu_chrf.py` | 单模型（基座 + 适配器）在测试集上的评测，产出 BLEU / chrF / 实体召回 |
| `run_eval_suite.py` + `eval_suite/` | 统一评测套件：离线打分或加载模型生成后打分。子模块：`config.py`（配置解析）、`data.py`（数据加载）、`model_adapters.py`（模型/离线适配）、`audio.py`（音频）、`text_metrics.py`（BLEU/chrF/LCS/严格 key recall）、`entity_backend.py`（HanLP NER）、`io_utils.py`、`runner.py`、`types.py` |
| `repeated_eval_testt_bleu_chrf.py` | 多次重复评测（统计稳定性） |
| `score_existing_predictions_bleu_chrf.py` | 对已有预测重新打分 |
| `recompute_offline_metrics.py` | 离线重算指标 |

### 4.3 我方模型评测与测试时干预对比（`scripts/`）

以下脚本评测 EARL 的基座/检查点，并实现论文中“训练式 vs 测试时实体干预”的对比，属于本文方法的一部分，保留在 `scripts/` 主目录。

| 文件 | 说明 |
|---|---|
| `run_gemma4_base_testt_eval.py` | 未微调 Gemma-4 端到端（同族基线） |
| `run_gemma4_entity_prompt_testt_eval.py` | 实体提示（oracle 软提示，性能上界） |
| `run_gemma4_entity_inject_testt_eval.py` | 词汇约束解码 / 实体注入 |
| `run_gemma4_mbr_testt_eval.py` | MBR / 实体重排 |

### 4.4 第三方基线（`scripts/baselines/`）

> 以下脚本把现成的第三方模型接入统一评测，用于复现论文中的对比行。它们依赖非 EARL 的外部权重/API，属 best-effort，可能随上游版本失效，**不影响 EARL 本体的训练与评测**。每个脚本顶部均有对应免责说明。

| 文件 | 基线 |
|---|---|
| `baselines/run_whisper_nllb_testt_eval.py` | 级联 Whisper-large-v3 → NLLB-200 |
| `baselines/run_qwen3_asr_nllb_testt_eval.py` | 级联 Qwen3-ASR → NLLB-200 |
| `baselines/run_qwen3_asr_madlad400_testt_eval.py` | 级联 Qwen3-ASR → madlad400 |
| `baselines/run_qwen3_asr_milmmt_testt_eval.py` | 级联 Qwen3-ASR → MilMMT |
| `baselines/run_wav2vec2_nllb_testt_eval.py` | 级联 wav2vec2-XLSR → NLLB-200 |
| `baselines/run_seamless_m4t_testt_eval.py` | SeamlessM4T v2 端到端 |
| `baselines/run_madlad400_mt_testt_eval.py` | madlad400 文本翻译 |
| `baselines/run_qwen2_audio_testt_eval.py` | Qwen2-Audio 端到端 |

> 闭源 API 基线（GPT-Audio、Qwen3-omni、GLM 级联等）通过评测套件的 API 适配器调用，凭证从环境变量读取（见 [SECURITY.md](../SECURITY.md)）。

### 4.5 基线训练（`scripts/baselines/`）

`baselines/train_madlad400_mt_sft.py`、`baselines/train_milmmt_text_sft.py`、`baselines/train_qwen25_omni_sft.py`、`baselines/train_seamless_m4t_sft.py`、`baselines/train_wav2vec2_kyrgyz_asr.py` —— 在本文语料上微调级联/端到端基线（用于数据有效性对比），同属第三方 best-effort 脚本。

### 4.6 数据准备

| 文件 | 作用 |
|---|---|
| `prepare_gemma4_sft_from_compact.py` | 由紧凑格式构建 Gemma-4 SFT JSONL |
| `prepare_qwen3_asr_sft_data.py` | 构建 ASR SFT 数据 |
| `ner_zh_gemma4.py` | 中文参考侧 NER（抽取金标实体） |
| `build_testt_ner_gemma4.py` | 为测试集构建实体 sidecar |
| `filter_dataset_by_repeated_metrics.py` | 按重复打分指标清洗数据 |
| `filter_jsonl_by_entity_sidecar.py` | 按实体 sidecar 过滤 JSONL |

---

## 5. 配置体系与覆盖优先级

### 5.1 GRPO 配置（`configs/fca_grpo/`）

- **`common.yaml`** —— 全局默认，分四段：
  - `runtime`：输出根、基座路径。
  - `data`：训练/验证/测试与实体 sidecar 路径。
  - `grpo`：GRPO 超参（目标、采样、奖励权重、实体奖励模式、LoRA、显存/精度、DeepSpeed 等）。
  - `sft`：SFT 超参。
- **`<experiment>.yaml`** —— 单个实验：`experiment_name`、`phase`、`mode`、`weights`（`bleu/chrf/key/ce`）、可选 `entity`（`entity_reward_mode` 等）、可选 `grpo_objective`。

**覆盖优先级（高 → 低）：命令行参数 > 实验 yaml > common.yaml > 脚本内默认。**

论文超参数配置 `stkg_stage2_full.yaml` 当前对齐论文设置：奖励权重 `BLEU:chrF:实体 = 0.3:0.5:0.2`、`entity_reward_mode: entity_substring`（最长公共子串软匹配）；采样 `G=4 / temperature=1.0 / top-k=50 / top-p=0.9`、`per_device_train_batch_size=3`、`kl_coef=0.02`、LoRA `r=16/α=32/dropout=0.05`。GRPO 目标不在该配置中固定，复现时请通过 `--grpo-objective group_relative_risk_kl` 或 `--grpo-objective clipped_grpo` 显式选择。

### 5.2 GRPO 目标函数

`--grpo-objective` 显式选择，两种可选：

- `group_relative_risk_kl`：组内对候选 log 概率以 `--group-policy-scale` 缩放后做 softmax 的期望风险，配合组内 z-score 优势与对参考策略的 KL 正则。
- `clipped_grpo`：PPO 式截断重要性比 + KL（由 `fca_grpo_clipped_runtime.py` 实现，`clipped_grpo_full.yaml` 使用与论文相同的奖励权重和实体奖励模式）。

### 5.3 实体奖励与 PER 开关

- `entity_reward_mode`：`none / key_recall / entity_em / entity_soft / entity_gemma_fuzzy / entity_substring`。论文主设置为 `entity_substring`（LCS 覆盖率）。
- **PER 默认纳入**实体奖励（覆盖七类）。如需排除人名（音译噪声大），训练脚本传 `--no-entity-include-per`，或在 `common.yaml` 设 `entity_include_per: false`，亦可在 `run_fca_grpo_experiments.py` 命令行传 `--no-entity-include-per`。

### 5.4 评测套件配置（`configs/eval_suite/`）

每个 yaml 含 `dataset`（数据与参考实体路径、格式）、`model`（`offline_predictions` 或具体模型适配）、`evaluation`（`mode`、`metrics`、`output_dir` 等）。

---

## 6. 运行流程

### 6.1 SFT

```bash
python scripts/train_gemma4_sft_qlora.py \
  --base-model-path $MODEL_ROOT/gemma-4-e4b-it \
  --train-data-path $DATA_ROOT/train.jsonl \
  --val-data-path $DATA_ROOT/dev.jsonl \
  --test-data-path $DATA_ROOT/test.jsonl \
  --output-root outputs/sft \
  --experiment-name gemma4_e4b_sft
```

### 6.2 GRPO（配置驱动）

```bash
# 单个实验
python scripts/run_fca_grpo_experiments.py run-one \
  --experiment stkg_stage2_full \
  --grpo-objective <group_relative_risk_kl|clipped_grpo> \
  --base-model-path $MODEL_ROOT/gemma-4-e4b-it \
  --sft-best-checkpoint outputs/sft/gemma4_e4b_sft/adapter_best

# 仅打印将执行的命令，不运行
python scripts/run_fca_grpo_experiments.py dry-run --experiment stkg_stage2_full \
  --grpo-objective <group_relative_risk_kl|clipped_grpo> \
  --sft-best-checkpoint <ckpt>

# 批量跑预设实验组
python scripts/run_fca_grpo_experiments.py run-all --sft-best-checkpoint <ckpt>
```

多卡：加 `--num-processes N --launcher torchrun`（或 `accelerate`），`--gpu-ids 0,1,...`。

### 6.3 评测

```bash
# 离线快速校验（无需模型/音频）
python scripts/run_eval_suite.py --config configs/eval_suite/offline_mini.yaml

# 套件评测（离线预测或模型生成）
python scripts/run_eval_suite.py --config configs/eval_suite/offline_predictions_example.yaml

# 单模型测试集评测
python scripts/eval_testt_bleu_chrf.py --base-model-path <base> --adapter-path <adapter> \
  --data-path <test.jsonl> --entity-path <test.ner.jsonl> --output-dir <out>
```

`run_fca_grpo_experiments.py` 在训练后会自动追加一次 `eval_testt_bleu_chrf.py` 评测。

---

## 7. 输出产物

### 7.1 训练（`<output_root>/<experiment_name>/`）

| 产物 | 说明 |
|---|---|
| `adapter_best/` | 验证最优的 LoRA 适配器 |
| `adapter_last/` | 末步适配器 |
| `run_config.json` | 解析后的运行配置 |
| `tensorboard/` | TensorBoard 日志（启用 `--enable-tensorboard` 时） |
| 严格断点目录 + `trainer_state.json` | 可续训检查点（`--keep-last-checkpoints` 控制保留数） |

经 `run_fca_grpo_experiments.py` 启动时，实验目录另含 `resolved_config.yaml`、`commands.txt`，以及评测子目录 `eval/`。

### 7.2 单模型评测（`eval_testt_bleu_chrf.py`，默认 `eval_outputs/<experiment_name>/`）

- `testt_predictions.jsonl` —— 逐句预测。
- `testt_metrics.json` —— 汇总指标。

### 7.3 评测套件（`run_eval_suite.py`，写入配置中的 `output_dir`）

- `metrics.summary.json` —— 汇总指标。
- `metrics.by_sample.jsonl` —— 逐样本指标。
- `predictions.jsonl` —— 预测文本。
- `reference_entities.jsonl` / `prediction_entities.jsonl` —— 参考/预测实体。
- `config.resolved.json` —— 解析后的评测配置。

> 这些产物（指标、预测、检查点、日志）均被 `.gitignore` 排除，应作为独立实验产物单独存放。

---

## 8. 评测指标

- **BLEU**：SacreBLEU，中文分词（`tok:zh`）。
- **chrF**：SacreBLEU chrF（字符 n-gram F 值）。
- **Entity-Recall（评测）**：`entity_key_recall`，**严格精确匹配**——金标实体须完整出现在译文中（同一归一化后逐字匹配）方计命中，覆盖七类实体。
- **实体软度量（训练奖励）**：`entity_substring` 用最长公共子串覆盖率给出按字符的连续得分，便于策略优化；评测阶段不使用软匹配，以免高估。

> 训练奖励的软 LCS 与评测的严格 key recall 是**两套不同度量**：前者用于稳定梯度，后者用于严谨报告。

---

## 9. 复现与发布

- 端到端复现步骤、超参与命令：[reproduction.md](reproduction.md)。
- 数据与实体 sidecar schema：[data.md](data.md)。
- 发布前检查项（无密钥/权重/大文件、占位 URL 替换、跑通测试与离线校验）：[release_checklist.md](release_checklist.md)。
