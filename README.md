# EARL：吉尔吉斯语—中文语音翻译

本仓库是 **EARL: Entity-Aware Reinforcement Learning for Low-Resource Kyrgyz-Chinese Speech Translation**（面向低资源吉尔吉斯语—中文语音翻译的实体感知强化学习）的代码发布版。

EARL 是一个两阶段训练框架：

1. 监督微调（SFT）建立吉尔吉斯语→中文语音翻译的基础策略。
2. 群体相对策略优化（GRPO）在 SFT 适配器基础上继续训练，优化由 BLEU、chrF 与参考侧实体召回构成的句级奖励。
   公开的 GRPO 入口提供两个可选目标：`group_relative_risk_kl` 与 `clipped_grpo`。二者均为带 KL 正则的 group-relative 目标，通过 `--grpo-objective` 显式选择，并可使用同一套论文超参数。

本仓库刻意只包含源码。大型模型检查点、完整训练数据、音频文件、API 密钥、缓存以及完整预测结果均未包含在内。

## 仓库结构

```text
configs/
  eval_suite/       评测示例
  fca_grpo/         GRPO 实验配置与目标变体
docs/               数据、复现与发布说明
examples/           极小的离线评测样例
scripts/            训练、GRPO 目标与核心评测代码
  baselines/        第三方基线（级联/端到端，best-effort，仅用于论文对比）
tests/              最小化的指标测试
```

## 命名说明

部分目录与配置名沿用了项目内部代号：

- `fca_grpo` / `stkg_*` —— 指标对齐的 GRPO 阶段（即 EARL 的第二训练阶段）及其实验配置。
- `testt`（例如 `converted_testt_format`、`testt.jsonl`）—— 吉尔吉斯语→中文语音翻译评测集及其 JSONL 格式的内部名称。

对应到论文术语：EARL = SFT（第一阶段）+ 指标对齐 GRPO（第二阶段）。

## 安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

如需 GPU 训练，请先安装与你的 CUDA 版本匹配的 PyTorch，再安装其余依赖。

## 测试

运行单元测试需要开发依赖（`pip install -r requirements-dev.txt`）：

```bash
pytest
```

## 离线快速校验

下面的命令运行一个极小的离线打分示例，不会加载任何语音模型或音频文件。

```bash
python scripts/run_eval_suite.py --config configs/eval_suite/offline_mini.yaml
```

预期输出写入：

```text
outputs/offline_mini/
```

## 主要训练入口

SFT：

```bash
python scripts/train_gemma4_sft_qlora.py \
  --base-model-path $MODEL_ROOT/gemma-4-e4b-it \
  --train-data-path $DATA_ROOT/train.jsonl \
  --val-data-path $DATA_ROOT/dev.jsonl \
  --test-data-path $DATA_ROOT/test.jsonl \
  --output-root outputs/sft \
  --experiment-name gemma4_e4b_sft
```

GRPO：

```bash
python scripts/run_fca_grpo_experiments.py run-one \
  --experiment stkg_stage2_full \
  --grpo-objective <group_relative_risk_kl|clipped_grpo> \
  --base-model-path $MODEL_ROOT/gemma-4-e4b-it \
  --sft-best-checkpoint outputs/sft/gemma4_e4b_sft/adapter_best
```

`stkg_stage2_full` 保存论文使用的奖励与采样超参数；GRPO 目标不在文档中指定为主目标，复现时请在 `group_relative_risk_kl` 与 `clipped_grpo` 中按实验需要显式选择。
`clipped_grpo_full` 提供同一套论文超参数下的 clipped-ratio 目标配置。

统一评测：

```bash
python scripts/run_eval_suite.py --config configs/eval_suite/offline_predictions_example.yaml
```

论文复现流程见 [docs/reproduction.md](docs/reproduction.md)。

## 文档

- [技术指南](docs/technical_guide.md)：数据位置、代码导览、配置体系、运行流程与输出产物。
- [数据格式](docs/data.md)：JSONL 与实体 sidecar 的 schema。
- [复现说明](docs/reproduction.md)：端到端复现流程。
- [发布检查清单](docs/release_checklist.md)：发布前检查项。

## 结果

论文结果的 JSON 文件未随本源码仓库一起打包。仓库保留了复现或重新打分所需的流程、配置模板、指标实现与最小示例。完整的指标汇总、预测结果与检查点请作为独立的实验产物单独存放。

## 数据

完整训练语料不在此仓库中再分发。请按 [docs/data.md](docs/data.md) 中描述的 `converted_translation_jsonl` 格式准备数据。参考侧实体 sidecar 使用包含 `id` 与 `entities` 的 JSONL 行。

## 安全

请勿提交 API 密钥或本地凭证。基于 API 的基线脚本从环境变量读取凭证。详见 [SECURITY.md](SECURITY.md)。

## 引用

见 [CITATION.cff](CITATION.cff)。

