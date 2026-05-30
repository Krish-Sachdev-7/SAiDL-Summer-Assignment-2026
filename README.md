# SAiDL Summer Assignment 2026

This repository contains my Core ML and Reinforcement Learning submissions for the SAiDL Summer Assignment 2026. The code is organized around Hydra configurations, W&B experiment logging, LaTeX reports, and CSV exports used to reproduce the report tables.

## Repository Layout

```text
core_ml/
  conf/                 Hydra configs for language-model experiments
  src/                  Transformer, attention, positional encoding, training, and evaluation code
  report/               Core ML LaTeX report, compiled PDF, figures, and CSV tables

rl/
  conf/                 Hydra configs for TD3, policy backbones, environments, and rewards
  src/                  TD3, replay buffers, Hopper wrappers, policy networks, and reward model code
  report/               RL LaTeX report, compiled PDF, figures, and CSV tables

shared/                 Shared utility code
```

## Environment

The experiments were run in Kaggle GPU notebooks. A local or notebook environment should include PyTorch, Hydra, W&B, Gymnasium MuJoCo, NumPy, Pandas, Matplotlib, Hugging Face datasets/tokenizers, and tqdm. The convenience dependency list is in `requirements.txt`.

For W&B logging, set credentials through the environment rather than hard-coding them:

```bash
export WANDB_API_KEY=...
```

## Core ML

The Core ML task studies long-context language modeling on WikiText-2. It includes a baseline decoder-only Transformer, sliding-window attention, sparse-block attention, GQA, RoPE, ALiBi, relative positional encoding, and convolution-attention hybrids.

Example command from `core_ml/`:

```bash
python src/train.py experiment.name=coreml_p1_baseline_full_abs_ctx1024 model.attention.type=full data.context_length=1024
```

The final Core ML report is available at:

```text
core_ml/report/core_ml_report.pdf
```

The associated report data is stored under:

```text
core_ml/report/data/
core_ml/report/tables/
core_ml/report/figures/
```

The full scalar W&B history exports for the selected Core ML report runs are stored in:

```text
core_ml/report/data/histories/
```

## Reinforcement Learning

The RL task studies TD3 on Hopper-v5 with an MLP actor and a causal Transformer actor over recent observation-action history. It includes the required baseline seeds, context-length sweep, hidden-velocity, observation-noise, delayed-reward, learned-reward, and attention-support experiments.

Example commands from `rl/`:

```bash
python src/train.py experiment.name=rl_p1_mlp_seed42 agent=td3_mlp env=hopper_full seed=42 total_steps=1000000
python src/train.py experiment.name=rl_p1_tr_L8_seed42 agent=td3_transformer env=hopper_full agent.actor.context_length=8 seed=42 total_steps=1000000
```

The final RL report is available at:

```text
rl/report/rl_report.pdf
```

The associated report data is stored under:

```text
rl/report/data/
rl/report/tables/
rl/report/figures/
```

The full scalar W&B history exports for the selected RL report runs are stored in:

```text
rl/report/data/histories/
```

The optional checkpoint-based attention export utility is:

```text
rl/report/export_attention_from_checkpoints.py
```

## Reproducibility Notes

Reports use final-tagged, successfully finished W&B runs and include CSV exports for the selected run sets, coverage audits, and derived tables. Failed or crashed runs are documented in the RL report only when a later successful run superseded them. Hardware labels are retained for RL because some runs moved from P100 to T4 during the final execution window; returns remain comparable by configuration, but wall-clock runtime should not be read as a controlled hardware benchmark.

The rerun command manifest is stored in:

```text
reproducibility/COMMAND_MANIFEST.md
reproducibility/command_manifest.csv
reproducibility/wandb_history_manifest.csv
```

To rebuild the reports with Tectonic or another LaTeX engine:

```bash
cd core_ml/report
tectonic core_ml_report.tex

cd ../../rl/report
tectonic rl_report.tex
```
