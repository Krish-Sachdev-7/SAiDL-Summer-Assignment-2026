# Command Manifest

This manifest records the commands needed to rerun the selected final Core ML and RL experiments used in the reports. Commands are also available as CSV in `reproducibility/command_manifest.csv`.

Run each command from the `workdir` shown in the CSV. W&B logging requires `WANDB_API_KEY` to be set in the environment.

## Core ML

### coreml_p1_baseline_full_abs_ctx1024

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/gyealu4a

```bash
cd core_ml
python src/train.py experiment.name=coreml_p1_baseline_full_abs_ctx1024 model=base pos_encoding=absolute conv=none data.context_length=1024 trainer.batch_size=4 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_sliding_window_ctx512

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/s7jtrjqq

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_sliding_window_ctx512 model=sliding_window pos_encoding=absolute conv=none data.context_length=512 trainer.batch_size=8 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_sliding_window_ctx1024

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/jyniv41v

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_sliding_window_ctx1024 model=sliding_window pos_encoding=absolute conv=none data.context_length=1024 trainer.batch_size=4 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_sliding_window_ctx2048

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/30zfwy42

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_sliding_window_ctx2048 model=sliding_window pos_encoding=absolute conv=none data.context_length=2048 trainer.batch_size=2 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_sliding_window_ctx4096

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/pnuksk52

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_sliding_window_ctx4096 model=sliding_window pos_encoding=absolute conv=none data.context_length=4096 trainer.batch_size=1 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_sparse_block_ctx512

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/c27xrb4g

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_sparse_block_ctx512 model=sparse_block pos_encoding=absolute conv=none data.context_length=512 trainer.batch_size=8 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_sparse_block_ctx1024

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/0gmcbzac

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_sparse_block_ctx1024 model=sparse_block pos_encoding=absolute conv=none data.context_length=1024 trainer.batch_size=4 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_sparse_block_ctx2048

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/0zx17pfo

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_sparse_block_ctx2048 model=sparse_block pos_encoding=absolute conv=none data.context_length=2048 trainer.batch_size=2 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_sparse_block_ctx4096

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/rf7qslez

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_sparse_block_ctx4096 model=sparse_block pos_encoding=absolute conv=none data.context_length=4096 trainer.batch_size=1 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_gqa_ctx512

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/vqwj1dkl

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_gqa_ctx512 model=gqa pos_encoding=absolute conv=none data.context_length=512 trainer.batch_size=8 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_gqa_ctx1024

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/mdbgwpzd

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_gqa_ctx1024 model=gqa pos_encoding=absolute conv=none data.context_length=1024 trainer.batch_size=4 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_gqa_ctx2048

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/2k9836jo

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_gqa_ctx2048 model=gqa pos_encoding=absolute conv=none data.context_length=2048 trainer.batch_size=2 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p2_gqa_ctx4096

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/0i87yv2e

```bash
cd core_ml
python src/train.py experiment.name=coreml_p2_gqa_ctx4096 model=gqa pos_encoding=absolute conv=none data.context_length=4096 trainer.batch_size=1 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p3_rope_train512

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/uz6g881m

```bash
cd core_ml
python src/train.py experiment.name=coreml_p3_rope_train512 model=base pos_encoding=rope conv=none data.context_length=512 trainer.batch_size=8 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p3_alibi_train512

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/38i0pnm1

```bash
cd core_ml
python src/train.py experiment.name=coreml_p3_alibi_train512 model=base pos_encoding=alibi conv=none data.context_length=512 trainer.batch_size=8 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p3_relative_train512

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/3b7o8z8s

```bash
cd core_ml
python src/train.py experiment.name=coreml_p3_relative_train512 model=base pos_encoding=relative conv=none data.context_length=512 trainer.batch_size=8 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p4_pre_attention_gqa_rope_ctx1024

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/e0lzy1ft

```bash
cd core_ml
python src/train.py experiment.name=coreml_p4_pre_attention_gqa_rope_ctx1024 model=gqa pos_encoding=rope conv=pre_attention data.context_length=1024 trainer.batch_size=4 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

### coreml_p4_gated_ffn_gqa_rope_ctx1024

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-core-ml/runs/h03r0qck

```bash
cd core_ml
python src/train.py experiment.name=coreml_p4_gated_ffn_gqa_rope_ctx1024 model=gqa pos_encoding=rope conv=gated_ffn data.context_length=1024 trainer.batch_size=4 trainer.max_steps=12000 trainer.eval_interval=500 trainer.eval_batches=40 trainer.checkpoint_interval=2500 seed=42 data.eval_context_lengths=[512,1024,2048] logging.wandb.tags=[final]
```

## Reinforcement Learning

### rl_p1_mlp_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/orduqlzt

```bash
cd rl
python src/train.py experiment.name=rl_p1_mlp_seed42 agent=td3_mlp env=hopper_full seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p1_mlp_seed43

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/12tb4cex

```bash
cd rl
python src/train.py experiment.name=rl_p1_mlp_seed43 agent=td3_mlp env=hopper_full seed=43 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p1_mlp_seed44

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/xe1bvoqz

```bash
cd rl
python src/train.py experiment.name=rl_p1_mlp_seed44 agent=td3_mlp env=hopper_full seed=44 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p1_tr_L4_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/k4mmxrji

```bash
cd rl
python src/train.py experiment.name=rl_p1_tr_L4_seed42 agent=td3_transformer env=hopper_full seed=42 agent.actor.context_length=4 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p1_tr_L8_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/jcs1nkci

```bash
cd rl
python src/train.py experiment.name=rl_p1_tr_L8_seed42 agent=td3_transformer env=hopper_full seed=42 agent.actor.context_length=8 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p1_tr_L16_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/d6s0vtx6

```bash
cd rl
python src/train.py experiment.name=rl_p1_tr_L16_seed42 agent=td3_transformer env=hopper_full seed=42 agent.actor.context_length=16 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000 +resume=/kaggle/input/datasets/krish247/check-1/SAiDL_Assignment/rl/ckpt_latest_rl.pt
```

### rl_p1_tr_L32_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/tc91exur

```bash
cd rl
python src/train.py experiment.name=rl_p1_tr_L32_seed42 agent=td3_transformer env=hopper_full seed=42 agent.actor.context_length=32 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000 +resume=/kaggle/input/datasets/krish247/check-1/SAiDL_Assignment/rl/ckpt_latest_rl.pt
```

### rl_p2_hidden_vel_mlp_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/96m7x5eb

```bash
cd rl
python src/train.py experiment.name=rl_p2_hidden_vel_mlp_seed42 agent=td3_mlp env=hopper_hidden_vel seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000 +resume=/kaggle/input/datasets/kkk7777/rl-output/SAiDL_Assignment/rl/ckpt_latest_rl.pt
```

### rl_p2_hidden_vel_tr_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/4hl8amlc

```bash
cd rl
python src/train.py experiment.name=rl_p2_hidden_vel_tr_seed42 agent=td3_transformer env=hopper_hidden_vel seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p2_noise01_mlp_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/lgmxbqai

```bash
cd rl
python src/train.py experiment.name=rl_p2_noise01_mlp_seed42 agent=td3_mlp env=hopper_noisy env.noise_sigma=0.1 seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p2_noise01_tr_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/3lt7il9g

```bash
cd rl
python src/train.py experiment.name=rl_p2_noise01_tr_seed42 agent=td3_transformer env=hopper_noisy env.noise_sigma=0.1 seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000 +resume=/kaggle/input/datasets/kkk7777/rl-output/SAiDL_Assignment/rl/ckpt_latest_rl.pt
```

### rl_p2_noise03_mlp_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/anlzqstp

```bash
cd rl
python src/train.py experiment.name=rl_p2_noise03_mlp_seed42 agent=td3_mlp env=hopper_noisy env.noise_sigma=0.3 seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p2_noise03_tr_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/f4lc5xjg

```bash
cd rl
python src/train.py experiment.name=rl_p2_noise03_tr_seed42 agent=td3_transformer env=hopper_noisy env.noise_sigma=0.3 seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000 +resume=/kaggle/input/datasets/kkk7777/rl-output/SAiDL_Assignment/rl/ckpt_latest_rl.pt
```

### rl_p2_delay10_mlp_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/523e0uha

```bash
cd rl
python src/train.py experiment.name=rl_p2_delay10_mlp_seed42 agent=td3_mlp env=hopper_delayed env.delay_k=10 seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p2_delay10_tr_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/hfiawzrg

```bash
cd rl
python src/train.py experiment.name=rl_p2_delay10_tr_seed42 agent=td3_transformer env=hopper_delayed env.delay_k=10 seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p2_delay30_mlp_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/15bzu83r

```bash
cd rl
python src/train.py experiment.name=rl_p2_delay30_mlp_seed42 agent=td3_mlp env=hopper_delayed env.delay_k=30 seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```

### rl_p2_delay30_tr_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/kkdfxotj

```bash
cd rl
python src/train.py experiment.name=rl_p2_delay30_tr_seed42 agent=td3_transformer env=hopper_delayed env.delay_k=30 seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000 +resume=/kaggle/input/datasets/kacsav/rl-output/SAiDL_Assignment/rl/ckpt_latest_rl.pt
```

### rl_p2_rlhf_tr_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/rvckskwt

```bash
cd rl
python src/train.py experiment.name=rl_p2_rlhf_tr_seed42 agent=td3_transformer env=hopper_full seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] agent.actor.context_length=8
```

### rl_p3_attn_support_full_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/78e2wcm0

```bash
cd rl
python src/train.py experiment.name=rl_p3_attn_support_full_seed42 agent=td3_transformer env=hopper_full seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000 +resume=/kaggle/input/datasets/kacsav/rl-output/SAiDL_Assignment/rl/ckpt_latest_rl.pt
```

### rl_p3_attn_support_hidden_seed42

W&B: https://wandb.ai/krishsachdev246-bits-pilani/saidl-rl/runs/lwsokb8n

```bash
cd rl
python src/train.py experiment.name=rl_p3_attn_support_hidden_seed42 agent=td3_transformer env=hopper_hidden_vel seed=42 total_steps=1000000 eval_interval=10000 eval_episodes=10 logging.wandb.tags=[final] checkpoint_interval=10000
```
