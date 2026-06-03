# RL Bonus Stability Changes - 2026-06-03

This note records the additional stability changes made after the first stabilized RL bonus runner. The purpose is to keep the report and later run interpretation tied to the actual code and runner settings.

## Context

The previous stability pass added observation normalization, best-eval checkpointing, conservative TD3 overrides, gradient clipping, and Algorithm Distillation checkpoint/source-policy fixes. A later sinusoidal positional-encoding rerun still showed a large gap between best return and final return, which suggests late TD3 policy deterioration rather than simple undertraining.

## New Source-Level Changes

1. Robust TD3 critic loss:
   - Added `agent.critic_loss=huber`.
   - Added `agent.huber_beta=1.0`.
   - Effect: critic updates become less sensitive to large temporal-difference outliers than plain MSE.

2. Target-Q clipping:
   - Added `agent.target_q_clip=5000.0`.
   - Effect: extreme bootstrapped target values are bounded before critic loss is computed. The value is intentionally loose enough not to cap normal Hopper returns but should suppress pathological critic spikes.

3. Decayed target policy smoothing:
   - Added `agent.target_noise_final=0.05`.
   - Added `agent.target_noise_decay_steps=300000`.
   - Added `agent.noise_clip_final=0.15`.
   - Added `agent.noise_clip_decay_steps=300000`.
   - Effect: TD3 target-action smoothing starts conservative but becomes less disruptive later in training, reducing late critic target noise after a useful policy has emerged.

4. Decayed exploration noise:
   - Added `agent.exploration_noise_final=0.01`.
   - Added `agent.exploration_noise_decay_steps=300000`.
   - Effect: actor exploration remains nonzero after warmup but no longer injects the same amount of action noise throughout the full one-million-step run.

5. Slower actor updates in the bonus runner:
   - Added `agent.policy_delay=3` to the RL bonus commands.
   - Effect: critics get more updates between actor updates, which can reduce actor movement driven by temporarily inaccurate Q estimates.

6. Additional diagnostics:
   - TD3 now reports target-noise schedule values, target-Q magnitude/clipping fraction, critic Q means, Q disagreement, and gradient norms.
   - The training loop logs the current exploration noise to W&B and writes the final exploration noise into the generated experiment docs.

## Runner-Level Changes

The stabilized RL bonus runner now applies the new controls to all TD3 bonus commands:

- `rl_bonus_pos_learned_stable_seed*`
- `rl_bonus_pos_sinusoidal_stable_seed*`
- `rl_bonus_pos_rope_stable_seed*`
- `rl_bonus_combined_L32_stable_seed*`
- `rl_bonus_xlstm_hidden_stable_seed*`
- `rl_bonus_xlstm_delayed_stable_seed*`

Algorithm Distillation is unchanged in this pass because the previous pass already added the relevant AD fixes: source-policy observation-normalizer restoration and best-loss checkpoint selection.

## Expected Effect on W&B Curves

These changes are expected to reduce, not eliminate, TD3 instability. A high-quality stable run should show:

- smaller `eval/final_minus_best` magnitude,
- fewer late collapses after an early peak,
- smoother or lower `critic*_loss`,
- bounded `target_q_abs_max`,
- low or zero `target_q_clipped_frac` during normal training,
- decreasing `exploration/noise_std`,
- decreasing `target_noise_std` and `target_noise_clip`.

If a run still has high best return but poor final return, it should still be reported as instability/late deterioration. The correct report treatment remains to compare best-checkpoint returns and final-checkpoint returns separately.

## Interpretation Caveat

These changes make the bonus reruns more defensible but do not guarantee monotonic policy improvement. TD3 with partial observability, delayed rewards, noisy observations, and sequence actors can still show seed variance and critic-driven collapse. For the report, the strongest evidence remains the combination of best returns, final returns, best-step timing, return standard deviation, and the new critic/noise diagnostics.
