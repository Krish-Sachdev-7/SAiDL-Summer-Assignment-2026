"""Core ML training entry point."""
import math
import time
import random
import atexit
import signal
import re
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from model import DecoderLM
    from data import build_dataloaders
    from evaluate import measure_peak_memory, measure_throughput, extrapolation_test
else:
    from .model import DecoderLM
    from .data import build_dataloaders
    from .evaluate import measure_peak_memory, measure_throughput, extrapolation_test

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from shared.doc_pipeline import update_experiment_docs


def set_seed(seed: int) -> None:
    """Seed the usual RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_rng_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [_as_cpu_rng_tensor(s) for s in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states[: torch.cuda.device_count()])


def _as_cpu_rng_tensor(value) -> torch.ByteTensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8).cpu()


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("._-")
    return text or "run"


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Warmup plus cosine LR."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, max_batches: int):
    """Evaluate loss and perplexity."""
    model.eval()
    losses = []
    it = iter(loader)
    with torch.no_grad():
        for _ in range(max_batches):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(float(loss.item()))

    if not losses:
        return float("nan"), float("nan")

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(min(20.0, mean_loss))
    return mean_loss, ppl


def maybe_init_wandb(cfg):
    """Start W&B when enabled."""
    if not bool(cfg.logging.wandb.enable):
        return None

    import wandb

    return wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        group=cfg.logging.wandb.group,
        tags=list(cfg.logging.wandb.tags),
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _write_local_run_summary(cfg, summary: dict) -> Path:
    out_dir = Path.cwd()
    report_dir = out_dir / "run_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    md_path = report_dir / "summary.md"
    latex_path = report_dir / "summary.tex"

    md = [
        "# Core ML Run Summary",
        "",
        f"- experiment_name: {cfg.experiment.name}",
        f"- attention_type: {cfg.model.attention.type}",
        f"- positional_encoding: {cfg.pos_encoding.type}",
        f"- context_length: {cfg.data.context_length}",
        f"- max_steps: {cfg.trainer.max_steps}",
        f"- final_train_loss: {summary['train_loss']:.6f}",
        f"- final_val_loss: {summary['val_loss']:.6f}",
        f"- final_val_perplexity: {summary['val_ppl']:.6f}",
        f"- throughput_tokens_per_sec: {summary['throughput']:.2f}",
        f"- inference_tokens_per_sec: {summary['inference_throughput']:.2f}",
        f"- peak_gpu_memory_gb: {summary['peak_mem_gb']:.4f}",
        f"- virtual_epoch: {summary['virtual_epoch']:.4f}",
        f"- sec_per_virtual_epoch: {summary['sec_per_virtual_epoch']:.2f}",
        f"- grad_norm: {summary['grad_norm']:.6f}",
        f"- train_loss_is_finite: {summary['train_loss_is_finite']}",
        f"- val_loss_is_finite: {summary['val_loss_is_finite']}",
        f"- wall_time_sec: {summary['wall_time_sec']:.2f}",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        "% Auto-generated by core_ml/src/train.py",
        "\\subsection{Run: " + str(cfg.experiment.name).replace("_", "\\_") + "}",
        "\\textbf{Configuration.} "
        + f"Attention={cfg.model.attention.type}, "
        + f"PE={cfg.pos_encoding.type}, "
        + f"Context={cfg.data.context_length}, "
        + f"Steps={cfg.trainer.max_steps}.",
        "",
        "\\textbf{Results.} "
        + f"Train loss={summary['train_loss']:.4f}, "
        + f"Validation loss={summary['val_loss']:.4f}, "
        + f"Validation perplexity={summary['val_ppl']:.2f}, "
        + f"Throughput={summary['throughput']:.1f} tokens/s, "
        + f"Inference throughput={summary['inference_throughput']:.1f} tokens/s, "
        + f"Peak GPU memory={summary['peak_mem_gb']:.2f} GB.",
    ]
    latex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    return md_path


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training loop."""
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg))

    run = maybe_init_wandb(cfg)
    train_loader, val_loader = build_dataloaders(cfg)

    model = DecoderLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )

    use_amp = bool(cfg.trainer.use_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    last_loss = float("nan")
    last_val_loss = float("nan")
    last_val_ppl = float("nan")
    last_throughput = 0.0
    last_inference_throughput = 0.0
    last_peak_gb = 0.0
    last_grad_norm = float("nan")
    last_virtual_epoch = 0.0
    last_sec_per_virtual_epoch = 0.0
    best_val_loss = float("inf")
    best_val_ppl = float("nan")
    best_step = 0
    checkpoint_saved = False
    checkpoint_dir = Path("checkpoints") / _slugify(str(cfg.experiment.name))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = checkpoint_dir / "ckpt_latest.pt"
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    final_ckpt_path = checkpoint_dir / "final_model.pt"

    def save_checkpoint(path: Path) -> None:
        nonlocal checkpoint_saved
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment_name": str(cfg.experiment.name),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "rng_state": _capture_rng_state(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "final_train_loss": last_loss,
                "final_val_loss": last_val_loss,
                "final_val_ppl": last_val_ppl,
                "throughput": last_throughput,
                "inference_throughput": last_inference_throughput,
                "grad_norm": last_grad_norm,
                "best_val_loss": best_val_loss,
                "best_val_ppl": best_val_ppl,
                "best_step": best_step,
                "wall_time_sec": time.time() - start,
                "peak_mem_gb": measure_peak_memory(),
            },
            path,
        )
        checkpoint_saved = True

    resume_path = str(
        OmegaConf.select(
            cfg,
            "resume",
            default=OmegaConf.select(cfg, "experiment.resume", default=""),
        )
        or ""
    )
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if "scaler" in ck and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        step = int(ck.get("step", 0))
        best_val_loss = float(ck.get("best_val_loss", ck.get("final_val_loss", best_val_loss)))
        best_val_ppl = float(ck.get("best_val_ppl", ck.get("final_val_ppl", best_val_ppl)))
        best_step = int(ck.get("best_step", step if math.isfinite(best_val_loss) else 0))
        if not math.isfinite(best_val_loss):
            best_val_loss = float("inf")
            best_val_ppl = float("nan")
            best_step = 0
        _restore_rng_state(ck.get("rng_state"))
        print(f"Resumed from {resume_path} at step {step}")

    model.train()
    train_iter = iter(train_loader)
    start = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def _save_on_exit(*_args):
        save_checkpoint(latest_ckpt_path)
        save_checkpoint(Path("ckpt_latest.pt"))

    atexit.register(_save_on_exit)

    def _handle_signal(signum, frame):
        save_checkpoint(latest_ckpt_path)
        save_checkpoint(Path("ckpt_latest.pt"))
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    while step < int(cfg.trainer.max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        lr = cosine_lr(
            step=step,
            max_steps=int(cfg.trainer.max_steps),
            base_lr=float(cfg.trainer.learning_rate),
            warmup_steps=int(cfg.trainer.warmup_steps),
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if not torch.isfinite(loss):
            last_loss = float(loss.detach().item())
            save_checkpoint(latest_ckpt_path)
            save_checkpoint(Path("ckpt_latest.pt"))
            raise RuntimeError(f"Non-finite training loss at step {step}: {last_loss}")

        scaler.scale(loss).backward()
        last_grad_norm = float("nan")
        if float(cfg.trainer.grad_clip) > 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float(cfg.trainer.grad_clip))
            last_grad_norm = float(grad_norm.detach().cpu().item())
        scaler.step(optimizer)
        scaler.update()

        last_loss = float(loss.item())
        step += 1

        ckpt_interval = int(getattr(cfg.trainer, "checkpoint_interval", 0))
        if ckpt_interval > 0 and step % ckpt_interval == 0:
            save_checkpoint(latest_ckpt_path)
            save_checkpoint(Path("ckpt_latest.pt"))

        if step % int(cfg.trainer.eval_interval) == 0 or step == 1:
            elapsed = time.time() - start
            tokens_seen = step * int(cfg.trainer.batch_size) * int(cfg.data.context_length)
            throughput = tokens_seen / max(elapsed, 1e-6)
            steps_per_virtual_epoch = max(1, len(train_loader))
            virtual_epoch = step / steps_per_virtual_epoch
            sec_per_virtual_epoch = elapsed / max(virtual_epoch, 1e-6)

            val_loss, val_ppl = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=int(cfg.trainer.eval_batches),
            )
            inference_throughput = measure_throughput(
                model=model,
                context_length=int(cfg.data.context_length),
                batch_size=int(cfg.trainer.batch_size),
                device=device,
                n_batches=int(getattr(cfg.trainer, "inference_benchmark_batches", 5)),
                use_amp=use_amp,
            )

            peak_gb = measure_peak_memory()
            last_peak_gb = float(peak_gb)
            print(
                f"step={step} train_loss={last_loss:.4f} val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.2f} tok/s={throughput:.1f} "
                f"infer_tok/s={inference_throughput:.1f} peak_mem_gb={peak_gb:.2f}"
            )

            last_val_loss = float(val_loss)
            last_val_ppl = float(val_ppl)
            last_throughput = float(throughput)
            last_inference_throughput = float(inference_throughput)
            last_virtual_epoch = float(virtual_epoch)
            last_sec_per_virtual_epoch = float(sec_per_virtual_epoch)

            if math.isfinite(last_val_loss) and last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_val_ppl = last_val_ppl
                best_step = step
                save_checkpoint(best_ckpt_path)
                save_checkpoint(Path("best_model.pt"))
                print(
                    f"Saved new best checkpoint at step={best_step} "
                    f"val_loss={best_val_loss:.4f} val_ppl={best_val_ppl:.2f}"
                )

            if run is not None:
                import wandb

                wandb.log(
                    {
                        "step": step,
                        "train/loss": last_loss,
                        "val/loss": last_val_loss,
                        "val/perplexity": last_val_ppl,
                        "val/best_loss": best_val_loss,
                        "val/best_perplexity": best_val_ppl,
                        "val/best_step": best_step,
                        "system/tokens_per_sec": last_throughput,
                        "system/inference_tokens_per_sec": last_inference_throughput,
                        "system/peak_gpu_mem_gb": float(peak_gb),
                        "system/virtual_epoch": last_virtual_epoch,
                        "system/sec_per_virtual_epoch": last_sec_per_virtual_epoch,
                        "stability/grad_norm": last_grad_norm,
                        "stability/train_loss_is_finite": float(math.isfinite(last_loss)),
                        "stability/val_loss_is_finite": float(math.isfinite(last_val_loss)),
                        "optim/lr": float(lr),
                    }
                )
            model.train()

    wall_time_sec = time.time() - start
    ckpt_path = final_ckpt_path
    save_checkpoint(ckpt_path)
    save_checkpoint(Path("final_model.pt"))
    print(f"Saved checkpoint to {ckpt_path}")

    eval_lengths = list(OmegaConf.select(cfg, "data.eval_context_lengths", default=[]) or [])
    if eval_lengths:
        print("Running extrapolation test...")
        extrap_results = extrapolation_test(model, cfg, eval_lengths, device)
        extrap_losses = {}
        for length, ppl in extrap_results.items():
            extrap_loss = float(min(20.0, math.log(max(float(ppl), 1e-12))))
            extrap_losses[int(length)] = extrap_loss
            print(f"  extrap ctx={length}: ppl={ppl:.2f} loss={extrap_loss:.4f}")
        if run is not None:
            import wandb

            wandb.log(
                {
                    **{f"extrap/ppl_L{L}": ppl for L, ppl in extrap_results.items()},
                    **{f"extrap/loss_L{L}": loss for L, loss in extrap_losses.items()},
                }
            )
    else:
        extrap_results = {}
        extrap_losses = {}

    summary = {
        "train_loss": last_loss,
        "val_loss": last_val_loss,
        "val_ppl": last_val_ppl,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else float("nan"),
        "best_val_ppl": best_val_ppl,
        "best_step": best_step,
        "throughput": last_throughput,
        "inference_throughput": last_inference_throughput,
        "peak_mem_gb": measure_peak_memory(),
        "virtual_epoch": last_virtual_epoch,
        "sec_per_virtual_epoch": last_sec_per_virtual_epoch,
        "grad_norm": last_grad_norm,
        "train_loss_is_finite": math.isfinite(last_loss),
        "val_loss_is_finite": math.isfinite(last_val_loss),
        "wall_time_sec": wall_time_sec,
        "extrap": {
            "ppl": extrap_results,
            "loss": extrap_losses,
        },
    }
    summary_path = _write_local_run_summary(cfg, summary)
    print(f"Saved local run summary to {summary_path}")

    doc_metrics = {
        "final_train_loss": round(last_loss, 6),
        "final_val_loss": round(last_val_loss, 6),
        "final_val_perplexity": round(last_val_ppl, 6),
        "best_val_loss": round(best_val_loss, 6) if math.isfinite(best_val_loss) else float("nan"),
        "best_val_perplexity": round(best_val_ppl, 6) if math.isfinite(best_val_ppl) else best_val_ppl,
        "best_step": best_step,
        "tokens_per_sec": round(last_throughput, 4),
        "inference_tokens_per_sec": round(last_inference_throughput, 4),
        "peak_gpu_memory_gb": round(measure_peak_memory(), 6),
        "virtual_epoch": round(last_virtual_epoch, 6),
        "sec_per_virtual_epoch": round(last_sec_per_virtual_epoch, 4),
        "grad_norm": round(last_grad_norm, 6) if math.isfinite(last_grad_norm) else last_grad_norm,
        "train_loss_is_finite": math.isfinite(last_loss),
        "val_loss_is_finite": math.isfinite(last_val_loss),
        "wall_time_sec": round(wall_time_sec, 4),
    }
    for L, ppl in extrap_results.items():
        doc_metrics[f"extrap_ppl_L{L}"] = round(float(ppl), 6)
    for L, loss in extrap_losses.items():
        doc_metrics[f"extrap_loss_L{L}"] = round(float(loss), 6)

    artifacts = update_experiment_docs(
        track="core_ml",
        experiment_name=str(cfg.experiment.name),
        cfg_dict=OmegaConf.to_container(cfg, resolve=True),
        metrics=doc_metrics,
        run_output_dir=str(Path.cwd()),
        repo_root=REPO_ROOT,
    )
    print(f"Updated docs: {artifacts.markdown_path}")
    print(f"Updated docs: {artifacts.include_index_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
