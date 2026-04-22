"""Hyperparameter Optimization sweep for JEPA v3 using Optuna.

Runs shortened training trials (30 epochs) with Bayesian optimization (TPE)
and pruning (MedianPruner) to efficiently find the best loss weights and LR.

Design decisions:
    - Architecture is FIXED (embed_dim=384, depth=6, predictor_depth=3) for fair
      comparison with v1/v2/VideoMAE. We only tune loss weights and optimizer settings.
    - 30-epoch trials: our v1/v2 data shows clear trends by epoch 30 (kNN peaks
      at epoch 30, divergence is visible by epoch 10).
    - MedianPruner kills trials worse than the running median after epoch 10,
      saving ~50% of GPU time on bad hyperparameter combos.
    - Results stored in SQLite so the study can be resumed across SLURM jobs.

Search space (4 hyperparameters):
    var_weight:    [0.25, 3.0]   log-uniform  (v1=1.0, v2=5.0 was too high)
    cov_weight:    [0.001, 0.05] log-uniform  (v2=0.04, v3=0.01, maybe even lower)
    lr:            [5e-5, 5e-4]  log-uniform  (v1/v2 used 1.5e-4)
    weight_decay:  [0.01, 0.1]   log-uniform  (v1/v2 used 0.05)

Usage:
    # Single GPU (runs all trials sequentially)
    python scripts/hpo_sweep.py --config configs/jepa_v3.yaml --n_trials 20

    # Resume a previous study
    python scripts/hpo_sweep.py --config configs/jepa_v3.yaml --n_trials 10 --study_name jepa_v3_hpo
"""

import os
import sys
import math
import signal
import argparse
import yaml
import copy
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    print("ERROR: Optuna not installed. Install with: pip install optuna")
    print("  In your HPC env: pip install optuna")
    sys.exit(1)

from data.dataset import ActiveMatterDataset
from models.encoder import SpatioTemporalViT, count_parameters
from models.jepa_v3 import VideoJEPAv3
from utils.training import set_seed, log_gpu_memory


# ---------------------------------------------------------------------------
# Graceful shutdown for SLURM preemption
# ---------------------------------------------------------------------------

class GracefulKiller:
    """Catches SIGTERM/SIGUSR1 from SLURM to stop the study between trials.

    SLURM sends SIGTERM ~30s before killing a job (preemption or walltime).
    Some clusters send SIGUSR1 even earlier as a warning.
    When caught, we set a flag so the current trial finishes its epoch,
    then the study stops cleanly — all completed trials stay in SQLite.
    """
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGTERM, self._handler)
        # SIGUSR1 is not available on Windows, but HPC is Linux
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, self._handler)

    def _handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n{'!'*60}")
        print(f"RECEIVED {sig_name} — stopping after current trial completes")
        print(f"Completed trials are safe in SQLite. Resubmit to resume.")
        print(f"{'!'*60}\n", flush=True)
        self.kill_now = True


# Global killer instance (set in main)
_killer = None


# ---------------------------------------------------------------------------
# Training helpers (copied from train_jepa_v3.py to avoid import complexity)
# ---------------------------------------------------------------------------

def get_ema_momentum(epoch, total_epochs, start=0.996, end=0.999):
    return end - (end - start) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2


def get_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# HPO training loop (lightweight — no checkpointing, no W&B, just val_loss)
# ---------------------------------------------------------------------------

def train_one_epoch_hpo(model, dataloader, optimizer, scaler, device,
                        ema_momentum, grad_clip, grad_accum_steps):
    model.train()
    total_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x, _) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        with autocast("cuda"):
            loss, pred_loss, var_loss, cov_loss, target_std = model(x)
            scaled_loss = loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            model.update_target_encoder(momentum=ema_momentum)

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def validate_hpo(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, _ in dataloader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda"):
            loss, _, _, _, _ = model(x)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def create_objective(cfg, train_loader, val_loader, device):
    """Factory that returns an Optuna objective function."""

    hpo_epochs = cfg.get("hpo_epochs", 30)
    warmup_epochs = cfg.get("warmup_epochs", 10)
    min_lr = cfg.get("min_lr", 1e-6)
    grad_clip = cfg.get("grad_clip", 1.0)
    grad_accum_steps = cfg.get("grad_accum_steps", 2)
    ema_start = cfg.get("ema_momentum_start", 0.996)
    ema_end = cfg.get("ema_momentum_end", 0.999)

    def objective(trial):
        # ---- Suggest hyperparameters ----
        var_weight = trial.suggest_float("var_weight", 0.25, 3.0, log=True)
        cov_weight = trial.suggest_float("cov_weight", 0.001, 0.05, log=True)
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)

        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: var_weight={var_weight:.4f}, "
              f"cov_weight={cov_weight:.4f}, lr={lr:.2e}, wd={weight_decay:.4f}")
        print(f"{'='*60}")

        # ---- Build model (fresh for each trial) ----
        encoder = SpatioTemporalViT(
            in_channels=cfg.get("in_channels", 11),
            patch_size=tuple(cfg.get("patch_size", [2, 16, 16])),
            embed_dim=cfg.get("embed_dim", 384),
            depth=cfg.get("encoder_depth", 6),
            num_heads=cfg.get("num_heads", 6),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            drop_rate=cfg.get("drop_rate", 0.0),
        )
        model = VideoJEPAv3(
            encoder=encoder,
            predictor_dim=cfg.get("predictor_dim", 256),
            predictor_depth=cfg.get("predictor_depth", 3),
            predictor_heads=cfg.get("predictor_heads", 4),
            context_frames=cfg.get("context_frames", 16),
            var_weight=var_weight,
            cov_weight=cov_weight,
            var_gamma=cfg.get("var_gamma", 1.0),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(cfg.get("betas", [0.9, 0.95])),
        )
        scaler = GradScaler("cuda")

        # ---- Training loop ----
        best_val = float("inf")

        for epoch in range(hpo_epochs):
            # Check for SLURM shutdown signal before starting a new epoch
            if _killer is not None and _killer.kill_now:
                print(f"  Trial {trial.number} interrupted by shutdown signal at epoch {epoch}")
                break

            current_lr = get_lr(epoch, hpo_epochs, warmup_epochs, lr, min_lr)
            set_lr(optimizer, current_lr)
            ema_m = get_ema_momentum(epoch, hpo_epochs, ema_start, ema_end)

            train_loss = train_one_epoch_hpo(
                model, train_loader, optimizer, scaler, device,
                ema_m, grad_clip, grad_accum_steps,
            )
            val_loss = validate_hpo(model, val_loader, device)

            best_val = min(best_val, val_loss)

            print(f"  Trial {trial.number} | Epoch {epoch:2d} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Best: {best_val:.4f}", flush=True)

            # Report intermediate value for pruning
            trial.report(val_loss, epoch)

            # Prune if this trial is clearly worse than the median
            if trial.should_prune():
                print(f"  ** Trial {trial.number} PRUNED at epoch {epoch} **")
                raise TrialPruned()

        # Clean up GPU memory for next trial
        del model, optimizer, scaler
        torch.cuda.empty_cache()

        print(f"  Trial {trial.number} COMPLETE — best val_loss: {best_val:.4f}")
        return best_val

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HPO sweep for JEPA v3")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--hpo_epochs", type=int, default=15,
                        help="Epochs per trial. 15 is enough to see trends with pruning kicking in at epoch 5.")
    parser.add_argument("--hpo_subsample", type=float, default=0.20,
                        help="Fraction of training data to use per trial (default: 0.20). "
                             "HPO only needs relative ranking, not the full dataset. "
                             "0.20 × 8750 = 1750 samples → ~100 batches/epoch at bs=16 → ~1 min/epoch.")
    parser.add_argument("--hpo_batch_size", type=int, default=16,
                        help="Batch size for HPO trials (default: 16). Larger than training bs=4 "
                             "because we're not memory-constrained in short trials and want faster epochs.")
    parser.add_argument("--study_name", type=str, default="jepa_v3_hpo", help="Optuna study name")
    parser.add_argument("--db_path", type=str, default=None, help="SQLite DB path for study persistence")
    args = parser.parse_args()

    # Load base config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for key in ["data_dir", "checkpoint_dir"]:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = os.path.expandvars(cfg[key])

    cfg["hpo_epochs"] = args.hpo_epochs

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Speed calculation for the user
    full_n = 8750  # approximate, shown in logs
    subsample_n = int(full_n * args.hpo_subsample)
    batches_per_epoch = subsample_n // args.hpo_batch_size
    est_min_per_epoch = batches_per_epoch * 3 / 60  # ~3s/batch on A100
    est_total_h = args.hpo_epochs * est_min_per_epoch * args.n_trials / 60

    print("=" * 60)
    print(f"HPO SWEEP: {args.study_name}")
    print(f"  Trials: {args.n_trials} | Epochs/trial: {args.hpo_epochs}")
    print(f"  Train subsample: {args.hpo_subsample:.0%} × ~{full_n} = ~{subsample_n} samples")
    print(f"  Batch size (HPO): {args.hpo_batch_size} → ~{batches_per_epoch} batches/epoch")
    print(f"  Estimated time: ~{est_min_per_epoch:.1f} min/epoch, "
          f"~{est_total_h:.1f} GPU-hours total (before pruning)")
    print(f"  Search space:")
    print(f"    var_weight:   [0.25, 3.0]  (log-uniform)")
    print(f"    cov_weight:   [0.001, 0.05] (log-uniform)")
    print(f"    lr:           [5e-5, 5e-4]  (log-uniform)")
    print(f"    weight_decay: [0.01, 0.1]   (log-uniform)")
    print("=" * 60)

    # ---- Data (shared across all trials) ----
    train_ds = ActiveMatterDataset(cfg["data_dir"], split="train",
                                    n_frames=cfg.get("n_frames", 32))
    try:
        val_ds = ActiveMatterDataset(cfg["data_dir"], split="val",
                                      n_frames=cfg.get("n_frames", 32))
    except FileNotFoundError:
        print("Val split not found — holding out 10%")
        n_val = max(1, int(0.1 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
        )

    # Subsample training data for HPO speed.
    # HPO only needs to rank hyperparameter combos, not learn the full dataset.
    # 20% of data gives the same relative ranking at 5x the speed.
    n_subsample = max(1, int(len(train_ds) * args.hpo_subsample))
    n_discard = len(train_ds) - n_subsample
    train_ds_hpo, _ = random_split(
        train_ds, [n_subsample, n_discard],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )
    print(f"Train subsample: {len(train_ds_hpo)}/{len(train_ds)} samples "
          f"({args.hpo_subsample:.0%}) for HPO speed")

    num_workers = cfg.get("num_workers", 4)
    hpo_batch_size = args.hpo_batch_size
    train_loader = DataLoader(
        train_ds_hpo, batch_size=hpo_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=hpo_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ---- Optuna study ----
    # SQLite storage for persistence across SLURM jobs
    if args.db_path:
        storage = f"sqlite:///{args.db_path}"
    else:
        db_dir = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), "hpo")
        os.makedirs(db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(db_dir, f'{args.study_name}.db')}"

    print(f"Study storage: {storage}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,  # resume previous study if DB exists
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,   # don't prune the first 3 trials (need baseline)
            n_warmup_steps=5,     # with 15 epochs total, prune after epoch 5 (warmup done)
        ),
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
    )

    # Set up graceful shutdown handler
    global _killer
    _killer = GracefulKiller()

    def stop_callback(study, trial):
        """Stop the study if SLURM sent a shutdown signal."""
        if _killer.kill_now:
            print("Stopping study due to shutdown signal...")
            study.stop()

    objective = create_objective(cfg, train_loader, val_loader, device)
    study.optimize(objective, n_trials=args.n_trials, callbacks=[stop_callback])

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("HPO SWEEP COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"  val_loss: {study.best_trial.value:.4f}")
    print(f"  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v:.6f}")

    # Print top-5 trials
    print(f"\nTop 5 trials:")
    print(f"{'Trial':>6} | {'val_loss':>10} | {'var_wt':>8} | {'cov_wt':>8} | {'lr':>10} | {'wd':>8} | {'Status':>10}")
    print("-" * 80)
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
    )
    for t in sorted_trials[:5]:
        p = t.params
        print(f"  {t.number:>4} | {t.value:>10.4f} | {p['var_weight']:>8.4f} | "
              f"{p['cov_weight']:>8.5f} | {p['lr']:>10.2e} | {p['weight_decay']:>8.4f} | "
              f"{t.state.name:>10}")

    # Generate best config YAML
    best_cfg = copy.deepcopy(cfg)
    best_cfg["experiment_name"] = "jepa_v3_tuned"
    best_cfg["var_weight"] = study.best_trial.params["var_weight"]
    best_cfg["cov_weight"] = study.best_trial.params["cov_weight"]
    best_cfg["lr"] = study.best_trial.params["lr"]
    best_cfg["weight_decay"] = study.best_trial.params["weight_decay"]
    best_cfg["epochs"] = 100  # full training with best params
    # Remove HPO-only keys
    best_cfg.pop("hpo_epochs", None)

    out_path = os.path.join(
        os.path.dirname(args.config), "jepa_v3_tuned.yaml"
    )
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config saved to: {out_path}")
    print(f"Run full training with: sbatch --job-name=jepa-v3t scripts/train.sh {out_path}")


if __name__ == "__main__":
    main()
