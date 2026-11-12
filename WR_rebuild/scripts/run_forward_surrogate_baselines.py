from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "WR_rebuild" / "scripts"))

from run_mf6_pilot_experiments import (  # noqa: E402
    ArrayDataset,
    Config,
    boundary_condition_loss,
    field_metrics,
    load_dataset_from_dir,
    make_config,
    pde_residual_loss,
)


class PlainCNNSurrogate(nn.Module):
    def __init__(self, in_channels: int = 6, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderDecoderSurrogate(nn.Module):
    def __init__(self, in_channels: int = 6, base: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base, 3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.GELU())
        self.dec2 = nn.Sequential(nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.GELU())
        self.dec1 = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y = self.enc3(self.enc2(self.enc1(x)))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.dec2(y)
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        y = self.dec1(y)
        return self.out(y)


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int = 6, base: int = 32):
        super().__init__()
        self.e1 = self.block(in_channels, base)
        self.e2 = self.block(base, base * 2)
        self.e3 = self.block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = self.block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = self.block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        y = self.up2(e3)
        if y.shape[-2:] != e2.shape[-2:]:
            y = F.interpolate(y, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.d2(torch.cat([y, e2], dim=1))
        y = self.up1(y)
        if y.shape[-2:] != e1.shape[-2:]:
            y = F.interpolate(y, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.d1(torch.cat([y, e1], dim=1))
        return self.out(y)


def make_model(arch: str, base: int) -> nn.Module:
    arch = arch.lower()
    if arch == "cnn":
        return PlainCNNSurrogate(base=base)
    if arch in {"encoder_decoder", "encdec"}:
        return EncoderDecoderSurrogate(base=base)
    if arch == "unet":
        return UNetSmall(base=base)
    raise ValueError(f"Unknown architecture: {arch}")


def load_config(source_dir: Path, profile: str) -> Config:
    cfg = make_config(profile)
    config_path = source_dir / "config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        valid = {f.name for f in fields(cfg)}
        for key, value in payload.items():
            if key in valid:
                if key in {"scenarios", "head_obs_counts"} and isinstance(value, list):
                    value = tuple(value)
                setattr(cfg, key, value)
    return cfg


def eval_model(model, loader, stats, cfg, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            pred_raw = pred[:, 0].cpu().numpy() * stats["head_std"] + stats["head_mean"]
            true_raw = y[:, 0].cpu().numpy() * stats["head_std"] + stats["head_mean"]
            for i in range(pred_raw.shape[0]):
                rmse, mae, r2 = field_metrics(true_raw[i], pred_raw[i])
                rows.append({"rmse": rmse, "mae": mae, "r2": r2})
            pde = float(pde_residual_loss(pred, x, stats, cfg).cpu().item())
            bc = float(boundary_condition_loss(pred, x, stats).cpu().item())
            for row in rows[-pred_raw.shape[0] :]:
                row["pde"] = pde
                row["bc"] = bc
    return rows


def train_variant(dataset, cfg, variant, out_dir, device):
    name = variant["name"]
    arch = variant["arch"]
    lam_pde = float(variant.get("lam_pde", 0.0))
    lam_bc = float(variant.get("lam_bc", 0.0))
    pde_warmup = int(variant.get("pde_warmup_epochs", cfg.pde_warmup_epochs))
    model = make_model(arch, cfg.base_channels).to(device)
    train_mask = dataset["split"] == "train"
    eval_mask = dataset["split"] == "eval"
    train_ds = ArrayDataset(dataset["x"][train_mask], dataset["head"][train_mask], dataset["stats"])
    eval_ds = ArrayDataset(dataset["x"][eval_mask], dataset["head"][eval_mask], dataset["stats"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    hist = []
    best = {
        "rmse": (float("inf"), "best_rmse.pt"),
        "pde": (float("inf"), "best_pde.pt"),
        "combined": (float("inf"), "best_combined.pt"),
    }
    ckpt_dir = out_dir / "checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        data_losses = []
        pde_losses = []
        bc_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            data_loss = F.mse_loss(pred, y)
            pde_loss = torch.tensor(0.0, device=device)
            bc_loss = torch.tensor(0.0, device=device)
            loss = data_loss
            if ep > pde_warmup:
                if lam_pde > 0:
                    pde_loss = pde_residual_loss(pred, x, dataset["stats"], cfg)
                    loss = loss + lam_pde * pde_loss
                if lam_bc > 0:
                    bc_loss = boundary_condition_loss(pred, x, dataset["stats"])
                    loss = loss + lam_bc * bc_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            data_losses.append(float(data_loss.detach().cpu().item()))
            pde_losses.append(float(pde_loss.detach().cpu().item()))
            bc_losses.append(float(bc_loss.detach().cpu().item()))
        eval_rows = eval_model(model, eval_loader, dataset["stats"], cfg, device)
        eval_rmse = float(np.mean([r["rmse"] for r in eval_rows]))
        eval_pde = float(np.mean([r["pde"] for r in eval_rows]))
        eval_bc = float(np.mean([r["bc"] for r in eval_rows]))
        combined = eval_rmse + 5.0e-4 * eval_pde + 0.2 * eval_bc
        row = {
            "variant": name,
            "epoch": ep,
            "arch": arch,
            "lam_pde": lam_pde,
            "lam_bc": lam_bc,
            "train_loss": float(np.mean(losses)),
            "train_data_loss": float(np.mean(data_losses)),
            "train_pde_loss": float(np.mean(pde_losses)),
            "train_bc_loss": float(np.mean(bc_losses)),
            "eval_rmse": eval_rmse,
            "eval_pde": eval_pde,
            "eval_bc": eval_bc,
            "eval_combined": combined,
        }
        hist.append(row)
        for key, value in [("rmse", eval_rmse), ("pde", eval_pde), ("combined", combined)]:
            if value < best[key][0]:
                best[key] = (value, best[key][1])
                torch.save(
                    {"model": model.state_dict(), "stats": dataset["stats"], "cfg": asdict(cfg), "variant": variant, "epoch": ep},
                    ckpt_dir / best[key][1],
                )
        if ep == 1 or ep % 20 == 0 or ep == cfg.epochs:
            print(
                f"{name} epoch {ep:04d} rmse={eval_rmse:.4f} pde={eval_pde:.2f} "
                f"bc={eval_bc:.5f} combined={combined:.4f}"
            )
    return hist


def evaluate_checkpoint(dataset, cfg, variant, checkpoint_kind, out_dir, device):
    name = variant["name"]
    model = make_model(variant["arch"], cfg.base_channels).to(device)
    ckpt = torch.load(out_dir / "checkpoints" / name / f"best_{checkpoint_kind}.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    eval_indices = np.where(dataset["split"] == "eval")[0]
    eval_ds = ArrayDataset(dataset["x"][eval_indices], dataset["head"][eval_indices], dataset["stats"])
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False)
    model.eval()
    rows = []
    with torch.no_grad():
        for pos, (x, y) in enumerate(loader):
            idx = eval_indices[pos]
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            pred_raw = pred[0, 0].cpu().numpy() * dataset["stats"]["head_std"] + dataset["stats"]["head_mean"]
            true_raw = y[0, 0].cpu().numpy() * dataset["stats"]["head_std"] + dataset["stats"]["head_mean"]
            rmse, mae, r2 = field_metrics(true_raw, pred_raw)
            rows.append(
                {
                    "variant": name,
                    "checkpoint": checkpoint_kind,
                    "arch": variant["arch"],
                    "case_id": int(dataset["case_id"][idx]),
                    "scenario": str(dataset["scenario"][idx]),
                    "head_rmse": rmse,
                    "head_mae": mae,
                    "head_r2": r2,
                    "pde_residual": float(pde_residual_loss(pred, x, dataset["stats"], cfg).cpu().item()),
                    "bc_loss": float(boundary_condition_loss(pred, x, dataset["stats"]).cpu().item()),
                }
            )
    return rows


def save_summary_plot(summary, out_dir):
    variants = [r["variant"] for r in summary]
    rmse = [r["head_rmse_mean"] for r in summary]
    pde = [r["pde_residual_mean"] for r in summary]
    bc = [r["bc_loss_mean"] for r in summary]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    for ax, values, title, ylabel in [
        (axes[0], rmse, "a  Head RMSE", "m"),
        (axes[1], pde, "b  PDE residual", "normalized residual"),
        (axes[2], bc, "c  Boundary loss", "normalized MSE"),
    ]:
        ax.bar(np.arange(len(variants)), values, color="#4c78a8", width=0.65)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(variants, rotation=35, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "forward_surrogate_baseline_ablation.png", dpi=260)
    fig.savefig(fig_dir / "forward_surrogate_baseline_ablation.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile", default="journal_highcontrast_train")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(source_dir, args.profile)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    dataset = load_dataset_from_dir(source_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [
        {"name": "CNN", "arch": "cnn", "lam_pde": 0.0, "lam_bc": 0.0, "pde_warmup_epochs": cfg.epochs + 1},
        {"name": "EncoderDecoder", "arch": "encoder_decoder", "lam_pde": 0.0, "lam_bc": 0.0, "pde_warmup_epochs": cfg.epochs + 1},
        {"name": "UNet_data", "arch": "unet", "lam_pde": 0.0, "lam_bc": 0.0, "pde_warmup_epochs": cfg.epochs + 1},
        {"name": "UNet_BC", "arch": "unet", "lam_pde": 0.0, "lam_bc": 0.2, "pde_warmup_epochs": cfg.pde_warmup_epochs},
        {"name": "UNet_PDE", "arch": "unet", "lam_pde": cfg.lam_pde, "lam_bc": 0.0, "pde_warmup_epochs": cfg.pde_warmup_epochs},
        {"name": "PI_UNet_PDE_BC", "arch": "unet", "lam_pde": cfg.lam_pde, "lam_bc": cfg.lam_bc, "pde_warmup_epochs": cfg.pde_warmup_epochs},
    ]
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    all_hist = []
    for variant in variants:
        all_hist.extend(train_variant(dataset, cfg, variant, out_dir, device))
    with (metrics_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_hist[0].keys()))
        writer.writeheader()
        writer.writerows(all_hist)
    case_rows = []
    for variant in variants:
        for checkpoint in ["rmse", "pde", "combined"]:
            case_rows.extend(evaluate_checkpoint(dataset, cfg, variant, checkpoint, out_dir, device))
    with (metrics_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)
    summary = []
    for variant in variants:
        for checkpoint in ["rmse", "pde", "combined"]:
            rows = [r for r in case_rows if r["variant"] == variant["name"] and r["checkpoint"] == checkpoint]
            summary.append(
                {
                    "variant": variant["name"],
                    "checkpoint": checkpoint,
                    "arch": variant["arch"],
                    "head_rmse_mean": float(np.mean([r["head_rmse"] for r in rows])),
                    "head_mae_mean": float(np.mean([r["head_mae"] for r in rows])),
                    "head_r2_mean": float(np.mean([r["head_r2"] for r in rows])),
                    "pde_residual_mean": float(np.mean([r["pde_residual"] for r in rows])),
                    "bc_loss_mean": float(np.mean([r["bc_loss"] for r in rows])),
                }
            )
    with (metrics_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_summary_plot([r for r in summary if r["checkpoint"] == "combined"], out_dir)
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

