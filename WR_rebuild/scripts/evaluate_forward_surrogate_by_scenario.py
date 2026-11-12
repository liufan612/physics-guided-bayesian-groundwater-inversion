from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "WR_rebuild" / "scripts"))

from run_mf6_pilot_experiments import (  # noqa: E402
    ArrayDataset,
    boundary_condition_loss,
    field_metrics,
    load_dataset_from_dir,
    load_surrogate,
    make_config,
    pde_residual_loss,
)


def load_config(source_dir: Path, profile: str):
    cfg = make_config(profile)
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return cfg
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    valid = {f.name for f in fields(cfg)}
    for key, value in payload.items():
        if key in valid:
            if key in {"scenarios", "head_obs_counts"} and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    return cfg


def predict_batches(model, dataset, indices: np.ndarray, device: torch.device, batch_size: int):
    stats = dataset["stats"]
    ds = ArrayDataset(dataset["x"][indices], dataset["head"][indices], stats)
    preds_raw = []
    true_raw = []
    pde_pred = []
    pde_true = []
    bc_pred = []
    bc_true = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            stop = min(start + batch_size, len(ds))
            xs = torch.stack([ds[i][0] for i in range(start, stop)]).to(device)
            ys = torch.stack([ds[i][1] for i in range(start, stop)]).to(device)
            pred = model(xs)
            pred_raw = pred[:, 0].cpu().numpy() * stats["head_std"] + stats["head_mean"]
            y_raw = ys[:, 0].cpu().numpy() * stats["head_std"] + stats["head_mean"]
            preds_raw.append(pred_raw)
            true_raw.append(y_raw)
            pde_pred.extend(
                pde_residual_loss(pred, xs, stats, cfg_global).detach().cpu().repeat(stop - start).numpy().tolist()
            )
            pde_true.extend(
                pde_residual_loss(ys, xs, stats, cfg_global).detach().cpu().repeat(stop - start).numpy().tolist()
            )
            bc_pred.extend(
                boundary_condition_loss(pred, xs, stats).detach().cpu().repeat(stop - start).numpy().tolist()
            )
            bc_true.extend(
                boundary_condition_loss(ys, xs, stats).detach().cpu().repeat(stop - start).numpy().tolist()
            )
    return (
        np.concatenate(preds_raw, axis=0),
        np.concatenate(true_raw, axis=0),
        np.asarray(pde_pred, dtype=np.float64),
        np.asarray(pde_true, dtype=np.float64),
        np.asarray(bc_pred, dtype=np.float64),
        np.asarray(bc_true, dtype=np.float64),
    )


def save_example_figure(out_path: Path, scenario: str, case_id: int, true_head: np.ndarray, pred_head: np.ndarray) -> None:
    err = pred_head - true_head
    h_vmin = float(np.nanpercentile(np.stack([true_head, pred_head]), 2))
    h_vmax = float(np.nanpercentile(np.stack([true_head, pred_head]), 98))
    e_abs = float(np.nanpercentile(np.abs(err), 98))
    panels = [
        (true_head, "a  MODFLOW6 head", "YlGnBu", h_vmin, h_vmax),
        (pred_head, "b  Surrogate head", "YlGnBu", h_vmin, h_vmax),
        (err, "c  Head error", "RdBu_r", -e_abs, e_abs),
    ]
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.55), constrained_layout=True)
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, loc="left", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.018)
        cb.ax.tick_params(labelsize=7, width=0.4, length=2)
    fig.suptitle(f"{scenario}, case {case_id:04d}", x=0.03, ha="left", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def save_summary_figure(out_path: Path, summary_rows: list[dict]) -> None:
    scenarios = [r["scenario"] for r in summary_rows]
    rmse = [r["head_rmse_mean"] for r in summary_rows]
    pde_ratio = [r["pde_residual_ratio_mean"] for r in summary_rows]
    bc = [r["bc_loss_mean"] for r in summary_rows]
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.45), constrained_layout=True)
    colors = ["#2a9d8f", "#457b9d", "#e76f51"]
    for ax, vals, title, ylabel in [
        (axes[0], rmse, "a  Head RMSE", "m"),
        (axes[1], pde_ratio, "b  PDE residual ratio", "pred / MODFLOW6"),
        (axes[2], bc, "c  Boundary loss", "normalized MSE"),
    ]:
        ax.bar(scenarios, vals, color=colors, width=0.62)
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile", default="journal_highcontrast_train")
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global cfg_global
    cfg_global = load_config(source_dir, args.profile)
    dataset = load_dataset_from_dir(source_dir)
    loaded = load_surrogate(dataset, cfg_global, source_dir)
    if loaded is None:
        raise FileNotFoundError(source_dir / "checkpoints" / "mf6_pilot_unet.pt")
    model, metrics = loaded
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eval_indices = np.where(dataset["split"] == "eval")[0]
    pred, true, pde_pred, pde_true, bc_pred, bc_true = predict_batches(
        model, dataset, eval_indices, device, args.batch_size
    )

    rows = []
    for pos, idx in enumerate(eval_indices):
        rmse, mae, r2 = field_metrics(true[pos], pred[pos])
        rows.append(
            {
                "case_id": int(dataset["case_id"][idx]),
                "scenario": str(dataset["scenario"][idx]),
                "head_rmse": rmse,
                "head_mae": mae,
                "head_r2": r2,
                "head_max_abs_error": float(np.max(np.abs(pred[pos] - true[pos]))),
                "pde_residual_pred": float(pde_pred[pos]),
                "pde_residual_true": float(pde_true[pos]),
                "pde_residual_ratio": float(pde_pred[pos] / max(pde_true[pos], 1e-12)),
                "bc_loss_pred": float(bc_pred[pos]),
                "bc_loss_true": float(bc_true[pos]),
            }
        )

    summary_rows = []
    for scenario in cfg_global.scenarios:
        srows = [r for r in rows if r["scenario"] == scenario]
        if not srows:
            continue
        summary_rows.append(
            {
                "scenario": scenario,
                "n_eval": len(srows),
                "head_rmse_mean": float(np.mean([r["head_rmse"] for r in srows])),
                "head_rmse_std": float(np.std([r["head_rmse"] for r in srows], ddof=0)),
                "head_mae_mean": float(np.mean([r["head_mae"] for r in srows])),
                "head_r2_mean": float(np.mean([r["head_r2"] for r in srows])),
                "pde_residual_pred_mean": float(np.mean([r["pde_residual_pred"] for r in srows])),
                "pde_residual_true_mean": float(np.mean([r["pde_residual_true"] for r in srows])),
                "pde_residual_ratio_mean": float(np.mean([r["pde_residual_ratio"] for r in srows])),
                "bc_loss_mean": float(np.mean([r["bc_loss_pred"] for r in srows])),
            }
        )

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "forward_surrogate_case_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (metrics_dir / "forward_surrogate_scenario_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (metrics_dir / "forward_surrogate_scenario_summary.json").write_text(
        json.dumps({"surrogate": metrics, "by_scenario": summary_rows}, indent=2), encoding="utf-8"
    )

    fig_dir = out_dir / "figures"
    save_summary_figure(fig_dir / "forward_surrogate_by_scenario.png", summary_rows)
    for scenario in cfg_global.scenarios:
        candidates = [i for i, idx in enumerate(eval_indices) if str(dataset["scenario"][idx]) == scenario]
        if not candidates:
            continue
        pos = candidates[0]
        case_id = int(dataset["case_id"][eval_indices[pos]])
        save_example_figure(fig_dir / f"case_{case_id:04d}_{scenario}_forward_surrogate.png", scenario, case_id, true[pos], pred[pos])

    print(json.dumps({"surrogate": metrics, "by_scenario": summary_rows}, indent=2))
    print(f"Saved outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
