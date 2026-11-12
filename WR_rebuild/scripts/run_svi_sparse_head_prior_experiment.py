from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "WR_rebuild" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_sparse_head_inverse_unet_experiment import SparseInverseUNet, make_sparse_inverse_input  # noqa: E402
from run_mf6_pilot_experiments import (  # noqa: E402
    build_kl_basis,
    choose_obs,
    ensure_train_normalization_stats,
    field_metrics,
    idw_interpolate,
    load_dataset_from_dir,
    load_surrogate,
    make_config,
    run_svi_inversion,
    save_case_plot,
)

DEFAULT_SOURCE = ROOT / "WR_rebuild" / "outputs" / "mf6_highcontrast_physics_surrogate"
DEFAULT_SURROGATE = ROOT / "WR_rebuild" / "outputs" / "mf6_highcontrast_physics_surrogate"
DEFAULT_INVERSE = ROOT / "WR_rebuild" / "outputs" / "sparse_head_inverse_highcontrast_k231_h300_e60" / "checkpoints" / "sparse_head_inverse_unet.pt"


def load_sparse_inverse_model(ckpt_path: Path, device: torch.device, base: int = 32) -> SparseInverseUNet:
    model = SparseInverseUNet(base=base).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_surrogate_stats(surrogate_dir: Path, device: torch.device) -> dict:
    payload = torch.load(surrogate_dir / "checkpoints" / "mf6_pilot_unet.pt", map_location=device)
    return payload["stats"]


def sparse_inverse_prior(model: SparseInverseUNet, x_template, head_true, logk_true, k_pts, head_pts, stats, device) -> np.ndarray:
    inp, _ = make_sparse_inverse_input(x_template, head_true, logk_true, k_pts, head_pts, stats)
    with torch.no_grad():
        pred_n = model(torch.from_numpy(inp[None]).to(device))[0, 0].cpu().numpy()
    return (pred_n * stats["logk_std"] + stats["logk_mean"]).astype(np.float32)


def select_eval_indices(dataset: dict, scenarios: tuple[str, ...], max_per_scenario: int | None) -> np.ndarray:
    eval_all = np.where(dataset["split"] == "eval")[0]
    if max_per_scenario is None:
        return eval_all
    selected = []
    for scenario in scenarios:
        sidx = [int(i) for i in eval_all if str(dataset["scenario"][i]) == scenario]
        selected.extend(sidx[:max_per_scenario])
    return np.asarray(selected, dtype=np.int64)


def corr_sd_abs_error(std: np.ndarray, true: np.ndarray, pred: np.ndarray) -> float:
    err = np.abs(pred - true).ravel()
    sd = std.ravel()
    if np.std(err) < 1e-12 or np.std(sd) < 1e-12:
        return float("nan")
    return float(np.corrcoef(sd, err)[0, 1])


def add_row(rows, case_id, scenario, n_k, n_h, method, prior_name, logk_true, result, head_true, elapsed):
    k_rmse, k_mae, k_r2 = field_metrics(logk_true, result["logk_mean"])
    h_rmse, h_mae, h_r2 = field_metrics(head_true, result["head_mean"])
    rows.append({
        "case_id": case_id,
        "scenario": scenario,
        "n_k_obs": n_k,
        "n_head_obs": n_h,
        "method": method,
        "prior": prior_name,
        "k_rmse": k_rmse,
        "k_mae": k_mae,
        "k_r2": k_r2,
        "head_rmse": h_rmse,
        "head_mae": h_mae,
        "head_r2": h_r2,
        "posterior_std_mean": float(np.mean(result["logk_std"])),
        "sd_abs_error_corr": corr_sd_abs_error(result["logk_std"], logk_true, result["logk_mean"]),
        "loss_initial": float(result["loss_initial"]),
        "loss_final": float(result["loss_final"]),
        "elapsed_pair_seconds": elapsed,
    })


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for key in sorted({(r["n_k_obs"], r["n_head_obs"], r["method"]) for r in rows}):
        n_k, n_h, method = key
        vals = [r for r in rows if r["n_k_obs"] == n_k and r["n_head_obs"] == n_h and r["method"] == method]
        summary[f"k{n_k}_h{n_h}_{method}"] = {
            "n_k_obs": n_k,
            "n_head_obs": n_h,
            "method": method,
            "k_rmse_mean": float(np.mean([r["k_rmse"] for r in vals])),
            "k_r2_mean": float(np.mean([r["k_r2"] for r in vals])),
            "head_rmse_mean": float(np.mean([r["head_rmse"] for r in vals])),
            "head_r2_mean": float(np.mean([r["head_r2"] for r in vals])),
            "posterior_std_mean": float(np.mean([r["posterior_std_mean"] for r in vals])),
            "sd_abs_error_corr_mean": float(np.nanmean([r["sd_abs_error_corr"] for r in vals])),
        }
    for n_k, n_h in sorted({(r["n_k_obs"], r["n_head_obs"]) for r in rows}):
        k0 = summary[f"k{n_k}_h{n_h}_svi_k_only"]["k_rmse_mean"]
        k1 = summary[f"k{n_k}_h{n_h}_svi_head_k"]["k_rmse_mean"]
        h0 = summary[f"k{n_k}_h{n_h}_svi_k_only"]["head_rmse_mean"]
        h1 = summary[f"k{n_k}_h{n_h}_svi_head_k"]["head_rmse_mean"]
        summary[f"k{n_k}_h{n_h}_target_check"] = {
            "n_k_obs": n_k,
            "n_head_obs": n_h,
            "k_improvement_percent": float(100 * (k0 - k1) / k0),
            "head_non_degrade": bool(h1 <= h0),
            "target_hit": bool(((k0 - k1) / k0) >= 0.10 and h1 <= h0),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--surrogate-dir", default=str(DEFAULT_SURROGATE))
    parser.add_argument("--inverse-ckpt", default=str(DEFAULT_INVERSE))
    parser.add_argument("--out-dir", default=str(ROOT / "WR_rebuild" / "outputs" / "svi_sparse_head_highcontrast"))
    parser.add_argument("--max-eval-cases-per-scenario", type=int, default=None)
    parser.add_argument("--k-obs-counts", default="231")
    parser.add_argument("--head-obs-counts", default="300")
    parser.add_argument("--svi-steps", type=int, default=500)
    parser.add_argument("--posterior-samples", type=int, default=20)
    parser.add_argument("--kl-rank", type=int, default=64)
    parser.add_argument("--kl-n-inducing", type=int, default=900)
    parser.add_argument("--kl-length-scale", type=float, default=12.0)
    parser.add_argument("--residual-scale", type=float, default=0.18)
    parser.add_argument("--head-noise", type=float, default=0.30)
    parser.add_argument("--logk-noise", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    k_counts = [int(x) for x in args.k_obs_counts.split(",") if x.strip()]
    h_counts = [int(x) for x in args.head_obs_counts.split(",") if x.strip()]

    cfg = make_config("journal_highcontrast_train")
    cfg.svi_steps = args.svi_steps
    cfg.posterior_samples = args.posterior_samples
    cfg.kl_rank = args.kl_rank
    cfg.kl_n_inducing = args.kl_n_inducing
    cfg.kl_length_scale = args.kl_length_scale
    cfg.residual_scale = args.residual_scale
    cfg.head_noise = args.head_noise
    cfg.logk_noise = args.logk_noise
    cfg.seed = args.seed
    dataset = load_dataset_from_dir(Path(args.source_dir))
    ensure_train_normalization_stats(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate, surrogate_metrics = load_surrogate(dataset, cfg, Path(args.surrogate_dir))
    if surrogate is None:
        raise FileNotFoundError("missing surrogate")
    surrogate.to(device)
    surrogate_stats = load_surrogate_stats(Path(args.surrogate_dir), device)
    inverse_model = load_sparse_inverse_model(Path(args.inverse_ckpt), device)
    kl_basis, kl_vals, _ = build_kl_basis(cfg, cfg.kl_rank)
    eval_indices = select_eval_indices(dataset, cfg.scenarios, args.max_eval_cases_per_scenario)

    rows = []
    saved = set()
    t_start = time.time()
    for n_k in k_counts:
        for n_h in h_counts:
            rng = np.random.default_rng(args.seed + 919)
            for idx in eval_indices:
                idx = int(idx)
                scenario = str(dataset["scenario"][idx])
                x_template = dataset["x"][idx]
                logk_true = dataset["logk"][idx]
                head_true = dataset["head"][idx]
                k_pts = choose_obs(cfg, rng, n_k)
                head_pts = choose_obs(cfg, rng, n_h)
                k_values = logk_true[k_pts[:, 0], k_pts[:, 1]]
                idw_prior = idw_interpolate(k_pts, k_values, cfg)
                head_prior = sparse_inverse_prior(inverse_model, x_template, head_true, logk_true, k_pts, head_pts, dataset["stats"], device)
                t0 = time.time()
                svi_k = run_svi_inversion(surrogate, x_template, logk_true, idw_prior, head_true, head_pts, k_pts, cfg, surrogate_stats, kl_basis, kl_vals, device, seed=args.seed + idx * 10 + n_k + n_h, use_head=False)
                svi_h = run_svi_inversion(surrogate, x_template, logk_true, head_prior, head_true, head_pts, k_pts, cfg, surrogate_stats, kl_basis, kl_vals, device, seed=args.seed + idx * 10 + n_k + n_h + 1, use_head=True)
                elapsed = time.time() - t0
                cid = int(dataset["case_id"][idx])
                add_row(rows, cid, scenario, n_k, n_h, "svi_k_only", "idw_k_only", logk_true, svi_k, head_true, elapsed)
                add_row(rows, cid, scenario, n_k, n_h, "svi_head_k", "sparse_head_inverse_prior", logk_true, svi_h, head_true, elapsed)
                if n_k == k_counts[-1] and n_h == h_counts[-1] and scenario not in saved:
                    save_case_plot(out_dir / "figures" / f"case_{cid:04d}_{scenario}.png", logk_true, svi_k["logk_mean"], svi_h["logk_mean"], svi_h["logk_std"], head_true, svi_h["head_mean"])
                    np.savez_compressed(out_dir / "figures" / f"case_{cid:04d}_{scenario}_fields.npz", true=logk_true, svi_k=svi_k["logk_mean"], svi_head=svi_h["logk_mean"], sd=svi_h["logk_std"], head_true=head_true, head_pred=svi_h["head_mean"])
                    saved.add(scenario)
                print(f"k={n_k} h={n_h} case={cid} {scenario}: K-only={rows[-2]['k_rmse']:.3f} head+K={rows[-1]['k_rmse']:.3f} corr={rows[-1]['sd_abs_error_corr']:.3f}")
    metrics = out_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "svi_sparse_head_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    summary["elapsed_total_seconds"] = time.time() - t_start
    summary["surrogate_best_eval_rmse"] = surrogate_metrics.get("best_eval_rmse")
    (metrics / "svi_sparse_head_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

