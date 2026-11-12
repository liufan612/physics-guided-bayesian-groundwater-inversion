from __future__ import annotations

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

from run_mf6_pilot_experiments import (  # noqa: E402
    build_kl_basis,
    choose_obs,
    field_metrics,
    idw_interpolate,
    load_dataset_from_dir,
    load_surrogate,
    make_config,
    run_svi_inversion,
)
from run_svi_sparse_head_prior_experiment import load_surrogate_stats  # noqa: E402


SOURCE = ROOT / "WR_rebuild" / "outputs" / "mf6_highcontrast_physics_surrogate"
OUT = ROOT / "R1_1_revision_20260926" / "r2_3_noise_sensitivity"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_from_dir(SOURCE)
    cfg = make_config("journal_highcontrast_train")
    cfg.svi_steps = 500
    cfg.posterior_samples = 40
    cfg.kl_rank = 64
    cfg.residual_scale = 0.18
    cfg.head_noise = 0.30
    cfg.logk_noise = 0.20
    cfg.seed = 20260927
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate, surrogate_metrics = load_surrogate(dataset, cfg, SOURCE)
    if surrogate is None:
        raise FileNotFoundError(SOURCE / "checkpoints" / "mf6_pilot_unet.pt")
    surrogate.to(device)
    stats = load_surrogate_stats(SOURCE, device)
    basis, vals, _ = build_kl_basis(cfg, cfg.kl_rank)
    eval_indices = []
    for scenario in cfg.scenarios:
        ids = np.where((dataset["split"] == "eval") & (dataset["scenario"].astype(str) == scenario))[0]
        eval_indices.append(int(ids[0]))

    # The central condition is the protocol used by the main joint-inversion
    # output. Sensitivity changes one likelihood scale at a time.
    conditions = {
        "base": (0.20, 0.30),
        "logk_half": (0.10, 0.30),
        "logk_double": (0.40, 0.30),
        "head_half": (0.20, 0.15),
        "head_double": (0.20, 0.60),
    }
    rows = []
    started = time.perf_counter()
    for pos, idx in enumerate(eval_indices):
        rng = np.random.default_rng(20260927 + pos * 1009)
        k_pts = choose_obs(cfg, rng, 231)
        h_pts = choose_obs(cfg, rng, 300)
        logk_true = dataset["logk"][idx]
        head_true = dataset["head"][idx]
        prior = idw_interpolate(k_pts, logk_true[k_pts[:, 0], k_pts[:, 1]], cfg)
        scenario = str(dataset["scenario"][idx])

        # A K-only result under the central scale gives a reference for
        # head-driven contraction without changing the prior or observations.
        cfg.logk_noise, cfg.head_noise = conditions["base"]
        konly = run_svi_inversion(
            surrogate, dataset["x"][idx], logk_true, prior, head_true, h_pts, k_pts,
            cfg, stats, basis, vals, device, seed=cfg.seed + idx * 37 + 100000, use_head=False
        )
        km = field_metrics(logk_true, konly["logk_mean"])
        rows.append({
            "scenario": scenario,
            "case_id": int(dataset["case_id"][idx]),
            "condition": "k_only_base",
            "sigma_logk": 0.20,
            "sigma_head_m": None,
            "k_rmse": km[0],
            "k_r2": km[2],
            "head_rmse_m": field_metrics(head_true, konly["head_mean"])[0],
            "head_r2": field_metrics(head_true, konly["head_mean"])[2],
            "posterior_sd_mean": float(np.mean(konly["logk_std"])),
            "interval_width_mean": float(np.mean(2 * 1.96 * konly["logk_std"])),
        })

        for cond_index, (cond, (sigma_k, sigma_h)) in enumerate(conditions.items()):
            cfg.logk_noise = sigma_k
            cfg.head_noise = sigma_h
            res = run_svi_inversion(
                surrogate, dataset["x"][idx], logk_true, prior, head_true, h_pts, k_pts,
                cfg, stats, basis, vals, device, seed=cfg.seed + idx * 37 + cond_index + 1, use_head=True
            )
            km = field_metrics(logk_true, res["logk_mean"])
            hm = field_metrics(head_true, res["head_mean"])
            rows.append({
                "scenario": scenario,
                "case_id": int(dataset["case_id"][idx]),
                "condition": cond,
                "sigma_logk": sigma_k,
                "sigma_head_m": sigma_h,
                "k_rmse": km[0],
                "k_r2": km[2],
                "head_rmse_m": hm[0],
                "head_r2": hm[2],
                "posterior_sd_mean": float(np.mean(res["logk_std"])),
                "interval_width_mean": float(np.mean(2 * 1.96 * res["logk_std"])),
            })

    # Avoid reporting a Python-process hash as a reproducibility key.
    report = {
        "protocol": "fixed IDW prior and fixed K/head observation locations per scenario; one scale varied at a time",
        "source_dir": str(SOURCE),
        "scenarios": [str(dataset["scenario"][i]) for i in eval_indices],
        "case_ids": [int(dataset["case_id"][i]) for i in eval_indices],
        "n_k_obs": 231,
        "n_head_obs": 300,
        "base_sigma_logk": 0.20,
        "base_sigma_head_m": 0.30,
        "conditions": conditions,
        "svi_steps": cfg.svi_steps,
        "posterior_samples": cfg.posterior_samples,
        "device": str(device),
        "surrogate_metrics": surrogate_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": len(rows),
    }
    with (OUT / "metrics" / "noise_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "metrics" / "noise_sensitivity_protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    for cond in ["k_only_base", *conditions.keys()]:
        vals_ = [r for r in rows if r["condition"] == cond]
        print(cond, {
            "k_rmse_mean": float(np.mean([r["k_rmse"] for r in vals_])),
            "head_rmse_mean": float(np.mean([r["head_rmse_m"] for r in vals_])),
            "posterior_sd_mean": float(np.mean([r["posterior_sd_mean"] for r in vals_])),
            "interval_width_mean": float(np.mean([r["interval_width_mean"] for r in vals_])),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
