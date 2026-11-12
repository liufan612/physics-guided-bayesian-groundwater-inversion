from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.mcmc import MCMC, NUTS
from pyro.ops.stats import effective_sample_size, split_gelman_rubin
from pyro.optim import Adam


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "WR_rebuild" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_mf6_pilot_experiments import (  # noqa: E402
    build_kl_basis,
    choose_obs,
    ensure_train_normalization_stats,
    field_metrics,
    load_dataset_from_dir,
    load_surrogate,
    make_config,
)
from run_svi_sparse_head_prior_experiment import (  # noqa: E402
    load_sparse_inverse_model,
    load_surrogate_stats,
    sparse_inverse_prior,
)


DEFAULT_SOURCE = ROOT / "WR_rebuild" / "outputs" / "mf6_highcontrast_physics_surrogate"
DEFAULT_SURROGATE = DEFAULT_SOURCE
DEFAULT_INVERSE = (
    ROOT
    / "WR_rebuild"
    / "outputs"
    / "sparse_head_inverse_highcontrast_k231_h300_e60"
    / "checkpoints"
    / "sparse_head_inverse_unet.pt"
)
DEFAULT_OUT = ROOT / "WR_rebuild" / "results" / "svi_mcmc_benchmark"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def hardware_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "pyro": pyro.__version__,
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_bytes": props.total_memory,
                "cuda_runtime": torch.version.cuda,
            }
        )
    return info


def find_case_index(dataset: dict[str, np.ndarray], case_id: int, scenario: str) -> int:
    matches = np.where(
        (dataset["case_id"].astype(np.int64) == int(case_id))
        & (dataset["scenario"].astype(str) == scenario)
        & (dataset["split"].astype(str) == "eval")
    )[0]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one eval case for case_id={case_id}, scenario={scenario}; got {matches}")
    return int(matches[0])


def reproduce_matched_observations(
    dataset: dict[str, np.ndarray],
    cfg: Any,
    target_index: int,
    n_k_obs: int,
    n_head_obs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 919)
    eval_indices = np.where(dataset["split"] == "eval")[0]
    for idx in eval_indices:
        k_points = choose_obs(cfg, rng, n_k_obs)
        head_points = choose_obs(cfg, rng, n_head_obs)
        if int(idx) == target_index:
            return k_points.astype(np.int64), head_points.astype(np.int64)
    raise RuntimeError(f"Target dataset index {target_index} was not encountered in eval split")


class SharedPosteriorTarget:
    def __init__(
        self,
        surrogate: torch.nn.Module,
        x_template: np.ndarray,
        prior_mean_field: np.ndarray,
        logk_true: np.ndarray,
        head_true: np.ndarray,
        k_points: np.ndarray,
        head_points: np.ndarray,
        stats: dict[str, Any],
        basis: torch.Tensor,
        eigenvalues: torch.Tensor,
        latent_rank: int,
        residual_scale: float,
        sigma_k: float,
        sigma_h: float,
        device: torch.device,
    ) -> None:
        self.surrogate = surrogate.eval()
        for parameter in self.surrogate.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.latent_rank = latent_rank
        self.sigma_k = float(sigma_k)
        self.sigma_h = float(sigma_h)
        self.basis = basis.to(device)
        self.eigenvalues = eigenvalues.to(device)
        sqrt_values = torch.sqrt(torch.clamp(self.eigenvalues, min=1e-8))
        self.coefficient_scale = residual_scale * sqrt_values / torch.sqrt(torch.sum(self.eigenvalues) + 1e-8)
        self.prior_mean = torch.from_numpy(prior_mean_field).float().to(device)
        self.logk_true = torch.from_numpy(logk_true).float().to(device)
        self.head_true = torch.from_numpy(head_true).float().to(device)
        self.x_base = torch.from_numpy(x_template.copy()).float().to(device)
        self.x_mean = torch.tensor(stats["x_mean"], dtype=torch.float32, device=device)
        self.x_std = torch.tensor(stats["x_std"], dtype=torch.float32, device=device)
        self.head_mean = torch.tensor(stats["head_mean"], dtype=torch.float32, device=device)
        self.head_std = torch.tensor(stats["head_std"], dtype=torch.float32, device=device)
        self.k_rows = torch.from_numpy(k_points[:, 0]).long().to(device)
        self.k_cols = torch.from_numpy(k_points[:, 1]).long().to(device)
        self.h_rows = torch.from_numpy(head_points[:, 0]).long().to(device)
        self.h_cols = torch.from_numpy(head_points[:, 1]).long().to(device)
        self.k_observed = self.logk_true[self.k_rows, self.k_cols]
        self.h_observed = self.head_true[self.h_rows, self.h_cols]
        self.forward_calls = 0

    def reset_counter(self) -> None:
        self.forward_calls = 0

    def logk_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.prior_mean + torch.sum(
            (self.coefficient_scale * z)[:, None, None] * self.basis,
            dim=0,
        )

    def forward_from_z(self, z: torch.Tensor, count: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        logk = self.logk_from_z(z)
        x_input = self.x_base.clone()
        x_input[0] = logk
        n_channels = self.x_mean.shape[0]
        x_input[:n_channels] = (
            x_input[:n_channels] - self.x_mean[:, None, None]
        ) / self.x_std[:, None, None]
        head_normalized = self.surrogate(x_input[None])[0, 0]
        head = head_normalized * self.head_std + self.head_mean
        if count:
            self.forward_calls += 1
        return logk, head

    def model(self) -> None:
        z = pyro.sample(
            "z",
            dist.Normal(
                torch.zeros(self.latent_rank, device=self.device),
                torch.ones(self.latent_rank, device=self.device),
            ).to_event(1),
        )
        logk, head = self.forward_from_z(z, count=True)
        pyro.sample(
            "k_obs",
            dist.Normal(logk[self.k_rows, self.k_cols], self.sigma_k).to_event(1),
            obs=self.k_observed,
        )
        pyro.sample(
            "h_obs",
            dist.Normal(head[self.h_rows, self.h_cols], self.sigma_h).to_event(1),
            obs=self.h_observed,
        )

    @torch.no_grad()
    def predict_heads(self, z_samples: np.ndarray, batch_size: int) -> tuple[np.ndarray, int]:
        z_tensor = torch.from_numpy(z_samples.astype(np.float32)).to(self.device)
        head_sum = torch.zeros_like(self.head_true)
        candidate_evaluations = 0
        n_channels = self.x_mean.shape[0]
        for start in range(0, len(z_tensor), batch_size):
            z_batch = z_tensor[start : start + batch_size]
            scaled = z_batch * self.coefficient_scale[None, :]
            logk_batch = self.prior_mean[None] + torch.einsum("br,rhw->bhw", scaled, self.basis)
            x_batch = self.x_base[None].repeat(len(z_batch), 1, 1, 1)
            x_batch[:, 0] = logk_batch
            x_batch[:, :n_channels] = (
                x_batch[:, :n_channels] - self.x_mean[None, :, None, None]
            ) / self.x_std[None, :, None, None]
            head_batch = self.surrogate(x_batch)[:, 0] * self.head_std + self.head_mean
            head_sum += head_batch.sum(dim=0)
            candidate_evaluations += len(z_batch)
        return (head_sum / len(z_tensor)).cpu().numpy().astype(np.float32), candidate_evaluations

    @torch.no_grad()
    def summarize_fields(
        self,
        z_samples: np.ndarray,
        cell_chunk: int = 512,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        z_tensor = torch.from_numpy(z_samples.astype(np.float32)).to(self.device)
        scaled = z_tensor * self.coefficient_scale[None, :]
        basis_flat = self.basis.reshape(self.latent_rank, -1)
        prior_flat = self.prior_mean.reshape(-1)
        mean_flat = torch.empty_like(prior_flat)
        std_flat = torch.empty_like(prior_flat)
        interval_width_sum = 0.0
        for start in range(0, prior_flat.numel(), cell_chunk):
            stop = min(start + cell_chunk, prior_flat.numel())
            field_chunk = prior_flat[start:stop][None] + scaled @ basis_flat[:, start:stop]
            mean_flat[start:stop] = field_chunk.mean(dim=0)
            std_flat[start:stop] = field_chunk.std(dim=0, unbiased=False)
            quantiles = torch.quantile(field_chunk, torch.tensor([0.025, 0.975], device=self.device), dim=0)
            interval_width_sum += float((quantiles[1] - quantiles[0]).sum().item())
        shape = tuple(self.prior_mean.shape)
        mean_field = mean_flat.reshape(shape).cpu().numpy().astype(np.float32)
        std_field = std_flat.reshape(shape).cpu().numpy().astype(np.float32)
        mean_interval_width = interval_width_sum / prior_flat.numel()
        return mean_field, std_field, float(mean_interval_width)


def run_svi(
    target: SharedPosteriorTarget,
    steps: int,
    sample_count: int,
    learning_rate: float,
    seed: int,
    out_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    target.reset_counter()
    if target.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target.device)
    guide = AutoDiagonalNormal(target.model)
    optimizer = Adam({"lr": learning_rate})
    svi = SVI(target.model, guide, optimizer, loss=Trace_ELBO())
    losses: list[float] = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        losses.append(float(svi.step()))
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            print(f"SVI step {step}/{steps}: loss={losses[-1]:.6f}", flush=True)
    elapsed = time.perf_counter() - start
    inference_calls = int(target.forward_calls)
    peak_memory = int(torch.cuda.max_memory_allocated(target.device)) if target.device.type == "cuda" else None

    loc = guide.loc.detach().cpu().numpy().astype(np.float32)
    scale = guide.scale.detach().cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(seed + 7001)
    z_samples = (loc[None] + scale[None] * rng.standard_normal((sample_count, len(loc)))).astype(np.float32)

    pd.DataFrame(
        {"iteration": np.arange(1, len(losses) + 1), "elbo_loss": losses, "elbo": -np.asarray(losses)}
    ).to_csv(out_dir / "figure_source_data" / "svi_elbo.csv", index=False)
    np.savez_compressed(out_dir / "posterior_samples_svi.npz", z=z_samples, loc=loc, scale=scale)
    metrics = {
        "algorithm": "SVI with AutoDiagonalNormal",
        "guide": "diagonal Gaussian",
        "optimizer": "Pyro Adam",
        "learning_rate": learning_rate,
        "iterations": steps,
        "posterior_latent_samples": sample_count,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "final_elbo": -losses[-1],
        "inference_wall_seconds": elapsed,
        "pi_unet_inference_evaluations": inference_calls,
        "peak_gpu_memory_bytes": peak_memory,
        "seed": seed,
        "variational_loc": loc,
        "variational_scale": scale,
    }
    write_json(out_dir / "svi_metrics.json", metrics)
    return z_samples, metrics


def run_nuts_chain(
    target: SharedPosteriorTarget,
    chain_index: int,
    warmup_steps: int,
    retained_samples: int,
    target_accept: float,
    max_tree_depth: int,
    seed: int,
    out_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    chain_path = out_dir / f"posterior_samples_mcmc_chain{chain_index + 1:02d}.npz"
    metric_path = out_dir / "chain_metrics" / f"chain_{chain_index + 1:02d}.json"
    if chain_path.exists() and metric_path.exists():
        print(f"Reusing completed MCMC chain {chain_index + 1}: {chain_path.name}", flush=True)
        return np.load(chain_path)["z"].astype(np.float32), json.loads(metric_path.read_text(encoding="utf-8"))

    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    target.reset_counter()
    if target.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target.device)

    timing: dict[str, float | None] = {"sample_start": None}
    run_start = time.perf_counter()

    def hook_fn(kernel: Any, samples: Any, stage: str, iteration: int) -> None:
        stage_lower = str(stage).lower()
        if "sample" in stage_lower and timing["sample_start"] is None:
            timing["sample_start"] = time.perf_counter()
        total = warmup_steps if "warm" in stage_lower else retained_samples
        stride = max(1, total // 10)
        if iteration == 0 or (iteration + 1) % stride == 0 or iteration + 1 == total:
            print(
                f"Chain {chain_index + 1} {stage} {iteration + 1}/{total}; forward_calls={target.forward_calls}",
                flush=True,
            )

    kernel = NUTS(
        target.model,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
        full_mass=False,
        jit_compile=False,
    )
    sampler = MCMC(
        kernel,
        num_samples=retained_samples,
        warmup_steps=warmup_steps,
        num_chains=1,
        hook_fn=hook_fn,
        disable_progbar=True,
        save_params=("z",),
    )
    sampler.run()
    finish = time.perf_counter()
    samples = sampler.get_samples(group_by_chain=True)["z"][0].detach().cpu().numpy().astype(np.float32)
    diagnostics = sampler.diagnostics()
    divergence_indices = diagnostics.get("divergences", {}).get("chain 0", [])
    acceptance_rate = diagnostics.get("acceptance rate", {}).get("chain 0")
    sample_start = timing["sample_start"]
    warmup_seconds = (sample_start - run_start) if sample_start is not None else None
    sampling_seconds = (finish - sample_start) if sample_start is not None else None
    peak_memory = int(torch.cuda.max_memory_allocated(target.device)) if target.device.type == "cuda" else None
    inverse_mass = {
        str(key): value.detach().cpu().numpy()
        for key, value in getattr(kernel, "inverse_mass_matrix", {}).items()
    }
    np.savez_compressed(
        chain_path,
        z=samples,
        divergence_indices=np.asarray(divergence_indices, dtype=np.int64),
        step_size=np.asarray([float(kernel.step_size)], dtype=np.float64),
        **{f"inverse_mass_{i}": value for i, value in enumerate(inverse_mass.values())},
    )
    metrics = {
        "chain": chain_index + 1,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "retained_samples": retained_samples,
        "target_accept_probability": target_accept,
        "max_tree_depth": max_tree_depth,
        "wall_seconds": finish - run_start,
        "warmup_seconds": warmup_seconds,
        "sampling_seconds": sampling_seconds,
        "pi_unet_inference_evaluations": int(target.forward_calls),
        "divergence_count": len(divergence_indices),
        "divergence_indices": divergence_indices,
        "acceptance_rate": acceptance_rate,
        "final_step_size": float(kernel.step_size),
        "peak_gpu_memory_bytes": peak_memory,
    }
    write_json(metric_path, metrics)
    return samples, metrics


def aggregate_mcmc(
    chain_samples: list[np.ndarray],
    chain_metrics: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    stacked = np.stack(chain_samples, axis=0).astype(np.float32)
    tensor = torch.from_numpy(stacked)
    rhat = split_gelman_rubin(tensor, chain_dim=0, sample_dim=1).cpu().numpy().astype(np.float64)
    ess = effective_sample_size(tensor, chain_dim=0, sample_dim=1).cpu().numpy().astype(np.float64)
    diagnostics_table = pd.DataFrame(
        {
            "latent_index_zero_based": np.arange(stacked.shape[-1]),
            "latent_mode": np.arange(1, stacked.shape[-1] + 1),
            "r_hat": rhat,
            "ess": ess,
        }
    )
    diagnostics_table.to_csv(out_dir / "figure_source_data" / "mcmc_diagnostics.csv", index=False)
    np.savez_compressed(out_dir / "posterior_samples_mcmc.npz", z=stacked)
    total_wall = float(sum(float(row["wall_seconds"]) for row in chain_metrics))
    metrics = {
        "algorithm": "NUTS",
        "chains": stacked.shape[0],
        "warmup_steps_per_chain": chain_metrics[0]["warmup_steps"],
        "retained_samples_per_chain": stacked.shape[1],
        "total_retained_samples": int(stacked.shape[0] * stacked.shape[1]),
        "sequential_chain_wall_seconds": total_wall,
        "warmup_seconds_total": sum(float(row["warmup_seconds"] or 0.0) for row in chain_metrics),
        "sampling_seconds_total": sum(float(row["sampling_seconds"] or 0.0) for row in chain_metrics),
        "pi_unet_inference_evaluations": sum(int(row["pi_unet_inference_evaluations"]) for row in chain_metrics),
        "divergence_count": sum(int(row["divergence_count"]) for row in chain_metrics),
        "acceptance_rate_by_chain": [row["acceptance_rate"] for row in chain_metrics],
        "r_hat_max": float(np.nanmax(rhat)),
        "r_hat_median": float(np.nanmedian(rhat)),
        "r_hat_fail_ge_1_01_count": int(np.sum(rhat >= 1.01)),
        "ess_min": float(np.nanmin(ess)),
        "ess_median": float(np.nanmedian(ess)),
        "peak_gpu_memory_bytes": max(
            [int(row["peak_gpu_memory_bytes"] or 0) for row in chain_metrics], default=0
        ),
        "chain_metrics": chain_metrics,
    }
    write_json(out_dir / "mcmc_metrics.json", metrics)
    return stacked, metrics


def jaccard_high_k(truth: np.ndarray, estimate: np.ndarray, quantile: float = 0.75) -> float:
    threshold = float(np.quantile(truth, quantile))
    true_mask = truth >= threshold
    estimate_mask = estimate >= threshold
    union = np.logical_or(true_mask, estimate_mask).sum()
    return float(np.logical_and(true_mask, estimate_mask).sum() / max(1, union))


def evaluate_method(
    name: str,
    z_samples: np.ndarray,
    target: SharedPosteriorTarget,
    truth_logk: np.ndarray,
    truth_head: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    start = time.perf_counter()
    mean_logk, sd_logk, interval_width = target.summarize_fields(z_samples)
    mean_head, predictive_evaluations = target.predict_heads(z_samples, batch_size=batch_size)
    postprocess_seconds = time.perf_counter() - start
    k_rmse, k_mae, k_r2 = field_metrics(truth_logk, mean_logk)
    h_rmse, h_mae, h_r2 = field_metrics(truth_head, mean_head)
    metrics = {
        "method": name,
        "common_posterior_draws": len(z_samples),
        "lnK_RMSE": k_rmse,
        "lnK_R2": k_r2,
        "lnK_MAE": k_mae,
        "head_RMSE": h_rmse,
        "head_R2": h_r2,
        "head_MAE": h_mae,
        "mean_posterior_SD_lnK": float(np.mean(sd_logk)),
        "median_posterior_SD_lnK": float(np.median(sd_logk)),
        "mean_95_interval_width_lnK": interval_width,
        "high_K_Jaccard": jaccard_high_k(truth_logk, mean_logk),
        "posterior_predictive_candidate_evaluations": predictive_evaluations,
        "postprocess_wall_seconds": postprocess_seconds,
    }
    return metrics, mean_logk, sd_logk, mean_head


def save_comparison_outputs(
    svi_samples: np.ndarray,
    mcmc_samples: np.ndarray,
    svi_metrics: dict[str, Any],
    mcmc_metrics: dict[str, Any],
    target: SharedPosteriorTarget,
    truth_logk: np.ndarray,
    truth_head: np.ndarray,
    common_draws_requested: int,
    posterior_batch_size: int,
    eigenvalues: np.ndarray,
    out_dir: Path,
    seed: int,
) -> None:
    flat_mcmc = mcmc_samples.reshape(-1, mcmc_samples.shape[-1])
    common_n = min(common_draws_requested, len(svi_samples), len(flat_mcmc))
    rng = np.random.default_rng(seed + 9117)
    svi_indices = rng.choice(len(svi_samples), size=common_n, replace=False)
    mcmc_indices = rng.choice(len(flat_mcmc), size=common_n, replace=False)
    svi_common = svi_samples[svi_indices]
    mcmc_common = flat_mcmc[mcmc_indices]

    svi_eval, svi_mean, svi_sd, svi_head = evaluate_method(
        "SVI", svi_common, target, truth_logk, truth_head, posterior_batch_size
    )
    mcmc_eval, mcmc_mean, mcmc_sd, mcmc_head = evaluate_method(
        "MCMC", mcmc_common, target, truth_logk, truth_head, posterior_batch_size
    )
    speedup = float(mcmc_metrics["sequential_chain_wall_seconds"] / svi_metrics["inference_wall_seconds"])
    svi_eval.update(
        {
            "inference_wall_seconds": svi_metrics["inference_wall_seconds"],
            "PI_U_Net_inference_evaluations": svi_metrics["pi_unet_inference_evaluations"],
            "speedup_relative_to_SVI": 1.0,
        }
    )
    mcmc_eval.update(
        {
            "inference_wall_seconds": mcmc_metrics["sequential_chain_wall_seconds"],
            "PI_U_Net_inference_evaluations": mcmc_metrics["pi_unet_inference_evaluations"],
            "speedup_relative_to_SVI": speedup,
        }
    )
    summary = pd.DataFrame([svi_eval, mcmc_eval])
    summary.to_csv(out_dir / "comparison_summary.csv", index=False)
    summary.to_csv(out_dir / "figure_source_data" / "comparison_summary.csv", index=False)

    selected_modes = np.asarray([0, len(eigenvalues) // 2, len(eigenvalues) - 1], dtype=np.int64)
    latent_rows: list[dict[str, Any]] = []
    for mode in selected_modes:
        for method, values in (("SVI", svi_common[:, mode]), ("MCMC", mcmc_common[:, mode])):
            for draw_index, value in enumerate(values):
                latent_rows.append(
                    {
                        "method": method,
                        "latent_index_zero_based": int(mode),
                        "latent_mode": int(mode + 1),
                        "draw": draw_index,
                        "value": float(value),
                    }
                )
    pd.DataFrame(latent_rows).to_csv(out_dir / "figure_source_data" / "latent_marginals.csv", index=False)

    latent_summary_rows = []
    for mode in selected_modes:
        for method, values in (("SVI", svi_common[:, mode]), ("MCMC", mcmc_common[:, mode])):
            centered = values - values.mean()
            skewness = float(np.mean(centered**3) / (np.std(values) ** 3 + 1e-12))
            latent_summary_rows.append(
                {
                    "method": method,
                    "latent_index_zero_based": int(mode),
                    "latent_mode": int(mode + 1),
                    "eigenvalue_relative": float(eigenvalues[mode]),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "skewness": skewness,
                }
            )
    pd.DataFrame(latent_summary_rows).to_csv(
        out_dir / "figure_source_data" / "latent_marginal_summary.csv", index=False
    )

    correlation = np.corrcoef(flat_mcmc, rowvar=False).astype(np.float32)
    np.save(out_dir / "figure_source_data" / "mcmc_latent_correlation.npy", correlation)
    off_diagonal = np.abs(correlation.copy())
    np.fill_diagonal(off_diagonal, 0.0)
    flat_indices = np.argsort(off_diagonal.ravel())[::-1]
    strongest = []
    seen: set[tuple[int, int]] = set()
    for flat_index in flat_indices:
        i, j = np.unravel_index(flat_index, off_diagonal.shape)
        pair = tuple(sorted((int(i), int(j))))
        if i == j or pair in seen:
            continue
        seen.add(pair)
        strongest.append({"mode_i": pair[0] + 1, "mode_j": pair[1] + 1, "correlation": float(correlation[pair])})
        if len(strongest) == 20:
            break
    write_json(out_dir / "figure_source_data" / "strongest_mcmc_correlations.json", {"pairs": strongest})

    np.savez_compressed(
        out_dir / "figure_source_data" / "posterior_field_summaries.npz",
        truth_logk=truth_logk.astype(np.float32),
        truth_head=truth_head.astype(np.float32),
        svi_mean_logk=svi_mean,
        svi_sd_logk=svi_sd,
        svi_mean_head=svi_head,
        mcmc_mean_logk=mcmc_mean,
        mcmc_sd_logk=mcmc_sd,
        mcmc_mean_head=mcmc_head,
        selected_modes=selected_modes,
        eigenvalues=eigenvalues.astype(np.float32),
        common_draws=np.asarray([common_n], dtype=np.int64),
    )

    table = pd.DataFrame(
        {
            "Metric": [
                "ln K RMSE",
                "ln K R2",
                "Head RMSE",
                "Head R2",
                "Mean posterior SD (ln K)",
                "Mean 95% interval width (ln K)",
                "Wall-clock time (s)",
                "PI-U-Net inference evaluations",
            ],
            "SVI": [
                svi_eval["lnK_RMSE"],
                svi_eval["lnK_R2"],
                svi_eval["head_RMSE"],
                svi_eval["head_R2"],
                svi_eval["mean_posterior_SD_lnK"],
                svi_eval["mean_95_interval_width_lnK"],
                svi_metrics["inference_wall_seconds"],
                svi_metrics["pi_unet_inference_evaluations"],
            ],
            "MCMC": [
                mcmc_eval["lnK_RMSE"],
                mcmc_eval["lnK_R2"],
                mcmc_eval["head_RMSE"],
                mcmc_eval["head_R2"],
                mcmc_eval["mean_posterior_SD_lnK"],
                mcmc_eval["mean_95_interval_width_lnK"],
                mcmc_metrics["sequential_chain_wall_seconds"],
                mcmc_metrics["pi_unet_inference_evaluations"],
            ],
        }
    )
    table.to_csv(out_dir / "table_main_svi_mcmc.csv", index=False)

    mcmc_metrics.update(
        {
            "posterior_metrics": mcmc_eval,
            "common_posterior_draws": common_n,
            "selected_latent_modes_one_based": (selected_modes + 1).tolist(),
            "strongest_off_diagonal_correlations": strongest,
        }
    )
    svi_metrics.update({"posterior_metrics": svi_eval, "common_posterior_draws": common_n})
    write_json(out_dir / "mcmc_metrics.json", mcmc_metrics)
    write_json(out_dir / "svi_metrics.json", svi_metrics)


def build_readme(out_dir: Path, args: argparse.Namespace, config_checksum: str) -> None:
    command = (
        f"& '{sys.executable}' '{Path(__file__).resolve()}' "
        f"--out-dir '{out_dir}' --case-id {args.case_id} --svi-steps {args.svi_steps} "
        f"--svi-samples {args.svi_samples} --warmup-steps {args.warmup_steps} "
        f"--mcmc-samples-per-chain {args.mcmc_samples_per_chain} --chains {args.chains} "
        f"--common-draws {args.common_draws} --resume"
    )
    figure_command = (
        f"& '{sys.executable}' '{SCRIPTS / 'make_svi_mcmc_benchmark_figures.py'}' --result-dir '{out_dir}'"
    )
    text = f"""# SVI-MCMC benchmark

This directory contains the channelized synthetic benchmark with an identical latent posterior target for SVI and NUTS.

## Configuration checksum

`{config_checksum}`

## Reproduce or resume inference

```powershell
{command}
```

## Regenerate figures only

```powershell
{figure_command}
```

The four NUTS chains are executed sequentially on the same GPU and checkpointed separately after each completed chain. Re-running with `--resume` skips completed SVI and chain artifacts.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled SVI versus NUTS benchmark")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--surrogate-dir", type=Path, default=DEFAULT_SURROGATE)
    parser.add_argument("--inverse-checkpoint", type=Path, default=DEFAULT_INVERSE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-id", type=int, default=160)
    parser.add_argument("--scenario", default="stripe")
    parser.add_argument("--n-k-observations", type=int, default=231)
    parser.add_argument("--n-head-observations", type=int, default=300)
    parser.add_argument("--latent-rank", type=int, default=64)
    parser.add_argument("--kl-inducing", type=int, default=900)
    parser.add_argument("--kl-length-scale", type=float, default=12.0)
    parser.add_argument("--residual-scale", type=float, default=0.18)
    parser.add_argument("--sigma-k", type=float, default=0.20)
    parser.add_argument("--sigma-h", type=float, default=0.30)
    parser.add_argument("--svi-steps", type=int, default=500)
    parser.add_argument("--svi-learning-rate", type=float, default=0.035)
    parser.add_argument("--svi-samples", type=int, default=5000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--mcmc-samples-per-chain", type=int, default=2000)
    parser.add_argument("--target-accept", type=float, default=0.80)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--common-draws", type=int, default=5000)
    parser.add_argument("--posterior-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty; use --resume or choose another directory: {out_dir}")
    for subdir in [out_dir, out_dir / "chain_metrics", out_dir / "figure_source_data"]:
        subdir.mkdir(parents=True, exist_ok=True)

    pyro.enable_validation(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = make_config("journal_highcontrast_train")
    cfg.kl_rank = args.latent_rank
    cfg.kl_n_inducing = args.kl_inducing
    cfg.kl_length_scale = args.kl_length_scale
    cfg.residual_scale = args.residual_scale
    cfg.logk_noise = args.sigma_k
    cfg.head_noise = args.sigma_h
    cfg.seed = args.seed

    dataset = load_dataset_from_dir(args.source_dir)
    ensure_train_normalization_stats(dataset)
    case_index = find_case_index(dataset, args.case_id, args.scenario)
    surrogate_result = load_surrogate(dataset, cfg, args.surrogate_dir)
    if surrogate_result is None:
        raise FileNotFoundError(args.surrogate_dir / "checkpoints" / "mf6_pilot_unet.pt")
    surrogate, surrogate_metrics = surrogate_result
    surrogate.to(device)
    surrogate_stats = load_surrogate_stats(args.surrogate_dir, device)
    inverse_model = load_sparse_inverse_model(args.inverse_checkpoint, device)
    basis, eigenvalues, _ = build_kl_basis(cfg, cfg.kl_rank)

    k_points, head_points = reproduce_matched_observations(
        dataset,
        cfg,
        case_index,
        args.n_k_observations,
        args.n_head_observations,
        args.seed,
    )
    x_template = dataset["x"][case_index]
    truth_logk = dataset["logk"][case_index].astype(np.float32)
    truth_head = dataset["head"][case_index].astype(np.float32)
    prior_mean = sparse_inverse_prior(
        inverse_model,
        x_template,
        truth_head,
        truth_logk,
        k_points,
        head_points,
        dataset["stats"],
        device,
    )

    target = SharedPosteriorTarget(
        surrogate=surrogate,
        x_template=x_template,
        prior_mean_field=prior_mean,
        logk_true=truth_logk,
        head_true=truth_head,
        k_points=k_points,
        head_points=head_points,
        stats=surrogate_stats,
        basis=basis,
        eigenvalues=eigenvalues,
        latent_rank=args.latent_rank,
        residual_scale=args.residual_scale,
        sigma_k=args.sigma_k,
        sigma_h=args.sigma_h,
        device=device,
    )

    shared_hashes = {
        "dataset": sha256_file(args.source_dir / "data" / "mf6_pilot_dataset.npz"),
        "surrogate_checkpoint": sha256_file(args.surrogate_dir / "checkpoints" / "mf6_pilot_unet.pt"),
        "inverse_prior_checkpoint": sha256_file(args.inverse_checkpoint),
        "truth_logk": sha256_array(truth_logk),
        "truth_head": sha256_array(truth_head),
        "prior_mean": sha256_array(prior_mean),
        "basis": sha256_array(basis.cpu().numpy()),
        "eigenvalues": sha256_array(eigenvalues.cpu().numpy()),
        "k_observation_indices": sha256_array(k_points),
        "head_observation_indices": sha256_array(head_points),
        "k_observation_values": sha256_array(truth_logk[k_points[:, 0], k_points[:, 1]]),
        "head_observation_values": sha256_array(truth_head[head_points[:, 0], head_points[:, 1]]),
    }
    target_definition = {
        "case_id": args.case_id,
        "dataset_index": case_index,
        "scenario": args.scenario,
        "n_k_observations": args.n_k_observations,
        "n_head_observations": args.n_head_observations,
        "latent_rank": args.latent_rank,
        "kl_inducing": args.kl_inducing,
        "kl_length_scale": args.kl_length_scale,
        "residual_scale": args.residual_scale,
        "sigma_k": args.sigma_k,
        "sigma_h": args.sigma_h,
        "latent_prior": "z ~ Normal(0, I)",
        "likelihood": "Normal K likelihood plus Normal hydraulic-head likelihood",
        "physics_residual_in_inversion": False,
        "shared_hashes": shared_hashes,
    }
    config_core = {
        "target": target_definition,
        "svi": {
            "steps": args.svi_steps,
            "learning_rate": args.svi_learning_rate,
            "guide": "AutoDiagonalNormal",
            "posterior_samples": args.svi_samples,
        },
        "mcmc": {
            "algorithm": "NUTS",
            "chains": args.chains,
            "warmup_steps_per_chain": args.warmup_steps,
            "retained_samples_per_chain": args.mcmc_samples_per_chain,
            "target_accept": args.target_accept,
            "max_tree_depth": args.max_tree_depth,
            "chain_execution": "sequential on one GPU",
        },
        "common_analysis_draws_requested": args.common_draws,
        "seed": args.seed,
        "hardware": hardware_info(device),
        "surrogate_metrics": surrogate_metrics,
        "source_paths": {
            "source_dir": args.source_dir,
            "surrogate_dir": args.surrogate_dir,
            "inverse_checkpoint": args.inverse_checkpoint,
        },
    }
    checksum_payload = json.dumps(json_ready(config_core), sort_keys=True).encode("utf-8")
    config_checksum = hashlib.sha256(checksum_payload).hexdigest()
    benchmark_config = {
        **config_core,
        "config_checksum": config_checksum,
        "svi_target_checksum": hashlib.sha256(json.dumps(target_definition, sort_keys=True).encode()).hexdigest(),
        "mcmc_target_checksum": hashlib.sha256(json.dumps(target_definition, sort_keys=True).encode()).hexdigest(),
        "target_identity_verified": True,
    }
    assert benchmark_config["svi_target_checksum"] == benchmark_config["mcmc_target_checksum"]
    write_json(out_dir / "benchmark_config.json", benchmark_config)
    np.savez_compressed(
        out_dir / "figure_source_data" / "benchmark_inputs.npz",
        truth_logk=truth_logk,
        truth_head=truth_head,
        prior_mean=prior_mean,
        k_points=k_points,
        head_points=head_points,
        k_values=truth_logk[k_points[:, 0], k_points[:, 1]],
        head_values=truth_head[head_points[:, 0], head_points[:, 1]],
        basis=basis.cpu().numpy(),
        eigenvalues=eigenvalues.cpu().numpy(),
    )
    build_readme(out_dir, args, config_checksum)
    print(f"Configuration checksum: {config_checksum}", flush=True)
    print("SVI and MCMC target checksums are identical.", flush=True)

    svi_sample_path = out_dir / "posterior_samples_svi.npz"
    svi_metric_path = out_dir / "svi_metrics.json"
    if args.resume and svi_sample_path.exists() and svi_metric_path.exists():
        print("Reusing completed SVI artifacts.", flush=True)
        svi_samples = np.load(svi_sample_path)["z"].astype(np.float32)
        svi_metrics = json.loads(svi_metric_path.read_text(encoding="utf-8"))
    else:
        svi_samples, svi_metrics = run_svi(
            target,
            steps=args.svi_steps,
            sample_count=args.svi_samples,
            learning_rate=args.svi_learning_rate,
            seed=args.seed + 100,
            out_dir=out_dir,
        )

    chain_samples: list[np.ndarray] = []
    chain_metrics: list[dict[str, Any]] = []
    for chain_index in range(args.chains):
        samples, metrics = run_nuts_chain(
            target,
            chain_index=chain_index,
            warmup_steps=args.warmup_steps,
            retained_samples=args.mcmc_samples_per_chain,
            target_accept=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            seed=args.seed + 1000 + chain_index * 101,
            out_dir=out_dir,
        )
        chain_samples.append(samples)
        chain_metrics.append(metrics)

    mcmc_samples, mcmc_metrics = aggregate_mcmc(chain_samples, chain_metrics, out_dir)
    save_comparison_outputs(
        svi_samples=svi_samples,
        mcmc_samples=mcmc_samples,
        svi_metrics=svi_metrics,
        mcmc_metrics=mcmc_metrics,
        target=target,
        truth_logk=truth_logk,
        truth_head=truth_head,
        common_draws_requested=args.common_draws,
        posterior_batch_size=args.posterior_batch_size,
        eigenvalues=eigenvalues.cpu().numpy(),
        out_dir=out_dir,
        seed=args.seed,
    )

    if not args.skip_figures:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "make_svi_mcmc_benchmark_figures.py"),
                "--result-dir",
                str(out_dir),
            ],
            check=True,
        )
    print(f"Benchmark completed: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
