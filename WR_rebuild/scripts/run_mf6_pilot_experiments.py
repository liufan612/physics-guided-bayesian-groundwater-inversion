from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import flopy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import scipy.linalg
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MF6 = ROOT / "mf6.7.0_win64" / "bin" / "mf6.exe"


@dataclass
class Config:
    nrow: int = 32
    ncol: int = 32
    delr: float = 40.0
    delc: float = 40.0
    n_train_per_scenario: int = 12
    n_eval_per_scenario: int = 1
    seed: int = 20260625
    epochs: int = 90
    batch_size: int = 12
    lr: float = 1e-3
    lam_pde: float = 0.0
    lam_bc: float = 0.10
    pde_warmup_epochs: int = 0
    base_channels: int = 24
    residual_scale: float = 0.7
    kl_rank: int = 48
    kl_n_inducing: int = 256
    kl_length_scale: float = 12.0
    svi_steps: int = 250
    posterior_samples: int = 40
    map_lr: float = 0.04
    head_obs_counts: tuple[int, ...] = (20, 80)
    k_obs_count: int = 80
    head_noise: float = 0.10
    logk_noise: float = 0.30
    right_head: float = 1.0
    recharge_rate: float = 1.0e-5
    use_wells: bool = False
    n_wells_min: int = 4
    n_wells_max: int = 6
    well_flux_abs_min: float = 20.0
    well_flux_abs_max: float = 80.0
    n_inversion_stresses: int = 1
    logk_contrast: float = 1.0
    scenarios: tuple[str, ...] = ("smooth", "mixed", "stripe")


def make_config(profile: str) -> Config:
    if profile == "smoke":
        return Config()
    if profile == "journal":
        return Config(
            nrow=150,
            ncol=150,
            delr=40.0,
            delc=40.0,
            n_train_per_scenario=30,
            n_eval_per_scenario=3,
            epochs=220,
            batch_size=3,
            lr=8e-4,
            lam_pde=0.05,
            base_channels=32,
            kl_rank=96,
            kl_n_inducing=900,
            svi_steps=900,
            posterior_samples=60,
            head_obs_counts=(40, 160, 300),
            k_obs_count=231,
        )
    if profile == "journal_quick":
        return Config(
            nrow=150,
            ncol=150,
            delr=40.0,
            delc=40.0,
            n_train_per_scenario=10,
            n_eval_per_scenario=1,
            epochs=90,
            batch_size=2,
            lr=8e-4,
            lam_pde=0.02,
            base_channels=24,
            kl_rank=64,
            kl_n_inducing=500,
            svi_steps=300,
            posterior_samples=30,
            head_obs_counts=(40, 300),
            k_obs_count=231,
        )
    if profile == "journal_train":
        return Config(
            nrow=150,
            ncol=150,
            delr=40.0,
            delc=40.0,
            n_train_per_scenario=50,
            n_eval_per_scenario=5,
            epochs=300,
            batch_size=3,
            lr=8e-4,
            lam_pde=0.0,
            base_channels=32,
            kl_rank=96,
            kl_n_inducing=900,
            svi_steps=900,
            posterior_samples=60,
            head_obs_counts=(40, 160, 300),
            k_obs_count=231,
        )
    if profile == "journal_identifiable_train":
        return Config(
            nrow=150,
            ncol=150,
            delr=40.0,
            delc=40.0,
            n_train_per_scenario=50,
            n_eval_per_scenario=5,
            epochs=320,
            batch_size=3,
            lr=8e-4,
            lam_pde=0.0,
            base_channels=32,
            kl_rank=96,
            kl_n_inducing=900,
            svi_steps=900,
            posterior_samples=60,
            head_obs_counts=(40, 160, 300),
            k_obs_count=231,
            recharge_rate=1.0e-5,
            use_wells=True,
        )
    if profile == "journal_highcontrast_train":
        return Config(
            nrow=150,
            ncol=150,
            delr=40.0,
            delc=40.0,
            n_train_per_scenario=50,
            n_eval_per_scenario=5,
            epochs=320,
            batch_size=3,
            lr=8e-4,
            lam_pde=0.0,
            base_channels=32,
            kl_rank=96,
            kl_n_inducing=900,
            svi_steps=900,
            posterior_samples=60,
            head_obs_counts=(40, 160, 300),
            k_obs_count=231,
            recharge_rate=1.0e-5,
            use_wells=True,
            logk_contrast=1.85,
        )
    raise ValueError(f"Unknown profile: {profile}")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def apply_logk_contrast(logk: np.ndarray, cfg: Config) -> np.ndarray:
    if abs(cfg.logk_contrast - 1.0) < 1e-8:
        return logk
    center = float(np.mean(logk))
    contrasted = center + cfg.logk_contrast * (logk - center)
    return np.clip(contrasted, -4.0, 4.5)


def sample_logk(scenario: str, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    h, w = cfg.nrow, cfg.ncol
    if scenario == "smooth":
        field = rng.normal(size=(h, w))
        field = gaussian_filter(field, sigma=rng.uniform(3.0, 6.0), mode="reflect")
        field = (field - field.mean()) / (field.std() + 1e-8)
        logk = rng.uniform(-0.2, 1.5) + rng.uniform(0.45, 0.95) * field
        return apply_logk_contrast(logk, cfg)

    if scenario == "mixed":
        large = gaussian_filter(rng.normal(size=(h, w)), sigma=rng.uniform(5.0, 8.0), mode="reflect")
        small = gaussian_filter(rng.normal(size=(h, w)), sigma=rng.uniform(1.0, 2.0), mode="reflect")
        field = 0.75 * large / (large.std() + 1e-8) + 0.35 * small / (small.std() + 1e-8)
        field = (field - field.mean()) / (field.std() + 1e-8)
        logk = rng.uniform(-0.1, 1.3) + rng.uniform(0.55, 1.05) * field
        return apply_logk_contrast(logk, cfg)

    if scenario == "stripe":
        base = gaussian_filter(rng.normal(size=(h, w)), sigma=4.0, mode="reflect")
        base = (base - base.mean()) / (base.std() + 1e-8)
        yy, xx = np.mgrid[0:h, 0:w]
        field = 0.45 * base
        for _ in range(rng.integers(1, 4)):
            angle = rng.uniform(-0.9, 0.9)
            center = rng.uniform(0, w)
            width = rng.uniform(1.2, 2.8)
            line_coord = xx * np.cos(angle) + yy * np.sin(angle)
            stripe = np.exp(-0.5 * ((line_coord - center) / width) ** 2)
            field += rng.uniform(1.2, 2.4) * stripe
        field = gaussian_filter(field, sigma=0.55, mode="reflect")
        logk = field - 0.25
        return apply_logk_contrast(logk, cfg)

    raise ValueError(f"Unknown scenario: {scenario}")


def make_left_boundary(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    y = np.linspace(0.0, 1.0, cfg.nrow)
    base = rng.uniform(8.0, 12.0)
    slope = rng.uniform(-1.0, 1.0)
    amp = rng.uniform(0.0, 0.5)
    wave = rng.choice([1, 2, 3])
    return base + slope * (y - 0.5) + amp * np.sin(2 * np.pi * wave * y)


def sample_source_sink_field(cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[tuple[int, int, int], float]], np.ndarray]:
    h, w = cfg.nrow, cfg.ncol
    recharge = np.full((h, w), cfg.recharge_rate, dtype=np.float32)
    zones = gaussian_filter(rng.normal(size=(h, w)), sigma=max(6.0, h / 18.0), mode="reflect")
    zones = (zones - zones.min()) / (zones.max() - zones.min() + 1e-8)
    recharge *= (0.55 + 0.9 * zones).astype(np.float32)
    q_equiv = recharge.copy()
    wells: list[tuple[tuple[int, int, int], float]] = []
    if cfg.use_wells:
        n_wells = int(rng.integers(cfg.n_wells_min, cfg.n_wells_max + 1))
        margin = max(8, h // 12)
        for _ in range(n_wells):
            i = int(rng.integers(margin, h - margin))
            j = int(rng.integers(margin, w - margin))
            sign = -1.0 if rng.random() < 0.65 else 1.0
            flux = sign * float(rng.uniform(cfg.well_flux_abs_min, cfg.well_flux_abs_max))
            wells.append(((0, i, j), flux))
            q_equiv[i, j] += flux / (cfg.delr * cfg.delc)
    return recharge.astype(np.float32), wells, q_equiv.astype(np.float32)


def build_input(logk: np.ndarray, left_bc: np.ndarray, q_field: np.ndarray, cfg: Config) -> np.ndarray:
    h, w = cfg.nrow, cfg.ncol
    left_map = np.tile(left_bc[:, None], (1, w))
    right_map = np.full((h, w), cfg.right_head, dtype=np.float32)
    left_mask = np.zeros((h, w), dtype=np.float32)
    right_mask = np.zeros((h, w), dtype=np.float32)
    left_mask[:, 0] = 1.0
    right_mask[:, -1] = 1.0
    return np.stack([logk, q_field, left_map, right_map, left_mask, right_mask]).astype(np.float32)


def run_mf6_case(
    logk: np.ndarray,
    left_bc: np.ndarray,
    recharge: np.ndarray,
    wells: list[tuple[tuple[int, int, int], float]],
    cfg: Config,
    mf6exe: Path,
    ws: Path,
) -> np.ndarray:
    ensure_empty_dir(ws)
    sim = flopy.mf6.MFSimulation(sim_name="gwf_case", version="mf6", exe_name=str(mf6exe), sim_ws=str(ws))
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)], time_units="DAYS")
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1e-5, inner_dvclose=1e-5)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf_case", save_flows=False)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=cfg.nrow,
        ncol=cfg.ncol,
        delr=cfg.delr,
        delc=cfg.delc,
        top=100.0,
        botm=0.0,
    )
    strt = np.tile(np.linspace(float(left_bc.mean()), cfg.right_head, cfg.ncol), (cfg.nrow, 1))
    flopy.mf6.ModflowGwfic(gwf, strt=strt)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=np.exp(logk)[None, :, :])
    flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)
    if wells:
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wells)
    chd_spd = []
    for i in range(cfg.nrow):
        chd_spd.append(((0, i, 0), float(left_bc[i])))
        chd_spd.append(((0, i, cfg.ncol - 1), cfg.right_head))
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="gwf_case.hds",
        budget_filerecord="gwf_case.cbc",
        saverecord=[("HEAD", "ALL")],
    )
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("MF6 failed:\n" + "\n".join(buff[-20:]))
    hds = flopy.utils.HeadFile(str(ws / "gwf_case.hds"))
    return hds.get_data(kstpkper=(0, 0))[0].astype(np.float32)


def generate_dataset(cfg: Config, mf6exe: Path, out_dir: Path) -> dict[str, np.ndarray]:
    data_dir = out_dir / "data"
    runs_dir = out_dir / "mf6_runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    idx = 0
    for scenario in cfg.scenarios:
        n_total = cfg.n_train_per_scenario + cfg.n_eval_per_scenario
        for j in range(n_total):
            logk = sample_logk(scenario, cfg, rng).astype(np.float32)
            left_bc = make_left_boundary(cfg, rng).astype(np.float32)
            recharge, wells, q_field = sample_source_sink_field(cfg, rng)
            head = run_mf6_case(logk, left_bc, recharge, wells, cfg, mf6exe, runs_dir / f"case_{idx:04d}_{scenario}")
            x = build_input(logk, left_bc, q_field, cfg)
            split = "train" if j < cfg.n_train_per_scenario else "eval"
            rows.append((x, head, logk, scenario, split, idx))
            idx += 1
    x = np.stack([r[0] for r in rows])
    head = np.stack([r[1] for r in rows])
    logk = np.stack([r[2] for r in rows])
    scenario = np.array([r[3] for r in rows])
    split = np.array([r[4] for r in rows])
    case_id = np.array([r[5] for r in rows])
    # Fit all normalization constants on the training cases only.  The
    # model-selection cases remain available for checkpoint selection, but do
    # not influence the representation seen by the optimizer.
    train_mask = split == "train"
    stats = {
        "x_mean": x[train_mask].mean(axis=(0, 2, 3)).tolist(),
        "x_std": (x[train_mask].std(axis=(0, 2, 3)) + 1e-6).tolist(),
        "head_mean": float(head[train_mask].mean()),
        "head_std": float(head[train_mask].std() + 1e-6),
        "logk_mean": float(logk[train_mask].mean()),
        "logk_std": float(logk[train_mask].std() + 1e-6),
        "normalization_fit_split": "train",
    }
    np.savez_compressed(data_dir / "mf6_pilot_dataset.npz", x=x, head=head, logk=logk, scenario=scenario, split=split, case_id=case_id)
    (data_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {"x": x, "head": head, "logk": logk, "scenario": scenario, "split": split, "case_id": case_id, "stats": stats}


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, stats: dict):
        self.x = x.copy()
        self.y = y.copy()
        xm = np.asarray(stats["x_mean"], dtype=np.float32)
        xs = np.asarray(stats["x_std"], dtype=np.float32)
        n_channels = len(xm)
        self.x[:, :n_channels] = (self.x[:, :n_channels] - xm[None, :, None, None]) / xs[None, :, None, None]
        self.y = (self.y - stats["head_mean"]) / stats["head_std"]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx][None, :, :])


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int = 6, base: int = 24):
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


def pde_residual_loss(pred_norm: torch.Tensor, x_norm: torch.Tensor, stats: dict, cfg: Config) -> torch.Tensor:
    head = pred_norm[:, 0] * float(stats["head_std"]) + float(stats["head_mean"])
    x_raw = denormalize_x_tensor(x_norm, stats)
    logk = x_raw[:, 0]
    q_field = x_raw[:, 1]
    transmissivity = torch.exp(torch.clamp(logk, -8.0, 8.0)) * 100.0
    area = cfg.delr * cfg.delc
    residual = q_field * area

    t_e = 2.0 * transmissivity[:, :, :-1] * transmissivity[:, :, 1:] / (
        transmissivity[:, :, :-1] + transmissivity[:, :, 1:] + 1e-8
    )
    c_e = t_e * cfg.delc / cfg.delr
    f_e = c_e * (head[:, :, 1:] - head[:, :, :-1])
    residual = residual.clone()
    residual[:, :, :-1] = residual[:, :, :-1] + f_e
    residual[:, :, 1:] = residual[:, :, 1:] - f_e

    t_s = 2.0 * transmissivity[:, :-1, :] * transmissivity[:, 1:, :] / (
        transmissivity[:, :-1, :] + transmissivity[:, 1:, :] + 1e-8
    )
    c_s = t_s * cfg.delr / cfg.delc
    f_s = c_s * (head[:, 1:, :] - head[:, :-1, :])
    residual[:, :-1, :] = residual[:, :-1, :] + f_s
    residual[:, 1:, :] = residual[:, 1:, :] - f_s

    interior = residual[:, :, 1:-1]
    scale = torch.mean(torch.abs((q_field * area).detach())) + 1.0
    return torch.mean((interior / scale) ** 2)


def boundary_condition_loss(pred_norm: torch.Tensor, x_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    head = pred_norm[:, 0] * float(stats["head_std"]) + float(stats["head_mean"])
    x_raw = denormalize_x_tensor(x_norm, stats)
    left_map = x_raw[:, 2]
    right_map = x_raw[:, 3]
    left_loss = torch.mean((head[:, :, 0] - left_map[:, :, 0]) ** 2)
    right_loss = torch.mean((head[:, :, -1] - right_map[:, :, -1]) ** 2)
    return (left_loss + right_loss) / (float(stats["head_std"]) ** 2 + 1e-8)


def denormalize_x_tensor(x_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    x_mean = torch.tensor(stats["x_mean"], device=x_norm.device, dtype=x_norm.dtype)
    x_std = torch.tensor(stats["x_std"], device=x_norm.device, dtype=x_norm.dtype)
    return x_norm * x_std[None, :, None, None] + x_mean[None, :, None, None]


def ensure_train_normalization_stats(dataset: dict[str, np.ndarray]) -> None:
    x = dataset["x"]
    head = dataset["head"]
    logk = dataset["logk"]
    stats = dataset["stats"]
    if (
        stats.get("normalization_fit_split") == "train"
        and len(stats.get("x_mean", [])) == x.shape[1]
        and len(stats.get("x_std", [])) == x.shape[1]
    ):
        return
    train_mask = dataset["split"] == "train"
    stats["x_mean"] = x[train_mask].mean(axis=(0, 2, 3)).tolist()
    stats["x_std"] = (x[train_mask].std(axis=(0, 2, 3)) + 1e-6).tolist()
    stats["head_mean"] = float(head[train_mask].mean())
    stats["head_std"] = float(head[train_mask].std() + 1e-6)
    stats["logk_mean"] = float(logk[train_mask].mean())
    stats["logk_std"] = float(logk[train_mask].std() + 1e-6)
    stats["normalization_fit_split"] = "train"


def load_surrogate(dataset: dict[str, np.ndarray], cfg: Config, out_dir: Path) -> tuple[UNetSmall, dict] | None:
    ckpt = out_dir / "checkpoints" / "mf6_pilot_unet.pt"
    if not ckpt.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall(base=cfg.base_channels).to(device)
    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model"])
    metrics_path = out_dir / "metrics" / "surrogate_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {"ckpt": str(ckpt), "best_eval_rmse": None}
    return model, metrics


def load_dataset_from_dir(source_dir: Path) -> dict[str, np.ndarray]:
    dataset_path = source_dir / "data" / "mf6_pilot_dataset.npz"
    stats_path = source_dir / "data" / "stats.json"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    loaded = np.load(dataset_path, allow_pickle=True)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    dataset = {k: loaded[k] for k in ["x", "head", "logk", "scenario", "split", "case_id"]}
    dataset["stats"] = stats
    ensure_train_normalization_stats(dataset)
    return dataset


def train_surrogate(dataset: dict[str, np.ndarray], cfg: Config, out_dir: Path) -> tuple[UNetSmall, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_train_normalization_stats(dataset)
    train_mask = dataset["split"] == "train"
    eval_mask = dataset["split"] == "eval"
    ds_train = ArrayDataset(dataset["x"][train_mask], dataset["head"][train_mask], dataset["stats"])
    ds_eval = ArrayDataset(dataset["x"][eval_mask], dataset["head"][eval_mask], dataset["stats"])
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_eval = DataLoader(ds_eval, batch_size=cfg.batch_size, shuffle=False)
    model = UNetSmall(base=cfg.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    hist = []
    best = math.inf
    ckpt = out_dir / "checkpoints" / "mf6_pilot_unet.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x, y in dl_train:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if cfg.lam_pde > 0 and ep > cfg.pde_warmup_epochs:
                physics_loss = pde_residual_loss(pred, x, dataset["stats"], cfg)
                physics_loss = physics_loss + cfg.lam_bc * boundary_condition_loss(pred, x, dataset["stats"])
                loss = loss + cfg.lam_pde * physics_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        eval_rmses = []
        eval_pdes = []
        eval_bcs = []
        with torch.no_grad():
            for x, y in dl_eval:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                pred_raw = pred.cpu().numpy()[:, 0] * dataset["stats"]["head_std"] + dataset["stats"]["head_mean"]
                y_raw = y.cpu().numpy()[:, 0] * dataset["stats"]["head_std"] + dataset["stats"]["head_mean"]
                eval_rmses.extend(np.sqrt(np.mean((pred_raw - y_raw) ** 2, axis=(1, 2))).tolist())
                eval_pdes.append(float(pde_residual_loss(pred, x, dataset["stats"], cfg).cpu().item()))
                eval_bcs.append(float(boundary_condition_loss(pred, x, dataset["stats"]).cpu().item()))
        val_rmse = float(np.mean(eval_rmses))
        hist.append(
            {
                "epoch": ep,
                "train_loss": float(np.mean(losses)),
                "eval_rmse": val_rmse,
                "eval_pde": float(np.mean(eval_pdes)),
                "eval_bc": float(np.mean(eval_bcs)),
            }
        )
        if val_rmse < best:
            best = val_rmse
            torch.save({"model": model.state_dict(), "stats": dataset["stats"], "cfg": asdict(cfg)}, ckpt)
        if ep == 1 or ep % 20 == 0 or ep == cfg.epochs:
            print(f"epoch {ep:04d} train_loss={np.mean(losses):.4e} eval_rmse={val_rmse:.4f} best={best:.4f}")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    metrics_path = out_dir / "metrics" / "surrogate_history.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        writer.writeheader()
        writer.writerows(hist)
    return model, {"best_eval_rmse": best, "ckpt": str(ckpt)}


def normalize_x(x_raw: np.ndarray, stats: dict) -> np.ndarray:
    x = x_raw.copy()
    xm = np.asarray(stats["x_mean"], dtype=np.float32)
    xs = np.asarray(stats["x_std"], dtype=np.float32)
    n_channels = len(xm)
    x[:n_channels] = (x[:n_channels] - xm[:, None, None]) / xs[:, None, None]
    return x


def surrogate_predict(model: UNetSmall, x_raw: np.ndarray, stats: dict, device: torch.device) -> np.ndarray:
    model.eval()
    x = normalize_x(x_raw, stats)
    with torch.no_grad():
        y = model(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0, 0]
    return y * stats["head_std"] + stats["head_mean"]


def idw_interpolate(points: np.ndarray, values: np.ndarray, cfg: Config, power: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0 : cfg.nrow, 0 : cfg.ncol]
    grid = np.column_stack([yy.ravel(), xx.ravel()])
    tree = cKDTree(points)
    k = min(12, len(points))
    dist_, idx = tree.query(grid, k=k)
    if k == 1:
        return values[idx].reshape(cfg.nrow, cfg.ncol)
    dist_ = np.maximum(dist_, 1e-6)
    weights = 1.0 / (dist_**power)
    interp = np.sum(weights * values[idx], axis=1) / np.sum(weights, axis=1)
    return interp.reshape(cfg.nrow, cfg.ncol).astype(np.float32)


def choose_obs(cfg: Config, rng: np.random.Generator, n: int) -> np.ndarray:
    all_pts = np.array([(i, j) for i in range(1, cfg.nrow - 1) for j in range(1, cfg.ncol - 1)], dtype=np.int64)
    idx = rng.choice(len(all_pts), size=n, replace=False)
    return all_pts[idx]


def build_kl_basis(cfg: Config, rank: int, length_scale: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if length_scale is None:
        length_scale = cfg.kl_length_scale
    yy, xx = np.mgrid[0 : cfg.nrow, 0 : cfg.ncol]
    coords = np.column_stack([yy.ravel(), xx.ravel()]).astype(np.float64)
    n = coords.shape[0]
    if n <= 5000:
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
        cov = np.exp(-0.5 * d2 / (length_scale**2)) + 1e-5 * np.eye(n)
        vals, vecs = scipy.linalg.eigh(cov, subset_by_index=[n - rank, n - 1])
        order = np.argsort(vals)[::-1]
        vals = vals[order].astype(np.float32)
        vecs = vecs[:, order].astype(np.float32)
    else:
        rng = np.random.default_rng(cfg.seed + 177)
        m = min(cfg.kl_n_inducing, n)
        inducing_idx = rng.choice(n, size=m, replace=False)
        inducing = coords[inducing_idx]
        d_mm = ((inducing[:, None, :] - inducing[None, :, :]) ** 2).sum(axis=2)
        k_mm = np.exp(-0.5 * d_mm / (length_scale**2)) + 1e-5 * np.eye(m)
        vals_m, vecs_m = scipy.linalg.eigh(k_mm, subset_by_index=[m - rank, m - 1])
        order = np.argsort(vals_m)[::-1]
        vals_m = vals_m[order]
        vecs_m = vecs_m[:, order]
        d_nm = ((coords[:, None, :] - inducing[None, :, :]) ** 2).sum(axis=2)
        k_nm = np.exp(-0.5 * d_nm / (length_scale**2))
        vecs = k_nm @ (vecs_m / np.maximum(vals_m[None, :], 1e-8))
        vecs = vecs / (np.linalg.norm(vecs, axis=0, keepdims=True) + 1e-8)
        vals = (vals_m * (n / m)).astype(np.float32)
        vecs = vecs.astype(np.float32)
    fields = vecs.T.reshape(rank, cfg.nrow, cfg.ncol)
    fields = fields - fields.mean(axis=(1, 2), keepdims=True)
    fields = fields / (fields.std(axis=(1, 2), keepdims=True) + 1e-8)
    vals = vals / (float(vals[0]) + 1e-8)
    return (
        torch.from_numpy(fields.astype(np.float32)),
        torch.from_numpy(vals.astype(np.float32)),
        torch.from_numpy(coords.astype(np.float32)),
    )


def run_svi_inversion(
    model: UNetSmall,
    x_template: np.ndarray,
    logk_true: np.ndarray,
    prior_mean_field: np.ndarray,
    head_true: np.ndarray,
    head_pts: np.ndarray,
    k_pts: np.ndarray,
    cfg: Config,
    stats: dict,
    kl_basis: torch.Tensor,
    kl_vals: torch.Tensor,
    device: torch.device,
    seed: int,
    use_head: bool,
) -> dict:
    pyro.clear_param_store()
    pyro.set_rng_seed(int(seed))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    basis = kl_basis.to(device)
    vals = kl_vals.to(device)
    sqrt_vals = torch.sqrt(torch.clamp(vals, min=1e-8))
    coeff_scale = cfg.residual_scale * sqrt_vals / torch.sqrt(torch.sum(vals) + 1e-8)
    h_idx = (torch.from_numpy(head_pts[:, 0]).long().to(device), torch.from_numpy(head_pts[:, 1]).long().to(device))
    k_idx = (torch.from_numpy(k_pts[:, 0]).long().to(device), torch.from_numpy(k_pts[:, 1]).long().to(device))
    head_obs = torch.from_numpy(head_true[head_pts[:, 0], head_pts[:, 1]]).float().to(device)
    k_obs = torch.from_numpy(logk_true[k_pts[:, 0], k_pts[:, 1]]).float().to(device)
    prior_mean = torch.from_numpy(prior_mean_field).float().to(device)
    x_base = torch.from_numpy(x_template.copy()).float().to(device)
    x_mean = torch.tensor(stats["x_mean"], device=device).float()
    x_std = torch.tensor(stats["x_std"], device=device).float()
    head_mean = torch.tensor(stats["head_mean"], device=device).float()
    head_std = torch.tensor(stats["head_std"], device=device).float()

    def forward_from_z(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logk = prior_mean + torch.sum((coeff_scale * z)[:, None, None] * basis, dim=0)
        x = x_base.clone()
        x[0] = logk
        n_channels = x_mean.shape[0]
        x[:n_channels] = (x[:n_channels] - x_mean[:, None, None]) / x_std[:, None, None]
        head_norm = model(x[None])[0, 0]
        head = head_norm * head_std + head_mean
        return logk, head

    def probabilistic_model():
        z = pyro.sample("z", dist.Normal(torch.zeros(cfg.kl_rank, device=device), torch.ones(cfg.kl_rank, device=device)).to_event(1))
        logk, head = forward_from_z(z)
        pyro.sample("k_obs", dist.Normal(logk[k_idx], cfg.logk_noise).to_event(1), obs=k_obs)
        if use_head:
            pyro.sample("h_obs", dist.Normal(head[h_idx], cfg.head_noise).to_event(1), obs=head_obs)
        pyro.deterministic("logk_field", logk)
        pyro.deterministic("head_field", head)

    guide = AutoDiagonalNormal(probabilistic_model)
    svi = SVI(probabilistic_model, guide, Adam({"lr": 0.035}), loss=Trace_ELBO())
    losses = []
    for step in range(1, cfg.svi_steps + 1):
        losses.append(float(svi.step()))
    predictive = Predictive(probabilistic_model, guide=guide, num_samples=cfg.posterior_samples, return_sites=("logk_field", "head_field"))
    post = predictive()
    logk_samples = post["logk_field"].detach().cpu().numpy()
    head_samples = post["head_field"].detach().cpu().numpy()
    logk_mean_post = logk_samples.mean(axis=0)
    logk_std_post = logk_samples.std(axis=0)
    head_mean_post = head_samples.mean(axis=0)
    return {
        "logk_mean": logk_mean_post,
        "logk_std": logk_std_post,
        "head_mean": head_mean_post,
        "loss_final": losses[-1],
        "loss_initial": losses[0],
    }


def run_map_inversion(
    model: UNetSmall,
    x_template: np.ndarray,
    logk_true: np.ndarray,
    prior_mean_field: np.ndarray,
    head_true: np.ndarray,
    head_pts: np.ndarray,
    k_pts: np.ndarray,
    cfg: Config,
    stats: dict,
    kl_basis: torch.Tensor,
    kl_vals: torch.Tensor,
    device: torch.device,
    seed: int,
    use_head: bool,
    init_z: np.ndarray | None = None,
) -> dict:
    torch.manual_seed(int(seed))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    basis = kl_basis.to(device)
    vals = kl_vals.to(device)
    sqrt_vals = torch.sqrt(torch.clamp(vals, min=1e-8))
    coeff_scale = cfg.residual_scale * sqrt_vals / torch.sqrt(torch.sum(vals) + 1e-8)
    h_idx = (torch.from_numpy(head_pts[:, 0]).long().to(device), torch.from_numpy(head_pts[:, 1]).long().to(device))
    k_idx = (torch.from_numpy(k_pts[:, 0]).long().to(device), torch.from_numpy(k_pts[:, 1]).long().to(device))
    head_obs = torch.from_numpy(head_true[head_pts[:, 0], head_pts[:, 1]]).float().to(device)
    k_obs = torch.from_numpy(logk_true[k_pts[:, 0], k_pts[:, 1]]).float().to(device)
    prior_mean = torch.from_numpy(prior_mean_field).float().to(device)
    x_base = torch.from_numpy(x_template.copy()).float().to(device)
    x_mean = torch.tensor(stats["x_mean"], device=device).float()
    x_std = torch.tensor(stats["x_std"], device=device).float()
    head_mean = torch.tensor(stats["head_mean"], device=device).float()
    head_std = torch.tensor(stats["head_std"], device=device).float()
    if init_z is None:
        z = torch.zeros(cfg.kl_rank, device=device, requires_grad=True)
    else:
        z = torch.from_numpy(init_z.astype(np.float32)).to(device).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=cfg.map_lr)

    def forward_from_z(z_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logk = prior_mean + torch.sum((coeff_scale * z_value)[:, None, None] * basis, dim=0)
        x = x_base.clone()
        x[0] = logk
        n_channels = x_mean.shape[0]
        x[:n_channels] = (x[:n_channels] - x_mean[:, None, None]) / x_std[:, None, None]
        head_norm = model(x[None])[0, 0]
        head = head_norm * head_std + head_mean
        return logk, head

    losses = []
    for _ in range(cfg.svi_steps):
        logk, head = forward_from_z(z)
        k_loss = torch.mean(((logk[k_idx] - k_obs) / cfg.logk_noise) ** 2)
        prior_loss = torch.mean(z**2)
        loss = k_loss + 0.02 * prior_loss
        if use_head:
            h_loss = torch.mean(((head[h_idx] - head_obs) / cfg.head_noise) ** 2)
            loss = loss + h_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        logk, head = forward_from_z(z)
        logk_np = logk.detach().cpu().numpy()
        head_np = head.detach().cpu().numpy()
        basis_np = basis.detach().cpu().numpy()
        sd = np.std((coeff_scale.detach().cpu().numpy())[:, None, None] * basis_np, axis=0)
    return {
        "logk_mean": logk_np,
        "logk_std": sd.astype(np.float32),
        "head_mean": head_np,
        "loss_final": losses[-1],
        "loss_initial": losses[0],
        "z_map": z.detach().cpu().numpy(),
    }


def run_map_multistress_inversion(
    model: UNetSmall,
    x_templates: list[np.ndarray],
    logk_true: np.ndarray,
    prior_mean_field: np.ndarray,
    head_trues: list[np.ndarray],
    head_pts_list: list[np.ndarray],
    k_pts: np.ndarray,
    cfg: Config,
    stats: dict,
    kl_basis: torch.Tensor,
    kl_vals: torch.Tensor,
    device: torch.device,
    seed: int,
    use_head: bool,
    init_z: np.ndarray | None = None,
) -> dict:
    torch.manual_seed(int(seed))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    basis = kl_basis.to(device)
    vals = kl_vals.to(device)
    sqrt_vals = torch.sqrt(torch.clamp(vals, min=1e-8))
    coeff_scale = cfg.residual_scale * sqrt_vals / torch.sqrt(torch.sum(vals) + 1e-8)
    k_idx = (torch.from_numpy(k_pts[:, 0]).long().to(device), torch.from_numpy(k_pts[:, 1]).long().to(device))
    k_obs = torch.from_numpy(logk_true[k_pts[:, 0], k_pts[:, 1]]).float().to(device)
    prior_mean = torch.from_numpy(prior_mean_field).float().to(device)
    x_bases = [torch.from_numpy(x.copy()).float().to(device) for x in x_templates]
    h_indices = [
        (torch.from_numpy(pts[:, 0]).long().to(device), torch.from_numpy(pts[:, 1]).long().to(device))
        for pts in head_pts_list
    ]
    h_obs = [
        torch.from_numpy(head[pts[:, 0], pts[:, 1]]).float().to(device)
        for head, pts in zip(head_trues, head_pts_list)
    ]
    x_mean = torch.tensor(stats["x_mean"], device=device).float()
    x_std = torch.tensor(stats["x_std"], device=device).float()
    head_mean = torch.tensor(stats["head_mean"], device=device).float()
    head_std = torch.tensor(stats["head_std"], device=device).float()
    if init_z is None:
        z = torch.zeros(cfg.kl_rank, device=device, requires_grad=True)
    else:
        z = torch.from_numpy(init_z.astype(np.float32)).to(device).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=cfg.map_lr)

    def logk_from_z(z_value: torch.Tensor) -> torch.Tensor:
        return prior_mean + torch.sum((coeff_scale * z_value)[:, None, None] * basis, dim=0)

    def predict_head(logk: torch.Tensor, x_base: torch.Tensor) -> torch.Tensor:
        x = x_base.clone()
        x[0] = logk
        n_channels = x_mean.shape[0]
        x[:n_channels] = (x[:n_channels] - x_mean[:, None, None]) / x_std[:, None, None]
        return model(x[None])[0, 0] * head_std + head_mean

    losses = []
    for _ in range(cfg.svi_steps):
        logk = logk_from_z(z)
        k_loss = torch.mean(((logk[k_idx] - k_obs) / cfg.logk_noise) ** 2)
        prior_loss = torch.mean(z**2)
        loss = k_loss + 0.02 * prior_loss
        if use_head:
            h_terms = []
            for x_base, h_idx, obs in zip(x_bases, h_indices, h_obs):
                head = predict_head(logk, x_base)
                h_terms.append(torch.mean(((head[h_idx] - obs) / cfg.head_noise) ** 2))
            loss = loss + torch.mean(torch.stack(h_terms))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        logk = logk_from_z(z)
        head0 = predict_head(logk, x_bases[0])
        logk_np = logk.detach().cpu().numpy()
        head_np = head0.detach().cpu().numpy()
        basis_np = basis.detach().cpu().numpy()
        sd = np.std((coeff_scale.detach().cpu().numpy())[:, None, None] * basis_np, axis=0)
    return {
        "logk_mean": logk_np,
        "logk_std": sd.astype(np.float32),
        "head_mean": head_np,
        "loss_final": losses[-1],
        "loss_initial": losses[0],
        "z_map": z.detach().cpu().numpy(),
    }


def finite_volume_residual(logk: torch.Tensor, head: torch.Tensor, q_field: torch.Tensor, cfg: Config) -> torch.Tensor:
    transmissivity = torch.exp(torch.clamp(logk, -8.0, 8.0)) * 100.0
    area = cfg.delr * cfg.delc
    res = q_field * area

    t_e = 2.0 * transmissivity[:, :-1] * transmissivity[:, 1:] / (
        transmissivity[:, :-1] + transmissivity[:, 1:] + 1e-8
    )
    c_e = t_e * cfg.delc / cfg.delr
    f_e = c_e * (head[:, 1:] - head[:, :-1])
    res = res.clone()
    res[:, :-1] = res[:, :-1] + f_e
    res[:, 1:] = res[:, 1:] - f_e

    t_s = 2.0 * transmissivity[:-1, :] * transmissivity[1:, :] / (
        transmissivity[:-1, :] + transmissivity[1:, :] + 1e-8
    )
    c_s = t_s * cfg.delr / cfg.delc
    f_s = c_s * (head[1:, :] - head[:-1, :])
    res[:-1, :] = res[:-1, :] + f_s
    res[1:, :] = res[1:, :] - f_s
    return res[:, 1:-1]


def run_physics_map_inversion(
    x_template: np.ndarray,
    logk_true: np.ndarray,
    prior_mean_field: np.ndarray,
    head_true: np.ndarray,
    head_pts: np.ndarray,
    k_pts: np.ndarray,
    cfg: Config,
    kl_basis: torch.Tensor,
    kl_vals: torch.Tensor,
    device: torch.device,
    seed: int,
    use_head: bool,
    init_z: np.ndarray | None = None,
    init_head: np.ndarray | None = None,
) -> dict:
    torch.manual_seed(int(seed))
    basis = kl_basis.to(device)
    vals = kl_vals.to(device)
    sqrt_vals = torch.sqrt(torch.clamp(vals, min=1e-8))
    coeff_scale = cfg.residual_scale * sqrt_vals / torch.sqrt(torch.sum(vals) + 1e-8)
    h_idx = (torch.from_numpy(head_pts[:, 0]).long().to(device), torch.from_numpy(head_pts[:, 1]).long().to(device))
    k_idx = (torch.from_numpy(k_pts[:, 0]).long().to(device), torch.from_numpy(k_pts[:, 1]).long().to(device))
    head_obs = torch.from_numpy(head_true[head_pts[:, 0], head_pts[:, 1]]).float().to(device)
    k_obs = torch.from_numpy(logk_true[k_pts[:, 0], k_pts[:, 1]]).float().to(device)
    prior_mean = torch.from_numpy(prior_mean_field).float().to(device)
    q_field = torch.from_numpy(x_template[1]).float().to(device)
    left_bc = torch.from_numpy(x_template[2, :, 0]).float().to(device)
    right_bc = torch.full((cfg.nrow,), float(x_template[3, 0, -1]), device=device)

    if init_z is None:
        z = torch.zeros(cfg.kl_rank, device=device, requires_grad=True)
    else:
        z = torch.from_numpy(init_z.astype(np.float32)).to(device).clone().detach().requires_grad_(True)
    if init_head is None:
        alpha = torch.linspace(0.0, 1.0, cfg.ncol, device=device)[None, :]
        head0 = (1.0 - alpha) * left_bc[:, None] + alpha * right_bc[:, None]
    else:
        head0 = torch.from_numpy(init_head.astype(np.float32)).to(device)
    head_var = head0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam(
        [
            {"params": [z], "lr": cfg.map_lr},
            {"params": [head_var], "lr": cfg.map_lr * 0.35},
        ]
    )

    def fields_from_vars() -> tuple[torch.Tensor, torch.Tensor]:
        logk = prior_mean + torch.sum((coeff_scale * z)[:, None, None] * basis, dim=0)
        head = head_var.clone()
        head[:, 0] = left_bc
        head[:, -1] = right_bc
        return logk, head

    losses = []
    res_scale = torch.mean(torch.abs(q_field * cfg.delr * cfg.delc)).detach() + 1.0
    for _ in range(cfg.svi_steps):
        logk, head = fields_from_vars()
        k_loss = torch.mean(((logk[k_idx] - k_obs) / cfg.logk_noise) ** 2)
        prior_loss = torch.mean(z**2)
        residual = finite_volume_residual(logk, head, q_field, cfg)
        pde_loss = torch.mean((residual / res_scale) ** 2)
        smooth_loss = torch.mean((head[:, 1:] - head[:, :-1]) ** 2) + torch.mean((head[1:, :] - head[:-1, :]) ** 2)
        loss = k_loss + 0.02 * prior_loss + 0.30 * pde_loss + 0.001 * smooth_loss
        if use_head:
            h_loss = torch.mean(((head[h_idx] - head_obs) / cfg.head_noise) ** 2)
            loss = loss + h_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        logk, head = fields_from_vars()
        logk_np = logk.detach().cpu().numpy()
        head_np = head.detach().cpu().numpy()
        basis_np = basis.detach().cpu().numpy()
        sd = np.std((coeff_scale.detach().cpu().numpy())[:, None, None] * basis_np, axis=0)
    return {
        "logk_mean": logk_np,
        "logk_std": sd.astype(np.float32),
        "head_mean": head_np,
        "loss_final": losses[-1],
        "loss_initial": losses[0],
        "z_map": z.detach().cpu().numpy(),
    }


def field_metrics(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae = float(np.mean(np.abs(true - pred)))
    r2 = float(r2_score(true.ravel(), pred.ravel()))
    return rmse, mae, r2


def save_case_plot(path: Path, logk_true, svi_konly_logk, svi_multi_logk, svi_std, head_true, svi_head):
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    logk_error = svi_multi_logk - logk_true
    head_error = svi_head - head_true
    k_vmin = float(np.nanpercentile(np.stack([logk_true, svi_konly_logk, svi_multi_logk]), 2))
    k_vmax = float(np.nanpercentile(np.stack([logk_true, svi_konly_logk, svi_multi_logk]), 98))
    h_vmin = float(np.nanpercentile(np.stack([head_true, svi_head]), 2))
    h_vmax = float(np.nanpercentile(np.stack([head_true, svi_head]), 98))
    e_abs = float(np.nanpercentile(np.abs(head_error), 98))

    path.parent.mkdir(parents=True, exist_ok=True)

    def draw_row(fig_path: Path, panels: list[tuple[np.ndarray, str, str, float, float]], figsize: tuple[float, float]) -> None:
        fig, axes = plt.subplots(1, len(panels), figsize=figsize, constrained_layout=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(title, loc="left", pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.018)
            cb.ax.tick_params(labelsize=7, width=0.4, length=2)
        fig.savefig(fig_path, dpi=220)
        fig.savefig(fig_path.with_suffix(".pdf"))
        plt.close(fig)

    k_abs = float(np.nanpercentile(np.abs(logk_error), 98))
    draw_row(
        path,
        [
            (logk_true, "a  True lnK", "viridis", k_vmin, k_vmax),
            (svi_konly_logk, "b  SVI K-only", "viridis", k_vmin, k_vmax),
            (svi_multi_logk, "c  SVI head + K", "viridis", k_vmin, k_vmax),
            (svi_std, "d  Posterior SD", "magma", 0.0, float(np.nanpercentile(svi_std, 98))),
            (logk_error, "e  lnK error", "RdBu_r", -k_abs, k_abs),
        ],
        (12.2, 2.8),
    )
    head_path = path.with_name(path.stem + "_head" + path.suffix)
    draw_row(
        head_path,
        [
            (head_true, "a  MODFLOW head", "YlGnBu", h_vmin, h_vmax),
            (svi_head, "b  Reconstructed head", "YlGnBu", h_vmin, h_vmax),
            (head_error, "c  Head error", "RdBu_r", -e_abs, e_abs),
        ],
        (7.5, 2.8),
    )


def run_experiments(
    dataset: dict[str, np.ndarray],
    model: UNetSmall,
    cfg: Config,
    out_dir: Path,
    mf6exe: Path,
    max_eval_cases_per_scenario: int | None = None,
    inversion_engine: str = "svi",
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rng = np.random.default_rng(cfg.seed + 99)
    kl_basis, kl_vals, _ = build_kl_basis(cfg, cfg.kl_rank)
    eval_indices_all = np.where(dataset["split"] == "eval")[0]
    if max_eval_cases_per_scenario is None:
        eval_indices = eval_indices_all
    else:
        chosen = []
        for scenario in cfg.scenarios:
            sidx = [i for i in eval_indices_all if str(dataset["scenario"][i]) == scenario]
            chosen.extend(sidx[:max_eval_cases_per_scenario])
        eval_indices = np.asarray(chosen, dtype=np.int64)
    rows = []
    for case_num, idx in enumerate(eval_indices):
        scenario = str(dataset["scenario"][idx])
        x_template = dataset["x"][idx]
        logk_true = dataset["logk"][idx]
        head_true = dataset["head"][idx]
        x_templates = [x_template]
        head_trues = [head_true]
        if inversion_engine == "multistress_map" and cfg.n_inversion_stresses > 1:
            for stress_id in range(1, cfg.n_inversion_stresses):
                left_bc_extra = make_left_boundary(cfg, rng).astype(np.float32)
                recharge_extra, wells_extra, q_extra = sample_source_sink_field(cfg, rng)
                stress_dir = out_dir / "mf6_inversion_stresses" / f"case_{int(dataset['case_id'][idx]):04d}_stress_{stress_id:02d}"
                head_extra = run_mf6_case(logk_true, left_bc_extra, recharge_extra, wells_extra, cfg, mf6exe, stress_dir)
                x_templates.append(build_input(logk_true, left_bc_extra, q_extra, cfg))
                head_trues.append(head_extra)
        k_pts = choose_obs(cfg, rng, cfg.k_obs_count)
        k_values = logk_true[k_pts[:, 0], k_pts[:, 1]]
        prior_mean_logk = idw_interpolate(k_pts, k_values, cfg)
        for n_head in cfg.head_obs_counts:
            head_pts = choose_obs(cfg, rng, n_head)
            head_pts_list = [head_pts] + [choose_obs(cfg, rng, n_head) for _ in range(1, len(head_trues))]
            t0 = time.time()
            if inversion_engine == "svi":
                svi_konly = run_svi_inversion(
                    model,
                    x_template,
                    logk_true,
                    prior_mean_logk,
                    head_true,
                    head_pts,
                    k_pts,
                    cfg,
                    dataset["stats"],
                    kl_basis,
                    kl_vals,
                    device,
                    seed=cfg.seed + idx * 10 + n_head + 1,
                    use_head=False,
                )
                svi = run_svi_inversion(
                    model,
                    x_template,
                    logk_true,
                    prior_mean_logk,
                    head_true,
                    head_pts,
                    k_pts,
                    cfg,
                    dataset["stats"],
                    kl_basis,
                    kl_vals,
                    device,
                    seed=cfg.seed + idx * 10 + n_head,
                    use_head=True,
                )
            elif inversion_engine == "map":
                svi_konly = run_map_inversion(
                    model,
                    x_template,
                    logk_true,
                    prior_mean_logk,
                    head_true,
                    head_pts,
                    k_pts,
                    cfg,
                    dataset["stats"],
                    kl_basis,
                    kl_vals,
                    device,
                    seed=cfg.seed + idx * 10 + n_head + 1,
                    use_head=False,
                )
                svi = run_map_inversion(
                    model,
                    x_template,
                    logk_true,
                    prior_mean_logk,
                    head_true,
                    head_pts,
                    k_pts,
                    cfg,
                    dataset["stats"],
                    kl_basis,
                    kl_vals,
                    device,
                    seed=cfg.seed + idx * 10 + n_head,
                    use_head=True,
                    init_z=svi_konly["z_map"],
                )
            elif inversion_engine == "physics_map":
                svi_konly = run_physics_map_inversion(
                    x_template,
                    logk_true,
                    prior_mean_logk,
                    head_true,
                    head_pts,
                    k_pts,
                    cfg,
                    kl_basis,
                    kl_vals,
                    device,
                    seed=cfg.seed + idx * 10 + n_head + 1,
                    use_head=False,
                )
                svi = run_physics_map_inversion(
                    x_template,
                    logk_true,
                    prior_mean_logk,
                    head_true,
                    head_pts,
                    k_pts,
                    cfg,
                    kl_basis,
                    kl_vals,
                    device,
                    seed=cfg.seed + idx * 10 + n_head,
                    use_head=True,
                    init_z=svi_konly["z_map"],
                    init_head=svi_konly["head_mean"],
                )
            elif inversion_engine == "multistress_map":
                svi_konly = run_map_multistress_inversion(
                    model,
                    x_templates,
                    logk_true,
                    prior_mean_logk,
                    head_trues,
                    head_pts_list,
                    k_pts,
                    cfg,
                    dataset["stats"],
                    kl_basis,
                    kl_vals,
                    device,
                    seed=cfg.seed + idx * 10 + n_head + 1,
                    use_head=False,
                )
                svi = run_map_multistress_inversion(
                    model,
                    x_templates,
                    logk_true,
                    prior_mean_logk,
                    head_trues,
                    head_pts_list,
                    k_pts,
                    cfg,
                    dataset["stats"],
                    kl_basis,
                    kl_vals,
                    device,
                    seed=cfg.seed + idx * 10 + n_head,
                    use_head=True,
                    init_z=svi_konly["z_map"],
                )
            else:
                raise ValueError(f"Unknown inversion engine: {inversion_engine}")
            elapsed = time.time() - t0
            konly_k_rmse, konly_k_mae, konly_k_r2 = field_metrics(logk_true, svi_konly["logk_mean"])
            konly_h_rmse, konly_h_mae, konly_h_r2 = field_metrics(head_true, svi_konly["head_mean"])
            svi_k_rmse, svi_k_mae, svi_k_r2 = field_metrics(logk_true, svi["logk_mean"])
            svi_h_rmse, svi_h_mae, svi_h_r2 = field_metrics(head_true, svi["head_mean"])
            rows.append(
                {
                    "case_id": int(dataset["case_id"][idx]),
                    "scenario": scenario,
                    "n_head_obs": n_head,
                    "n_k_obs": cfg.k_obs_count,
                    "method": "svi_k_only",
                    "k_rmse": konly_k_rmse,
                    "k_mae": konly_k_mae,
                    "k_r2": konly_k_r2,
                    "head_rmse": konly_h_rmse,
                    "head_mae": konly_h_mae,
                    "head_r2": konly_h_r2,
                    "posterior_std_mean": float(np.mean(svi_konly["logk_std"])),
                    "elapsed_seconds": np.nan,
                    "loss_initial": svi_konly["loss_initial"],
                    "loss_final": svi_konly["loss_final"],
                }
            )
            row = {
                "case_id": int(dataset["case_id"][idx]),
                "scenario": scenario,
                "n_head_obs": n_head,
                "n_k_obs": cfg.k_obs_count,
                    "method": "svi_multisource",
                "k_rmse": svi_k_rmse,
                "k_mae": svi_k_mae,
                "k_r2": svi_k_r2,
                "head_rmse": svi_h_rmse,
                "head_mae": svi_h_mae,
                "head_r2": svi_h_r2,
                "posterior_std_mean": float(np.mean(svi["logk_std"])),
                "elapsed_seconds": elapsed,
                "loss_initial": svi["loss_initial"],
                "loss_final": svi["loss_final"],
            }
            rows.append(row)
            if n_head == cfg.head_obs_counts[-1] and case_num < 3:
                save_case_plot(
                    out_dir / "figures" / f"case_{int(dataset['case_id'][idx]):04d}_{scenario}.png",
                    logk_true,
                    svi_konly["logk_mean"],
                    svi["logk_mean"],
                    svi["logk_std"],
                    head_true,
                    svi["head_mean"],
                )
            print(
                f"case={int(dataset['case_id'][idx])} {scenario} n_head={n_head} "
                f"K-only K_RMSE={konly_k_rmse:.3f} "
                f"Head+K K_RMSE={svi_k_rmse:.3f} H_RMSE={svi_h_rmse:.3f}"
            )
    return rows


def write_metrics(rows: list[dict], out_dir: Path) -> None:
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (metrics_dir / "pilot_inversion_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in sorted({r["method"] for r in rows}):
        mrows = [r for r in rows if r["method"] == method]
        summary[method] = {}
        for scenario in sorted({r["scenario"] for r in mrows}):
            srows = [r for r in mrows if r["scenario"] == scenario]
            summary[method][scenario] = {
                "k_rmse_mean": float(np.mean([r["k_rmse"] for r in srows])),
                "k_r2_mean": float(np.mean([r["k_r2"] for r in srows])),
                "head_rmse_mean": float(np.mean([r["head_rmse"] for r in srows])),
                "head_r2_mean": float(np.mean([r["head_r2"] for r in srows])),
            }
    (metrics_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["smoke", "journal_quick", "journal_train", "journal_identifiable_train", "journal_highcontrast_train", "journal"],
        default="smoke",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--mf6", default=str(DEFAULT_MF6))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-inversion", action="store_true")
    parser.add_argument("--reuse-surrogate", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-eval-cases-per-scenario", type=int, default=None)
    parser.add_argument("--kl-rank", type=int, default=None)
    parser.add_argument("--kl-n-inducing", type=int, default=None)
    parser.add_argument("--kl-length-scale", type=float, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--lam-pde", type=float, default=None)
    parser.add_argument("--lam-bc", type=float, default=None)
    parser.add_argument("--pde-warmup-epochs", type=int, default=None)
    parser.add_argument("--svi-steps", type=int, default=None)
    parser.add_argument("--posterior-samples", type=int, default=None)
    parser.add_argument("--head-noise", type=float, default=None)
    parser.add_argument("--logk-noise", type=float, default=None)
    parser.add_argument("--head-obs-counts", default=None)
    parser.add_argument("--k-obs-count", type=int, default=None)
    parser.add_argument("--inversion-engine", choices=["svi", "map", "physics_map", "multistress_map"], default="svi")
    parser.add_argument("--n-inversion-stresses", type=int, default=None)
    args = parser.parse_args()
    cfg = make_config(args.profile)
    if args.kl_rank is not None:
        cfg.kl_rank = args.kl_rank
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.kl_n_inducing is not None:
        cfg.kl_n_inducing = args.kl_n_inducing
    if args.kl_length_scale is not None:
        cfg.kl_length_scale = args.kl_length_scale
    if args.residual_scale is not None:
        cfg.residual_scale = args.residual_scale
    if args.lam_pde is not None:
        cfg.lam_pde = args.lam_pde
    if args.lam_bc is not None:
        cfg.lam_bc = args.lam_bc
    if args.pde_warmup_epochs is not None:
        cfg.pde_warmup_epochs = args.pde_warmup_epochs
    if args.svi_steps is not None:
        cfg.svi_steps = args.svi_steps
    if args.posterior_samples is not None:
        cfg.posterior_samples = args.posterior_samples
    if args.head_noise is not None:
        cfg.head_noise = args.head_noise
    if args.logk_noise is not None:
        cfg.logk_noise = args.logk_noise
    if args.head_obs_counts is not None:
        cfg.head_obs_counts = tuple(int(x) for x in args.head_obs_counts.split(",") if x.strip())
    if args.k_obs_count is not None:
        cfg.k_obs_count = args.k_obs_count
    if args.n_inversion_stresses is not None:
        cfg.n_inversion_stresses = args.n_inversion_stresses
    set_seed(cfg.seed)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "WR_rebuild" / "outputs" / f"mf6_{args.profile}"
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    mf6exe = Path(args.mf6)
    if not mf6exe.exists():
        raise FileNotFoundError(mf6exe)
    dataset_path = out_dir / "data" / "mf6_pilot_dataset.npz"
    if source_dir != out_dir and (source_dir / "data" / "mf6_pilot_dataset.npz").exists():
        dataset = load_dataset_from_dir(source_dir)
    elif dataset_path.exists():
        dataset = load_dataset_from_dir(out_dir)
    else:
        dataset = generate_dataset(cfg, mf6exe, out_dir)
    loaded_surrogate = load_surrogate(dataset, cfg, source_dir) if args.reuse_surrogate else None
    if loaded_surrogate is not None:
        model, surrogate_metrics = loaded_surrogate
        print(f"Reused surrogate checkpoint: {surrogate_metrics.get('ckpt')}")
    else:
        model, surrogate_metrics = train_surrogate(dataset, cfg, out_dir)
    if not args.skip_inversion:
        rows = run_experiments(dataset, model, cfg, out_dir, mf6exe, args.max_eval_cases_per_scenario, args.inversion_engine)
        write_metrics(rows, out_dir)
    (out_dir / "metrics" / "surrogate_metrics.json").write_text(json.dumps(surrogate_metrics, indent=2), encoding="utf-8")
    print(f"Done. Outputs saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




