from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "WR_rebuild" / "outputs" / "mf6_highcontrast_physics_surrogate"


def load_dataset(source_dir: Path) -> dict:
    data = np.load(source_dir / "data" / "mf6_pilot_dataset.npz", allow_pickle=True)
    stats = json.loads((source_dir / "data" / "stats.json").read_text(encoding="utf-8"))
    return {**{k: data[k] for k in ["x", "head", "logk", "scenario", "split", "case_id"]}, "stats": stats}


def choose_obs(shape: tuple[int, int], rng: np.random.Generator, n: int) -> np.ndarray:
    h, w = shape
    all_pts = np.array([(i, j) for i in range(1, h - 1) for j in range(1, w - 1)], dtype=np.int64)
    return all_pts[rng.choice(len(all_pts), size=n, replace=False)]


def idw_interpolate(points: np.ndarray, values: np.ndarray, shape: tuple[int, int], power: float = 2.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.column_stack([yy.ravel(), xx.ravel()])
    tree = cKDTree(points)
    k = min(12, len(points))
    dist, idx = tree.query(grid, k=k)
    if k == 1:
        return values[idx].reshape(h, w).astype(np.float32)
    dist = np.maximum(dist, 1e-6)
    weights = 1.0 / (dist**power)
    interp = np.sum(weights * values[idx], axis=1) / np.sum(weights, axis=1)
    return interp.reshape(h, w).astype(np.float32)


def make_sparse_inverse_input(x: np.ndarray, head: np.ndarray, logk: np.ndarray, k_pts: np.ndarray, head_pts: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    h, w = logk.shape
    k_values = logk[k_pts[:, 0], k_pts[:, 1]]
    idw = idw_interpolate(k_pts, k_values, (h, w))
    k_mask = np.zeros((h, w), dtype=np.float32)
    k_obs = np.zeros((h, w), dtype=np.float32)
    k_mask[k_pts[:, 0], k_pts[:, 1]] = 1.0
    k_obs[k_pts[:, 0], k_pts[:, 1]] = k_values
    h_mask = np.zeros((h, w), dtype=np.float32)
    h_obs = np.zeros((h, w), dtype=np.float32)
    h_mask[head_pts[:, 0], head_pts[:, 1]] = 1.0
    h_obs[head_pts[:, 0], head_pts[:, 1]] = head[head_pts[:, 0], head_pts[:, 1]]
    q = (x[1] - stats["x_mean"][1]) / stats["x_std"][1]
    left = (x[2] - stats["x_mean"][2]) / stats["x_std"][2]
    right = (x[3] - stats["x_mean"][3]) / stats["x_std"][3]
    idw_n = (idw - stats["logk_mean"]) / stats["logk_std"]
    k_obs_n = (k_obs - stats["logk_mean"]) / stats["logk_std"] * k_mask
    h_obs_n = (h_obs - stats["head_mean"]) / stats["head_std"] * h_mask
    target = (logk - stats["logk_mean"]) / stats["logk_std"]
    inp = np.stack([idw_n, k_mask, k_obs_n, h_obs_n, h_mask, q, left, right]).astype(np.float32)
    return inp, target.astype(np.float32)


class SparseInverseDataset(Dataset):
    def __init__(self, dataset: dict, indices: np.ndarray, k_obs_count: int, head_obs_count: int, augment: int, seed: int):
        self.dataset = dataset
        self.indices = indices
        self.k_obs_count = k_obs_count
        self.head_obs_count = head_obs_count
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices) * self.augment

    def __getitem__(self, item: int):
        case_pos = item // self.augment
        aug_id = item % self.augment
        idx = int(self.indices[case_pos])
        rng = np.random.default_rng(self.seed + idx * 1009 + aug_id * 9176)
        logk = self.dataset["logk"][idx]
        k_pts = choose_obs(logk.shape, rng, self.k_obs_count)
        head_pts = choose_obs(logk.shape, rng, self.head_obs_count)
        inp, target = make_sparse_inverse_input(self.dataset["x"][idx], self.dataset["head"][idx], logk, k_pts, head_pts, self.dataset["stats"])
        return torch.from_numpy(inp), torch.from_numpy(target[None])


class SparseInverseUNet(nn.Module):
    def __init__(self, in_channels: int = 8, base: int = 32):
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
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU())

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


def field_metrics(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae = float(np.mean(np.abs(true - pred)))
    r2 = float(r2_score(true.ravel(), pred.ravel()))
    return rmse, mae, r2


def save_plot(path: Path, true: np.ndarray, idw: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    err = pred - true
    k_vmin = float(np.nanpercentile(np.stack([true, idw, pred]), 2))
    k_vmax = float(np.nanpercentile(np.stack([true, idw, pred]), 98))
    e_abs = float(np.nanpercentile(np.abs(err), 98))
    panels = [(true, "a  True lnK", "viridis", k_vmin, k_vmax), (idw, "b  K-only IDW", "viridis", k_vmin, k_vmax), (pred, "c  Sparse-head prior", "viridis", k_vmin, k_vmax), (err, "d  lnK error", "RdBu_r", -e_abs, e_abs)]
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 4, figsize=(9.8, 2.7), constrained_layout=True)
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, loc="left", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def evaluate(model: nn.Module, dataset: dict, eval_idx: np.ndarray, k_obs_count: int, head_obs_count: int, seed: int, out_dir: Path | None) -> float:
    device = next(model.parameters()).device
    model.eval()
    rows=[]; rmses=[]; saved=set(); stats=dataset['stats']
    with torch.no_grad():
        for pos, idx in enumerate(eval_idx):
            idx=int(idx); rng=np.random.default_rng(seed+idx*1009+424242)
            logk=dataset['logk'][idx]; k_pts=choose_obs(logk.shape,rng,k_obs_count); h_pts=choose_obs(logk.shape,rng,head_obs_count)
            inp,target=make_sparse_inverse_input(dataset['x'][idx],dataset['head'][idx],logk,k_pts,h_pts,stats)
            pred_n=model(torch.from_numpy(inp[None]).to(device))[0,0].cpu().numpy()
            pred=pred_n*stats['logk_std']+stats['logk_mean']; idw=inp[0]*stats['logk_std']+stats['logk_mean']
            idw_rmse,idw_mae,idw_r2=field_metrics(logk,idw); inv_rmse,inv_mae,inv_r2=field_metrics(logk,pred)
            rmses.append(inv_rmse); scenario=str(dataset['scenario'][idx]); cid=int(dataset['case_id'][idx])
            rows.append({'case_id':cid,'scenario':scenario,'method':'idw_k_only','k_rmse':idw_rmse,'k_mae':idw_mae,'k_r2':idw_r2})
            rows.append({'case_id':cid,'scenario':scenario,'method':'sparse_head_inverse_prior','k_rmse':inv_rmse,'k_mae':inv_mae,'k_r2':inv_r2})
            if out_dir is not None and scenario not in saved:
                save_plot(out_dir/'figures'/f'case_{cid:04d}_{scenario}.png',logk,idw,pred); saved.add(scenario)
    if out_dir is not None:
        (out_dir/'metrics').mkdir(parents=True,exist_ok=True)
        with (out_dir/'metrics'/'sparse_inverse_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
        summary={}
        for method in sorted({r['method'] for r in rows}):
            vals=[r for r in rows if r['method']==method]
            summary[method]={'k_rmse_mean':float(np.mean([r['k_rmse'] for r in vals])),'k_r2_mean':float(np.mean([r['k_r2'] for r in vals]))}
        (out_dir/'metrics'/'sparse_inverse_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return float(np.mean(rmses))


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-dir',default=str(DEFAULT_SOURCE))
    parser.add_argument('--out-dir',default=str(ROOT/'WR_rebuild'/'outputs'/'sparse_head_inverse_highcontrast'))
    parser.add_argument('--k-obs-count',type=int,default=231)
    parser.add_argument('--head-obs-count',type=int,default=300)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--augment',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=3)
    parser.add_argument('--base',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20260625)
    parser.add_argument('--eval-only',action='store_true')
    args=parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    dataset=load_dataset(Path(args.source_dir)); train_idx=np.where(dataset['split']=='train')[0]; eval_idx=np.where(dataset['split']=='eval')[0]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SparseInverseUNet(base=args.base).to(device); ckpt=out_dir/'checkpoints'/'sparse_head_inverse_unet.pt'
    if args.eval_only:
        model.load_state_dict(torch.load(ckpt,map_location=device)['model'])
        print('Eval only', evaluate(model,dataset,eval_idx,args.k_obs_count,args.head_obs_count,args.seed,out_dir)); return 0
    ds_train=SparseInverseDataset(dataset,train_idx,args.k_obs_count,args.head_obs_count,args.augment,args.seed)
    dl_train=DataLoader(ds_train,batch_size=args.batch_size,shuffle=True,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    ckpt.parent.mkdir(parents=True,exist_ok=True); hist=[]; best=math.inf; t0=time.time()
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in dl_train:
            x=x.to(device); y=y.to(device); pred=model(x); loss=F.mse_loss(pred,y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu().item()))
        eval_rmse=evaluate(model,dataset,eval_idx,args.k_obs_count,args.head_obs_count,args.seed,None)
        hist.append({'epoch':ep,'train_loss':float(np.mean(losses)),'eval_rmse':eval_rmse})
        if eval_rmse<best:
            best=eval_rmse; torch.save({'model':model.state_dict(),'args':vars(args)},ckpt)
        if ep==1 or ep%10==0 or ep==args.epochs:
            print(f'epoch={ep:03d} train_loss={np.mean(losses):.4e} eval_k_rmse={eval_rmse:.4f} best={best:.4f}')
    (out_dir/'metrics').mkdir(exist_ok=True)
    with (out_dir/'metrics'/'sparse_inverse_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(hist[0].keys())); writer.writeheader(); writer.writerows(hist)
    model.load_state_dict(torch.load(ckpt,map_location=device)['model'])
    final=evaluate(model,dataset,eval_idx,args.k_obs_count,args.head_obs_count,args.seed,out_dir)
    print(f'Done in {time.time()-t0:.1f}s best={best:.4f} final={final:.4f}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
