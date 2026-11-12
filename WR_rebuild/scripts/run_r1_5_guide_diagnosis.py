"""Matched diagonal/full-covariance guide diagnosis, with exact K-only reference.

Same observation locations, IDW mean, basis and likelihood as R1-4 for the
first retained case of each scenario. Gaussian lnK moments are analytic;
head means use draws. No added observation noise: scales are likelihood SDs.
"""
from pathlib import Path
import argparse
import csv
import json
import time

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer.autoguide.initialization import init_to_value
from pyro.optim import ClippedAdam

from run_mf6_pilot_experiments import (build_kl_basis, choose_obs, field_metrics,
    idw_interpolate, load_dataset_from_dir, load_surrogate, make_config)
from run_svi_sparse_head_prior_experiment import load_surrogate_stats
from run_svi_mcmc_benchmark import SharedPosteriorTarget

ROOT = Path(__file__).resolve().parents[2]


def moments(target, mu, cov):
    A = (target.basis * target.coefficient_scale[:, None, None]).reshape(target.latent_rank, -1).T
    mean = target.prior_mean.flatten() + A @ mu
    var = torch.sum((A @ cov) * A, dim=1).clamp_min(0)
    sd = var.sqrt()
    truth = target.logk_true.flatten()
    error = (mean-truth).abs()
    corr = torch.corrcoef(torch.stack((sd, error)))[0, 1]
    metrics = field_metrics(truth.cpu().numpy(), mean.cpu().numpy())
    zsd = cov.diag().sqrt()
    corr_z = cov / (zsd[:, None] * zsd[None, :])
    off = ~torch.eye(len(mu), dtype=torch.bool, device=mu.device)
    return dict(logk_rmse=float(metrics[0]), logk_r2=float(metrics[2]),
        sd_mean=float(sd.mean()), width95_mean=float(3.919927969*sd.mean()),
        coverage95_fraction=float((error <= 1.9599639845*sd).float().mean()),
        spread_error_r=float(corr), latent_cov_trace=float(cov.trace()),
        latent_abs_corr_offdiag_mean=float(corr_z[off].abs().mean()),
        latent_abs_corr_offdiag_max=float(corr_z[off].abs().max())), mean, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=ROOT/'R1_1_revision_20260926/guide_diagnosis')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', default='11,22,33')
    ap.add_argument('--head-draws', type=int, default=256)
    args = ap.parse_args()
    out=args.out_dir; out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg=make_config('journal_highcontrast_train')
    cfg.seed=20260926; cfg.kl_rank=64; cfg.residual_scale=.18
    cfg.head_noise=.30; cfg.logk_noise=.20
    source=ROOT/'WR_rebuild/outputs/mf6_highcontrast_physics_surrogate'
    data=load_dataset_from_dir(source)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net,_=load_surrogate(data,cfg,source); net.to(device)
    stats=load_surrogate_stats(source,device)
    basis,vals,_=build_kl_basis(cfg,64)
    rows=[]; refs=[]; seed_values=[int(s) for s in args.seeds.split(',')]
    protocol=dict(steps=args.steps,seeds=seed_values,rank=64,residual_scale=.18,
        sigma_logk=.20,sigma_head_m=.30,observation_seed=cfg.seed,
        prior='same IDW from 231 K values; data reused in K likelihood',
        observation_values='truth at observation locations; no noise realization added',
        guide_init='mean zero, scale 0.1, full-rank factor identity',
        optimizer='ClippedAdam, lr 0.035 decaying geometrically to 0.0035, clip_norm 10',
        head_draws=args.head_draws,field_moments='exact affine Gaussian propagation',
        device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        python_torch=str(torch.__version__),pyro=pyro.__version__,source=str(source),
        scope='3 retained fields x 3 optimization seeds; no high-fidelity posterior claim')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    start=time.time()
    for scenario_pos,scenario in enumerate(cfg.scenarios):
        ids=np.where((data['split']=='eval')&(data['scenario'].astype(str)==scenario))[0]
        idx=int(ids[0]); cid=int(data['case_id'][idx])
        # R1-4 full run case positions 0, 5, 10.
        rng=np.random.default_rng(cfg.seed+scenario_pos*5*1009)
        kp=choose_obs(cfg,rng,231); hp=choose_obs(cfg,rng,900)[:300]
        truth=data['logk'][idx]
        prior=idw_interpolate(kp,truth[kp[:,0],kp[:,1]],cfg)
        target=SharedPosteriorTarget(net,data['x'][idx],prior,truth,data['head'][idx],kp,hp,
            stats,basis,vals,64,.18,.20,.30,device)
        A=(target.basis*target.coefficient_scale[:,None,None]).reshape(64,-1).T
        Ak=A[(kp[:,0]*cfg.ncol+kp[:,1])].double()
        delta=(target.k_observed-target.prior_mean[target.k_rows,target.k_cols]).double()
        precision=torch.eye(64,device=device,dtype=torch.float64)+Ak.T@Ak/.20**2
        exact_cov=torch.linalg.inv(precision).float()
        exact_mu=(torch.linalg.solve(precision,Ak.T@delta/.20**2)).float()
        diag_opt_cov=torch.diag(1/precision.diag()).float()
        for name,cov in [('exact_k_only',exact_cov),('optimal_diagonal_k_only',diag_opt_cov)]:
            with torch.no_grad():
                row,m,sd=moments(target,exact_mu,cov)
            row.update(case_id=cid,scenario=scenario,reference=name); refs.append(row)
            np.savez_compressed(out/f'case{cid}_{name}.npz',mu=exact_mu.cpu().numpy(),
                covariance=cov.cpu().numpy(),mean=m.cpu().numpy(),sd=sd.cpu().numpy())
        np.savez_compressed(out/f'case{cid}_inputs.npz',k_points=kp,head_points=hp,prior=prior,
            truth_logk=truth,truth_head=data['head'][idx],basis=basis.numpy(),eigenvalues=vals.numpy())
        for seed in seed_values:
            for use_head in [False,True]:
                def model():
                    z=pyro.sample('z',dist.Normal(torch.zeros(64,device=device),torch.ones(64,device=device)).to_event(1))
                    if use_head:
                        logk,head=target.forward_from_z(z)
                    else:
                        logk=target.logk_from_z(z)
                    pyro.sample('k_obs',dist.Normal(logk[target.k_rows,target.k_cols],.20).to_event(1),obs=target.k_observed)
                    if use_head:
                        pyro.sample('h_obs',dist.Normal(head[target.h_rows,target.h_cols],.30).to_event(1),obs=target.h_observed)
                for family in ['diagonal','fullrank']:
                    pyro.clear_param_store(); pyro.set_rng_seed(seed)
                    klass=AutoDiagonalNormal if family=='diagonal' else AutoMultivariateNormal
                    guide=klass(model,init_loc_fn=init_to_value(values={'z':torch.zeros(64,device=device)}),init_scale=.1)
                    opt=ClippedAdam({'lr':.035,'lrd':.1**(1/args.steps),'clip_norm':10.})
                    svi=SVI(model,guide,opt,Trace_ELBO())
                    losses=[]; snapshots=[]; t=time.time()
                    for step in range(1,args.steps+1):
                        loss=float(svi.step())
                        if not np.isfinite(loss): raise RuntimeError(f'Nonfinite ELBO case {cid}, {family}, {step}')
                        losses.append(loss)
                        if step in {500,1000,args.steps}:
                            with torch.no_grad():
                                q=guide.get_posterior()
                                mu=q.mean.detach()
                                cov=torch.diag(q.variance) if family=='diagonal' else q.covariance_matrix.detach()
                                row,mean,sd=moments(target,mu,cov)
                                row.update(case_id=cid,scenario=scenario,seed=seed,head_n=300 if use_head else 0,
                                    guide=family,step=step,seconds=time.time()-t,
                                    loss_last100_mean=float(np.mean(losses[-100:])))
                                if not use_head:
                                    refcov=diag_opt_cov if family=='diagonal' else exact_cov
                                    row['cov_relative_error_to_family_optimum']=float(torch.linalg.norm(cov-refcov)/torch.linalg.norm(refcov))
                                    row['mean_rmse_to_exact']=float(torch.mean((mu-exact_mu)**2).sqrt())
                                else:
                                    row['cov_relative_error_to_family_optimum']=None
                                    row['mean_rmse_to_exact']=None
                                snapshots.append(row)
                    with torch.no_grad():
                        # Common draw seed for comparisons; analytic lnK summaries need no draws.
                        torch.manual_seed(seed+12345)
                        z=q.sample((args.head_draws,)).cpu().numpy()
                        hmean,_=target.predict_heads(z,batch_size=8)
                        row['head_rmse_m']=float(field_metrics(data['head'][idx],hmean)[0])
                    stem=f'case{cid}_seed{seed}_h{row["head_n"]}_{family}'
                    np.savez_compressed(out/f'{stem}.npz',mu=mu.cpu().numpy(),covariance=cov.cpu().numpy(),
                        mean=mean.cpu().numpy(),sd=sd.cpu().numpy(),head_mean=hmean,losses=np.array(losses))
                    rows.extend(snapshots)
                    # Save after each completed run, with no inferred resume behavior.
                    (out/'runs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                    print(f'{stem}: sd={row["sd_mean"]:.5f} r={row["spread_error_r"]:.4f} rmse={row["logk_rmse"]:.4f} elapsed={time.time()-start:.1f}s',flush=True)
    (out/'analytic_references.json').write_text(json.dumps(refs,indent=2),encoding='utf-8')
    fields=sorted({k for row in rows for k in row})
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (out/'completion.json').write_text(json.dumps({'completed_runs':len(rows)//3,
        'rows':len(rows),'elapsed_seconds':time.time()-start},indent=2),encoding='utf-8')
    print('COMPLETED',flush=True)


if __name__=='__main__': main()
