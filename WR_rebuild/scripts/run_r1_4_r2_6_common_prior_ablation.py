"""Common-prior head-density ablation for Reviewer 1-4 and Reviewer 2-6.

Both K-only and K+head updates use the same IDW prior constructed only from
the 231 direct-K observations.  The head count is the only changed likelihood
input.  This script is intentionally separate from the archived experiment
entry point, whose head+K branch uses a different observation-informed prior.
"""
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch

from run_mf6_pilot_experiments import (
    build_kl_basis, choose_obs, field_metrics, idw_interpolate,
    load_dataset_from_dir, load_surrogate, make_config, run_svi_inversion,
)
from run_svi_sparse_head_prior_experiment import load_surrogate_stats

ROOT = Path(__file__).resolve().parents[2]

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-dir',default=str(ROOT/'WR_rebuild/outputs/mf6_highcontrast_physics_surrogate'))
    ap.add_argument('--surrogate-dir',default=str(ROOT/'WR_rebuild/outputs/mf6_highcontrast_physics_surrogate'))
    ap.add_argument('--out-dir',default=str(ROOT/'WR_article_final_20260826/R1_1_revision_20260926/common_prior_head_ablation'))
    ap.add_argument('--head-counts',default='0,5,10,20,40,80,160,300,900')
    ap.add_argument('--max-per-scenario',type=int,default=1)
    ap.add_argument('--svi-steps',type=int,default=500)
    ap.add_argument('--posterior-samples',type=int,default=40)
    ap.add_argument('--head-noise',type=float,default=0.30)
    ap.add_argument('--logk-noise',type=float,default=0.20)
    ap.add_argument('--rank',type=int,default=64)
    ap.add_argument('--residual-scale',type=float,default=0.18)
    ap.add_argument('--seed',type=int,default=20260926)
    args=ap.parse_args()
    out=Path(args.out_dir); (out/'metrics').mkdir(parents=True,exist_ok=True)
    dataset=load_dataset_from_dir(Path(args.source_dir))
    cfg=make_config('journal_highcontrast_train')
    cfg.svi_steps=args.svi_steps; cfg.posterior_samples=args.posterior_samples
    cfg.kl_rank=args.rank; cfg.residual_scale=args.residual_scale
    cfg.head_noise=args.head_noise; cfg.logk_noise=args.logk_noise; cfg.seed=args.seed
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    surrogate,_=load_surrogate(dataset,cfg,Path(args.surrogate_dir))
    if surrogate is None: raise FileNotFoundError('surrogate checkpoint not found')
    surrogate.to(device); stats=load_surrogate_stats(Path(args.surrogate_dir),device)
    basis,vals,_=build_kl_basis(cfg,cfg.kl_rank)
    eval_idx=[]
    for scenario in cfg.scenarios:
        ids=np.where((dataset['split']=='eval')&(dataset['scenario'].astype(str)==scenario))[0]
        eval_idx.extend(ids[:args.max_per_scenario].tolist())
    counts=[int(x) for x in args.head_counts.split(',') if x.strip()]
    # One fixed K observation set and one nested head permutation per case.
    rows=[]; started=time.time()
    for case_pos,idx in enumerate(eval_idx):
        rng=np.random.default_rng(args.seed+case_pos*1009)
        k_pts=choose_obs(cfg,rng,231)
        all_head=choose_obs(cfg,rng,900)
        logk=dataset['logk'][idx]; head=dataset['head'][idx]; x=dataset['x'][idx]
        prior=idw_interpolate(k_pts,logk[k_pts[:,0],k_pts[:,1]],cfg)
        for n_h in counts:
            h_pts=all_head[:n_h]
            common=dict(model=surrogate,x_template=x,logk_true=logk,prior_mean_field=prior,
                        head_true=head,head_pts=h_pts,k_pts=k_pts,cfg=cfg,stats=stats,
                        kl_basis=basis,kl_vals=vals,device=device,seed=args.seed+idx*37+n_h,use_head=True)
            t=time.time(); joint=run_svi_inversion(**common); joint_elapsed=time.time()-t
            # The K-only target uses the same prior and the same K observations.
            t=time.time(); konly=run_svi_inversion(**{**common,'seed':args.seed+idx*37+n_h+100000,'use_head':False}); konly_elapsed=time.time()-t
            km=field_metrics(logk,joint['logk_mean']); hm=field_metrics(head,joint['head_mean'])
            km0=field_metrics(logk,konly['logk_mean']); hm0=field_metrics(head,konly['head_mean'])
            for method,res,kmx,hmx,elapsed in [('common_prior_head_k',joint,km,hm,joint_elapsed),('common_prior_k_only',konly,km0,hm0,konly_elapsed)]:
                rows.append({'case_id':int(dataset['case_id'][idx]),'scenario':str(dataset['scenario'][idx]),'n_k_obs':231,'n_head_obs':n_h,'method':method,'prior':'same_idw_k_prior','k_rmse':kmx[0],'k_mae':kmx[1],'k_r2':kmx[2],'head_rmse':hmx[0],'head_mae':hmx[1],'head_r2':hmx[2],'posterior_std_mean':float(res['logk_std'].mean()),'loss_initial':float(res['loss_initial']),'loss_final':float(res['loss_final']),'elapsed_seconds':float(elapsed)})
            print(f'case={dataset["case_id"][idx]} scenario={dataset["scenario"][idx]} h={n_h} joint_rmse={km[0]:.4f} konly_rmse={km0[0]:.4f}',flush=True)
    with (out/'metrics/common_prior_head_ablation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={'protocol':'same IDW prior from 231 K observations; nested head locations; K-only vs K+head','cases':len(eval_idx),'head_counts':counts,'config':vars(args),'device':str(device),'elapsed_seconds':time.time()-started,'rows':len(rows)}
    (out/'metrics/protocol.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
