"""Run a small data-generation, surrogate-training, and SVI example."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'WR_rebuild/scripts'))
from run_mf6_pilot_experiments import (
    Config, build_kl_basis, choose_obs, generate_dataset,
    idw_interpolate, run_svi_inversion, set_seed, train_surrogate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mf6', required=True, type=Path)
    parser.add_argument('--out-dir', type=Path, default=root / 'runs/quickstart')
    args = parser.parse_args()
    config = Config(nrow=16, ncol=16, n_train_per_scenario=1, n_eval_per_scenario=1,
                    epochs=1, batch_size=2, base_channels=8, kl_rank=4,
                    kl_n_inducing=16, svi_steps=3, posterior_samples=4,
                    head_obs_counts=(8,), k_obs_count=8, lam_pde=1e-5)
    set_seed(config.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    data = generate_dataset(config, args.mf6.resolve(), args.out_dir)
    model, _ = train_surrogate(data, config, args.out_dir)
    index = int(np.flatnonzero(data['split'] == 'eval')[0])
    rng = np.random.default_rng(config.seed)
    k_points = choose_obs(config, rng, config.k_obs_count)
    head_points = choose_obs(config, rng, config.head_obs_counts[0])
    prior = idw_interpolate(k_points, data['logk'][index][k_points[:, 0], k_points[:, 1]], config)
    basis, values, _ = build_kl_basis(config, config.kl_rank)
    result = run_svi_inversion(
        model, data['x'][index], data['logk'][index], prior, data['head'][index],
        head_points, k_points, config, data['stats'], basis, values,
        next(model.parameters()).device, config.seed, True,
    )
    assert np.isfinite(result['logk_mean']).all()
    assert np.isfinite(result['head_mean']).all()
    np.savez_compressed(args.out_dir / 'posterior_summary.npz',
                        logk_mean=result['logk_mean'], logk_sd=result['logk_std'],
                        head_mean=result['head_mean'])
    print(f'Completed six MODFLOW simulations, one training epoch, and three SVI updates in {args.out_dir}')


if __name__ == '__main__':
    main()
