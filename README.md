# Physics guided variational Bayesian groundwater inversion

Synthetic data, model checkpoints, and executable code accompanying “Beyond Hydraulic-Head Fitting: Physics-Guided Variational Bayesian Inversion of Heterogeneous Conductivity Fields”.

The workflow generates heterogeneous conductivity fields, solves the groundwater flow equation with MODFLOW 6, trains a differentiable physics-guided U-Net, and estimates latent conductivity coefficients with stochastic variational inference. A shared-posterior NUTS benchmark supports comparison of posterior means, spread, and coefficient dependence.

## Installation

Use Python 3.11 and install the Python dependencies.

```bash
python -m pip install -r requirements.txt
```

The archived GPU experiment used PyTorch 2.9.1 with CUDA 12.8 and an NVIDIA GeForce RTX 5060. Install the appropriate PyTorch build for your hardware from [PyTorch](https://pytorch.org/get-started/locally/). The code also supports CPU execution.

Data generation requires MODFLOW 6. The study used version 6.7.0. Download the executable for your operating system from the [official MODFLOW 6 releases](https://github.com/MODFLOW-USGS/modflow6/releases). Supply its local path through `--mf6`.

## Quick example

This example generates six 16 × 16 cases, trains for one epoch, and performs three SVI updates. It checks the simulation, training, and inference entry points.

```bash
python examples/quickstart.py --mf6 /path/to/mf6 --out-dir runs/quickstart
```

## Released data and checkpoints

The dataset contains 165 fields on a 150 × 150 grid, covering smooth, mixed-scale, and channelized heterogeneity. Each scenario has 50 training fields and five model-selection/evaluation fields. The grid spacing is 40 m and the confined layer thickness is 100 m.

| Path | Contents |
| --- | --- |
| `WR_rebuild/outputs/mf6_highcontrast_physics_surrogate/data/` | Complete conductivity fields, six-channel model inputs, MODFLOW head solutions, case identifiers, split labels, and normalization metadata |
| `WR_rebuild/outputs/mf6_highcontrast_physics_surrogate/checkpoints/` | Trained forward U-Net |
| `WR_rebuild/outputs/sparse_head_inverse_highcontrast_k231_h300_e60/` | Trained sparse-observation prior-mean estimator and training records |
| `WR_rebuild/outputs/svi_sparse_head_highcontrast_k231_h300_aligned/` | Reconstruction metrics and field arrays for the three illustrated cases |
| `WR_rebuild/results/svi_mcmc_benchmark/` | SVI and NUTS latent samples, four individual NUTS chains, observation and basis arrays, posterior field summaries, timing, and diagnostics |
| `configs/synthetic_forward.json` | Synthetic simulation and forward-training settings |

The NPZ dataset stores `x` with shape `(165, 6, 150, 150)`, `head` and `logk` with shape `(165, 150, 150)`, and one-dimensional `scenario`, `split`, and `case_id` arrays. The six input channels contain ln K, the source/sink field, left boundary head, right boundary head, left boundary mask, and right boundary mask. Hydraulic conductivity uses m/day, head uses m, and `logk` is its natural logarithm. `split="eval"` identifies the model-selection/evaluation cases used during checkpoint selection.

## Generate synthetic data

```bash
python WR_rebuild/scripts/generate_synthetic_data.py --mf6 /path/to/mf6 --out-dir runs/generated_data
```

This saves the complete generated NPZ dataset, normalization metadata, configuration, and individual MODFLOW workspaces. The generation routine is `generate_dataset` in `run_mf6_pilot_experiments.py`.

## Train the forward surrogate

```bash
python WR_rebuild/scripts/train_synthetic_surrogate.py --data-dir WR_rebuild/outputs/mf6_highcontrast_physics_surrogate --out-dir runs/trained_forward
```

The released configuration specifies 220 epochs, batch size 3, learning rate 0.0008, `lambda_pde=0.00001`, `lambda_bc=0.2`, and a 120-epoch physical-penalty warmup. After warmup, the loss is `MSE + lambda_pde * (finite_volume_residual + lambda_bc * boundary_loss)`. Training writes its checkpoint and loss history into the output directory. To train on newly generated cases, set `--data-dir runs/generated_data`.

## Train the synthetic prior-mean estimator

```bash
python WR_rebuild/scripts/run_sparse_head_inverse_unet_experiment.py --source-dir WR_rebuild/outputs/mf6_highcontrast_physics_surrogate --out-dir runs/trained_prior --epochs 60 --k-obs-count 231 --head-obs-count 300 --augment 5
```

The estimator supplies the observation-informed mean around which the synthetic KLE-derived conductivity perturbation is inferred. The released SVI–NUTS benchmark uses 64 latent coefficients, 231 conductivity observations, and 300 head observations.

Inference reads the normalization constants stored with the released dataset to reproduce the archived prior-mean estimate. Forward retraining fits its normalization on the training fields. Version 1.0.1 makes the dataset loader preserve the saved normalization and applies training-statistic estimation during forward training.

## Run the SVI–NUTS benchmark

```bash
python WR_rebuild/scripts/run_svi_mcmc_benchmark.py --out-dir runs/svi_nuts --resume
```

The command uses the supplied data and checkpoints. Its defaults reproduce the archived benchmark configuration with 500 SVI updates, four sequential NUTS chains, 1,000 warmup steps and 2,000 retained draws per chain, and 5,000 draws for common posterior analysis. Re-running with `--resume` reuses completed chain results. To use retrained models, supply `--surrogate-dir runs/trained_forward` and `--inverse-checkpoint runs/trained_prior/checkpoints/sparse_head_inverse_unet.pt`.

The archived channelized-case results are available directly in `table_main_svi_mcmc.csv`.

| Quantity | SVI | NUTS |
| --- | --- | --- |
| Posterior-mean ln K RMSE | 0.733502 | 0.732628 |
| Mean posterior SD of ln K | 0.084707 | 0.088299 |
| Inference wall time in seconds | 7.02494 | 974.47903 |

These timings give an approximately 139-fold inference speedup on the archived hardware. Offline data generation and training are accounted for separately from this inference comparison.

Regenerate the benchmark figures from the supplied outputs.

```bash
python WR_rebuild/scripts/make_svi_mcmc_benchmark_figures.py --result-dir WR_rebuild/results/svi_mcmc_benchmark
```

The scripts directory also contains the original forward-baseline, common-prior head-observation, noise-sensitivity, and variational-guide experiments. Their command-line options are available through `--help`.

## Citation and license

Version 1.0.1 is described in `CITATION.cff`. Cite the versioned release when reusing the code, synthetic data, or checkpoints. The repository is distributed under the MIT license in `LICENSE`.
