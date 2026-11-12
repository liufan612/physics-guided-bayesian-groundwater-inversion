from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, norm


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


COLORS = {
    "SVI": "#3F6FB5",
    "MCMC": "#C56A3A",
    "neutral": "#5B6573",
    "chains": ["#355C9A", "#5D8FC1", "#B85C38", "#D99A5E"],
    "modes": ["#355C9A", "#4D927D", "#B85C38"],
}


def add_panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.10,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_main_figure(result_dir: Path) -> None:
    fields = np.load(result_dir / "figure_source_data" / "posterior_field_summaries.npz")
    summary = pd.read_csv(result_dir / "comparison_summary.csv").set_index("method")
    latent = pd.read_csv(result_dir / "figure_source_data" / "latent_marginals.csv")
    latent_summary = pd.read_csv(result_dir / "figure_source_data" / "latent_marginal_summary.csv")

    truth = fields["truth_logk"]
    svi_mean = fields["svi_mean_logk"]
    mcmc_mean = fields["mcmc_mean_logk"]
    svi_sd = fields["svi_sd_logk"]
    mcmc_sd = fields["mcmc_sd_logk"]
    selected_modes = fields["selected_modes"].astype(int)
    scale_labels = ["Large scale", "Intermediate scale", "Fine scale"]

    value_min = float(min(truth.min(), svi_mean.min(), mcmc_mean.min()))
    value_max = float(max(truth.max(), svi_mean.max(), mcmc_mean.max()))
    sd_max = float(max(svi_sd.max(), mcmc_sd.max()))

    fig, axes = plt.subplots(2, 3, figsize=(7.20, 4.95), constrained_layout=True)
    image_specs = [
        (axes[0, 0], truth, "Reference ln K", "a"),
        (axes[0, 1], svi_mean, "SVI posterior mean", "b"),
        (axes[0, 2], mcmc_mean, "MCMC posterior mean", "c"),
    ]
    image_handles = []
    for ax, image, title, letter in image_specs:
        handle = ax.imshow(image, cmap="viridis", vmin=value_min, vmax=value_max, interpolation="nearest")
        image_handles.append(handle)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("#4B4B4B")
        add_panel_label(ax, letter)

    axes[0, 1].text(
        0.03,
        0.04,
        f"RMSE = {summary.loc['SVI', 'lnK_RMSE']:.3f}\n$R^2$ = {summary.loc['SVI', 'lnK_R2']:.3f}",
        transform=axes[0, 1].transAxes,
        fontsize=6.5,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
    )
    axes[0, 2].text(
        0.03,
        0.04,
        f"RMSE = {summary.loc['MCMC', 'lnK_RMSE']:.3f}\n$R^2$ = {summary.loc['MCMC', 'lnK_R2']:.3f}",
        transform=axes[0, 2].transAxes,
        fontsize=6.5,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
    )
    cbar_fields = fig.colorbar(
        image_handles[0], ax=list(axes[0, :]), orientation="horizontal", fraction=0.045, pad=0.04, aspect=50
    )
    cbar_fields.set_label("ln K", fontsize=7)
    cbar_fields.ax.tick_params(labelsize=6.5, length=2)

    sd_handles = []
    for ax, image, title, letter in [
        (axes[1, 0], svi_sd, "SVI posterior SD", "d"),
        (axes[1, 1], mcmc_sd, "MCMC posterior SD", "e"),
    ]:
        handle = ax.imshow(image, cmap="magma_r", vmin=0.0, vmax=sd_max, interpolation="nearest")
        sd_handles.append(handle)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("#4B4B4B")
        add_panel_label(ax, letter)
    cbar_sd = fig.colorbar(
        sd_handles[0], ax=[axes[1, 0], axes[1, 1]], orientation="horizontal", fraction=0.045, pad=0.04, aspect=34
    )
    cbar_sd.set_label("Posterior SD (ln K)", fontsize=7)
    cbar_sd.ax.tick_params(labelsize=6.5, length=2)

    ax = axes[1, 2]
    add_panel_label(ax, "f")
    offsets = np.arange(len(selected_modes))[::-1] * 1.15
    all_values = latent["value"].to_numpy()
    x_min, x_max = np.quantile(all_values, [0.002, 0.998])
    margin = 0.08 * max(1e-6, x_max - x_min)
    x_grid = np.linspace(x_min - margin, x_max + margin, 400)
    for position, mode in enumerate(selected_modes):
        color = COLORS["modes"][position]
        offset = offsets[position]
        mcmc_values = latent[
            (latent["method"] == "MCMC") & (latent["latent_index_zero_based"] == mode)
        ]["value"].to_numpy()
        if np.std(mcmc_values) > 1e-9:
            density = gaussian_kde(mcmc_values)(x_grid)
            density = density / max(density.max(), 1e-12)
        else:
            density = np.zeros_like(x_grid)
        svi_row = latent_summary[
            (latent_summary["method"] == "SVI")
            & (latent_summary["latent_index_zero_based"] == mode)
        ].iloc[0]
        svi_density = norm.pdf(x_grid, loc=svi_row["mean"], scale=max(svi_row["std"], 1e-8))
        svi_density = svi_density / max(svi_density.max(), 1e-12)
        ax.fill_between(x_grid, offset, offset + density, color=color, alpha=0.20, linewidth=0)
        ax.plot(x_grid, offset + density, color=color, linewidth=1.25)
        ax.plot(x_grid, offset + svi_density, color=color, linewidth=1.25, linestyle=(0, (3, 2)))
        ax.axhline(offset, color="#C7CCD1", linewidth=0.55)
        ax.text(
            x_min - margin * 0.65,
            offset + 0.48,
            scale_labels[position],
            color=color,
            fontsize=6.5,
            va="center",
        )
    ax.set_title("Posterior marginals across spatial scales", fontsize=8, pad=4)
    ax.set_xlabel("Latent coefficient z")
    ax.set_ylabel("Normalized density + offset")
    ax.set_yticks([])
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(-0.05, offsets[0] + 1.18)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1.25, label="MCMC density"),
            Line2D([0], [0], color="black", linewidth=1.25, linestyle=(0, (3, 2)), label="SVI Gaussian"),
        ],
        loc="upper right",
        fontsize=6.2,
        handlelength=2.2,
    )
    ax.grid(axis="x", color="#E1E4E8", linewidth=0.55)

    save_publication_figure(fig, result_dir / "fig_main_svi_mcmc")


def make_mcmc_convergence_figure(result_dir: Path) -> None:
    samples = np.load(result_dir / "posterior_samples_mcmc.npz")["z"]
    fields = np.load(result_dir / "figure_source_data" / "posterior_field_summaries.npz")
    selected_modes = fields["selected_modes"].astype(int)
    scale_labels = ["Large-scale\ncoefficient", "Intermediate-scale\ncoefficient", "Fine-scale\ncoefficient"]
    metrics = json.loads((result_dir / "mcmc_metrics.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(3, 1, figsize=(7.20, 4.8), sharex=True, constrained_layout=True)
    draws = np.arange(1, samples.shape[1] + 1)
    for row, mode in enumerate(selected_modes):
        ax = axes[row]
        for chain in range(samples.shape[0]):
            ax.plot(
                draws,
                samples[chain, :, mode],
                color=COLORS["chains"][chain % len(COLORS["chains"])],
                linewidth=0.55,
                alpha=0.85,
                label=f"Chain {chain + 1}",
            )
        ax.set_ylabel(scale_labels[row])
        ax.grid(axis="y", color="#E1E4E8", linewidth=0.55)
        add_panel_label(ax, chr(ord("a") + row))
    axes[-1].set_xlabel("Post-warm-up draw")
    axes[0].legend(ncol=samples.shape[0], fontsize=6.3, loc="upper right")
    axes[0].text(
        0.01,
        0.03,
        f"max $\\hat{{R}}$ = {metrics['r_hat_max']:.3f}; min ESS = {metrics['ess_min']:.0f}; "
        f"divergences = {metrics['divergence_count']}",
        transform=axes[0].transAxes,
        fontsize=6.2,
        color="#5B6573",
    )
    save_publication_figure(fig, result_dir / "fig_si_mcmc_convergence")


def make_correlation_figure(result_dir: Path) -> None:
    correlation = np.load(result_dir / "figure_source_data" / "mcmc_latent_correlation.npy")
    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("Latent coefficient rank")
    ax.set_ylabel("Latent coefficient rank")
    ax.set_title("MCMC posterior correlation among latent coefficients", fontsize=8)
    ticks = np.arange(0, correlation.shape[0], 8)
    ax.set_xticks(ticks, ticks + 1)
    ax.set_yticks(ticks, ticks + 1)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson correlation")
    cbar.ax.tick_params(labelsize=6.5, length=2)
    add_panel_label(ax, "a")
    save_publication_figure(fig, result_dir / "fig_si_mcmc_correlation")


def make_svi_elbo_figure(result_dir: Path) -> None:
    data = pd.read_csv(result_dir / "figure_source_data" / "svi_elbo.csv")
    window = max(5, min(51, len(data) // 20 * 2 + 1))
    smoothed = data["elbo_loss"].rolling(window=window, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(7.20, 2.9), constrained_layout=True)
    ax.plot(data["iteration"], data["elbo_loss"], color="#9CB4D5", linewidth=0.65, alpha=0.75, label="Iteration loss")
    ax.plot(data["iteration"], smoothed, color=COLORS["SVI"], linewidth=1.45, label=f"Centered moving mean (n={window})")
    ax.set_xlabel("SVI iteration")
    ax.set_ylabel("ELBO loss (−ELBO)")
    ax.set_title("SVI convergence under the matched channelized benchmark", fontsize=8)
    ax.grid(axis="y", color="#E1E4E8", linewidth=0.6)
    ax.legend(fontsize=6.5)
    add_panel_label(ax, "a")
    save_publication_figure(fig, result_dir / "fig_si_svi_elbo")


def make_additional_metrics_figure(result_dir: Path) -> None:
    summary = pd.read_csv(result_dir / "comparison_summary.csv").set_index("method")
    metric_specs = [
        ("lnK_MAE", "ln K MAE", "a", 3),
        ("head_MAE", "Hydraulic-head MAE", "b", 3),
        ("high_K_Jaccard", "High-K Jaccard overlap", "c", 3),
        ("mean_posterior_SD_lnK", "Mean posterior SD (ln K)", "d", 4),
        ("median_posterior_SD_lnK", "Median posterior SD (ln K)", "e", 4),
        ("mean_95_interval_width_lnK", "Mean 95% interval width (ln K)", "f", 3),
    ]
    source_rows = []
    for column, label, _, _ in metric_specs:
        for method in ["SVI", "MCMC"]:
            source_rows.append({"metric": label, "method": method, "value": summary.loc[method, column]})
    pd.DataFrame(source_rows).to_csv(
        result_dir / "figure_source_data" / "si_additional_metrics.csv", index=False
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.20, 4.45), constrained_layout=True)
    for ax, (column, title, letter, decimals) in zip(axes.ravel(), metric_specs):
        values = [summary.loc["SVI", column], summary.loc["MCMC", column]]
        bars = ax.bar(
            [0, 1],
            values,
            width=0.58,
            color=[COLORS["SVI"], COLORS["MCMC"]],
            edgecolor="#4B4B4B",
            linewidth=0.55,
        )
        ax.set_xticks([0, 1], ["SVI", "MCMC"])
        ax.set_title(title, fontsize=8, pad=5)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y", color="#E1E4E8", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                f"{value:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
            )
        add_panel_label(ax, letter)
    fig.suptitle(
        "Additional posterior-reconstruction metrics for the matched channelized benchmark",
        fontsize=9,
    )
    save_publication_figure(fig, result_dir / "fig_si_additional_metrics")


def make_latent_summary_figure(result_dir: Path) -> None:
    svi_samples = np.load(result_dir / "posterior_samples_svi.npz")["z"]
    mcmc_samples = np.load(result_dir / "posterior_samples_mcmc.npz")["z"].reshape(-1, svi_samples.shape[1])
    fields = np.load(result_dir / "figure_source_data" / "posterior_field_summaries.npz")
    selected = fields["selected_modes"].astype(int)
    scale_names = ["Large scale", "Intermediate scale", "Fine scale"]

    def summarize(samples: np.ndarray, method: str) -> pd.DataFrame:
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        centered = samples - mean[None]
        skewness = np.mean(centered**3, axis=0) / (std**3 + 1e-12)
        return pd.DataFrame(
            {
                "method": method,
                "coefficient_rank": np.arange(1, samples.shape[1] + 1),
                "mean": mean,
                "std": std,
                "skewness": skewness,
            }
        )

    latent_summary = pd.concat(
        [summarize(svi_samples, "SVI"), summarize(mcmc_samples, "MCMC")], ignore_index=True
    )
    latent_summary.to_csv(
        result_dir / "figure_source_data" / "si_all_latent_summary.csv", index=False
    )
    svi = latent_summary[latent_summary["method"] == "SVI"].set_index("coefficient_rank")
    mcmc = latent_summary[latent_summary["method"] == "MCMC"].set_index("coefficient_rank")

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.75), constrained_layout=True)

    ax = axes[0]
    ax.scatter(mcmc["mean"], svi["mean"], s=12, color="#9AA2AB", alpha=0.75, linewidth=0)
    limits = [min(mcmc["mean"].min(), svi["mean"].min()), max(mcmc["mean"].max(), svi["mean"].max())]
    margin = 0.06 * (limits[1] - limits[0])
    limits = [limits[0] - margin, limits[1] + margin]
    ax.plot(limits, limits, color="#4B4B4B", linestyle=(0, (3, 2)), linewidth=0.9)
    for color, mode, label in zip(COLORS["modes"], selected, scale_names):
        rank = mode + 1
        ax.scatter(mcmc.loc[rank, "mean"], svi.loc[rank, "mean"], s=28, color=color, edgecolor="white", linewidth=0.5, zorder=3, label=label)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("MCMC posterior mean")
    ax.set_ylabel("SVI posterior mean")
    ax.set_title("All latent posterior means", fontsize=8)
    ax.grid(color="#E1E4E8", linewidth=0.55)
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.scatter(mcmc["std"], svi["std"], s=12, color="#9AA2AB", alpha=0.75, linewidth=0)
    limits = [0.0, max(mcmc["std"].max(), svi["std"].max()) * 1.06]
    ax.plot(limits, limits, color="#4B4B4B", linestyle=(0, (3, 2)), linewidth=0.9)
    for color, mode in zip(COLORS["modes"], selected):
        rank = mode + 1
        ax.scatter(mcmc.loc[rank, "std"], svi.loc[rank, "std"], s=28, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("MCMC posterior SD")
    ax.set_ylabel("SVI posterior SD")
    ax.set_title("All latent posterior spreads", fontsize=8)
    ax.grid(color="#E1E4E8", linewidth=0.55)
    add_panel_label(ax, "b")

    ax = axes[2]
    ranks = mcmc.index.to_numpy()
    ax.axhline(0.0, color="#9AA2AB", linewidth=0.7)
    ax.plot(ranks, mcmc["skewness"], color=COLORS["MCMC"], linewidth=1.0)
    ax.scatter(ranks, mcmc["skewness"], s=8, color=COLORS["MCMC"], linewidth=0)
    for color, mode in zip(COLORS["modes"], selected):
        rank = mode + 1
        ax.scatter(rank, mcmc.loc[rank, "skewness"], s=30, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel("Latent coefficient rank")
    ax.set_ylabel("MCMC marginal skewness")
    ax.set_title("Non-Gaussian marginal shape", fontsize=8)
    ax.grid(axis="y", color="#E1E4E8", linewidth=0.55)
    add_panel_label(ax, "c")
    axes[0].legend(fontsize=5.8, loc="upper left", handletextpad=0.3)
    save_publication_figure(fig, result_dir / "fig_si_latent_summary")


def make_chain_computation_figure(result_dir: Path) -> None:
    metrics = json.loads((result_dir / "mcmc_metrics.json").read_text(encoding="utf-8"))
    chains = pd.DataFrame(metrics["chain_metrics"])
    chains["chain_label"] = [f"Chain {int(value)}" for value in chains["chain"]]
    chains.to_csv(result_dir / "figure_source_data" / "si_chain_computation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.8), constrained_layout=True)
    x = np.arange(len(chains))
    axes[0].bar(x, chains["warmup_seconds"], color="#9CB4D5", width=0.62, label="Warm-up")
    axes[0].bar(
        x,
        chains["sampling_seconds"],
        bottom=chains["warmup_seconds"],
        color=COLORS["SVI"],
        width=0.62,
        label="Post-warm-up sampling",
    )
    axes[0].set_xticks(x, chains["chain_label"])
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Chain-level runtime decomposition", fontsize=8)
    axes[0].grid(axis="y", color="#E1E4E8", linewidth=0.6)
    axes[0].legend(fontsize=6.2)
    add_panel_label(axes[0], "a")

    bars = axes[1].bar(x, chains["pi_unet_inference_evaluations"], color=COLORS["MCMC"], width=0.62)
    axes[1].set_xticks(x, chains["chain_label"])
    axes[1].set_ylabel("PI-U-Net evaluations")
    axes[1].set_title("Chain-level forward-model cost", fontsize=8)
    axes[1].grid(axis="y", color="#E1E4E8", linewidth=0.6)
    for bar, calls, acceptance in zip(
        bars, chains["pi_unet_inference_evaluations"], chains["acceptance_rate"]
    ):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + chains["pi_unet_inference_evaluations"].max() * 0.025,
            f"{int(calls):,}\nacc.={acceptance:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.1,
        )
    axes[1].set_ylim(0, chains["pi_unet_inference_evaluations"].max() * 1.20)
    add_panel_label(axes[1], "b")
    save_publication_figure(fig, result_dir / "fig_si_chain_computation")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate publication figures for the SVI-MCMC benchmark")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    make_main_figure(result_dir)
    make_mcmc_convergence_figure(result_dir)
    make_correlation_figure(result_dir)
    make_svi_elbo_figure(result_dir)
    make_additional_metrics_figure(result_dir)
    make_latent_summary_figure(result_dir)
    make_chain_computation_figure(result_dir)
    print(f"Benchmark figures written to {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
