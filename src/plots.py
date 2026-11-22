import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

LABELS = {
    "classical-kkt": "Classical KKT", "transformer": "Transformer",
    "transformer-kkt": "Transformer + KKT", "interior-ridge": "Interior ridge", "oracle": "Oracle",
}


def plot_experiments(recovery, stability, output):
    metrics = (("relative_parameter_error", "Relative parameter error"),
               ("marginal_cost_rmse", "Held-out marginal-cost RMSE"),
               ("context_kkt_rmse", "Context KKT residual RMSE"),
               ("heldout_dispatch_rmse", "Held-out dispatch RMSE"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, (metric, title) in zip(axes.flat, metrics):
        for method, label in LABELS.items():
            rows = [r for r in recovery if r["experiment"] == "A" and r["method"] == method]
            ax.plot([r["m"] for r in rows], [r[metric] for r in rows], ".-", label=label)
        ax.set(title=title, xlabel="Context records", ylabel="Mean across tasks")
        ax.set_xscale("log", base=2)
        ax.set_xticks([r["m"] for r in rows], labels=[str(r["m"]) for r in rows])
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize="small")
    fig.savefig(output / "A_context_length.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    regimes = ("interior-rich", "mixed", "boundary-heavy")
    positions = np.arange(len(regimes))
    for idx, (method, label) in enumerate(LABELS.items()):
        rows = [next(r for r in recovery if r["experiment"] == "B"
                     and r["method"] == method and r["group"] == regime) for regime in regimes]
        ax.bar(positions + (idx - 2) * .16, [r["relative_parameter_error"] for r in rows],
               width=.16, label=label)
    ax.set(xticks=positions, xticklabels=regimes,
           title=f"Recovery by observation regime (m = {rows[0]['m']})",
           ylabel="Mean relative parameter error")
    ax.legend(loc="upper left", fontsize="small")
    fig.savefig(output / "B_informativeness.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for method in ("transformer", "transformer-kkt"):
        rows = [r for r in stability if r["method"] == method]
        ax.errorbar([r["m"] for r in rows], [r["mean_delta"] for r in rows],
                    yerr=[1.96 * r["se"] for r in rows], fmt="o-", capsize=3, label=LABELS[method])
    lengths = np.array([r["m"] for r in rows])
    start = next(r for r in stability if r["method"] == "transformer" and r["m"] == lengths[0])
    ax.plot(lengths, lengths[0] * start["mean_delta"] / lengths, "--", label="Reference 1/m")
    ax.set_xscale("log", base=2)
    ax.set(yscale="log", xlabel="Context records",
           ylabel="Mean L2 parameter change", title="One valid same-task record replacement")
    ax.set_xticks(lengths, labels=[str(m) for m in lengths])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output / "C_stability.png", dpi=160)
    plt.close(fig)
