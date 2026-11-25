"""Create presentation-size plots from the saved course-project results."""

from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report_assets import read, select

OUTPUT = Path(__file__).resolve().parents[1] / "results/presentation"
MODELS = {
    "transformer": ("Transformer", "#0072B2", "o"),
    "norm-controlled": ("Norm cap 0.85", "#D55E00", "s"),
    "deepsets": ("DeepSets", "#009E73", "^"),
    "ridge": ("Ridge", "#555555", "v"),
}


def finish(fig, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(OUTPUT / f"{name}.pdf")
    fig.savefig(OUTPUT / f"{name}.png", dpi=160)
    plt.close(fig)


def length_axis(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values, labels=[str(value) for value in values])
    ax.xaxis.set_minor_locator(NullLocator())


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 17,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "pdf.fonttype": 42})
    lengths = (4, 8, 16, 32, 64)
    recovery = read("transformer_stability/recovery_summary.csv")
    stability = read("transformer_stability/stability_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    for name in ("transformer", "norm-controlled", "deepsets"):
        label, color, marker = MODELS[name]
        for ax, rows, metric in ((axes[0], stability, "mean_delta"),
                                 (axes[1], recovery, "relative_error")):
            values = [select(rows, m=m, model=name) for m in lengths]
            ax.errorbar(lengths, [float(row[f"{metric}_mean"]) for row in values],
                        yerr=[float(row[f"{metric}_sd"]) for row in values],
                        label=label, color=color, marker=marker, capsize=3, markersize=5)
            length_axis(ax, lengths)
            ax.set_xlabel("Context records")
    axes[0].set(yscale="log", ylabel="Mean L2 prediction change", title="Same-task replacement sensitivity")
    axes[1].set(ylabel="Relative parameter error", title="Recovery on unseen tasks")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=13)
    finish(fig, "architecture")

    pairs = read("leverage_stability/identifying_pair_summary.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.7), layout="constrained")
    for name, (label, color, marker) in MODELS.items():
        rows = [select(pairs, model=name, m=m) for m in lengths]
        ax.errorbar(lengths, [float(row["target_direction_response_mean"]) for row in rows],
                    yerr=[float(row["target_direction_response_sd"]) for row in rows],
                    label=label, color=color, marker=marker, capsize=3)
    length_axis(ax, lengths)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Context records", ylabel="Directional response",
           ylim=(-.03, 1.1))
    ax.legend(fontsize=13, ncols=2, loc="upper center", bbox_to_anchor=(.55, .84))
    finish(fig, "identifying_response")

    matched = read("leverage_stability/matched_summary.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.7), layout="constrained")
    positions = np.arange(4)
    for offset, case, color in ((-.18, "low-leverage", "#8AB5D1"),
                                (.18, "high-leverage", "#D55E00")):
        rows = [select(matched, case=case, intervention="price-0.01", model=name)
                for name in MODELS]
        ax.bar(positions + offset, [float(row["mean_delta_mean"]) for row in rows], width=.34,
               yerr=[float(row["mean_delta_sd"]) for row in rows], color=color, capsize=3,
               label=case.replace("-", " ").capitalize())
    ax.set_xticks(positions, [v[0] for v in MODELS.values()])
    ax.set(yscale="log", ylabel="Mean L2 prediction change")
    ax.legend(loc="upper left", fontsize=14)
    finish(fig, "matched_response")

    risks = read("task_scaling/risks_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    counts = (128, 256, 512, 1024, 1536)
    for ax, study, title in zip(axes, ("equal-passes", "equal-steps"),
                                ("14 passes per source task", "448 optimizer updates")):
        rows = [select(risks, study=study, condition="uniform", T=count, m=32) for count in counts]
        ax.errorbar(counts, [float(row["transfer_relative_error_mean"]) for row in rows],
                    yerr=[float(row["transfer_relative_error_sd"]) for row in rows],
                    color="#0072B2", marker="o", capsize=3)
        length_axis(ax, counts)
        ax.set(xlabel="Distinct source tasks", ylabel="Unseen relative parameter error", title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    finish(fig, "transfer")
    # Interpret Theorem 2 using the measured response, rather than the loose C.
    target_gap = float(np.sqrt(1.6))
    pair = select(pairs, model="transformer", m=64)
    mean_delta = float(pair["mean_delta_mean"])
    larger_error = float(pair["max_absolute_error_mean"])
    floor = max(0., (target_gap - mean_delta) / 2)
    raw = [row for row in read("leverage_stability/identifying_pair_raw.csv")
           if row["model"] == "transformer" and int(row["m"]) == 64]
    if len(raw) != 3:
        raise ValueError("Expected three original Transformer checkpoints.")
    for row in raw:
        required_error = max(0., (target_gap - float(row["delta"])) / 2)
        if float(row["max_absolute_error"]) + 1e-6 < required_error:
            raise ValueError(f"Recorded pair violates the geometric bound: {row['seed']}")
    macros = {
        "PairTargetGap": target_gap, "PairMeanDelta": mean_delta,
        "PairErrorFloor": floor, "PairObservedError": larger_error,
    }
    (OUTPUT / "pair_response_check.tex").write_text(
        "% Derived from G identifying_pair_summary.csv; absolute parameter units.\n"
        + "".join(r"\newcommand{\%s}{%.3f}" % (name, value) + "\n"
                  for name, value in macros.items()),
        encoding="utf-8", newline="\n")
    print("Created four vector figures and the measured-response theorem check.")


if __name__ == "__main__":
    main()
