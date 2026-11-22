"""Render the Transformer study figures from saved summary tables."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, PercentFormatter

MODELS = {
    "transformer": ("Transformer", "#0072B2", "o"),
    "norm-controlled": ("Norm-controlled", "#D55E00", "s"),
    "deepsets": ("DeepSets", "#009E73", "^"),
    "transformer-kkt": ("Transformer + KKT", "#CC79A7", "D"),
    "exact-prior": ("Exact KKT, fixed prior", "#555555", "v"),
    "exact-learned": ("Exact KKT, learned anchor", "#E69F00", "P"),
}


def read(folder, name):
    with (folder / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series(ax, rows, key, metric, label, color, marker):
    rows = sorted(rows, key=lambda row: int(row[key]))
    x = [int(row[key]) for row in rows]
    y = np.array([float(row[f"{metric}_mean"]) for row in rows])
    sd = np.array([float(row[f"{metric}_sd"]) for row in rows])
    ax.errorbar(x, y, yerr=sd, color=color, marker=marker, label=label,
                capsize=2.5, linewidth=1.5, markersize=4)
    return x, y


def log_ticks(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, labels=[str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def finish(fig, path):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
    for extension in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=220)
    plt.close(fig)


def plot_d(folder):
    stability = read(folder, "stability_summary")
    recovery = read(folder, "recovery_summary")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, (label, color, marker) in MODELS.items():
        rows = [r for r in stability if r["model"] == name]
        if rows:
            x, y = series(axes[0], rows, "m", "mean_delta", label, color, marker)
            if name == "transformer":
                axes[0].plot(x, y[0] * x[0] / np.array(x), "--", color="#777777",
                             label="Reference 1/m", linewidth=1.2)
        rows = [r for r in recovery if r["model"] == name]
        series(axes[1], rows, "m", "relative_error", label, color, marker)
    axes[0].set(yscale="log", ylabel="Mean L2 prediction change", title="A  Record replacement")
    axes[1].set(ylabel="Relative parameter error", title="B  Recovery on unseen tasks")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        log_ticks(ax, x)
        ax.set_xlabel("Context records")
        ax.legend(fontsize=7.5)
    fig.suptitle("Transformer architecture comparison  |  mean ± seed SD")
    finish(fig, folder / "architecture_comparison")


def plot_e(folder):
    risk = read(folder, "risks_summary")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for study, ax, title in zip(("equal-passes", "equal-steps"), axes[:2],
                                ("A  Equal task passes", "B  Equal optimizer steps")):
        rows = [r for r in risk if r["study"] == study and r["m"] == "32"]
        for metric, label, color, mark in (
                ("source_risk", "Source risk", "#009E73", "^"),
                ("transfer_risk", "Unseen risk", "#0072B2", "o"),
                ("shift_risk", "Shifted risk", "#D55E00", "s")):
            ticks, _ = series(ax, rows, "T", metric, label, color, mark)
        log_ticks(ax, ticks)
        ax.set(title=title, xlabel="Distinct source tasks", ylabel="Span-scaled MSE", yscale="log")
        ax.legend(fontsize=8)
    counts = sorted({int(row["T"]) for row in risk if row["study"] == "equal-steps"})
    for count, color in zip(counts, plt.cm.viridis(np.linspace(.07, .86, len(counts)))):
        rows = [r for r in risk if r["study"] == "equal-steps" and int(r["T"]) == count]
        if rows:
            ticks, _ = series(axes[2], rows, "m", "transfer_risk", f"T = {count}", color, "o")
    log_ticks(axes[2], ticks)
    axes[2].set(title="C  Context-length dependence", xlabel="Context records",
                ylabel="Unseen span-scaled MSE", yscale="log")
    axes[2].legend(fontsize=8)
    fig.suptitle("Transfer across optimization tasks  |  mean ± seed SD")
    finish(fig, folder / "transfer_scaling")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, study, title in zip(axes, ("equal-records", "diversity"),
                                ("A  Matched record and update budgets", "B  Source diversity at fixed T")):
        rows = [r for r in risk if r["study"] == study]
        for index, condition in enumerate(dict.fromkeys(r["condition"] for r in rows)):
            subset = [r for r in rows if r["condition"] == condition]
            name = (f"T = {subset[0]['T']}, train m = {condition[7:]}"
                    if study == "equal-records" else condition.title())
            ticks, _ = series(ax, subset, "m", "transfer_relative_error", name,
                              ("#0072B2", "#D55E00")[index], ("o", "s")[index])
        log_ticks(ax, ticks)
        ax.set(title=title, xlabel="Evaluation context records", ylabel="Unseen parameter error")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
    finish(fig, folder / "transfer_controls")


def plot_f(folder):
    contexts = read(folder, "contexts_summary")
    labels = ["well-conditioned", "nearly-redundant", "mixed", "boundary-heavy"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    x = np.arange(len(labels))
    for index, name in enumerate(("transformer", "exact-prior", "exact-learned")):
        label, color, _ = MODELS[name]
        rows = [next(r for r in contexts if r["condition"] == condition and r["model"] == name)
                for condition in labels]
        axes[0].bar(x + (index - 1) * .25, [float(r["relative_error_mean"]) for r in rows],
                    width=.25, yerr=[float(r["relative_error_sd"]) for r in rows],
                    color=color, label=label, capsize=2)
    axes[0].set(title="A  Cost recovery", ylabel="Relative parameter error")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=7)
    for index, metric in enumerate(("min_gram_eigenvalue", "admissible_diameter")):
        axes[1].scatter(x, [float(r[f"{metric}_mean"]) for r in rows],
                        marker=("o", "s")[index], s=46, label=metric.replace("_", " "),
                        color=("#0072B2", "#D55E00")[index])
    axes[1].set(title="B  Information diagnostics", ylabel="Value (symlog scale)")
    axes[1].set_yscale("symlog", linthresh=1e-5)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, ["Well-\nconditioned", "Nearly\nredundant", "Mixed", "Boundary-\nheavy"])
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    fig.suptitle(f"Matched tasks, {config['info_context_length']} observations per context")
    finish(fig, folder / "context_information")

    interventions = read(folder, "interventions_summary")
    interventions = [r for r in interventions if r["intervention"] != "shuffle"]
    names = ["Replace high", "Replace redundant", "Drop high", "Drop redundant",
             "Keep interior", "Keep boundary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    y = np.arange(len(names))
    for ax, metric, title in zip(axes, ("delta", "relative_error_change"),
                                ("A  Prediction sensitivity", "B  Change in recovery error")):
        ax.barh(y, [float(r[f"{metric}_mean"]) for r in interventions],
                xerr=[float(r[f"{metric}_sd"]) for r in interventions],
                color=["#0072B2"] * 2 + ["#009E73"] * 2 + ["#D55E00"] * 2, capsize=2)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set(title=title, xlabel=("L2 prediction change" if metric == "delta"
                                   else "Relative error change"))
        if metric == "relative_error_change":
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Matched interventions  |  mean ± seed SD")
    finish(fig, folder / "context_interventions")


def plot_g(folder):
    pairs = read(folder, "identifying_pair_summary")
    matched = read(folder, "matched_summary")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), layout="constrained")
    labels = {
        **{name: MODELS[name] for name in ("transformer", "norm-controlled", "deepsets")},
        "statistics-mlp": ("Statistics MLP", "#CC79A7", "D"),
        "ridge": ("Ridge", "#555555", "v"),
    }
    for ax, metric, title, ylabel in (
            (axes[0], "mean_delta", "A  Identifying-record response", "L2 prediction change"),
            (axes[1], "max_absolute_error", "B  Recovery on the same pair", "Larger L2 parameter error")):
        for name, (label, color, marker) in labels.items():
            subset = [row for row in pairs if row["model"] == name]
            ticks, _ = series(ax, subset, "m", metric, label, color, marker)
        log_ticks(ax, ticks)
        ax.set(title=title, xlabel="Context records", ylabel=ylabel, yscale="log")
    axes[0].legend(fontsize=7, loc="lower left")

    ax = axes[2]
    positions = np.arange(len(labels))
    for offset, case, color in ((-.18, "low-leverage", "#8AB5D1"),
                                (.18, "high-leverage", "#D55E00")):
        rows = [next(row for row in matched if row["model"] == name
                     and row["case"] == case and row["intervention"] == "price-0.01")
                for name in labels]
        ax.bar(positions + offset, [float(row["mean_delta_mean"]) for row in rows], width=.34,
               yerr=[float(row["mean_delta_sd"]) for row in rows], capsize=2,
               color=color, label=case.replace("-", " ").capitalize())
    ax.set_xticks(positions, ["TF", "Norm TF", "DeepSets", "Stats MLP", "Ridge"])
    ax.set(title="C  Matched focal price change", ylabel="Mean L2 prediction change", yscale="log")
    ax.legend(fontsize=7)
    fig.suptitle("Information-dependent stability  |  mean ± seed SD")
    finish(fig, folder / "leverage_stability")


def plot_followup(folder):
    pairs = read(folder, "identification_summary")
    caps = read(folder, "cap_sweep_summary")
    noise = read(folder, "noise_summary")
    transfer = read(folder, "matched_transfer_summary")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    cap_values = sorted({row["model"] for row in caps if row["model"] != "uncapped"}, key=float)
    labels = {"uncapped": ("Uncapped", "#0072B2", "o")}
    for index, cap in enumerate(cap_values):
        color = plt.cm.plasma((index + .5) / len(cap_values))
        labels[cap] = (f"Cap {float(cap):g}", color, ("s", "^", "D", "P")[index % 4])
    labels["ridge"] = ("Ridge", "#555555", "v")
    max_length = max(int(row["m"]) for row in pairs)
    for ax, metric, title, ylabel in (
            (axes[0, 0], "target_direction_response", "A  Recovery of target direction", "Fraction of true parameter gap"),
            (axes[0, 1], "max_absolute_error", "B  Recovery on the same pair", "Larger L2 error"),
            (axes[0, 2], "max_path_edge", "C  Largest step along replacement path", "L2 prediction change")):
        for name, (label, color, marker) in labels.items():
            subset = [row for row in pairs if row["model"] == name and int(row["m"]) == max_length]
            ticks, _ = series(ax, subset, "k", metric, label, color, marker)
        log_ticks(ax, ticks)
        ax.set(title=title, xlabel=f"Identifying records out of {max_length}", ylabel=ylabel)
    axes[0, 0].axhline(1, linestyle=":", color="#777777", linewidth=1)
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(fontsize=7, ncols=2)

    ax = axes[1, 0]
    for name, (label, color, marker) in labels.items():
        if name == "ridge":
            continue
        row = next(r for r in caps if r["model"] == name and int(r["m"]) == max_length)
        ax.errorbar(float(row["mean_delta_mean"]), float(row["relative_error_mean"]),
                    xerr=float(row["mean_delta_sd"]), yerr=float(row["relative_error_sd"]),
                    color=color, marker=marker, capsize=2.5, label=label)
    ax.set(title="D  Same-task replacements", xlabel=f"Mean L2 change at m = {max_length}",
           ylabel="Relative parameter error")
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax = axes[1, 1]
    cases = {"well-conditioned": ("Well-conditioned", "#0072B2"),
             "nearly-redundant": ("Nearly redundant", "#D55E00"),
             "boundary-heavy": ("Boundary-heavy", "#009E73")}
    for case, (label, color) in cases.items():
        for name, style in (("transformer", "-"), ("ridge", "--")):
            subset = sorted((r for r in noise if r["case"] == case and r["model"] == name),
                            key=lambda r: float(r["sigma"]))
            ax.errorbar([float(r["sigma"]) for r in subset],
                        [float(r["relative_error_mean"]) for r in subset],
                        yerr=[float(r["relative_error_sd"]) for r in subset], color=color,
                        linestyle=style, marker=".", capsize=2, label=f"{label}, {name}")
    ax.set(title="E  Recovery under price noise", xlabel="Price noise standard deviation",
           ylabel="Relative parameter error")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=6.5)

    ax = axes[1, 2]
    task_counts = sorted({int(r["T"]) for r in transfer})
    colors = plt.cm.viridis(np.linspace(.14, .8, len(task_counts)))
    for count, color in zip(task_counts, colors):
        subset = [r for r in transfer if int(r["T"]) == count]
        lengths, _ = series(ax, subset, "m", "relative_error", f"T = {count}", color, "o")
    log_ticks(ax, lengths)
    ax.set(title="F  Transfer at matched training budgets", xlabel="Evaluation context records",
           ylabel="Unseen relative parameter error")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=7)
    fig.suptitle("Transformer follow-up experiments  |  mean ± seed SD")
    finish(fig, folder / "followup_experiments")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--stage", choices=("D", "E", "F", "G", "followup", "all"), default="all")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 9.5, "svg.fonttype": "none"})
    if args.stage == "G":
        plot_g(args.output / "leverage_stability")
        return
    if args.stage == "followup":
        plot_followup(args.output / "leverage_followup")
        return
    for stage, folder, plot in (
            ("D", "transformer_stability", plot_d), ("E", "task_scaling", plot_e),
            ("F", "context_informativeness", plot_f)):
        if args.stage in (stage, "all"):
            plot(args.output / folder)


if __name__ == "__main__":
    main()
