"""Rebuild the course-report tables and figure from the saved experiment CSVs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/course_report"


def read(path):
    with (ROOT / "results" / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(rows, **filters):
    matches = [row for row in rows if all(row[key] == str(value) for key, value in filters.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one result for {filters}, found {len(matches)}")
    return matches[0]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    recovery = read("transformer_stability/recovery_summary.csv")
    sensitivity = read("transformer_stability/stability_summary.csv")
    transfer = read("task_scaling/risks_summary.csv")
    contexts = read("context_informativeness/contexts_summary.csv")
    matched = read("leverage_stability/matched_summary.csv")
    pairs = read("leverage_stability/identifying_pair_summary.csv")
    models = ("transformer", "norm-controlled", "deepsets")
    lengths = (4, 8, 16, 32, 64)
    architecture = []
    for m in lengths:
        changes = [float(select(sensitivity, m=m, model=name)["mean_delta_mean"]) for name in models]
        errors = [100 * float(select(recovery, m=m, model=name)["relative_error_mean"]) for name in models]
        architecture.append(f"{m} & " + " & ".join([f"{v:.5f}" for v in changes] +
                                                   [f"{v:.2f}" for v in errors]) + r" \\")
    (OUTPUT / "architecture_table.tex").write_text(
        r"\begin{tabular}{r rrr rrr}" + "\n" + r"\toprule" + "\n"
        + r"&\multicolumn{3}{c}{Mean prediction change}&\multicolumn{3}{c}{Recovery error (\%)}\\"
        + "\n" + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n"
        + r"$m$&Transformer&Norm cap&DeepSets&Transformer&Norm cap&DeepSets\\"
        + "\n" + r"\midrule" + "\n" + "\n".join(architecture)
        + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")

    transfer_rows = []
    for count in (128, 256, 512, 1024, 1536):
        left = select(transfer, study="equal-passes", T=count, m=32)
        right = select(transfer, study="equal-steps", T=count, m=32)
        transfer_rows.append(f"{count:,} & {int(float(left['training_updates_mean']))} & "
                             f"{100 * float(left['transfer_relative_error_mean']):.2f} & "
                             f"{100 * float(right['transfer_relative_error_mean']):.2f}" + r" \\")
    (OUTPUT / "transfer_table.tex").write_text(
        r"\begin{tabular}{r rrr}\toprule" + "\n"
        + r"Source tasks&Equal-pass updates&Equal-pass error (\%)&Equal-step error (\%)\\"
        + "\n" + r"\midrule" + "\n" + "\n".join(transfer_rows)
        + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")
    (OUTPUT / "transfer_compact.tex").write_text(
        r"\begin{tabular}{rrrr}\toprule" + "\n"
        + r"$T$&Updates&Equal pass&Equal step\\" + "\n" + r"\midrule" + "\n"
        + "\n".join(transfer_rows) + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")

    gram_labels = ("0.03257", r"3.72\times10^{-6}", "0.02429", "0")
    labels = ("Well-conditioned", "Nearly redundant", "Mixed", "Boundary-heavy")
    conditions = ("well-conditioned", "nearly-redundant", "mixed", "boundary-heavy")
    context_rows = []
    for label, case, gram in zip(labels, conditions, gram_labels):
        row = select(contexts, condition=case, model="transformer")
        diameter = float(row["admissible_diameter_mean"])
        context_rows.append(
            f"{label} & $" + gram + f"$ & {diameter:.3f} & $"
            + f"{100 * float(row['relative_error_mean']):.2f}"
            + r"\pm" + f"{100 * float(row['relative_error_sd']):.2f}$" + r" \\")
    (OUTPUT / "context_table.tex").write_text(
        r"\begin{tabular}{lrrr}\toprule" + "\n"
        + r"Context&$\mu$&Diameter&$E$ (\%)\\"
        + "\n" + r"\midrule" + "\n" + "\n".join(context_rows)
        + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")
    compact_rows = [row.split(" & ") for row in context_rows]
    (OUTPUT / "context_compact.tex").write_text(
        r"\begin{tabular}{lrr}\toprule" + "\n" + r"Context&$\mu$&Error (\%)\\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(" & ".join((row[0], row[1], row[3])) for row in compact_rows)
        + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.05), layout="constrained")
    colors = ("#0072B2", "#D55E00", "#009E73", "#555555")
    labels = ("Transformer", "Norm cap", "DeepSets", "Ridge")
    for name, label, color in zip((*models, "ridge"), labels, colors):
        rows = [select(pairs, model=name, m=m) for m in lengths]
        axes[0].errorbar(lengths, [float(row["target_direction_response_mean"]) for row in rows],
                         yerr=[float(row["target_direction_response_sd"]) for row in rows],
                         marker="o", color=color, label=label, markersize=3, capsize=2, linewidth=1.2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(lengths, labels=[str(m) for m in lengths])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(xlabel="Context records", ylabel="True direction recovered",
                title="(a) Valid cross-task identifying pair")
    axes[0].legend(fontsize=6, ncols=2, loc="center right")
    positions = list(range(4))
    for shift, case, color in ((-.18, "low-leverage", "#8AB5D1"),
                               (.18, "high-leverage", "#D55E00")):
        rows = [select(matched, case=case, intervention="price-0.01", model=name)
                for name in (*models, "ridge")]
        axes[1].bar([x + shift for x in positions], [float(row["mean_delta_mean"]) for row in rows],
                    yerr=[float(row["mean_delta_sd"]) for row in rows],
                    width=.34, capsize=2, color=color, label=case.replace("-", " "))
    axes[1].set(yscale="log", ylabel="Mean L2 prediction change",
                title="(b) Matched focal price perturbation")
    axes[1].set_xticks(positions, ["TF", "Norm TF", "DeepSets", "Ridge"])
    axes[1].legend(fontsize=6, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "information_response.pdf")
    fig.savefig(OUTPUT / "information_response.png", dpi=220)
    plt.close(fig)
    print("Generated report tables and vector figure from saved D-G results.")


if __name__ == "__main__":
    main()
