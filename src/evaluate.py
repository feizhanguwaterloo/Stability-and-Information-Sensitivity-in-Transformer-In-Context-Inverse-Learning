import argparse
import csv
import json
import os
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

from .data import sample_context, split_tasks
from .dispatch import INTERIOR, Observations, solve_dispatch
from .inverse import interior_ridge, kkt_correct
from .kkt import equality_gram, residual
from .train import fit, predict

METHODS = ("classical-kkt", "transformer", "transformer-kkt", "interior-ridge", "oracle")
REGIMES = ("interior-rich", "mixed", "boundary-heavy")
METRICS = ("relative_parameter_error", "marginal_cost_rmse", "context_kkt_rmse",
           "heldout_dispatch_rmse", "min_interior_eigenvalue", "both_interior_fraction")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nested_contexts(tasks, lengths, rng):
    full = [sample_context(theta, max(lengths), rng) for theta in tasks]
    return {m: [obs.take(slice(m)) for obs in full] for m in lengths}


def fit_classical(method, anchor, obs, gamma):
    solver = interior_ridge if method == "interior-ridge" else kkt_correct
    return solver(anchor, obs, gamma)


def tune_gammas(config, validation_tasks, contexts, anchors, prior):
    rows = []
    selected = {}
    for method in ("classical-kkt", "transformer-kkt", "interior-ridge"):
        scores = []
        for gamma in config["gamma_grid"]:
            errors = []
            for m, batch in contexts.items():
                seeds = anchors[m] if method == "transformer-kkt" else [prior] * len(batch)
                for theta, obs, anchor in zip(validation_tasks, batch, seeds):
                    estimate = fit_classical(method, anchor, obs, gamma)
                    errors.append(np.linalg.norm(estimate - theta) / np.linalg.norm(theta))
            score = float(np.mean(errors))
            scores.append(score)
            rows.append({"method": method, "gamma": gamma, "validation_relative_error": score})
        selected[method] = config["gamma_grid"][int(np.argmin(scores))]
    return selected, rows


def rmse(x):
    return np.sqrt(np.mean(np.square(x)))


def recovery_rows(experiment, group, tasks, contexts, anchors, prior, gammas, heldout):
    rows = []
    for idx, (theta, obs, anchor, query) in enumerate(zip(tasks, contexts, anchors, heldout)):
        estimates = (
            kkt_correct(prior, obs, gammas["classical-kkt"]),
            anchor,
            kkt_correct(anchor, obs, gammas["transformer-kkt"]),
            interior_ridge(prior, obs, gammas["interior-ridge"]),
            theta,
        )
        min_eigenvalue = np.linalg.eigvalsh(equality_gram(obs)).min()
        interior_fraction = np.mean(np.all(obs.status == INTERIOR, axis=1))
        for method, estimate in zip(METHODS, estimates):
            diff = (estimate - theta).reshape(2, 2)
            reconstructed = solve_dispatch(estimate, query.demand)
            rows.append({
                "experiment": experiment, "group": group, "task": idx,
                "m": len(obs.demand), "method": method,
                "relative_parameter_error": np.linalg.norm(diff) / np.linalg.norm(theta),
                "marginal_cost_rmse": rmse(diff[:, 0] + diff[:, 1] * query.dispatch),
                "context_kkt_rmse": rmse(residual(estimate, obs)),
                "heldout_dispatch_rmse": rmse(reconstructed.dispatch - query.dispatch),
                "min_interior_eigenvalue": min_eigenvalue,
                "both_interior_fraction": interior_fraction,
            })
    return rows


def stability_rows(m, tasks, contexts, anchors, model, config, gamma, rng):
    variants = []
    indices = []
    for idx, (theta, obs) in enumerate(zip(tasks, contexts)):
        for trial in range(config["replacement_trials"]):
            new = sample_context(theta, 1, rng)
            position = rng.integers(m)
            variant = Observations(*(field.copy() for field in obs))
            for field, replacement in zip(variant, new):
                field[position] = replacement[0]
            variants.append(variant)
            indices.append((idx, trial))
    modified = predict(model, variants, config["batch_size"])
    corrected = [kkt_correct(anchor, obs, gamma) for anchor, obs in zip(anchors, contexts)]
    rows = []
    for (idx, trial), obs, anchor in zip(indices, variants, modified):
        adjusted = kkt_correct(anchor, obs, gamma)
        for method, delta in (("transformer", anchor - anchors[idx]),
                              ("transformer-kkt", adjusted - corrected[idx])):
            rows.append({"m": m, "task": idx, "trial": trial, "method": method,
                         "parameter_delta": np.linalg.norm(delta)})
    return rows


def summarize(recovery, stability):
    groups = {}
    for row in recovery:
        key = (row["experiment"], row["group"], row["m"], row["method"])
        groups.setdefault(key, []).append(row)
    recovery_summary = []
    for (experiment, group, m, method), rows in groups.items():
        recovery_summary.append({
            "experiment": experiment, "group": group, "m": m, "method": method,
            "tasks": len(rows),
            **{metric: np.mean([row[metric] for row in rows]) for metric in METRICS},
        })

    clusters = {}
    for row in stability:
        key = (row["m"], row["method"], row["task"])
        clusters.setdefault(key, []).append(row["parameter_delta"])
    groups = {}
    for (m, method, _), deltas in clusters.items():
        groups.setdefault((m, method), []).append(np.mean(deltas))
    stability_summary = [
        {"m": m, "method": method, "tasks": len(deltas),
         "mean_delta": np.mean(deltas), "se": np.std(deltas, ddof=1) / np.sqrt(len(deltas))}
        for (m, method), deltas in groups.items()
    ]
    return recovery_summary, stability_summary


def run(config, output):
    from .plots import plot_experiments

    output.mkdir(parents=True, exist_ok=True)
    train, validation, test = split_tasks(config["seed"], config["task_counts"])
    prior = train.mean(axis=0)
    rngs = [np.random.default_rng(stream)
            for stream in np.random.SeedSequence([config["seed"], 2]).spawn(5)]
    lengths = config["lengths"]
    valid_contexts = nested_contexts(validation, lengths, rngs[0])
    model, history, best_epoch = fit(config, train, validation, valid_contexts)
    write_csv(output / "training.csv", history)
    batch_size = config["batch_size"]
    valid_anchors = {m: predict(model, batch, batch_size) for m, batch in valid_contexts.items()}
    gammas, tuning = tune_gammas(config, validation, valid_contexts, valid_anchors, prior)
    write_csv(output / "validation_gamma.csv", tuning)
    print(f"Selected checkpoint: epoch {best_epoch}. Selected gamma: {gammas}", flush=True)

    test_contexts = nested_contexts(test, lengths, rngs[1])
    heldout = [sample_context(theta, config["heldout_demands"], rngs[2]) for theta in test]
    recovery = []
    stability = []
    for m, contexts in test_contexts.items():
        anchors = predict(model, contexts, batch_size)
        recovery.extend(recovery_rows("A", "uniform", test, contexts, anchors, prior, gammas, heldout))
        stability.extend(stability_rows(m, test, contexts, anchors, model, config,
                                        gammas["transformer-kkt"], rngs[3]))
    for regime in REGIMES:
        contexts = [sample_context(theta, config["regime_context_length"], rngs[4], regime)
                    for theta in test]
        anchors = predict(model, contexts, batch_size)
        recovery.extend(recovery_rows("B", regime, test, contexts, anchors, prior, gammas, heldout))

    recovery_summary, stability_summary = summarize(recovery, stability)
    for name, rows in (("recovery_tasks", recovery), ("stability_pairs", stability),
                       ("recovery_summary", recovery_summary), ("stability_summary", stability_summary)):
        write_csv(output / f"{name}.csv", rows)
    plot_experiments(recovery_summary, stability_summary, output)
    metadata = {
        "config": config, "best_epoch": best_epoch, "selected_gamma": gammas,
        "prior": prior.tolist(), "device": str(next(model.parameters()).device),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": str(torch.__version__)},
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate inverse ED estimators")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot"))
    args = parser.parse_args()
    # Deterministic CUDA matrix operations require this before initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(config, args.output)


if __name__ == "__main__":
    main()
