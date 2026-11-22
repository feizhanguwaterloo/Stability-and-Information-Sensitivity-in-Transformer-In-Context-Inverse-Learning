"""Transformer-centered experiments, separate from the original pilot."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .data import HIGH, LOW, sample_context, split_tasks
from .dispatch import Observations
from .evaluate import nested_contexts, rmse, write_csv
from .identified import admissible_diameter, exact_kkt_geometry, exact_kkt_project
from .inverse import kkt_correct
from .model import DeepSets, NormControlledTransformer, SetTransformer, attention_operators
from .train import fit, predict

FOLDERS = {"D": "transformer_stability", "E": "task_scaling", "F": "context_informativeness"}


def stream(seed, channel):
    return np.random.default_rng(np.random.SeedSequence([seed, 886, channel]))


def relative_error(estimate, theta):
    return np.linalg.norm(estimate - theta, axis=-1) / np.linalg.norm(theta, axis=-1)


def recovery_metrics(estimate, theta, query):
    diff = (estimate - theta).reshape(2, 2)
    return {"relative_error": relative_error(estimate, theta),
            "marginal_cost_rmse": rmse(diff[:, 0] + diff[:, 1] * query.dispatch)}


def annotate(rows, seed, **fields):
    return [{"seed": seed, **fields, **row} for row in rows]


def summarize(rows, keys, metrics):
    """Average within each seed first, then compute across-seed SD."""
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, {}).setdefault(row["seed"], []).append(row)
    per_seed, pooled = [], []
    for key, seeds in groups.items():
        identifiers = dict(zip(keys, key))
        estimates = []
        for seed, bucket in seeds.items():
            values = {metric: float(np.mean([r[metric] for r in bucket])) for metric in metrics}
            per_seed.append({**identifiers, "seed": seed, "n": len(bucket), **values})
            estimates.append(values)
        means = {f"{metric}_mean": float(np.mean([r[metric] for r in estimates]))
                 for metric in metrics}
        deviations = {f"{metric}_sd": float(np.std([r[metric] for r in estimates], ddof=1))
                      if len(estimates) > 1 else 0.0 for metric in metrics}
        pooled.append({**identifiers, "seeds": len(estimates), **means, **deviations})
    return per_seed, pooled


def save_summary(output, name, rows, keys, metrics, save_raw=True):
    if save_raw:
        write_csv(output / f"{name}_raw.csv", rows)
    per_seed, pooled = summarize(rows, keys, metrics)
    write_csv(output / f"{name}_seeds.csv", per_seed)
    write_csv(output / f"{name}_summary.csv", pooled)


def replace(obs, index, new):
    result = Observations(*(array.copy() for array in obs))
    for field, value in zip(result, new):
        field[index] = value[0]
    return result


def train_model(config, seed, source, validation, valid_contexts, label,
                factory=SetTransformer, **overrides):
    fit_config = {**config, "seed": seed, "verbose": False, **overrides}
    model, history, epoch = fit(fit_config, source, validation, valid_contexts, factory)
    print(f"seed={seed} {label}: epoch {epoch}, validation error "
          f"{history[epoch - 1]['validation_relative_error']:.5f}", flush=True)
    return model, epoch, history


def experiment_d(config, output):
    lengths, batch = config["lengths"], config["batch_size"]
    factories = {"transformer": SetTransformer,
                 "norm-controlled": partial(NormControlledTransformer, config["attention_cap"]),
                 "deepsets": DeepSets}
    recovery, stability, trial_rows, diagnostics, norms, histories = [], [], [], [], [], []
    for seed in config["seeds"]:
        source, valid, test = tasks(config, seed)
        valid_contexts = nested_contexts(valid, lengths, stream(seed, 1))
        context_batches = nested_contexts(test, lengths, stream(seed, 2))
        query_rng = stream(seed, 3)
        queries = [sample_context(theta, 64, query_rng) for theta in test]
        models = {}
        for name, factory in factories.items():
            model, epoch, history = train_model(config, seed, source, valid, valid_contexts,
                                                name, factory)
            models[name] = model
            histories.extend(annotate(history, seed, model=name))
            diagnostics.append({"seed": seed, "model": name, "selected_epoch": epoch,
                                "parameters": sum(p.numel() for p in model.parameters())})
            if name == "transformer":
                torch.save(model.state_dict(), output / f"transformer_{seed}.pt")
            if name != "deepsets":
                for index, weight in enumerate(attention_operators(model)):
                    value = float(torch.linalg.matrix_norm(weight.detach(), ord=2))
                    norms.append({"seed": seed, "model": name, "layer": index // 4 + 1,
                                  "operator": ("Q", "K", "V", "output")[index % 4], "norm": value})
                    if name == "norm-controlled" and value > config["attention_cap"] + 2e-6:
                        raise RuntimeError(f"attention norm exceeds cap: {value}")

        prior = source.mean(axis=0)
        rng = stream(seed, 4)
        trials = config["replacement_trials"]
        for m, contexts in context_batches.items():
            variants = [replace(obs, rng.integers(m), sample_context(theta, 1, rng))
                        for theta, obs in zip(test, contexts) for _ in range(trials)]
            base = {name: predict(model, contexts, batch) for name, model in models.items()}
            modified = {name: predict(model, variants, batch) for name, model in models.items()}
            gamma = config["correction_gamma"]
            base["transformer-kkt"] = np.array([kkt_correct(anchor, obs, gamma)
                                                for anchor, obs in zip(base["transformer"], contexts)])
            modified["transformer-kkt"] = np.array([kkt_correct(anchor, obs, gamma)
                                                    for anchor, obs in zip(modified["transformer"], variants)])
            for name, predictions in modified.items():
                changes = np.linalg.norm(predictions - np.repeat(base[name], trials, axis=0), axis=1)
                trial_rows.extend({"seed": seed, "m": m, "model": name, "task": j // trials,
                                   "trial": j % trials, "delta": delta}
                                  for j, delta in enumerate(changes))
                stability.append({"seed": seed, "m": m, "model": name, "mean_delta": changes.mean(),
                                  "p95_delta": np.quantile(changes, .95), "max_delta": changes.max()})
            for idx, (theta, obs, query) in enumerate(zip(test, contexts, queries)):
                vertices = exact_kkt_geometry(obs)
                diameter = admissible_diameter(vertices)
                estimates = {name: values[idx] for name, values in base.items()}
                estimates["exact-prior"] = exact_kkt_project(prior, obs, vertices)
                estimates["exact-learned"] = exact_kkt_project(estimates["transformer"], obs, vertices)
                for name, estimate in estimates.items():
                    recovery.append({"seed": seed, "m": m, "model": name, "task": idx,
                                     "identification": "point" if diameter <= 1e-9 else "partial",
                                     **recovery_metrics(estimate, theta, query)})
    write_csv(output / "replacement_trials.csv", trial_rows)
    write_csv(output / "architecture.csv", diagnostics)
    write_csv(output / "attention_norms.csv", norms)
    write_csv(output / "training.csv", histories)
    save_summary(output, "stability", stability, ("m", "model"),
                 ("mean_delta", "p95_delta", "max_delta"), save_raw=False)
    metrics = ("relative_error", "marginal_cost_rmse")
    save_summary(output, "recovery", recovery, ("m", "model"), metrics)
    save_summary(output, "recovery_by_identification", recovery,
                 ("m", "model", "identification"), metrics, save_raw=False)


def tasks(config, seed):
    return split_tasks(seed, (max(config["source_counts"]),
                              config["validation_tasks"], config["test_tasks"]))


def load_transformer(config, seed, root):
    output = root / FOLDERS["D"]
    if json.loads((output / "config.json").read_text(encoding="utf-8")) != config:
        raise ValueError("Experiment D must use the same study configuration")
    model = SetTransformer()
    model.load_state_dict(torch.load(output / f"transformer_{seed}.pt", map_location="cpu",
                                     weights_only=True))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/transformer_study.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--stage", choices=("D", "E", "F", "all"), default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    for stage in ("D", "E", "F") if args.stage == "all" else (args.stage,):
        output = args.output / FOLDERS[stage]
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if stage == "D":
            experiment_d(config, output)
        elif stage == "E":
            from .study_transfer import experiment_e
            experiment_e(config, output, args.output)
        else:
            from .study_info import experiment_f
            experiment_f(config, output, args.output)


if __name__ == "__main__":
    main()
