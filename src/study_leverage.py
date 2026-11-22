"""Experiment G, with clean leverage controls and a separate price perturbation."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .data import observation_features, sample_context
from .dispatch import solve_dispatch
from .evaluate import nested_contexts, write_csv
from .leverage import (StatisticsMLP, condition_number, crossed_backgrounds, hat_leverage,
                       ridge_estimate, theorem_pair, transformer_bound)
from .model import DeepSets, NormControlledTransformer
from .study import (load_transformer, relative_error, replace, save_summary,
                    stream, summarize, tasks, train_model)
from .study_info import context_design
from .train import predict


def paired_rows(seed, case, models, ridge, original, modified, targets, new_targets, batch):
    entries = []
    diagnostics = []
    for before, after in zip(original, modified):
        difference = observation_features(after) - observation_features(before)
        diagnostics.append((condition_number(before), np.linalg.norm(difference)))
    ridge_before, ridge_after = [np.array([ridge(obs) for obs in contexts])
                                 for contexts in (original, modified)]
    ridge_change = ridge_after - ridge_before
    for name, model in {**models, "ridge": None}.items():
        base, changed = ((ridge_before, ridge_after) if model is None else
                         (predict(model, original, batch), predict(model, modified, batch)))
        vectors = changed - base
        errors = relative_error(base, targets)
        new_errors = relative_error(changed, new_targets)
        absolute_errors = np.linalg.norm(base - targets, axis=1)
        new_absolute_errors = np.linalg.norm(changed - new_targets, axis=1)
        for idx, (vector, error, new_error, (condition, displacement)) in enumerate(
                zip(vectors, errors, new_errors, diagnostics)):
            target_diff = new_targets[idx] - targets[idx]
            separation = np.linalg.norm(target_diff)
            entries.append({
                "seed": seed, "case": case, "model": name, "task": idx, "m": len(original[idx].demand),
                "condition_number": condition, "feature_displacement": displacement,
                "delta": np.linalg.norm(vector), "relative_error": error,
                "replacement_relative_error": new_error,
                "max_absolute_error": max(absolute_errors[idx], new_absolute_errors[idx]),
                "ridge_response_distance": np.linalg.norm(vector - ridge_change[idx]),
                "target_direction_response": (vector @ target_diff / separation**2
                                              if separation else 0.0),
            })
    return entries


def detailed_summary(output, name, rows, keys, extra=()):
    write_csv(output / f"{name}_raw.csv", rows)
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, {}).setdefault(row["seed"], []).append(row)
    per_seed = []
    global_stats = {}
    for key, seed_groups in grouped.items():
        pooled = []
        for seed, records in seed_groups.items():
            delta = np.asarray([r["delta"] for r in records])
            per_seed.append({
                "seed": seed, **dict(zip(keys, key)), "n": len(delta),
                "mean_delta": delta.mean(), "p95_delta": np.quantile(delta, .95),
                "max_delta": delta.max(),
                **{metric: np.mean([r[metric] for r in records]) for metric in extra},
            })
            pooled.extend(delta)
        global_stats[key] = {"pooled_p95": np.quantile(pooled, .95), "maximum_sampled": max(pooled)}
    write_csv(output / f"{name}_seeds.csv", per_seed)
    _, summary = summarize(per_seed, keys, ("mean_delta", "p95_delta", "max_delta", *extra))
    for row in summary:
        row.update(global_stats[tuple(row[k] for k in keys)])
    write_csv(output / f"{name}_summary.csv", summary)


def fit_ridge(penalties, validation, contexts, prior):
    errors = []
    for penalty in penalties:
        values = [relative_error(np.array([ridge_estimate(obs, prior, penalty) for obs in group]),
                                 validation).mean() for group in contexts.values()]
        errors.append(float(np.mean(values)))
    return penalties[int(np.argmin(errors))], errors


def run(config, study_config, root, output):
    lengths, batch = study_config["lengths"], study_config["batch_size"]
    size = config["context_length"]
    if size < 4 or size > 64 or size % 2:
        raise ValueError("matched-context length must be even and between 4 and 64")
    within, matched, stress, bounds, tuning = [], [], [], [], []
    torch.set_num_threads(study_config["cpu_threads"])
    for seed in study_config["seeds"]:
        source, valid, test = tasks(study_config, seed)
        valid_contexts = nested_contexts(valid, lengths, stream(seed, 1))
        prior = source.mean(axis=0)
        penalty, scores = fit_ridge(config["ridge_penalties"], valid, valid_contexts, prior)
        tuning.extend({"seed": seed, "penalty": p, "validation_relative_error": error,
                       "selected": p == penalty} for p, error in zip(config["ridge_penalties"], scores))
        ridge = partial(ridge_estimate, prior=prior, penalty=penalty)
        models = {"transformer": load_transformer(study_config, seed, root)}
        factories = {
            "norm-controlled": partial(NormControlledTransformer, study_config["attention_cap"]),
            "deepsets": DeepSets, "statistics-mlp": StatisticsMLP,
        }
        for name, factory in factories.items():
            models[name], _, _ = train_model(study_config, seed, source, valid, valid_contexts,
                                             f"G {name}", factory)
        for name in ("transformer", "norm-controlled"):
            constant, score = transformer_bound(models[name])
            bounds.append({"seed": seed, "model": name, "log10_upper_C": constant,
                           "max_score_upper": score, "output_box_diameter": np.linalg.norm(
                               models[name].span.detach().cpu().numpy())})

        rng = stream(seed, 510)
        designs = [context_design(theta, size, rng)[0] for theta in test]
        trials = config["replacement_trials"]
        repeated = np.repeat(test, trials, axis=0)
        for label in ("uniform", "well-conditioned", "nearly-redundant", "boundary-heavy"):
            contexts = ([sample_context(theta, size, rng) for theta in test] if label == "uniform"
                        else [design[label] for design in designs])
            original = [obs for obs in contexts for _ in range(trials)]
            modified = []
            for theta, obs in zip(repeated, original):
                index = rng.integers(size)
                # Keep replacement demands inside each context's sampled interval.
                record = solve_dispatch(theta, rng.uniform(obs.demand.min(), obs.demand.max(), 1))
                modified.append(replace(obs, index, record))
            rows = paired_rows(seed, label, models, ridge, original, modified, repeated, repeated, batch)
            for row in rows:
                row["trial"] = row["task"] % trials
                row["task"] //= trials
            within.extend(rows)

        crossed = [list(crossed_backgrounds(theta, size)) for theta in test]
        repeated = np.repeat(test, 2, axis=0)
        for offset, label in enumerate(("high-leverage", "low-leverage")):
            records = [design[index] for design in crossed for index in (offset, offset + 2)]
            original = [row[2] for row in records]
            modified = [replace(obs, -1, row[3]) for obs, row in zip(original, records)]
            leverage = [hat_leverage(obs) for obs in original]
            perturbations = [("clean", original, modified)]
            for noise in config["price_noise_amplitudes"]:
                # Change only the measured price of the same focal record.
                minus, plus = [], []
                for obs in original:
                    focal = obs.take(slice(-1, None))
                    minus.append(replace(obs, -1, focal._replace(price=focal.price - noise)))
                    plus.append(replace(obs, -1, focal._replace(price=focal.price + noise)))
                perturbations.append((f"price-{noise:g}", minus, plus))
            for intervention, left, right in perturbations:
                for row in paired_rows(seed, label, models, ridge, left, right, repeated, repeated, batch):
                    idx = row["task"]
                    row.update(task=idx // 2, side=records[idx][0], intervention=intervention,
                               focal_leverage=leverage[idx])
                    matched.append(row)

        for m in lengths:
            theta, contexts = theorem_pair(m)
            stress.extend(paired_rows(seed, "identifying-record", models, ridge,
                                      contexts[:1], contexts[1:], theta[:1], theta[1:], batch))
        print(f"seed={seed} G completed, ridge penalty={penalty:g}", flush=True)

    metrics = ("relative_error", "replacement_relative_error", "feature_displacement")
    detailed_summary(output, "regimes", within, ("case", "model"), metrics)
    detailed_summary(output, "matched", matched, ("case", "intervention", "model"),
                     (*metrics, "focal_leverage", "condition_number", "ridge_response_distance"))
    detailed_summary(output, "identifying_pair", stress, ("m", "model"),
                     (*metrics, "condition_number", "max_absolute_error", "target_direction_response"))
    relation = []
    for seed in study_config["seeds"]:
        for model in sorted({row["model"] for row in matched}):
            for intervention in sorted({row["intervention"] for row in matched}):
                selected = [row for row in matched if row["model"] == model
                            and row["seed"] == seed and row["intervention"] == intervention]
                rho = spearmanr([row["condition_number"] for row in selected],
                                [row["delta"] for row in selected]).statistic
                relation.append({"seed": seed, "model": model, "intervention": intervention,
                                 "spearman_condition_vs_delta": rho})
    save_summary(output, "condition_relationship", relation, ("model", "intervention"),
                 ("spearman_condition_vs_delta",), save_raw=False)
    write_csv(output / "norm_bounds.csv", bounds)
    write_csv(output / "ridge_validation.csv", tuning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/leverage_study.json"))
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    study_config = json.loads(Path(config["study_config"]).read_text(encoding="utf-8"))
    output = args.root / "leverage_stability"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"experiment": config, "training": study_config},
                                                  indent=2) + "\n", encoding="utf-8")
    run(config, study_config, args.root, output)


if __name__ == "__main__":
    main()
