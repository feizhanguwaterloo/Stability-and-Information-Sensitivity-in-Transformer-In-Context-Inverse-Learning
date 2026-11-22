"""Controlled follow-ups to the Transformer leverage and transfer studies."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .data import HIGH, LOW, sample_context
from .dispatch import INTERIOR, Observations, solve_dispatch
from .evaluate import nested_contexts, write_csv
from .kkt import equality_gram
from .leverage import ridge_estimate
from .model import NormControlledTransformer, attention_operators
from .study import load_transformer, relative_error, replace, save_summary, stream, tasks, train_model
from .study_info import context_design
from .study_leverage import detailed_summary
from .study_transfer import fresh_batches, population_metrics
from .train import predict


def identifying_family(config, rng):
    """Draw pairs that share pivot dispatch and price and differ at a second demand."""
    count = config["pair_count"]
    pivot = rng.uniform(.44, .56, size=(count, 2))
    direction = np.sign(pivot[:, 0] - .5)
    mid_price = rng.uniform(2.04, 2.13, size=count)
    mid_slope = rng.uniform(1.32, 1.48, size=count)
    slope_gap = config["target_distance"] / np.sqrt(2 + np.square(pivot).sum(axis=1))
    step = direction * config["price_distance"] / slope_gap

    def coefficients(slopes):
        intercepts = mid_price[:, None] - slopes[:, None] * pivot
        return np.stack((intercepts, np.broadcast_to(slopes[:, None], pivot.shape)), axis=-1).reshape(-1, 4)

    left = coefficients(mid_slope - slope_gap / 2)
    right = coefficients(mid_slope + slope_gap / 2)
    demand = np.column_stack((pivot.sum(axis=1), pivot.sum(axis=1) + 2 * step))
    if np.any((left < LOW) | (left > HIGH) | (right < LOW) | (right > HIGH)):
        raise ValueError("constructed parameters are outside the model domain")
    return left, right, demand


def identifying_path(theta_a, theta_b, demand, m, k):
    records_a, records_b = [solve_dispatch(theta, demand) for theta in (theta_a, theta_b)]
    if (not np.all(records_a.status == INTERIOR) or not np.all(records_b.status == INTERIOR)
            or not np.allclose(records_a.dispatch, records_b.dispatch, atol=1e-12, rtol=0)
            or abs(records_a.price[0] - records_b.price[0]) > 1e-12 or not 0 < k < m):
        raise ValueError("invalid identifying pair")
    indices = np.r_[np.zeros(m - k, dtype=int), np.ones(k, dtype=int)]
    left = records_a.take(indices)
    right = left._replace(price=np.r_[left.price[:-k], np.repeat(records_b.price[1], k)])
    path = [left]
    for index in range(m - k, m - 1):
        price = path[-1].price.copy()
        price[index] = records_b.price[1]
        path.append(left._replace(price=price))
    return [*path, right]


def stress_rows(config, training, seed, models, prior):
    left, right, demands = identifying_family(config, stream(seed, 600))
    target_delta = right - left
    target_distance = config["target_distance"]
    count = len(left)
    rows = []
    for m in config["pair_lengths"]:
        for k in config["informative_counts"]:
            paths = [identifying_path(a, b, d, m, k) for a, b, d in zip(left, right, demands)]
            contexts = [obs for path in paths for obs in path]
            for name, model in {**models, "ridge": None}.items():
                values = (predict(model, contexts, training["batch_size"]) if model is not None else
                          np.array([ridge_estimate(obs, prior, 1e-6) for obs in contexts]))
                values = values.reshape(count, k + 1, 4)
                endpoint = values[:, -1] - values[:, 0]
                delta = np.linalg.norm(endpoint, axis=1)
                max_edge = np.linalg.norm(np.diff(values, axis=1), axis=2).max(axis=1)
                maximum_error = np.maximum(np.linalg.norm(values[:, 0] - left, axis=1),
                                           np.linalg.norm(values[:, -1] - right, axis=1))
                error_floor = np.maximum(target_distance - delta, 0) / 2
                if np.any(maximum_error + 1e-12 < error_floor):
                    raise RuntimeError("the two-point inequality failed")
                alignment = np.einsum("ij,ij->i", endpoint, target_delta) / target_distance**2
                for i in range(count):
                    rows.append({
                        "seed": seed, "model": name, "m": m, "k": k, "pair": i,
                        "focal_side": int(np.sign(demands[i, 1] - demands[i, 0])),
                        "delta": delta[i], "max_path_edge": max_edge[i],
                        "max_absolute_error": maximum_error[i], "error_floor": error_floor[i],
                        "target_direction_response": alignment[i],
                        "min_gram_eigenvalue": np.linalg.eigvalsh(equality_gram(paths[i][0])).min(),
                    })
    return rows


def noise_groups(task_set, m, rng):
    cases = ("well-conditioned", "nearly-redundant", "boundary-heavy")
    designs = [context_design(theta, m, rng)[0] for theta in task_set]
    return {case: [design[case] for design in designs] for case in cases}


def noisy_contexts(contexts, draws, sigma):
    return [obs._replace(price=obs.price + sigma * noise) for obs, noise in zip(contexts, draws)]


def noise_rows(config, training, seed, model, source, validation, test):
    size = config["noise_length"]
    prior = source.mean(axis=0)
    repeats = config["noise_replicates"]
    valid_groups = noise_groups(validation, size, stream(seed, 610))
    test_groups = noise_groups(test, size, stream(seed, 611))
    validation_noise = stream(seed, 612).normal(size=(len(validation) * repeats, size))
    test_noise = stream(seed, 613).normal(size=(len(test) * repeats, size))
    targets = np.repeat(test, repeats, axis=0)
    validation_targets = np.repeat(validation, repeats, axis=0)
    rows, tuning = [], []
    for case, observations in test_groups.items():
        valid_contexts = [obs for obs in valid_groups[case] for _ in range(repeats)]
        test_contexts = [obs for obs in observations for _ in range(repeats)]
        eigenvalues = [np.linalg.eigvalsh(equality_gram(obs)).min() for obs in observations]
        for sigma in config["noise_sigma"]:
            valid_noisy = noisy_contexts(valid_contexts, validation_noise, sigma)
            test_noisy = noisy_contexts(test_contexts, test_noise, sigma)
            errors = []
            for penalty in config["ridge_penalties"]:
                estimates = np.array([ridge_estimate(obs, prior, penalty) for obs in valid_noisy])
                errors.append(relative_error(np.clip(estimates, LOW, HIGH), validation_targets).mean())
            best = int(np.argmin(errors))
            penalty = config["ridge_penalties"][best]
            tuning.extend({"seed": seed, "case": case, "sigma": sigma, "penalty": p,
                           "validation_error": error, "selected": j == best}
                          for j, (p, error) in enumerate(zip(config["ridge_penalties"], errors)))
            estimates = {
                "transformer": predict(model, test_noisy, training["batch_size"]),
                "ridge": np.clip([ridge_estimate(obs, prior, penalty) for obs in test_noisy], LOW, HIGH),
            }
            for name, predictions in estimates.items():
                for idx, error in enumerate(relative_error(predictions, targets)):
                    rows.append({"seed": seed, "case": case, "sigma": sigma, "model": name,
                                 "task": idx // repeats, "trial": idx % repeats, "relative_error": error,
                                 "min_gram_eigenvalue": eigenvalues[idx // repeats]})
    return rows, tuning


def cap_rows(config, training, seed, root, source, validation, test, validation_contexts):
    models = {"uncapped": load_transformer(training, seed, root)}
    for cap in config["attention_caps"]:
        models[str(cap)], _, _ = train_model(training, seed, source, validation, validation_contexts,
                                             f"cap {cap:g}", partial(NormControlledTransformer, cap))
    norms = [{"seed": seed, "model": name, "max_operator_norm": max(
        float(torch.linalg.matrix_norm(weight.detach(), ord=2)) for weight in attention_operators(model))}
        for name, model in models.items()]
    for row in norms:
        if row["model"] != "uncapped" and row["max_operator_norm"] > float(row["model"]) + 2e-6:
            raise RuntimeError("trained attention operator exceeds its cap")
    rng = stream(seed, 620)
    rows = []
    for m, observations in nested_contexts(test, training["lengths"], rng).items():
        replacements = [replace(obs, rng.integers(m), sample_context(theta, 1, rng))
                        for obs, theta in zip(observations, test)]
        for name, model in models.items():
            estimates = predict(model, observations, training["batch_size"])
            shifted = predict(model, replacements, training["batch_size"])
            for i, error in enumerate(relative_error(estimates, test)):
                rows.append({"seed": seed, "model": name, "m": m, "task": i,
                             "relative_error": error, "delta": np.linalg.norm(estimates[i] - shifted[i])})
    return models, rows, norms


def transfer_rows(config, training, seed, source, validation, test, validation_contexts):
    lengths, batch = training["lengths"], training["batch_size"]
    length = config["transfer_train_length"]
    targets = fresh_batches(test, lengths, seed, 630, training["risk_contexts_per_task"])
    rows = []
    for count in config["transfer_counts"]:
        if count % batch:
            raise ValueError("each task count must be a multiple of batch size")
        model, _, history = train_model(
            training, seed, source[:count], validation, validation_contexts, f"matched T={count}",
            lengths=[length], epochs=config["transfer_epochs"], use_final_checkpoint=True,
            steps_per_epoch=config["transfer_steps_per_epoch"], track_budget=True,
            match_observation_draws=True)
        updates = sum(row["updates"] for row in history)
        records = sum(row["training_records"] for row in history)
        for m in lengths:
            risk, error = population_metrics(model, test, targets, m, batch)
            rows.append({"seed": seed, "T": count, "m": m, "relative_error": error,
                         "risk": risk, "updates": updates, "records": records})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/followup.json"))
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = json.loads(Path(config["study_config"]).read_text(encoding="utf-8"))
    output = args.root / "leverage_followup"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"followup": config, "training": training}, indent=2)
                                        + "\n", encoding="utf-8")
    torch.set_num_threads(training["cpu_threads"])
    stress, noise, noise_tuning, caps, cap_norms, transfer = [], [], [], [], [], []
    for seed in training["seeds"]:
        source, validation, test = tasks(training, seed)
        validation_contexts = nested_contexts(validation, training["lengths"], stream(seed, 1))
        models, rows, norms = cap_rows(config, training, seed, args.root, source, validation,
                                       test, validation_contexts)
        caps.extend(rows)
        cap_norms.extend(norms)
        stress.extend(stress_rows(config, training, seed, models, source.mean(axis=0)))
        rows, tuning = noise_rows(config, training, seed, models["uncapped"], source, validation, test)
        noise.extend(rows)
        noise_tuning.extend(tuning)
        transfer.extend(transfer_rows(config, training, seed, source, validation, test, validation_contexts))
    detailed_summary(output, "identification", stress, ("m", "k", "model"),
                     ("max_path_edge", "max_absolute_error", "error_floor",
                      "target_direction_response", "min_gram_eigenvalue"))
    detailed_summary(output, "cap_sweep", caps, ("m", "model"), ("relative_error",))
    write_csv(output / "cap_norms.csv", cap_norms)
    save_summary(output, "noise", noise, ("case", "sigma", "model"),
                 ("relative_error", "min_gram_eigenvalue"))
    write_csv(output / "noise_tuning.csv", noise_tuning)
    save_summary(output, "matched_transfer", transfer, ("T", "m"),
                 ("relative_error", "risk", "updates", "records"))


if __name__ == "__main__":
    main()
