"""Matched contexts and valid-record interventions."""

import numpy as np
from scipy.stats import wasserstein_distance

from .data import observation_features, sample_context
from .dispatch import INTERIOR, Observations, solve_dispatch
from .identified import admissible_diameter, exact_kkt_geometry, exact_kkt_project
from .kkt import equality_gram
from .study import load_transformer, recovery_metrics, replace, save_summary, stream, tasks
from .train import predict

CONDITIONS = ("well-conditioned", "nearly-redundant", "mixed", "boundary-heavy")
INTERVENTIONS = ("high-replacement", "redundant-replacement", "drop-high", "drop-redundant",
                 "keep-interior", "keep-boundary", "shuffle")


def context_design(theta, size, rng):
    """Build every context from two pools of valid outcomes for the same task."""
    grid = solve_dispatch(theta, np.linspace(0, 2, 2049)[1:-1])
    interior = grid.take(np.flatnonzero(np.all(grid.status == INTERIOR, axis=1)))
    boundary = sample_context(theta, 128, rng, "boundary-heavy")
    half = size // 2
    # Pair records about the same interior center, matching demand means exactly.
    center = (len(interior.demand) - 1) // 2
    wide = np.rint(np.linspace(-.95 * center, .95 * center, size)).astype(int)
    narrow = np.r_[np.arange(-half, 0), np.arange(1, half + 1)]
    well = interior.take(center + wide)
    near = interior.take(center + narrow)
    boundary_rows = boundary.take(np.arange(size))
    mixed = Observations(*(np.concatenate((a[::2], b[:half])) for a, b in
                           zip(well, boundary_rows)))

    # Shift each chosen record by the same number of grid increments.
    shift = 2 * (center // 10)
    core = np.r_[np.arange(-half + 1, 0), np.arange(1, half)]
    base = interior.take(center + np.r_[core, 3 * shift, shift // 2])
    variants = {
        "high-replacement": replace(base, -2, interior.take([center + 2 * shift])),
        "redundant-replacement": replace(base, -1, interior.take([center - shift // 2])),
        "drop-high": base.take(np.delete(np.arange(size), -2)),
        "drop-redundant": base.take(slice(None, -1)),
        "keep-interior": mixed.take(slice(None, size - half)),
        "keep-boundary": mixed.take(slice(size - half, None)),
        "shuffle": well.take(rng.permutation(size)),
    }
    refs = {label: base for label in INTERVENTIONS[:4]}
    refs.update({"keep-interior": mixed, "keep-boundary": mixed, "shuffle": well})
    return dict(zip(CONDITIONS, (well, near, mixed, boundary_rows))), variants, refs


def information(obs):
    return np.linalg.eigvalsh(equality_gram(obs)).min()


def experiment_f(config, output, root):
    size, batch = config["info_context_length"], config["batch_size"]
    if size < 4 or size > 64 or size % 2:
        raise ValueError("informativeness context size must be even and between 4 and 64")
    contexts_rows, intervention_rows = [], []
    for seed in config["seeds"]:
        source, _, test = tasks(config, seed)
        model = load_transformer(config, seed, root)
        prior = source.mean(axis=0)
        rng, query_rng = stream(seed, 400), stream(seed, 401)
        designs = [context_design(theta, size, rng) for theta in test]
        queries = [sample_context(theta, 64, query_rng) for theta in test]
        for label in CONDITIONS:
            contexts = [design[0][label] for design in designs]
            for idx, (theta, query, obs, anchor) in enumerate(
                    zip(test, queries, contexts, predict(model, contexts, batch))):
                vertices = exact_kkt_geometry(obs)
                diameter = admissible_diameter(vertices)
                estimates = {"transformer": anchor,
                             "exact-prior": exact_kkt_project(prior, obs, vertices),
                             "exact-learned": exact_kkt_project(anchor, obs, vertices)}
                for name, estimate in estimates.items():
                    contexts_rows.append({
                        "seed": seed, "task": idx, "condition": label, "model": name,
                        "identification": "point" if diameter <= 1e-9 else "partial",
                        "min_gram_eigenvalue": information(obs), "admissible_diameter": diameter,
                        "both_interior_fraction": np.mean(np.all(obs.status == INTERIOR, axis=1)),
                        "demand_mean": obs.demand.mean(), "demand_sd": obs.demand.std(),
                        "demand_w1_to_well": wasserstein_distance(
                            obs.demand, designs[idx][0]["well-conditioned"].demand),
                        **recovery_metrics(estimate, theta, query),
                    })

        for label in INTERVENTIONS:
            variants = [design[1][label] for design in designs]
            references = [design[2][label] for design in designs]
            estimates = predict(model, variants, batch)
            baseline = predict(model, references, batch)
            for idx, (theta, query, obs, ref, estimate, anchor) in enumerate(
                    zip(test, queries, variants, references, estimates, baseline)):
                metrics = recovery_metrics(estimate, theta, query)
                base_metrics = recovery_metrics(anchor, theta, query)
                if label.endswith("-replacement"):
                    features = observation_features(obs) - observation_features(ref)
                    displacement = np.linalg.norm(features, axis=1).max()
                else:
                    displacement = 0.0
                intervention_rows.append({
                    "seed": seed, "task": idx, "intervention": label, "m": len(obs.demand),
                    "delta": np.linalg.norm(estimate - anchor),
                    "relative_error_change": metrics["relative_error"] - base_metrics["relative_error"],
                    "marginal_rmse_change": metrics["marginal_cost_rmse"] - base_metrics["marginal_cost_rmse"],
                    "gram_drop": information(ref) - information(obs),
                    "min_gram_eigenvalue": information(obs),
                    "admissible_diameter": admissible_diameter(exact_kkt_geometry(obs)),
                    "demand_w1": wasserstein_distance(obs.demand, ref.demand),
                    "replacement_feature_distance": displacement,
                    **metrics,
                })
        print(f"seed={seed} F contexts and interventions complete", flush=True)

    save_summary(output, "contexts", contexts_rows, ("condition", "model", "identification"),
                 ("relative_error", "marginal_cost_rmse", "min_gram_eigenvalue",
                  "admissible_diameter", "both_interior_fraction", "demand_mean", "demand_sd",
                  "demand_w1_to_well"))
    save_summary(output, "interventions", intervention_rows, ("intervention", "m"),
                 ("delta", "relative_error", "relative_error_change", "marginal_cost_rmse",
                  "marginal_rmse_change", "gram_drop", "min_gram_eigenvalue",
                  "admissible_diameter", "demand_w1", "replacement_feature_distance"))
