"""Task-count, compute-budget, and distribution-shift comparisons."""

import numpy as np

from .data import HIGH, LOW
from .evaluate import nested_contexts
from .study import load_transformer, save_summary, stream, tasks, train_model
from .train import predict


def fresh_batches(tasks, lengths, seed, channel, replicates):
    return [nested_contexts(tasks, lengths, stream(seed, channel + index))
            for index in range(replicates)]


def population_metrics(model, tasks, batches, length, batch_size):
    errors = [predict(model, contexts[length][:len(tasks)], batch_size) - tasks
              for contexts in batches]
    errors = np.stack(errors)
    risk = np.square(errors / (HIGH - LOW)).mean()
    relative = (np.linalg.norm(errors, axis=-1) / np.linalg.norm(tasks, axis=-1)).mean()
    return float(risk), float(relative)


def experiment_e(config, output, root):
    import csv

    lengths, batch_size = config["lengths"], config["batch_size"]
    replicates = config["risk_contexts_per_task"]
    count_max = max(config["source_counts"])
    d_training = root / "transformer_stability" / "training.csv"
    with d_training.open(newline="", encoding="utf-8") as handle:
        d_history = list(csv.DictReader(handle))
    rows = []
    for seed in config["seeds"]:
        source, valid, test = tasks(config, seed)
        valid_contexts = nested_contexts(valid, lengths, stream(seed, 1))
        source_batches = fresh_batches(source, lengths, seed, 100, replicates)
        target_batches = fresh_batches(test, lengths, seed, 200, replicates)
        # Shift density within the original support, preserving feasible output bounds.
        ood = LOW + (HIGH - LOW) * stream(seed, 301).beta(5, 1.5, size=test.shape)
        ood_batches = fresh_batches(ood, lengths, seed, 200, replicates)
        diversity_count = config["diversity_tasks"]
        narrow = (LOW + HIGH) / 2 + config["narrow_source_width"] * (
            source[:diversity_count] - (LOW + HIGH) / 2)
        narrow_batches = fresh_batches(narrow, lengths, seed, 100, replicates)
        shared = {}

        def score(model, source_tasks, batches, study, condition, epoch, updates, records):
            for m in lengths:
                source_risk, source_error = population_metrics(model, source_tasks, batches, m, batch_size)
                id_risk, id_error = population_metrics(model, test, target_batches, m, batch_size)
                ood_risk, ood_error = population_metrics(model, ood, ood_batches, m, batch_size)
                rows.append({
                    "seed": seed, "study": study, "condition": condition,
                    "T": len(source_tasks), "m": m, "selected_epoch": epoch,
                    "training_updates": updates, "training_records": records,
                    "source_risk": source_risk, "transfer_risk": id_risk,
                    "transfer_gap": id_risk - source_risk, "shift_risk": ood_risk,
                    "source_relative_error": source_error, "transfer_relative_error": id_error,
                    "shift_relative_error": ood_error,
                })

        for count in config["source_counts"]:
            if count == count_max:
                model = load_transformer(config, seed, root)
                histories = [row for row in d_history if int(row["seed"]) == seed
                             and row["model"] == "transformer"]
                best = min(histories, key=lambda row: float(row["validation_relative_error"]))
                epoch = int(best["epoch"])
                steps = (count + batch_size - 1) // batch_size
                # Reproduce the record-count draws of the reused D training run.
                rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
                records = 0
                for _ in range(config["epochs"]):
                    rng.permutation(count)
                    for start in range(0, count, batch_size):
                        m = int(rng.choice(lengths))
                        records += min(batch_size, count - start) * m
                        rng.uniform(0, 2, size=(min(batch_size, count - start), m))
                updates = steps * config["epochs"]
            else:
                model, epoch, history = train_model(config, seed, source[:count], valid,
                                                    valid_contexts, f"E equal passes T={count}",
                                                    track_budget=True)
                updates = sum(row["updates"] for row in history)
                records = sum(row["training_records"] for row in history)
            shared[count] = model, epoch, updates, records
            score(model, source[:count], source_batches, "equal-passes", "uniform",
                  epoch, updates, records)

        for count in config["source_counts"]:
            steps = config["matched_steps_per_epoch"]
            if count == steps * batch_size:
                model, epoch, updates, records = shared[count]
            else:
                model, epoch, history = train_model(config, seed, source[:count], valid,
                                                    valid_contexts, f"E equal steps T={count}",
                                                    steps_per_epoch=steps, track_budget=True)
                updates = sum(row["updates"] for row in history)
                records = sum(row["training_records"] for row in history)
            score(model, source[:count], source_batches, "equal-steps", "uniform",
                  epoch, updates, records)

        for count, m, batch in config["matched_records"]:
            model, epoch, history = train_model(config, seed, source[:count], valid, valid_contexts,
                                                f"E matched records T={count} m={m}",
                                                lengths=[m], batch_size=batch, track_budget=True)
            score(model, source[:count], source_batches, "equal-records", f"train-m{m}",
                  epoch, sum(row["updates"] for row in history),
                  sum(row["training_records"] for row in history))

        broad, epoch, updates, records = shared[diversity_count]
        score(broad, source[:diversity_count], source_batches, "diversity", "broad",
              epoch, updates, records)
        model, epoch, history = train_model(config, seed, narrow, valid, valid_contexts,
                                            "E narrow source", track_budget=True)
        score(model, narrow, narrow_batches, "diversity", "narrow", epoch,
              sum(row["updates"] for row in history),
              sum(row["training_records"] for row in history))

    metrics = ("source_risk", "transfer_risk", "transfer_gap", "shift_risk",
               "source_relative_error", "transfer_relative_error", "shift_relative_error",
               "training_updates", "training_records")
    save_summary(output, "risks", rows, ("study", "condition", "T", "m"), metrics,
                 save_raw=False)
