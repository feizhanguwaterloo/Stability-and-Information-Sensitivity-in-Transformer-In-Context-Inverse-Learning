from copy import deepcopy
from itertools import islice

import numpy as np
import torch

from .data import observation_features, sample_context
from .model import SetTransformer


@torch.inference_mode()
def predict(model, contexts, batch_size):
    model.eval()
    device = next(model.parameters()).device
    batches = []
    for start in range(0, len(contexts), batch_size):
        features = np.stack([observation_features(obs)
                             for obs in contexts[start:start + batch_size]])
        batches.append(model(torch.as_tensor(features, device=device)).cpu().numpy())
    return np.concatenate(batches)


def task_batches(rng, count, batch_size):
    if count < 1 or batch_size < 1:
        raise ValueError("training requires at least one task and a positive batch size")
    while True:
        order = rng.permutation(count)
        for start in range(0, count, batch_size):
            yield order[start:start + batch_size]


def fit(config, train_tasks, validation_tasks, validation_contexts, model_factory=SetTransformer):
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config["cpu_threads"])
    requested = config["device"]
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_factory().to(torch.device(requested))
    project = getattr(model, "project_attention", None)
    if project is not None:
        project()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 1]))
    task_rng = (np.random.default_rng(np.random.SeedSequence([config["seed"], 2]))
                if config.get("match_observation_draws") else rng)
    batch_size = config["batch_size"]
    best_score = np.inf
    history = []

    steps = config.get("steps_per_epoch", (len(train_tasks) + batch_size - 1) // batch_size)
    batches = task_batches(task_rng, len(train_tasks), batch_size)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        batch_losses = []
        records = 0
        for indices in islice(batches, steps):
            tasks = train_tasks[indices]
            size = int(rng.choice(config["lengths"]))
            records += len(tasks) * size
            contexts = [sample_context(theta, size, rng) for theta in tasks]
            features = torch.as_tensor(np.stack([observation_features(obs) for obs in contexts]),
                                       device=requested)
            targets = torch.as_tensor(tasks, dtype=torch.float32, device=requested)
            loss = ((model(features) - targets) / model.span).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if project is not None:
                project()
            batch_losses.append(loss.item())

        scores = []
        for contexts in validation_contexts.values():
            estimates = predict(model, contexts, batch_size)
            scores.append(np.mean(np.linalg.norm(estimates - validation_tasks, axis=1)
                                  / np.linalg.norm(validation_tasks, axis=1)))
        score = float(np.mean(scores))
        if config.get("use_final_checkpoint") or score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if config.get("verbose", True):
            print(f"epoch {epoch:02d} train={np.mean(batch_losses):.5f} val={score:.5f}", flush=True)
        history.append({"epoch": epoch, "train_scaled_mse": np.mean(batch_losses),
                        "validation_relative_error": score})
        if config.get("track_budget"):
            history[-1].update(updates=steps, training_records=records)

    model.load_state_dict(best_state)
    return model.eval(), history, best_epoch
