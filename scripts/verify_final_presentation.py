"""Read saved outputs and verify FINAL slide bounds; never train or edit results."""

import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "results/leverage_stability/identifying_pair_raw.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["model"] == "transformer" and int(r["m"]) == 64]
    assert len(rows) == 3 and len({r["seed"] for r in rows}) == 3
    gap = math.sqrt(1.6)
    checks = []
    for row in rows:
        response = float(row["target_direction_response"])
        delta = float(row["delta"])
        maximum_error = float(row["max_absolute_error"])
        error_a = float(row["relative_error"]) * math.sqrt(2 * (1.6**2 + 1.0**2))
        error_b = float(row["replacement_relative_error"]) * math.sqrt(2 * (1.2**2 + 1.8**2))
        assert math.isclose(max(error_a, error_b), maximum_error, abs_tol=1e-12)
        directional = gap / 2 * abs(1 - response)
        euclidean = max(0., (gap - delta) / 2)
        assert maximum_error + 1e-6 >= directional >= euclidean - 1e-6
        checks.append(dict(seed=int(row["seed"]), r=response, delta=delta,
                           directional_bound=directional, euclidean_bound=euclidean,
                           error_a=error_a, error_b=error_b,
                           maximum_endpoint_error=maximum_error))
    avg = dict(directional_bound=mean(r["directional_bound"] for r in checks),
               euclidean_bound=mean(r["euclidean_bound"] for r in checks),
               maximum_endpoint_error=mean(r["maximum_endpoint_error"] for r in checks),
               r=mean(r["r"] for r in checks))
    assert all(r["r"] < 1 for r in checks)
    assert math.isclose(avg["directional_bound"], gap / 2 * abs(1 - avg["r"]),
                        abs_tol=1e-12)
    with (ROOT / "results/leverage_stability/identifying_pair_summary.csv").open() as stream:
        summary = next(r for r in csv.DictReader(stream)
                       if r["model"] == "transformer" and int(r["m"]) == 64)
    assert math.isclose(avg["maximum_endpoint_error"],
                        float(summary["max_absolute_error_mean"]), abs_tol=1e-12)
    assert math.isclose(avg["r"], float(summary["target_direction_response_mean"]),
                        abs_tol=1e-12)
    output = dict(target_gap=gap, checkpoints=checks, checkpoint_means=avg,
                  averaging="mean_j(max(e_Aj,e_Bj)); not max(mean_j(e_Aj),mean_j(e_Bj))",
                  tolerance=1e-6,
                  source="results/leverage_stability/identifying_pair_raw.csv")
    (ROOT / "COURSE_PROJECT_FINAL_CHECKS.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    macros = dict(FinalGap=gap, FinalDirectional=avg["directional_bound"],
                  FinalEuclidean=avg["euclidean_bound"],
                  FinalObserved=avg["maximum_endpoint_error"])
    (ROOT / "COURSE_PROJECT_FINAL_NUMBERS.tex").write_text(
        "% Generated from saved per-checkpoint outputs; absolute parameter units.\n"
        + "".join(r"\newcommand{\%s}{%.3f}" % (name, value) + "\n"
                  for name, value in macros.items()), encoding="utf-8", newline="\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
