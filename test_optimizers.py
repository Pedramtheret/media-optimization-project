"""
test_optimizers.py

Validates optimizers.py (random search, GA, PSO, DE, NSGA-II) against a
SYNTHETIC stand-in for FBA, so the search logic is proven correct
independent of the real, slower cobra model and independent of any
medium-mapping bugs.

RUN THIS FIRST, before run_optimization.py, any time optimizers.py or
media_components.py changes. If this fails, the bug is in the
optimization logic; if it passes but run_optimization.py still looks
wrong, the bug is almost certainly in the model/medium mapping instead
(check test_integration.py's regression test next).

Beyond checking each algorithm individually, this ALSO checks that all
four AGREE with each other on the same problem (similar cost ballpark) --
that cross-agreement is itself a strong correctness signal: four
independent algorithms converging on similar answers is exactly the kind
of check that would have caught the original penalty-collapse bug (GA
cost=0.0 and PSO cost=0.0 would have flagged instantly against DE's very
different, non-zero answer).

    python test_optimizers.py
"""

import numpy as np
from dataclasses import dataclass
from optimizers import (
    random_search, solve_ga, solve_pso, solve_de,
    pareto_front, select_recommended_from_front, run_all_optimizers,
)


@dataclass
class FakeResult:
    growth: float
    cost: float


REQUIREMENTS = np.array([10.0, 4.0, 2.0, 1.0, 0.5])
COSTS = np.array([0.02, 0.05, 0.1, 0.3, 0.5])
MAX_GROWTH = 0.8  # h^-1
BOUNDS = [(0.0, 20.0), (0.0, 8.0), (0.0, 4.0), (0.0, 2.0), (0.0, 1.0)]
THRESHOLD = 0.9


def fake_evaluate(x):
    x = np.asarray(x)
    ratios = x / REQUIREMENTS
    growth = MAX_GROWTH * np.clip(np.min(ratios), 0, 1.3)
    cost = float(np.sum(COSTS * x))
    return FakeResult(growth=float(growth), cost=cost)


def check_feasible(label, res, growth_ref):
    print(f"   {label:12s} feasible={res.feasible}  cost={res.cost:.4f}  "
          f"growth={res.growth:.4f} ({100*res.growth/growth_ref:.1f}% of ref)"
          + (f"   note: {res.note}" if res.note else ""))
    if res.feasible:
        assert res.growth >= THRESHOLD * growth_ref - 1e-2, \
            f"{label} claims feasible but violates the {THRESHOLD*100:.0f}% constraint!"


def main():
    growth_ref = MAX_GROWTH

    print("=" * 60)
    print("1) Random search baseline")
    rs_res, rs_history = random_search(fake_evaluate, BOUNDS, growth_ref, threshold=THRESHOLD, n_samples=3000, seed=1)
    check_feasible("random", rs_res, growth_ref)
    assert rs_res.feasible, "random search should find SOME feasible point in this easy synthetic problem"

    print("=" * 60)
    print("2) GA (pymoo, constrained)")
    ga_res = solve_ga(fake_evaluate, BOUNDS, growth_ref, threshold=THRESHOLD, pop_size=50, n_gen=100, seed=1)
    check_feasible("GA", ga_res, growth_ref)
    assert ga_res.feasible, "GA must find a feasible solution on this easy synthetic problem"
    assert ga_res.cost > 0.01, "GA cost collapsing near 0 is exactly the old penalty-method failure signature"

    print("=" * 60)
    print("3) PSO (pymoo, constrained)")
    pso_res = solve_pso(fake_evaluate, BOUNDS, growth_ref, threshold=THRESHOLD, pop_size=40, n_gen=100, seed=1)
    check_feasible("PSO", pso_res, growth_ref)
    assert pso_res.feasible
    assert pso_res.cost > 0.01

    print("=" * 60)
    print("4) DE (scipy, constrained)")
    de_res = solve_de(fake_evaluate, BOUNDS, growth_ref, threshold=THRESHOLD, maxiter=200, popsize=20, seed=1)
    check_feasible("DE", de_res, growth_ref)
    assert de_res.feasible
    assert de_res.cost > 0.01

    print("=" * 60)
    print("5) Cross-agreement check: GA, PSO, DE should land in a similar cost ballpark")
    costs = [ga_res.cost, pso_res.cost, de_res.cost]
    spread = (max(costs) - min(costs)) / min(costs)
    print(f"   costs = {[round(c, 4) for c in costs]}   relative spread = {spread:.2%}")
    assert spread < 0.5, (
        "GA/PSO/DE disagree by more than 50% on an identical constrained problem -- "
        "on a problem this simple that's a red flag, not just noise"
    )

    print("=" * 60)
    print("6) NSGA-II true Pareto front (no constraint) + threshold filter")
    front = pareto_front(fake_evaluate, BOUNDS, pop_size=60, n_gen=100, seed=1)
    print(f"   front has {len(front)} non-dominated points")
    costs_front = [p["cost"] for p in front]
    growths_front = [p["growth"] for p in front]
    assert all(c2 >= c1 - 1e-6 for c1, c2 in zip(sorted(costs_front), sorted(costs_front)[1:]))
    assert max(growths_front) >= THRESHOLD * growth_ref, "front should reach at least the 90% region somewhere"

    recommended = select_recommended_from_front(front, growth_ref, threshold=THRESHOLD)
    assert recommended is not None
    print(f"   front-derived recommendation: cost={recommended['cost']:.4f}  "
          f"growth={100*recommended['growth']/growth_ref:.1f}% of ref")
    assert abs(recommended["cost"] - de_res.cost) / de_res.cost < 0.5, (
        "NSGA-II's front-derived recommendation should roughly agree with DE's direct constrained solve"
    )

    print("=" * 60)
    print("7) run_all_optimizers() convenience wrapper")
    results, front2, rs_hist2 = run_all_optimizers(fake_evaluate, BOUNDS, growth_ref, threshold=THRESHOLD, seed=1)
    for name, res in results.items():
        print(f"   {name:15s} feasible={res.feasible}  cost={res.cost if res.feasible else float('nan'):.4f}")
        assert res.feasible, f"{name} should be feasible on this easy synthetic problem"

    print("=" * 60)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
