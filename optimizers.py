"""
optimizers.py  (v3 -- GA, PSO, DE, and NSGA-II integrated in one file,
                      all using REAL constraint handling, no manual penalty)

Why the penalty method was dropped entirely
--------------------------------------------
v1 used a hand-rolled penalty: fitness = cost + W * max(0, deficit).
The failure mode you hit is generic to that approach: if growth(x) == 0
for nearly every x the optimizer tries (which happened here because of
a medium-construction bug), the penalty term collapses to a CONSTANT
(since deficit = threshold*growth_ref - 0 is the same number regardless
of x). Minimizing "cost + constant" is mathematically identical to
minimizing cost ALONE -- the search silently stops caring about growth
at all and walks straight to cost=0. That's exactly "optimizing cost
without even looking at growth."

v3 fixes this at the ALGORITHM level, not just the medium level: every
solver here uses a REAL constraint mechanism that can never degenerate
like that, and every solver reports explicitly whether it actually found
a feasible point -- no more silently-wrong "confident-looking" answers.

Four algorithms, one shared interface
--------------------------------------
Every function/class here consumes `evaluate_fn(x) -> object with .growth
and .cost` (normally GEMEvaluator.evaluate) and shares ONE Problem
definition (_ConstrainedMediaProblem) for the three constrained,
single-objective solvers:

  - random_search        : naive uniform sampling baseline (kept hand-written
                            on purpose -- there's no "off the shelf" substitute
                            for "just sample randomly")
  - solve_ga              : pymoo GA          (single-objective, constrained)
  - solve_pso             : pymoo PSO         (single-objective, constrained)
  - solve_de              : scipy differential_evolution (constrained via
                             a real NonlinearConstraint)
  - pareto_front          : pymoo NSGA-II     (TRUE 2-objective: minimize cost,
                             maximize growth, NO constraint baked in --
                             the 90% requirement is applied AFTERWARDS as a
                             filter over the resulting front)

run_all_optimizers() runs all four back to back on the identical problem
so you can compare them directly, which is the point of testing them
together.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from scipy.optimize import differential_evolution, NonlinearConstraint

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination


# ----------------------------------------------------------------------
# Shared result type
# ----------------------------------------------------------------------
@dataclass
class OptResult:
    algorithm: str
    x: Optional[np.ndarray]
    cost: float
    growth: float
    feasible: bool
    note: str = ""


# ----------------------------------------------------------------------
# 1. Random search (baseline) -- kept hand-written, see module docstring
# ----------------------------------------------------------------------
def random_search(evaluate_fn, bounds, growth_ref, threshold=0.9, n_samples=2000, seed=None) -> OptResult:
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    d = len(bounds)

    history = []  # (growth, cost, feasible) for every sample -> raw cloud for plotting
    best_x, best_cost, best_res = None, np.inf, None

    for _ in range(n_samples):
        x = lb + rng.random(d) * (ub - lb)
        res = evaluate_fn(x)
        feasible = res.growth >= threshold * growth_ref
        history.append((res.growth, res.cost, feasible))
        if feasible and res.cost < best_cost:
            best_x, best_cost, best_res = x, res.cost, res

    if best_x is None:
        return OptResult("random_search", None, np.inf, 0.0, False,
                          note=f"no feasible point found in {n_samples} samples"), history
    return OptResult("random_search", best_x, best_res.cost, best_res.growth, True), history


# ----------------------------------------------------------------------
# Shared pymoo Problem: minimize cost(x), s.t. growth(x) >= threshold*growth_ref
# pymoo convention: feasible iff g(x) <= 0, so g(x) = threshold*growth_ref - growth(x)
# ----------------------------------------------------------------------
class _ConstrainedMediaProblem(ElementwiseProblem):
    def __init__(self, evaluate_fn, bounds, growth_ref, threshold):
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        super().__init__(n_var=len(bounds), n_obj=1, n_ieq_constr=1, xl=lb, xu=ub)
        self.evaluate_fn = evaluate_fn
        self.growth_ref = growth_ref
        self.threshold = threshold

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.evaluate_fn(x)
        out["F"] = [res.cost]
        out["G"] = [self.threshold * self.growth_ref - res.growth]


def _run_pymoo_single_objective(algorithm_name, algorithm, evaluate_fn, bounds, growth_ref,
                                 threshold, n_gen, seed, verbose):
    problem = _ConstrainedMediaProblem(evaluate_fn, bounds, growth_ref, threshold)
    res = pymoo_minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=seed, verbose=verbose)

    if res.X is None:
        return OptResult(algorithm_name, None, np.inf, 0.0, False,
                          note="pymoo found no feasible individual across the whole run")

    x = res.X
    result = evaluate_fn(x)
    cv = float(np.atleast_1d(res.CV)[0]) if res.CV is not None else 0.0
    feasible = cv <= 1e-6
    note = "" if feasible else f"best individual still violates constraint by {cv:.4f}"
    return OptResult(algorithm_name, x, result.cost, result.growth, feasible, note=note)


# ----------------------------------------------------------------------
# 2. Genetic Algorithm -- pymoo GA (constrained, single-objective)
# ----------------------------------------------------------------------
def solve_ga(evaluate_fn, bounds, growth_ref, threshold=0.9,
             pop_size=60, n_gen=150, seed=None, verbose=False) -> OptResult:
    algorithm = GA(pop_size=pop_size)
    return _run_pymoo_single_objective("GA (pymoo)", algorithm, evaluate_fn, bounds,
                                        growth_ref, threshold, n_gen, seed, verbose)


# ----------------------------------------------------------------------
# 3. Particle Swarm Optimization -- pymoo PSO (constrained, single-objective)
# ----------------------------------------------------------------------
def solve_pso(evaluate_fn, bounds, growth_ref, threshold=0.9,
              pop_size=40, n_gen=150, seed=None, verbose=False) -> OptResult:
    algorithm = PSO(pop_size=pop_size)
    return _run_pymoo_single_objective("PSO (pymoo)", algorithm, evaluate_fn, bounds,
                                        growth_ref, threshold, n_gen, seed, verbose)


# ----------------------------------------------------------------------
# 4. Differential Evolution -- scipy, with a REAL NonlinearConstraint
# ----------------------------------------------------------------------
def solve_de(evaluate_fn, bounds, growth_ref, threshold=0.9,
             maxiter=200, popsize=20, seed=None, disp=False) -> OptResult:
    def cost_objective(x):
        return evaluate_fn(x).cost

    def growth_constraint_fn(x):
        return evaluate_fn(x).growth

    nlc = NonlinearConstraint(growth_constraint_fn, lb=threshold * growth_ref, ub=np.inf)

    result = differential_evolution(
        cost_objective, bounds=bounds, constraints=(nlc,),
        maxiter=maxiter, popsize=popsize, seed=seed, polish=False, disp=disp,
    )
    eval_res = evaluate_fn(result.x)
    feasible = eval_res.growth >= threshold * growth_ref - 1e-6
    note = "" if result.success and feasible else (
        f"scipy success={result.success}; " + (result.message or "")
        + ("" if feasible else f"  [constraint violated: growth={eval_res.growth:.4f} < {threshold*growth_ref:.4f}]")
    )
    return OptResult("DE (scipy)", result.x, eval_res.cost, eval_res.growth, feasible, note=note)


# ----------------------------------------------------------------------
# 5. True multi-objective Pareto front -- pymoo NSGA-II (NO constraint)
# ----------------------------------------------------------------------
class _BiObjectiveMediaProblem(ElementwiseProblem):
    """minimize [cost(x), -growth(x)] -- no floor on growth at all."""
    def __init__(self, evaluate_fn, bounds):
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        super().__init__(n_var=len(bounds), n_obj=2, n_ieq_constr=0, xl=lb, xu=ub)
        self.evaluate_fn = evaluate_fn

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.evaluate_fn(x)
        out["F"] = [res.cost, -res.growth]


def pareto_front(evaluate_fn, bounds, pop_size=80, n_gen=120, seed=None, verbose=False) -> List[dict]:
    problem = _BiObjectiveMediaProblem(evaluate_fn, bounds)
    algorithm = NSGA2(pop_size=pop_size)
    res = pymoo_minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=seed, verbose=verbose)

    front = []
    if res.X is not None:
        X = res.X if res.X.ndim > 1 else res.X.reshape(1, -1)
        F = res.F if res.F.ndim > 1 else res.F.reshape(1, -1)
        for x, f in zip(X, F):
            front.append({"x": x, "cost": float(f[0]), "growth": float(-f[1])})
    front.sort(key=lambda p: p["cost"])
    return front


def select_recommended_from_front(front, growth_ref, threshold=0.9) -> Optional[dict]:
    feasible = [p for p in front if p["growth"] >= threshold * growth_ref]
    if not feasible:
        return None
    return min(feasible, key=lambda p: p["cost"])


# ----------------------------------------------------------------------
# Run all four together (this is the actual "test all 4 together" entry point)
# ----------------------------------------------------------------------
def run_all_optimizers(evaluate_fn, bounds, growth_ref, threshold=0.9, seed=42, verbose=False):
    """
    Runs random search, GA, PSO, DE (all solving the identical constrained
    problem) plus NSGA-II (solving the unconstrained bi-objective version,
    then filtered to the threshold for comparison), and returns everything
    needed for a side-by-side comparison table + combined plot.
    """
    results = {}

    rs_result, rs_history = random_search(evaluate_fn, bounds, growth_ref, threshold, seed=seed)
    results["random_search"] = rs_result

    results["ga"] = solve_ga(evaluate_fn, bounds, growth_ref, threshold, seed=seed, verbose=verbose)
    results["pso"] = solve_pso(evaluate_fn, bounds, growth_ref, threshold, seed=seed, verbose=verbose)
    results["de"] = solve_de(evaluate_fn, bounds, growth_ref, threshold, seed=seed)

    front = pareto_front(evaluate_fn, bounds, seed=seed, verbose=verbose)
    recommended = select_recommended_from_front(front, growth_ref, threshold)
    if recommended is not None:
        results["nsga2"] = OptResult("NSGA-II (pymoo, front-filtered)", recommended["x"],
                                      recommended["cost"], recommended["growth"], True)
    else:
        results["nsga2"] = OptResult("NSGA-II (pymoo, front-filtered)", None, np.inf, 0.0, False,
                                      note="no point on the Pareto front reached the threshold")

    return results, front, rs_history
