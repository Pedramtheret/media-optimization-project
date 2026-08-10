"""
run_optimization.py

ONE unified pipeline (replaces the old run_optimization / run_optimization2
split) for B. longum NCC2705 on MRS-derived media:

  1. load model, build CultureMedium, measure realistic growth reference
  2. essentiality scan
  3. run ALL FOUR optimizers on the identical 90%-constrained problem,
     side by side: random search (baseline), GA (pymoo), PSO (pymoo),
     DE (scipy) -- plus NSGA-II (pymoo) for the true Pareto front, whose
     result is filtered post-hoc at the 90% threshold for direct
     comparison against the other three
  4. comparison table across all four/five results
  5. Pareto front plot with all optimizer solutions marked on it
  6. sensitivity analysis around the best solution found
  7. recommended-media table
  8. top-flux report (+ tryptophan/kynurenine flag)

BEFORE running this on your real model:
    python test_optimizers.py     # validates GA/PSO/DE/NSGA-II logic
    python test_integration.py    # validates the medium-mapping + the
                                   # composite-overwrite regression test
Both should print ALL CHECKS PASSED. If either fails, fix that first --
this script's output is only trustworthy once both pass.

Point MODEL_PATH at your downloaded AGORA2 SBML file and run:
    python run_optimization.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import cobra

from media_components import build_bifidobacterium_mrs_medium
from gem_evaluator import GEMEvaluator
from optimizers import run_all_optimizers
from pathlib import Path

MODEL_PATH = Path("D:/workspace/Hakim project/AGORA2_SBML/Bifidobacterium_longum_NCC2705.xml")
GROWTH_THRESHOLD = 0.90


def main():
    print("Loading model ...")
    model = cobra.io.read_sbml_model(MODEL_PATH)

    medium = build_bifidobacterium_mrs_medium()
    evaluator = GEMEvaluator(model, medium)

    # ---- 1. realistic reference growth ----
    growth_ref = evaluator.measure_reference()
    print(f"\nReference growth rate (richest MRS-derived medium): {growth_ref:.4f} h^-1")
    if growth_ref <= 0:
        print("Reference growth is 0 -- check the essentiality scan below before optimizing.")
        sys.exit(1)

    # ---- 2. essentiality scan ----
    print("\nEssentiality scan (component removed -> resulting growth):")
    for name, g, pct in evaluator.essentiality_scan():
        flag = "  <-- ESSENTIAL" if pct < 5 else ("  <-- minor/no effect, consider dropping from search" if pct > 99 else "")
        print(f"   {name:35s} growth={g:.4f}  ({pct:5.1f}% of ref){flag}")
    print("\n   NOTE: if this still shows every single component as 100% essential (0% growth when\n"
          "   ANY one is removed, even with the other 8 wide open), that's now a genuine signal\n"
          "   about this model's biomass equation rather than the old overwrite bug -- worth a\n"
          "   quick manual look at the biomass reaction's stoichiometry if so.")

    bounds = medium.bounds

    # ---- 3. run all four optimizers together ----
    print(f"\nRunning random search, GA, PSO, DE (all constrained to >={GROWTH_THRESHOLD*100:.0f}% growth), "
          f"plus NSGA-II Pareto front ...")
    results, front, rs_history = run_all_optimizers(
        evaluator.evaluate, bounds, growth_ref, threshold=GROWTH_THRESHOLD, seed=42, verbose=True
    )

    # ---- 4. comparison table ----
    print("\n" + "=" * 72)
    print("COMPARISON ACROSS ALL FOUR (FIVE) METHODS")
    print("=" * 72)
    print(f"   {'Method':28s} {'Feasible':>9s} {'Cost':>10s} {'Growth %ref':>12s}")
    for key in ["random_search", "ga", "pso", "de", "nsga2"]:
        res = results[key]
        cost_str = f"{res.cost:.4f}" if res.feasible else "n/a"
        growth_str = f"{100*res.growth/growth_ref:.1f}%" if res.feasible else "n/a"
        print(f"   {res.algorithm:28s} {str(res.feasible):>9s} {cost_str:>10s} {growth_str:>12s}"
              + (f"   ({res.note})" if res.note else ""))

    feasible_results = {k: r for k, r in results.items() if r.feasible and k != "random_search"}
    if not feasible_results:
        print("\nNo constrained optimizer (GA/PSO/DE/NSGA-II) found a feasible solution.\n"
              "This means the 90% threshold is not reachable anywhere within your current\n"
              "component bounds -- widen upper bounds (especially on the extract components)\n"
              "before re-running, rather than trusting any single result here.")
        sys.exit(1)

    best_key = min(feasible_results, key=lambda k: feasible_results[k].cost)
    best = feasible_results[best_key]
    print(f"\nBest feasible solution overall: {best.algorithm}  (cost={best.cost:.4f}, "
          f"growth={100*best.growth/growth_ref:.1f}% of ref)")

    # ---- 5. plot ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if rs_history:
        rs_growth = [100 * g / growth_ref for g, c, feas in rs_history]
        rs_cost = [c for g, c, feas in rs_history]
        ax.scatter(rs_cost, rs_growth, s=8, alpha=0.15, color="gray", label="random search (raw samples)")
    if front:
        front_growth = [100 * p["growth"] / growth_ref for p in front]
        front_cost = [p["cost"] for p in front]
        ax.plot(front_cost, front_growth, "-", color="#0B4F52", linewidth=2, alpha=0.7, label="NSGA-II Pareto front")

    markers = {"ga": ("o", "#D9752B"), "pso": ("s", "#3F8C87"), "de": ("^", "#B5544A"), "nsga2": ("*", "#6A4C93")}
    for key, (marker, color) in markers.items():
        res = results[key]
        if res.feasible:
            ax.scatter([res.cost], [100 * res.growth / growth_ref], marker=marker, s=140,
                       color=color, edgecolor="black", zorder=5, label=res.algorithm)

    ax.axhline(GROWTH_THRESHOLD * 100, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label=f"{GROWTH_THRESHOLD*100:.0f}% growth threshold")
    ax.set_xlabel("Medium cost (placeholder $ units -- replace with your literature costs)")
    ax.set_ylabel("Growth (% of reference)")
    ax.set_title("Growth vs. Cost \u2014 B. longum NCC2705 on MRS-derived medium\nAll four methods compared")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("pareto_front.png", dpi=150)
    print("\nSaved plot -> pareto_front.png")

    # ---- 6. sensitivity analysis around the best solution ----
    print(f"\nLocal sensitivity analysis around the best solution ({best.algorithm}):")
    for name, dg, dc in evaluator.sensitivity_analysis(best.x):
        print(f"   {name:35s} d(growth)/d(x) = {dg:+.5f}   d(cost)/d(x) = {dc:+.5f}")

    # ---- 7. recommended media table ----
    print(f"\nRecommended medium (cheapest feasible solution, from {best.algorithm}):")
    print(f"   {'Component':35s} {'Value':>10s}")
    for comp, val in zip(medium.components, best.x):
        print(f"   {comp.name:35s} {val:10.4f}")
    print(f"   {'TOTAL COST':35s} {best.cost:10.4f}")
    print(f"   {'GROWTH':35s} {best.growth:10.4f}  ({100*best.growth/growth_ref:.1f}% of ref)")

    # ---- 8. flux interpretation ----
    print("\nTop fluxes at the recommended medium:")
    top_fluxes, trp_related = evaluator.top_flux_report(best.x, n=15)
    for rxn_id, name, subsystem, flux in top_fluxes:
        print(f"   {rxn_id:25s} {flux:+9.3f}   {name} [{subsystem}]")

    if trp_related:
        print("\nTryptophan/kynurenine-related reactions found in the network:")
        for rxn_id, name, subsystem, flux in trp_related:
            print(f"   {rxn_id:25s} {flux:+9.3f}   {name} [{subsystem}]")
    else:
        print("\nNo tryptophan/kynurenine-labeled reactions found among nonzero fluxes at this medium.")


if __name__ == "__main__":
    main()
