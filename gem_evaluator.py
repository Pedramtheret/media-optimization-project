"""
gem_evaluator.py

Wraps a cobra GEM + a CultureMedium into one object with a single method,
`evaluate(x)`, that every optimizer (random search, GA, PSO, epsilon-constraint
sweep) calls identically. Also provides essentiality scanning, local
sensitivity analysis, and a top-flux report for interpreting the solution.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class EvalResult:
    growth: float
    cost: float
    feasible: bool  # solver status was optimal (not infeasible)


class GEMEvaluator:
    def __init__(self, model, medium, growth_ref: float = None):
        """
        model       : a cobra.Model (already loaded)
        medium      : a CultureMedium instance (see media_components.py)
        growth_ref  : the realistic 100% reference growth rate (h^-1),
                      e.g. what you measured on full-bound MRS. If not
                      given yet, call `.measure_reference()` first.
        """
        self.model = model
        self.medium = medium
        self.growth_ref = growth_ref

    def measure_reference(self, x=None) -> float:
        """Run FBA at a given (or default = richest) point and store it as growth_ref."""
        if x is None:
            x = self.medium.default_vector()
        result = self.evaluate(x)
        self.growth_ref = result.growth
        return self.growth_ref

    def evaluate(self, x) -> EvalResult:
        """Apply candidate medium `x`, solve FBA, return growth + cost."""
        with self.model as m:  # cobra context manager -> auto-reverts bounds after
            self.medium.apply_to_model(m, x)
            solution = m.optimize()
            growth = solution.objective_value if solution.status == "optimal" else 0.0
            feasible = solution.status == "optimal" and growth is not None
        cost = self.medium.cost(x)
        return EvalResult(growth=growth or 0.0, cost=cost, feasible=feasible)

    def growth_fraction(self, x) -> float:
        """Growth achieved as a fraction of growth_ref (0-1+)."""
        if not self.growth_ref:
            raise ValueError("growth_ref not set -- call measure_reference() first")
        return self.evaluate(x).growth / self.growth_ref

    # ------------------------------------------------------------------
    # Essentiality / auxotrophy scan: zero out one component at a time
    # ------------------------------------------------------------------
    def essentiality_scan(self) -> List[Tuple[str, float, float]]:
        """
        For each component, set it to its lower bound (usually 0) while
        keeping all others at the reference vector, and record the
        resulting growth. Returns list of (name, growth, pct_of_ref)
        sorted by most damaging first. Use this to find your REAL
        decision variables (large drop = essential; ~no drop = drop it
        from the optimization, it isn't limiting).
        """
        if not self.growth_ref:
            self.measure_reference()
        base = self.medium.default_vector()
        rows = []
        for i, comp in enumerate(self.medium.components):
            x = list(base)
            x[i] = comp.lb
            g = self.evaluate(x).growth
            pct = 100.0 * g / self.growth_ref if self.growth_ref else 0.0
            rows.append((comp.name, g, pct))
        rows.sort(key=lambda r: r[2])
        return rows

    # ------------------------------------------------------------------
    # Local sensitivity analysis around a given solution (finite differences)
    # ------------------------------------------------------------------
    def sensitivity_analysis(self, x_star, rel_step: float = 0.05) -> List[Tuple[str, float, float]]:
        """
        Around a given optimized point x_star, perturb each component by
        +rel_step (relative to its ub) and measure d(growth)/d(component)
        and d(cost)/d(component) via forward finite differences.
        Returns (name, d_growth, d_cost) sorted by |d_growth| descending.
        This tells you which ingredients the optimum is most sensitive to
        -- exactly the "sensitivity analysis for cost of ingredients" step.
        """
        base_result = self.evaluate(x_star)
        rows = []
        for i, comp in enumerate(self.medium.components):
            step = rel_step * (comp.ub - comp.lb if comp.ub > comp.lb else 1.0)
            x_pert = list(x_star)
            x_pert[i] = min(comp.ub, x_pert[i] + step)
            pert_result = self.evaluate(x_pert)
            d_growth = (pert_result.growth - base_result.growth) / step if step else 0.0
            d_cost = (pert_result.cost - base_result.cost) / step if step else 0.0
            rows.append((comp.name, d_growth, d_cost))
        rows.sort(key=lambda r: -abs(r[1]))
        return rows

    # ------------------------------------------------------------------
    # Flux interpretation: top fluxes at a given medium
    # ------------------------------------------------------------------
    def top_flux_report(self, x, n: int = 15, exclude_exchanges: bool = False):
        """
        Solve FBA at `x` and return the n reactions with the largest
        |flux|, as (reaction_id, name, subsystem, flux). Also flags any
        tryptophan/kynurenine-related reactions found, for the thesis tie-in.
        """
        with self.model as m:
            self.medium.apply_to_model(m, x)
            solution = m.optimize()
            if solution.status != "optimal":
                return [], []
            flux_series = solution.fluxes
            rows = []
            for rxn in m.reactions:
                if exclude_exchanges and rxn in m.exchanges:
                    continue
                f = flux_series.get(rxn.id, 0.0)
                if abs(f) < 1e-9:
                    continue
                subsystem = getattr(rxn, "subsystem", "") or ""
                rows.append((rxn.id, rxn.name, subsystem, f))
            rows.sort(key=lambda r: -abs(r[3]))

            trp_related = [r for r in rows if "trp" in r[0].lower() or "kyn" in r[0].lower()
                           or "trp" in (r[1] or "").lower() or "kynurenine" in (r[1] or "").lower()]
        return rows[:n], trp_related

    # ------------------------------------------------------------------
    # Diagnostic: find missing essential nutrients
    # ------------------------------------------------------------------
    def find_unlocking_nutrients(self, x=None, candidate_bound: float = 1000.0, verbose: bool = True):
        """
        USE THIS WHEN growth is 0 even at your medium's richest point.

        Your CultureMedium only opens a curated SUBSET of the exchange
        reactions the model actually supports (whatever your components +
        fixed_open cover). If growth is stuck at 0 across the whole
        search, it usually means that subset is missing something the
        model's biomass equation genuinely requires -- a nutrient you
        never included at all, not one you bounded too tightly.

        This tests every exchange reaction the model has that your
        medium does NOT currently cover, one at a time, opened at a
        generous bound on top of your current medium, and reports which
        ones move growth above your current baseline. That turns
        "something is missing, not sure what" into a short, concrete
        list.

        Returns a list of (exchange_id, resulting_growth) for every
        reaction that unlocked growth, sorted best-first.
        """
        if x is None:
            x = self.medium.default_vector()

        covered = set()
        for comp in self.medium.components:
            covered.update(comp.exchange_ids())
        for fb in self.medium.fixed_open:
            covered.add(fb.exchange_id)
        covered.update(self.medium.closed)

        all_exchange_ids = [rxn.id for rxn in self.model.exchanges]
        uncovered = [rid for rid in all_exchange_ids if rid not in covered]

        baseline = self.evaluate(x).growth
        if verbose:
            print(f"Baseline growth at your medium's richest point: {baseline:.4f}")
            print(f"Your medium covers {len(covered)} of {len(all_exchange_ids)} exchange reactions "
                  f"the model has -- testing the other {len(uncovered)} one at a time...")

        hits = []
        for rid in uncovered:
            with self.model as m:
                self.medium.apply_to_model(m, x)
                if rid in m.reactions:
                    m.reactions.get_by_id(rid).lower_bound = -candidate_bound
                sol = m.optimize()
                g = sol.objective_value if sol.status == "optimal" else 0.0
            if g and g > baseline + 1e-9:
                hits.append((rid, g))
                if verbose:
                    rxn_name = self.model.reactions.get_by_id(rid).name if rid in self.model.reactions else ""
                    print(f"   {rid:20s} {rxn_name:35s} -> growth = {g:.4f}   <-- UNLOCKS GROWTH")

        hits.sort(key=lambda h: -h[1])
        if verbose and not hits:
            print("   No single uncovered reaction unlocked growth on its own -- the model may need "
                  "TWO OR MORE missing nutrients simultaneously. Try opening several of the biggest "
                  "candidate groups together (e.g. all remaining sugars, or all remaining cofactors) "
                  "and re-run this scan on what's left.")
        return hits

    # ------------------------------------------------------------------
    # Diagnostic: top-down group-removal scan (for when single-reaction
    # scanning finds nothing -- i.e. multiple nutrients are jointly required)
    # ------------------------------------------------------------------
    def group_removal_scan(self, groups: "Dict[str, List[str]]", candidate_bound: float = 1000.0,
                            verbose: bool = True):
        """
        USE THIS WHEN find_unlocking_nutrients() finds no single fix.

        Starts from ALL exchange reactions the model has, opened wide
        (candidate_bound each) -- i.e. the model's full 155-reaction
        default medium, which you already confirmed reaches nonzero
        growth. Then removes one named GROUP of reactions at a time
        (closing every reaction in that group back to 0) and reports how
        much growth drops. A group whose removal collapses growth to ~0
        is a group containing something essential; narrow further within
        that group afterward (e.g. remove its members one at a time, or
        split it into smaller groups and repeat).

        `groups` : dict of {group_name: [exchange_id, ...]}. You define
        the groups (e.g. "sugars", "vitamins", "trace_metals") -- this
        makes the search fast (few groups) rather than slow (155
        individual reactions), and the grouping itself is diagnostic:
        whichever CATEGORY breaks growth tells you what's missing without
        needing to test every reaction one by one.

        Returns list of (group_name, growth_with_group_removed, pct_of_full)
        sorted most-damaging first.
        """
        all_exchange_ids = [rxn.id for rxn in self.model.exchanges]

        with self.model as m:
            for rid in all_exchange_ids:
                m.reactions.get_by_id(rid).lower_bound = -candidate_bound
            full_solution = m.optimize()
            full_growth = full_solution.objective_value if full_solution.status == "optimal" else 0.0
        if verbose:
            print(f"Full 155-reaction baseline (everything open at {candidate_bound}): growth = {full_growth:.4f}")
        if full_growth <= 1e-9:
            if verbose:
                print("   WARNING: even fully open, growth is ~0 on this call -- that contradicts your "
                      "earlier 67.4 h^-1 result. Double check you're passing the SAME model object/file "
                      "here as when you got that number.")
            return []

        grouped_ids = set()
        for ids in groups.values():
            grouped_ids.update(ids)
        ungrouped = [rid for rid in all_exchange_ids if rid not in grouped_ids]
        if ungrouped and verbose:
            print(f"   NOTE: {len(ungrouped)} exchange reactions are not in any group you defined "
                  f"and will stay open throughout (not tested): {ungrouped[:10]}"
                  f"{' ...' if len(ungrouped) > 10 else ''}")

        rows = []
        for gname, ids in groups.items():
            with self.model as m:
                for rid in all_exchange_ids:
                    m.reactions.get_by_id(rid).lower_bound = -candidate_bound
                for rid in ids:
                    if rid in m.reactions:
                        m.reactions.get_by_id(rid).lower_bound = 0
                sol = m.optimize()
                g = sol.objective_value if sol.status == "optimal" else 0.0
            pct = 100.0 * g / full_growth
            rows.append((gname, g, pct))
            if verbose:
                flag = "  <-- REMOVING THIS GROUP KILLS GROWTH (contains something essential)" if pct < 5 else ""
                print(f"   remove '{gname}':  growth = {g:.4f}  ({pct:5.1f}% of full){flag}")

        rows.sort(key=lambda r: r[2])
        return rows

    # ------------------------------------------------------------------
    # Diagnostic: narrow a flagged group down to the exact reaction(s)
    # ------------------------------------------------------------------
    def narrow_group(self, reaction_ids: "List[str]", candidate_bound: float = 1000.0, verbose: bool = True):
        """
        USE THIS after group_removal_scan() flags a group as essential.

        Same idea as group_removal_scan, but tests each reaction in the
        flagged group INDIVIDUALLY (starting from the full 155-open
        baseline) rather than as a whole group -- pinpoints exactly which
        member(s) of the group are load-bearing versus just along for
        the ride.

        Returns list of (exchange_id, growth_with_this_one_removed, pct_of_full)
        sorted most-damaging first.
        """
        all_exchange_ids = [rxn.id for rxn in self.model.exchanges]
        with self.model as m:
            for rid in all_exchange_ids:
                m.reactions.get_by_id(rid).lower_bound = -candidate_bound
            full_solution = m.optimize()
            full_growth = full_solution.objective_value if full_solution.status == "optimal" else 0.0
        if full_growth <= 1e-9:
            if verbose:
                print("Full-open baseline is 0 -- can't narrow from here, check model consistency first.")
            return []

        rows = []
        for rid in reaction_ids:
            with self.model as m:
                for full_id in all_exchange_ids:
                    m.reactions.get_by_id(full_id).lower_bound = -candidate_bound
                if rid in m.reactions:
                    m.reactions.get_by_id(rid).lower_bound = 0
                sol = m.optimize()
                g = sol.objective_value if sol.status == "optimal" else 0.0
            pct = 100.0 * g / full_growth
            rows.append((rid, g, pct))
            if verbose:
                rxn_name = self.model.reactions.get_by_id(rid).name if rid in self.model.reactions else "?"
                flag = "  <-- ESSENTIAL (on its own)" if pct < 5 else ""
                print(f"   remove {rid:18s} ({rxn_name:30s}) -> growth = {g:.4f}  ({pct:5.1f}% of full){flag}")

        rows.sort(key=lambda r: r[2])
        return rows
