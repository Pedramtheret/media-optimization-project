"""
test_integration.py

Tests media_components.py + gem_evaluator.py TOGETHER against a mock
"cobra model" (mimics reactions, exchanges, lower_bound, context-manager
revert, optimize()) -- exercises the real code path without needing the
actual AGORA2 file.

Test #0 below is a PERMANENT REGRESSION TEST for the composite-overwrite
bug: multiple CompositeComponents that share exchange ids (exactly like
Peptone/Meat extract/Yeast extract sharing amino acids) must have their
contributions SUM, not have later components silently discard earlier
ones. Keep this test forever -- it's cheap to run and it's exactly the
test that would have caught the original bug before it ever reached a
real model.

    python test_integration.py
"""

import numpy as np
from media_components import (
    SimpleComponent, MultiSimpleComponent, CompositeComponent, FixedBound, CultureMedium
)
from gem_evaluator import GEMEvaluator


# ---- minimal mock of the cobra interface we actually use ----
class MockReaction:
    def __init__(self, rxn_id, name="", subsystem=""):
        self.id = rxn_id
        self.name = name or rxn_id
        self.subsystem = subsystem
        self.lower_bound = 0.0
        self.upper_bound = 1000.0


class MockReactionList(list):
    """
    Mimics cobrapy's DictList closely enough for these tests: iterating
    yields Reaction OBJECTS (like real cobra), while still supporting
    get_by_id() and `"id_string" in reactions` lookups by id -- this is
    what a naive dict-based mock got wrong (iterating gave string keys
    instead of objects, which real cobrapy never does).
    """
    def __init__(self, rxn_dict):
        super().__init__(rxn_dict.values())
        self._by_id = rxn_dict

    def get_by_id(self, rxn_id):
        return self._by_id[rxn_id]

    def __contains__(self, item):
        rxn_id = item.id if hasattr(item, "id") else item
        return rxn_id in self._by_id

    def __getitem__(self, key):
        # allow both list-style m.reactions[0] and dict-style m.reactions["EX_x(e)"]
        if isinstance(key, str):
            return self._by_id[key]
        return super().__getitem__(key)


class MockSolution:
    def __init__(self, status, objective_value, fluxes):
        self.status = status
        self.objective_value = objective_value
        self.fluxes = fluxes


class MockModel:
    """
    Growth = Liebig's law across THREE pools, deliberately built so amino
    acids can ONLY be supplied by the sum of overlapping extract-like
    components -- this is what actually exercises the aggregation fix,
    not just glucose.
        growth = 0.8 * min(glucose/15, total_amino_acid_uptake/6, 1.3)
    """
    def __init__(self):
        ids = (["EX_glc_D(e)", "EX_pi(e)", "EX_nh4(e)", "EX_ac(e)", "EX_mg2(e)",
                "EX_so4(e)", "EX_mn2(e)", "EX_h(e)", "EX_h2o(e)", "EX_co2(e)",
                "EX_k(e)", "EX_na1(e)", "EX_cl(e)", "EX_o2(e)"]
               + [f"EX_{aa}(e)" for aa in ["ala_L", "asn_L", "asp_L", "cys_L", "gln_L", "glu_L",
                                            "gly", "his_L", "ile_L", "leu_L", "lys_L", "met_L",
                                            "phe_L", "pro_L", "ser_L", "thr_L", "trp_L", "tyr_L", "val_L"]]
               + [f"EX_{v}(e)" for v in ["btn", "fol", "nac", "pnto_R", "pydam", "pydx", "pydxn",
                                          "ribflv", "thm", "thf", "5mthf", "cbl1", "adocbl", "4abz", "dpcoa"]]
               + [f"EX_{m}(e)" for m in ["fe2", "fe3", "zn2", "cobalt2", "cu2"]]
               + [f"EX_{n}(e)" for n in ["ade", "gua", "hxan", "xan", "orot"]]
               + ["BIOMASS_rxn"])
        self.reactions = MockReactionList({i: MockReaction(i) for i in ids})
        self.exchanges = [rxn for rxn in self.reactions if rxn.id.startswith("EX_")]
        self._snapshot = None

    def __enter__(self):
        self._snapshot = {rxn.id: rxn.lower_bound for rxn in self.reactions}
        return self

    def __exit__(self, *args):
        for rxn_id, lb in self._snapshot.items():
            self.reactions.get_by_id(rxn_id).lower_bound = lb

    def optimize(self):
        glc_uptake = -self.reactions.get_by_id("EX_glc_D(e)").lower_bound
        aa_rxns = [rxn for rxn in self.reactions if rxn.id.startswith("EX_") and rxn.id.endswith("_L(e)")]
        aa_pool = sum(-rxn.lower_bound for rxn in aa_rxns if rxn.lower_bound < 0)

        growth = 0.8 * min(glc_uptake / 15.0, max(aa_pool, 1e-9) / 6.0, 1.3)
        growth = max(growth, 0.0)
        fluxes = {"BIOMASS_rxn": growth, "EX_glc_D(e)": -glc_uptake}
        status = "optimal" if growth > 0 or glc_uptake >= 0 else "infeasible"
        return MockSolution(status="optimal", objective_value=growth, fluxes=fluxes)


def build_test_medium():
    components = [
        SimpleComponent("Glucose", "EX_glc_D(e)", cost_per_unit=0.003, lb=0.0, ub=20.0, category="carbon"),
        MultiSimpleComponent("Magnesium sulfate", {"EX_mg2(e)": 1.0, "EX_so4(e)": 1.0},
                              cost_per_unit=0.02, lb=0.0, ub=1.0, category="ion_major"),
        SimpleComponent("Manganese", "EX_mn2(e)", cost_per_unit=0.08, lb=0.0, ub=0.2, category="ion_trace_expensive"),
        # THREE overlapping composites, exactly mirroring peptone/meat/yeast extract
        CompositeComponent("Peptone level", {f"EX_{aa}(e)": 1.0 for aa in ["ala_L", "leu_L", "lys_L", "val_L"]},
                            cost_per_unit=0.05, lb=0.0, ub=10.0, category="extract"),
        CompositeComponent("Meat extract level", {f"EX_{aa}(e)": 0.6 for aa in ["ala_L", "leu_L", "lys_L", "val_L"]},
                            cost_per_unit=0.08, lb=0.0, ub=5.0, category="extract"),
        CompositeComponent("Yeast extract level", {f"EX_{aa}(e)": 0.4 for aa in ["ala_L", "leu_L", "lys_L", "val_L"]},
                            cost_per_unit=0.09, lb=0.0, ub=5.0, category="extract"),
    ]
    fixed_open = [FixedBound("EX_h(e)", 1000.0), FixedBound("EX_h2o(e)", 1000.0)]
    return CultureMedium(components, fixed_open=fixed_open, closed=["EX_o2(e)"])


def main():
    model = MockModel()
    medium = build_test_medium()
    evaluator = GEMEvaluator(model, medium)

    print("=" * 60)
    print("0) REGRESSION TEST -- composite components must SUM, not overwrite")
    # Peptone=0, Meat extract=ub, Yeast extract=ub, everything else at ub
    x = medium.default_vector()
    peptone_idx = medium.names.index("Peptone level")
    x[peptone_idx] = 0.0
    totals = medium.aggregate_bounds(x)
    got = totals["EX_ala_L(e)"]
    expected = 0.6 * 5.0 + 0.4 * 5.0  # Meat + Yeast contributions, Peptone=0 contributes 0
    print(f"   EX_ala_L(e) aggregate with Peptone=0, Meat=ub, Yeast=ub: got={got}  expected={expected}")
    assert abs(got - expected) < 1e-9, (
        f"REGRESSION: composite components are overwriting instead of summing! "
        f"got {got}, expected {expected}. This is the original bug -- do not ship this."
    )
    growth_with_peptone_zeroed = evaluator.evaluate(x).growth
    growth_ref_check = evaluator.evaluate(medium.default_vector()).growth
    print(f"   growth with only Peptone zeroed: {growth_with_peptone_zeroed:.4f} "
          f"(reference at full: {growth_ref_check:.4f})")
    assert growth_with_peptone_zeroed > 0.5 * growth_ref_check, (
        "Zeroing ONE of three overlapping extract components should NOT collapse growth "
        "to near-zero when the other two are still fully open -- if it does, contributions "
        "are still being overwritten somewhere."
    )
    print("   PASSED -- overlapping components now correctly sum")

    print("=" * 60)
    print("1) measure_reference() at richest point")
    growth_ref = evaluator.measure_reference()
    print(f"   growth_ref = {growth_ref:.4f}")
    assert growth_ref > 0

    print("=" * 60)
    print("2) evaluate() at a starved point (glucose=0) should give ~0 growth")
    x2 = medium.default_vector()
    x2[medium.names.index("Glucose")] = 0.0
    starved = evaluator.evaluate(x2)
    print(f"   growth={starved.growth:.4f}  cost={starved.cost:.4f}")
    assert starved.growth < 1e-6

    print("=" * 60)
    print("3) context manager must revert bounds between calls (no state leakage)")
    _ = evaluator.evaluate(medium.default_vector())
    lb_after = model.reactions["EX_glc_D(e)"].lower_bound
    print(f"   EX_glc_D(e).lower_bound after evaluate() call = {lb_after}")
    assert lb_after == 0.0

    print("=" * 60)
    print("4) essentiality_scan() -- with the fix, removing ONE extract should be mild, not fatal")
    scan = evaluator.essentiality_scan()
    for name, g, pct in scan:
        print(f"   {name:20s} growth={g:.4f}  ({pct:5.1f}% of ref)")
    scan_dict = {name: pct for name, g, pct in scan}
    assert scan_dict["Glucose"] < 5, "glucose (sole carbon source) should still be essential"
    assert scan_dict["Peptone level"] > 40, (
        "with the fix, losing ONE of three overlapping amino-acid sources (Meat/Yeast still open) "
        "should NOT be near-fatal -- if it is, the overwrite bug is back"
    )

    print("=" * 60)
    print("5) sensitivity_analysis()")
    x_star = [15.0, 0.5, 0.1, 3.0, 2.0, 2.0]
    for name, dg, dc in evaluator.sensitivity_analysis(x_star):
        print(f"   {name:20s} d(growth)/dx={dg:+.5f}  d(cost)/dx={dc:+.5f}")

    print("=" * 60)
    print("6) top_flux_report()")
    top, trp = evaluator.top_flux_report(x_star, n=5)
    for rxn_id, name, subsystem, flux in top:
        print(f"   {rxn_id:15s} {flux:+8.3f}")
    assert any(r[0] == "BIOMASS_rxn" for r in top)

    print("=" * 60)
    print("ALL INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
