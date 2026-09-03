"""
find_missing_from_mrs.py

CORRECTS a real methodological error from the previous two narrowing
scripts (narrow_misc_other.py, and the deleted find_additive_essentiality.py):
both tested whether misc_other reactions matter against the FULLY-OPEN
155-reaction default medium as the baseline. In that condition, ~130+ other
exchange reactions are simultaneously wide open, so almost any single
metabolite (or small combination) is trivially replaceable through some
alternate route -- finding "no pair matters" against that backdrop tells
you almost nothing about whether it matters for YOUR actual medium.

The right question is: starting from the current MRS-derived medium
(media_components.py's build_bifidobacterium_mrs_medium(), at its own
upper bounds -- NOT the unconstrained default) does adding any single
misc_other reaction, or any combination, change growth at all? This is
the same logic as the original group_removal_scan that actually found
the real fix many turns ago, just applied at the individual/pairwise
level this time instead of whole-group.

Two directions are tested, since both are informative:
  (a) ADDING each misc_other reaction on top of the current MRS medium --
      does anything in this list currently HELP growth beyond what MRS
      already provides? (answers: "is misc_other relevant to MRS at all")
  (b) For any that DO help, how much, and is the effect large enough to
      be worth citing/including as a real MRS-plausible addition?

    python find_missing_from_mrs.py
"""

import cobra
from media_components_updated import build_bifidobacterium_mrs_medium
from gem_evaluator import GEMEvaluator

from pathlib import Path

MODEL_PATH = Path("D:/workspace/Hakim project/AGORA2_SBML/Bifidobacterium_longum_NCC2705.xml")

MISC_OTHER = ['EX_4abut(e)', 'EX_C02528(e)', 'EX_HC02191(e)', 'EX_HC02192(e)', 'EX_HC02193(e)',
              'EX_M01989(e)', 'EX_M03134(e)', 'EX_butam(e)', 'EX_dma(e)', 'EX_fcsn(e)',
              'EX_h2s(e)', 'EX_met_D(e)', 'EX_metsox_S_L(e)', 'EX_no3(e)', 'EX_norval_L(e)',
              'EX_peamn(e)', 'EX_ppi(e)', 'EX_taur(e)', 'EX_urea(e)']

ADD_BOUND = 1.0  # modest, MRS-plausible uptake allowance to test with -- not 1000


def main():
    model = cobra.io.read_sbml_model(MODEL_PATH)
    medium = build_bifidobacterium_mrs_medium()
    evaluator = GEMEvaluator(model, medium)

    # ---- baseline: your ACTUAL current MRS medium, at its own bounds ----
    baseline_x = medium.default_vector()
    baseline_growth = evaluator.evaluate(baseline_x).growth
    print(f"Baseline growth on the current MRS-derived medium (NOT the unconstrained default): "
          f"{baseline_growth:.4f}\n")

    if baseline_growth <= 1e-9:
        print("Baseline is already 0 -- fix that first (this is the same zero-growth debugging")
        print("from earlier in the project); testing additions on top of a broken baseline")
        print("won't give a meaningful answer.")
        return

    print(f"Testing each misc_other reaction ADDED on top of this MRS baseline, one at a time,")
    print(f"at a modest bound ({ADD_BOUND} mmol/gDW/h, not 1000):\n")

    helpful = []
    for rid in MISC_OTHER:
        if rid not in model.reactions:
            continue
        with model as m:
            medium.apply_to_model(m, baseline_x)
            m.reactions.get_by_id(rid).lower_bound = -ADD_BOUND
            sol = m.optimize()
            g = sol.objective_value if sol.status == "optimal" else 0.0
        delta = g - baseline_growth
        rxn = model.reactions.get_by_id(rid)
        flag = "  <-- HELPS beyond current MRS medium" if delta > 1e-6 else ""
        print(f"   {rid:20s} ({rxn.name:30s}) -> growth = {g:.4f}  (delta = {delta:+.4f}){flag}")
        if delta > 1e-6:
            helpful.append((rid, rxn.name, delta))

    print("\n" + "=" * 70)
    if helpful:
        print("Reaction(s) that IMPROVE growth beyond the current MRS medium:")
        for rid, name, delta in helpful:
            print(f"   {rid}  ({name})  +{delta:.4f} h^-1")
        print("\nFor each of these, the real question is biological, not computational:")
        print("is this metabolite plausibly present in peptone, meat extract, or yeast")
        print("extract specifically? If yes with a citation -> add it to that extract's")
        print("profile. If no -> this is evidence NCC2705 may need supplementation beyond")
        print("plain MRS, which is a legitimate finding worth stating directly.")
    else:
        print("NONE of the misc_other reactions improve growth beyond what the current MRS")
        print("medium already provides. This is actually a clean, positive result: it means")
        print("misc_other was never truly relevant to your MRS-grounded medium in the first")
        print("place -- its earlier flagged essentiality was an artifact of testing against")
        print("the unconstrained default medium, not a real gap in your MRS model. Nothing to")
        print("add here; the earlier zero-growth fix (whatever specifically resolved it) is")
        print("the real, load-bearing change, and this list is a dead end -- correctly ruled")
        print("out, not unresolved.")


if __name__ == "__main__":
    main()
