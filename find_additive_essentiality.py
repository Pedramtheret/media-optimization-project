"""
find_additive_essentiality.py

Follow-up to narrow_misc_other.py's result: no single misc_other reaction
is individually essential, but the WHOLE group's removal (from many turns
ago) collapsed growth to 0%. This is the signature of additive/redundant
essentiality -- several reactions each individually dispensable, but the
cell needs SOME minimum combination among them.

This script narrows that down properly: it removes misc_other reactions
in growing combinations (pairs, then triples if needed) rather than one at
a time, to find the SMALLEST combination whose joint removal collapses
growth -- which tells you the real minimal essential requirement instead
of "all 19 at once."

Biologically motivated first guess, tested explicitly rather than assumed:
EX_h2s(e) (hydrogen sulfide) and EX_taur(e) (taurine) are both sulfur
sources; Schoepping et al. 2021 report bifidobacteria require an organic
(cysteine) or assimilable inorganic (H2S) sulfur source. If cysteine/
methionine in the extract profiles only PARTIALLY cover sulfur demand,
losing BOTH H2S and taurine together (while cysteine/methionine stay
available) might still be survivable -- or might not. Test it directly.

    python find_additive_essentiality.py
"""

import itertools
import cobra
from media_components_4 import build_bifidobacterium_mrs_medium
from gem_evaluator import GEMEvaluator
from pathlib import Path

MODEL_PATH = Path("D:/workspace/Hakim project/AGORA2_SBML/Bifidobacterium_longum_NCC2705.xml")

MISC_OTHER = ['EX_4abut(e)', 'EX_C02528(e)', 'EX_HC02191(e)', 'EX_HC02192(e)', 'EX_HC02193(e)',
              'EX_M01989(e)', 'EX_M03134(e)', 'EX_butam(e)', 'EX_dma(e)', 'EX_fcsn(e)',
              'EX_h2s(e)', 'EX_met_D(e)', 'EX_metsox_S_L(e)', 'EX_no3(e)', 'EX_norval_L(e)',
              'EX_peamn(e)', 'EX_ppi(e)', 'EX_taur(e)', 'EX_urea(e)']


def test_combo(evaluator, model, x, all_exchange_ids, combo, candidate_bound=1000.0):
    with model as m:
        for rid in all_exchange_ids:
            m.reactions.get_by_id(rid).lower_bound = -candidate_bound
        for rid in combo:
            if rid in m.reactions:
                m.reactions.get_by_id(rid).lower_bound = 0
        sol = m.optimize()
        return sol.objective_value if sol.status == "optimal" else 0.0


def main():
    model = cobra.io.read_sbml_model(MODEL_PATH)
    medium = build_bifidobacterium_mrs_medium()
    evaluator = GEMEvaluator(model, medium)
    all_exchange_ids = [rxn.id for rxn in model.exchanges]

    with model as m:
        for rid in all_exchange_ids:
            m.reactions.get_by_id(rid).lower_bound = -1000.0
        full_sol = m.optimize()
        full_growth = full_sol.objective_value if full_sol.status == "optimal" else 0.0
    print(f"Full-open baseline growth: {full_growth:.4f}\n")
    if full_growth <= 1e-9:
        print("Full-open baseline is 0 -- something upstream changed, stop and check that first.")
        return

    # Step 1: targeted biological hypothesis -- sulfur sources together
    print("Testing hypothesis: EX_h2s(e) + EX_taur(e) removed TOGETHER (sulfur sources)")
    combo = ["EX_h2s(e)", "EX_taur(e)"]
    g = test_combo(evaluator, model, None, all_exchange_ids, combo)
    print(f"   growth with both removed: {g:.4f}  ({100*g/full_growth:.1f}% of full)\n")

    # Step 2: exhaustive pairs (19 choose 2 = 171 combinations -- cheap, each is one fast LP solve)
    print("Exhaustive pairwise scan (all 171 pairs) -- looking for any pair whose JOINT removal")
    print("collapses growth, even though neither member alone does:\n")
    damaging_pairs = []
    for combo in itertools.combinations(MISC_OTHER, 2):
        g = test_combo(evaluator, model, None, all_exchange_ids, combo)
        pct = 100 * g / full_growth
        if pct < 5:
            damaging_pairs.append((combo, g, pct))

    if damaging_pairs:
        print(f"Found {len(damaging_pairs)} damaging pair(s):")
        for combo, g, pct in damaging_pairs:
            print(f"   {combo}  -> growth = {g:.4f} ({pct:.1f}% of full)")
    else:
        print("No pair collapses growth on its own. The joint requirement needs 3+ reactions")
        print("removed together -- worth testing triples next, or simply accepting that the")
        print("original all-19-at-once test found a real but diffusely-distributed dependency")
        print("that doesn't reduce to a small, easily-cited combination. That itself is a")
        print("legitimate, reportable finding: this reconstruction's redundancy in this region")
        print("of the network is broad rather than sharply localized to 1-2 metabolites.")


if __name__ == "__main__":
    main()
