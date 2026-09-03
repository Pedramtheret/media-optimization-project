"""
find_load_bearing_ingredient.py

Closes the remaining gap from this debugging thread: we've confirmed
misc_other is a dead end (find_missing_from_mrs.py), and we know Yeast
extract is flagged ESSENTIAL in the whole-component essentiality scan --
but "yeast extract" is one CompositeComponent bundling ~40 individual
exchange reactions (19 amino acids + ~15 vitamin-group entries + 6 trace
metals + 5 nucleotides). We have never actually identified WHICH of those
~40 is the specific thing making yeast extract essential. That's a real
gap in the paper trail: right now you can say "growth is nonzero" but not
"growth is nonzero because of X."

Same corrected logic as find_missing_from_mrs.py: test against the ACTUAL
current MRS medium (not the unconstrained default), by zeroing out ONE
exchange reaction INSIDE yeast extract's profile at a time -- not the whole
composite decision variable -- while everything else in the medium stays
exactly as it is. This directly separates "yeast extract matters because of
its amino acids" from "...because of one specific vitamin" from "...because
of one specific trace metal."

    python find_load_bearing_ingredient.py
"""

import cobra
from media_components_updated import build_bifidobacterium_mrs_medium, AMINO_ACIDS, NUCLEOTIDES
from gem_evaluator import GEMEvaluator

from pathlib import Path

MODEL_PATH = Path("D:/workspace/Hakim project/AGORA2_SBML/Bifidobacterium_longum_NCC2705.xml")


def main():
    model = cobra.io.read_sbml_model(MODEL_PATH)
    medium = build_bifidobacterium_mrs_medium()
    evaluator = GEMEvaluator(model, medium)

    yeast = next(c for c in medium.components if c.name == "Yeast extract level")
    baseline_x = medium.default_vector()
    baseline_growth = evaluator.evaluate(baseline_x).growth
    print(f"Baseline growth on current MRS medium (yeast extract at its full profile): "
          f"{baseline_growth:.4f}\n")

    if baseline_growth <= 1e-9:
        print("Baseline is 0 -- fix that first before this test means anything.")
        return

    # group yeast extract's ~40 exchange ids by category for readable output
    categories = {
        "amino_acids": [f"EX_{aa}(e)" for aa in AMINO_ACIDS],
        "nucleotides": [f"EX_{n}(e)" for n in NUCLEOTIDES],
    }
    categorized = set()
    for ids in categories.values():
        categorized.update(ids)
    categories["vitamins_and_trace_metals"] = [rid for rid in yeast.profile if rid not in categorized]

    print(f"Yeast extract's profile has {len(yeast.profile)} individual exchange reactions.")
    print("Testing each one individually: zero it out, keep everything else (including the")
    print("rest of yeast extract's own profile) exactly at the current medium's levels.\n")

    results = []
    for cat_name, ids in categories.items():
        print(f"--- {cat_name} ({len(ids)} reactions) ---")
        for rid in ids:
            if rid not in yeast.profile:
                continue
            # FIX: the naive "zero the whole aggregated reaction bound" approach
            # can't distinguish "yeast extract's share of cysteine matters" from
            # "cysteine matters at all, from ANY source" -- because once
            # CultureMedium sums peptone+meat+yeast's contributions into one
            # shared exchange-reaction bound, that bound has no memory of which
            # component supplied what. Zeroing the whole reaction removes ALL
            # THREE sources at once, not just yeast extract's.
            #
            # Correct test: rebuild the aggregate WITHOUT yeast extract's
            # contribution to this one reaction specifically, by subtracting
            # only yeast's own weighted amount from the full total, and set
            # the bound to that reduced (not zeroed) value -- this isolates
            # "what if yeast extract, specifically, didn't supply this."
            full_totals = medium.aggregate_bounds(baseline_x)
            yeast_share = yeast.contribution(baseline_x[medium.components.index(yeast)]).get(rid, 0.0)
            reduced_total = full_totals.get(rid, 0.0) - yeast_share

            with model as m:
                medium.apply_to_model(m, baseline_x)
                if rid in m.reactions:
                    # set to the total WITHOUT yeast's contribution, not to 0 --
                    # this correctly leaves peptone/meat's shares intact
                    m.reactions.get_by_id(rid).lower_bound = -abs(reduced_total) if reduced_total > 0 else 0
                sol = m.optimize()
                g = sol.objective_value if sol.status == "optimal" else 0.0
            pct = 100 * g / baseline_growth
            flag = "  <-- LOAD-BEARING (yeast extract's SHARE specifically matters)" if pct < 95 else ""
            if pct < 99.9:  # only print ones that show ANY effect, to keep output readable
                print(f"   {rid:18s} -> growth = {g:.4f}  ({pct:5.1f}% of baseline, "
                      f"yeast contributed {yeast_share:.3f} of {full_totals.get(rid, 0.0):.3f} total){flag}")
            results.append((cat_name, rid, g, pct))

    print("\n" + "=" * 70)
    load_bearing = [r for r in results if r[3] < 95]
    if load_bearing:
        print("Reaction(s) inside yeast extract's profile that are individually load-bearing:")
        for cat_name, rid, g, pct in sorted(load_bearing, key=lambda r: r[3]):
            rxn = model.reactions.get_by_id(rid) if rid in model.reactions else None
            name = rxn.name if rxn else "?"
            print(f"   [{cat_name}] {rid}  ({name})  -> {pct:.1f}% of baseline growth remains")
        print("\nThis tells you EXACTLY what to cite/verify for your report, instead of the")
        print("whole 'yeast extract' block being an opaque essential requirement.")
    else:
        print("No single ingredient inside yeast extract's profile is individually load-bearing")
        print("at this threshold -- essentiality is likely coming from the AMINO ACID pool as a")
        print("whole (removing one of 19 amino acids barely dents total nitrogen/carbon supply,")
        print("but the group scan already confirmed the composite variable overall is essential).")
        print("If so, that's a coherent, expected result: amino acids are yeast extract's bulk")
        print("contribution, so essentiality being diffuse across many of them rather than")
        print("concentrated in one is the correct, unsurprising shape for that finding.")


if __name__ == "__main__":
    main()