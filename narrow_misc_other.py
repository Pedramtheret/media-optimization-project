"""
narrow_misc_other.py

Answers the open question from this conversation: which SPECIFIC reaction
inside the 'misc_other' group was actually essential, and does it correspond
to something real MRS ingredients (peptone/meat extract/yeast extract) could
plausibly supply -- or does its presence in the model at all mean either (a)
the GEM reconstruction has a real gap, or (b) NCC2705 needs supplementation
beyond plain MRS to grow in reality.

Run this against your existing project files (needs gem_evaluator.py,
debug_zero_growth.py's _build_groups, and your model file all present in
the same directory).

    python narrow_misc_other.py
"""

import cobra
from media_components_4 import build_bifidobacterium_mrs_medium
from gem_evaluator import GEMEvaluator
from debug_zero_growth import _build_groups, FULL_DEFAULT_MEDIUM_IDS
from pathlib import Path

MODEL_PATH = Path("D:/workspace/Hakim project/AGORA2_SBML/Bifidobacterium_longum_NCC2705.xml")


def main():
    model = cobra.io.read_sbml_model(MODEL_PATH)
    medium = build_bifidobacterium_mrs_medium()
    evaluator = GEMEvaluator(model, medium)

    groups = _build_groups(medium)
    misc = groups.get("misc_other", [])
    print(f"misc_other currently contains {len(misc)} candidate reactions: {misc}\n")

    print("Narrowing to find which SPECIFIC reaction(s) are essential on their own")
    print("(starting from all 155 reactions open, testing each misc_other member alone):\n")
    results = evaluator.narrow_group(misc)

    essential = [r for r in results if r[2] < 5]
    print("\n" + "=" * 70)
    if essential:
        print("REACTION(S) THAT ARE INDIVIDUALLY ESSENTIAL:")
        for rid, g, pct in essential:
            rxn = model.reactions.get_by_id(rid) if rid in model.reactions else None
            name = rxn.name if rxn else "?"
            print(f"\n   {rid}  ({name})")
            print(f"   growth when removed: {g:.4f} ({pct:.1f}% of full)")
            if rxn is not None:
                print(f"   subsystem: {getattr(rxn, 'subsystem', '(none listed)')}")
                print(f"   reaction formula: {rxn.reaction}")
        print("\nNext step (manual, biological judgment call -- not automatable):")
        print("   Look up whether the metabolite(s) in the reaction(s) above are")
        print("   plausibly present in peptone, meat extract, or yeast extract")
        print("   specifically. If yes -> add it to the relevant extract's profile")
        print("   with a citation. If no -> this may indicate either a genuine GEM")
        print("   reconstruction gap, or that NCC2705 needs supplementation beyond")
        print("   plain MRS to grow -- both are legitimate, reportable findings,")
        print("   not something to silently patch around.")
    else:
        print("No single misc_other reaction is individually essential -- growth")
        print("in your current v3.1 medium is likely coming from a DIFFERENT,")
        print("already-covered source (e.g. the expanded yeast extract profile),")
        print("not from anything in misc_other. This would mean misc_other's")
        print("original essentiality (found many turns ago) was answered by a")
        print("change elsewhere, not resolved directly -- worth confirming which.")


if __name__ == "__main__":
    main()
