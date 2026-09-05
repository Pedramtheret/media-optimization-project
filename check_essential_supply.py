"""
check_essential_supply.py

Your debug_zero_growth.py run already narrowed the problem to exactly six
reactions -- every other reaction in the model is either non-essential or
redundant. Growth is 0.0000 on your medium, and these six are the ONLY
things that can cause a hard zero on their own:

    EX_ribflv(e)   riboflavin       -- trace, yeast-extract-only source
    EX_cobalt2(e)  Co2+             -- trace metal, yeast-extract-only source
    EX_cu2(e)      Cu2+             -- trace metal, yeast-extract-only source
    EX_fe2(e)      Fe2+             -- trace metal, yeast-extract-only source
    EX_zn2(e)      Zn2+             -- trace metal, yeast-extract-only source
    EX_lys_L(e)    L-lysine         -- BULK amino acid, multiple sources

This script does two things for each of the six:
  1. Prints the ACTUAL numeric bound your medium currently applies
     (medium.aggregate_bounds) -- catches "it's literally zero" bugs
     immediately, no guessing.
  2. Pulls the REAL required amount from the model's own biomass
     reaction -- its stoichiometric coefficient for that metabolite IS
     "mmol required per gDW of biomass produced," a ground-truth number
     that doesn't depend on whether you're thinking in rate-mode or
     yield-mode (see Marinos, Kaleta & Waschina 2020, PLOS ONE
     15(8):e0236890, "Step 6" -- checking what's actually limiting via
     the model itself rather than guessing).

Interpretation of the printed ratio (supply / required-per-gDW):
  - If supply is EXACTLY 0.0 for lysine specifically -> bug, not biology.
    Bulk amino acids shouldn't be able to hit a hard zero from a real
    extract dose. Go find the coverage gap (typo, dict not merged into
    the profile, wrong exchange id) before doing anything else.
  - If supply is nonzero but the ratio is astronomically small for one
    or more of the four trace metals / riboflavin -> that's your real
    bottleneck, quantified. Compare it against what real MRS is known to
    support (B. longum grows to high density on real MRS -- see Cheng et
    al. 2025) before concluding NCC2705 "can't" grow on it: an
    implausibly large biomass coefficient for a single cofactor is also
    a known GEM curation artifact worth checking against a second,
    well-curated model (e.g. an E. coli reconstruction) for the same
    metabolite, not just accepted at face value.

    python check_essential_supply.py
"""

import cobra
from media_converted import build_bifidobacterium_mrs_medium
from pathlib import Path

MODEL_PATH = Path("D:/workspace/Hakim project/AGORA2_SBML/Bifidobacterium_longum_NCC2705.xml")

CRITICAL = {
    "EX_ribflv(e)":  "ribflv",
    "EX_cobalt2(e)": "cobalt2",
    "EX_cu2(e)":     "cu2",
    "EX_fe2(e)":     "fe2",
    "EX_zn2(e)":     "zn2",
    "EX_lys_L(e)":   "lys_L",
}


def find_biomass_reaction(model):
    """The model's own objective reaction -- almost always the biomass pseudo-reaction."""
    for rxn in model.reactions:
        if rxn.objective_coefficient != 0:
            return rxn
    # fallback: search by id/name if objective_coefficient isn't set for some reason
    for rxn in model.reactions:
        if "biomass" in rxn.id.lower():
            return rxn
    raise RuntimeError("Could not find a biomass/objective reaction -- inspect model.objective manually.")


def find_coefficient(biomass_rxn, bare_id):
    """Find this metabolite's stoichiometric coefficient in the biomass equation,
    trying the common compartment-suffix conventions this model might use."""
    for suffix in ["_c", "[c]", "_c0", ""]:
        candidate_id = f"{bare_id}{suffix}"
        for met, coeff in biomass_rxn.metabolites.items():
            if met.id == candidate_id:
                return met.id, coeff
    return None, None


def main():
    print("Loading model ...")
    model = cobra.io.read_sbml_model(MODEL_PATH)
    medium = build_bifidobacterium_mrs_medium()
    biomass_rxn = find_biomass_reaction(model)
    print(f"Biomass reaction: {biomass_rxn.id}\n")

    x = medium.default_vector()
    totals = medium.aggregate_bounds(x)

    print(f"{'exchange id':16s} {'your supply':>16s} {'biomass needs/gDW':>20s} {'max feasible growth from this alone':>38s}")
    for ex_id, bare_id in CRITICAL.items():
        supply = totals.get(ex_id, 0.0)
        met_id, coeff = find_coefficient(biomass_rxn, bare_id)
        if coeff is None:
            print(f"{ex_id:16s} {supply:16.6e}   could not locate '{bare_id}' in biomass equation "
                  f"-- check compartment suffix manually")
            continue
        required_per_gDW = abs(coeff)
        max_feasible = supply / required_per_gDW if required_per_gDW else float("inf")
        flag = "  <-- ZERO SUPPLY -- BUG, not biology" if supply <= 0 else (
               "  <-- SEVERELY LIMITING" if max_feasible < 0.01 else "")
        print(f"{ex_id:16s} {supply:16.6e} {required_per_gDW:20.6e} {max_feasible:38.6e}{flag}")

    print("\nWhichever row has the smallest 'max feasible growth from this alone' value is your")
    print("binding bottleneck overall -- FBA can't do better than the worst of these six even if")
    print("every other nutrient were infinite.")


if __name__ == "__main__":
    main()
