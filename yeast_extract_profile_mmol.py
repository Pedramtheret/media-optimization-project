"""
yeast_extract_profile_mmol.py

Converts your yeast_extract_composition.xlsx-derived mass-fraction dicts
(AMINO_ACIDS, VITAMIN_GROUPS_G_PER_100G, YEAST_TRACE_METALS_G_PER_100G)
into a single profile dict in the correct units for CompositeComponent:
    mmol of nutrient PER GRAM OF YEAST EXTRACT

This is deliberately NOT mmol/L, because "yeast extract" is an undefined
mixture with no molecular weight -- it can't be converted to moles itself.
Only the individual metabolites inside it (each with a real MW) can be.

How this plugs into your CultureMedium:
    - The decision variable `x` for the "Yeast extract level" component
      stays in GRAMS OF EXTRACT PER LITRE (never converted to moles).
    - This module's output dict is the `profile` weight: mmol nutrient
      per gram of extract.
    - CompositeComponent.contribution(x) = weight * x therefore gives
      mmol/L directly, matching your (separately fixed) glucose/phosphate/
      MgSO4/etc. components, which should also be expressed in mmol/L.
    - Set that component's `ub` to a realistic DOSE IN GRAMS, e.g.
      somewhere between classic MRS's ~4-5 g/L and Cheng et al.'s
      optimized 19.524 g/L -- not an arbitrary tuned number.

One bug fixed from your pasted code: YEAST_TRACE_METALS_G_PER_100G had
fe3 and mn2 dividing by 10,000 instead of 100,000 (inconsistent with
zn2/cobalt2/cu2, which correctly divide by 100,000 to convert mg/100g dry
weight -> g/g mass fraction). Fixed below -- see FIXED_YEAST_TRACE_METALS.

Molecular weights: PubChem (Kim S, Chen J, Cheng T, et al. (2019)
"PubChem 2019 update." Nucleic Acids Res 47:D1102-D1109).
"""

from typing import Dict


# ----------------------------------------------------------------------
# Your data, as pasted -- UNCHANGED (already correct mass-fraction basis)
# ----------------------------------------------------------------------
AMINO_ACIDS: Dict[str, float] = {
    "ala_L": 4.8 / 100,
    "arg_L": 3.2 / 100,
    "asn_L": 4 / 100,
    "gln_L": 5 / 100,
    "asp_L": 6.5 / 100,
    "cys_L": 0.6 / 100,
    "glu_L": 11.5 / 100,
    "gly": 3 / 100,
    "his_L": 1.4 / 100,
    "ile_L": 3.1 / 100,
    "leu_L": 4.2 / 100,
    "lys_L": 4.6 / 100,
    "met_L": 0.9 / 100,
    "phe_L": 2.6 / 100,
    "pro_L": 2.3 / 100,
    "ser_L": 2.4 / 100,
    "thr_L": 2.6 / 100,
    "trp_L": 0.9 / 100,
    "tyr_L": 1.1 / 100,
    "val_L": 4.1 / 100,
}

VITAMIN_GROUPS_G_PER_100G: Dict[str, float] = {
    "thm": 21 / 1000000,
    "ribflv": 125 / 1000000,
    "pnto_R": 105 / 1000000,
    "pydxn": 24 / 1000000,
    "pydx": 70 / 1000000,
    "pydam": 70 / 1000000,
    "nac": 600 / 1000000,
    "cbl1": 3 / 1000000,
    "fol": 6 / 1000000,
    "btn": 4 / 1000000,
    "thf": 0.002 / 1000000,
    "5mthf": 0.002 / 1000000,
    "adocbl": 0.01 / 1000000,
    "4abz": 0.002 / 1000000,
    "dpcoa": 0.002 / 1000000,
}

# ---- FIXED: fe3 and mn2 now divide by 100,000 like the other four ----
FIXED_YEAST_TRACE_METALS_G_PER_100G: Dict[str, float] = {
    "zn2": 13.6 / 100000,
    "fe2": 1.76 / 100000,
    "fe3": 1.76 / 100000,     # was /10000 -- 10x too concentrated, now consistent
    "mn2": 8 / 100000,        # was /10000 -- 10x too concentrated, now consistent
    "cobalt2": 0.05 / 100000,
    "cu2": 0.3 / 100000,
}

NUCLEOBASE_MW_G_PER_MOL = {
    "ade":  135.13,   # adenine,       C5H5N5
    "gua":  151.13,   # guanine,       C5H5N5O
    "hxan": 136.11,   # hypoxanthine,  C5H4N4O
    "xan":  152.11,   # xanthine,      C5H4N4O2
    "orot": 156.10,   # orotate (orotic acid), C5H4N2O4 -- a pyrimidine
                       # precursor, not a base itself, but grouped with
                       # these in your code and in FULL_DEFAULT_MEDIUM_IDS
}

# ----------------------------------------------------------------------
# Molecular weights (g/mol) for every metabolite above. PubChem-sourced.
# ----------------------------------------------------------------------
MW_G_PER_MOL: Dict[str, float] = {
    "ala_L": 89.09, "arg_L": 174.20, "asn_L": 132.12, "gln_L": 146.14,
    "asp_L": 133.10, "cys_L": 121.16, "glu_L": 147.13, "gly": 75.07,
    "his_L": 155.16, "ile_L": 131.17, "leu_L": 131.17, "lys_L": 146.19,
    "met_L": 149.21, "phe_L": 165.19, "pro_L": 115.13, "ser_L": 105.09,
    "thr_L": 119.12, "trp_L": 204.23, "tyr_L": 181.19, "val_L": 117.15,

    "thm": 337.27, "ribflv": 376.36, "pnto_R": 219.23, "pydxn": 169.18,
    "pydx": 167.16, "pydam": 168.19, "nac": 123.11, "cbl1": 1355.37,
    "fol": 441.40, "btn": 244.31, "thf": 445.43, "5mthf": 459.46,
    "adocbl": 1579.61, "4abz": 137.14, "dpcoa": 767.55,

    "zn2": 65.38, "fe2": 55.85, "fe3": 55.85, "mn2": 54.94,
    "cobalt2": 58.93, "cu2": 63.55,
    
    "ade":  135.13, "gua":  151.13, "hxan": 136.11, "xan":  152.11, "orot": 156.10,  
}

MEAT_VITAMNINE_PROFILE = {key: value * 0.3 for key, value in VITAMIN_GROUPS_G_PER_100G.items()}
MEAT_NUCLEOTIDE_PROFILE = {key: value * 2 for key, value in NUCLEOBASE_MW_G_PER_MOL.items()}


def _to_mmol_per_g_extract(mass_fraction_dicts) -> Dict[str, float]:
    """Merge {bare_id: mass_fraction} dicts -> {EX_id(e): mmol/g extract}."""
    profile: Dict[str, float] = {}
    for d in mass_fraction_dicts:
        for met_id, mass_fraction in d.items():
            mw = MW_G_PER_MOL[met_id]
            mmol_per_g = (mass_fraction / mw) * 1000.0
            profile[f"EX_{met_id}(e)"] = mmol_per_g
    return profile


def _to_single_mmol_per_g_extract(mass_fraction_dict):
    """Convert {bare_id: mass_fraction} -> {EX_id(e): mmol/g extract}."""
    profile = {}

    for met_id, mass_fraction in mass_fraction_dict.items():
        mw = MW_G_PER_MOL[met_id]
        mmol_per_g = (mass_fraction / mw) * 1000.0
        profile[f"EX_{met_id}(e)"] = mmol_per_g

    return profile




# ----------------------------------------------------------------------
# THE RESULT: drop this straight in as the `profile=` argument of your
# "Yeast extract level" CompositeComponent. Units: mmol nutrient / g
# yeast extract. Remember: `x` for this component = grams extract / L.
# ----------------------------------------------------------------------

AMINO_ACIDS_MMOL: Dict[str, float] = _to_single_mmol_per_g_extract(AMINO_ACIDS)
NUCLEOTIDES_MMOL: Dict[str, float] = _to_single_mmol_per_g_extract(NUCLEOBASE_MW_G_PER_MOL)
YEAST_EXTRACT_PROFILE_MMOL_PER_G: Dict[str, float] = _to_mmol_per_g_extract(
    [AMINO_ACIDS, VITAMIN_GROUPS_G_PER_100G, FIXED_YEAST_TRACE_METALS_G_PER_100G, NUCLEOBASE_MW_G_PER_MOL]
)

MEAT_EXTRACT_PROFILE_MMOL_PER_G: Dict[str, float] = _to_mmol_per_g_extract(
    [AMINO_ACIDS, MEAT_VITAMNINE_PROFILE, MEAT_NUCLEOTIDE_PROFILE]
)

PEPTONE_PROFILE_MMOL_PER_G: Dict[str, float] = _to_mmol_per_g_extract(
    [AMINO_ACIDS]
)



if __name__ == "__main__":
    print(f"{len(YEAST_EXTRACT_PROFILE_MMOL_PER_G)} exchange reactions in the profile.\n")
    print(f"{'exchange id':16s} {'mmol/g extract':>16s}   {'at ub=19.524 g/L (Cheng et al.)':>32s}")
    for ex_id, w in sorted(YEAST_EXTRACT_PROFILE_MMOL_PER_G.items(), key=lambda kv: -kv[1]):
        supplied_at_ub = w * 19.524
        print(f"{ex_id:16s} {w:16.6e}   {supplied_at_ub:32.6e} mmol/L")

    print("\nSanity range check: amino acids should land ~0.3-1.3 mmol/g extract;")
    print("vitamins/trace metals should land many orders of magnitude lower --")
    print("that gap is real biology, not a bug, as long as it comes from a")
    print("consistent mmol/g basis (which it now does).")
