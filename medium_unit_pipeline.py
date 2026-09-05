"""
medium_unit_pipeline.py

Fixes the root cause diagnosed in this debugging thread: every bound in
media_components_updated.py was a real g/L (or mg/kg-derived) concentration
reused DIRECTLY as an mmol/gDW/h exchange-flux bound, with no unit
conversion at all. That's an invalid move -- concentration and rate are
different physical quantities -- and it happens to look fine for
macro-nutrients (bulk g/L numbers coincidentally overlap with plausible
mmol/gDW/h ranges) while collapsing completely for genuine trace
nutrients (vitamins/metals in yeast extract), which is exactly the
symptom you saw.

METHODOLOGICAL BASIS (read before changing anything):
  Marinos G, Kaleta C, Waschina S (2020) "Defining the nutritional input
  for genome-scale metabolic models: A roadmap." PLOS ONE 15(8):e0236890.
  https://doi.org/10.1371/journal.pone.0236890

  This paper formalizes exactly the fork you're stuck on (see their
  "Step 5"): exchange bounds can be set up in one of two ways, and they
  are NOT interchangeable:

    (a) RATE MODE   -- bound in mmol/gDW/h, FBA objective = growth RATE
                       (h^-1). Requires bounds derived from uptake
                       KINETICS (Monod/Michaelis-Menten), not from how
                       much of the nutrient is in the flask. This is the
                       standard COBRA convention (e.g. glucose commonly
                       set to ~18.5 mmol/gDW/h as "a biologically
                       realistic uptake rate" -- see the COBRA Toolbox
                       FBA tutorial -- a KINETIC number, not a
                       concentration).

    (b) YIELD MODE  -- bound in mmol/L (literal total nutrient available
                       per litre of medium, via molecular weight only),
                       FBA objective = biomass YIELD (g dry weight per
                       litre of medium), NOT an h^-1 rate.

  Your code has been putting yield-mode numbers into a rate-mode slot and
  reading the result out as if it were an h^-1 rate. That mismatch, not
  your amino-acid/vitamin coverage, is almost certainly why growth
  collapsed once you started using real (non-placeholder) concentrations.

  This module implements YIELD MODE throughout, because it only requires
  molecular-weight conversion (which you can do exactly, for every
  nutrient), not uptake kinetics for B. longum (which nobody has measured
  and which you cannot look up). Once you switch, you must also relabel
  your reported "growth_ref" as a predicted biomass YIELD (g DCW/L), not
  a growth rate in h^-1 -- that relabeling is not cosmetic, it's the
  actual fix. See Marinos et al. 2020, Results ("E. coli growth in
  modelled LB-medium") for a worked example of this exact reinterpretation,
  including how they validated their yield prediction (1.54 g/L) against
  an independent OD600-based estimate (~1.5 g/L; OD600-to-dry-weight
  conversion factors from Milo R, Jorgensen P, Moran U, Weber G, Springer M
  (2010) "BioNumbers -- the database of key numbers in molecular and cell
  biology." Nucleic Acids Res 38:D750-D753).

  If you later want literal h^-1 rate predictions, that requires real
  uptake-rate/kinetic data for B. longum (or a closely related organism)
  -- see Teusink B, Wiersma A, Molenaar D, Francke C, de Vos WM, Siezen RJ,
  Smid EJ (2006) "Analysis of growth of Lactobacillus plantarum WCFS1 on a
  complex medium using a genome-scale metabolic model." J Biol Chem
  281:40041-40048, which is the closest published precedent (a
  closely-related LAB species, complex medium, GEM) for how that
  calibration (q_s = mu / Y_x/s from measured fermentation time-courses)
  is actually done. Don't attempt this without real time-course data --
  it needs measured glucose-depletion and biomass curves, not literature
  guesses, or you'll just reintroduce the same "arbitrary number" problem
  a layer deeper (a risk Marinos et al. explicitly call out: "in numerous
  instances arbitrary numbers are used for nutritional constraints").
"""

from typing import Dict


# ----------------------------------------------------------------------
# Molecular weights (g/mol), covering every metabolite currently used in
# media_components_updated.py's AMINO_ACIDS / VITAMIN_GROUPS / trace
# metals / nucleotides. Source: PubChem (Kim S, Chen J, Cheng T, et al.
# (2019) "PubChem 2019 update: improved access to chemical data." Nucleic
# Acids Res 47:D1102-D1109) -- the same database Marinos et al. 2020 (their
# Table 2) recommend for exactly this step. Values rounded to 2 dp.
# ----------------------------------------------------------------------
MW_G_PER_MOL: Dict[str, float] = {
    # Amino acids (free acid forms, as used in exchange reactions)
    "ala_L": 89.09, "arg_L": 174.20, "asn_L": 132.12, "asp_L": 133.10,
    "cys_L": 121.16, "gln_L": 146.14, "glu_L": 147.13, "gly": 75.07,
    "his_L": 155.16, "ile_L": 131.17, "leu_L": 131.17, "lys_L": 146.19,
    "met_L": 149.21, "phe_L": 165.19, "pro_L": 115.13, "ser_L": 105.09,
    "thr_L": 119.12, "trp_L": 204.23, "tyr_L": 181.19, "val_L": 117.15,

    # B-vitamins / cofactors
    "thm": 337.27, "ribflv": 376.36, "pnto_R": 219.23, "pydxn": 169.18,
    "pydx": 167.16, "pydam": 168.19, "nac": 123.11, "cbl1": 1355.37,
    "fol": 441.40, "btn": 244.31, "thf": 445.43, "5mthf": 459.46,
    "adocbl": 1579.61, "4abz": 137.14, "dpcoa": 767.55,

    # Trace metals (atomic weights -- these are ions, not molecules)
    "zn2": 65.38, "fe2": 55.85, "fe3": 55.85, "mn2": 54.94,
    "cobalt2": 58.93, "cu2": 63.55,

    # Nucleobases
    "ade": 135.13, "gua": 151.13, "hxan": 136.11, "xan": 152.11,
    "orot": 156.10,

    # Simple/salt-derived ingredients used as SimpleComponent/MultiSimpleComponent
    "glc_D": 180.16, "pi": 174.18, "nh4": 18.04, "ac": 60.05,
    "na1": 22.99, "mg2": 24.31, "so4": 96.06, "mn2_salt": 54.94,
}


def mass_fraction_to_mmol_per_g(mass_fraction_g_per_g: float, metabolite_id: str) -> float:
    """
    Convert a composition-sheet mass fraction (g nutrient / g extract,
    dimensionless) into mmol of that nutrient per gram of extract.

    This is the fix for CompositeComponent profiles (peptone/meat/yeast
    extract): use this instead of the raw mass fraction as the profile
    weight. `x` (the decision variable) then means "grams of this extract
    added per litre of medium," and contribution(x) = weight * x yields
    mmol of nutrient per litre -- directly comparable to every other
    component once you also fix those (see concentration_gL_to_mmolL).
    """
    mw = MW_G_PER_MOL[metabolite_id]
    return (mass_fraction_g_per_g / mw) * 1000.0


def concentration_gL_to_mmolL(concentration_g_per_L: float, metabolite_id: str) -> float:
    """
    Convert a real recipe concentration (g/L) into mmol/L -- the fix for
    SimpleComponent/MultiSimpleComponent bounds (glucose, phosphate,
    MgSO4, ...), which were previously just the raw g/L number reused
    with no conversion at all.
    """
    mw = MW_G_PER_MOL[metabolite_id]
    return (concentration_g_per_L / mw) * 1000.0


# ----------------------------------------------------------------------
# Worked example: rebuilding the yeast-extract profile and the glucose
# bound the correct way. Apply the SAME pattern to the rest of your
# components (peptone, meat extract, phosphate, ammonium, MgSO4, MnSO4).
# ----------------------------------------------------------------------

# --- Example 1: glucose, a SimpleComponent ---
# OLD (wrong): ub = 27.36  (this is Cheng et al.'s g/L figure, reused
#              directly as an mmol/gDW/h bound -- a category error)
# NEW (yield mode): convert the same real g/L figure to mmol/L
GLUCOSE_UB_MMOL_PER_L = concentration_gL_to_mmolL(27.36, "glc_D")
print(f"Glucose: 27.36 g/L  ->  {GLUCOSE_UB_MMOL_PER_L:.2f} mmol/L ")
    
PHOSPHATE_UB_MMOL_PER_L = concentration_gL_to_mmolL(2.0, "pi")
print(f"phosphate : -> {PHOSPHATE_UB_MMOL_PER_L:.2f} mmol/L")
    
AMMUNIOM_UB_MMOL_PER_L = concentration_gL_to_mmolL(2.0, "nh4")
print(f"ammonuioum: -> {AMMUNIOM_UB_MMOL_PER_L} mmmol/L")
    
ACETATE_UB_MMOL_PER_L = concentration_gL_to_mmolL(5.0, "ac")
NA_UB_MMOL_PER_L = concentration_gL_to_mmolL(5.0, "na1")
MG_UB_MMOL_PER_L = concentration_gL_to_mmolL(0.8, "mg2")
SULFATE_UB_MMOL_PER_L = concentration_gL_to_mmolL(0.8, "so4")
MN_UB_MMOL_PER_L = concentration_gL_to_mmolL(0.09, "mn2")
MNSULFATE_UB_MMOL_PER_L = concentration_gL_to_mmolL(0.09, "so4")
OVERALSULFATE_UB_MMOL_PER_L = MNSULFATE_UB_MMOL_PER_L + SULFATE_UB_MMOL_PER_L

if __name__ == "__main__":
    example_mass_fractions = {
        "ala_L": 0.048, "glu_L": 0.115,          # amino acids, dominant mass fraction
        "ribflv": 1.25e-4, "cbl1": 3e-6,          # vitamins, genuinely trace by mass
        "zn2": 1.36e-4, "cobalt2": 5e-7,          # trace metals
    }
    print("\nYeast extract profile, mass-fraction weight vs. correct molar weight:")
    for met_id, frac in example_mass_fractions.items():
        molar_weight = mass_fraction_to_mmol_per_g(frac, met_id)
        print(f"   {met_id:10s}  mass fraction={frac:.2e} g/g  ->  "
              f"{molar_weight:.4e} mmol/g extract")

    # At a realistic dose (Cheng et al.'s own optimized recipe: 19.524 g/L
    # yeast extract -- use this as your ub, not 1000):
    dose_g_per_L = 19.524
    print(f"\nAt a realistic dose of {dose_g_per_L} g/L yeast extract:")
    for met_id, frac in example_mass_fractions.items():
        molar_weight = mass_fraction_to_mmol_per_g(frac, met_id)
        supplied_mmol_per_L = molar_weight * dose_g_per_L
        print(f"   {met_id:10s}  ->  {supplied_mmol_per_L:.4e} mmol/L supplied")

    print("\nNOTE: whatever FBA growth value you get after switching every\n"
          "component to this basis is a predicted biomass YIELD (g DCW per\n"
          "litre of medium), NOT an h^-1 growth rate. Relabel growth_ref /\n"
          "reference growth accordingly throughout your code and report.\n"
          "Validate it the way Marinos et al. (2020) validated LB: convert\n"
          "Cheng et al.'s reported CFU/mL (or your own OD600 data, if you\n"
          "have any) to an approximate g DCW/L using a literature OD600-to-\n"
          "dry-weight factor, and check the two numbers are in the same\n"
          "ballpark -- that's your external sanity check, not just internal\n"
          "consistency between your own optimizers.")
