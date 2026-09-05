"""
debug_zero_growth.py

Diagnostic pipeline for "growth is 0 even at my medium's richest point":

  STAGE 1 (find_unlocking_nutrients): tests each uncovered reaction alone.
     Already run -- found nothing, meaning multiple nutrients are jointly
     required.

  STAGE 2 (group_removal_scan): top-down. Starts from all 155 exchange
     reactions open (confirmed reaches 67.4 h^-1), removes one named
     CATEGORY at a time, checks how much growth drops. Already run --
     flagged 'trace_metals' and 'misc_other' as essential.

  STAGE 3 (narrow_group): pinpoints the exact reaction(s) within a
     flagged group, one at a time.

  STAGE 4 (pantethine check): literature-guided targeted check. Schoepping
     et al. 2021 (npj Syst Biol Appl) found that two OTHER Bifidobacterium
     GEMs (B. animalis BB-12, B. longum BB-46) have an absolute, structural
     requirement for pantethine (not ordinary pantothenate/vitamin B5) as
     a coenzyme-A precursor, because both strains lack the genes to build
     CoA starting from pantothenate directly. Neither their generic medium
     nor ours supplies pantethine. This checks whether the model even HAS
     a pantethine exchange reaction under likely naming conventions.

GROUPS is built PROGRAMMATICALLY below: start from every exchange
reaction id in the model's original default medium (the 155-reaction
list from the start of this project), subtract whatever your
CultureMedium already covers (media_components.py), and bucket what's
left into categories by simple substring rules. Deliberately mechanical
rather than hand-typed, since an earlier hand-typed version accidentally
included reactions the medium already covered.

    python debug_zero_growth.py
"""

import re
import cobra
from media_converted import build_bifidobacterium_mrs_medium
from gem_evaluator import GEMEvaluator
from pathlib import Path

MODEL_PATH = Path("D:/workspace/Hakim project/AGORA2_SBML/Bifidobacterium_longum_NCC2705.xml")

# The model's original 155-reaction default medium, from the very first
# message in this project. Kept as a literal list (rather than re-derived
# from the model at runtime) so this script still works for the group
# scan even before you've confirmed the model loads correctly elsewhere.
FULL_DEFAULT_MEDIUM_IDS = [
    "EX_12ppd_S(e)", "EX_4abut(e)", "EX_4abz(e)", "EX_4ahmmp(e)", "EX_5fura(e)", "EX_5mthf(e)",
    "EX_7a_czp(e)", "EX_C02528(e)", "EX_HC02191(e)", "EX_HC02192(e)", "EX_HC02193(e)",
    "EX_M01989(e)", "EX_M03134(e)", "EX_ac(e)", "EX_acald(e)", "EX_ade(e)", "EX_adocbl(e)",
    "EX_ala_L(e)", "EX_alaasp(e)", "EX_alagln(e)", "EX_alaglu(e)", "EX_alagly(e)", "EX_alahis(e)",
    "EX_alaleu(e)", "EX_alathr(e)", "EX_anzp(e)", "EX_arab_L(e)", "EX_arabinogal(e)", "EX_arabttr(e)",
    "EX_asn_L(e)", "EX_asp_L(e)", "EX_biomass(e)", "EX_btn(e)", "EX_butam(e)", "EX_ca2(e)",
    "EX_cbl1(e)", "EX_cd2(e)", "EX_cgly(e)", "EX_chlphncl(e)", "EX_cholate(e)", "EX_cl(e)",
    "EX_co2(e)", "EX_cobalt2(e)", "EX_cu2(e)", "EX_cys_L(e)", "EX_czp(e)", "EX_dchac(e)",
    "EX_dextrin(e)", "EX_dgchol(e)", "EX_dhna(e)", "EX_dma(e)", "EX_dpcoa(e)", "EX_drib(e)",
    "EX_etoh(e)", "EX_fcsn(e)", "EX_fe2(e)", "EX_fe3(e)", "EX_fol(e)", "EX_for(e)", "EX_fru(e)",
    "EX_fum(e)", "EX_gal(e)", "EX_galam(e)", "EX_gchola(e)", "EX_glc_D(e)", "EX_glcur(e)",
    "EX_gln_L(e)", "EX_glu_L(e)", "EX_gly(e)", "EX_glyasn(e)", "EX_glyasp(e)", "EX_glyc(e)",
    "EX_glycys(e)", "EX_glygln(e)", "EX_glyglu(e)", "EX_glyleu(e)", "EX_glymet(e)", "EX_glyphe(e)",
    "EX_glypro(e)", "EX_glytyr(e)", "EX_gua(e)", "EX_h(e)", "EX_h2o(e)", "EX_h2s(e)", "EX_hg2(e)",
    "EX_his_L(e)", "EX_hxan(e)", "EX_ile_L(e)", "EX_isomal(e)", "EX_k(e)", "EX_kesto(e)",
    "EX_kestopt(e)", "EX_kestottr(e)", "EX_lac_L(e)", "EX_lactl(e)", "EX_lcts(e)", "EX_leu_L(e)",
    "EX_lys_L(e)", "EX_malt(e)", "EX_malthx(e)", "EX_malttr(e)", "EX_mantr(e)", "EX_melib(e)",
    "EX_met_D(e)", "EX_met_L(e)", "EX_metala(e)", "EX_metsox_S_L(e)", "EX_mg2(e)", "EX_mn2(e)",
    "EX_mqn7(e)", "EX_mqn8(e)", "EX_na1(e)", "EX_nac(e)", "EX_nchlphncl(e)", "EX_nh4(e)",
    "EX_no3(e)", "EX_norval_L(e)", "EX_nzp(e)", "EX_o2(e)", "EX_orot(e)", "EX_pb(e)", "EX_peamn(e)",
    "EX_phe_L(e)", "EX_pi(e)", "EX_pnto_R(e)", "EX_ppi(e)", "EX_pro_L(e)", "EX_pydam(e)",
    "EX_pydx(e)", "EX_pydxn(e)", "EX_raffin(e)", "EX_rib_D(e)", "EX_ribflv(e)", "EX_rmn(e)",
    "EX_ser_L(e)", "EX_so4(e)", "EX_stys(e)", "EX_succ(e)", "EX_sucr(e)", "EX_taur(e)",
    "EX_tchola(e)", "EX_tdchola(e)", "EX_tdechola(e)", "EX_thf(e)", "EX_thm(e)", "EX_thr_L(e)",
    "EX_trp_L(e)", "EX_turan_D(e)", "EX_tyr_L(e)", "EX_urea(e)", "EX_val_L(e)", "EX_xan(e)",
    "EX_xyl_D(e)", "EX_xylottr(e)", "EX_zn2(e)",
]


def _categorize(rxn_id: str) -> str:
    """Simple substring-based bucketing for whatever's left after subtracting covered ids."""
    core = rxn_id[3:-3]  # strip "EX_" and "(e)"
    if re.search(r"chol|dchac", core):
        return "bile_acids_and_host_derived"
    if core in {"mqn7", "mqn8", "dhna"}:
        return "menaquinones_and_quinone_precursors"
    if core in {"ca2", "cd2", "cobalt2", "cu2", "fe2", "fe3", "hg2", "mn2", "pb", "zn2"}:
        return "trace_metals"
    if core in {"ade", "gua", "hxan", "xan", "orot"}:
        return "nucleobases_and_related"
    if core in {"btn", "fol", "nac", "pnto_R", "pydam", "pydx", "pydxn", "ribflv", "thf", "thm",
                "5mthf", "cbl1", "adocbl", "4abz", "dpcoa", "4ahmmp", "5fura", "7a_czp", "anzp",
                "chlphncl", "nchlphncl", "czp", "nzp"}:
        return "vitamins_and_cofactors_not_in_medium"
    if re.match(r"ala|gly[a-z]|metala|cgly", core) and core not in {"ala_L", "gly"}:
        return "peptides_and_dipeptides"
    if core in {"fru", "gal", "lcts", "sucr", "malt", "dextrin", "raffin", "melib", "arab_L",
                "xyl_D", "rib_D", "drib", "rmn", "arabinogal", "arabttr", "isomal", "kesto",
                "kestopt", "kestottr", "malthx", "malttr", "mantr", "stys", "turan_D",
                "xylottr", "glyc", "glcur", "galam", "ac", "acald", "etoh", "for", "fum",
                "lac_L", "lactl", "succ", "12ppd_S"}:
        return "other_sugars_and_carbon"
    return "misc_other"


def _build_groups(medium) -> "dict[str, list[str]]":
    covered = set()
    for comp in medium.components:
        covered.update(comp.exchange_ids())
    for fb in medium.fixed_open:
        covered.add(fb.exchange_id)
    covered.update(medium.closed)
    covered.add("EX_biomass(e)")  # not a real nutrient, exclude from testing

    remaining = [rid for rid in FULL_DEFAULT_MEDIUM_IDS if rid not in covered]

    groups: "dict[str, list[str]]" = {}
    for rid in remaining:
        groups.setdefault(_categorize(rid), []).append(rid)
    return groups



def main():
    print("Loading model ...")
    model = cobra.io.read_sbml_model(MODEL_PATH)
    medium = build_bifidobacterium_mrs_medium()
    evaluator = GEMEvaluator(model, medium)

    groups = _build_groups(medium)  # BUGFIX: this was referencing an undefined bare `GROUPS` name before

    growth_ref = evaluator.measure_reference()
    if 0 < growth_ref < 1e-6:
        print(f"Growth at richest point: {growth_ref!r} -- solver floating-point noise, treating as 0.")
        growth_ref = 0.0
    else:
        print(f"Growth at richest point: {growth_ref:.4f} h^-1")

    if growth_ref > 1e-6:
        print("Nonzero -- you're not blocked anymore. Safe to move on to run_optimization.py.")
        return

    print("\n--- STAGE 1: single uncovered reaction scan ---")
    hits = evaluator.find_unlocking_nutrients()
    if hits:
        print(f"\nFound {len(hits)} single-reaction fix(es) -- add the top one to fixed_open "
              f"in media_components.py and re-run.")
        return

    print("\n--- STAGE 2: top-down group-removal scan (starting from all 155 open) ---")
    group_results = evaluator.group_removal_scan(groups)
    if not group_results:
        return

    damaging = [r for r in group_results if r[2] < 5]
    print("\n" + "=" * 60)
    if not damaging:
        print("No single group's removal collapsed growth on its own -- the essential nutrient(s)")
        print("are split across categories. Try removing PAIRS of groups together next.")
        return

    print("Group(s) whose removal collapses growth:")
    for gname, g, pct in damaging:
        print(f"   {gname}  ({pct:.1f}% of full growth remains without it)")

    print("\n--- STAGE 3: narrowing flagged group(s) to exact reaction(s) ---")
    for gname, g, pct in damaging:
        print(f"\nNarrowing '{gname}':")
        evaluator.narrow_group(groups[gname])

    print("\n--- STAGE 4: literature-guided check -- pantethine (Schoepping et al. 2021) ---")
    print("Neither B. animalis BB-12 nor B. longum BB-46 can build coenzyme A from ordinary")
    print("pantothenate alone -- both require pantethine directly, and it's absent from our")
    print("medium entirely. Checking whether this model has a pantethine exchange reaction,")
    print("under each plausible naming convention, and whether opening it changes anything:\n")
    pantethine_candidates = ["EX_ptth(e)", "EX_pnto_R(e)", "EX_pan4p(e)", "EX_dpcoa(e)",
                              "EX_4ppan(e)", "EX_4ppcys(e)", "EX_coa(e)"]
    present = [rid for rid in pantethine_candidates if rid in model.reactions]
    print(f"   Candidates present in this model: {present if present else 'NONE of the guessed IDs exist in this model'}")
    if present:
        evaluator.narrow_group(present)
    else:
        print("   None of the standard pantethine/CoA-precursor exchange IDs exist in this "
              "reconstruction under the names we guessed. Worth grepping the model's full "
              "reaction list directly for 'ptth', 'pan4p', 'coa', or '4ppan' to confirm whether "
              "pantethine metabolism is modeled at all here, and under what ID, before concluding "
              "it isn't the issue -- NCC2705 may use different BiGG-style naming than what we guessed.")


if __name__ == "__main__":
    main()
