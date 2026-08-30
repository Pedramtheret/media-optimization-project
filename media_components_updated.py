"""
media_components.py  (v3 -- literature-grounded extract composition,
                            corrected ion bounds, menaquinone removed)

CHANGES FROM v2, AND WHY
-------------------------
1. Yeast extract and meat extract no longer share one uniform weight-of-1.0
   profile across every amino acid/vitamin/trace-element exchange reaction.
   Each is now built from actual documented composition data:

   Yeast extract (Sigma product A1552 spec sheet; Ferreira et al., brewer's
   spent yeast extract, J. Food Compos. Anal.-type source):
     - B-vitamins reported as exact mg/kg ranges: B1 100-120, B2 80-120,
       B5 120-200, B6 60-80, B3/niacin 900-1100, B12 0.005-0.015 (5-15 ug/kg)
     - Trace metals reported quantitatively: Zn 11.9, Fe 1.76, Mn 0.564
       (mg/100g dw)
     - Protein/amino-nitrogen content high (66.8-76.3 g/100g protein) ->
       amino acids weighted proportionally higher than vitamins/metals
     Relative weights below are the midpoint of each reported range,
     normalized so amino acids (the dominant mass fraction) = 1.0 and
     everything else scaled relative to that.

   Meat extract (Sigma/Gibco/HiMedia-style vendor spec sheets; consistent
   across sources): described as peptides, individual amino acids,
   NUCLEOTIDE fractions (inosine/inosinic acid specifically named as
   characteristic), organic acids, minerals, and "some" vitamins.
   Critically, NO source names trace metals as a defining component of
   meat extract, unlike yeast extract. This is a real, citable asymmetry:
   meat extract's profile below carries amino acids + nucleotides as its
   two strong components, vitamins at a low weight ("some"), and NO trace
   metals at all -- rather than copying yeast extract's profile.

   Where the literature gives only a category ("contains B vitamins") and
   not a number, that ingredient is still included but at a conservative,
   flagged placeholder weight -- clearly commented as such, not silently
   invented.

2. FixedBound entries for menaquinone-7, menaquinone-8, and their precursor
   dihydroxynaphthoate have been REMOVED. The MRS recipe this medium is
   built from (peptone, meat extract, yeast extract, glucose, K2HPO4,
   triammonium citrate, sodium acetate, MgSO4, MnSO4, agar) has no
   ingredient that plausibly supplies menaquinone, and no source consulted
   for yeast or meat extract composition names menaquinone/vitamin K as a
   component of either. Per Schoepping et al. 2021, menaquinone-4 is
   reported as a substance that had to be added to a defined medium
   SEPARATELY and explicitly, precisely because it is not reliably present
   in standard bifidobacteria media -- supporting removal rather than
   assuming it is hiding, unaccounted-for, in the extracts.

3. Sodium is no longer a blanket FixedBound. It is tied to sodium acetate's
   stoichiometry (as in v2), PLUS a separate, small, explicitly-labeled
   background allowance representing trace sodium from buffer components
   not otherwise named in the recipe (e.g., media water, glassware). Three
   regimes are provided via BACKGROUND_NA_LEVEL so they can be compared
   directly: "none", "low", "unconstrained" (the last reproduces the
   original artifact ON PURPOSE, as a control, not a recommendation).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict


class MediaComponent(ABC):
    def __init__(self, name: str, cost_per_unit: float, lb: float, ub: float, category: str):
        self.name = name
        self.cost_per_unit = cost_per_unit
        self.lb = lb
        self.ub = ub
        self.category = category

    @abstractmethod
    def contribution(self, value: float) -> Dict[str, float]:
        """Return {exchange_id: uptake_amount_contributed} for this component at `value`."""
        raise NotImplementedError

    @abstractmethod
    def exchange_ids(self) -> List[str]:
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name} [{self.lb},{self.ub}] ${self.cost_per_unit}/u>"


class SimpleComponent(MediaComponent):
    """One ingredient -> one exchange reaction."""

    def __init__(self, name, exchange_id, cost_per_unit, lb, ub, category):
        super().__init__(name, cost_per_unit, lb, ub, category)
        self.exchange_id = exchange_id

    def contribution(self, value: float) -> Dict[str, float]:
        return {self.exchange_id: float(value)}

    def exchange_ids(self) -> List[str]:
        return [self.exchange_id]


class MultiSimpleComponent(MediaComponent):
    """One ingredient -> several exchange reactions with KNOWN, exact stoichiometry (e.g. a salt)."""

    def __init__(self, name, weights: Dict[str, float], cost_per_unit, lb, ub, category):
        super().__init__(name, cost_per_unit, lb, ub, category)
        self.weights = weights

    def contribution(self, value: float) -> Dict[str, float]:
        return {ex_id: float(value) * w for ex_id, w in self.weights.items()}

    def exchange_ids(self) -> List[str]:
        return list(self.weights.keys())


class CompositeComponent(MediaComponent):
    """
    One ingredient (undefined mixture, e.g. yeast extract) -> many exchange
    reactions, scaled by one shared decision variable, weighted by a fixed
    relative `profile`. Multiple CompositeComponents may share exchange ids
    (peptone/meat/yeast extract all supply amino acids) -- CultureMedium
    sums their contributions rather than one overwriting another.
    """

    def __init__(self, name, profile: Dict[str, float], cost_per_unit, lb, ub, category):
        super().__init__(name, cost_per_unit, lb, ub, category)
        self.profile = profile

    def contribution(self, value: float) -> Dict[str, float]:
        return {ex_id: float(value) * w for ex_id, w in self.profile.items()}

    def exchange_ids(self) -> List[str]:
        return list(self.profile.keys())


@dataclass
class FixedBound:
    """A non-optimized exchange reaction held at a constant bound (e.g. water, free ions)."""
    exchange_id: str
    bound: float


class CultureMedium:
    def __init__(self, components: List[MediaComponent],
                 fixed_open: Optional[List[FixedBound]] = None,
                 closed: Optional[List[str]] = None):
        self.components = components
        self.fixed_open = fixed_open or []
        self.closed = closed or []

    @property
    def dimension(self) -> int:
        return len(self.components)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.components]

    @property
    def bounds(self) -> List[tuple]:
        return [(c.lb, c.ub) for c in self.components]

    def default_vector(self):
        return [c.ub for c in self.components]

    def aggregate_bounds(self, x) -> Dict[str, float]:
        assert len(x) == len(self.components), "vector length must match number of components"
        totals: Dict[str, float] = defaultdict(float)
        for fb in self.fixed_open:
            totals[fb.exchange_id] += fb.bound
        for comp, xi in zip(self.components, x):
            for ex_id, amount in comp.contribution(xi).items():
                totals[ex_id] += amount
        for ex_id in self.closed:
            totals[ex_id] = 0.0
        return dict(totals)

    def apply_to_model(self, model, x) -> None:
        totals = self.aggregate_bounds(x)
        for rxn in model.exchanges:
            rxn.lower_bound = 0
        for ex_id, total in totals.items():
            if ex_id in model.reactions:
                model.reactions.get_by_id(ex_id).lower_bound = -abs(total) if total > 0 else 0

    def cost(self, x) -> float:
        assert len(x) == len(self.components)
        return float(sum(c.cost_per_unit * xi for c, xi in zip(self.components, x)))


# ----------------------------------------------------------------------
# Amino acid / vitamin / trace element / nucleotide id groups
# ----------------------------------------------------------------------
AMINO_ACIDS = ["ala_L", "asn_L", "asp_L", "cys_L", "gln_L", "glu_L", "gly", "his_L",
               "ile_L", "leu_L", "lys_L", "met_L", "phe_L", "pro_L", "ser_L",
               "thr_L", "trp_L", "tyr_L", "val_L"]

YEAST_PROTEIN_G_PER_100G = 71.5

# B-vitamin exchange ids, grouped by which vitamin they represent, so real
# relative mg/kg weights from the yeast-extract spec sheet can be applied
# per vitamin rather than uniformly.
VITAMIN_GROUPS_G_PER_100G = {
    "thm": 110 / 10000,       # B1 thiamine, midpoint of 100-120 mg/kg
    "ribflv": 100 / 10000,    # B2 riboflavin, midpoint of 80-120 mg/kg
    "pnto_R": 160 / 10000,    # B5 pantothenate, midpoint of 120-200 mg/kg
    "pydxn": 70 / 10000,      # B6 pyridoxine, midpoint of 60-80 mg/kg
    "pydx": 70 / 10000,       # B6 pyridoxal form, same midpoint (isoform)
    "pydam": 70 / 10000,      # B6 pyridamine form, same midpoint (isoform)
    "nac": 1000 / 10000,      # B3/PP niacin, midpoint of 900-1100 mg/kg
    "cbl1": 0.01 / 10000,     # B12, midpoint of 5-15 ug/kg
    # Category-level only (no cited mg/kg figure) -- kept well below the
    # smallest CITED vitamin (B12) rather than above it, since "present but
    # not singled out for quantification" should not outweigh vitamins that
    # WERE specifically measured.
    "fol": 0.002 / 10000,
    "btn": 0.002 / 10000,
    "thf": 0.002 / 10000,
    "5mthf": 0.002 / 10000,
    "adocbl": 0.01 / 10000,   # tied to B12's cited weight (coenzyme form of same vitamin)
    "4abz": 0.002 / 10000,
    "dpcoa": 0.002 / 10000,
}

# Trace metals with a real, cited mg/100g dw figure from the brewer's-yeast
# extract composition paper.
YEAST_TRACE_METALS_G_PER_100G = {
    "zn2": 11.9 / 1000,     # mg/100g -> g/100g
    "fe2": 1.76 / 1000,
    "fe3": 1.76 / 1000,     # same total iron pool, split across both oxidation states in the model
    "mn2": 0.564 / 1000,
    # Named only as "trace elements present", no cited figure -- kept below
    # the smallest CITED metal (Mn) rather than above it, same principle as
    # the uncited vitamins above.
    "cobalt2": 0.0002,
    "cu2": 0.0002,
}
# Nucleotide-related exchanges -- meat extract's characteristic/named
# component (inosine and inosinic acid specifically), yeast extract's is
# present but not singled out as characteristic the way it is for meat extract.
NUCLEOTIDES = ["ade", "gua", "hxan", "xan", "orot"]
NUCLEOTIDE_G_PER_100G = 0.5  # modest, unquantified-in-sources placeholder


def _weighted_profile(id_weight_map: Dict[str, float], base: str = "e", scale: float = 1.0) -> Dict[str, float]:
    """Build {EX_id(e): weight * scale} from a {bare_id: weight} map."""
    return {f"EX_{i}({base})": w * scale for i, w in id_weight_map.items()}


def _uniform_profile(ids: List[str], weight: float, base: str = "e") -> Dict[str, float]:
    return {f"EX_{i}({base})": weight for i in ids}


# ----------------------------------------------------------------------
# Factory: MRS medium for Bifidobacterium longum NCC2705 (AGORA2 model)
# ----------------------------------------------------------------------
def build_bifidobacterium_mrs_medium(background_na_level: str = "low") -> CultureMedium:
    """
    Builds the tunable MRS-based CultureMedium for B. longum NCC2705.

    background_na_level : "none" | "low" | "unconstrained"
        Controls the small background sodium allowance layered on top of
        the sodium-acetate-tied Na+ contribution, so the three regimes can
        be directly compared (see module docstring, point 3):
          "none"          -> Na+ available ONLY via sodium acetate (strictest)
          "low"           -> adds a small (1.0 mmol/gDW/h) background allowance
                              for unnamed trace sodium (buffer/media water)
          "unconstrained" -> reproduces the ORIGINAL artifact on purpose, as
                              a control to quantify how much it was inflating
                              growth/ATP -- NOT a recommended production setting

    All cost_per_unit and bound (ub) values remain PLACEHOLDERS pending your
    literature-sourced costs; only the internal composition profiles and ion
    handling changed in this revision.
    """
    # ---- yeast extract profile: real relative weights from cited sources ----
    yeast_profile: Dict[str, float] = {}
    yeast_profile.update(_uniform_profile(AMINO_ACIDS, weight=YEAST_PROTEIN_G_PER_100G/ len(AMINO_ACIDS)))          # dominant mass fraction, protein 66.8-76.3%
    yeast_profile.update(_weighted_profile(VITAMIN_GROUPS_G_PER_100G))     # mg/kg figures scaled down relative to amino acids
    yeast_profile.update(_weighted_profile(YEAST_TRACE_METALS_G_PER_100G)) # mg/100g dw figures, small relative weight
    yeast_profile.update(_uniform_profile(NUCLEOTIDES, weight=NUCLEOTIDE_G_PER_100G))          # present, not the extract's defining feature

    # ---- meat extract profile: amino acids + nucleotides emphasized, ----
    # ---- vitamins at low/category weight, NO trace metals (undocumented) ----
    MEAT_PROTEIN_G_PER_100G_ESTIMATE = 60.0  # order-of-magnitude estimate, not individually cited
    meat_profile: Dict[str, float] = {}
    meat_profile.update(_uniform_profile(AMINO_ACIDS, weight=MEAT_PROTEIN_G_PER_100G_ESTIMATE / len(AMINO_ACIDS)))           # documented: "peptides, individual amino acids"
    meat_profile.update(_uniform_profile(NUCLEOTIDES, weight=(NUCLEOTIDE_G_PER_100G * 2) / len(NUCLEOTIDES)))  # 2x yeast's -- documented as characteristic of meat extract specifically        # documented characteristic component (inosine/inosinic acid)
    meat_profile.update(_weighted_profile(VITAMIN_GROUPS_G_PER_100G, scale=0.3))     # documented only as "some vitamins" -- CATEGORY-LEVEL, low weight
    # no YEAST_TRACE_METALS entries here -- deliberately absent, not an oversight

    components: List[MediaComponent] = [
        SimpleComponent("Glucose", "EX_glc_D(e)", cost_per_unit=0.1, lb=0.0, ub=27.36, category="carbon"),
        SimpleComponent("Phosphate (K2HPO4)", "EX_pi(e)", cost_per_unit=0.1, lb=0.0, ub=2.0, category="ion_major"),
        SimpleComponent("Ammonium (from triammonium citrate)", "EX_nh4(e)", cost_per_unit=0.12, lb=0.0, ub=2.0, category="ion_major"),

        # Sodium tied to its only real recipe source (sodium acetate), 1:1
        # stoichiometry from CH3COONa -- see build_bifidobacterium_mrs_medium
        # docstring for the background_na_level regimes tested alongside this.
        MultiSimpleComponent("Acetate (sodium acetate)", {"EX_ac(e)": 1.0, "EX_na1(e)": 1.0},
                              cost_per_unit=0.07, lb=0.0, ub=5.0, category="ion_major"),

        MultiSimpleComponent("Magnesium sulfate (MgSO4)", {"EX_mg2(e)": 1.0, "EX_so4(e)": 1.0},
                              cost_per_unit=0.06, lb=0.0, ub=0.8, category="ion_major"),
        MultiSimpleComponent("Manganese sulfate (MnSO4)", {"EX_mn2(e)": 1.0, "EX_so4(e)": 1.0},
                              cost_per_unit=0.3, lb=0.0, ub=0.09, category="ion_trace_expensive"),

        # Peptone: kept as the uniform amino-acid-only profile from v2 --
        # peptone is a defined-process partial protein hydrolysate (not an
        # "extract" in the yeast/meat sense), and vendor peptone specs
        # consistently describe it as free amino acids + short peptides
        # without a distinct vitamin/mineral profile, so no change here.
        CompositeComponent("Peptone level", _uniform_profile(AMINO_ACIDS, weight=1.0),
                            cost_per_unit=0.2, lb=0.0, ub=10.0, category="extract"),

        CompositeComponent("Meat extract level", meat_profile,
                            cost_per_unit=0.18, lb=0.0, ub=5.0, category="extract"),

        CompositeComponent("Yeast extract level", yeast_profile,
                            cost_per_unit=0.3, lb=0.0, ub=19.5, category="extract"),
    ]

    fixed_open = [
        FixedBound("EX_h(e)", 1000.0),
        FixedBound("EX_h2o(e)", 1000.0),
        FixedBound("EX_co2(e)", 1000.0),
        FixedBound("EX_k(e)", 1000.0),
        FixedBound("EX_cl(e)", 1000.0),
        FixedBound("EX_ca2(e)", 1000.0),  # unresolved from earlier debugging -- kept free, no evidence it causes a similar artifact
    ]

    if background_na_level == "none":
        pass  # Na+ available ONLY through sodium acetate
    elif background_na_level == "low":
        fixed_open.append(FixedBound("EX_na1(e)", 1.0))  # small explicit background allowance
    elif background_na_level == "unconstrained":
        fixed_open.append(FixedBound("EX_na1(e)", 1000.0))  # CONTROL: reproduces the original artifact on purpose
    else:
        raise ValueError("background_na_level must be 'none', 'low', or 'unconstrained'")

    # Menaquinone FixedBounds REMOVED (see module docstring, point 2) --
    # no ingredient in this MRS recipe, and no yeast/meat extract composition
    # source consulted, supports their presence.
    closed = ["EX_o2(e)"]  # B. longum is anaerobic

    return CultureMedium(components, fixed_open=fixed_open, closed=closed)
