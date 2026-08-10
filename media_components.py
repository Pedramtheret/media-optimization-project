"""
media_components.py  (v2 -- fixes the composite-overwrite bug)

BUG THIS VERSION FIXES:
    v1 had each MediaComponent directly SET an exchange reaction's
    lower_bound via `model.reactions.get_by_id(ex_id).lower_bound = -value`.
    When multiple components legitimately share an exchange reaction (e.g.
    Peptone, Meat extract, and Yeast extract ALL supply alanine), applying
    them one after another meant each one silently overwrote the previous
    one's contribution -- only the LAST component in the list actually
    determined the final bound for any shared exchange id. This made most
    of the medium's amino-acid/vitamin capacity depend on a single
    component (whichever was last), which collapsed to ~0 whenever the
    optimizer reduced that one component to cut cost, even with the other
    two wide open.

FIX:
    Components no longer touch the model directly. Each one exposes
    `contribution(value) -> Dict[exchange_id, uptake_amount]`. CultureMedium
    SUMS these contributions across all components per exchange id, and
    applies the summed total to the model in a single pass. Overlap between
    components is now handled the way real overlapping media ingredients
    should be: additively, not by one silently discarding another.
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
        self.category = category  # "carbon","ion_major","ion_trace","extract",...

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
    """
    One ingredient -> several exchange reactions it DEFINITELY, distinctly
    supplies (e.g. MgSO4 -> both Mg2+ and SO4--), each at its own fixed
    molar weight relative to the component's single decision variable.
    Distinct from CompositeComponent only in naming/intent: this is for
    ingredients with a KNOWN, small, exact stoichiometry (like a salt),
    not an undefined mixture.
    """

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
    reactions, all scaled by the SAME single decision variable, weighted by
    a fixed relative `profile`. Multiple CompositeComponents are allowed to
    (and, for peptone/meat/yeast extract, are EXPECTED to) share exchange
    ids -- CultureMedium sums their contributions rather than letting one
    overwrite another.
    """

    def __init__(self, name, profile: Dict[str, float], cost_per_unit, lb, ub, category):
        super().__init__(name, cost_per_unit, lb, ub, category)
        self.profile = profile  # exchange_id -> relative weight

    def contribution(self, value: float) -> Dict[str, float]:
        return {ex_id: float(value) * w for ex_id, w in self.profile.items()}

    def exchange_ids(self) -> List[str]:
        return list(self.profile.keys())


@dataclass
class FixedBound:
    """A non-optimized exchange reaction held at a constant bound (e.g. water, free ions)."""
    exchange_id: str
    bound: float  # max uptake, mmol/gDW/h


class CultureMedium:
    """
    A CultureMedium is an ordered list of MediaComponents (the decision
    vector the optimizer controls) plus a set of FixedBounds (always-open,
    non-costed background components like water/protons/free ions), and
    a set of reactions to explicitly close (e.g. O2 -> 0 for anaerobic growth).

    apply_to_model() now works in two passes:
      1. AGGREGATE: sum every component's contribution() per exchange id
         (plus fixed_open bounds) into one dict of totals.
      2. APPLY: close every exchange reaction, then set each aggregated
         total once. Reactions in `closed` are forced to 0 regardless of
         what any component contributed to them (e.g. deliberately
         excluding O2 even if some component's profile touched it by
         mistake).
    """

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
        """Start at the upper bound of each component (a 'rich' starting point)."""
        return [c.ub for c in self.components]

    def aggregate_bounds(self, x) -> Dict[str, float]:
        """Sum every component's contribution + fixed_open, per exchange id. (Used by apply_to_model,
        exposed publicly too since it's useful for debugging/inspecting what a candidate x actually opens.)"""
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
        # 1) close everything first (clean slate)
        for rxn in model.exchanges:
            rxn.lower_bound = 0
        # 2) apply the SUMMED total for every exchange id touched by any
        #    component or fixed_open -- this is the fix: one write per
        #    exchange id, computed from the sum of all contributors, not
        #    N sequential overwriting writes.
        for ex_id, total in totals.items():
            if ex_id in model.reactions:
                model.reactions.get_by_id(ex_id).lower_bound = -abs(total) if total > 0 else 0

    def cost(self, x) -> float:
        assert len(x) == len(self.components)
        return float(sum(c.cost_per_unit * xi for c, xi in zip(self.components, x)))


# ----------------------------------------------------------------------
# Factory: MRS medium for Bifidobacterium longum NCC2705 (AGORA2 model)
# ----------------------------------------------------------------------
def build_bifidobacterium_mrs_medium() -> CultureMedium:
    """
    Builds the tunable MRS-based CultureMedium for B. longum NCC2705.

    Bound (`ub`) and `cost_per_unit` values here are the placeholders from
    before -- REPLACE with your literature-sourced values (you mentioned
    you already updated these; drop your numbers in here before running).
    """
    amino_acids = ["ala_L", "asn_L", "asp_L", "cys_L", "gln_L", "glu_L", "gly", "his_L",
                   "ile_L", "leu_L", "lys_L", "met_L", "phe_L", "pro_L", "ser_L",
                   "thr_L", "trp_L", "tyr_L", "val_L"]
    vitamins = ["btn", "fol", "nac", "pnto_R", "pydam", "pydx", "pydxn", "ribflv",
                "thm", "thf", "5mthf", "cbl1", "adocbl", "4abz", "dpcoa"]
    trace_metals = ["fe2", "fe3", "zn2", "cobalt2", "cu2"]
    nucleotides = ["ade", "gua", "hxan", "xan", "orot"]

    def profile(ids, weight):
        return {f"EX_{i}(e)": weight for i in ids}

    components: List[MediaComponent] = [
        # --- directly mapped, individually tunable ingredients ---
        SimpleComponent("Glucose", "EX_glc_D(e)", cost_per_unit=0.1, lb=0.0, ub=27.36, category="carbon"),
        SimpleComponent("Phosphate (K2HPO4)", "EX_pi(e)", cost_per_unit=0.1, lb=0.0, ub=2.0, category="ion_major"),
        SimpleComponent("Ammonium (from triammonium citrate)", "EX_nh4(e)", cost_per_unit=0.12, lb=0.0, ub=2.0, category="ion_major"),
        SimpleComponent("Acetate (sodium acetate)", "EX_ac(e)", cost_per_unit=0.07, lb=0.0, ub=5.0, category="ion_major"),

        # FIX: magnesium sulfate and manganese sulfate BOTH contribute sulfate;
        # MultiSimpleComponent + the new summing aggregator means their so4
        # contributions correctly ADD UP instead of one silently discarding the other.
        MultiSimpleComponent("Magnesium sulfate (MgSO4)", {"EX_mg2(e)": 1.0, "EX_so4(e)": 1.0},
                              cost_per_unit=0.06, lb=0.0, ub=0.8, category="ion_major"),
        MultiSimpleComponent("Manganese sulfate (MnSO4)", {"EX_mn2(e)": 1.0, "EX_so4(e)": 1.0},
                              cost_per_unit=0.3, lb=0.0, ub=0.09, category="ion_trace_expensive"),

        # ADDED based on Schoepping et al. 2021 (npj Syst Biol Appl): two
        # OTHER Bifidobacterium GEMs (B. animalis BB-12, B. longum BB-46)
        # both have an absolute, structural requirement for pantethine as
        # a coenzyme-A precursor -- they lack the genes to build CoA from
        # ordinary pantothenate alone. Our medium had no pantethine
        # component at all. NOTE: verify the exact exchange id this
        # specific NCC2705 reconstruction uses (guessed as EX_ptth(e) below
        # -- confirm/correct via debug_zero_growth.py's Stage 4 output)
        # before trusting this to actually do anything.
        
        #SimpleComponent("Pantethine", "ptth[c]", cost_per_unit=0.15, lb=0.0, ub=1.0, category="vitamin_essential"),

        # --- undefined mixtures, each optimized as ONE unit; profiles
        #     deliberately OVERLAP (all three touch the amino acids) --
        #     that's realistic, and now handled correctly via summation. ---
        CompositeComponent(
            "Peptone level", profile(amino_acids, weight=1.0),
            cost_per_unit=0.2, lb=0.0, ub=10.0, category="extract"),
        CompositeComponent(
            "Meat extract level",
            {**profile(amino_acids, weight=0.6), **profile(vitamins, weight=0.3),
             **profile(trace_metals, weight=0.2)},
            cost_per_unit=0.18, lb=0.0, ub=5.0, category="extract"),
        CompositeComponent(
            "Yeast extract level",
            {**profile(amino_acids, weight=0.4), **profile(vitamins, weight=1.0),
             **profile(nucleotides, weight=0.5), **profile(trace_metals, weight=0.3)},
            cost_per_unit=0.3, lb=0.0, ub=19.5, category="extract"),
    ]

    fixed_open = [
        FixedBound("EX_h(e)", 1000.0),
        FixedBound("EX_h2o(e)", 1000.0),
        FixedBound("EX_co2(e)", 1000.0),
        FixedBound("EX_k(e)", 1000.0),
        FixedBound("EX_na1(e)", 1000.0),
        FixedBound("EX_cl(e)", 1000.0),
        # ADDED after debugging zero-growth-everywhere: these were entirely
        # absent from the original medium (not just tightly bounded), and
        # are structural/cofactor requirements analogous to water or simple
        # ions -- not sensible cost-optimization variables. Confirm with
        # GEMEvaluator.find_unlocking_nutrients() that these (and only
        # these) are what your specific model needs before trusting this list.
        FixedBound("EX_ca2(e)", 1000.0),     # calcium -- common structural/biomass requirement
        FixedBound("EX_mqn7(e)", 1000.0),    # menaquinone-7 (vitamin K2) -- electron carrier
        FixedBound("EX_mqn8(e)", 1000.0),    # menaquinone-8 (vitamin K2) -- electron carrier
        FixedBound("EX_dhna(e)", 1000.0),    # 1,4-dihydroxy-2-naphthoate -- menaquinone precursor
    ]
    # B. longum is anaerobic -> keep O2 closed by default. Agar has no
    # metabolite, so it never appears as a component or fixed_open at all.
    closed = ["EX_o2(e)"]

    return CultureMedium(components, fixed_open=fixed_open, closed=closed)
