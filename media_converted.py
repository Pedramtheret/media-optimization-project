from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
from yeast_extract_profile_mmol import YEAST_EXTRACT_PROFILE_MMOL_PER_G, MEAT_EXTRACT_PROFILE_MMOL_PER_G, PEPTONE_PROFILE_MMOL_PER_G
from medium_unit_pipeline import GLUCOSE_UB_MMOL_PER_L, PHOSPHATE_UB_MMOL_PER_L, ACETATE_UB_MMOL_PER_L, NA_UB_MMOL_PER_L, SULFATE_UB_MMOL_PER_L
from medium_unit_pipeline import MG_UB_MMOL_PER_L, OVERALSULFATE_UB_MMOL_PER_L, MN_UB_MMOL_PER_L, AMMUNIOM_UB_MMOL_PER_L, MNSULFATE_UB_MMOL_PER_L


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
    yeast_profile.update(_weighted_profile(YEAST_EXTRACT_PROFILE_MMOL_PER_G))          # dominant mass fraction, protein 66.8-76.3%
    
    # ---- meat extract profile: amino acids + nucleotides emphasized, ----
    # ---- vitamins at low/category weight, NO trace metals (undocumented) ----
    meat_profile: Dict[str, float] = {}
    meat_profile.update(_weighted_profile(MEAT_EXTRACT_PROFILE_MMOL_PER_G))        # documented: "peptides, individual amino acids"
    

    components: List[MediaComponent] = [
        SimpleComponent("Glucose", "EX_glc_D(e)", cost_per_unit=0.0148, lb=0.0, ub=GLUCOSE_UB_MMOL_PER_L, category="carbon"),
        SimpleComponent("Phosphate (K2HPO4)", "EX_pi(e)", cost_per_unit=0.28, lb=0.0, ub=PHOSPHATE_UB_MMOL_PER_L, category="ion_major"),
        SimpleComponent("Ammonium (from triammonium citrate)", "EX_nh4(e)", cost_per_unit=0.0950, lb=0.0, ub=AMMUNIOM_UB_MMOL_PER_L, category="ion_major"),

        # Sodium tied to its only real recipe source (sodium acetate), 1:1
        # stoichiometry from CH3COONa -- see build_bifidobacterium_mrs_medium
        # docstring for the background_na_level regimes tested alongside this.
        MultiSimpleComponent("Acetate (sodium acetate)", {"EX_ac(e)": 1.0, "EX_na1(e)": 1.0},
                              cost_per_unit=0.01094, lb=0.0, ub=ACETATE_UB_MMOL_PER_L, category="ion_major"),

        MultiSimpleComponent("Magnesium sulfate (MgSO4)", {"EX_mg2(e)": 1.0, "EX_so4(e)": 1.0},
                              cost_per_unit=0.0137, lb=0.0, ub=MG_UB_MMOL_PER_L, category="ion_major"),
        MultiSimpleComponent("Manganese sulfate (MnSO4)", {"EX_mn2(e)": 1.0, "EX_so4(e)": 1.0},
                              cost_per_unit=0.2301, lb=0.0, ub=MN_UB_MMOL_PER_L, category="ion_trace_expensive"),

        # Peptone: kept as the uniform amino-acid-only profile from v2 --
        # peptone is a defined-process partial protein hydrolysate (not an
        # "extract" in the yeast/meat sense), and vendor peptone specs
        # consistently describe it as free amino acids + short peptides
        # without a distinct vitamin/mineral profile, so no change here.
        CompositeComponent("Peptone level", _uniform_profile(PEPTONE_PROFILE_MMOL_PER_G, weight=1.0),
                            cost_per_unit=0.1640, lb=0.0, ub=10.0, category="extract"),

        CompositeComponent("Meat extract level", meat_profile,
                            cost_per_unit=0.3952, lb=0.0, ub=8.0, category="extract"),

        CompositeComponent("Yeast extract level", yeast_profile,
                            cost_per_unit=0.1315, lb=0.0, ub=20.0, category="extract"),
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

    
    
