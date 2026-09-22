"""Independent query-aware adaptive visual search."""

from .frontend import FrontendResult, materialize_frontend
from .binary import (
    NegativeCoverage,
    NegativeGateDecision,
    StrictNoConfig,
    evaluate_negative_gate,
)
from .config import (
    BranchAcceptanceConfig,
    IndependentSearchConfig,
    ObservationGeometryConfig,
    VerificationConfig,
)
from .decision import (
    GlobalCost,
    cumulative_phase_records,
    evaluate_global_gate,
    select_accepted_hypothesis,
    source_image_identity,
)
from .method import run_independent_sample
from .grounding import (
    GroundingLabelDistribution,
    GroundingRecord,
    GroundingVerification,
    RoleGrounding,
    TargetInstance,
    TargetInstanceRegistry,
    compatible_confirmation,
    verify_target_grounding,
)
from .semantics import (
    LabelDistribution,
    OptionCatalog,
    OptionEntry,
    OptionSupport,
    OptionSupportVector,
    build_option_catalog,
    verify_option_support,
)

__all__ = [
    "BranchAcceptanceConfig",
    "FrontendResult",
    "GroundingLabelDistribution",
    "GroundingRecord",
    "GroundingVerification",
    "GlobalCost",
    "IndependentSearchConfig",
    "LabelDistribution",
    "NegativeGateDecision",
    "NegativeCoverage",
    "ObservationGeometryConfig",
    "OptionCatalog",
    "OptionEntry",
    "OptionSupport",
    "OptionSupportVector",
    "RoleGrounding",
    "StrictNoConfig",
    "TargetInstance",
    "TargetInstanceRegistry",
    "VerificationConfig",
    "build_option_catalog",
    "compatible_confirmation",
    "cumulative_phase_records",
    "materialize_frontend",
    "evaluate_global_gate",
    "evaluate_negative_gate",
    "run_independent_sample",
    "select_accepted_hypothesis",
    "source_image_identity",
    "verify_option_support",
    "verify_target_grounding",
]
