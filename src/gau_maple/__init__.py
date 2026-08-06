"""Public API for Gau_MAPLE."""

from .client import ServerMetadata, evaluate_via_server, ping_server
from .config import GauMapleConfig, ServerDefinition, load_config
from .frequency import (
    FrequencyAnalysis,
    HessianComparison,
    compare_hessians,
    finite_difference_hessian,
    harmonic_frequency_analysis,
)
from .gaussian_io import (
    parse_external_input,
    parse_external_output,
    write_external_output,
)
from .invocation import GaussianInvocation, parse_gaussian_invocation
from .maple_backend import BackendCapabilities, MapleBackend
from .models import ExternalRequest, ExternalResult
from .profiles import MapleProfile

__all__ = [
    "BackendCapabilities",
    "ExternalRequest",
    "ExternalResult",
    "FrequencyAnalysis",
    "GaussianInvocation",
    "HessianComparison",
    "GauMapleConfig",
    "MapleBackend",
    "MapleProfile",
    "ServerDefinition",
    "ServerMetadata",
    "compare_hessians",
    "evaluate_via_server",
    "finite_difference_hessian",
    "harmonic_frequency_analysis",
    "load_config",
    "parse_external_input",
    "parse_external_output",
    "parse_gaussian_invocation",
    "ping_server",
    "write_external_output",
]

__version__ = "0.10.0"
