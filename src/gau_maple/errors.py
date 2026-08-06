"""Project-specific exception hierarchy."""


class GauMapleError(Exception):
    """Base exception for Gau_MAPLE."""


class InvocationError(GauMapleError, ValueError):
    """Raised when Gaussian External process arguments are invalid."""


class ExternalFormatError(GauMapleError, ValueError):
    """Raised when Gaussian External input/output is malformed."""


class UnitConversionError(GauMapleError, ValueError):
    """Raised when an array cannot be converted safely."""


class ProfileError(GauMapleError, ValueError):
    """Raised when a MAPLE backend profile is invalid."""


class ConfigError(GauMapleError, ValueError):
    """Raised when Gau_MAPLE TOML configuration is invalid."""


class MapleUnavailableError(GauMapleError, ImportError):
    """Raised when the active Python environment cannot import MAPLE/ASE."""


class BackendCapabilityError(GauMapleError, RuntimeError):
    """Raised when a selected calculator cannot satisfy a request safely."""


class BackendExecutionError(GauMapleError, RuntimeError):
    """Raised when MAPLE calculator construction or evaluation fails."""


class ProtocolError(GauMapleError, ValueError):
    """Raised when the local server/client protocol is malformed."""


class ServerConnectionError(GauMapleError, ConnectionError):
    """Raised when the local Unix socket server cannot be reached."""


class RemoteServerError(GauMapleError, RuntimeError):
    """Raised when a Gau_MAPLE server reports an evaluation failure."""


class ServerStartupError(GauMapleError, RuntimeError):
    """Raised when a persistent Gau_MAPLE server cannot start safely."""
