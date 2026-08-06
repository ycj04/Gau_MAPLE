"""MAPLE calculator adapter for Gaussian External requests.

This module is the only Block-2 layer that imports MAPLE or ASE, and those
imports are lazy.  Consequently the Gaussian protocol tests remain runnable in
an environment where MAPLE is absent.

MAPLE's public calculator contract used here is:

* ``SetCalculator(...).set_calculator()`` returns an ASE-compatible calculator.
* Final ``results['energy']`` is Hartree.
* Final ``results['forces']`` is Hartree/Angstrom.
* ``get_hessian(atoms)`` returns Hartree/Angstrom^2 with shape ``(3N, 3N)``.

Gau_MAPLE converts only the coordinate derivatives required by Gaussian.  It
must not apply an additional eV-to-Hartree conversion because MAPLE has already
performed that conversion in its calculator layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import (
    BackendCapabilityError,
    BackendExecutionError,
    MapleUnavailableError,
)
from .models import ExternalRequest, ExternalResult
from .profiles import MapleProfile
HESSIAN_SYMMETRY_ATOL = 1.0e-6
HESSIAN_SYMMETRY_RTOL = 1.0e-6


from .units import (
    maple_forces_to_gaussian_gradient,
    maple_hessian_to_gaussian,
    positions_bohr_to_angstrom,
)


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    calculator_class: str
    model_names: tuple[str, ...]
    supports_charge_multiplicity: bool
    supports_pbc: bool
    supported_hessian_modes: tuple[str, ...]
    implemented_properties: tuple[str, ...]


class MapleBackend:
    """One lazily constructed and reusable MAPLE calculator instance.

    The object is deliberately *not* thread-safe.  The persistent server puts
    one lock around each cached backend because ASE calculators mutate
    ``atoms`` and ``results`` during evaluation.
    """

    def __init__(
        self,
        profile: MapleProfile,
        *,
        log_path: str | Path,
        calculator_factory: type | None = None,
        atoms_factory: Callable[..., Any] | None = None,
        all_changes: Any | None = None,
    ) -> None:
        self.profile = profile
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._calculator_factory = calculator_factory
        self._atoms_factory = atoms_factory
        self._all_changes = all_changes
        self._calculator: Any | None = None
        self._capabilities: BackendCapabilities | None = None

    @staticmethod
    def _import_set_calculator() -> type:
        try:
            from maple.function.calculator.set_calculator import SetCalculator
        except Exception as exc:  # import errors can originate in optional runtimes
            raise MapleUnavailableError(
                "Could not import MAPLE SetCalculator. Activate a MAPLE-capable "
                "Python environment and confirm that 'import maple' resolves to "
                "the intended MAPLE installation."
            ) from exc
        return SetCalculator

    def _import_ase_helpers(self) -> tuple[Callable[..., Any], Any]:
        try:
            from ase import Atoms
            from ase.calculators.calculator import all_changes
        except Exception as exc:
            raise MapleUnavailableError(
                "Could not import ASE from the active MAPLE environment."
            ) from exc
        return Atoms, all_changes

    def _make_atoms(self, request: ExternalRequest) -> Any:
        atoms_factory = self._atoms_factory
        if atoms_factory is None:
            atoms_factory, imported_all_changes = self._import_ase_helpers()
            self._atoms_factory = atoms_factory
            if self._all_changes is None:
                self._all_changes = imported_all_changes

        positions = positions_bohr_to_angstrom(request.positions_bohr)
        try:
            atoms = atoms_factory(
                numbers=request.atomic_numbers.tolist(),
                positions=positions,
                pbc=False,
            )
        except TypeError:
            # Small injected fakes used by unit tests may omit the pbc keyword.
            atoms = atoms_factory(
                numbers=request.atomic_numbers.tolist(),
                positions=positions,
            )
        if not hasattr(atoms, "info"):
            raise BackendExecutionError("Atoms object does not expose an 'info' mapping.")
        atoms.info["charge"] = int(request.charge)
        atoms.info["mult"] = int(request.multiplicity)
        return atoms

    @staticmethod
    @contextmanager
    def _isolated_plugin_environment():
        """Prevent shell activation hooks from injecting unrelated calculators.

        Gau_MAPLE profiles load required plugins explicitly through
        ``model_options.module``.  Clearing ``MAPLE_CALCULATOR_PLUGINS`` only
        while MAPLE constructs the calculator prevents an environment such as
        ``uma_env`` from trying to import ``maple_mace_native``.  The caller's
        shell value is restored immediately afterwards.
        """
        sentinel = object()
        previous = os.environ.get("MAPLE_CALCULATOR_PLUGINS", sentinel)
        os.environ["MAPLE_CALCULATOR_PLUGINS"] = ""
        try:
            yield
        finally:
            if previous is sentinel:
                os.environ.pop("MAPLE_CALCULATOR_PLUGINS", None)
            else:
                os.environ["MAPLE_CALCULATOR_PLUGINS"] = str(previous)

    def _build_calculator(self, atoms: Any) -> Any:
        factory_cls = self._calculator_factory or self._import_set_calculator()
        kwargs = self.profile.factory_kwargs()
        kwargs.update(output=str(self.log_path), atoms=atoms)
        try:
            with self._isolated_plugin_environment():
                calculator = factory_cls(**kwargs).set_calculator()
        except Exception as exc:
            raise BackendExecutionError(
                f"Failed to initialize MAPLE profile '{self.profile.name}' "
                f"(model='{self.profile.model}', device='{self.profile.device}'): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._capabilities = self._inspect_capabilities(calculator)
        return calculator

    @staticmethod
    def _inspect_capabilities(calculator: Any) -> BackendCapabilities:
        cls = type(calculator)
        return BackendCapabilities(
            calculator_class=f"{cls.__module__}.{cls.__name__}",
            model_names=tuple(str(x) for x in getattr(cls, "MODEL_NAMES", ())),
            supports_charge_multiplicity=bool(
                getattr(cls, "SUPPORTS_CHARGE_MULT", False)
            ),
            supports_pbc=bool(getattr(cls, "SUPPORTS_PBC", False)),
            supported_hessian_modes=tuple(
                str(x) for x in getattr(cls, "SUPPORTED_HESSIAN_MODES", ())
            ),
            implemented_properties=tuple(
                str(x) for x in getattr(calculator, "implemented_properties", ())
            ),
        )

    def _ensure_calculator(self, atoms: Any) -> Any:
        if self._calculator is None:
            self._calculator = self._build_calculator(atoms)
        return self._calculator

    @property
    def capabilities(self) -> BackendCapabilities | None:
        """Capabilities become available after the first calculator build."""
        return self._capabilities

    def _validate_request_capabilities(
        self,
        request: ExternalRequest,
        calculator: Any,
    ) -> None:
        capabilities = self._capabilities or self._inspect_capabilities(calculator)
        self._capabilities = capabilities

        if self.profile.reject_mm_charges and not np.allclose(
            request.mm_charges,
            0.0,
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise BackendCapabilityError(
                "Gaussian supplied non-zero per-atom MM charges, but Gau_MAPLE "
                "Gau_MAPLE does not implement an electrostatic-embedding/ONIOM "
                "path. Refusing to ignore them silently."
            )

        non_default_charge = request.charge != 0
        non_default_multiplicity = request.multiplicity != 1

        if non_default_charge and self.profile.charge_policy == "neutral_only":
            raise BackendCapabilityError(
                f"Profile {self.profile.name!r} has charge_policy='neutral_only'; "
                f"refusing request with charge={request.charge}."
            )

        if (
            non_default_multiplicity
            and self.profile.multiplicity_policy == "singlet_only"
        ):
            raise BackendCapabilityError(
                f"Profile {self.profile.name!r} has "
                "multiplicity_policy='singlet_only'; refusing request with "
                f"multiplicity={request.multiplicity}."
            )

        requires_calculator_state_support = (
            non_default_charge
            and self.profile.charge_policy in {"calculator", "supported"}
        ) or (
            non_default_multiplicity
            and self.profile.multiplicity_policy in {"calculator", "supported"}
        )
        if (
            self.profile.strict_charge_multiplicity
            and requires_calculator_state_support
            and not capabilities.supports_charge_multiplicity
        ):
            raise BackendCapabilityError(
                f"Calculator {capabilities.calculator_class} declares "
                "SUPPORTS_CHARGE_MULT=False, but profile "
                f"{self.profile.name!r} permits this request with "
                f"charge={request.charge}, multiplicity={request.multiplicity}."
            )

        if request.derivative_order >= 1 and "forces" not in capabilities.implemented_properties:
            raise BackendCapabilityError(
                f"Calculator {capabilities.calculator_class} does not advertise forces."
            )

        if request.derivative_order == 2:
            if not hasattr(calculator, "get_hessian"):
                raise BackendCapabilityError(
                    f"Calculator {capabilities.calculator_class} has no get_hessian()."
                )
            requested_mode = self.profile.model_options.get("hessian")
            if requested_mode is not None and capabilities.supported_hessian_modes:
                mode = str(requested_mode).strip().lower()
                if mode not in capabilities.supported_hessian_modes:
                    raise BackendCapabilityError(
                        f"Requested hessian='{mode}' but "
                        f"{capabilities.calculator_class} supports only "
                        f"{capabilities.supported_hessian_modes}."
                    )

    @staticmethod
    def _finite_scalar(value: Any, *, name: str) -> float:
        try:
            result = float(np.asarray(value).reshape(()))
        except Exception as exc:
            raise BackendExecutionError(f"{name} is not a scalar: {value!r}") from exc
        if not np.isfinite(result):
            raise BackendExecutionError(f"{name} is NaN or infinity.")
        return result

    @staticmethod
    def _finite_array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape:
            raise BackendExecutionError(
                f"{name} must have shape {shape}, got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise BackendExecutionError(f"{name} contains NaN or infinity.")
        return array

    def evaluate(self, request: ExternalRequest) -> ExternalResult:
        """Evaluate one Gaussian request through a cached MAPLE calculator."""
        atoms = self._make_atoms(request)
        calculator = self._ensure_calculator(atoms)
        self._validate_request_capabilities(request, calculator)

        properties = ["energy"]
        if request.derivative_order >= 1:
            properties.append("forces")

        try:
            calculator.calculate(
                atoms,
                properties=properties,
                system_changes=self._all_changes,
            )
        except Exception as exc:
            raise BackendExecutionError(
                f"MAPLE evaluation failed for profile '{self.profile.name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        results = getattr(calculator, "results", None)
        if not isinstance(results, dict):
            raise BackendExecutionError("Calculator did not expose a results dictionary.")
        if "energy" not in results:
            raise BackendExecutionError("Calculator results are missing 'energy'.")

        energy = self._finite_scalar(results["energy"], name="MAPLE energy")
        gradient = None
        gaussian_hessian = None

        if request.derivative_order >= 1:
            if "forces" not in results:
                raise BackendExecutionError("Calculator results are missing 'forces'.")
            forces = self._finite_array(
                results["forces"],
                name="MAPLE forces",
                shape=(request.natoms, 3),
            )
            gradient = maple_forces_to_gaussian_gradient(forces)

        if request.derivative_order == 2:
            try:
                maple_hessian = calculator.get_hessian(atoms)
            except Exception as exc:
                raise BackendExecutionError(
                    f"MAPLE Hessian evaluation failed for profile "
                    f"'{self.profile.name}': {type(exc).__name__}: {exc}"
                ) from exc
            maple_hessian = self._finite_array(
                maple_hessian,
                name="MAPLE Hessian",
                shape=(request.ndof, request.ndof),
            )
            # Autograd backends often evaluate in float32.  Their raw Hessian
            # can therefore differ from its transpose at the ~1e-7 level even
            # though the exact second derivative is symmetric.  Treat only a
            # materially asymmetric matrix as an error, then explicitly
            # symmetrize before passing it to Gaussian.
            asymmetry = float(np.max(np.abs(maple_hessian - maple_hessian.T)))
            hessian_scale = float(np.max(np.abs(maple_hessian)))
            symmetry_tolerance = (
                HESSIAN_SYMMETRY_ATOL
                + HESSIAN_SYMMETRY_RTOL * hessian_scale
            )
            if asymmetry > symmetry_tolerance:
                raise BackendExecutionError(
                    "MAPLE Hessian is materially non-symmetric; "
                    f"maximum |H-H.T|={asymmetry:.3e}, "
                    f"allowed={symmetry_tolerance:.3e}."
                )
            maple_hessian = 0.5 * (maple_hessian + maple_hessian.T)
            gaussian_hessian = maple_hessian_to_gaussian(maple_hessian)

        return ExternalResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient,
            hessian_hartree_per_bohr2=gaussian_hessian,
        ).validated_for(request)
