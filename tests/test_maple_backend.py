from __future__ import annotations

import numpy as np
import pytest

from gau_maple.errors import BackendCapabilityError, BackendExecutionError
from gau_maple.maple_backend import MapleBackend
from gau_maple.models import ExternalRequest
from gau_maple.profiles import MapleProfile
from gau_maple.units import BOHR_TO_ANGSTROM


class FakeAtoms:
    def __init__(self, *, numbers, positions, pbc=False):
        self.numbers = np.asarray(numbers, dtype=int)
        self.positions = np.asarray(positions, dtype=float)
        self.pbc = pbc
        self.info = {}


class GoodCalculator:
    MODEL_NAMES = ("fake",)
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    implemented_properties = ["energy", "forces", "free_energy", "hessian"]

    def __init__(self):
        self.results = {}
        self.calls = 0

    def calculate(self, atoms, properties, system_changes):
        self.calls += 1
        self.results = {"energy": -1.25}
        if "forces" in properties:
            self.results["forces"] = np.full((len(atoms.numbers), 3), 2.0)

    def get_hessian(self, atoms):
        return np.eye(3 * len(atoms.numbers)) * 4.0


class SlightlyAsymmetricCalculator(GoodCalculator):
    def get_hessian(self, atoms):
        hessian = super().get_hessian(atoms)
        hessian[0, 1] += 1.2e-7
        return hessian


class MateriallyAsymmetricCalculator(GoodCalculator):
    def get_hessian(self, atoms):
        hessian = super().get_hessian(atoms)
        hessian[0, 1] += 1.0e-3
        return hessian


class NoChargeCalculator(GoodCalculator):
    SUPPORTS_CHARGE_MULT = False


class BadForceCalculator(GoodCalculator):
    def calculate(self, atoms, properties, system_changes):
        self.results = {"energy": -1.0, "forces": np.zeros((1, 3))}


class FakeSetCalculator:
    created = []
    calculator_type = GoodCalculator

    def __init__(self, **kwargs):
        type(self).created.append(kwargs)

    def set_calculator(self):
        return type(self).calculator_type()


def request(order=1, charge=0, mult=1, mm=None):
    if mm is None:
        mm = np.zeros(2)
    return ExternalRequest(
        atomic_numbers=np.array([1, 1]),
        positions_bohr=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        derivative_order=order,
        charge=charge,
        multiplicity=mult,
        mm_charges=mm,
    )


def backend(tmp_path, *, calculator_type=GoodCalculator, **profile_kwargs):
    FakeSetCalculator.created = []
    FakeSetCalculator.calculator_type = calculator_type
    profile = MapleProfile(name="fake-profile", model="fake", **profile_kwargs)
    return MapleBackend(
        profile,
        log_path=tmp_path / "backend.log",
        calculator_factory=FakeSetCalculator,
        atoms_factory=FakeAtoms,
        all_changes=("all",),
    )


def test_order1_builds_atoms_passes_factory_options_and_converts_force(tmp_path):
    adapter = backend(
        tmp_path,
        model_options={"hessian": "analytic", "module": "plugin.mod"},
    )
    result = adapter.evaluate(request(order=1))

    assert result.energy_hartree == pytest.approx(-1.25)
    assert np.allclose(result.gradient_hartree_per_bohr, -2.0 * BOHR_TO_ANGSTROM)
    assert len(FakeSetCalculator.created) == 1
    kwargs = FakeSetCalculator.created[0]
    assert kwargs["model"] == "fake"
    assert kwargs["model_options"]["module"] == "plugin.mod"
    assert kwargs["atoms"].info == {"charge": 0, "mult": 1}
    assert np.allclose(kwargs["atoms"].positions[1, 0], 1.4 * BOHR_TO_ANGSTROM)


def test_calculator_is_cached_across_requests(tmp_path):
    adapter = backend(tmp_path)
    adapter.evaluate(request(order=0))
    adapter.evaluate(request(order=1))
    assert len(FakeSetCalculator.created) == 1


def test_order2_converts_hessian(tmp_path):
    adapter = backend(tmp_path, model_options={"hessian": "analytic"})
    result = adapter.evaluate(request(order=2))
    assert result.hessian_hartree_per_bohr2.shape == (6, 6)
    assert np.allclose(
        result.hessian_hartree_per_bohr2,
        np.eye(6) * 4.0 * BOHR_TO_ANGSTROM**2,
    )


def test_rejects_unsupported_charge_multiplicity(tmp_path):
    adapter = backend(tmp_path, calculator_type=NoChargeCalculator)
    with pytest.raises(BackendCapabilityError, match="SUPPORTS_CHARGE_MULT=False"):
        adapter.evaluate(request(order=1, charge=-1, mult=2))


def test_can_explicitly_relax_charge_policy(tmp_path):
    adapter = backend(
        tmp_path,
        calculator_type=NoChargeCalculator,
        strict_charge_multiplicity=False,
    )
    result = adapter.evaluate(request(order=0, charge=-1, mult=2))
    assert result.energy_hartree == pytest.approx(-1.25)


def test_rejects_nonzero_mm_charges(tmp_path):
    adapter = backend(tmp_path)
    with pytest.raises(BackendCapabilityError, match="non-zero per-atom MM charges"):
        adapter.evaluate(request(order=1, mm=np.array([0.1, 0.0])))


def test_rejects_bad_force_shape(tmp_path):
    adapter = backend(tmp_path, calculator_type=BadForceCalculator)
    with pytest.raises(BackendExecutionError, match="must have shape"):
        adapter.evaluate(request(order=1))


def test_small_float32_hessian_asymmetry_is_symmetrized(tmp_path):
    adapter = backend(
        tmp_path,
        calculator_type=SlightlyAsymmetricCalculator,
        model_options={"hessian": "analytic"},
    )
    result = adapter.evaluate(request(order=2))
    hessian = result.hessian_hartree_per_bohr2
    assert np.array_equal(hessian, hessian.T)


def test_material_hessian_asymmetry_is_rejected(tmp_path):
    adapter = backend(
        tmp_path,
        calculator_type=MateriallyAsymmetricCalculator,
        model_options={"hessian": "analytic"},
    )
    with pytest.raises(BackendExecutionError, match="materially non-symmetric"):
        adapter.evaluate(request(order=2))


def test_calculator_build_isolates_inherited_plugin_environment(tmp_path, monkeypatch):
    observed = {}

    class RecordingSetCalculator(FakeSetCalculator):
        def __init__(self, **kwargs):
            import os

            observed["during"] = os.environ.get("MAPLE_CALCULATOR_PLUGINS")
            super().__init__(**kwargs)

    monkeypatch.setenv("MAPLE_CALCULATOR_PLUGINS", "maple_mace_native")
    profile = MapleProfile(
        name="fake-profile",
        model="fake",
        model_options={"module": "explicit.plugin"},
    )
    adapter = MapleBackend(
        profile,
        log_path=tmp_path / "backend.log",
        calculator_factory=RecordingSetCalculator,
        atoms_factory=FakeAtoms,
        all_changes=("all",),
    )
    adapter.evaluate(request(order=0))

    assert observed["during"] == ""
    assert __import__("os").environ["MAPLE_CALCULATOR_PLUGINS"] == "maple_mace_native"



def test_profile_can_support_charge_but_reject_open_shell(tmp_path):
    adapter = backend(
        tmp_path,
        charge_policy="supported",
        multiplicity_policy="singlet_only",
    )
    charged = adapter.evaluate(request(order=0, charge=1, mult=1))
    assert charged.energy_hartree == pytest.approx(-1.25)
    with pytest.raises(BackendCapabilityError, match="multiplicity_policy='singlet_only'"):
        adapter.evaluate(request(order=0, charge=0, mult=2))


def test_profile_can_reject_charge_and_accept_multiplicity(tmp_path):
    adapter = backend(
        tmp_path,
        charge_policy="neutral_only",
        multiplicity_policy="supported",
    )
    open_shell = adapter.evaluate(request(order=0, charge=0, mult=2))
    assert open_shell.energy_hartree == pytest.approx(-1.25)
    with pytest.raises(BackendCapabilityError, match="charge_policy='neutral_only'"):
        adapter.evaluate(request(order=0, charge=1, mult=1))


def test_supported_policy_still_checks_calculator_declaration(tmp_path):
    adapter = backend(
        tmp_path,
        calculator_type=NoChargeCalculator,
        charge_policy="supported",
        multiplicity_policy="supported",
    )
    with pytest.raises(BackendCapabilityError, match="SUPPORTS_CHARGE_MULT=False"):
        adapter.evaluate(request(order=0, charge=1, mult=2))
