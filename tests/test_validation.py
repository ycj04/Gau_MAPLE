import numpy as np

from gau_maple.errors import RemoteServerError
from gau_maple.models import ExternalResult
from gau_maple.validation import (
    PROFILE_ELECTRONIC_STATE_CAPABILITIES,
    run_capability_probes,
    run_stability_probes,
    water_request,
)


def test_water_request_has_gaussian_units_and_gradient_order():
    request = water_request()
    assert request.natoms == 3
    assert request.derivative_order == 1
    assert request.charge == 0
    assert request.multiplicity == 1
    assert np.isclose(request.positions_bohr[1, 0], 0.75716 / 0.529177210903)


def test_capability_matrix_classifies_charge_and_multiplicity_independently():
    def evaluator(_config, profile, request):
        assert request.derivative_order == 0
        supports_charge, supports_multiplicity = (
            PROFILE_ELECTRONIC_STATE_CAPABILITIES[profile]
        )
        reject = (
            request.charge != 0 and not supports_charge
        ) or (
            request.multiplicity != 1 and not supports_multiplicity
        )
        if reject:
            if request.charge != 0 and not supports_charge:
                raise RemoteServerError(
                    "BackendCapabilityError: charge_policy='neutral_only'"
                )
            raise RemoteServerError(
                "BackendCapabilityError: multiplicity_policy='singlet_only'"
            )
        return ExternalResult(energy_hartree=-1.0)

    result = run_capability_probes(object(), evaluator=evaluator)
    assert len(result) == 3 * len(PROFILE_ELECTRONIC_STATE_CAPABILITIES)
    assert all(item.passed for item in result)

    aimnet2 = {
        item.case: item
        for item in result
        if item.profile == "aimnet2"
    }
    assert aimnet2["charge_closed_shell"].expected == "accept"
    assert aimnet2["open_shell_doublet"].expected == "reject"
    assert aimnet2["charged_open_shell"].expected == "reject"


def test_stability_probe_passes_deterministic_backend():
    def evaluator(_config, profile, request):
        energy = -float(request.natoms) - len(profile) * 1.0e-4
        gradient = np.full((request.natoms, 3), 0.001 * request.natoms)
        return ExternalResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient,
        )

    result = run_stability_probes(
        object(), profiles=("aimnet2", "uma-s-1p2"), repeats=3, evaluator=evaluator
    )
    assert len(result) == 2
    assert all(item.passed for item in result)
    assert all(item.atom_count_roundtrip_passed for item in result)


def test_stability_probe_detects_state_drift():
    calls = {"count": 0}

    def evaluator(_config, _profile, request):
        calls["count"] += 1
        return ExternalResult(
            energy_hartree=-1.0 + calls["count"] * 1.0e-5,
            gradient_hartree_per_bohr=np.zeros((request.natoms, 3)),
        )

    result = run_stability_probes(
        object(), profiles=("aimnet2",), repeats=3, evaluator=evaluator
    )
    assert len(result) == 1
    assert not result[0].passed
    assert result[0].energy_span_hartree > 1.0e-9


def test_interleave_probe_detects_cross_profile_stability():
    from types import SimpleNamespace
    from gau_maple.validation import run_interleave_probes

    class Config:
        def server_for_profile(self, profile):
            return SimpleNamespace(name="maple_server" if profile.startswith("a") else "meta_server")

    def evaluator(_config, profile, request):
        energy = -1.0 - len(profile) * 1.0e-3
        return ExternalResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=np.full((request.natoms, 3), len(profile) * 1.0e-4),
        )

    probes = run_interleave_probes(
        Config(), profiles=("aimnet2", "aimnet2nse", "uma-s-1p2"), cycles=3,
        evaluator=evaluator,
    )
    assert len(probes) == 2
    assert all(item.passed for item in probes)


def test_meta_profiles_use_relaxed_default_stability_tolerance():
    calls = {"uma-s-1p2": 0, "esen-sm-conserving-all": 0}

    def evaluator(_config, profile, request):
        calls[profile] += 1
        jitter = (calls[profile] % 2) * 2.0e-8
        return ExternalResult(
            energy_hartree=-float(request.natoms) + jitter,
            gradient_hartree_per_bohr=np.full((request.natoms, 3), jitter),
        )

    result = run_stability_probes(
        object(),
        profiles=("uma-s-1p2", "esen-sm-conserving-all"),
        repeats=3,
        evaluator=evaluator,
    )
    assert all(item.passed for item in result)
    assert all(item.energy_atol_hartree == 1.0e-7 for item in result)
    assert all(item.gradient_atol_hartree_per_bohr == 1.0e-7 for item in result)
    assert all(type(item.atom_count_roundtrip_passed) is bool for item in result)


def test_validation_json_serializes_numpy_scalars(tmp_path):
    import json
    from gau_maple.validation import _write_json

    path = tmp_path / "report.json"
    _write_json(path, {"ok": np.bool_(True), "value": np.float32(1.5)})
    payload = json.loads(path.read_text())
    assert payload == {"ok": True, "value": 1.5}
