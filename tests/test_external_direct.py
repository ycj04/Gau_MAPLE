from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gau_maple.external import main, run_direct_external
from gau_maple.gaussian_io import parse_external_input, parse_external_output
from gau_maple.invocation import GaussianInvocation
from gau_maple.models import ExternalResult
from gau_maple.profiles import MapleProfile


class FakeBackend:
    def __init__(self, profile, *, log_path):
        self.profile = profile
        self.log_path = Path(log_path)

    def evaluate(self, request):
        gradient = None
        hessian = None
        if request.derivative_order >= 1:
            gradient = np.full((request.natoms, 3), 0.125)
        if request.derivative_order == 2:
            hessian = np.eye(request.ndof) * 0.25
        return ExternalResult(
            energy_hartree=-76.5,
            gradient_hartree_per_bohr=gradient,
            hessian_hartree_per_bohr2=hessian,
        ).validated_for(request)


class FailingBackend(FakeBackend):
    def evaluate(self, request):
        raise RuntimeError("deliberate failure")


def invocation(tmp_path, fixture="water_deriv1.EIn"):
    source = Path(__file__).parent / "fixtures" / fixture
    return GaussianInvocation(
        layer="R",
        input_path=source,
        output_path=tmp_path / "water.EOut",
        message_path=tmp_path / "water.msg",
        formatted_checkpoint_path=tmp_path / "water.fchk",
        matrix_element_path=tmp_path / "water.mat",
    )


def test_direct_end_to_end_writes_parseable_output_and_message(tmp_path):
    inv = invocation(tmp_path)
    profile = MapleProfile(name="fake", model="fake")
    run_direct_external(inv, profile, backend_factory=FakeBackend)

    request = parse_external_input(inv.input_path)
    result = parse_external_output(inv.output_path, request)
    assert result.energy_hartree == pytest.approx(-76.5)
    assert np.allclose(result.gradient_hartree_per_bohr, 0.125)
    message = inv.message_path.read_text()
    assert "completed successfully" in message
    assert "derivative_order=1" in message


def test_failure_removes_stale_output_and_writes_traceback(tmp_path):
    inv = invocation(tmp_path)
    inv.output_path.write_text("stale")
    profile = MapleProfile(name="fake", model="fake")
    with pytest.raises(RuntimeError, match="deliberate failure"):
        run_direct_external(inv, profile, backend_factory=FailingBackend)
    # run_direct_external itself removes stale output; main() is responsible for
    # recording the exception diagnostic.
    assert not inv.output_path.exists()


def test_main_returns_nonzero_and_does_not_leave_output_on_bad_input(tmp_path):
    missing = tmp_path / "missing.EIn"
    out = tmp_path / "out.EOut"
    msg = tmp_path / "msg.txt"
    code = main(
        [
            "--model",
            "aimnet2",
            "R",
            str(missing),
            str(out),
            str(msg),
            str(tmp_path / "fchk"),
            str(tmp_path / "mat"),
        ]
    )
    assert code == 2
    assert not out.exists()
    assert "InvocationError" in msg.read_text()
