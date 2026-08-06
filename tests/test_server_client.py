from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from gau_maple.client import evaluate_via_server, ping_server
from gau_maple.errors import RemoteServerError
from gau_maple.gaussian_io import parse_external_input
from gau_maple.models import ExternalResult
from gau_maple.profiles import MapleProfile
from gau_maple.server import GauMapleUnixServer


class FakeBackend:
    instances = 0

    def __init__(self, profile, *, log_path):
        type(self).instances += 1
        self.profile = profile
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
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
        raise RuntimeError("deliberate remote failure")


def start_server(tmp_path, backend_factory=FakeBackend):
    socket_path = tmp_path / "server.sock"
    profile = MapleProfile(name="fake-profile", model="fake", device="cpu")
    server = GauMapleUnixServer(
        socket_path,
        profile,
        log_path=tmp_path / "server.log",
        backend_factory=backend_factory,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, socket_path


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_ping_and_repeated_evaluation_reuse_one_backend(tmp_path):
    FakeBackend.instances = 0
    server, thread, socket_path = start_server(tmp_path)
    try:
        metadata = ping_server(socket_path, expect_profile="fake-profile")
        assert metadata.model == "fake"
        assert metadata.request_count == 0

        request = parse_external_input(Path(__file__).parent / "fixtures" / "water_deriv1.EIn")
        result1, meta1 = evaluate_via_server(request, socket_path)
        result2, meta2 = evaluate_via_server(request, socket_path)
        assert result1.energy_hartree == pytest.approx(-76.5)
        assert np.allclose(result2.gradient_hartree_per_bohr, 0.125)
        assert meta1.request_count == 1
        assert meta2.request_count == 2
        assert FakeBackend.instances == 1
    finally:
        stop_server(server, thread)
    assert not socket_path.exists()


def test_profile_mismatch_is_rejected(tmp_path):
    server, thread, socket_path = start_server(tmp_path)
    try:
        with pytest.raises(RemoteServerError, match="Unknown server profile"):
            ping_server(socket_path, expect_profile="wrong-profile")
    finally:
        stop_server(server, thread)


def test_remote_backend_failure_is_returned_with_traceback(tmp_path):
    server, thread, socket_path = start_server(tmp_path, backend_factory=FailingBackend)
    try:
        request = parse_external_input(Path(__file__).parent / "fixtures" / "water_deriv1.EIn")
        with pytest.raises(RemoteServerError, match="deliberate remote failure"):
            evaluate_via_server(request, socket_path)
    finally:
        stop_server(server, thread)


def test_multi_profile_server_routes_and_counts_independently(tmp_path):
    from gau_maple.server import GauMapleMultiProfileServer

    class RoutedBackend(FakeBackend):
        def evaluate(self, request):
            energy = -1.0 if self.profile.name == "alpha" else -2.0
            return ExternalResult(energy_hartree=energy).validated_for(request)

    socket_path = tmp_path / "multi.sock"
    profiles = {
        "alpha": MapleProfile(name="alpha", model="fake-a"),
        "beta": MapleProfile(name="beta", model="fake-b"),
    }
    server = GauMapleMultiProfileServer(
        socket_path,
        profiles,
        server_name="test_server",
        log_path=tmp_path / "server.log",
        backend_factory=RoutedBackend,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = parse_external_input(Path(__file__).parent / "fixtures" / "water_deriv0.EIn")
        alpha, alpha_meta = evaluate_via_server(
            request,
            socket_path,
            profile_name="alpha",
            expect_server="test_server",
        )
        beta, beta_meta = evaluate_via_server(
            request,
            socket_path,
            profile_name="beta",
            expect_server="test_server",
        )
        assert alpha.energy_hartree == pytest.approx(-1.0)
        assert beta.energy_hartree == pytest.approx(-2.0)
        assert alpha_meta.profile_name == "alpha"
        assert beta_meta.profile_name == "beta"
        status = ping_server(socket_path, expect_server="test_server")
        assert status.available_profiles == ("alpha", "beta")
        assert status.profile_statuses["alpha"]["request_count"] == 1
        assert status.profile_statuses["beta"]["request_count"] == 1
        assert status.request_count == 2
    finally:
        stop_server(server, thread)


def test_preload_failure_isolated_to_one_profile(tmp_path):
    from gau_maple.server import GauMapleMultiProfileServer

    class SelectiveBackend(FakeBackend):
        def evaluate(self, request):
            if self.profile.name == "bad":
                raise RuntimeError("cannot load bad model")
            return ExternalResult(energy_hartree=-3.0).validated_for(request)

    socket_path = tmp_path / "preload.sock"
    server = GauMapleMultiProfileServer(
        socket_path,
        {
            "good": MapleProfile(name="good", model="good"),
            "bad": MapleProfile(name="bad", model="bad"),
        },
        server_name="preload_server",
        log_path=tmp_path / "server.log",
        backend_factory=SelectiveBackend,
        preload=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        metadata = ping_server(socket_path, expect_server="preload_server")
        assert metadata.profile_statuses["good"]["preload_state"] == "loaded"
        assert metadata.profile_statuses["bad"]["preload_state"] == "failed"
        assert "cannot load bad model" in metadata.profile_statuses["bad"]["preload_error"]
        request = parse_external_input(Path(__file__).parent / "fixtures" / "water_deriv0.EIn")
        result, _ = evaluate_via_server(request, socket_path, profile_name="good")
        assert result.energy_hartree == pytest.approx(-3.0)
    finally:
        stop_server(server, thread)
