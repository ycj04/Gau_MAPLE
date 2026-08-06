from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from gau_maple.external import main, run_socket_external
from gau_maple.gaussian_io import parse_external_input, parse_external_output
from gau_maple.invocation import GaussianInvocation
from gau_maple.models import ExternalResult
from gau_maple.profiles import MapleProfile
from gau_maple.server import GauMapleUnixServer


class FakeBackend:
    def __init__(self, profile, *, log_path):
        self.profile = profile

    def evaluate(self, request):
        gradient = None
        if request.derivative_order >= 1:
            gradient = np.full((request.natoms, 3), 0.125)
        return ExternalResult(
            energy_hartree=-76.5,
            gradient_hartree_per_bohr=gradient,
        ).validated_for(request)


def start_server(tmp_path):
    socket_path = tmp_path / "external.sock"
    server = GauMapleUnixServer(
        socket_path,
        MapleProfile(name="fake-profile", model="fake"),
        log_path=tmp_path / "server.log",
        backend_factory=FakeBackend,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, socket_path


def invocation(tmp_path):
    source = Path(__file__).parent / "fixtures" / "water_deriv1.EIn"
    return GaussianInvocation(
        layer="R",
        input_path=source,
        output_path=tmp_path / "water.EOut",
        message_path=tmp_path / "water.msg",
        formatted_checkpoint_path=tmp_path / "water.fchk",
        matrix_element_path=tmp_path / "water.mat",
    )


def test_socket_external_writes_gaussian_output(tmp_path):
    server, thread, socket_path = start_server(tmp_path)
    try:
        inv = invocation(tmp_path)
        run_socket_external(inv, socket_path, expect_profile="fake-profile")
        request = parse_external_input(inv.input_path)
        result = parse_external_output(inv.output_path, request)
        assert result.energy_hartree == -76.5
        assert np.allclose(result.gradient_hartree_per_bohr, 0.125)
        message = inv.message_path.read_text()
        assert "mode=socket" in message
        assert "request_count=1" in message
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_main_socket_mode_end_to_end(tmp_path):
    server, thread, socket_path = start_server(tmp_path)
    try:
        inv = invocation(tmp_path)
        code = main(
            [
                "--socket",
                str(socket_path),
                "--expect-profile",
                "fake-profile",
                "R",
                str(inv.input_path),
                str(inv.output_path),
                str(inv.message_path),
                str(inv.formatted_checkpoint_path),
                str(inv.matrix_element_path),
            ]
        )
        assert code == 0
        assert inv.output_path.is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_main_multi_profile_socket_selection(tmp_path):
    from gau_maple.server import GauMapleMultiProfileServer

    class RoutedBackend(FakeBackend):
        def evaluate(self, request):
            energy = -10.0 if self.profile.name == "alpha" else -20.0
            gradient = None
            if request.derivative_order >= 1:
                gradient = np.zeros((request.natoms, 3))
            return ExternalResult(
                energy_hartree=energy,
                gradient_hartree_per_bohr=gradient,
            ).validated_for(request)

    socket_path = tmp_path / "multi-external.sock"
    server = GauMapleMultiProfileServer(
        socket_path,
        {
            "alpha": MapleProfile(name="alpha", model="a"),
            "beta": MapleProfile(name="beta", model="b"),
        },
        server_name="multi_server",
        log_path=tmp_path / "server.log",
        backend_factory=RoutedBackend,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        inv = invocation(tmp_path)
        code = main(
            [
                "--socket",
                str(socket_path),
                "--profile",
                "beta",
                "--expect-server",
                "multi_server",
                "R",
                str(inv.input_path),
                str(inv.output_path),
                str(inv.message_path),
                str(inv.formatted_checkpoint_path),
                str(inv.matrix_element_path),
            ]
        )
        assert code == 0
        request = parse_external_input(inv.input_path)
        result = parse_external_output(inv.output_path, request)
        assert result.energy_hartree == -20.0
        assert "server='multi_server'" in inv.message_path.read_text()
        assert "profile='beta'" in inv.message_path.read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_main_toml_profile_routing_end_to_end(tmp_path):
    socket_path = tmp_path / "config-routed.sock"
    profile = MapleProfile(name="beta", model="fake")
    server = GauMapleUnixServer(
        socket_path,
        profile,
        log_path=tmp_path / "server.log",
        backend_factory=FakeBackend,
    )
    # The single-profile compatibility server uses profile.name as server_name.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "profiles.toml"
    config.write_text(
        f'''
[project]
runtime_dir = "{tmp_path}"

[profiles.beta]
model = "fake"

[servers.beta]
executable = "/bin/true"
profiles = ["beta"]
socket = "{socket_path}"
pid_file = "{tmp_path}/beta.pid"
log = "{tmp_path}/beta.log"
stdout = "{tmp_path}/beta.stdout"
preload = false
startup_timeout = 5
shutdown_timeout = 2
''',
        encoding="utf-8",
    )
    try:
        inv = invocation(tmp_path)
        code = main(
            [
                "--config",
                str(config),
                "--profile",
                "beta",
                "R",
                str(inv.input_path),
                str(inv.output_path),
                str(inv.message_path),
                str(inv.formatted_checkpoint_path),
                str(inv.matrix_element_path),
            ]
        )
        assert code == 0
        request = parse_external_input(inv.input_path)
        result = parse_external_output(inv.output_path, request)
        assert result.energy_hartree == -76.5
        assert "server='beta'" in inv.message_path.read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
