from __future__ import annotations

import socket
from pathlib import Path

import numpy as np

from gau_maple.gaussian_io import parse_external_input
from gau_maple.protocol import (
    encode_message,
    make_evaluate_message,
    receive_message,
    request_from_payload,
    request_to_payload,
    result_from_payload,
    result_to_payload,
    send_message,
)
from gau_maple.models import ExternalResult


def fixture(name: str):
    return parse_external_input(Path(__file__).parent / "fixtures" / name)


def test_request_and_result_payload_roundtrip():
    request = fixture("water_deriv2.EIn")
    restored = request_from_payload(request_to_payload(request))
    assert np.array_equal(restored.atomic_numbers, request.atomic_numbers)
    assert np.allclose(restored.positions_bohr, request.positions_bohr)
    assert restored.derivative_order == 2

    result = ExternalResult(
        energy_hartree=-76.5,
        gradient_hartree_per_bohr=np.ones((3, 3)) * 0.1,
        hessian_hartree_per_bohr2=np.eye(9) * 0.2,
    ).validated_for(request)
    restored_result = result_from_payload(result_to_payload(result), request)
    assert restored_result.energy_hartree == -76.5
    assert np.allclose(restored_result.gradient_hartree_per_bohr, 0.1)
    assert np.allclose(restored_result.hessian_hartree_per_bohr2, np.eye(9) * 0.2)


def test_framed_message_roundtrip_over_socketpair():
    request = fixture("water_deriv1.EIn")
    message = make_evaluate_message(request, "aimnet2")
    left, right = socket.socketpair()
    try:
        send_message(left, message)
        received = receive_message(right)
    finally:
        left.close()
        right.close()
    assert received == message
    assert received["profile"] == "aimnet2"
    assert len(encode_message(message)) > 12
