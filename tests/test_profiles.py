import pytest

from gau_maple.errors import ProfileError
from gau_maple.profiles import MapleProfile


def test_profile_normalizes_and_copies_options():
    original = {"module": "maple_mace_native", "hessian": "analytic"}
    profile = MapleProfile(
        name=" Native OMol ",
        model="MACEOMOL_NATIVE",
        device="cpu",
        model_options=original,
    )
    original["module"] = "changed"

    assert profile.name == "Native OMol"
    assert profile.model == "maceomol_native"
    assert profile.model_options["module"] == "maple_mace_native"
    assert profile.factory_kwargs()["model"] == "maceomol_native"


def test_profile_rejects_empty_model():
    with pytest.raises(ProfileError, match="model must not be empty"):
        MapleProfile(name="bad", model=" ")
