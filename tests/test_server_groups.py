from gau_maple.server_groups import get_server_group


def test_builtin_server_groups_have_expected_runtime_split():
    maple_name, maple_profiles = get_server_group("maple")
    meta_name, meta_profiles = get_server_group("meta")
    assert maple_name == "maple_server"
    assert meta_name == "meta_server"
    assert "aimnet2" in maple_profiles
    assert "maceomol_native" in maple_profiles
    assert "uma-s-1p2" not in maple_profiles
    assert set(meta_profiles) == {"uma-s-1p2", "esen-sm-conserving-all"}
    assert meta_profiles["esen-sm-conserving-all"].model_options["model_path"].endswith(
        "esen_sm_conserving_all.pt"
    )
