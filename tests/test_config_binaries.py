from kon import AVAILABLE_BINARIES, config


def test_available_binaries_is_set():
    assert isinstance(AVAILABLE_BINARIES, set)


def test_available_binaries_contains_valid_entries():
    valid_binaries = {"rg", "fd"}
    assert AVAILABLE_BINARIES.issubset(valid_binaries)


def test_config_binaries_property():
    binaries_config = config.binaries
    assert hasattr(binaries_config, "has")
    assert hasattr(binaries_config, "rg")
    assert hasattr(binaries_config, "fd")


def test_config_binaries_has_method():
    # Test with a known available binary
    if "rg" in AVAILABLE_BINARIES:
        assert config.binaries.has("rg") is True

    # Test with a nonexistent binary
    assert config.binaries.has("nonexistent_binary") is False


def test_config_binaries_properties():
    # The properties should match the AVAILABLE_BINARIES set
    assert config.binaries.rg == ("rg" in AVAILABLE_BINARIES)
    assert config.binaries.fd == ("fd" in AVAILABLE_BINARIES)
