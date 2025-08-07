"""Basic tests for ts_profiler package."""



def test_package_import():
    """Test that the package can be imported."""
    import ts_profiler
    assert ts_profiler.__version__ == "0.1.0"


def test_submodules_import():
    """Test that submodules can be imported."""
    from ts_profiler import config, features, utils
    assert config is not None
    assert features is not None
    assert utils is not None
