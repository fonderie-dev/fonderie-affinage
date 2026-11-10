from importlib.metadata import version

import fonderie_affinage


def test_version_matches_installed_metadata() -> None:
    assert fonderie_affinage.__version__ == version("fonderie-affinage")
