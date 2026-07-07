from .synergy import (
    read_synergy_data,
    SynergyData,
)

from .titration import (
    read_titration_data,
    plot_titration,
)

def _get_version():
    import os
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib

    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")

    with open(pyproject_path, "rb") as pyproject_file:
        return tomllib.load(pyproject_file)["project"]["version"]


try:
    __version__ = _get_version()
except Exception:
    __version__ = "Unknown"
