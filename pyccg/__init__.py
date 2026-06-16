from .synergy import (
    read_CCG_synergy_data,
    CCG_synergy,
    read_CTG_synergy_data,
    CTG_synergy,
)

from .titration import (
    read_CCG_titration_data,
    plot_CCG_titration,
    read_CTG_titration_data, 
    plot_CTG_titration,
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
