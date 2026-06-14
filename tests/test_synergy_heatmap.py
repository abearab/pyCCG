import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import importlib.util
from pathlib import Path

_SYNERGY_PATH = Path(__file__).resolve().parents[1] / "pyctg" / "synergy.py"
_SPEC = importlib.util.spec_from_file_location("pyctg_synergy", _SYNERGY_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
CTG_synergy = _MODULE.CTG_synergy


def _example_df():
    return pd.DataFrame(
        {
            "cell_type": ["A"] * 4,
            "replicate": [1] * 4,
            "drug_a": [0.0, 1.0, 0.0, 1.0],
            "drug_b": [0.0, 0.0, 1.0, 1.0],
            "viability": [1.0, 0.8, 0.7, 0.4],
        }
    )


def test_plot_synergy_heatmap_default_labels_and_colorbar():
    fig, ax = plt.subplots()
    CTG_synergy(_example_df(), "drug_a", "drug_b").plot_synergy_heatmap(
        query='cell_type == "A"', ax=ax
    )

    assert len(fig.axes) == 2
    assert ax.get_xlabel() == "\ndrug_a"
    assert ax.get_ylabel() == "drug_b\n"
    plt.close(fig)


def test_plot_synergy_heatmap_without_colorbar():
    fig, ax = plt.subplots()
    CTG_synergy(_example_df(), "drug_a", "drug_b").plot_synergy_heatmap(
        query='cell_type == "A"', ax=ax, colorbar=False
    )

    assert len(fig.axes) == 1
    plt.close(fig)
