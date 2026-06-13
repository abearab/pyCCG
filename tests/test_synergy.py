import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
import sys
import types

py50_stub = types.ModuleType("py50")
py50_stub.Calculator = object
py50_stub.PlotCurve = object
py50_stub.CBMARKERS = []
py50_stub.CBPALETTE = []
sys.modules.setdefault("py50", py50_stub)

from pyctg.synergy import CTG_synergy
import pyctg.synergy as synergy_module


def _sample_ctg_synergy():
    df = pd.DataFrame(
        {
            "cell_type": ["NUGC-3"] * 4,
            "replicate": [1, 1, 1, 1],
            "drugA": [0.0, 1.0, 0.0, 1.0],
            "drugB": [0.0, 0.0, 1.0, 1.0],
            "viability": [1.0, 0.9, 0.85, 0.7],
        }
    )
    return CTG_synergy(df=df, wide_treatment="drugA", narrow_treatment="drugB")


def test_plot_synergy_heatmap_accepts_single_element_ndarray_axis(monkeypatch):
    ctg = _sample_ctg_synergy()
    fig, ax = plt.subplots()
    ax_array = np.array([ax], dtype=object)
    captured = {}

    def fake_plot_heatmap(*_, **kwargs):
        captured["ax"] = kwargs["ax"]

    monkeypatch.setattr(synergy_module, "plot_heatmap", fake_plot_heatmap)
    ctg.plot_synergy_heatmap(query='cell_type=="NUGC-3"', ax=ax_array, colorbar=True)
    plt.close(fig)

    assert captured["ax"] is ax


def test_plot_synergy_heatmap_uses_first_axis_for_multi_axes_ndarray(monkeypatch):
    ctg = _sample_ctg_synergy()
    fig, axes = plt.subplots(2, 1)
    captured = {}

    def fake_plot_heatmap(*_, **kwargs):
        captured["ax"] = kwargs["ax"]

    monkeypatch.setattr(synergy_module, "plot_heatmap", fake_plot_heatmap)
    ctg.plot_synergy_heatmap(query='cell_type=="NUGC-3"', ax=axes, colorbar=True)
    plt.close(fig)

    assert captured["ax"] is axes.flat[0]


def test_plot_synergy_heatmap_rejects_empty_ndarray_axes():
    ctg = _sample_ctg_synergy()

    with pytest.raises(ValueError, match="ax cannot be an empty ndarray"):
        ctg.plot_synergy_heatmap(
            query='cell_type=="NUGC-3"',
            ax=np.array([], dtype=object),
            colorbar=True,
        )
