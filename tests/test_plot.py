"""plot: one-call charting, matplotlib/plotly/seaborn backends."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import newsindex as ni


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _result():
    rows = []
    for country, base in [("Nigeria", 80.0), ("Kenya", 95.0)]:
        for i, month in enumerate(["2024-01", "2024-02", "2024-03", "2024-04"]):
            rows.append(
                {
                    "country": country,
                    "year_month": month,
                    "EPU": base + i * 5.0,
                    "M_t": 1.0 + i * 0.1,
                    "mu": 1.0,
                    "n_sites": 2,
                }
            )
    return pd.DataFrame(rows)


def _regions_result():
    rows = []
    for group in ["West", "East"]:
        for i, month in enumerate(["2024-01", "2024-02", "2024-03"]):
            rows.append(
                {
                    "group": group,
                    "year_month": month,
                    "z": 0.1 * i,
                    "index": 100.0 + i,
                    "n_countries": 3,
                    "weight_covered": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _counts():
    rows = []
    for month in ["2024-01", "2024-02", "2024-03"]:
        rows.append(
            {
                "country": "Nigeria",
                "site": "The Punch",
                "year_month": month,
                "epu_articles": 3,
                "all_articles": 10,
            }
        )
    return pd.DataFrame(rows)


def _baseline():
    return ni.index.Baseline(
        start="2015-01",
        end="2023-12",
        mu={"Nigeria": 1.0, "Kenya": 1.2},
        sigma={"Nigeria": {"The Punch": 0.5, "Guardian": 0.3}, "Kenya": {"Daily Nation": 0.4}},
    )


# --- plot() -----------------------------------------------------------------

def test_plot_matplotlib_default():
    fig = ni.plot(_result())
    assert isinstance(fig, plt.Figure)


@pytest.mark.parametrize("kind", ["line", "bar", "area"])
def test_plot_kinds(kind):
    fig = ni.plot(_result(), kind=kind)
    assert isinstance(fig, plt.Figure)


def test_plot_bad_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        ni.plot(_result(), kind="scatter")


def test_plot_bad_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        ni.plot(_result(), backend="excel")


def test_plot_plotly_backend():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    fig = ni.plot(_result(), backend="plotly")
    assert isinstance(fig, go.Figure)


def test_plot_seaborn_backend():
    pytest.importorskip("seaborn")
    fig = ni.plot(_result(), backend="seaborn")
    assert isinstance(fig, plt.Figure)


# --- plot_compare -------------------------------------------------------

def test_plot_compare_subset_countries():
    fig = ni.plot_compare(_result(), countries=["Nigeria"])
    assert isinstance(fig, plt.Figure)


def test_plot_compare_all_countries_default():
    fig = ni.plot_compare(_result())
    assert isinstance(fig, plt.Figure)


# --- plot_regions ---------------------------------------------------------

def test_plot_regions():
    fig = ni.plot_regions(_regions_result())
    assert isinstance(fig, plt.Figure)


# --- plot_heatmap -----------------------------------------------------------

def test_plot_heatmap():
    fig = ni.plot_heatmap(_result())
    assert isinstance(fig, plt.Figure)


def test_plot_heatmap_seaborn():
    pytest.importorskip("seaborn")
    fig = ni.plot_heatmap(_result(), backend="seaborn")
    assert isinstance(fig, plt.Figure)


# --- plot_rank --------------------------------------------------------------

def test_plot_rank():
    fig = ni.plot_rank(_result(), "2024-02")
    assert isinstance(fig, plt.Figure)


def test_plot_rank_missing_month_raises():
    with pytest.raises(ValueError, match="2099-01"):
        ni.plot_rank(_result(), "2099-01")


# --- plot_rolling -------------------------------------------------------

def test_plot_rolling():
    fig = ni.plot_rolling(_result(), window=2)
    assert isinstance(fig, plt.Figure)


# --- plot_distribution --------------------------------------------------

@pytest.mark.parametrize("kind", ["hist", "box"])
def test_plot_distribution(kind):
    fig = ni.plot_distribution(_result(), kind=kind)
    assert isinstance(fig, plt.Figure)


def test_plot_distribution_bad_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        ni.plot_distribution(_result(), kind="violin")


# --- plot_components ------------------------------------------------------

def test_plot_components():
    fig = ni.plot_components(_counts())
    assert isinstance(fig, plt.Figure)


# --- plot_baseline ----------------------------------------------------------

def test_plot_baseline():
    fig = ni.plot_baseline(_baseline())
    assert isinstance(fig, plt.Figure)


# --- plot_correlation -------------------------------------------------------

def test_plot_correlation():
    external = pd.Series([1.0, 2.0, 3.0, 4.0], index=["2024-01", "2024-02", "2024-03", "2024-04"])
    single_country = _result()[_result()["country"] == "Nigeria"]
    fig = ni.plot_correlation(single_country, external)
    assert isinstance(fig, plt.Figure)


def test_plot_correlation_multi_country_raises():
    external = pd.Series([1.0, 2.0], index=["2024-01", "2024-02"])
    with pytest.raises(ValueError, match="single-country"):
        ni.plot_correlation(_result(), external)


def test_plot_correlation_dataframe_external():
    ext = pd.DataFrame({"year_month": ["2024-01", "2024-02", "2024-03", "2024-04"], "vix": [1.0, 2.0, 3.0, 4.0]})
    single_country = _result()[_result()["country"] == "Kenya"]
    fig = ni.plot_correlation(single_country, ext)
    assert isinstance(fig, plt.Figure)


# --- plot_dashboard ---------------------------------------------------------

def test_plot_dashboard():
    fig = ni.plot_dashboard(_result())
    assert isinstance(fig, plt.Figure)


def test_plot_dashboard_plotly_raises():
    with pytest.raises(NotImplementedError):
        ni.plot_dashboard(_result(), backend="plotly")


# --- theme --------------------------------------------------------------

def test_theme_applies_rcparams():
    ni.theme()
    assert plt.rcParams["axes.grid"] is True


def test_theme_bad_style_raises():
    with pytest.raises(ValueError, match="style"):
        ni.theme(style="dark")


def test_theme_non_matplotlib_backend_raises():
    with pytest.raises(NotImplementedError):
        ni.theme(backend="plotly")


# --- save_plot ------------------------------------------------------------

def test_save_plot_matplotlib(tmp_path):
    fig = ni.plot(_result())
    path = tmp_path / "chart.png"
    ni.save_plot(fig, path)
    assert path.exists() and path.stat().st_size > 0


def test_save_plot_plotly_html(tmp_path):
    pytest.importorskip("plotly")
    fig = ni.plot(_result(), backend="plotly")
    path = tmp_path / "chart.html"
    ni.save_plot(fig, path)
    assert path.exists() and path.stat().st_size > 0


def test_save_plot_plotly_image(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    fig = ni.plot(_result(), backend="plotly")
    path = tmp_path / "chart.png"
    ni.save_plot(fig, path)
    assert path.exists() and path.stat().st_size > 0


def test_save_plot_unknown_type_raises(tmp_path):
    with pytest.raises(TypeError):
        ni.save_plot(object(), tmp_path / "chart.png")
