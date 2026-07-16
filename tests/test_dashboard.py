"""ROUND 3 - Dashboard smoke test using Streamlit's AppTest harness.

Launches app.py headless, renders every page, drives the main controls, and
asserts no exceptions and that copilot panels / charts produce output.
"""
import pytest

import config
import generate_data

try:
    from streamlit.testing.v1 import AppTest
    HAVE_APPTEST = True
except Exception:  # pragma: no cover - very old streamlit
    HAVE_APPTEST = False

pytestmark = pytest.mark.skipif(not HAVE_APPTEST, reason="streamlit AppTest unavailable")

APP = str(config.ROOT / "app.py")


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
    if not config.FLEET_DATA_CSV.exists() or not config.BATTERY_DATA_CSV.exists():
        generate_data.main()


def _run_page(label: str):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    # Select the page in the sidebar radio and re-run.
    at.sidebar.radio[0].set_value(label).run()
    assert not at.exception
    return at


def test_home_renders():
    at = _run_page("🏠 Home")
    assert len(at.metric) >= 3  # three headline metric cards


def test_battery_page_renders_and_explains():
    at = _run_page("🔋 Battery Health")
    assert not at.exception
    # A copilot explanation should appear as markdown/text on the page.
    assert len(at.markdown) > 0


def test_readiness_page_renders():
    at = _run_page("🚚 Fleet Readiness")
    assert not at.exception
    assert len(at.dataframe) >= 1  # the ranked fleet table


def test_carbon_page_renders():
    at = _run_page("🌱 Carbon Savings")
    assert not at.exception
    assert len(at.metric) >= 3


def test_unusual_input_does_not_crash():
    """Set the readiness min-score slider to its max; the app must not crash."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("🚚 Fleet Readiness").run()
    assert not at.exception
    if at.slider:
        at.slider[0].set_value(100).run()
        assert not at.exception
