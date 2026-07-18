"""ROUND 3 - Dashboard smoke test using Streamlit's AppTest harness.

Launches app.py headless, renders every page, and asserts no exceptions and
that content is produced. Kept at the default (small) fleet size for speed.
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

PAGES = [
    "🏠 Command Center",
    "🔋 Battery Health",
    "🚚 Fleet Readiness",
    "🔗 Supply-Chain Risk",
    "🛠️ Maintenance & Charging",
    "🌱 Carbon Impact",
    "🧪 Scenario Lab",
    "🛰️ Digital Twin",
    "⚙️ System & Monitoring",
]


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
    generate_data.main()  # deterministic default fleet


def _run_page(label: str):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(label).run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("label", PAGES)
def test_every_page_renders(label):
    at = _run_page(label)
    assert len(at.markdown) > 0


def test_readiness_table_and_slider():
    at = _run_page("🚚 Fleet Readiness")
    assert len(at.dataframe) >= 1
    if at.slider:
        at.slider[0].set_value(100).run()
        assert not at.exception, at.exception
