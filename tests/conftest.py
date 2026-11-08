import pytest

from ooma.adapters.hyperliquid import HyperliquidAdapter


@pytest.fixture(autouse=True)
def fast_adapter(monkeypatch, request):
    if "live" not in request.keywords:                    # throttle only matters against the real API
        monkeypatch.setattr(HyperliquidAdapter, "MIN_GAP", 0.0)
