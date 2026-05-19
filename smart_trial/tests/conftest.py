import os

import pytest


@pytest.fixture(autouse=True)
def use_mock_models(monkeypatch):
    """Run tests without live API keys unless explicitly disabled."""
    if os.environ.get("SMART_TRIAL_LIVE_API", "").lower() in ("1", "true", "yes"):
        return
    monkeypatch.setenv("SMART_TRIAL_USE_MOCK", "1")
