import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import pytest

from app import config, db


@pytest.fixture()
def temp_env(tmp_path, monkeypatch):
    """Redirect all storage into a temp dir and provide a fresh DB."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_VOICE_MODEL_DIR", raising=False)
    # Reload config so module constants point at the temp directory
    import importlib
    importlib.reload(config)
    importlib.reload(db)
    importlib.reload(sys.modules["app.storage"])
    for sub in ("voices", "samples", "generated"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    yield data_dir
    # reload back to defaults is unnecessary per test since each test reloads
