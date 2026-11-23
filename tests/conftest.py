from pathlib import Path

import pytest

from src.utils.config import read_config


@pytest.fixture
def config():
    return read_config(Path(__file__).resolve().parents[1] / "configs" / "strategy_config.yaml")


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parents[1]
