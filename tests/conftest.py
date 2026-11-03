"""
Shared pytest fixtures for the sublinear-Markov-semigroup replication suite.

All experiments use synthetic inputs; no market-data API is required.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data: tests that require live market-data API credentials "
        "(skipped when DATALOADER_API_KEY is absent)",
    )


# ---------------------------------------------------------------------------
# Small shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def x_grid_small() -> np.ndarray:
    """Coarse 1-D state grid for fast unit tests."""
    return np.linspace(-2.0, 2.0, 41)


@pytest.fixture(scope="session")
def x_grid_fine() -> np.ndarray:
    """Fine 1-D state grid."""
    return np.linspace(-3.0, 3.0, 121)


@pytest.fixture(scope="session")
def rng_fixed() -> np.random.Generator:
    """Deterministic RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def controls_small() -> list[dict]:
    """Tiny control set (3 controls) for fast tests."""
    return [
        {"sigma": 0.5, "b": -0.1, "c": 0.0},
        {"sigma": 1.0, "b":  0.0, "c": 0.0},
        {"sigma": 1.5, "b":  0.1, "c": -0.05},
    ]


@pytest.fixture(scope="session")
def alpha_small() -> float:
    """alpha parameter for Example 4.1 diffusion."""
    return 0.25  # in (0, 0.5) as required by the paper


@pytest.fixture(scope="session")
def smooth_payoff(x_grid_small: np.ndarray) -> np.ndarray:
    """Smooth bounded payoff on x_grid_small."""
    return np.exp(-0.5 * x_grid_small ** 2)
