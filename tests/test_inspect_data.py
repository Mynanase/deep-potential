from __future__ import annotations

import numpy as np
import pytest

from experiments.inspect_data import summarize_eta


def test_summarize_eta():
    eta = np.array(
        [
            [3.0, 4.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )

    summary = summarize_eta(eta)

    assert summary["shape"] == (2, 6)
    np.testing.assert_allclose(summary["radius"], [0.0, 2.5, 5.0])
    np.testing.assert_allclose(summary["speed"], [2.0, 2.5, 3.0])


def test_summarize_eta_rejects_empty_data():
    with pytest.raises(ValueError, match="empty"):
        summarize_eta(np.empty((0, 6), dtype=np.float32))
