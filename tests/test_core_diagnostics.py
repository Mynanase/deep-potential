from __future__ import annotations

import numpy as np
import pytest

from dpjax.diagnostics import compare_df_samples


def test_df_diagnostics_compare_only_two_phase_space_arrays():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(256, 6)).astype(np.float32)
    generated = rng.normal(loc=0.1, size=(128, 6)).astype(np.float32)

    diagnostics = compare_df_samples(reference, generated, bins=16)

    assert diagnostics.reference_count == 256
    assert diagnostics.generated_count == 128
    assert diagnostics.bin_edges.shape == (6, 17)
    assert diagnostics.reference_density.shape == (6, 16)
    assert diagnostics.generated_density.shape == (6, 16)
    assert diagnostics.reference_mean.shape == (6,)


def test_df_diagnostics_reject_non_six_dimensional_inputs():
    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        compare_df_samples(
            np.zeros((10, 5), dtype=np.float32),
            np.zeros((10, 6), dtype=np.float32),
        )
