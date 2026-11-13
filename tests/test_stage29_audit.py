"""Unit checks for the independent Stage 29 audit primitives."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from eqmarl_slam import diagnostics29 as audit
from eqmarl_slam.budgeted_entanglement import classical_value, xor_sdp_upper

ROOT = Path(__file__).resolve().parents[1]


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, 0.5]
    assert audit.holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert audit.holm([1.0, 1.0]) == [1.0, 1.0]


@pytest.mark.parametrize("seed", range(5))
def test_exact_signed_rank_matches_scipy(seed):
    differences = np.random.default_rng(seed).normal(0.01, 0.05, size=12)
    expected = stats.wilcoxon(differences, alternative="greater", method="exact").pvalue
    assert audit.exact_signed_rank_greater(differences) == pytest.approx(expected, abs=1e-12)


def test_percentile_bootstrap_is_seeded_and_ordered():
    values = np.linspace(-1, 1, 12)
    low, high = audit.percentile_bootstrap(values, 7, 2000, 2.5)
    assert low < values.mean() < high
    assert audit.percentile_bootstrap(values, 7, 2000, 2.5) == [low, high]


def test_certificate_primitives_agree_with_stage21_definitions():
    requests = json.loads((ROOT / "configs/data/qip_v1/development-request-types.json").read_text())["types"][:3]
    for request in requests:
        assert audit.classical_optimum(request) == pytest.approx(classical_value(request)[0], abs=1e-12)
        assert audit.xor_quantum_upper(request) == pytest.approx(xor_sdp_upper(request)["win_upper"], abs=1e-6)
        # Equal measurement angles give perfect correlation, i.e. the "always equal" strategy.
        predicate = np.asarray(request["predicate"]); weights = np.asarray(request["weights"], float)
        assert audit.bell_value_at_angles(request, [0.0] * 6) == pytest.approx((weights * (predicate == 0)).sum() / weights.sum())
