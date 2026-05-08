"""Validation tests for the RAAN-vs-phase torus regularizer (cluster_phase_space.py).

Run with:
    python -m pytest test_cluster_phase_space.py -v
or:
    python test_cluster_phase_space.py
"""

from __future__ import annotations

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from cluster_phase_space import (
    wrap_to_360,
    wrap_to_pm180,
    circular_distance_deg,
    unwrap_angle_series_deg,
    compute_phase_variable_deg,
    build_torus_target_slots,
    build_assignment_cost_matrix,
    solve_slot_assignment,
    compute_slot_fit_loss,
    compute_circular_gap_loss_deg,
    estimate_drift_deg_per_day,
    compute_drift_loss,
    fit_torus_lattice,
    compute_raan_phase_regularizer,
    compute_torus_consistency_penalty,
    TorusLattice,
    TorusRegularizerResult,
    ShellTorusSummary,
)


# ======================================================================
# Angular wrapping tests
# ======================================================================

class TestAngularWrapping:
    def test_wrap_to_360_basic(self):
        assert wrap_to_360(0.0) == pytest.approx(0.0)
        assert wrap_to_360(360.0) == pytest.approx(0.0)
        assert wrap_to_360(361.0) == pytest.approx(1.0)
        assert wrap_to_360(-1.0) == pytest.approx(359.0)

    def test_wrap_to_360_array(self):
        arr = np.array([0.0, 180.0, 360.0, -90.0, 450.0])
        result = wrap_to_360(arr)
        expected = np.array([0.0, 180.0, 0.0, 270.0, 90.0])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_wrap_to_pm180(self):
        assert wrap_to_pm180(0.0) == pytest.approx(0.0)
        assert wrap_to_pm180(180.0) == pytest.approx(-180.0)
        assert wrap_to_pm180(181.0) == pytest.approx(-179.0)
        assert wrap_to_pm180(-179.0) == pytest.approx(-179.0)

    def test_circular_distance_near_zero(self):
        """Distance between angles near 0/360 boundary."""
        assert circular_distance_deg(1.0, 359.0) == pytest.approx(2.0, abs=1e-10)
        assert circular_distance_deg(359.0, 1.0) == pytest.approx(-2.0, abs=1e-10)

    def test_circular_distance_identical(self):
        assert circular_distance_deg(45.0, 45.0) == pytest.approx(0.0)

    def test_circular_distance_opposite(self):
        assert abs(circular_distance_deg(0.0, 180.0)) == pytest.approx(180.0)

    def test_unwrap_monotonic(self):
        """Unwrapped series should be monotonically increasing for constant drift."""
        theta = np.mod(np.arange(100) * 10.0, 360.0)  # 0, 10, ..., 350, 0, 10, ...
        unwrapped = unwrap_angle_series_deg(theta)
        diffs = np.diff(unwrapped)
        assert np.all(diffs > 0), "Unwrapped series should be monotonically increasing"


# ======================================================================
# Phase variable computation
# ======================================================================

class TestPhaseVariable:
    def test_mean_anomaly_mode(self):
        oe = np.array([6928.0, 0.001, np.radians(53.0), np.radians(30.0),
                        np.radians(45.0), np.radians(60.0)])
        psi = compute_phase_variable_deg(oe, mode="raan_mean_anomaly")
        assert psi == pytest.approx(60.0, abs=0.01)

    def test_mean_longitude_mode(self):
        oe = np.array([6928.0, 0.001, np.radians(53.0), np.radians(30.0),
                        np.radians(45.0), np.radians(60.0)])
        psi = compute_phase_variable_deg(oe, mode="raan_mean_longitude")
        expected = wrap_to_360(45.0 + 30.0 + 60.0)
        assert psi == pytest.approx(expected, abs=0.01)

    def test_argument_of_latitude_near_circular(self):
        """For nearly circular orbits, u = omega + f ≈ omega + M."""
        oe = np.array([6928.0, 1e-8, np.radians(53.0), np.radians(30.0),
                        np.radians(45.0), np.radians(60.0)])
        psi = compute_phase_variable_deg(oe, mode="raan_argument_of_latitude")
        expected_approx = wrap_to_360(30.0 + 60.0)
        assert psi == pytest.approx(expected_approx, abs=0.1)

    def test_invalid_mode_raises(self):
        oe = np.array([6928.0, 0.001, 0.9, 0.5, 0.8, 1.0])
        with pytest.raises(ValueError, match="Unsupported phase mode"):
            compute_phase_variable_deg(oe, mode="invalid_mode")


# ======================================================================
# Torus lattice
# ======================================================================

class TestTorusLattice:
    def test_single_plane_single_slot(self):
        lat = build_torus_target_slots(1, 1)
        assert lat.n_slots == 1
        assert lat.target_raan_deg.shape == (1,)
        assert lat.target_phase_deg.shape == (1,)

    def test_3_planes_4_slots(self):
        lat = build_torus_target_slots(3, 4)
        assert lat.n_slots == 12
        assert lat.target_raan_deg.shape == (12,)
        # Verify RAAN spacing: 120 deg apart
        unique_raans = np.unique(np.round(lat.target_raan_deg, 6))
        assert len(unique_raans) == 3
        # Verify phase spacing within a plane: 90 deg apart
        plane0_mask = np.round(lat.target_raan_deg, 6) == unique_raans[0]
        plane0_phases = np.sort(lat.target_phase_deg[plane0_mask])
        gaps = np.diff(plane0_phases)
        np.testing.assert_allclose(gaps, 90.0, atol=1e-10)

    def test_eta_shifts_phase(self):
        """Non-zero eta should shift phase in later planes."""
        lat0 = build_torus_target_slots(3, 4, eta=0.0)
        lat1 = build_torus_target_slots(3, 4, eta=0.5)
        # Plane 0 should be identical
        p0_mask_0 = np.round(lat0.target_raan_deg, 6) == np.round(lat0.target_raan_deg[0], 6)
        p0_mask_1 = np.round(lat1.target_raan_deg, 6) == np.round(lat1.target_raan_deg[0], 6)
        np.testing.assert_allclose(
            np.sort(lat0.target_phase_deg[p0_mask_0]),
            np.sort(lat1.target_phase_deg[p0_mask_1]),
            atol=1e-10,
        )
        # Later planes should differ
        assert not np.allclose(
            np.sort(lat0.target_phase_deg),
            np.sort(lat1.target_phase_deg),
        )


# ======================================================================
# Assignment
# ======================================================================

class TestAssignment:
    def test_perfect_lattice_zero_cost(self):
        """If satellites sit exactly on target slots, assignment cost ≈ 0."""
        lat = build_torus_target_slots(3, 4)
        J, row, col = compute_slot_fit_loss(
            lat.target_raan_deg, lat.target_phase_deg, lat,
        )
        assert J == pytest.approx(0.0, abs=1e-8)
        assert len(row) == lat.n_slots

    def test_more_sats_than_slots(self):
        """Extra satellites should still be handled."""
        lat = build_torus_target_slots(2, 2)  # 4 slots
        rng = np.random.default_rng(123)
        raan = rng.uniform(0, 360, 10)
        phase = rng.uniform(0, 360, 10)
        J, row, col = compute_slot_fit_loss(raan, phase, lat)
        assert len(row) <= 4  # at most n_slots assigned
        assert J > 0.0

    def test_fewer_sats_than_slots(self):
        """Fewer sats than slots — still works."""
        lat = build_torus_target_slots(3, 4)  # 12 slots
        raan = np.array([0.0, 120.0])
        phase = np.array([0.0, 0.0])
        J, row, col = compute_slot_fit_loss(raan, phase, lat)
        assert len(row) == 2


# ======================================================================
# Gap-uniformity
# ======================================================================

class TestGapLoss:
    def test_uniform_spacing_zero_loss(self):
        """Perfectly uniform spacing → J_gap ≈ 0."""
        angles = np.array([0.0, 90.0, 180.0, 270.0])
        assert compute_circular_gap_loss_deg(angles) == pytest.approx(0.0, abs=1e-12)

    def test_single_angle_zero_loss(self):
        assert compute_circular_gap_loss_deg(np.array([42.0])) == 0.0

    def test_clustered_angles_nonzero_loss(self):
        """All angles in a 10-deg band → large gap loss."""
        angles = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        J = compute_circular_gap_loss_deg(angles)
        assert J > 1.0


# ======================================================================
# Drift estimation
# ======================================================================

class TestDrift:
    def test_zero_drift(self):
        """Constant angle → drift = 0."""
        angles = np.full(100, 45.0)
        times = np.linspace(0, 10, 100)
        rate = estimate_drift_deg_per_day(angles, times)
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_known_drift(self):
        """5 deg/day drift."""
        times = np.linspace(0, 100, 1000)
        angles = np.mod(45.0 + 5.0 * times, 360.0)
        rate = estimate_drift_deg_per_day(angles, times)
        assert rate == pytest.approx(5.0, abs=0.01)

    def test_drift_loss_identical_rates(self):
        """If all reps have same drift, variance = 0."""
        times = np.linspace(0, 100, 500)
        series1 = np.mod(10.0 + 3.0 * times, 360.0)
        series2 = np.mod(50.0 + 3.0 * times, 360.0)
        raan = [series1, series2]
        phase = [series1 * 0, series2 * 0]  # zero phase drift
        J = compute_drift_loss(raan, phase, [times, times])
        assert J == pytest.approx(0.0, abs=0.01)


# ======================================================================
# Torus consistency penalty
# ======================================================================

class TestTorusConsistencyPenalty:
    def test_identical_summaries_zero(self):
        s = ShellTorusSummary(omega0_deg=45.0, psi0_deg=90.0, eta=0.5)
        assert compute_torus_consistency_penalty(s, s) == pytest.approx(0.0, abs=1e-12)

    def test_different_omega0(self):
        s1 = ShellTorusSummary(omega0_deg=0.0, psi0_deg=0.0, eta=0.0)
        s2 = ShellTorusSummary(omega0_deg=10.0, psi0_deg=0.0, eta=0.0)
        penalty = compute_torus_consistency_penalty(s1, s2)
        assert penalty == pytest.approx(100.0, abs=0.01)  # 10^2 * 1.0


# ======================================================================
# Lattice fitting
# ======================================================================

class TestLatticeFitting:
    def test_perfect_lattice_recovery(self):
        """Fitting should recover a lattice from perfect data."""
        lat = build_torus_target_slots(3, 4, omega0_deg=15.0, psi0_deg=20.0, eta=0.0)
        fitted = fit_torus_lattice(
            lat.target_raan_deg, lat.target_phase_deg,
            n_planes=3, slots_per_plane=4, fit_eta=False,
        )
        J, _, _ = compute_slot_fit_loss(
            lat.target_raan_deg, lat.target_phase_deg, fitted,
        )
        assert J < 1.0, f"Lattice fit should recover near-perfect assignment, got J={J}"


# ======================================================================
# Integration: disabled config doesn't break baseline
# ======================================================================

class TestDisabledConfig:
    def test_disabled_returns_empty_result(self):
        """With no results and disabled config, should return default."""
        result = compute_raan_phase_regularizer(
            results=[],
            mode="raan_mean_anomaly",
            n_planes=1,
            slots_per_plane=1,
        )
        assert result.J_torus == 0.0
        assert result.lattice is None


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
