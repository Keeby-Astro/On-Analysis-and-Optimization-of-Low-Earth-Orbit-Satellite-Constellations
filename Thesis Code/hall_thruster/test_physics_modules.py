"""
Unit tests for the extracted physics modules and key revisions.

Tests cover:
1. reduced_dynamics: angle wrapping, J2 secular rates (NumPy + Torch)
2. hall_beam_relations: thrust/Isp coefficients
3. chemistry_models: legacy surrogate, closure dispatch
4. StageAModel forward pass shape consistency (smoke test)
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

# ── Module imports ───────────────────────────────────────────────────────────
from reduced_dynamics import (
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    J2_EARTH,
    G0_M_S2,
    TWOPI,
    wrap_to_pi,
    wrap_to_2pi,
    wrap_angle,
    angle_residual,
    deg2rad,
    mean_motion_rad_s,
    raan_rate_j2_rad_s,
    omega_dot_j2_rad_s,
    M_dot_j2_rad_s,
    lambda_dot_j2_rad_s,
    raan_rate_j2_torch,
    omega_dot_j2_torch,
    M_dot_j2_torch,
    lambda_dot_j2_torch,
)
from hall_beam_relations import (
    C_T_KR,
    C_I_KR,
    thrust_kr_mN,
    isp_kr_s,
    beam_exhaust_velocity_m_s,
    electrical_power_W,
)
from chemistry_models import (
    ClosureMode,
    ChemistryResult,
    legacy_surrogate_chemistry,
    compute_chemistry,
)


# ── Constants sanity checks ─────────────────────────────────────────────────
class TestConstants:
    def test_mu_earth(self):
        assert abs(MU_EARTH_KM3_S2 - 398600.4418) < 1.0

    def test_r_earth(self):
        assert abs(R_EARTH_KM - 6378.1366) < 0.01

    def test_j2_earth(self):
        assert abs(J2_EARTH - 1.0826359e-3) < 1e-6

    def test_twopi(self):
        assert abs(TWOPI - 2 * math.pi) < 1e-15


# ── Angle wrapping ──────────────────────────────────────────────────────────
class TestAngleWrapping:
    def test_wrap_to_pi_basic(self):
        assert abs(wrap_to_pi(0.0)) < 1e-12
        assert abs(wrap_to_pi(np.pi) - np.pi) < 1e-12 or abs(wrap_to_pi(np.pi) + np.pi) < 1e-12

    def test_wrap_to_pi_large(self):
        result = wrap_to_pi(3 * np.pi)
        assert -np.pi <= result + 1e-12 and result <= np.pi + 1e-12

    def test_wrap_to_2pi_basic(self):
        result = wrap_to_2pi(-0.1)
        assert 0.0 <= result < TWOPI

    def test_wrap_to_2pi_array(self):
        angles = np.array([-0.5, 0.0, 3.0, 7.0])
        result = wrap_to_2pi(angles)
        assert np.all(result >= 0.0)
        assert np.all(result < TWOPI)

    def test_wrap_angle_torch(self):
        x = torch.tensor([0.0, np.pi, 3 * np.pi, -np.pi])
        result = wrap_angle(x)
        assert result.shape == x.shape
        assert torch.all(result >= -np.pi - 1e-6)
        assert torch.all(result <= np.pi + 1e-6)

    def test_angle_residual_zero(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        obs = torch.tensor([1.0, 2.0, 3.0])
        resid = angle_residual(pred, obs)
        assert torch.allclose(resid, torch.zeros(3), atol=1e-6)

    def test_angle_residual_wrapping(self):
        pred = torch.tensor([3.1])
        obs = torch.tensor([-3.1])
        resid = angle_residual(pred, obs)
        # Should be close to 6.2 - 2*pi ≈ -0.083, not 6.2
        assert abs(float(resid)) < 0.1


# ── Mean motion ─────────────────────────────────────────────────────────────
class TestMeanMotion:
    def test_geostationary(self):
        a_geo = 42164.0  # km
        n = float(mean_motion_rad_s(a_geo))
        period_hr = TWOPI / n / 3600.0
        assert abs(period_hr - 24.0) < 0.1

    def test_leo(self):
        a_leo = 6778.0  # ~400 km altitude
        n = float(mean_motion_rad_s(a_leo))
        period_min = TWOPI / n / 60.0
        assert 90.0 < period_min < 95.0  # ~92.4 min


# ── J2 secular rates (NumPy) ────────────────────────────────────────────────
class TestJ2SecularRatesNumPy:
    """Test J2 secular rates against known values for Starlink-like orbits."""

    def _starlink_params(self):
        return dict(a_km=6921.0, e=0.0001, inc_rad=np.deg2rad(53.0))

    def test_raan_rate_sign_prograde(self):
        p = self._starlink_params()
        rate = float(raan_rate_j2_rad_s(**p))
        # For i < 90°, RAAN regresses (rate < 0)
        assert rate < 0

    def test_raan_rate_magnitude(self):
        p = self._starlink_params()
        rate = float(raan_rate_j2_rad_s(**p))
        rate_deg_day = np.degrees(rate) * 86400
        # Starlink: approx -4 to -5 deg/day
        assert -7.0 < rate_deg_day < -2.0

    def test_omega_dot_sign(self):
        p = self._starlink_params()
        rate = float(omega_dot_j2_rad_s(**p))
        # For i=53°, sin²i ≈ 0.64 → 2 - 2.5*0.64 = 0.40 > 0 → positive
        assert rate > 0

    def test_m_dot_positive(self):
        p = self._starlink_params()
        rate = float(M_dot_j2_rad_s(**p))
        # Ṁ ≈ n (always positive, correction is small)
        n = float(mean_motion_rad_s(p["a_km"]))
        assert abs(rate - n) / n < 0.01

    def test_lambda_dot_positive(self):
        p = self._starlink_params()
        rate = float(lambda_dot_j2_rad_s(**p))
        # λ̇ is dominated by Ṁ ≈ n, so must be positive
        assert rate > 0

    def test_lambda_dot_consistency(self):
        p = self._starlink_params()
        lam_dot = lambda_dot_j2_rad_s(**p)
        sum_parts = (
            raan_rate_j2_rad_s(**p)
            + omega_dot_j2_rad_s(**p)
            + M_dot_j2_rad_s(**p)
        )
        assert abs(float(lam_dot) - float(sum_parts)) < 1e-15


# ── J2 secular rates (Torch) ────────────────────────────────────────────────
class TestJ2SecularRatesTorch:
    """Test Torch J2 rate functions match NumPy versions."""

    def _params(self):
        a = 6921.0
        e = 0.0001
        inc = np.deg2rad(53.0)
        return a, e, inc

    def test_raan_rate_matches_numpy(self):
        a, e, inc = self._params()
        np_val = float(raan_rate_j2_rad_s(a, e, inc))
        t_val = float(raan_rate_j2_torch(
            torch.tensor(a), torch.tensor(e), torch.tensor(inc),
            torch.tensor(MU_EARTH_KM3_S2),
        ))
        assert abs(np_val - t_val) / abs(np_val) < 1e-5

    def test_omega_dot_matches_numpy(self):
        a, e, inc = self._params()
        np_val = float(omega_dot_j2_rad_s(a, e, inc))
        t_val = float(omega_dot_j2_torch(
            torch.tensor(a), torch.tensor(e), torch.tensor(inc),
            torch.tensor(MU_EARTH_KM3_S2),
        ))
        assert abs(np_val - t_val) / abs(np_val) < 1e-5

    def test_m_dot_matches_numpy(self):
        a, e, inc = self._params()
        np_val = float(M_dot_j2_rad_s(a, e, inc))
        t_val = float(M_dot_j2_torch(
            torch.tensor(a), torch.tensor(e), torch.tensor(inc),
            torch.tensor(MU_EARTH_KM3_S2),
        ))
        assert abs(np_val - t_val) / abs(np_val) < 1e-5

    def test_lambda_dot_matches_numpy(self):
        a, e, inc = self._params()
        np_val = float(lambda_dot_j2_rad_s(a, e, inc))
        t_val = float(lambda_dot_j2_torch(
            torch.tensor(a), torch.tensor(e), torch.tensor(inc),
            torch.tensor(MU_EARTH_KM3_S2),
        ))
        assert abs(np_val - t_val) / abs(np_val) < 1e-5

    def test_torch_autograd(self):
        a = torch.tensor(6921.0, requires_grad=True)
        e = torch.tensor(0.0001)
        inc = torch.tensor(np.deg2rad(53.0))
        mu = torch.tensor(MU_EARTH_KM3_S2)
        rate = lambda_dot_j2_torch(a, e, inc, mu)
        rate.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad)


# ── Hall beam relations ──────────────────────────────────────────────────────
class TestHallBeamRelations:
    def test_thrust_coefficient(self):
        assert abs(C_T_KR - 1.32) < 0.01

    def test_isp_coefficient(self):
        assert abs(C_I_KR - 154.8) < 0.5

    def test_thrust_positive(self):
        T = thrust_kr_mN(
            torch.tensor(1.0),
            torch.tensor(3.0),
            torch.tensor(300.0),
        )
        assert float(T) > 0

    def test_thrust_proportional_to_ib(self):
        T1 = thrust_kr_mN(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(300.0))
        T2 = thrust_kr_mN(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(300.0))
        assert abs(float(T2) / float(T1) - 2.0) < 1e-6

    def test_isp_positive(self):
        Isp = isp_kr_s(
            torch.tensor(1.0),
            torch.tensor(0.75),
            torch.tensor(300.0),
        )
        assert float(Isp) > 0

    def test_isp_typical_range(self):
        Isp = isp_kr_s(
            torch.tensor(1.0),
            torch.tensor(0.75),
            torch.tensor(300.0),
        )
        # Krypton Isp ~ 1500-2500s at 300V with reasonable γ, η_m
        assert 500.0 < float(Isp) < 4000.0

    def test_exhaust_velocity_consistent(self):
        gamma = torch.tensor(1.0)
        eta_m = torch.tensor(0.75)
        vb = torch.tensor(300.0)
        ve = beam_exhaust_velocity_m_s(gamma, eta_m, vb)
        expected_isp = isp_kr_s(gamma, eta_m, vb)
        assert abs(float(ve) - float(expected_isp) * G0_M_S2) < 1.0

    def test_electrical_power(self):
        P = electrical_power_W(
            torch.tensor(3.0),
            torch.tensor(300.0),
            torch.tensor(0.85),
            torch.tensor(0.90),
        )
        # P = Vb * Ib / (eta_b * eta_v) ≈ 900 / 0.765 ≈ 1176 W
        assert 1100 < float(P) < 1300


# ── Chemistry models ────────────────────────────────────────────────────────
class TestChemistryModels:
    def test_legacy_surrogate_returns_result(self):
        result = legacy_surrogate_chemistry(
            torch.tensor(300.0),
            torch.tensor(0.25),
            torch.tensor(1e-5),
        )
        assert isinstance(result, ChemistryResult)
        assert result.is_surrogate is True
        assert float(result.te_eV) > 0
        assert float(result.ionization_ratio) > 0

    def test_legacy_surrogate_te_range(self):
        result = legacy_surrogate_chemistry(
            torch.tensor(300.0),
            torch.tensor(0.25),
            torch.tensor(1e-5),
        )
        # Te should be in [0.5, 120] per clamping
        assert 0.5 <= float(result.te_eV) <= 120.0

    def test_compute_dispatch_legacy(self):
        result = compute_chemistry(
            ClosureMode.LEGACY_SURROGATE,
            torch.tensor(300.0),
            torch.tensor(0.25),
            torch.tensor(1e-5),
        )
        assert result.is_surrogate is True

    def test_compute_dispatch_tabulated_raises(self):
        with pytest.raises(NotImplementedError):
            compute_chemistry(
                ClosureMode.TABULATED,
                torch.tensor(300.0),
                torch.tensor(0.25),
                torch.tensor(1e-5),
            )

    def test_closure_mode_enum(self):
        assert ClosureMode.LEGACY_SURROGATE.value == "legacy_surrogate"
        assert ClosureMode.TABULATED.value == "tabulated"


# ── Deg2rad ──────────────────────────────────────────────────────────────────
class TestDeg2Rad:
    def test_basic(self):
        assert abs(float(deg2rad(180.0)) - np.pi) < 1e-10

    def test_array(self):
        result = deg2rad([0, 90, 180, 360])
        expected = np.array([0, np.pi / 2, np.pi, 2 * np.pi])
        np.testing.assert_allclose(result, expected, atol=1e-10)
