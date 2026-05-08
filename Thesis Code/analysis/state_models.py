"""Shared state-vector generation helpers for TLE analytics.

Why this module exists:
- Most pipeline analytics operate on TLE-derived mean-element proxy quantities
  (e.g., sma, true anomaly proxies, secular-rate diagnostics).
- A subset of analyses needs Cartesian states synchronized at a common epoch.
- For those use cases, SGP4 propagation is optional and configurable.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from orbital_mechanics import sv_from_coe

try:
    from sgp4.api import Satrec, jday

    _HAS_SGP4 = True
except Exception:
    Satrec = None
    jday = None
    _HAS_SGP4 = False

DEFAULT_STATE_MODEL = "classical"
SUPPORTED_STATE_MODELS = ("classical", "sgp4_preferred", "sgp4_required")
STATE_MODEL_ALIASES = {"sgp4": "sgp4_preferred"}

def _normalize_state_model(state_model: str | None) -> str:
    value = DEFAULT_STATE_MODEL if state_model is None else str(state_model).strip().lower()
    value = STATE_MODEL_ALIASES.get(value, value)
    if value not in SUPPORTED_STATE_MODELS:
        raise ValueError(f"Unsupported state_model='{state_model}'. "
                         f"Expected one of {SUPPORTED_STATE_MODELS}.")
    return value

def normalize_state_model(state_model: str | None) -> str:
    """Normalize user/runtime state-model strings to a supported policy name."""
    return _normalize_state_model(state_model)

def has_sgp4() -> bool:
    """Return True when sgp4.api is importable in this environment."""
    return bool(_HAS_SGP4)

def _normalize_epoch_datetime(epoch: Any) -> datetime | None:
    if epoch is None:
        return None

    ts = pd.to_datetime(epoch, errors="coerce")
    if pd.isna(ts):
        return None

    if isinstance(ts, pd.DatetimeIndex):
        if len(ts) == 0:
            return None
        ts = ts[0]

    ts_obj = pd.Timestamp(ts)
    if ts_obj.tzinfo is not None:
        ts_obj = ts_obj.tz_convert("UTC").tz_localize(None)
    return ts_obj.to_pydatetime()

def _state_from_row_classical(row: Any, mu: float) -> tuple[np.ndarray, np.ndarray]:
    e = row.get("ecc", np.nan)
    h = row.get("specific_angular_momentum", np.nan)
    sma = row.get("sma", np.nan)

    if (not np.isfinite(h) or h <= 0.0) and np.isfinite(sma) and sma > 0.0 and np.isfinite(e):
        h = np.sqrt(mu * sma * (1.0 - e**2))

    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("Unable to construct valid specific angular momentum from row data")

    i = np.deg2rad(row["inc"])
    raan = np.deg2rad(row["raan"])
    argp = np.deg2rad(row["aop"])
    true_anomaly_deg = row.get("true_anomaly", row.get("mean_anomaly", np.nan))
    theta = np.deg2rad(true_anomaly_deg)
    r_vec, v_vec = sv_from_coe((h, e, raan, i, argp, theta), mu)
    return np.asarray(r_vec, dtype=np.float64), np.asarray(v_vec, dtype=np.float64)

@lru_cache(maxsize=20000)
def _satrec_from_lines_cached(line1: str, line2: str):
    if not _HAS_SGP4 or Satrec is None:
        return None
    return Satrec.twoline2rv(str(line1), str(line2))

def build_satrec_from_row(row: Any):
    """Build and cache a Satrec from raw TLE lines stored in a row-like object."""
    line1 = row.get("tle_line1_raw")
    line2 = row.get("tle_line2_raw")

    if line1 is None or line2 is None:
        return None

    line1_str = str(line1).strip()
    line2_str = str(line2).strip()
    if not line1_str or not line2_str:
        return None

    if not _HAS_SGP4:
        return None

    try:
        return _satrec_from_lines_cached(line1_str, line2_str)
    except Exception:
        return None

def _base_meta(state_model_requested: str) -> dict[str, Any]:
    return {"state_model_requested": state_model_requested, "state_model_used": None,
            "state_frame": None, "frame": None, "source": None, "state_source": None,
            "epoch_source": None, "element_semantics": None, "sgp4_error_code": None,
            "fallback_used": False, "ok": False, "error": None}

def _propagate_with_satrec(satrec, epoch_dt: datetime) -> tuple[int, np.ndarray, np.ndarray]:
    jd, fr = jday(epoch_dt.year, epoch_dt.month, epoch_dt.day, epoch_dt.hour, epoch_dt.minute,
                  epoch_dt.second + epoch_dt.microsecond / 1e6)
    err, r_km, v_kms = satrec.sgp4(jd, fr)
    return int(err), np.asarray(r_km, dtype=np.float64), np.asarray(v_kms, dtype=np.float64)

def propagate_row_to_teme_state(row: Any, epoch: Any, *, state_model: str = DEFAULT_STATE_MODEL) -> dict[str, Any]:
    """Attempt SGP4 propagation and return a structured result payload.

    The function always reports TEME when SGP4 succeeds and does not silently
    rename the output frame.
    """
    requested = _normalize_state_model(state_model)
    meta = _base_meta(requested)
    meta["epoch_source"] = "provided_epoch" if epoch is not None else "row_timestamp"

    epoch_dt = _normalize_epoch_datetime(epoch)
    if epoch_dt is None:
        meta["error"] = "Invalid or missing epoch for SGP4 propagation"
        return {"ok": False, "r_km": None, "v_kms": None, "meta": meta}

    if not _HAS_SGP4:
        meta["error"] = "sgp4 package unavailable"
        return {"ok": False, "r_km": None, "v_kms": None, "meta": meta}

    satrec = build_satrec_from_row(row)
    if satrec is None:
        meta["error"] = "Missing or invalid raw TLE lines"
        return {"ok": False, "r_km": None, "v_kms": None, "meta": meta}

    try:
        err, r_km, v_kms = _propagate_with_satrec(satrec, epoch_dt)
    except Exception as exc:
        meta["error"] = f"SGP4 propagation failed: {exc}"
        return {"ok": False, "r_km": None, "v_kms": None, "meta": meta}

    meta["sgp4_error_code"] = int(err)
    if err != 0:
        meta["error"] = f"SGP4 returned non-zero error code: {err}"
        return {"ok": False, "r_km": None, "v_kms": None, "meta": meta}

    meta["state_model_used"] = "sgp4"
    meta["state_frame"] = "TEME"
    meta["frame"] = "TEME"
    meta["source"] = "sgp4_tle"
    meta["state_source"] = "sgp4_tle"
    meta["element_semantics"] = "SGP4 TEME state from TLE mean elements"
    meta["ok"] = True
    return {"ok": True, "r_km": r_km, "v_kms": v_kms, "meta": meta}

def state_from_row(row: Any, epoch: Any = None, mu: float = 398600.4418,
                   state_model: str = DEFAULT_STATE_MODEL) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a state vector from a row according to requested policy.

    Policies:
    - classical: always use approximate reconstruction from mean-element proxies.
    - sgp4_preferred: use SGP4 when available; fallback to classical with explicit metadata.
    - sgp4_required: require SGP4 and fail explicitly when unavailable.
    """
    requested = _normalize_state_model(state_model)

    if requested == "classical":
        r_km, v_kms = _state_from_row_classical(row, mu)
        meta = _base_meta(requested)
        meta.update({"state_model_used": "classical", "state_frame": "proxy_inertial",
                     "frame": "proxy_inertial", "source": "derived_columns",
                     "state_source": "derived_columns",
                     "epoch_source": "provided_epoch" if epoch is not None else "row_timestamp",
                     "element_semantics": "TLE-derived Kepler proxy diagnostics", "ok": True})
        return r_km, v_kms, meta

    effective_epoch = epoch
    if effective_epoch is None and hasattr(row, "get"):
        effective_epoch = row.get("timestamp")

    sgp4_result = propagate_row_to_teme_state(row, effective_epoch, state_model=requested)
    if sgp4_result["ok"]:
        return sgp4_result["r_km"], sgp4_result["v_kms"], sgp4_result["meta"]

    if requested == "sgp4_required":
        err = sgp4_result["meta"].get("error") or "SGP4 state generation failed"
        raise RuntimeError(f"state_model=sgp4_required failed: {err}")

    # sgp4_preferred fallback path.
    r_km, v_kms = _state_from_row_classical(row, mu)
    meta = sgp4_result["meta"].copy()
    meta.update({"state_model_used": "classical", "state_frame": "proxy_inertial",
                 "frame": "proxy_inertial", "source": "derived_columns", "state_source": "derived_columns",
                   "element_semantics": "TLE-derived Kepler proxy diagnostics", "fallback_used": True, "ok": True})
    return r_km, v_kms, meta

__all__ = ["DEFAULT_STATE_MODEL", "SUPPORTED_STATE_MODELS", "normalize_state_model", 
           "build_satrec_from_row", "has_sgp4", "propagate_row_to_teme_state", "state_from_row"]