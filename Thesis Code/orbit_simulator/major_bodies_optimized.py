# In this file, we import the initial state vector for the major bodies in
# the system (Sun-Earth-Moon) from the JPL Horizons database.

from astroquery.jplhorizons import Horizons
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random
import time
import os
import threading
from pathlib import Path
from requests.exceptions import RequestException
from functions_optimized import xyz2orb
from major_bodies_parameters_optimized import constants


# Thread-safe in-memory cache for current process.
_MB_MEM_CACHE_LOCK = threading.Lock()
_MB_MEM_CACHE = {}

# Optional persistent cache to avoid repeated Horizons calls across runs.
# Set MAJOR_BODIES_CACHE_DISABLE=1 to disable disk cache.
_MB_DISK_CACHE_DISABLED = os.getenv("MAJOR_BODIES_CACHE_DISABLE", "0") == "1"
_MB_DISK_CACHE_DIR = Path(
    os.getenv(
        "MAJOR_BODIES_CACHE_DIR",
        str(Path(__file__).resolve().parent / ".major_bodies_cache"),
    )
)


def _normalize_epoch_date(start_date: str) -> str:
    # Safety: normalize cache key date to YYYY-MM-DD and validate format.
    return datetime.strptime(str(start_date), '%Y-%m-%d').strftime('%Y-%m-%d')


def _cache_key_body(body: str, frame: str, start_date: str):
    return (str(body), str(frame).lower().strip(), _normalize_epoch_date(start_date))


def _cache_key_path(key):
    body, frame, epoch_date = key
    safe_name = f"body_{body}_frame_{frame}_epoch_{epoch_date}.npz"
    return _MB_DISK_CACHE_DIR / safe_name


def _load_body_disk_cache(key):
    if _MB_DISK_CACHE_DISABLED:
        return None

    path = _cache_key_path(key)
    if not path.is_file():
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            x = np.asarray(data["x"], dtype=np.float64)
            jd = float(data["jd"])
    except Exception:
        # Ignore corrupted cache entries and fall back to Horizons.
        return None

    if x.shape != (6,):
        return None

    return x, jd


def _save_body_disk_cache(key, x_state, jd):
    if _MB_DISK_CACHE_DISABLED:
        return

    try:
        _MB_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_key_path(key)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        # Safety: atomic replace avoids partially-written cache files.
        with tmp_path.open("wb") as f:
            np.savez_compressed(f, x=np.asarray(x_state, dtype=np.float64), jd=float(jd))
        os.replace(tmp_path, path)
    except Exception:
        # Cache write failures should never break the simulation path.
        pass


def _fetch_body_state_from_horizons(body, frame, start_date, au, day):
    # Fallback behavior preserved: use existing Horizons query function.
    vec = vector(body, frame, start_date)
    x_state = extract_state_vector(vec, au, day)
    jd = float(vec['datetime_jd'][0])
    return np.asarray(x_state, dtype=np.float64), jd


def _load_state_for_body_cached(body, frame, start_date, au, day):
    key = _cache_key_body(body, frame, start_date)

    with _MB_MEM_CACHE_LOCK:
        cached = _MB_MEM_CACHE.get(key)
    if cached is not None:
        return np.asarray(cached[0], dtype=np.float64), float(cached[1])

    disk_cached = _load_body_disk_cache(key)
    if disk_cached is not None:
        with _MB_MEM_CACHE_LOCK:
            _MB_MEM_CACHE[key] = (np.asarray(disk_cached[0], dtype=np.float64), float(disk_cached[1]))
        return np.asarray(disk_cached[0], dtype=np.float64), float(disk_cached[1])

    x_state, jd = _fetch_body_state_from_horizons(body, frame, start_date, au, day)

    with _MB_MEM_CACHE_LOCK:
        _MB_MEM_CACHE[key] = (np.asarray(x_state, dtype=np.float64), float(jd))
    _save_body_disk_cache(key, x_state, jd)
    return x_state, jd


def _is_transient_horizons_error(exc: Exception) -> bool:
    if isinstance(exc, RequestException):
        return True

    msg = str(exc).lower()
    transient_markers = (
        '503',
        '502',
        '504',
        '429',
        'service temporarily unavailable',
        'too many requests',
        'timed out',
        'connection reset',
        'temporarily unavailable',
    )
    return any(marker in msg for marker in transient_markers)

def vector(body, frame, start_date, max_retries=6, base_backoff_s=0.6):
    # Input:
    #     body - the body's name e.g. '301' (The Moon)
    #     frame - the reference - 'ecliptic' or 'earth' (equatorial)
    #     start_date - first date interval 'yyy-mm-dd'

    # Adding one day to the start date
    # Optimization: Using standard python datetime is fast enough here
    date_1 = datetime.strptime(start_date, '%Y-%m-%d')
    date_2 = date_1 + timedelta(days=1)
    end_date = date_2.strftime('%Y-%m-%d')

    # Query the JPL Horizons database
    # Note: Network I/O is the bottleneck here, addressed in load_mb via threading
    last_exc = None
    for attempt in range(max(1, int(max_retries))):
        try:
            obj = Horizons(id=body, location='500@399',
                           epochs={'start': start_date, 'stop': end_date,
                                   'step': '1d'})
            state_vec_mb = obj.vectors(refplane=frame)
            return state_vec_mb
        except Exception as exc:
            last_exc = exc
            is_retryable = _is_transient_horizons_error(exc)
            has_next_attempt = attempt < int(max_retries) - 1

            if not is_retryable or not has_next_attempt:
                break

            sleep_s = base_backoff_s * (2.0 ** attempt) + random.uniform(0.0, 0.25)
            print(f"Horizons retry {attempt + 1}/{max_retries - 1} for body={body}, date={start_date} "
                  f"after transient error: {exc}")
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to fetch Horizons vectors for body={body}, frame={frame}, date={start_date} "
        f"after {max_retries} attempts"
    ) from last_exc

def orbital_elements(body, frame, start_date):
    # Input:
    #     body - the body's name e.g. '301' (The Moon)
    #     frame - the reference - 'ecliptic' or 'earth' (equatorial)
    #     start_date - first date interval 'yyy-mm-dd'

    date_1 = datetime.strptime(start_date, '%Y-%m-%d')
    date_2 = date_1 + timedelta(days=1)
    end_date = date_2.strftime('%Y-%m-%d')

    # Query the JPL Horizons database
    obj = Horizons(id=body, location='500@399',
                   epochs={'start': start_date, 'stop': end_date,
                           'step': '10d'})
    oe_mb = obj.elements(refplane=frame)
    return oe_mb

# Helper function to extract state vector components
def extract_state_vector(vec, au, day):
    # Build state in one contiguous buffer to avoid temporary arrays.
    out = np.empty(6, dtype=np.float64)
    vel_scale = au / day
    out[0] = float(vec['x'][0]) * au
    out[1] = float(vec['y'][0]) * au
    out[2] = float(vec['z'][0]) * au
    out[3] = float(vec['vx'][0]) * vel_scale
    out[4] = float(vec['vy'][0]) * vel_scale
    out[5] = float(vec['vz'][0]) * vel_scale
    return out

# Function to load the major bodies' initial conditions
# Input: frame - the reference frame - 'ecliptic' or 'earth'
#        start_date - the first date interval 'yyyy-mm-dd'
def load_mb(frame, start_date):
    const = constants()

    au = const.au  # km
    day = const.day  # seconds

    start_date = _normalize_epoch_date(start_date)

    print(f"\nEpoch: {start_date}")
    print(f"Reference frame: {frame.capitalize()} mean equator and equinox of J2000.0")

    # Exactly three network requests are submitted here; avoid oversubscribing threads.
    fetch_workers = max(1, int(os.getenv("MAJOR_BODIES_FETCH_WORKERS", "3")))
    with ThreadPoolExecutor(max_workers=min(3, fetch_workers)) as executor:
        future_sun = executor.submit(_load_state_for_body_cached, '10', frame, start_date, au, day)
        future_earth = executor.submit(_load_state_for_body_cached, '399', frame, start_date, au, day)
        future_moon = executor.submit(_load_state_for_body_cached, '301', frame, start_date, au, day)

        # .result() blocks until the specific thread is done
        x_sun, jd_sun = future_sun.result()
        x_earth, _ = future_earth.result()
        x_moon, _ = future_moon.result()

    jd = float(jd_sun)

    # Transfer the initial conditions to the state vector
    # Pre-allocate array
    Xb = np.zeros((3, 6), dtype=np.float64)

    # Sun
    Xb[0, :] = x_sun
    # Earth
    Xb[1, :] = x_earth
    # Moon
    Xb[2, :] = x_moon

    return Xb, jd


def _self_test_cache_roundtrip():
    # Lightweight self-test: validates cache path and output shape/dtype without network.
    import tempfile

    global _MB_DISK_CACHE_DIR

    class _FakeVec(dict):
        pass

    def _fake_vector(body, frame, start_date, max_retries=6, base_backoff_s=0.6):
        base = float(int(body))
        return _FakeVec({
            'x': np.array([base + 1.0]), 'y': np.array([base + 2.0]), 'z': np.array([base + 3.0]),
            'vx': np.array([base + 0.1]), 'vy': np.array([base + 0.2]), 'vz': np.array([base + 0.3]),
            'datetime_jd': np.array([2460310.5])
        })

    vec_backup = globals()['vector']
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _MB_DISK_CACHE_DIR = Path(tmp)
            globals()['vector'] = _fake_vector

            with _MB_MEM_CACHE_LOCK:
                _MB_MEM_CACHE.clear()

            xb0, jd0 = load_mb('earth', '2025-01-01')
            assert xb0.shape == (3, 6)
            assert xb0.dtype == np.float64
            assert np.isfinite(jd0)

            # Clear in-memory cache to force disk-cache path.
            with _MB_MEM_CACHE_LOCK:
                _MB_MEM_CACHE.clear()

            xb1, jd1 = load_mb('earth', '2025-01-01')
            assert xb1.shape == (3, 6)
            assert xb1.dtype == np.float64
            assert float(jd1) == float(jd0)
            assert np.allclose(xb0, xb1)
    finally:
        globals()['vector'] = vec_backup
    print("major_bodies_optimized self-test passed")

########################################################################
# Test the function
if __name__ == '__main__':
    if os.getenv("MAJOR_BODIES_SELF_TEST", "0") == "1":
        _self_test_cache_roundtrip()
        raise SystemExit(0)

    # Optimization: Defined constants locally for the test block to ensure independence
    au = 149597870.7  # km
    day = 86400.0  # seconds

    # Physical parameters of the major bodies
    Msun = 1.989e30  # kg
    Mearth = 5.972e24  # kg
    Mmoon = 7.34767309e22  # kg
    G = 6.67430e-20  # km^3/kg/s^2

    GMsun = G * Msun
    GMearth = G * Mearth
    GMmoon = G * Mmoon

    # Initial date (epoch - YYYY-MM-DD)
    epoch = '2025-01-01'

    # Note: In the __main__ block we still call 'vector' sequentially for the
    # individual print statements. This is fine for testing. The primary
    # optimization is in load_mb.

    # Earth
    body = '399'
    vec_earth = vector(body, 'earth', epoch)

    # Table header
    print(f"\nEpoch: {epoch}")
    print(vec_earth['datetime_jd'][0])

    print("\n")
    print("Earth (Central body)")
    X_earth = extract_state_vector(vec_earth, au, day)
    print(f"R_earth = {np.linalg.norm(X_earth[:3])} km")
    print(f"V_earth = {np.linalg.norm(X_earth[3:])} km/s")
    print(f"X_earth = {X_earth}")

    # Moon
    body = '301'
    vec_moon = vector(body, 'earth', epoch)

    print("\nMoon (State vector from JPL Horizons)")
    X_moon = extract_state_vector(vec_moon, au, day)
    print(f"R_moon = {np.linalg.norm(X_moon[:3])} km")
    print(f"V_moon = {np.linalg.norm(X_moon[3:])} km/s")
    print(f"X_moon = {X_moon}")

    oe_moon = np.zeros(6)
    mu = GMearth + GMmoon
    oe_moon = xyz2orb(mu, X_moon[:3], X_moon[3:])
    print("\nMoon (Orbital elements from state vector)")
    print(f"a = {oe_moon[0]} km")
    print(f"e = {oe_moon[1]}")
    print(f"i = {np.rad2deg(oe_moon[2])} deg")
    print(f"w = {np.rad2deg(oe_moon[3])} deg")
    print(f"Omega = {np.rad2deg(oe_moon[4])} deg")
    print(f"Me = {np.rad2deg(oe_moon[5])} deg")

    jpl_oe_moon = orbital_elements(body, 'earth', epoch)
    print("\nMoon from JPL Horizons")
    print(f"a = {jpl_oe_moon['a'][0]*au} km")
    print(f"e = {jpl_oe_moon['e'][0]}")
    print(f"i = {jpl_oe_moon['incl'][0]} deg")
    print(f"w = {jpl_oe_moon['w'][0]} deg")
    print(f"Omega = {jpl_oe_moon['Omega'][0]} deg")
    print(f"Me = {jpl_oe_moon['M'][0]} deg")

    # Sun
    print("\nSun (State vector from JPL Horizons)")
    body = '10'
    vec_sun = vector(body, 'earth', epoch)

    X_sun = extract_state_vector(vec_sun, au, day)
    print(f"R_sun = {np.linalg.norm(X_sun[:3])} km")
    print(f"V_sun = {np.linalg.norm(X_sun[3:])} km/s")
    print(f"X_sun = {X_sun}")

    oe_sun = np.zeros(6)
    mu = GMsun + GMearth
    oe_sun = xyz2orb(mu, X_sun[:3], X_sun[3:])
    print("\nSun (Orbital elements from state vector)")
    print(f"a = {oe_sun[0]} km")
    print(f"e = {oe_sun[1]}")
    print(f"i = {np.rad2deg(oe_sun[2])} deg")
    print(f"w = {np.rad2deg(oe_sun[3])} deg")
    print(f"Omega = {np.rad2deg(oe_sun[4])} deg")
    print(f"Me = {np.rad2deg(oe_sun[5])} deg")

    # Test the function load_mb
    # Check the structure of the returned value from load_mb
    print("\n--- Testing Parallel load_mb ---")
    Xb, jd = load_mb('earth', epoch)
    print(f"Structure of Xb: {Xb}")

    print("\nMajor bodies' initial conditions")
    # Sun
    print("\nSun")
    print(f"Xb_sun = {Xb[0, :]}")
    # Earth
    print("\nEarth")
    print(f"Xb_earth = {Xb[1, :]}")
    # Moon
    print("\nMoon")
    print(f"Xb_moon = {Xb[2, :]}")