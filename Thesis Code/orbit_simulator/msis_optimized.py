"""
Unified MSIS utilities for optimized_orbit_simulator.

This module consolidates functionality that was previously split across
multiple MSIS helper scripts into a single implementation.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numba import njit

# SW inputs for MSIS driver
@dataclass(frozen=True)
class SwDay:
    date: Date
    ap_bins: Tuple[float, float, float, float, float, float, float, float]
    ap_avg: float
    f107_obs: float
    f107a_center81: float

def _parse_date_iso_fast(s: str) -> Date:
    # Faster than strptime for fixed ISO format YYYY-MM-DD.
    return Date(int(s[0:4]), int(s[5:7]), int(s[8:10]))

def load_sw_csv(path: str | Path) -> Dict[Date, SwDay]:
    path = Path(path)
    out: Dict[Date, SwDay] = {}
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return out

        idx = {h.strip(): i for i, h in enumerate(header)}
        if "DATE" not in idx:
            raise ValueError("CSV missing DATE column")

        def _idx_or_missing(name: str) -> int:
            if name not in idx:
                raise ValueError(f"CSV missing required column: {name}")
            return int(idx[name])

        i_date = _idx_or_missing("DATE")
        i_ap_avg = _idx_or_missing("AP_AVG")
        i_f107_obs = _idx_or_missing("F10.7_OBS")
        i_f107a = _idx_or_missing("F10.7_OBS_CENTER81")
        i_ap = [_idx_or_missing(f"AP{k}") for k in range(1, 9)]

        for row in reader:
            if len(row) < len(header):
                continue

            ds = row[i_date].strip()
            if not ds:
                continue

            try:
                d = _parse_date_iso_fast(ds)
            except Exception:
                continue

            ap_avg_raw = row[i_ap_avg].strip()
            f107_obs_raw = row[i_f107_obs].strip()
            f107a_raw = row[i_f107a].strip()
            if (not ap_avg_raw) or (not f107_obs_raw) or (not f107a_raw):
                continue

            try:
                ap_avg = float(ap_avg_raw)
                f107_obs = float(f107_obs_raw)
                f107a = float(f107a_raw)
            except ValueError:
                continue

            ap_list = []
            malformed_ap = False
            for j in i_ap:
                ap_raw = row[j].strip()
                if not ap_raw:
                    ap_list.append(ap_avg)
                    continue
                try:
                    ap_list.append(float(ap_raw))
                except ValueError:
                    malformed_ap = True
                    break

            if malformed_ap:
                continue

            out[d] = SwDay(date=d, ap_bins=tuple(ap_list), ap_avg=ap_avg,
                           f107_obs=f107_obs, f107a_center81=f107a)
    return out

def _ap_for_bin(sw: Dict[Date, SwDay], t: datetime) -> float:
    day_obj = sw.get(t.date())
    if day_obj is None:
        raise KeyError(f"Missing SW row for {t.date()}")

    bin_idx = t.hour // 3
    return day_obj.ap_bins[bin_idx]

def build_ap_history(sw: Dict[Date, SwDay], t: datetime) -> Tuple[float, ...]:
    day_obj = sw.get(t.date())
    if day_obj is None:
        raise KeyError(f"Missing SW row for {t.date()}")

    ap1 = day_obj.ap_avg
    ap2 = _ap_for_bin(sw, t)
    ap3 = _ap_for_bin(sw, t - timedelta(hours=3))
    ap4 = _ap_for_bin(sw, t - timedelta(hours=6))
    ap5 = _ap_for_bin(sw, t - timedelta(hours=9))

    s6 = 0.0
    for h in range(12, 34, 3):
        s6 += _ap_for_bin(sw, t - timedelta(hours=h))
    ap6 = s6 * 0.125

    s7 = 0.0
    for h in range(36, 58, 3):
        s7 += _ap_for_bin(sw, t - timedelta(hours=h))
    ap7 = s7 * 0.125

    return (ap1, ap2, ap3, ap4, ap5, ap6, ap7)

def build_driver_inputs(sw: Dict[Date, SwDay], start_date: Date,
                        end_date_inclusive: Date, utsecs: List[int]) -> List[tuple]:
    rows = []
    d = start_date
    one_day = timedelta(days=1)
    ut_deltas = [timedelta(seconds=u) for u in utsecs]

    while d <= end_date_inclusive:
        today = sw.get(d)
        prev = sw.get(d - one_day)

        if not today or not prev:
            raise KeyError(f"Missing SW data around {d}")

        sfluxavg = today.f107a_center81
        sflux_prev = prev.f107_obs
        doy = d.timetuple().tm_yday
        yyyymmdd = d.year * 10000 + d.month * 100 + d.day

        dt_base = datetime(d.year, d.month, d.day)

        for i, ut in enumerate(utsecs):
            t = dt_base + ut_deltas[i]
            ap = build_ap_history(sw, t)
            rows.append((yyyymmdd, doy, ut, sfluxavg, sflux_prev, ap))

        d += one_day

    return rows

def msis_sw_inputs_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sw", required=False)
    p.add_argument("--start", required=False)
    p.add_argument("--end", required=False)
    p.add_argument("--utsecs", default="0,21600,43200,64800")
    p.add_argument("--out", required=False)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        msis_sw_inputs_self_test()
        return 0

    if not args.sw or not args.start or not args.end or not args.out:
        raise ValueError("--sw, --start, --end, and --out are required unless --self-test is used")

    sw = load_sw_csv(args.sw)
    start = _parse_date_iso_fast(args.start)
    end = _parse_date_iso_fast(args.end)
    utsecs = [int(x) for x in args.utsecs.split(",") if x.strip()]

    rows = build_driver_inputs(sw, start, end, utsecs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["yyyymmdd", "doy", "utsec", "sfluxavg", "sflux",
                     "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], *r[5]])

    return 0

def msis_sw_inputs_self_test() -> None:
    import tempfile

    csv_text = ( "DATE,AP_AVG,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,F10.7_OBS,F10.7_OBS_CENTER81\n"
                "2024-12-30,9,1,2,3,4,5,6,7,8,119,114\n"
                "2024-12-31,9.5,1,2,3,4,5,6,7,8,119.5,114.5\n"
                "2025-01-01,10,1,2,3,4,5,6,7,8,120,115\n"
                "2025-01-02,11,1,2,3,4,5,6,7,8,121,116\n"
                "2025-01-03,12,1,2,3,4,5,6,7,8,122,117\n")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sw.csv"
        p.write_text(csv_text, encoding="utf-8")
        sw = load_sw_csv(p)
        assert len(sw) == 5
        rows = build_driver_inputs(sw, Date(2025, 1, 2), Date(2025, 1, 2), [0])
        assert len(rows) == 1
        ap = rows[0][5]
        assert len(ap) == 7
        assert ap[0] == 11.0
    print("msis_sw_inputs self-test passed")

# Grid pack/compression
@dataclass(frozen=True)
class GridSpec:
    lat0: float = -90.0
    dlat: float = 5.0
    nlat: int = 37

    lon0: float = 0.0
    dlon: float = 5.0
    nlon: int = 72

    alt_min_km: float = 115.0
    alt_step_km: float = 5.0
    nz: int = 98

    @property
    def alt_max_km(self) -> float:
        return self.alt_min_km + self.alt_step_km * (self.nz - 1)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nlat, self.nlon, self.nz)

_PACK_PAT = re.compile(r"^rho_(\d{8})_ut(\d{2})_f32\.bin$")

def _iter_raw_bins(grid_dir: Path) -> Iterable[Tuple[str, int, Path]]:
    for p in sorted(grid_dir.glob("rho_*_ut*_f32.bin")):
        m = _PACK_PAT.match(p.name)
        if not m:
            continue
        yyyymmdd = m.group(1)
        hh = int(m.group(2))
        utsec = hh * 3600
        date_key = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
        yield date_key, utsec, p

def _compress_one_raw_to_zst(src: Path, dst: Path, level: int) -> None:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'zstandard'. Install with: pip install zstandard") from exc

    # Use a temp file so partial writes do not leave a truncated destination.
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    cctx = zstd.ZstdCompressor(level=level, write_content_size=True)
    with src.open("rb") as fin, tmp.open("wb") as fout:
        cctx.copy_stream(fin, fout)
    tmp.replace(dst)

def compress_worker(args: Tuple[Path, Path, int]) -> None:
    src, dst, level = args
    _compress_one_raw_to_zst(src, dst, int(level))

def validate_raw_file(p: Path, spec: GridSpec) -> None:
    expected_bytes = spec.nlat * spec.nlon * spec.nz * 4
    if p.stat().st_size != expected_bytes:
        raise ValueError(f"Size mismatch {p.name}: {p.stat().st_size} != {expected_bytes}")

def write_meta(grid_dir: Path, spec: GridSpec, utsecs: List[int]) -> None:
    meta_path = grid_dir / "grid_meta.txt"
    lines = [f"nlat={spec.nlat}",
             f"nlon={spec.nlon}",
             f"nz={spec.nz}",
             f"lat0={spec.lat0}",
             f"dlat={spec.dlat}",
             f"lon0={spec.lon0}",
             f"dlon={spec.dlon}",
             f"alt_min_km={spec.alt_min_km}",
             f"alt_max_km={spec.alt_max_km}",
             f"alt_step_km={spec.alt_step_km}",
             f"utsecs={','.join(str(int(u)) for u in utsecs)}",
             "utsec=0"]
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_index(grid_dir: Path, entries: List[Tuple[str, int, Path]]) -> None:
    index_path = grid_dir / "grid_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "utsec", "file"])
        for date_key, utsec, path in entries:
            w.writerow([date_key, int(utsec), path.name])

def pack_msis_grid(grid_dir: str | Path, level: int = 10, keep_raw: bool = False,
                   workers: int | None = None, chunksize: int = 8) -> int:
    grid_dir = Path(grid_dir)
    spec = GridSpec()

    tasks = []
    entries_out: List[Tuple[str, int, Path]] = []
    utsecs_seen = set()
    raw_files_to_remove = []

    t0 = time.perf_counter()
    for date_key, utsec, raw in _iter_raw_bins(grid_dir):
        validate_raw_file(raw, spec)
        zst = raw.with_suffix(raw.suffix + ".zst")

        tasks.append((raw, zst, int(level)))
        entries_out.append((date_key, utsec, zst))
        utsecs_seen.add(int(utsec))

        if not keep_raw:
            raw_files_to_remove.append(raw)
    t_scan = time.perf_counter() - t0

    if not entries_out:
        raise RuntimeError(f"No matching raw grid files found under {grid_dir}")

    t1 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(compress_worker, tasks, chunksize=max(1, int(chunksize))))
    t_compress = time.perf_counter() - t1

    t2 = time.perf_counter()
    if not keep_raw:
        for p in raw_files_to_remove:
            p.unlink(missing_ok=True)

    utsecs = sorted(utsecs_seen)
    write_meta(grid_dir, spec, utsecs)
    write_index(grid_dir, entries_out)
    t_finalize = time.perf_counter() - t2

    print(f"Timing | scan+validate: {t_scan:.2f} s")
    print(f"Timing | compress: {t_compress:.2f} s")
    print(f"Timing | finalize/index: {t_finalize:.2f} s")
    print("Done.")
    return 0

def pack_msis_grid_streaming(grid_dir: str | Path, level: int = 10, keep_raw: bool = False,
                             progress_every: int = 250) -> int:
    """Compress raw grids sequentially and optionally delete each .bin immediately."""
    grid_dir = Path(grid_dir)
    spec = GridSpec()

    entries_out: List[Tuple[str, int, Path]] = []
    utsecs_seen = set()
    processed = 0

    t0 = time.perf_counter()
    for date_key, utsec, raw in _iter_raw_bins(grid_dir):
        validate_raw_file(raw, spec)
        zst = raw.with_suffix(raw.suffix + ".zst")
        _compress_one_raw_to_zst(raw, zst, int(level))

        entries_out.append((date_key, utsec, zst))
        utsecs_seen.add(int(utsec))
        processed += 1

        if not keep_raw:
            raw.unlink(missing_ok=True)

        if progress_every > 0 and processed % int(progress_every) == 0:
            print(f"Progress | compressed {processed} files")

    if not entries_out:
        raise RuntimeError(f"No matching raw grid files found under {grid_dir}")

    utsecs = sorted(utsecs_seen)
    write_meta(grid_dir, spec, utsecs)
    write_index(grid_dir, entries_out)

    t_total = time.perf_counter() - t0
    print(f"Timing | streaming pack total: {t_total:.2f} s")
    print("Done.")
    return 0

def msis_grid_pack_main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compress MSIS grids")
    ap.add_argument("--grid-dir", required=False, help="Directory containing rho_*.bin files")
    ap.add_argument("--level", type=int, default=10, help="zstd compression level")
    ap.add_argument("--keep-raw", action="store_true", help="Keep uncompressed .bin files")
    ap.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    ap.add_argument("--chunksize", type=int, default=8, help="Executor map chunksize for compression tasks")
    ap.add_argument("--stream-delete", action="store_true",
                    help="Compress sequentially and delete each raw .bin immediately to reduce peak disk usage")
    ap.add_argument("--self-test", action="store_true", help="Run module self-test and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        msis_grid_pack_self_test()
        return 0

    if not args.grid_dir:
        raise ValueError("--grid-dir is required unless --self-test is used")

    print(f"Scanning {Path(args.grid_dir)}...")
    mode = "streaming" if args.stream_delete else "parallel"
    print(f"Compression level: {int(args.level)} | mode: {mode} | chunksize: {int(args.chunksize)}")

    if args.stream_delete:
        if args.workers not in (None, 1):
            print("Note: --workers is ignored when --stream-delete is enabled.")
        return pack_msis_grid_streaming(args.grid_dir, level=args.level, keep_raw=args.keep_raw)

    return pack_msis_grid(args.grid_dir, level=args.level, keep_raw=args.keep_raw,
                          workers=args.workers, chunksize=args.chunksize)

def msis_grid_pack_self_test() -> None:
    import tempfile

    try:
        import zstandard  # noqa: F401
    except ImportError:
        print("msis_grid_pack self-test skipped (zstandard not installed)")
        return

    with tempfile.TemporaryDirectory() as td:
        grid_dir = Path(td)
        spec = GridSpec()
        arr = np.zeros(spec.shape, dtype=np.float32)
        raw = grid_dir / "rho_20250101_ut00_f32.bin"
        arr.tofile(raw)

        rc = pack_msis_grid(grid_dir, level=3, keep_raw=True, workers=1, chunksize=1)
        assert rc == 0

        meta = (grid_dir / "grid_meta.txt").read_text(encoding="utf-8")
        assert "nlat=" in meta and "utsecs=" in meta

        idx = (grid_dir / "grid_index.csv").read_text(encoding="utf-8")
        assert "date,utsec,file" in idx
        assert "rho_20250101_ut00_f32.bin.zst" in idx

    print("msis_grid_pack self-test passed")

# Precomputed grid loading
_ZSTD_DCTX_LOCK = threading.Lock()
_ZSTD_DCTX = None
_ZSTD_DCTX_LOCAL = threading.local()  # thread-local decompressor for safe concurrent use

@dataclass(frozen=True)
class MsisGridMeta:
    nlat: int
    nlon: int
    nz: int
    lat0: float
    dlat: float
    lon0: float
    dlon: float
    alt_min_km: float
    alt_max_km: float
    alt_step_km: float
    utsec: float
    utsecs: Tuple[int, ...] = ()

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nlat, self.nlon, self.nz)

def parse_date(date_str: str) -> Date:
    """Compatibility helper: parse YYYY-MM-DD into datetime.date."""
    return Date.fromisoformat(date_str)

def date_add_days(d: Date, n_days: int) -> Date:
    """Compatibility helper: add integer days to a datetime.date."""
    return d + timedelta(days=int(n_days))

def load_meta(grid_dir: str | Path) -> MsisGridMeta:
    grid_dir = Path(grid_dir)
    meta_path = grid_dir / "grid_meta.txt"
    txt = meta_path.read_text(encoding="utf-8")

    kv = {}
    for line in txt.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    nlat = int(kv["nlat"])
    nlon = int(kv["nlon"])
    alt_min = float(kv["alt_min_km"])
    alt_max = float(kv["alt_max_km"])
    alt_step = float(kv["alt_step_km"])
    utsec = float(kv.get("utsec", "0"))

    uts_str = kv.get("utsecs", "")
    if uts_str:
        utsecs = tuple(int(x) for x in uts_str.split(",") if x.strip())
    else:
        utsecs = ()

    if nlat <= 0 or nlon <= 0:
        raise ValueError(f"Invalid grid dimensions in {meta_path}: nlat={nlat}, nlon={nlon}")
    if not np.isfinite(alt_step) or alt_step <= 0.0:
        raise ValueError(f"Invalid alt_step_km in {meta_path}: {alt_step}")
    if not np.isfinite(alt_max - alt_min) or alt_max < alt_min:
        raise ValueError(f"Invalid altitude bounds in {meta_path}: [{alt_min}, {alt_max}]")

    nz = int(kv.get("nz", round((alt_max - alt_min) / alt_step) + 1))
    if nz <= 0:
        raise ValueError(f"Invalid nz in {meta_path}: {nz}")

    return MsisGridMeta(nlat=nlat, nlon=nlon, nz=nz, lat0=float(kv["lat0"]),
                        dlat=float(kv["dlat"]), lon0=float(kv["lon0"]),
                        dlon=float(kv["dlon"]), alt_min_km=alt_min,
                        alt_max_km=alt_max, alt_step_km=alt_step,
                        utsec=utsec, utsecs=utsecs)

_RE_GRID_FILE = re.compile(r"^rho_(\d{8})_ut(\d{2})")

class MsisGridIndex:
    def __init__(self, grid_dir: str | Path):
        self.grid_dir = Path(grid_dir)
        index_path = self.grid_dir / "grid_index.csv"
        self._map_date_ut: Dict[Tuple[str, int], Path] = {}
        self._map_date: Dict[str, Path] = {}

        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                has_ut = "utsec" in (reader.fieldnames or [])
                has_date = "date" in (reader.fieldnames or [])
                has_file = "file" in (reader.fieldnames or [])
                if not has_date or not has_file:
                    raise ValueError(f"Malformed grid index header in {index_path}; expected date,file[,utsec]")

                for row in reader:
                    d = row.get("date", "").strip()
                    p_str = row.get("file", "").strip()
                    if not d or not p_str:
                        continue

                    ut = int(float(row.get("utsec", "0"))) if has_ut else 0

                    path = Path(p_str)
                    if not path.is_absolute():
                        cand = self.grid_dir / path
                        path = cand if cand.exists() else self.grid_dir / path.name

                    if has_ut:
                        self._map_date_ut[(d, ut)] = path
                    else:
                        self._map_date[d] = path
        else:
            for p in sorted(self.grid_dir.glob("rho_*")):
                if not p.is_file():
                    continue
                m = _RE_GRID_FILE.match(p.name)
                if m:
                    ymd = m.group(1)
                    ut = int(m.group(2)) * 3600
                    k = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
                    self._map_date_ut[(k, ut)] = p

    def path_for_date_ut(self, dt: Date, utsec: int) -> Path:
        k = dt.strftime("%Y-%m-%d")
        return self._map_date_ut.get((k, utsec)) or self._map_date.get(k) or _raise(
            KeyError(f"No grid for {k} ut={utsec}"))

def _raise(e):
    raise e

def _expected_grid_counts(meta: MsisGridMeta):
    n = int(meta.nlat) * int(meta.nlon) * int(meta.nz)
    if n <= 0:
        raise ValueError(f"Invalid grid shape from metadata: {meta.shape}")
    return n, n * np.dtype(np.float32).itemsize

def _get_reusable_zstd_decompressor():
    """Return a per-thread ZstdDecompressor (thread-safe for concurrent streaming)."""
    dctx = getattr(_ZSTD_DCTX_LOCAL, "dctx", None)
    if dctx is not None:
        return dctx

    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    _ZSTD_DCTX_LOCAL.dctx = dctx
    return dctx

def memmap_grid(path: str | Path, meta: MsisGridMeta) -> np.ndarray:
    """Compatibility alias for legacy callers; supports raw and .zst grids."""
    return load_grid(path, meta)

def load_grid(path: str | Path, meta: MsisGridMeta) -> np.ndarray:
    path = Path(path)
    count, expected_bytes = _expected_grid_counts(meta)

    if path.suffix == ".zst":
        dctx = _get_reusable_zstd_decompressor()

        arr = np.empty(count, dtype=np.float32)
        out_mv = memoryview(arr).cast("B")
        written = 0

        with path.open("rb") as fin:
            with dctx.stream_reader(fin) as reader:
                while written < expected_bytes:
                    n_read = reader.readinto(out_mv[written:])
                    if n_read is None or n_read <= 0:
                        break
                    written += int(n_read)

        if written != expected_bytes:
            raise ValueError(f"Decompressed mismatch for {path.name}: got {written} bytes, expected {expected_bytes} "
                             f"bytes for shape {meta.shape}")

        out = arr.reshape(meta.shape)
        if out.dtype != np.float32 or not out.flags.c_contiguous:
            out = np.ascontiguousarray(out, dtype=np.float32)
        return out

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(f"Cannot open grid file {path}") from exc

    if file_size != expected_bytes:
        raise ValueError(f"Raw grid mismatch for {path.name}: got {file_size} bytes, expected {expected_bytes} "
                         f"bytes for shape {meta.shape}")

    mm = np.memmap(path, dtype=np.float32, mode="r", shape=meta.shape)
    if mm.dtype != np.float32 or not mm.flags.c_contiguous:
        return np.ascontiguousarray(mm, dtype=np.float32)
    return mm

def msis_precomputed_self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        grid_dir = Path(td)
        (grid_dir / "grid_meta.txt").write_text("\n".join(["nlat=2", "nlon=3", "nz=4", "lat0=-90",
                                                           "dlat=90", "lon0=0", "dlon=120",
                                                           "alt_min_km=115", "alt_max_km=130",
                                                           "alt_step_km=5", "utsecs=0,21600",
                                                           "utsec=0"]) + "\n", encoding="utf-8")
        (grid_dir / "grid_index.csv").write_text("date,utsec,file\n2025-01-01,0,rho_20250101_ut00_f32.bin\n",
                                                 encoding="utf-8")

        meta = load_meta(grid_dir)
        assert meta.shape == (2, 3, 4)
        idx = MsisGridIndex(grid_dir)

        raw = np.arange(24, dtype=np.float32).reshape(meta.shape)
        raw_path = grid_dir / "rho_20250101_ut00_f32.bin"
        raw.tofile(raw_path)

        p = idx.path_for_date_ut(Date(2025, 1, 1), 0)
        out_raw = load_grid(p, meta)
        assert out_raw.shape == meta.shape
        assert out_raw.dtype == np.float32
        assert np.allclose(np.asarray(out_raw), raw)
        if isinstance(out_raw, np.memmap):
            out_raw._mmap.close()
        del out_raw

        try:
            import zstandard as zstd

            zst_path = raw_path.with_suffix(raw_path.suffix + ".zst")
            cctx = zstd.ZstdCompressor(level=3, write_content_size=True)
            with raw_path.open("rb") as fin, zst_path.open("wb") as fout:
                cctx.copy_stream(fin, fout)
            out_zst = load_grid(zst_path, meta)
            assert out_zst.shape == meta.shape
            assert out_zst.dtype == np.float32
            assert out_zst.flags.c_contiguous
            assert np.allclose(out_zst, raw)
        except ImportError:
            pass

    print("msis_precomputed self-test passed")

# Numba drag interpolation on precomputed grids
@njit(cache=True, inline="always")
def _wrap_lon_deg(lon_deg: float) -> float:
    return lon_deg % 360.0

@njit(cache=True, inline="always")
def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

@njit(cache=True, nogil=True)
def _interp_trilinear_dual(grid0: np.ndarray, grid1: np.ndarray, mix: float, lat_deg: float,
                           lon_deg: float, alt_km: float, lat0: float, dlat: float,
                           lon0: float, dlon: float, alt_min_km: float, alt_step_km: float) -> float:
    nlat, nlon, nz = grid0.shape

    lat_max = lat0 + dlat * (nlat - 1)
    if lat_deg < lat0:
        lat_deg = lat0
    elif lat_deg > lat_max:
        lat_deg = lat_max

    alt_max_km = alt_min_km + alt_step_km * (nz - 1)
    if alt_km < alt_min_km or alt_km > alt_max_km:
        return 0.0

    lon_deg = lon_deg % 360.0

    ilat_f = (lat_deg - lat0) / dlat
    ilon_f = (lon_deg - lon0) / dlon
    iz_f = (alt_km - alt_min_km) / alt_step_km

    ilat0 = int(ilat_f)
    iz0 = int(iz_f)

    if ilat0 >= nlat - 1:
        ilat0 = nlat - 2
    if iz0 >= nz - 1:
        iz0 = nz - 2

    ilon0 = int(np.floor(ilon_f))

    wlat = ilat_f - ilat0
    wlon = ilon_f - np.floor(ilon_f)
    wz = iz_f - iz0

    ilat1 = ilat0 + 1
    iz1 = iz0 + 1
    j0 = ilon0 % nlon
    j1 = (ilon0 + 1) % nlon

    c000 = float(grid0[ilat0, j0, iz0])
    c001 = float(grid0[ilat0, j0, iz1])
    c010 = float(grid0[ilat0, j1, iz0])
    c011 = float(grid0[ilat0, j1, iz1])
    c100 = float(grid0[ilat1, j0, iz0])
    c101 = float(grid0[ilat1, j0, iz1])
    c110 = float(grid0[ilat1, j1, iz0])
    c111 = float(grid0[ilat1, j1, iz1])

    d000 = float(grid1[ilat0, j0, iz0])
    d001 = float(grid1[ilat0, j0, iz1])
    d010 = float(grid1[ilat0, j1, iz0])
    d011 = float(grid1[ilat0, j1, iz1])
    d100 = float(grid1[ilat1, j0, iz0])
    d101 = float(grid1[ilat1, j0, iz1])
    d110 = float(grid1[ilat1, j1, iz0])
    d111 = float(grid1[ilat1, j1, iz1])

    c00 = c000 + (c010 - c000) * wlon
    c01 = c001 + (c011 - c001) * wlon
    c10 = c100 + (c110 - c100) * wlon
    c11 = c101 + (c111 - c101) * wlon

    d00 = d000 + (d010 - d000) * wlon
    d01 = d001 + (d011 - d001) * wlon
    d10 = d100 + (d110 - d100) * wlon
    d11 = d101 + (d111 - d101) * wlon

    c0 = c00 + (c01 - c00) * wz
    c1 = c10 + (c11 - c10) * wz

    d0 = d00 + (d01 - d00) * wz
    d1 = d10 + (d11 - d10) * wz

    val0 = c0 + (c1 - c0) * wlat
    val1 = d0 + (d1 - d0) * wlat

    return val0 + (val1 - val0) * mix

@njit(cache=True, nogil=True)
def ecef_from_eci_zrot(x_eci: float, y_eci: float, z_eci: float, theta: float) -> tuple[float, float, float]:
    c = np.cos(theta)
    s = np.sin(theta)
    return (x_eci * c + y_eci * s, -x_eci * s + y_eci * c, z_eci)

@njit(cache=True, nogil=True)
def geodetic_lat_lon_alt_wgs84_km(x: float, y: float, z: float) -> tuple[float, float, float]:
    a = 6378.137
    e2 = 6.69437999014e-3

    p2 = x * x + y * y
    p = np.sqrt(p2)
    lon = np.arctan2(y, x)

    if p < 1e-12:
        return (90.0 if z >= 0 else -90.0, _wrap_lon_deg(np.degrees(lon)), np.abs(z) - 6356.7523)

    lat = np.arctan2(z, p * (1.0 - e2))
    alt = 0.0

    for _ in range(4):
        s = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * s * s)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + alt)))

    return (np.degrees(lat), _wrap_lon_deg(np.degrees(lon)), alt)

@njit(cache=True, nogil=True)
def _atm_drag_msis_grid_daysec(Xsat: np.ndarray, Cd: float, AtoM: float, earth_spin: float,
                               GST0: float, t_abs: float, sec_in_day: float, grid_ut00: np.ndarray,
                               grid_ut06: np.ndarray, grid_ut12: np.ndarray, grid_ut18: np.ndarray,
                               grid_tomorrow_ut00: np.ndarray, earth_Re: float, lat0: float,
                               dlat: float, lon0: float, dlon: float, alt_min_km: float,
                               alt_step_km: float, out: np.ndarray) -> None:
    theta = GST0 + earth_spin * t_abs
    rx, ry, rz = Xsat[0], Xsat[1], Xsat[2]
    vx, vy, vz = Xsat[3], Xsat[4], Xsat[5]

    x_ecef, y_ecef, z_ecef = ecef_from_eci_zrot(rx, ry, rz, theta)
    lat_deg, lon_deg, alt_km = geodetic_lat_lon_alt_wgs84_km(x_ecef, y_ecef, z_ecef)

    if sec_in_day < 21600.0:
        g0, g1 = grid_ut00, grid_ut06
        w = sec_in_day * 4.62962962962963e-5
    elif sec_in_day < 43200.0:
        g0, g1 = grid_ut06, grid_ut12
        w = (sec_in_day - 21600.0) * 4.62962962962963e-5
    elif sec_in_day < 64800.0:
        g0, g1 = grid_ut12, grid_ut18
        w = (sec_in_day - 43200.0) * 4.62962962962963e-5
    else:
        g0, g1 = grid_ut18, grid_tomorrow_ut00
        w = (sec_in_day - 64800.0) * 4.62962962962963e-5

    rho = _interp_trilinear_dual(g0, g1, w, lat_deg, lon_deg, alt_km, lat0, dlat,
                                 lon0, dlon, alt_min_km, alt_step_km)

    rho_km3 = rho * 1.0e9

    vrel0 = vx - (-earth_spin * ry)
    vrel1 = vy - (earth_spin * rx)
    vrel2 = vz

    vrel = np.sqrt(vrel0 * vrel0 + vrel1 * vrel1 + vrel2 * vrel2)
    fac = -0.5 * rho_km3 * Cd * (AtoM * 1e-6) * vrel

    out[0] = fac * vrel0
    out[1] = fac * vrel1
    out[2] = fac * vrel2

@njit(cache=True, nogil=True)
def atm_drag_msis_grid(Xsat: np.ndarray, Cd: float, AtoM: float, earth_spin: float, GST0: float,
                       t_abs: float, grid_ut00: np.ndarray, grid_ut06: np.ndarray,
                       grid_ut12: np.ndarray, grid_ut18: np.ndarray, grid_tomorrow_ut00: np.ndarray,
                       earth_Re: float, lat0: float, dlat: float, lon0: float, dlon: float,
                       alt_min_km: float, alt_step_km: float, out: np.ndarray) -> None:
    sec_in_day = t_abs % 86400.0
    _atm_drag_msis_grid_daysec(Xsat, Cd, AtoM, earth_spin, GST0, t_abs, sec_in_day, grid_ut00,
                               grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00, earth_Re,
                               lat0, dlat, lon0, dlon, alt_min_km, alt_step_km, out)

def main(argv: Optional[List[str]] = None) -> int:
    """
    Unified CLI for consolidated MSIS utilities.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if len(args) == 0:
        print("Usage: msis_optimized.py <command> [options]\n"
              "Commands:\n"
              "  sw-inputs            Build MSIS driver inputs from SW CSV\n"
              "  pack                 Compress raw rho_*.bin grids and write meta/index\n"
              "  self-test-sw         Run SW input builder self-test\n"
              "  self-test-pack       Run grid packer self-test\n"
              "  self-test-precomputed Run precomputed grid loader self-test")
        return 2

    command = args[0].strip().lower()
    rest = args[1:]

    if command == "sw-inputs":
        return msis_sw_inputs_main(rest)
    if command == "pack":
        return msis_grid_pack_main(rest)
    if command == "self-test-sw":
        msis_sw_inputs_self_test()
        return 0
    if command == "self-test-pack":
        msis_grid_pack_self_test()
        return 0
    if command == "self-test-precomputed":
        msis_precomputed_self_test()
        return 0

    raise ValueError(f"Unknown command '{command}'. Use sw-inputs, pack, or self-test-* commands.")

__all__ = ["SwDay", "_parse_date_iso_fast", "load_sw_csv", "_ap_for_bin", "build_ap_history",
           "build_driver_inputs", "msis_sw_inputs_main", "msis_sw_inputs_self_test", "GridSpec",
           "_iter_raw_bins", "compress_worker", "validate_raw_file", "write_meta", "write_index",
           "pack_msis_grid", "pack_msis_grid_streaming", "msis_grid_pack_main", "msis_grid_pack_self_test", "MsisGridMeta",
           "parse_date", "date_add_days", "load_meta", "MsisGridIndex", "memmap_grid", "load_grid",
           "msis_precomputed_self_test", "_wrap_lon_deg", "_clamp", "_interp_trilinear_dual",
           "ecef_from_eci_zrot", "geodetic_lat_lon_alt_wgs84_km", "_atm_drag_msis_grid_daysec",
           "atm_drag_msis_grid", "main"]

if __name__ == "__main__":
    raise SystemExit(main())