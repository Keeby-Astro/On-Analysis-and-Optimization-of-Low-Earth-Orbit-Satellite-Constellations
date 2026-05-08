from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path


def main() -> int:
    grid_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("full_out")
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'zstandard'. Install with: pip install zstandard") from exc

    if not grid_dir.is_dir():
        raise FileNotFoundError(f"Grid directory not found: {grid_dir}")

    pat = re.compile(r"^rho_(\d{8})_ut(\d{2})_f32\.bin$")
    nlat, nlon, nz = 37, 72, 98
    expected_bytes = nlat * nlon * nz * 4

    bins = sorted(grid_dir.glob("rho_*_ut*_f32.bin"))
    if not bins:
        raise RuntimeError(f"No matching raw grid files found under {grid_dir}")

    cctx = zstd.ZstdCompressor(level=level, write_content_size=True)
    entries: list[tuple[str, int, str]] = []
    utsecs_seen: set[int] = set()

    t0 = time.perf_counter()
    processed = 0
    for p in bins:
        m = pat.match(p.name)
        if not m:
            continue

        size = p.stat().st_size
        if size != expected_bytes:
            raise ValueError(f"Size mismatch {p.name}: {size} != {expected_bytes}")

        yyyymmdd = m.group(1)
        hh = int(m.group(2))
        utsec = hh * 3600
        date_key = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

        zst = p.with_suffix(p.suffix + ".zst")
        tmp = zst.with_suffix(zst.suffix + ".tmp")
        with p.open("rb") as fin, tmp.open("wb") as fout:
            cctx.copy_stream(fin, fout)
        tmp.replace(zst)

        # Delete each raw bin as soon as its zst is written.
        p.unlink(missing_ok=True)

        entries.append((date_key, utsec, zst.name))
        utsecs_seen.add(utsec)
        processed += 1

        if processed % 250 == 0:
            print(f"Progress | compressed {processed} files")

    if not entries:
        raise RuntimeError(f"No valid rho_*.bin files found under {grid_dir}")

    utsecs = sorted(utsecs_seen)

    meta_lines = [
        f"nlat={nlat}",
        f"nlon={nlon}",
        f"nz={nz}",
        "lat0=-90.0",
        "dlat=5.0",
        "lon0=0.0",
        "dlon=5.0",
        "alt_min_km=115.0",
        "alt_max_km=600.0",
        "alt_step_km=5.0",
        f"utsecs={','.join(str(int(u)) for u in utsecs)}",
        "utsec=0",
    ]
    (grid_dir / "grid_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    with (grid_dir / "grid_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "utsec", "file"])
        for date_key, utsec, fname in entries:
            w.writerow([date_key, int(utsec), fname])

    dt = time.perf_counter() - t0
    print(f"Done | files={processed} | elapsed_s={dt:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
