# Precomputed NRLMSIS grids (dynamic atmosphere)

This folder supports two drag atmosphere options in the orbit simulator:

- `atm_model = 0`: USSA76 (static; current default)
- `atm_model = 1`: NRLMSIS grids (dynamic; driven by SW-Last5Years)

The NRLMSIS path uses **geodetic (WGS84)** latitude and altitude end-to-end.

For the `hall_thruster` workflow, use `full_out` for MSIS grids/metadata so `out_grid` stays isolated for other projects.

## Grid spec (current)

- Geodetic latitude: -90..90 deg, step 5 deg  (nlat=37)
- Longitude: 0..355 deg, step 5 deg (nlon=72)
- Geodetic altitude: 115..600 km, step 5 km (nz=98)
- Time snapshots: 00/06/12/18 UT (4 grids per day)
- Stored value: total mass density `rho` in **kg/m^3** (float32)

## Consolidated tooling

MSIS helper code is now consolidated into one module:

- `hall_thruster/msis_optimized.py`

## 1) Build driver inputs from SW-Last5Years.csv

Generates a compact CSV that the Fortran grid generator can read.

```powershell
py hall_thruster/msis_optimized.py sw-inputs `
  --sw SW-Last5Years.csv `
  --start 2023-07-01 `
  --end 2026-01-01 `
  --utsecs 0,21600,43200,64800 `
  --out full_out/driver_inputs.csv
```

Notes:
- Uses `F10.7_OBS_CENTER81` for `sfluxavg`.
- Uses previous day's `F10.7_OBS` for `sflux` (as required by MSIS).
- Constructs storm-time `ap(1:7)` from `AP1..AP8` and prior-day history.

## 2) Compile the grid generator (Fortran)

The MSIS readme shows how to compile on Windows with `gfortran`. This adds one extra source file:

```powershell
gfortran -O3 -cpp -o msis_grid_sw_driver.exe `
  alt2gph.F90 msis_constants.F90 msis_init.F90 msis_gfn.F90 msis_tfn.F90 msis_dfn.F90 msis_calc.F90 `
  msis_grid_sw_driver.F90
```

Optional:
- Add `-DDBLE` for double precision internally.

## 3) Generate raw grids

```powershell
./msis_grid_sw_driver.exe full_out/driver_inputs.csv full_out msis20.parm
```

This writes many raw float32 files:

- `rho_YYYYMMDD_utHH_f32.bin`

## 4) Compress + write meta/index

Requires zstd Python package:

```powershell
py -m pip install zstandard
```

Pack/compress:

```powershell
py hall_thruster/msis_optimized.py pack --grid-dir full_out
```

This produces:
- `rho_YYYYMMDD_utHH_f32.bin.zst`
- `grid_meta.txt`
- `grid_index.csv`

By default it deletes the raw `.bin` files after packing. Add `--keep-raw` to keep them.

## 5) Run the orbit simulator with MSIS grids

In `hall_thruster/constellation_simulator_optimized.py` set:

- `atm_model = 1`
- `msis_grid_dir = r'...\full_out'`
- `msis_grid_start_date = '2023-07-01'`

The simulator will load the needed UT grids for each day segment and compute drag using:
- WGS84 geodetic latitude + altitude (ECEF -> geodetic conversion)
- spatial trilinear interpolation (lat/lon/alt)
- linear interpolation between UT snapshots (00/06/12/18), including 18->24 using tomorrow 00

## File format compatibility

The runtime supports:
- raw `.bin` (memmap) and
- `.zst` compressed grids.

`grid_meta.txt` and `grid_index.csv` are the authoritative mapping.