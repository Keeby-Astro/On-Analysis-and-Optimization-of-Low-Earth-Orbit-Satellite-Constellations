import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from array import array

from semi_major_axis import semi_major_axis_vector
from mean_to_true_anomaly import mean_to_true_anomaly_vector
from specific_angular_momentum import specific_angular_momentum_vector

# Utility for parsing the drag term string in TLE line 1
def parse_drag_term(drag_term_str):
    """Parse TLE B* drag term (fixed-width mantissa/exponent, no explicit 'E').

    Typical examples (after stripping):
      '40221-3' -> 0.40221e-3
      '17042-4' -> 0.17042e-4
      '-10681-3' -> -0.10681e-3
      '00000-0' -> 0.0
    """
    s = str(drag_term_str).strip()
    if not s:
        return np.nan

    # Fall back to float() for unexpected formats.
    if len(s) < 7:
        try:
            return float(s)
        except Exception:
            return np.nan

    exp_sign = s[-2]
    exp_digit = s[-1]
    mantissa = s[:-2]

    sign = ''
    if mantissa and mantissa[0] in '+-':
        sign = mantissa[0]
        mant_digits = mantissa[1:]
    else:
        mant_digits = mantissa

    # Mantissa is typically 5 digits (no decimal point). Keep it robust.
    mant_digits = mant_digits.replace('.', '').strip()
    if not mant_digits:
        return np.nan

    try:
        return float(f"{sign}0.{mant_digits}e{exp_sign}{exp_digit}")
    except Exception:
        try:
            return float(s)
        except Exception:
            return np.nan


def load_all_tle_data(folder_paths, *, only_files=None, derived=None):
    """Load TLE data from one or more folders.

    Parameters
    ----------
    folder_paths : str | list[str]
        One folder path or a list of folder paths.
    only_files : None | list[str] | set[str]
        If provided, only parses these filenames (e.g., {'sat1008.txt'}).
        Matching is case-insensitive on basename.
    derived : None | bool | iterable[str]
        Controls which derived columns are computed/returned.
        - None/True  -> compute all: {'sma','true_anomaly','specific_angular_momentum'}
        - False      -> compute none
        - iterable   -> compute subset
    """
    if isinstance(folder_paths, (str, os.PathLike)):
        folder_paths = [str(folder_paths)]

    if derived is None or derived is True:
        derived_set = {"sma", "true_anomaly", "specific_angular_momentum"}
    elif derived is False:
        derived_set = set()
    else:
        derived_set = {str(x) for x in derived}

    only_files_set = None
    if only_files is not None:
        only_files_set = {os.path.basename(str(f)).lower() for f in only_files}

    # Accumulate parsed fields.
    # Use compact numeric buffers to reduce memory overhead vs Python float/int lists.
    sat_ids = []
    timestamps = []
    tle_epoch_list = []

    inclination_buf = array('d')
    raan_buf = array('d')
    eccentricity_buf = array('d')
    arg_of_perigee_buf = array('d')
    mean_anomaly_buf = array('d')
    mean_motion_buf = array('d')

    launch_year_buf = array('i')
    launch_num_buf = array('i')

    launch_piece_list = []
    classification_list = []
    int_designator_list = []

    ballistic_coeff_buf = array('d')
    drag_term_buf = array('d')

    filenames_array = []

    year_start_cache = {}

    # Bind frequently-used callables/attributes locally (small speed win in tight loops)
    sat_ids_append = sat_ids.append
    timestamps_append = timestamps.append
    tle_epoch_append = tle_epoch_list.append
    filenames_append = filenames_array.append
    launch_piece_append = launch_piece_list.append
    classification_append = classification_list.append
    int_designator_append = int_designator_list.append

    inc_append = inclination_buf.append
    raan_append = raan_buf.append
    ecc_append = eccentricity_buf.append
    aop_append = arg_of_perigee_buf.append
    ma_append = mean_anomaly_buf.append
    mm_append = mean_motion_buf.append
    ly_append = launch_year_buf.append
    ln_append = launch_num_buf.append
    bc_append = ballistic_coeff_buf.append
    drag_append = drag_term_buf.append

    parse_drag = parse_drag_term
    year_start_cache_get = year_start_cache.get

    def _iter_nonblank_lines(fh):
        for raw_ln in fh:
            ln = raw_ln.rstrip()
            if ln:
                yield ln

    for folder in folder_paths:
        if not os.path.isdir(folder):
            print(f"Warning: folder not found: {folder}")
            continue

        for entry in os.scandir(folder):
            if not entry.is_file():
                continue
            filename = entry.name
            if not filename.lower().endswith('.txt'):
                continue
            if only_files_set is not None and filename.lower() not in only_files_set:
                continue

            path = entry.path

            with open(path, 'r') as fh:
                # Stream non-blank lines in blocks of 3: [name, line1, line2]
                it = _iter_nonblank_lines(fh)
                for _name, tle_line1, tle_line2 in zip(it, it, it):

                    # Sanity check field widths
                    if len(tle_line1) < 32 or len(tle_line2) < 63:
                        continue

                    # Parse epoch from line1 (chars 18–32)
                    epoch_str = tle_line1[18:32].strip()
                    if not epoch_str:
                        continue
                    try:
                        # 'YYDDD.DDDDDDDD': year, day-of-year
                        yyddd = float(epoch_str)
                        yy = int(yyddd // 1000)
                        ddd = yyddd - yy * 1000
                        year = 2000 + yy if yy < 57 else 1900 + yy
                        year_start = year_start_cache_get(year)
                        if year_start is None:
                            year_start = datetime(year, 1, 1)
                            year_start_cache[year] = year_start
                        epoch_dt = year_start + timedelta(days=ddd - 1)

                        # Parse orbital elements from line2
                        inc = float(tle_line2[8:16])
                        raan = float(tle_line2[17:25])
                        ecc = float('0.' + tle_line2[26:33].strip())
                        aop = float(tle_line2[34:42])
                        ma_deg = float(tle_line2[43:51])
                        mm = float(tle_line2[52:63])

                        # Parse launch/designator from line1
                        classification = tle_line1[7]
                        ly = int(tle_line1[9:11])
                        ln = int(tle_line1[11:14])
                        lp = tle_line1[14:17].strip()
                        intl_des = tle_line1[9:17].strip()

                        # Parse ballistic & drag terms from line1
                        bc = float(tle_line1[33:43])
                        drag_str = tle_line1[53:62]
                        drag_term = parse_drag(drag_str)

                    except Exception:
                        # Skip malformed blocks
                        continue

                    # Accumulate data
                    sat_ids_append(filename)
                    timestamps_append(epoch_dt)
                    tle_epoch_append(epoch_str)
                    inc_append(inc)
                    raan_append(raan)
                    ecc_append(ecc)
                    aop_append(aop)
                    ma_append(ma_deg)
                    mm_append(mm)
                    ly_append(ly)
                    ln_append(ln)
                    launch_piece_append(lp)
                    classification_append(classification)
                    int_designator_append(intl_des)
                    bc_append(bc)
                    drag_append(drag_term)
                    filenames_append(filename)

    # Convert to NumPy (views when possible)
    mm_arr = np.frombuffer(mean_motion_buf, dtype=np.float64)
    ecc_arr = np.frombuffer(eccentricity_buf, dtype=np.float64)
    ma_deg_arr = np.frombuffer(mean_anomaly_buf, dtype=np.float64)

    # Compute derived quantities only if requested
    sma_arr = None
    ta_deg_arr = None
    h_arr = None

    need_sma = ("sma" in derived_set) or ("specific_angular_momentum" in derived_set)
    if need_sma:
        sma_arr = semi_major_axis_vector(mm_arr)

    if "true_anomaly" in derived_set:
        ma_rad_arr = np.radians(ma_deg_arr)
        ta_deg_arr = mean_to_true_anomaly_vector(ma_rad_arr, ecc_arr)

    if "specific_angular_momentum" in derived_set:
        if sma_arr is None:
            sma_arr = semi_major_axis_vector(mm_arr)
        h_arr = specific_angular_momentum_vector(sma_arr, ecc_arr)

    # DataFrame construction
    data = {
        'sat_id': sat_ids,
        'timestamp': timestamps,
        'tle_epoch': tle_epoch_list,
        'ecc': ecc_arr,
        'inc': np.frombuffer(inclination_buf, dtype=np.float64),
        'raan': np.frombuffer(raan_buf, dtype=np.float64),
        'aop': np.frombuffer(arg_of_perigee_buf, dtype=np.float64),
        'mean_anomaly': ma_deg_arr,
        'mean_motion': mm_arr,
        'launch_year': np.frombuffer(launch_year_buf, dtype=np.int32),
        'launch_num': np.frombuffer(launch_num_buf, dtype=np.int32),
        'launch_piece': launch_piece_list,
        'classification': classification_list,
        'international_designator': int_designator_list,
        'ballistic_coefficient': np.frombuffer(ballistic_coeff_buf, dtype=np.float64),
        'drag_term': np.frombuffer(drag_term_buf, dtype=np.float64),
    }

    if sma_arr is not None:
        data['sma'] = sma_arr
    if ta_deg_arr is not None:
        data['true_anomaly'] = ta_deg_arr
    if h_arr is not None:
        data['specific_angular_momentum'] = h_arr

    df = pd.DataFrame(data)
    
    return df, filenames_array