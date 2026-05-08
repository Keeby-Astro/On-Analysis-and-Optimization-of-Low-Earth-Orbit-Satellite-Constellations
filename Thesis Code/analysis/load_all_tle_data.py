import os
import warnings
import numpy as np
import pandas as pd
from datetime import UTC, datetime, timedelta
from array import array

from constants import (
    COMPATIBILITY_ALIAS_MAP,
    PHASE_SEMANTICS_TRUE_ANOMALY_PROXY,
    PHASE_VARIABLE_TRUE_ANOMALY,
)
from orbital_features import (mean_to_true_anomaly_vector,
                              semi_major_axis_vector,
                              specific_angular_momentum_vector)

def _bool_from_env(var_name, default=False):
    value = os.getenv(var_name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _tle_checksum_is_valid(tle_line):
    """Validate TLE checksum (mod 10)."""
    if len(tle_line) < 69:
        return False
    checksum_char = tle_line[68]
    if not checksum_char.isdigit():
        return False

    total = 0
    for char in tle_line[:68]:
        if char.isdigit():
            total += int(char)
        elif char == '-':
            total += 1

    return (total % 10) == int(checksum_char)

def _parse_assumed_decimal(field):
    """Parse TLE assumed-decimal scientific format (e.g., '+34124-4')."""
    s = str(field).strip()
    if not s:
        return np.nan

    try:
        return float(s)
    except Exception:
        pass

    if len(s) < 2:
        return np.nan

    exp_sign = s[-2]
    exp_digits = s[-1:]
    mantissa = s[:-2]

    sign = ""
    if mantissa and mantissa[0] in "+-":
        sign = mantissa[0]
        mantissa = mantissa[1:]

    mantissa_digits = mantissa.replace(" ", "").replace(".", "")
    if not mantissa_digits:
        return np.nan

    try:
        return float(f"{sign}0.{mantissa_digits}e{exp_sign}{exp_digits}")
    except Exception:
        return np.nan

def _parse_epoch_to_datetime(epoch_str, year_start_cache):
    """Convert TLE epoch YYDDD.DDDDDDDD to datetime."""
    yyddd = float(epoch_str)
    yy = int(yyddd // 1000)
    ddd = yyddd - yy * 1000
    year = 2000 + yy if yy < 57 else 1900 + yy

    year_start = year_start_cache.get(year)
    if year_start is None:
        year_start = datetime(year, 1, 1)
        year_start_cache[year] = year_start

    return year_start + timedelta(days=ddd - 1)

def _parse_tle_line1(line1):
    """Parse TLE line 1 using fixed-width columns."""
    if len(line1) < 68:
        raise ValueError("line 1 is too short")

    norad_cat_id = line1[2:7].strip()
    classification = line1[7:8].strip()
    launch_year_2digit = int(line1[9:11])
    launch_num = int(line1[11:14])
    launch_piece = line1[14:17].strip()
    international_designator = line1[9:17].strip()
    launch_year_full = 2000 + launch_year_2digit if launch_year_2digit < 57 else 1900 + launch_year_2digit

    epoch_str = line1[18:32].strip()
    mean_motion_dot = float(line1[33:43].strip())
    mean_motion_ddot = _parse_assumed_decimal(line1[44:52])
    bstar = _parse_assumed_decimal(line1[53:61])

    ephemeris_type = line1[62:63].strip()
    element_number_raw = line1[64:68].strip()
    element_number = int(element_number_raw) if element_number_raw else -1

    return {"norad_cat_id": norad_cat_id,
            "classification": classification,
            "launch_year": launch_year_2digit,
            "launch_year_full": launch_year_full,
            "launch_num": launch_num,
            "launch_piece": launch_piece,
            "international_designator": international_designator,
            "tle_epoch": epoch_str,
            "mean_motion_dot": mean_motion_dot,
            "mean_motion_ddot": mean_motion_ddot,
            "bstar": bstar,
            "ephemeris_type": ephemeris_type,
            "element_number": element_number}

def _infer_element_set_format(_line0, _line1, _line2):
    """Scaffolding hook for future GP/OMM ingestion format routing."""
    return "TLE"

def _parse_tle_line2(line2):
    """Parse TLE line 2 using fixed-width columns."""
    if len(line2) < 63:
        raise ValueError("line 2 is too short")

    norad_cat_id = line2[2:7].strip()
    inc = float(line2[8:16])
    raan = float(line2[17:25])
    ecc_digits = line2[26:33].strip()
    ecc = float(f"0.{ecc_digits}")
    aop = float(line2[34:42])
    mean_anomaly = float(line2[43:51])
    mean_motion = float(line2[52:63])

    return {"norad_cat_id": norad_cat_id,
            "inc": inc,
            "raan": raan,
            "ecc": ecc,
            "aop": aop,
            "mean_anomaly": mean_anomaly,
            "mean_motion": mean_motion}

def _iter_three_line_records(file_path):
    """Yield (line0_name, line1, line2, record_index) from a historical 3LE file."""
    records = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n\r")
            if line.strip():
                records.append((line_no, line))

    triplet_count = len(records) // 3
    for idx in range(triplet_count):
        base = idx * 3
        (line0_no, line0), (line1_no, line1), (line2_no, line2) = records[base:base + 3]
        yield line0, line1, line2, idx + 1, (line0_no, line1_no, line2_no)

    if len(records) % 3 != 0:
        warnings.warn(f"File '{os.path.basename(file_path)}' has trailing non-empty lines outside complete 3LE triplets; trailing lines were ignored.",
                      RuntimeWarning)

def load_all_tle_data(folder_paths, *, only_files=None, derived=None):
    """Load TLE data from one or more folders.

        Notes on semantics
            - Parsed TLE fields are SGP4-compatible public-product mean-element data.
             - Keplerized quantities computed from those fields are diagnostic proxies,
                not conjunction-grade osculating states.
            - See CelesTrak TLE/GP documentation and Vallado et al. (AIAA-2006-6753)
                for formal mean-element semantics.

    Parameters
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

    # Optional warning-only checksum validation (default off to preserve legacy behavior).
    validate_checksums = _bool_from_env("TLE_VALIDATE_CHECKSUM", default=False)

    # Accumulate parsed fields.
    # Use compact numeric buffers to reduce memory overhead vs Python float/int lists.
    sat_ids = []
    timestamps = []
    tle_epoch_list = []
    norad_cat_id_list = []
    ephemeris_type_list = []
    source_filename_list = []
    object_name_list = []
    element_set_format_list = []
    object_id_list = []
    sat_id_kind_list = []
    raw_tle_line1_list = []
    raw_tle_line2_list = []
    parse_timestamp_utc_list = []
    checksum_line1_valid_list = []
    checksum_line2_valid_list = []
    parser_quality_flag_list = []

    inclination_buf = array('d')
    raan_buf = array('d')
    eccentricity_buf = array('d')
    arg_of_perigee_buf = array('d')
    mean_anomaly_buf = array('d')
    mean_motion_buf = array('d')

    launch_year_buf = array('i')
    launch_year_full_buf = array('i')
    launch_num_buf = array('i')
    element_number_buf = array('i')

    launch_piece_list = []
    classification_list = []
    int_designator_list = []

    mean_motion_dot_buf = array('d')
    mean_motion_ddot_buf = array('d')
    bstar_buf = array('d')

    filenames_array = []

    year_start_cache = {}

    # Bind frequently-used callables/attributes locally (small speed win in tight loops)
    sat_ids_append = sat_ids.append
    timestamps_append = timestamps.append
    tle_epoch_append = tle_epoch_list.append
    filenames_append = filenames_array.append
    norad_append = norad_cat_id_list.append
    ephem_append = ephemeris_type_list.append
    source_file_append = source_filename_list.append
    object_name_append = object_name_list.append
    element_set_format_append = element_set_format_list.append
    object_id_append = object_id_list.append
    sat_id_kind_append = sat_id_kind_list.append
    raw_tle1_append = raw_tle_line1_list.append
    raw_tle2_append = raw_tle_line2_list.append
    parse_timestamp_append = parse_timestamp_utc_list.append
    checksum_l1_append = checksum_line1_valid_list.append
    checksum_l2_append = checksum_line2_valid_list.append
    parser_quality_append = parser_quality_flag_list.append
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
    ly_full_append = launch_year_full_buf.append
    ln_append = launch_num_buf.append
    element_number_append = element_number_buf.append
    mmdot_append = mean_motion_dot_buf.append
    mmddot_append = mean_motion_ddot_buf.append
    bstar_append = bstar_buf.append

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

            for object_name, tle_line1, tle_line2, rec_idx, line_nums in _iter_three_line_records(path):
                if not tle_line1.startswith('1'):
                    continue
                if not tle_line2.startswith('2'):
                    continue

                checksum_l1 = _tle_checksum_is_valid(tle_line1)
                checksum_l2 = _tle_checksum_is_valid(tle_line2)

                if validate_checksums:
                    if not checksum_l1:
                        warnings.warn(f"Checksum mismatch in line 1 for triplet #{rec_idx} in '{filename}' (warning only).",
                                      RuntimeWarning)
                    if not checksum_l2:
                        warnings.warn(f"Checksum mismatch in line 2 for triplet #{rec_idx} in '{filename}' (warning only).",
                                      RuntimeWarning)

                try:
                    parsed_l1 = _parse_tle_line1(tle_line1)
                    parsed_l2 = _parse_tle_line2(tle_line2)
                    if parsed_l1['norad_cat_id'] != parsed_l2['norad_cat_id']:
                        warnings.warn(f"Skipping malformed triplet #{rec_idx} in '{filename}': NORAD mismatch line1={parsed_l1['norad_cat_id']} line2={parsed_l2['norad_cat_id']}.", RuntimeWarning)
                        continue

                    epoch_dt = _parse_epoch_to_datetime(parsed_l1['tle_epoch'], year_start_cache)
                except Exception as exc:
                    warnings.warn(f"Skipping malformed triplet #{rec_idx} in '{filename}': {exc}",
                                  RuntimeWarning)
                    continue

                # sat_id remains filename-based for backward compatibility.
                sat_ids_append(filename)
                timestamps_append(epoch_dt)
                tle_epoch_append(parsed_l1['tle_epoch'])
                inc_append(parsed_l2['inc'])
                raan_append(parsed_l2['raan'])
                ecc_append(parsed_l2['ecc'])
                aop_append(parsed_l2['aop'])
                ma_append(parsed_l2['mean_anomaly'])
                mm_append(parsed_l2['mean_motion'])
                ly_append(parsed_l1['launch_year'])
                ly_full_append(parsed_l1['launch_year_full'])
                ln_append(parsed_l1['launch_num'])
                launch_piece_append(parsed_l1['launch_piece'])
                classification_append(parsed_l1['classification'])
                int_designator_append(parsed_l1['international_designator'])
                mmdot_append(parsed_l1['mean_motion_dot'])
                mmddot_append(parsed_l1['mean_motion_ddot'])
                bstar_append(parsed_l1['bstar'])
                norad_append(parsed_l1['norad_cat_id'])
                ephem_append(parsed_l1['ephemeris_type'])
                element_number_append(parsed_l1['element_number'])
                source_file_append(filename)
                object_name_append(object_name.strip())
                element_set_format_append(_infer_element_set_format(object_name, tle_line1, tle_line2))
                object_id_append(parsed_l1['norad_cat_id'])
                sat_id_kind_append('filename')
                raw_tle1_append(tle_line1)
                raw_tle2_append(tle_line2)
                parse_timestamp_append(datetime.now(UTC))
                checksum_l1_append(checksum_l1)
                checksum_l2_append(checksum_l2)
                parser_quality_append('ok' if (checksum_l1 and checksum_l2) else 'checksum_warning')
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
    data = {'sat_id': sat_ids,
            'timestamp': timestamps,
            'tle_epoch': tle_epoch_list,
            'norad_cat_id': norad_cat_id_list,
            'ecc': ecc_arr,
            'inc': np.frombuffer(inclination_buf, dtype=np.float64),
            'raan': np.frombuffer(raan_buf, dtype=np.float64),
            'aop': np.frombuffer(arg_of_perigee_buf, dtype=np.float64),
            'mean_anomaly': ma_deg_arr,
            'mean_motion': mm_arr,
            'launch_year': np.frombuffer(launch_year_buf, dtype=np.int32),
            'launch_year_full': np.frombuffer(launch_year_full_buf, dtype=np.int32),
            'launch_num': np.frombuffer(launch_num_buf, dtype=np.int32),
            'launch_piece': launch_piece_list,
            'classification': classification_list,
            'international_designator': int_designator_list,
            # Legacy compatibility aliases: these names are preserved for downstream callers.
            # ballistic_coefficient is a compatibility alias for mean_motion_dot,
            # and is not a physical ballistic coefficient.
            'ballistic_coefficient': np.frombuffer(mean_motion_dot_buf, dtype=np.float64),
            'drag_term': np.frombuffer(bstar_buf, dtype=np.float64),
            'ephemeris_type': ephemeris_type_list,
            'element_number': np.frombuffer(element_number_buf, dtype=np.int32),
            'mean_motion_dot': np.frombuffer(mean_motion_dot_buf, dtype=np.float64),
            'mean_motion_ddot': np.frombuffer(mean_motion_ddot_buf, dtype=np.float64),
            'bstar': np.frombuffer(bstar_buf, dtype=np.float64),
            'source_filename': source_filename_list,
            'object_name': object_name_list,
            'element_set_format': element_set_format_list,
            'object_id': object_id_list,
            'sat_id_kind': sat_id_kind_list,
            'parse_timestamp_utc': parse_timestamp_utc_list,
            'checksum_line1_valid': checksum_line1_valid_list,
            'checksum_line2_valid': checksum_line2_valid_list,
            'parser_quality_flag': parser_quality_flag_list,
            # Raw lines are retained as internal scaffolding for optional SGP4 usage.
            'tle_line1_raw': raw_tle_line1_list,
            'tle_line2_raw': raw_tle_line2_list}

    data['has_raw_tle'] = [bool(str(l1).strip()) and bool(str(l2).strip())
                           for l1, l2 in zip(raw_tle_line1_list, raw_tle_line2_list)]

    if sma_arr is not None:
        data['sma'] = sma_arr
    if ta_deg_arr is not None:
        data['true_anomaly'] = ta_deg_arr
    if h_arr is not None:
        data['specific_angular_momentum'] = h_arr

    # Canonical proxy aliases (additive, backward compatible).
    if sma_arr is not None:
        data['sma_kepler_proxy_km'] = sma_arr
    if ta_deg_arr is not None:
        data['true_anomaly_kepler_proxy_deg'] = ta_deg_arr
    if h_arr is not None:
        data['specific_angular_momentum_kepler_proxy_km2_s'] = h_arr

    data['ballistic_coefficient_alias_of'] = COMPATIBILITY_ALIAS_MAP['ballistic_coefficient']
    data['drag_term_alias_of'] = COMPATIBILITY_ALIAS_MAP['drag_term']
    data['source_semantics'] = 'catalog_mean_elements_tle_gp'
    data['derived_semantics'] = 'keplerized_proxy_diagnostics'

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(['sat_id', 'timestamp'], kind='mergesort').reset_index(drop=True)

    numeric_cols = [
        'inc',
        'raan',
        'ecc',
        'aop',
        'mean_anomaly',
        'mean_motion',
        'sma',
        'true_anomaly',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'true_anomaly' in df.columns:
        if 'true_anomaly_kepler_proxy_deg' not in df.columns:
            df['true_anomaly_kepler_proxy_deg'] = df['true_anomaly']
        df['phase_variable'] = PHASE_VARIABLE_TRUE_ANOMALY
        df['phase_semantics'] = PHASE_SEMANTICS_TRUE_ANOMALY_PROXY

    filenames_array = df['sat_id'].tolist()

    return df, filenames_array