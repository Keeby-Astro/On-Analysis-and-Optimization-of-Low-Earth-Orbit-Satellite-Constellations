import pandas as pd
from datetime import datetime

def filter_tle_data_by_date(tle_data, target_date, time_tolerance):
    """
    Filter TLE data for a specific date using optimized approaches.

    Parameters:
        tle_data (pd.DataFrame): 
            DataFrame containing TLE data. Must have a 'timestamp' column 
            (dtype datetime64[ns]) or a DatetimeIndex.
        target_date (datetime or pd.Timestamp): 
            Target date for filtering.
        time_tolerance (timedelta or pd.Timedelta): 
            Time tolerance around the target date.

    Returns:
        pd.DataFrame: Filtered TLE data.
    """
    # Basic validations
    if tle_data.empty:
        raise ValueError("TLE data is empty.")
    
    if not isinstance(target_date, (datetime, pd.Timestamp)):
        raise TypeError("target_date must be a datetime or pd.Timestamp object.")
    
    # Convert to pandas Timestamp for consistency
    target_date = pd.Timestamp(target_date)

    # Calculate start/end
    start_date = target_date - time_tolerance
    end_date   = target_date + time_tolerance

    # If the TLE DataFrame has a DatetimeIndex, do a slice
    if isinstance(tle_data.index, pd.DatetimeIndex):
        # Ensure the index is sorted to allow fast slicing
        if not tle_data.index.is_monotonic_increasing:
            tle_data = tle_data.sort_index()
        
        # Return the slice between start_date and end_date
        return tle_data.loc[start_date:end_date]

    # Otherwise, filter using a timestamp column efficiently
    else:
        if 'timestamp' not in tle_data.columns:
            raise KeyError("TLE data does not contain a 'timestamp' column.")
        
        # Ensure 'timestamp' column is a proper datetime type
        if not pd.api.types.is_datetime64_ns_dtype(tle_data['timestamp']):
            tle_data['timestamp'] = pd.to_datetime(tle_data['timestamp'], errors='coerce')
        
        # Use between for a single, efficient boolean mask
        mask = tle_data['timestamp'].between(start_date, end_date)
        return tle_data[mask]