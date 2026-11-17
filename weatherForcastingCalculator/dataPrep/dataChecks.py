############################################################################################
"""
Data checks for the weather forecasting calculator.

checkSyntheticInputs: fills defaults, casts types and validates inputs for the synthetic
                      station generator (same idea as checkData in Production-Prediction-ML).
checkWeatherData:     validates any station dataset in the cached format the calculator
                      reads (weatherDataJSONObject.json + Weather Stations Info.csv), whether
                      it came from NOAA USCRN or from the synthetic generator.
"""
############################################################################################
import os
import json
import numpy as np
import pandas as pd
from datetime import date

############################################################################################
# USCRN hourly02 column layout, same order the NOAA parser uses
WEATHER_DATA_COLUMNS = [
    'WBANNO', 'UTC_DATE', 'UTC_TIME', 'LST_DATE', 'LST_TIME', 'CRX_VN', 'LONGITUDE', 'LATITUDE',
    'T_CALC', 'T_HR_AVG', 'T_MAX', 'T_MIN', 'P_CALC',
    'SOLARAD', 'SOLARAD_FLAG', 'SOLARAD_MAX', 'SOLARAD_MAX_FLAG', 'SOLARAD_MIN', 'SOLARAD_MIN_FLAG',
    'SUR_TEMP_TYPE', 'SUR_TEMP', 'SUR_TEMP_FLAG', 'SUR_TEMP_MAX', 'SUR_TEMP_MAX_FLAG', 'SUR_TEMP_MIN', 'SUR_TEMP_MIN_FLAG',
    'RH_HR_AVG', 'RH_HR_AVG_FLAG',
    'SOIL_MOISTURE_5', 'SOIL_MOISTURE_10', 'SOIL_MOISTURE_20', 'SOIL_MOISTURE_50', 'SOIL_MOISTURE_100',
    'SOIL_TEMP_5', 'SOIL_TEMP_10', 'SOIL_TEMP_20', 'SOIL_TEMP_50', 'SOIL_TEMP_100',
]
STATION_COLUMNS = ['City', 'Longitude', 'Latitude', 'Elevation']
DATA_FILE       = 'weatherDataJSONObject.json'
STATIONS_FILE   = 'Weather Stations Info.csv'

# Physical sanity ranges used by the checks (real or synthetic data)
VALUE_RANGES = {
    'T_CALC': (-60.0, 60.0), 'T_HR_AVG': (-60.0, 60.0), 'T_MAX': (-60.0, 60.0), 'T_MIN': (-60.0, 60.0),
    'P_CALC': (0.0, 300.0), 'SOLARAD': (0.0, 1500.0), 'RH_HR_AVG': (0.0, 100.0),
    'SOIL_MOISTURE_5': (0.0, 1.0), 'SOIL_MOISTURE_100': (0.0, 1.0),
    'LATITUDE': (-90.0, 90.0), 'LONGITUDE': (-180.0, 180.0),
}
CORE_COLUMNS = ['T_HR_AVG', 'T_MAX', 'T_MIN', 'P_CALC', 'SOLARAD', 'RH_HR_AVG']

############################################################################################
SYNTHETIC_DEFAULTS = {
    'name':                 'Synthetic_Stations',
    'random_seed':          5195,
    'output_dir':           'Data/Synthetic',
    'n_stations':           20,
    'start_date':           '2024-01-01',
    'n_days':               90,
    'lat_min':              25.0,
    'lat_max':              49.0,
    'lon_min':              -124.0,
    'lon_max':              -67.0,
    # Climate knobs
    'lapse_rate_c_per_km':  6.5,
    'lat_gradient_c_per_deg': 0.6,
    'seasonal_amp_base_c':  5.0,
    'diurnal_amp_c':        6.0,
    'synoptic_std_c':       3.0,
    'synoptic_persistence': 0.98,      # hourly AR(1) coefficient
    'spatial_corr_km':      800.0,
    'p_dry_to_wet':         0.03,      # hourly Markov chain for rain
    'p_wet_to_wet':         0.70,
    'rain_mean_mm':         1.2,
    # Data quality
    'missing_fraction':     0.01,      # random single-hour gaps in measured columns
    'soil_outage_fraction': 0.25,      # share of stations with a multi-day soil sensor outage
    # Size guard
    'max_rows':             2000000,
    'allow_large':          False,
}

############################################################################################
def checkSyntheticInputs(inputData={}):
    """
    Fill defaults, cast types and validate synthetic weather inputs.
    Collects every problem and raises once so the UI can show them together.
    """
    ########################################################################################
    data = dict(SYNTHETIC_DEFAULTS)
    data.update({k: v for k, v in inputData.items() if v is not None})
    errors = []
    for key in ['random_seed', 'n_stations', 'n_days', 'max_rows']:
        try: data[key] = int(data[key])
        except (TypeError, ValueError): errors.append(f"{key} must be an integer, got {data[key]!r}")
    for key, default in SYNTHETIC_DEFAULTS.items():
        if isinstance(default, float):
            try: data[key] = float(data[key])
            except (TypeError, ValueError): errors.append(f"{key} must be a number, got {data[key]!r}")
    data['allow_large'] = bool(data['allow_large'])
    if errors: raise ValueError("Input check failed:\n  - " + "\n  - ".join(errors))
    ########################################################################################
    try: date.fromisoformat(str(data['start_date']))
    except ValueError: errors.append(f"start_date must be YYYY-MM-DD, got {data['start_date']!r}")
    if data['n_stations'] < 1:  errors.append("n_stations must be >= 1")
    if data['n_stations'] > 999: errors.append("n_stations must be <= 999")
    if data['n_days'] < 1:      errors.append("n_days must be >= 1")
    if not (-90 <= data['lat_min'] < data['lat_max'] <= 90):     errors.append("need -90 <= lat_min < lat_max <= 90")
    if not (-180 <= data['lon_min'] < data['lon_max'] <= 180):   errors.append("need -180 <= lon_min < lon_max <= 180")
    if not (0 <= data['synoptic_persistence'] < 1):              errors.append("synoptic_persistence must be in [0, 1)")
    for key in ['p_dry_to_wet', 'p_wet_to_wet']:
        if not (0 <= data[key] <= 1): errors.append(f"{key} must be in [0, 1]")
    for key in ['missing_fraction', 'soil_outage_fraction']:
        if not (0 <= data[key] < 0.5): errors.append(f"{key} must be in [0, 0.5)")
    for key in ['diurnal_amp_c', 'synoptic_std_c', 'spatial_corr_km', 'rain_mean_mm']:
        if data[key] < 0 or (key in ('spatial_corr_km', 'rain_mean_mm') and data[key] == 0): errors.append(f"{key} must be > 0")
    ########################################################################################
    data['n_hours'] = data['n_days'] * 24
    n_rows = data['n_hours'] * data['n_stations']
    data['n_rows'] = n_rows
    if n_rows > data['max_rows'] and not data['allow_large']:
        errors.append(f"{n_rows:,} station-hours requested (limit {data['max_rows']:,}). "
                      f"Use fewer stations or days, or set allow_large")
    if errors: raise ValueError("Input check failed:\n  - " + "\n  - ".join(errors))
    return data

############################################################################################
def stationTimes(stationDict={}):
    """UTC timestamps for one station from UTC_DATE (YYYYMMDD) and UTC_TIME (HHMM, 2400 allowed)."""
    d = pd.to_datetime(pd.Series(stationDict['UTC_DATE']).astype(float).astype(int).astype(str), format='%Y%m%d', errors='coerce')
    hhmm = pd.Series(stationDict['UTC_TIME']).astype(float).astype(int)
    return d + pd.to_timedelta(hhmm // 100, unit='h') + pd.to_timedelta(hhmm % 100, unit='m')

############################################################################################
def checkWeatherData(dataDir='', max_nan_fraction=0.2, max_soil_nan_fraction=0.9, max_out_of_range_fraction=0.001):
    """
    Validate a cached station dataset before modeling. Works on real USCRN data and synthetic data.
    :param dataDir:                   Folder holding weatherDataJSONObject.json and Weather Stations Info.csv
    :param max_nan_fraction:          Max NaN share allowed in core columns (temperature, precip, solar, RH)
    :param max_soil_nan_fraction:     Max NaN share for soil columns before it is flagged (often missing in real data)
    :param max_out_of_range_fraction: Max share of values outside physical ranges per column
    :return:                          dict with passed, errors, warnings, summary
    """
    ########################################################################################
    errors, warnings, summary = [], [], {}
    data_path, stations_path = os.path.join(dataDir, DATA_FILE), os.path.join(dataDir, STATIONS_FILE)
    for p in (data_path, stations_path):
        if not os.path.exists(p): errors.append(f"missing file: {p}")
    if errors: return {'passed': False, 'errors': errors, 'warnings': warnings, 'summary': summary}
    ########################################################################################
    with open(data_path) as f: weather = json.load(f)
    df_st = pd.read_csv(stations_path)
    missing_cols = [c for c in STATION_COLUMNS if c not in df_st.columns]
    if missing_cols: errors.append(f"stations file missing columns: {missing_cols}")
    if not isinstance(weather, dict) or not weather: errors.append("station data must be a non-empty dict of stations")
    if errors: return {'passed': False, 'errors': errors, 'warnings': warnings, 'summary': summary}
    ########################################################################################
    in_csv, in_json = set(df_st['City'].astype(str)), set(weather.keys())
    if in_json - in_csv: warnings.append(f"{len(in_json - in_csv)} stations in data but not in stations file")
    if in_csv - in_json: errors.append(f"{len(in_csv - in_json)} stations in stations file have no data")
    ########################################################################################
    n_rows, gaps, bad_stations = 0, 0, 0
    for name, st in weather.items():
        miss = [c for c in WEATHER_DATA_COLUMNS if c not in st]
        if miss:
            errors.append(f"station {name} missing columns: {miss[:5]}{'...' if len(miss) > 5 else ''}"); bad_stations += 1; continue
        lengths = {len(st[c]) for c in WEATHER_DATA_COLUMNS}
        if len(lengths) != 1:
            errors.append(f"station {name}: columns have different lengths {sorted(lengths)[:4]}"); bad_stations += 1; continue
        n = lengths.pop(); n_rows += n
        ####################################################################################
        t = stationTimes(st)
        if t.isna().any(): errors.append(f"station {name}: unparseable UTC_DATE/UTC_TIME"); continue
        dt = t.diff().dropna()
        if (dt <= pd.Timedelta(0)).any(): errors.append(f"station {name}: timestamps not strictly increasing")
        gaps += int((dt > pd.Timedelta(hours=1)).sum())
        ####################################################################################
        for col, (lo, hi) in VALUE_RANGES.items():
            v = np.asarray(st[col], dtype=float)
            v = v[~np.isnan(v)]
            if len(v) and np.mean((v < lo) | (v > hi)) > max_out_of_range_fraction:
                errors.append(f"station {name}: {col} has values outside {lo}..{hi}")
        for col in CORE_COLUMNS:
            if np.mean(np.isnan(np.asarray(st[col], dtype=float))) > max_nan_fraction:
                errors.append(f"station {name}: {col} is more than {max_nan_fraction:.0%} missing")
        tmin, tmax = np.asarray(st['T_MIN'], float), np.asarray(st['T_MAX'], float)
        ok = ~np.isnan(tmin) & ~np.isnan(tmax)
        if ok.any() and np.mean(tmin[ok] > tmax[ok] + 0.05) > 0.001: errors.append(f"station {name}: T_MIN above T_MAX")
        soil = np.asarray(st['SOIL_MOISTURE_5'], float)
        if np.mean(np.isnan(soil)) > max_soil_nan_fraction: warnings.append(f"station {name}: soil data mostly missing")
    ########################################################################################
    if gaps: warnings.append(f"{gaps} gaps longer than one hour across all stations")
    summary = {'n_stations': len(weather), 'n_rows': n_rows, 'bad_stations': bad_stations,
               'lat_range': [float(df_st['Latitude'].min()), float(df_st['Latitude'].max())],
               'lon_range': [float(df_st['Longitude'].min()), float(df_st['Longitude'].max())]}
    return {'passed': len(errors) == 0, 'errors': errors, 'warnings': warnings, 'summary': summary}
############################################################################################
