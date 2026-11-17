############################################################################################
"""
Synthetic hourly weather station generator.

Produces the same cached format the calculator reads after a NOAA USCRN pull:
    weatherDataJSONObject.json   {station: {USCRN column: [hourly values]}}, cleaned (NaN for missing)
    Weather Stations Info.csv    City, Longitude, Latitude, Elevation
    manifest.json                inputs, date range, stations, files

Physics is simple but connected, so models have real signal to find:
    temperature  = climate (latitude, elevation lapse rate) + seasonal + diurnal + spatially
                   correlated synoptic anomaly (AR(1) in time, distance-decay across stations)
    rain         = hourly wet/dry Markov chain, gamma amounts, more likely when the anomaly is cold
    solar        = clear-sky from solar elevation, cut by rain
    RH           = from temperature and dewpoint spread, higher when raining
    surface/soil = surface heated by sun; soil temps lagged and damped with depth
    soil moisture = bucket filled by rain, drains back toward a base level, slower at depth

Nothing here pulls from the network. Output goes to a gitignored folder and stays local.
"""
############################################################################################
import os
import sys
import json
import numpy as np
import pandas as pd
from os.path import dirname, join, isabs, abspath

try:
    from dataPrep.dataChecks import checkSyntheticInputs, WEATHER_DATA_COLUMNS, DATA_FILE, STATIONS_FILE
    from dataPrep.inputWrapper import standardInputWrapper
except ImportError:
    from .dataChecks import checkSyntheticInputs, WEATHER_DATA_COLUMNS, DATA_FILE, STATIONS_FILE
    from .inputWrapper import standardInputWrapper

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))

############################################################################################
def haversineKm(lat, lon):
    """Pairwise great-circle distance matrix in km."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

############################################################################################
def emaAlongTime(x, tau_hours):
    """Exponential moving average along axis 1 (stations x hours). Larger tau = slower, damped response."""
    alpha = 1.0 / max(tau_hours, 1.0)
    out = np.empty_like(x)
    out[:, 0] = x[:, :24].mean(axis=1)
    for h in range(1, x.shape[1]):
        out[:, h] = out[:, h - 1] + alpha * (x[:, h] - out[:, h - 1])
    return out

############################################################################################
def buildStations(data={}, rng=None):
    """Station locations and elevations. Elevation follows a broad western high-terrain trend plus noise."""
    ########################################################################################
    n = data['n_stations']
    lat = rng.uniform(data['lat_min'], data['lat_max'], n)
    lon = rng.uniform(data['lon_min'], data['lon_max'], n)
    elev_m = 2200.0 * np.exp(-((lon + 110.0) / 9.0) ** 2) + 300.0 * np.exp(-((lon + 80.0) / 4.0) ** 2)
    elev_m = np.clip(elev_m + rng.normal(0, 150, n), 0, None)
    names = [f"SYN_Station_{i + 1:03d}" for i in range(n)]
    return pd.DataFrame({'City': names, 'Longitude': np.round(lon, 4), 'Latitude': np.round(lat, 4),
                         'Elevation': np.round(elev_m, 1)})

############################################################################################
def buildWeather(data={}, stations=None, rng=None):
    """All hourly fields as (stations x hours) arrays."""
    ########################################################################################
    S, H = len(stations), data['n_hours']
    lat, lon = stations['Latitude'].to_numpy(), stations['Longitude'].to_numpy()
    elev_km = stations['Elevation'].to_numpy() / 1000.0
    t_utc = pd.Timestamp(data['start_date']) + pd.to_timedelta(np.arange(1, H + 1), unit='h')  # hour-ending
    doy = (t_utc.dayofyear.to_numpy() + t_utc.hour.to_numpy() / 24.0)[None, :]
    local_solar_hr = (t_utc.hour.to_numpy()[None, :] + lon[:, None] / 15.0) % 24
    ########################################################################################
    # Synoptic anomaly: AR(1) in time, correlated across stations by distance
    cov = np.exp(-haversineKm(lat, lon) / data['spatial_corr_km'])
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(S))
    phi = data['synoptic_persistence']
    shocks = L @ rng.standard_normal((S, H)) * np.sqrt(1 - phi ** 2)
    anom = np.empty((S, H)); anom[:, 0] = L @ rng.standard_normal(S)
    for h in range(1, H): anom[:, h] = phi * anom[:, h - 1] + shocks[:, h]
    anom *= data['synoptic_std_c']
    ########################################################################################
    # Temperature (deg C)
    annual_mean = 27.0 - data['lat_gradient_c_per_deg'] * (lat - 25.0) - data['lapse_rate_c_per_km'] * elev_km
    seasonal_amp = data['seasonal_amp_base_c'] + 0.4 * (lat - 25.0)
    seasonal = seasonal_amp[:, None] * np.cos(2 * np.pi * (doy - 200.0) / 365.25)
    diurnal = data['diurnal_amp_c'] * np.cos(2 * np.pi * (local_solar_hr - 15.0) / 24.0)
    ########################################################################################
    # Rain: Markov chain, colder-than-normal hours a bit more likely to turn wet
    wet = np.zeros((S, H), dtype=bool)
    u = rng.random((S, H))
    p01 = np.clip(data['p_dry_to_wet'] * (1.0 - 0.15 * anom / max(data['synoptic_std_c'], 1e-6)), 0, 1)
    for h in range(1, H):
        wet[:, h] = np.where(wet[:, h - 1], u[:, h] < data['p_wet_to_wet'], u[:, h] < p01[:, h])
    shape = 0.6
    precip = np.where(wet, rng.gamma(shape, data['rain_mean_mm'] / shape, (S, H)), 0.0)
    ########################################################################################
    diurnal_eff = diurnal * np.where(wet, 0.4, 1.0)            # cloudy hours have a flatter daily cycle
    t_avg = annual_mean[:, None] + seasonal + diurnal_eff + anom
    t_max = t_avg + np.abs(rng.normal(0.5, 0.25, (S, H)))
    t_min = t_avg - np.abs(rng.normal(0.5, 0.25, (S, H)))
    t_calc = t_avg + rng.normal(0, 0.15, (S, H))
    ########################################################################################
    # Solar (W/m2): clear sky from solar elevation, reduced when wet
    decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81.0) / 365.0)
    hour_angle = np.radians(15.0 * (local_solar_hr - 12.0))
    la = np.radians(lat)[:, None]
    sin_elev = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(hour_angle)
    clear = 1000.0 * np.clip(sin_elev, 0, None) ** 1.15
    solar = clear * np.where(wet, rng.uniform(0.1, 0.35, (S, H)), rng.uniform(0.75, 1.0, (S, H)))
    ########################################################################################
    # Relative humidity from dewpoint spread
    spread = np.clip(6.0 + 0.04 * elev_km[:, None] * 100 + 0.5 * diurnal_eff + rng.normal(0, 1.5, (S, H)), 0.3, None)
    spread = np.where(wet, rng.uniform(0.0, 1.0, (S, H)), spread)
    td = t_avg - spread
    rh = 100.0 * np.exp(17.625 * td / (243.04 + td) - 17.625 * t_avg / (243.04 + t_avg))
    rh = np.clip(rh, 3.0, 100.0)
    ########################################################################################
    # Surface and soil temperature
    sur = t_avg + 0.012 * solar - 1.0 + rng.normal(0, 0.4, (S, H))
    soil_t = {d: emaAlongTime(sur, tau) for d, tau in [(5, 6), (10, 12), (20, 30), (50, 120), (100, 360)]}
    ########################################################################################
    # Soil moisture (m3/m3): bucket per depth
    soil_m = {}
    for d, (gain, decay) in {5: (0.010, 0.010), 10: (0.006, 0.006), 20: (0.0035, 0.003),
                             50: (0.0015, 0.0012), 100: (0.0008, 0.0005)}.items():
        base = 0.12 + 0.02 * rng.standard_normal(S)
        sm = np.empty((S, H)); sm[:, 0] = base + 0.05
        for h in range(1, H):
            sm[:, h] = sm[:, h - 1] + gain * precip[:, h] - decay * (sm[:, h - 1] - base)
        soil_m[d] = np.clip(sm, 0.02, 0.5)
    ########################################################################################
    return {'t_utc': t_utc, 't_calc': t_calc, 't_avg': t_avg, 't_max': t_max, 't_min': t_min, 'precip': precip,
            'solar': solar, 'rh': rh, 'sur': sur, 'soil_t': soil_t, 'soil_m': soil_m}

############################################################################################
def toStationDicts(data={}, stations=None, fields={}, rng=None):
    """Pack arrays into the cleaned USCRN dict format, then knock out values to mimic real gaps."""
    ########################################################################################
    S, H = len(stations), data['n_hours']
    t_utc = fields['t_utc']
    utc_date = t_utc.strftime('%Y%m%d').astype(float).to_numpy()
    utc_time = (t_utc.hour * 100).astype(float).to_numpy()
    out = {}
    for i, row in stations.iterrows():
        offset = int(round(row['Longitude'] / 15.0))
        t_lst = t_utc + pd.Timedelta(hours=offset)
        zeros = np.zeros(H)
        st = {
            'WBANNO': np.full(H, 90000.0 + i), 'UTC_DATE': utc_date, 'UTC_TIME': utc_time,
            'LST_DATE': t_lst.strftime('%Y%m%d').astype(float).to_numpy(), 'LST_TIME': (t_lst.hour * 100).astype(float).to_numpy(),
            'CRX_VN': np.full(H, 2.622), 'LONGITUDE': np.full(H, row['Longitude']), 'LATITUDE': np.full(H, row['Latitude']),
            'T_CALC': fields['t_calc'][i], 'T_HR_AVG': fields['t_avg'][i], 'T_MAX': fields['t_max'][i], 'T_MIN': fields['t_min'][i],
            'P_CALC': fields['precip'][i],
            'SOLARAD': fields['solar'][i], 'SOLARAD_FLAG': zeros,
            'SOLARAD_MAX': fields['solar'][i] * 1.12, 'SOLARAD_MAX_FLAG': zeros,
            'SOLARAD_MIN': fields['solar'][i] * 0.85, 'SOLARAD_MIN_FLAG': zeros,
            'SUR_TEMP_TYPE': ['C'] * H,
            'SUR_TEMP': fields['sur'][i], 'SUR_TEMP_FLAG': zeros,
            'SUR_TEMP_MAX': fields['sur'][i] + 0.8, 'SUR_TEMP_MAX_FLAG': zeros,
            'SUR_TEMP_MIN': fields['sur'][i] - 0.8, 'SUR_TEMP_MIN_FLAG': zeros,
            'RH_HR_AVG': fields['rh'][i], 'RH_HR_AVG_FLAG': zeros,
        }
        for d in (5, 10, 20, 50, 100):
            st[f'SOIL_MOISTURE_{d}'] = fields['soil_m'][d][i]
            st[f'SOIL_TEMP_{d}'] = fields['soil_t'][d][i]
        ####################################################################################
        # Random single-hour gaps in measured columns, plus a soil sensor outage on some stations
        measured = ['T_CALC', 'T_HR_AVG', 'T_MAX', 'T_MIN', 'P_CALC', 'SOLARAD', 'RH_HR_AVG', 'SUR_TEMP']
        for col in measured + [c for c in st if c.startswith('SOIL_')]:
            v = np.round(np.asarray(st[col], dtype=float), 3)
            v[rng.random(H) < data['missing_fraction']] = np.nan
            st[col] = v
        if rng.random() < data['soil_outage_fraction'] and H > 48:
            start = rng.integers(0, H - 24); length = int(rng.integers(24, min(24 * 14, H - start) + 1))
            for col in [c for c in st if c.startswith('SOIL_')]: st[col][start:start + length] = np.nan
        ####################################################################################
        out[row['City']] = {c: (st[c] if isinstance(st[c], list) else [None if np.isnan(x) else float(x) for x in st[c]])
                            for c in WEATHER_DATA_COLUMNS}
    return out

############################################################################################
def generateSyntheticWeather(inputData={}):
    """
    Build and write a synthetic station dataset.
    :param inputData: Flat dict (from standardInputWrapper). Missing keys fall back to generic defaults.
    :return:          (manifest dict, output folder)
    """
    ########################################################################################
    data = checkSyntheticInputs(inputData=inputData)
    rng = np.random.default_rng(data['random_seed'])
    out = data['output_dir'] if isabs(data['output_dir']) else join(REPO_ROOT, data['output_dir'])
    os.makedirs(out, exist_ok=True)
    print(f"Building {data['n_stations']} stations x {data['n_hours']} hours ({data['n_rows']:,} rows)")
    ########################################################################################
    stations = buildStations(data=data, rng=rng)
    fields = buildWeather(data=data, stations=stations, rng=rng)
    weather = toStationDicts(data=data, stations=stations, fields=fields, rng=rng)
    ########################################################################################
    with open(join(out, DATA_FILE), 'w') as f: json.dump(weather, f, allow_nan=False)
    stations.to_csv(join(out, STATIONS_FILE), index=False)
    manifest = {
        'name': data['name'], 'synthetic': True, 'random_seed': data['random_seed'],
        'files': {'data': DATA_FILE, 'stations': STATIONS_FILE},
        'start_utc': str(fields['t_utc'][0]), 'end_utc': str(fields['t_utc'][-1]),
        'n_stations': data['n_stations'], 'n_hours': data['n_hours'],
        'inputs': data,
    }
    with open(join(out, 'manifest.json'), 'w') as f: json.dump(manifest, f, indent=2, default=str)
    print(f"Synthetic weather data written to {out}")
    return manifest, out

############################################################################################
def main():
    thisDirName = dirname(abspath(__file__))
    fileName = sys.argv[1] if len(sys.argv) > 1 else join(thisDirName, '..', '..', 'tests', 'in_jsons', 'testCase01.json')
    with open(fileName) as json_file: inJson = json.load(json_file)
    generateSyntheticWeather(inputData=standardInputWrapper(inJson=inJson))

############################################################################################
if __name__ == "__main__":
    main()
############################################################################################
