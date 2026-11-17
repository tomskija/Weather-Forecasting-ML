"""
Tests for the synthetic weather generator and the weather data checks.
"""
import os
import json
import numpy as np
import pandas as pd
import pytest

from dataPrep.dataChecks import checkSyntheticInputs, checkWeatherData, WEATHER_DATA_COLUMNS, DATA_FILE, STATIONS_FILE
from dataPrep.syntheticData import generateSyntheticWeather


@pytest.fixture(scope='module')
def tinyWeather(tmp_path_factory):
    out = tmp_path_factory.mktemp('weather')
    manifest, data_dir = generateSyntheticWeather({'output_dir': str(out), 'n_stations': 5, 'n_days': 10})
    with open(os.path.join(data_dir, DATA_FILE)) as f: weather = json.load(f)
    return manifest, data_dir, weather


# Input checks ##############################################################################
def test_defaults_pass():
    data = checkSyntheticInputs({})
    assert data['n_hours'] == 90 * 24


@pytest.mark.parametrize('overrides, message', [
    ({'start_date': '01/01/2024'}, 'YYYY-MM-DD'),
    ({'lat_min': 50, 'lat_max': 40}, 'lat_min < lat_max'),
    ({'p_wet_to_wet': 1.5}, 'p_wet_to_wet'),
    ({'n_stations': 0}, 'n_stations'),
    ({'n_stations': 500, 'n_days': 365}, 'allow_large'),
])
def test_bad_inputs_raise(overrides, message):
    with pytest.raises(ValueError, match=message):
        checkSyntheticInputs(overrides)


# Generated data ###########################################################################
def test_passes_checks(tinyWeather):
    _, data_dir, _ = tinyWeather
    report = checkWeatherData(data_dir)
    assert report['passed'], report['errors']
    assert report['summary']['n_stations'] == 5


def test_uscrn_layout(tinyWeather):
    _, data_dir, weather = tinyWeather
    st = pd.read_csv(os.path.join(data_dir, STATIONS_FILE))
    assert list(st.columns) == ['City', 'Longitude', 'Latitude', 'Elevation']
    assert set(st['City']) == set(weather)
    for station in weather.values():
        assert list(station) == WEATHER_DATA_COLUMNS
        assert len({len(v) for v in station.values()}) == 1


def test_physical_relationships(tinyWeather):
    _, _, weather = tinyWeather
    s = {k: np.asarray(v, dtype=float) for k, v in next(iter(weather.values())).items() if k != 'SUR_TEMP_TYPE'}
    ok = ~np.isnan(s['T_MIN']) & ~np.isnan(s['T_MAX'])
    assert np.all(s['T_MIN'][ok] <= s['T_MAX'][ok])
    assert np.nanmin(s['P_CALC']) >= 0 and np.nanmax(s['RH_HR_AVG']) <= 100
    hour = s['LST_TIME'] // 100
    night = (hour <= 3) | (hour >= 22)
    assert np.nanmax(s['SOLARAD'][night]) < 50
    assert np.nanstd(s['SOIL_TEMP_100']) < np.nanstd(s['T_HR_AVG'])


def test_same_seed_same_data(tmp_path):
    _, a = generateSyntheticWeather({'output_dir': str(tmp_path / 'a'), 'n_stations': 3, 'n_days': 3})
    _, b = generateSyntheticWeather({'output_dir': str(tmp_path / 'b'), 'n_stations': 3, 'n_days': 3})
    assert open(os.path.join(a, DATA_FILE)).read() == open(os.path.join(b, DATA_FILE)).read()


def test_check_catches_bad_values(tinyWeather, tmp_path):
    _, data_dir, weather = tinyWeather
    bad = json.loads(json.dumps(weather))
    name = next(iter(bad))
    bad[name]['RH_HR_AVG'] = [150.0] * len(bad[name]['RH_HR_AVG'])
    del bad[name]['T_MAX']
    other = list(bad)[1]
    bad[other]['RH_HR_AVG'] = [150.0] * len(bad[other]['RH_HR_AVG'])
    with open(tmp_path / DATA_FILE, 'w') as f: json.dump(bad, f)
    pd.read_csv(os.path.join(data_dir, STATIONS_FILE)).to_csv(tmp_path / STATIONS_FILE, index=False)
    report = checkWeatherData(str(tmp_path))
    assert not report['passed']
    assert any('missing columns' in e for e in report['errors'])
    assert any('RH_HR_AVG' in e for e in report['errors'])


def test_check_missing_files(tmp_path):
    report = checkWeatherData(str(tmp_path))
    assert not report['passed'] and 'missing file' in report['errors'][0]
