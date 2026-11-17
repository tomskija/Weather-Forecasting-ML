###############################################################################
"""
End-to-end tests of Calculator.calculate. Synthetic and cached modes only (no network).
"""
import json
import asyncio
import pytest
from os.path import dirname, join, abspath

from Calculator import calculate

IN_JSON = join(dirname(abspath(__file__)), 'in_jsons', 'testCase01.json')


def loadInJson(overrides={}):
    with open(IN_JSON) as f: inJson = json.load(f)
    params = {p['name']: p for p in inJson['data']['params']}
    for name, value in overrides.items():
        if name in params: params[name]['value'] = value
        else: inJson['data']['params'].append({'name': name, 'value': value, 'valueType': 2})
    return inJson


@pytest.fixture(autouse=True)
def mlflowStore(tmp_path, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'sqlite:///' + str(tmp_path / 'mlflow.db'))


###############################################################################
def test_synthetic_run(tmp_path):
    out = asyncio.run(calculate(inJson=loadInJson({'output_dir': str(tmp_path / 'w'), 'n_stations': 4, 'n_days': 5})))
    assert not isinstance(out, str), out
    result = out.flatten()
    assert result['data_check_passed'] is True and result['n_stations'] == 4


def test_cached_run(tmp_path):
    first = asyncio.run(calculate(inJson=loadInJson({'output_dir': str(tmp_path / 'w'), 'n_stations': 3, 'n_days': 3}))).flatten()
    out = asyncio.run(calculate(inJson=loadInJson({'data_source': 'cached', 'data_dir': first['data_dir']})))
    assert not isinstance(out, str), out
    assert out.flatten()['n_stations'] == 3


@pytest.mark.parametrize('overrides, message', [
    ({'problemType': 'PROB09'}, 'problemType'),
    ({'data_source': 'satellite'}, 'data_source'),
    ({'data_source': 'cached', 'data_dir': '/nope'}, 'cached data_source needs'),
    ({'data_source': 'noaa'}, 'idealDates'),
])
def test_bad_inputs_return_message(overrides, message):
    out = asyncio.run(calculate(inJson=loadInJson(overrides)))
    assert isinstance(out, str) and message in out
