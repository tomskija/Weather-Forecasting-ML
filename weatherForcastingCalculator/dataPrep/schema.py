############################################################################################
"""
Input form definition for the UI (served at GET /schema).

Defaults come straight from the data check defaults so there's one source of truth.
Field types:
    number, integer, boolean, string, select     sent as params
    list_number, list_string                     sent as tables
show_if hides a field unless another field has one of the listed values.
"""
############################################################################################
try:
    from dataPrep.dataChecks import SYNTHETIC_DEFAULTS as D
except ImportError:
    from .dataChecks import SYNTHETIC_DEFAULTS as D

############################################################################################
def field(name='', label='', type='number', default=None, help='', options=None, min=None, max=None, show_if=None):
    f = {'name': name, 'label': label or name, 'type': type, 'default': default,
         'kind': 'table' if type.startswith('list_') else 'param'}
    if help: f['help'] = help
    if options is not None: f['options'] = options
    if min is not None: f['min'] = min
    if max is not None: f['max'] = max
    if show_if: f['show_if'] = show_if
    return f

############################################################################################
def getSchema():
    synthetic = {'data_source': ['synthetic']}
    return {
        'name': 'weather',
        'title': 'Weather Forecasting',
        'description': 'Hourly weather station data from NOAA USCRN or a synthetic network, set up for feature analysis, spatial modeling and forecasting.',
        'groups': [
            {'name': 'Run', 'fields': [
                field('name', 'Run name', 'string', 'Weather_Results'),
                field('problemType', 'Problem type', 'select', 'PROB01', options=['PROB01', 'PROB02', 'PROB03', 'PROB04'],
                      help='PROB01 feature analysis, PROB02 geospatial, PROB03 forecasting, PROB04 advanced'),
                field('data_source', 'Data source', 'select', 'synthetic', options=['synthetic', 'cached', 'noaa']),
                field('data_dir', 'Cached data folder', 'string', '', show_if={'data_source': ['cached']},
                      help='Folder with weatherDataJSONObject.json and Weather Stations Info.csv'),
                field('idealDates', 'Years to pull', 'list_number', [2024], show_if={'data_source': ['noaa']}),
                field('batchSize', 'Stations per batch', 'integer', 20, min=1, show_if={'data_source': ['noaa']}),
                field('random_seed', 'Random seed', 'integer', D['random_seed'], show_if=synthetic),
            ]},
            {'name': 'Stations', 'fields': [
                field('n_stations', 'Stations', 'integer', D['n_stations'], min=1, max=999, show_if=synthetic),
                field('start_date', 'Start date', 'string', D['start_date'], show_if=synthetic, help='YYYY-MM-DD'),
                field('n_days', 'Days', 'integer', D['n_days'], min=1, show_if=synthetic),
                field('lat_min', 'Latitude min', 'number', D['lat_min'], show_if=synthetic),
                field('lat_max', 'Latitude max', 'number', D['lat_max'], show_if=synthetic),
                field('lon_min', 'Longitude min', 'number', D['lon_min'], show_if=synthetic),
                field('lon_max', 'Longitude max', 'number', D['lon_max'], show_if=synthetic),
            ]},
            {'name': 'Climate', 'fields': [
                field('lapse_rate_c_per_km', 'Lapse rate (C/km)', 'number', D['lapse_rate_c_per_km'], show_if=synthetic),
                field('lat_gradient_c_per_deg', 'Latitude gradient (C/deg)', 'number', D['lat_gradient_c_per_deg'], show_if=synthetic),
                field('diurnal_amp_c', 'Daily swing (C)', 'number', D['diurnal_amp_c'], min=0, show_if=synthetic),
                field('synoptic_std_c', 'Weather anomaly size (C)', 'number', D['synoptic_std_c'], min=0, show_if=synthetic),
                field('synoptic_persistence', 'Anomaly persistence (hourly)', 'number', D['synoptic_persistence'], min=0, max=0.999, show_if=synthetic),
                field('spatial_corr_km', 'Anomaly reach (km)', 'number', D['spatial_corr_km'], min=1, show_if=synthetic),
                field('p_dry_to_wet', 'Chance dry hour turns wet', 'number', D['p_dry_to_wet'], min=0, max=1, show_if=synthetic),
                field('p_wet_to_wet', 'Chance wet hour stays wet', 'number', D['p_wet_to_wet'], min=0, max=1, show_if=synthetic),
                field('rain_mean_mm', 'Mean hourly rain (mm)', 'number', D['rain_mean_mm'], min=0, show_if=synthetic),
            ]},
            {'name': 'Data Quality', 'fields': [
                field('missing_fraction', 'Missing hours (fraction)', 'number', D['missing_fraction'], min=0, max=0.5, show_if=synthetic),
                field('soil_outage_fraction', 'Stations with soil outage', 'number', D['soil_outage_fraction'], min=0, max=0.5, show_if=synthetic),
                field('max_nan_fraction', 'Max missing allowed in checks', 'number', 0.2, min=0, max=1),
            ]},
        ],
    }
############################################################################################
