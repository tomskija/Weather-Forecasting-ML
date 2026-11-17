# Weather-Forecasting-ML

**By: Jackson R. Tomski**

This project pulls hourly weather station data from NOAA's U.S. Climate Reference Network (USCRN), cleans it into one structure per station, and sets it up for ML work: feature analysis, spatial interpolation between stations, and time series forecasting.

The data side is built around an async pipeline that pulls every station for every year you ask for in parallel. There's also a synthetic station generator so the whole thing can run and be tested without hitting NOAA, and so models can be checked against data where the physics behind it is known.

![Weather Stations Map](weatherForcastingCalculator/Figures/Weather%20Stations%20Map.png)

---

## What's Inside

### NOAA Data Pipeline

`utils/dataPrepAndParser.py` scrapes the USCRN hourly02 archive. For each year it lists the station files, then pulls them in batches with `aiohttp` and `asyncio.gather`. The parallel version runs in about 500 seconds versus about 700 for the original serial version.

After the pull:

1. **Common stations only** across all years, so every station has a full record.
2. **Stacked by station**, so each station is one dictionary of 38 USCRN columns covering every year.
3. **Cleaned**, with the USCRN missing codes (-99, -9999) replaced by NaN.
4. **Station info** (name, lat, lon, elevation) saved to CSV and plotted on the map above. Elevation comes from the Open-Elevation API.

The result is cached as `weatherDataJSONObject.json` plus `Weather Stations Info.csv`, so you only pull from NOAA once.

### Synthetic Station Generator

`dataPrep/syntheticData.py` writes the exact same cached format, so everything downstream can't tell the difference. The fields are simple but connected to each other:

| Field | How it's built |
|-------|----------------|
| Temperature | Climate from latitude and elevation (lapse rate), plus seasonal and daily cycles, plus a weather anomaly that is correlated in time and shared by nearby stations |
| Rain | Wet/dry spells from a Markov chain, gamma-distributed amounts, a little more likely when it's colder than normal |
| Solar | Clear sky from the sun's elevation, cut down when it's raining |
| Humidity | From temperature and dewpoint, higher when it's raining |
| Surface and soil temperature | Surface heated by the sun, soil temperature lagged and damped with depth |
| Soil moisture | A bucket at each depth that fills with rain and drains back down, slower when deeper |

It also adds random missing hours and multi-day soil sensor outages on some stations, since real USCRN data has both.

### Data Checks

**`checkWeatherData`** runs on any cached dataset, real or synthetic. It checks the 38 columns are present and the same length, timestamps are hourly and increasing, values are physically reasonable (temperature, rain, solar, humidity, soil moisture, lat/lon), T_MIN isn't above T_MAX, and core columns aren't mostly missing. Soil gaps are a warning, not an error, since a lot of real stations are missing soil data. **`checkSyntheticInputs`** validates the generator inputs and reports every problem at once.

### Calculator

`Calculator.py` follows the same flow as Production-Prediction-ML:

```
input JSON → checkData → get data → checkWeatherData → MLflow logging → output wrapper
```

`data_source` can be:

| `data_source` | What it does |
|---------------|--------------|
| `synthetic` | Builds a synthetic station set (default, no network) |
| `cached` | Loads an existing JSON and CSV from `data_dir` |
| `noaa` | Pulls from NOAA for the years in `idealDates` (needs internet, slow) |

---

## Problem Types

| Type | What it is |
|------|------------|
| `PROB01` | Feature analysis: correlation, uncertainty, feature importance |
| `PROB02` | Geospatial modeling: interpolate between stations to predict at unobserved locations |
| `PROB03` | Time series forecasting: classical baselines through ML forecasters |
| `PROB04` | More advanced methods, e.g. physics-informed NNs in a Bayesian framework |

Set `problemType` in the input JSON to pick one.

---

## Getting Started

**You need:** Docker Desktop and VS Code with the Dev Containers extension.

```bash
git clone https://github.com/tomskija/Weather-Forecasting-ML.git
cd Weather-Forecasting-ML
```

Open the folder in VS Code and select **Reopen in Container**. Docker Compose starts three services: the calculator, its FastAPI server, and an MLflow tracking server.

**Run the calculator:**
```bash
python weatherForcastingCalculator/Calculator.py
```

**Run the test suite:**
```bash
pytest tests -v
```

**Without VS Code:**
```bash
docker compose up -d --build
docker compose exec calculator pytest tests -v
docker compose exec calculator python weatherForcastingCalculator/Calculator.py
```

| Service   | URL                   |
|-----------|-----------------------|
| API       | http://localhost:8002  |
| API Docs  | http://localhost:8002/docs |
| MLflow UI | http://localhost:5002 |

MLflow is on 5002 so it can run next to my other repos.

---

## API Endpoints

| Method | Endpoint                | Description                                        |
|--------|-------------------------|----------------------------------------------------|
| `GET`  | `/health`               | Check if the API is running                        |
| `GET`  | `/ready`                | Check the calculator and its schema load           |
| `GET`  | `/schema`               | Input form definition (fields, types, defaults, groups) used by the UI |
| `POST` | `/calculate`            | Submit a run with `{"inJson": {...}}`, returns `job_id` immediately |
| `GET`  | `/runs/{job_id}/status` | Poll run status: pending / queued / running / complete / failed |
| `GET`  | `/jobs/{job_id}/output` | Output params and tables once the run is complete |
| `GET`  | `/jobs`                 | List every job the API has seen                    |

Runs go one at a time per calculator in a background thread, and each API run writes to its own folder under `Data/Synthetic/` so runs don't overwrite each other. The full interactive docs are at `http://localhost:8002/docs`. This is the same contract as Production-Prediction-ML, which is what lets [ML-Platform-UI](https://github.com/tomskija/ML-Platform-UI) drive every calculator the same way.

---

## Configuration

Inputs use the same `params` / `tables` JSON format as Production-Prediction-ML. Anything left out falls back to a default.

| Parameter | What it does |
|-----------|--------------|
| `problemType` | `PROB01` to `PROB04` |
| `data_source` | `synthetic`, `cached` or `noaa` |
| `data_dir` | Folder with the cached JSON and CSV (for `cached`) |
| `idealDates` | List of years to pull (for `noaa`) |
| `batchSize` | Stations pulled at once per year (for `noaa`) |
| `n_stations`, `start_date`, `n_days` | Size and time range of the synthetic set |
| `lat_min`, `lat_max`, `lon_min`, `lon_max` | Where synthetic stations go |
| `synoptic_std_c`, `spatial_corr_km` | How big weather anomalies are and how far they reach |
| `p_dry_to_wet`, `p_wet_to_wet`, `rain_mean_mm` | Rain behavior |
| `missing_fraction`, `soil_outage_fraction` | How gappy the synthetic data is |
| `random_seed` | Same seed, same dataset |

See `tests/in_jsons/testCase01.json` for an example. Generated data goes to `Data/Synthetic/`, which is gitignored.

---

## Project Structure

```
Weather-Forecasting-ML/
├── .devcontainer/devcontainer.json     # Dev container config
├── .github/workflows/ci.yml            # Builds the Docker stack, runs pytest and the calculator
├── weatherForcastingCalculator/
│   ├── Calculator.py                   # Orchestrator
│   ├── dataPrep/
│   │   ├── syntheticData.py            # Synthetic station generator
│   │   ├── schema.py                   # Input form definition served at /schema
│   │   ├── dataChecks.py               # Input and dataset checks
│   │   ├── inputWrapper.py             # JSON input parser
│   │   ├── outputWrapper.py            # Output wrapper
│   │   └── mlFlowConfig.py             # MLflow (Docker server or local SQLite)
│   ├── utils/
│   │   ├── dataPrepAndParser.py        # Async NOAA USCRN pull
│   │   ├── dataCleaning.py             # Missing codes, station info, map
│   │   └── utils.py                    # JSON helpers
│   ├── Data/weatherDataColumnInfo.xlsx # USCRN column descriptions
│   └── Figures/                        # Station map
├── tests/
│   ├── in_jsons/                       # pytest input configs
│   ├── test_synthetic_data.py
│   ├── test_api.py
│   └── test_calculator.py
├── api.py                              # FastAPI, same calculator contract as PPML
├── conftest.py                         # pytest path config
├── Dockerfile                          # Python 3.11
├── Dockerfile.mlflow                   # MLflow server
├── docker-compose.yml                  # calculator, api, mlflow
└── requirements.txt
```

---

## Data

Real data comes from NOAA's [U.S. Climate Reference Network](https://www.ncei.noaa.gov/access/crn/) hourly product. Column definitions are in `weatherForcastingCalculator/Data/weatherDataColumnInfo.xlsx`. Pulled and synthetic data are both gitignored.

---

## Dependencies

Python 3.11 · numpy · pandas · scipy · scikit-learn · matplotlib · seaborn · aiohttp · beautifulsoup4 · requests · openpyxl · mlflow · pytest · fastapi · uvicorn · pydantic
