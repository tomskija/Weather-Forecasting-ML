##########################################################
import os
import json
import time
import asyncio
import numpy as np
import pandas as pd
import mlflow
from os.path import dirname, join, isabs, abspath
from dataPrep.inputWrapper import standardInputWrapper
from dataPrep.outputWrapper import standardOutputWrapper
from dataPrep.mlFlowConfig import setupMLFlow
from dataPrep.dataChecks import checkWeatherData, DATA_FILE, STATIONS_FILE
from dataPrep.syntheticData import generateSyntheticWeather, REPO_ROOT

PACKAGE_DIR = dirname(abspath(__file__))
PROBLEM_TYPES = ['PROB01', 'PROB02', 'PROB03', 'PROB04']


##########################################################
def checkData(inputData={}):
    """
    Calculator-level input checks. Synthetic generator inputs are checked inside
    generateSyntheticWeather (checkSyntheticInputs).
    data_source:
        synthetic  build a synthetic station set (no network)
        cached     read weatherDataJSONObject.json + Weather Stations Info.csv from data_dir
        noaa       pull USCRN hourly data from NOAA with the async parser (network, slow)
    """
    ######################################################
    inputData["name"] = inputData.get("name", "Weather_Results")
    inputData["problemType"] = str(inputData.get("problemType", "PROB01")).upper()
    inputData["data_source"] = str(inputData.get("data_source", "synthetic")).lower()
    inputData["max_nan_fraction"] = float(inputData.get("max_nan_fraction", 0.2))
    ######################################################
    if inputData["problemType"] not in PROBLEM_TYPES:
        raise ValueError(f"problemType must be one of {PROBLEM_TYPES}, got {inputData['problemType']!r}")
    if inputData["data_source"] not in ("synthetic", "cached", "noaa"):
        raise ValueError(f"data_source must be 'synthetic', 'cached' or 'noaa', got {inputData['data_source']!r}")
    if inputData["data_source"] == "cached":
        d = inputData.get("data_dir", join(PACKAGE_DIR, "Data"))
        inputData["data_dir"] = d if isabs(d) else join(REPO_ROOT, d)
        for f in (DATA_FILE, STATIONS_FILE):
            if not os.path.exists(join(inputData["data_dir"], f)):
                raise ValueError(f"cached data_source needs {f} in {inputData['data_dir']}")
    if inputData["data_source"] == "noaa":
        years = inputData.get("idealDates", [])
        if not years or not all(1990 <= int(y) <= 2100 for y in years):
            raise ValueError("noaa data_source needs idealDates, a list of years")
        inputData["idealDates"] = [int(y) for y in years]
        inputData["batchSize"] = int(inputData.get("batchSize", 20))
        inputData["booleanRunSeriesVsParallel"] = int(inputData.get("booleanRunSeriesVsParallel", 0))
        inputData["parseDataBool"] = 1
    ######################################################
    return inputData


##########################################################
async def getData(inputData={}):
    """Returns (weather dict, stations DataFrame, folder the data lives in)."""
    ######################################################
    if inputData["data_source"] == "synthetic":
        _, data_dir = generateSyntheticWeather(inputData=inputData)
    elif inputData["data_source"] == "cached":
        data_dir = inputData["data_dir"]
    else:
        # Network pull, writes to weatherForcastingCalculator/Data like before
        from utils.dataPrepAndParser import getCleanedDataStructure
        await getCleanedDataStructure(inputData=inputData, dirname=PACKAGE_DIR)
        data_dir = join(PACKAGE_DIR, "Data")
    ######################################################
    with open(join(data_dir, DATA_FILE)) as f:
        weatherDataDictObjectCleaned = json.load(f)
    dfStations = pd.read_csv(join(data_dir, STATIONS_FILE))
    return weatherDataDictObjectCleaned, dfStations, data_dir


##########################################################
async def calculate(inJson={}):
    ######################################################
    setupMLFlow()
    ######################################################
    try:
        inputData = standardInputWrapper(inJson=inJson)
        output_wrapper = standardOutputWrapper()
        inputData = checkData(inputData=inputData)
        startTime = time.time()
        ##################################################
        with mlflow.start_run(run_name=f"pipeline_{inputData['name']}") as run:
            mlflow.set_tags({"data_source": inputData["data_source"], "problemType": inputData["problemType"]})
            weatherDataDictObjectCleaned, dfStations, data_dir = await getData(inputData=inputData)
            ##############################################
            report = checkWeatherData(dataDir=data_dir, max_nan_fraction=inputData["max_nan_fraction"])
            mlflow.log_params({"n_stations": report["summary"].get("n_stations", 0),
                               "n_rows": report["summary"].get("n_rows", 0)})
            mlflow.log_metrics({"data_check_passed": float(report["passed"]),
                                "n_errors": len(report["errors"]), "n_warnings": len(report["warnings"])})
            if not report["passed"]:
                raise ValueError("Data check failed:\n  - " + "\n  - ".join(report["errors"][:20]))
            ##############################################
            if inputData["problemType"] == "PROB01":
                # Feature analysis (correlation / uncertainty / feature importance stats and plots)
                pass
            if inputData["problemType"] == "PROB02":
                # Geospatial modeling to get a 2D/3D map, then predict away from stations
                pass
            if inputData["problemType"] == "PROB03":
                # Time series forecasting with standard ML approaches
                pass
            if inputData["problemType"] == "PROB04":
                # Further methods, e.g. physics informed NNs in a Bayesian framework
                pass
            ##############################################
            output_wrapper.add_param(name="data_check_passed", value=report["passed"], value_type=1)
            output_wrapper.add_param(name="n_stations", value=len(weatherDataDictObjectCleaned))
            output_wrapper.add_param(name="data_dir", value=data_dir, value_type=3)
            output_wrapper.add_param(name="mlflow_run_id", value=run.info.run_id, value_type=3)
            output_wrapper.add_table(name="stations", array=dfStations["City"].astype(str).tolist())
            output_wrapper.add_table(name="warnings", array=report["warnings"])
        ##################################################
        print("The total numerical time is " + str(np.round(time.time() - startTime, 5)) + " seconds")
        print("Completed Calculation")
        return output_wrapper
    except Exception as e:
        print("\nError Message: " + str(e) + "\n")
        return str(e)


##########################################################
async def main():
    ######################################################
    fileDirName = join(dirname(abspath(__file__)), "..", "tests", "in_jsons", "testCase01.json")
    with open(fileDirName) as json_file:
        inJson = json.load(json_file)
    output_wrapper = await calculate(inJson=inJson)
    if not isinstance(output_wrapper, str):
        print(json.dumps(output_wrapper.flatten(), indent=2))


##########################################################
if __name__ == "__main__":
    asyncio.run(main())
##########################################################
