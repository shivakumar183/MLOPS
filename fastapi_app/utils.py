import pandas as pd
import h2o

CATEGORICAL_BOOL_COLS = [
    "season_name_spring",
    "season_name_summer",
    "season_name_winter",
    "weather_desc_heavy_rain_storm",
    "weather_desc_light_snow_rain",
    "weather_desc_mist",
]

FEATURE_COLUMNS = [
    "holiday","workingday","temp","atemp","humidity","windspeed",
    "casual","registered","hour","day","weekday","month","year",
    "season_name_spring","season_name_summer","season_name_winter",
    "weather_desc_heavy_rain_storm","weather_desc_light_snow_rain",
    "weather_desc_mist"
]

def convert_to_h2o(record_dict):
    df = pd.DataFrame([record_dict])
    df = df[FEATURE_COLUMNS]

    for col in CATEGORICAL_BOOL_COLS:
        df[col] = df[col].apply(lambda x: "True" if x else "False")

    hf = h2o.H2OFrame(df)

    for col in CATEGORICAL_BOOL_COLS:
        hf[col] = hf[col].asfactor()

    return hf


def convert_batch_to_h2o(list_of_dicts):
    df = pd.DataFrame(list_of_dicts)
    df = df[FEATURE_COLUMNS]

    for col in CATEGORICAL_BOOL_COLS:
        df[col] = df[col].apply(lambda x: "True" if x else "False")

    hf = h2o.H2OFrame(df)

    for col in CATEGORICAL_BOOL_COLS:
        hf[col] = hf[col].asfactor()

    return hf
