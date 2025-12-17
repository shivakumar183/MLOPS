from pydantic import BaseModel
from typing import List

class BikeRecord(BaseModel):
    holiday: int
    workingday: int
    temp: float
    atemp: float
    humidity: float
    windspeed: float
    casual: float
    registered: float
    hour: int
    day: int
    weekday: int
    month: int
    year: int
    season_name_spring: bool
    season_name_summer: bool
    season_name_winter: bool
    weather_desc_heavy_rain_storm: bool
    weather_desc_light_snow_rain: bool
    weather_desc_mist: bool

class BatchRequest(BaseModel):
    records: List[BikeRecord]
