import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    #parsing datetime to different categories of its own
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day 
    df["weekday"] = df["datetime"].dt.weekday 
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    #mapping seasong values as given in the kaggle website
    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    df["season_name"] = df["season"].map(season_map)

    #same for weather category
    weather_map = {1: "clear",2: "mist",3: "light_snow_rain",4: "heavy_rain_storm"}
    df["weather_desc"] = df["weather"].map(weather_map)

    #droping datetime cause we wont be using it
    df = df.drop(columns=["datetime"])

    #forward and backward fill to fill up the null values we are using
    #ffill() and bfill() directly without fillna cause its being depricated
    df = df.ffill().bfill()

    #encoding the categorical values we have introduced earlier 
    cat_cols = ["season_name", "weather_desc"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Drop original labeled categorical columns
    df = df.drop(columns=["season", "weather"])

    return df

def main():
    print("Loading raw dataset...")
    df = pd.read_csv('data/data.csv')

    print("Preprocessing...")
    cleaned = preprocess(df)

    cleaned.to_csv('data/cleaned.csv', index=False)
    print("Saved cleaned dataset → cleaned.csv")
    print(f"Final shape: {cleaned.shape}")


if __name__ == "__main__":
    main()
