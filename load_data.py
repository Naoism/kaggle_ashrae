import pandas as pd
import numpy as np
import datetime
from meteocalc import feels_like, Temp
from logging import getLogger

TRAIN_DATA = '../input/train.csv'
TEST_DATA = "../input/test.csv"
BUILDING_DATA = "../input/building_metadata.csv"
WEATHER_TRAIN_DATA = "../input/weather_train.csv"
WEATHER_TEST_DATA = "../input/weather_test.csv"

logger = getLogger(__name__)


def read_csv(path):
    df = pd.read_csv(path)
    return df


def load_train_data():
    logger.debug("enter")
    df = pd.read_csv(TRAIN_DATA)
    logger.debug("exit")
    return df


def load_test_data():
    logger.debug("enter")
    df = pd.read_csv(TEST_DATA)
    logger.debug("exit")
    return df


def load_building_data():
    logger.debug("enter")
    df = pd.read_csv(BUILDING_DATA)
    logger.debug("exit")
    return df


def load_weather_train_data():
    logger.debug("enter")
    df = pd.read_csv(WEATHER_TRAIN_DATA)
    logger.debug("exit")
    return df


def load_weather_test_data():
    logger.debug("enter")
    df = pd.read_csv(WEATHER_TEST_DATA)
    logger.debug("exit")
    return df


def load_fill_weather_data(weather_df):
    
    logger.debug("enter")
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])

        weather_df = weather_df.reset_index(drop=True)

    # FIX Time Zone
    zone_dict={0:4,1:0,2:7,3:4,4:7,5:0,6:4,7:4,8:4,9:5,10:7,11:4,12:0,13:5,14:4,15:4}
    def set_localtime(df):
        for sid, zone in zone_dict.items():
            sids = df.site_id == sid
            df.loc[sids, 'timestamp'] = df[sids].timestamp - pd.offsets.Hour(zone)
        

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month

    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)


    def get_meteorological_features(data):

        
        def calculate_rh(df):
            df['relative_humidity'] = 100 * (np.exp((17.625 * df['dew_temperature']) / (243.04 + df['dew_temperature'])) / np.exp((17.625 * df['air_temperature'])/(243.04 + df['air_temperature'])))

            
        def calculate_fl(df):
            flike_final = []
            flike = []
            # calculate Feels Like temperature
            for i in range(len(df)):
                at = df['air_temperature'][i]
                rh = df['relative_humidity'][i]
                ws = df['wind_speed'][i]
                flike.append(feels_like(Temp(at, unit = 'C'), rh, ws))
            for i in range(len(flike)):
                flike_final.append(flike[i].f)
            df['feels_like'] = flike_final
            del flike_final, flike, at, rh, ws
        calculate_rh(data)
        calculate_fl(data)
        return data

    
    weather_df = get_meteorological_features(weather_df)

    logger.debug("exit")

    return weather_df


if __name__ == "__main__":
    weather_df = load_weather_train_data()

    weather_df = load_fill_weather_data(weather_df)

    print(weather_df.head())
    print(weather_df.isnull().sum())
