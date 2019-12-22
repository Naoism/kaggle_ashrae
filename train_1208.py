import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from meteocalc import feels_like, Temp
import datetime
import gc
from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
from load_data import load_train_data, load_test_data, load_building_data, load_weather_train_data, load_weather_test_data, load_fill_weather_data
from reduce_memory import reduce_mem_usage

logger = getLogger(__name__)

DIR = "result_tmp/"

if __name__ == "__main__":

    
    log_fmt = Formatter("%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ")
    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + "train.py.log", "a")
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info("start")

    train_df = load_train_data()
    building_df = load_building_data()
    weather_df = load_weather_train_data()
    

    ## Remove outliers
    #train_df = train_df [ train_df['building_id'] != 1099 ]
    # All electricity meter is 0 until May 20 for site_id == 0. I will remove these data from training data.
    train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    zone_dict={0:4,1:0,2:7,3:4,4:7,5:0,6:4,7:4,8:4,9:5,10:7,11:4,12:0,13:5,14:4,15:4}

            
    def add_lag_feature(weather_df, window=3):
        group_df = weather_df.groupby('site_id')
        cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
        rolled = group_df[cols].rolling(window=window, min_periods=0)
        lag_mean = rolled.mean().reset_index().astype(np.float16)
        lag_max = rolled.max().reset_index().astype(np.float16)
        lag_min = rolled.min().reset_index().astype(np.float16)
        lag_std = rolled.std().reset_index().astype(np.float16)
        for col in cols:
            weather_df['{}_mean_lag(window)'.format(col)] = lag_mean[col]
            weather_df['{}_max_lag(window)'.format(col)] = lag_max[col]
            weather_df['{}_min_lag(window)'.format(col)] = lag_min[col]
            weather_df['{}_std_lag(window)'.format(col)] = lag_std[col]
    

    def features_engineering(df):
        # Sort by timestamp
        df.sort_values("timestamp")
        df.reset_index(drop=True)

        # Add more features
        df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.weekday
        holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04","2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26", "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04", "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25", "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04", "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25", "2019-01-01"]
        df["group"] = df["timestamp"].dt.month
        df["group"].replace((1, 2, 3, 4), 1, inplace=True)
        df["group"].replace((5, 6, 7, 8), 2, inplace=True)
        df["group"].replace((9, 10, 11, 12), 3, inplace=True)
        df["is_holiday"] = (df.timestamp.isin(holidays)).astype(int)
        df['square_feet'] =  np.log1p(df['square_feet'])

        # Remove Unused Columns
        drop = ["timestamp"]
        df = df.drop(drop, axis=1)
        gc.collect()

        # Encode Categorical Data
        le = LabelEncoder()
        df["primary_use"] = le.fit_transform(df["primary_use"])

        return df


    weather_df = load_fill_weather_data(weather_df)

    add_lag_feature(weather_df)

    train_df = reduce_mem_usage(train_df,use_float16=True)
    building_df = reduce_mem_usage(building_df,use_float16=True)
    weather_df = reduce_mem_usage(weather_df,use_float16=True)


    train_df = train_df.merge(building_df, left_on='building_id',right_on='building_id',how='left')
    train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    del weather_df
    gc.collect()

    
    train_df = features_engineering(train_df)


    target = np.log1p(train_df["meter_reading"])
    features = train_df.drop('meter_reading', axis = 1)
    del train_df
    gc.collect()

    use_cols = features.columns.values

    logger.debug("train columns: {} {}".format(use_cols.shape, use_cols))
    logger.info("data preparation end {}".format(features.shape))

    categorical_features = ["building_id", "site_id", "meter", "primary_use", "is_holiday", "dayofweek"]
    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 1280,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse",
    }

    kf = KFold(n_splits=3)
    models = []
    list_rmse = [] 
    
    for train_index,test_index in tqdm(kf.split(features)):
        train_features = features.loc[train_index]
        train_target = target.loc[train_index]

        test_features = features.loc[test_index]
        test_target = target.loc[test_index]

        d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_features, free_raw_data=False)
        d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_features, free_raw_data=False)

        model = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)
        models.append(model)

        y_pred = model.predict(test_features, num_iteration=model.best_iteration)

        rmse_ = np.sqrt(mean_squared_error(test_target, y_pred))
        list_rmse.append(rmse_)
    
        logger.info("rmse:{}".format(rmse_))
        del train_features, train_target, test_features, test_target, d_training, d_test
        gc.collect()

    del features, target
    gc.collect()

    logger.info("train end")


    test_df = load_test_data()
    row_ids = test_df["row_id"]
    test_df.drop("row_id", axis=1, inplace=True)
    test_df = reduce_mem_usage(test_df)

    test_df = test_df.merge(building_df,left_on='building_id',right_on='building_id',how='left')
    del building_df
    gc.collect()


    weather_df = load_weather_test_data()
    weather_df = load_fill_weather_data(weather_df)
    add_lag_feature(weather_df)
    weather_df = reduce_mem_usage(weather_df)


    test_df = test_df.merge(weather_df,how='left',on=['timestamp','site_id'])
    del weather_df
    gc.collect()

    test_df = features_engineering(test_df)

    logger.info("data preparation end{}".format(test_df.shape))

    results = []
    for model in models:
        if  results == []:
            results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
        else:
            results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
            del model
            gc.collect()

    del test_df, models
    gc.collect()

    results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None)})
    del row_ids,results
    gc.collect()
    results_df.to_csv(DIR + "submission_1208.csv", index=False)

    logger.info("end")
