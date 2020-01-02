import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from meteocalc import feels_like, Temp
import datetime
import gc
from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
from load_data import load_train_data, load_test_data, load_building_data, load_weather_train_data, load_weather_te\
st_data, load_fill_weather_data
from reduce_memory import reduce_mem_usage
seed = 1


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

# eliminate bad rows                                                                                            
    bad_rows = pd.read_csv('../input/rows_to_drop.csv')
    train_df.drop(bad_rows.loc[:, '0'], inplace = True)
    train_df.reset_index(drop = True, inplace = True)


    def add_lag_feature(weather_df, window):
        group_df = weather_df.groupby('site_id')
        cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
        rolled = group_df[cols].rolling(window=window, min_periods=0)
        lag_mean = rolled.mean().reset_index().astype(np.float16)
        lag_max = rolled.max().reset_index().astype(np.float16)
        lag_min = rolled.min().reset_index().astype(np.float16)
        lag_std = rolled.std().reset_index().astype(np.float16)
        lag_median = rolled.median().reset_index().astype(np.float16)
        lag_skew = rolled.skew().reset_index().astype(np.float16)
        
        for col in cols:
            weather_df['{}_mean_lag{}'.format(col, window)] = lag_mean[col]
            weather_df['{}_max_lag{}'.format(col, window)] = lag_max[col]
            weather_df['{}_min_lag{}'.format(col, window)] = lag_min[col]
            weather_df['{}_std_lag{}'.format(col, window)] = lag_std[col]
            weather_df['{}_median_lag{}'.format(col, window)] = lag_median[col]
            weather_df['{}_skew_lag{}'.format(col, window)] = lag_skew[col]
            
            
    def features_engineering(df):
        # Sort by timestamp                                                                     
        df.sort_values("timestamp")
        df.reset_index(drop=True)
        # Add more features                                                                     
        df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.weekday
        df["year"] = df["timestamp"].dt.year
        holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04","2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26", "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04", "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25", "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04", "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25", "2019-01-01"]
        df["month"] = df["timestamp"].dt.month
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
    
    add_lag_feature(weather_df, 3)
    add_lag_feature(weather_df, 18)
    
    train_df = reduce_mem_usage(train_df,use_float16=True)
    building_df = reduce_mem_usage(building_df,use_float16=True)
    weather_df = reduce_mem_usage(weather_df,use_float16=True)
    
    train_df = train_df.merge(building_df, left_on='building_id',right_on='building_id',how='left')
    train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    del weather_df
    gc.collect()
    
    #fixing site_id 0                                                                           
    train_df.loc[train_df['site_id']==0, 'meter_reading'] = train_df.loc[train_df['site_id']==0, 'meter_reading'] * 0.2931
    
    train_df = features_engineering(train_df)
    
    train_df["log_meter_reading"] = np.log1p(train_df["meter_reading"])
    
    categorical_features = ["building_id", "meter", "primary_use", "is_holiday", "dayofweek"]
    all_features = [col for col in train_df.columns if col not in ["site_id", "meter_reading", "log_meter_reading", "year"]]
    
    logger.info("train columns: {}".format(all_features))
    logger.info("data preparation end {}".format(train_df.shape))
    
    month_train = 10
    month_valid = 11
    month_test = 12
    
    cv = 3
    all_params = {"num_leaves":[5, 10, 20, 50],
                  "objective":["regression"],
                  "metric":["rmse"],
                  "learning_rate":[0.1],
                  "boosting":["gbdt"],
                  "min_data_in_leaf":[0,5,15,300],
                  "max_depth":[-1,3,7],
                  "colsample_bytree":[0.1,1.0],
                  "lambda_l1":[1,10],
                  "lambda_l2":[1,10],
                  "seed":[1]}
    models = {}
    cv_scores = {"site_id": [], "cv_score": []}
    
    min_rmse = 100
    
    min_params = None
    
    for params in tqdm(list(ParameterGrid(all_params))):
        rmse = 0
        for site_id in tqdm(range(16), desc="site_id"):
            print(cv, "fold CV for site_id:", site_id)
            models[site_id] = []
            
            X_train_site = train_df[train_df["site_id"]==site_id].reset_index(drop=True)
            y_train_site = X_train_site["log_meter_reading"]
            y_pred_train_site = np.zeros(X_train_site.shape[0])
            
            score = 0
            
            X_train = X_train_site[X_train_site["month"] <= month_train]
            X_train = X_train[all_features]
            X_valid = X_train_site[X_train_site["month"] == month_valid]
            X_valid = X_valid[all_features]
            
            y_train = X_train_site[X_train_site["month"] <= month_train]
            y_train = y_train["log_meter_reading"]
            y_valid = X_train_site[X_train_site["month"] == month_valid]
            y_valid = y_valid["log_meter_reading"]
            
            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
            
            dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
            
            watchlist = [dtrain, dvalid]
            
            model_lgb = lgb.train(params, train_set=dtrain, valid_sets=watchlist, verbose_eval=20, early_stopping_rounds=20)
            models[site_id].append(model_lgb)
            
            y_pred_valid = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
            
            rmse += np.sqrt(mean_squared_error(y_valid, y_pred_valid))
        
        rmse = rmse/16
        
        if min_rmse < rmse:
            min_rmse = rmse
            min_params = params
    
    
    logger.info("minimum params:{}".format(max_params))
    logger.info("minimum auc:{}".format(max_auc))
    logger.info("train end")
