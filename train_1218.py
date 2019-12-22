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
from load_data import load_train_data, load_test_data, load_building_data, load_weather_train_data, load_weather_test_data, load_fill_weather_data
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

    # read data
    train_df = load_train_data()
    building_df = load_building_data()
    weather_df = load_weather_train_data()

    # eliminate bad rows                                                                        
    bad_rows = pd.read_csv('../input/rows_to_drop.csv')
    train_df.drop(bad_rows.loc[:, '0'], inplace = True)
    train_df.reset_index(drop = True, inplace = True)
    del bad_rows
    gc.collect()

    # lag feature
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


    # add features about time
    def features_engineering(df):
        # Add more features                                                                     
        df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.weekday
        holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04","2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26", "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04", "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25", "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04", "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25", "2019-01-01"]
        df["month"] = df["timestamp"].dt.month
        df["month_cat"] = df["month"]
        df["month_cat"].replace((1, 2, 3, 4), 1, inplace=True)
        df["month_cat"].replace((5, 6, 7, 8), 2, inplace=True)
        df["month_cat"].replace((9, 10, 11, 12), 3, inplace=True)
        df["is_holiday"] = (df.timestamp.isin(holidays)).astype(int)
        df['square_feet'] =  np.log1p(df['square_feet'])

        le = LabelEncoder()
        df["primary_use"] = le.fit_transform(df["primary_use"])
        return df


    weather_df = load_fill_weather_data(weather_df)

    add_lag_feature(weather_df, 3)
    add_lag_feature(weather_df, 18)

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

    train_df = reduce_mem_usage(train_df,use_float16=True)

    categorical_features = ["building_id", "meter", "primary_use", "is_holiday", "dayofweek", "site_id"]
    all_features = [col for col in train_df.columns if col not in ["meter_reading", "log_meter_reading", "month", "timestamp"]]
    
    logger.info("train columns: {}".format(all_features))
    logger.info("data preparation end {}".format(train_df.shape))
    
    cv = 3
    NO1_param = {"num_leaves":41,
                 "objective":"regression",
                 "metric":"rmse",
                 "learning_rate":0.049,
                 "bagging_freq":5,
                 "bagging_fraction":0.51,
                 "feature_fraction":0.81,
                 }
    NO2_param = {"num_leaves":500,
                 "objective":"regression",
                 "metric":"rmse",
                 "learning_rate":0.05,
                 "boosting":"gbdt",
                 "subsample":0.4,
                 "feature_fraction":0.7,
                 "n_jobs":-1,
                 "seed":50,
                 }
    NO3_param = {"num_leaves":3160,                                                            
                 "objective":"regression",                                                     
                 "metric":"rmse",                                                              
                 "learning_rate":0.03,                                                         
                 "subsample":0.5,                                                              
                 "n_jobs":-1,                                                                  
                 "seed":50,                                                                    
                 "feature_fraction":0.7,                                                       
                 "boosting":"gbdt"
                 }

    models = {}

    for site_id in tqdm(range(16), desc="site_id"):

        print(cv, "fold CV for site_id:", site_id)

        models[site_id] = []

        X_train_site = train_df[train_df["site_id"]==site_id].reset_index(drop=True)
        y_train_site = X_train_site["log_meter_reading"]
        
        X_train = X_train_site.sample(frac=0.01, random_state=5)
        X_train = X_train[all_features]
        X_valid = X_train_site[X_train_site["month"]==12]
        X_valid = X_valid[all_features]
        
        y_train = X_train_site.sample(frac=0.01, random_state=5)
        y_train = y_train["log_meter_reading"]
        y_valid = X_train_site[X_train_site["month"]==12]
        y_valid = y_valid["log_meter_reading"]
        
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
        
        watchlist = [dtrain, dvalid]
        
        model_lgb = lgb.train(NO1_param, train_set=dtrain, valid_sets=watchlist, verbose_eval=20, early_stopping_rounds=20)
        models[site_id].append(model_lgb)
        
        
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
        
        watchlist = [dtrain, dvalid]
        
        model_lgb = lgb.train(NO2_param, train_set=dtrain, valid_sets=watchlist, verbose_eval=20, early_stopping_rounds=20)
        models[site_id].append(model_lgb)


        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
        
        watchlist = [dtrain, dvalid]                                                           
        
        model_lgb = lgb.train(NO3_param, train_set=dtrain, valid_sets=watchlist, verbose_eval=20, early_stopping_rounds=20)                                                                   
        models[site_id].append(model_lgb)                                                      
    del X_train, X_valid, y_train, y_valid, X_train_site, y_train_site, dtrain, dvalid, watchlist, train_df, model_lgb,
    gc.collect()

    logger.info("train end")

    test_df = load_test_data()
    test_df = test_df.merge(building_df,left_on='building_id',right_on='building_id',how='left')
    del building_df
    gc.collect()

    test_df_0 = test_df[test_df["site_id"]==0]
    test_df_1 = test_df[test_df["site_id"]==1]
    test_df_2 = test_df[test_df["site_id"]==2]
    test_df_3 = test_df[test_df["site_id"]==3]
    test_df_4 = test_df[test_df["site_id"]==4]
    test_df_5 = test_df[test_df["site_id"]==5]
    test_df_6 = test_df[test_df["site_id"]==6]
    test_df_7 = test_df[test_df["site_id"]==7]
    test_df_8 = test_df[test_df["site_id"]==8]
    test_df_9 = test_df[test_df["site_id"]==9]
    test_df_10 = test_df[test_df["site_id"]==10]
    test_df_11 = test_df[test_df["site_id"]==11]
    test_df_12 = test_df[test_df["site_id"]==12]
    test_df_13 = test_df[test_df["site_id"]==13]
    test_df_14 = test_df[test_df["site_id"]==14]
    test_df_15 = test_df[test_df["site_id"]==15]
    del test_df
    gc.collect()

    weather_df = load_weather_test_data()
    weather_df = load_fill_weather_data(weather_df)

    add_lag_feature(weather_df, 3)
    add_lag_feature(weather_df, 18)
    
    weather_df = reduce_mem_usage(weather_df)
    
    # test_df = test_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])                                                                           
    test_df_0 = test_df_0.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_1 = test_df_1.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_2 = test_df_2.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_3 = test_df_3.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_4 = test_df_4.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_5 = test_df_5.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_6 = test_df_6.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_7 = test_df_7.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_8 = test_df_8.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_9 = test_df_9.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_10 = test_df_10.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_11 = test_df_11.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_12 = test_df_12.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_13 = test_df_13.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_14 = test_df_14.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    test_df_15 = test_df_15.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    del weather_df
    gc.collect()

    #test_df = features_engineering(test_df)                                                    
    #test_df = reduce_mem_usage(test_df)                                                        
    test_df_0 = features_engineering(test_df_0)
    test_df_0 = reduce_mem_usage(test_df_0)
    test_df_1 = features_engineering(test_df_1)
    test_df_1 = reduce_mem_usage(test_df_1)
    test_df_2 = features_engineering(test_df_2)
    test_df_2 = reduce_mem_usage(test_df_2)
    test_df_3 = features_engineering(test_df_3)
    test_df_3 = reduce_mem_usage(test_df_3)
    test_df_4 = features_engineering(test_df_4)
    test_df_4 = reduce_mem_usage(test_df_4)
    test_df_5 = features_engineering(test_df_5)
    test_df_5 = reduce_mem_usage(test_df_5)
    test_df_6 = features_engineering(test_df_6)
    test_df_6 = reduce_mem_usage(test_df_6)
    test_df_7 = features_engineering(test_df_7)
    test_df_7 = reduce_mem_usage(test_df_7)
    test_df_8 = features_engineering(test_df_8)
    test_df_8 = reduce_mem_usage(test_df_8)
    test_df_9 = features_engineering(test_df_9)
    test_df_9 = reduce_mem_usage(test_df_9)
    test_df_10 = features_engineering(test_df_10)
    test_df_10 = reduce_mem_usage(test_df_10)
    test_df_11 = features_engineering(test_df_11)
    test_df_11 = reduce_mem_usage(test_df_11)
    test_df_12 = features_engineering(test_df_12)
    test_df_12 = reduce_mem_usage(test_df_12)
    test_df_13 = features_engineering(test_df_13)
    test_df_13 = reduce_mem_usage(test_df_13)
    test_df_14 = features_engineering(test_df_14)
    test_df_14 = reduce_mem_usage(test_df_14)
    test_df_15 = features_engineering(test_df_15)
    test_df_15 = reduce_mem_usage(test_df_15)

    #logger.info("data preparation end{}".format(test_df.shape))                                

    df_test_sites = []

    X_test_site = test_df_0.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 0)

    for fold in range(cv):
        model_lgb = models[0][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    y_pred_test_site *= 3.4118
    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)

    print("Scoring for site_id", 0, "completed\n")
    del test_df_0
    gc.collect()

    X_test_site = test_df_1.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 1)

    for fold in range(cv):
        model_lgb = models[1][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 1, "completed\n")
    del test_df_1
    gc.collect()

    X_test_site = test_df_2.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 2)

    for fold in range(cv):
        model_lgb = models[2][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 2, "completed\n")
    del test_df_2
    gc.collect()

    X_test_site = test_df_3.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])

    print("Scoring for site_id", 3)
    for fold in range(cv):
        model_lgb = models[3][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 3, "completed\n")
    del test_df_3
    gc.collect()

    X_test_site = test_df_4.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 4)

    for fold in range(cv):
        model_lgb = models[4][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 4, "completed\n")
    del test_df_4
    gc.collect()
    
    X_test_site = test_df_5.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 5)

    for fold in range(cv):
        model_lgb = models[5][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 5, "completed\n")
    del test_df_5
    gc.collect()

    X_test_site = test_df_6.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 6)

    for fold in range(cv):
        model_lgb = models[6][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 6, "completed\n")

    del test_df_6
    gc.collect()
    
    X_test_site = test_df_7.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 7)
    
    for fold in range(cv):
        model_lgb = models[7][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 7, "completed\n")
    del test_df_7
    gc.collect()

    X_test_site = test_df_8.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 8)

    for fold in range(cv):
        model_lgb = models[8][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 8, "completed\n")
    del test_df_8
    gc.collect()

    X_test_site = test_df_9.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 9)

    for fold in range(cv):
        model_lgb = models[9][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 9, "completed\n")
    del test_df_9
    gc.collect()

    X_test_site = test_df_10.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 10)

    for fold in range(cv):
        model_lgb = models[10][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 10, "completed\n")
    del test_df_10
    gc.collect()

    X_test_site = test_df_11.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 11)

    for fold in range(cv):
        model_lgb = models[11][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 11, "completed\n")
    del test_df_11
    gc.collect()

    X_test_site = test_df_12.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 12)

    for fold in range(cv):
        model_lgb = models[12][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 12, "completed\n")
    del test_df_12
    gc.collect()

    X_test_site = test_df_13.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 13)

    for fold in range(cv):
        model_lgb = models[13][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 13, "completed\n")
    del test_df_13
    gc.collect()

    X_test_site = test_df_14.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 14)

    for fold in range(cv):
        model_lgb = models[14][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 14, "completed\n")    
    del test_df_14
    gc.collect()

    X_test_site = test_df_15.copy()
    row_ids_site = X_test_site.row_id
    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])
    print("Scoring for site_id", 15)

    for fold in range(cv):
        model_lgb = models[15][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)
    print("Scoring for site_id", 15, "completed\n")
    del test_df_15, X_test_site, df_test_site
    gc.collect()

    submit = pd.concat(df_test_sites)
    submit.meter_reading = np.clip(np.expm1(submit.meter_reading), 0, a_max=None)
    submit.to_csv(DIR + "submission_1218.csv", index=False)

    logger.info("end")
