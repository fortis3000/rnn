"""Predicting on the test set with the model"""

import pickle
import datetime as dt

import pandas as pd
import numpy as np

from create_data import build_train_period, build_dataset

from data_config import get_prev, get_day_number, get_column_name
from initial_data import (
    sales,
    get_days,
)

from linear_model import metrics, evaluate


def custom_round(float_value, threshold=0.5):
    """Round float_value to integer using custom threshold"""

    if float_value <= 0:
        return 0

    int_part = int(float_value)

    res = -1 * (int_part - float_value)

    return int_part if res <= threshold else int_part + 1


def find_thres(preds, real_values, threshold_range, metric_to_follow="mse"):
    """Brute-force method to choose the best round threshold"""

    best_metric_to_follow = np.inf  # -np.inf dor maximizing task
    best_t = 0

    best_metrics = {}

    for t in threshold_range:

        current_metrics = evaluate(
            list(map(lambda x: custom_round(x, t), preds)), real_values, metrics
        )

        if current_metrics[metric_to_follow] <= best_metric_to_follow:
            best_metrics = current_metrics
            best_t = t

    return best_t, best_metrics


def predict_step_by_step(
    model, target_list, target_exists=True, threshold=False, save=False
):
    """Validate train model on target_list provided"""

    global sales
    base_columns = sorted(set(sales.columns) - set(get_days(sales)))

    # save predictions
    preds_df = pd.DataFrame(index=sales.index)

    for target in target_list:

        target_col = get_column_name(target)

        df_test = build_dataset(
            base_columns=base_columns,
            target_list=[target],
            period=28,
            datasets_num=1,
            test=True,
        )

        preds = model.predict(df_test[model.feature_name()])

        if threshold:
            preds = list(map(lambda x: custom_round(x, threshold), preds))

        preds_df = preds_df.merge(
            pd.DataFrame({target_col: preds}, index=df_test.index),
            left_index=True,
            right_index=True,
        )

        if not target_exists:
            sales = sales.merge(
                pd.DataFrame({target_col: preds}, index=df_test.index),
                left_index=True,
                right_index=True,
            )

    return preds_df


def validate(preds_df):
    """Before or after postprocessing"""
    global sales, metrics
    all_metrics = {key: list().copy() for key, val in metrics.items()}

    for col in preds_df.keys():

        cur_metrics = evaluate(preds_df[col], sales[col], metrics)
        {key: all_metrics[key].append(val) for key, val in cur_metrics.items()}

    mean_metrics = {key: np.mean(val) for key, val in all_metrics.items()}
    median_metrics = {key: np.median(val) for key, val in all_metrics.items()}
    print(f"All metrics: {all_metrics}")
    print(f"Mean metrics: {mean_metrics}")
    print(f"Median metrics: {median_metrics}")
    return all_metrics


def postprocess_predictions(preds_df, choose_threshold=False):
    # step by step predictions using sales dataframe
    global sales

    if choose_threshold:
        thresholds = []

        for col in preds_df.keys():
            t, _ = find_thres(preds_df[col], sales[col], np.arange(0.1, 1, 0.05))
            thresholds.append(t)

        threshold_found = np.median(thresholds)

        print("Thresholds", thresholds)
        print(f"Applying threshold {threshold_found}")

        preds_df = preds_df.applymap(
            lambda x: custom_round(x, threshold=threshold_found)
        )

    return preds_df


import lightgbm as lgb

from lgbm_regr import get_dt_str

from transfer_df import upload_pickled, usecols

if __name__ == "__main__":

    # model = pickle.load(open("models/elastic_model_2020-05-04_20:25:51", "rb"))

    # model = lgb.Booster(model_file=r"models/booster_2020-05-24_10:48:36.txt")

    # val_list = list(range(1912, 1914))

    # preds_df = predict_step_by_step(
    #     model, target_list=val_list, target_exists=True, threshold=False
    # )

    # validate(preds_df)

    # validate(postprocess_predictions(preds_df, choose_threshold=True))

    # val_list or target_list
    # preds_df = preds_df.rename(
    #     {f"d_{x}": f"F{num}" for num, x in enumerate(val_list, start=1)}, axis=1
    # )

    # preds_df.to_csv("preds.csv", sep=",", header=True, index=True)

    #####
    # SUBMISSION
    #####

    model = lgb.Booster(
        model_file=r"models/lgbm_2020-06-02_21:39:29/booster_2020-06-03_06:01:27.txt"
    )

    # val_list = list(range(1914, 1914 + 28 + 1))

    # preds_df = predict_step_by_step(
    #     model, target_list=val_list, target_exists=False, threshold=False
    # )

    # # val_list or target_list
    # preds_df = preds_df.rename(
    #     {f"d_{x}": f"F{num}" for num, x in enumerate(val_list, start=1)}, axis=1
    # )

    # preds_df.to_csv(f"submission_{get_dt_str()}.csv", sep=",", header=True, index=True)

    upload_pickled()

