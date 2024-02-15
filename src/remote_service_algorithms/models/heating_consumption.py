from copy import deepcopy
from copy import deepcopy
from datetime import timedelta
from random import sample, seed
from typing import Union

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.remote_service_algorithms.features.enums import FeatureName
from src.remote_service_algorithms.models.enums import ModelName
from src.remote_service_algorithms.shared.data_utils import convert_json, model_from_bytes
from src.remote_service_algorithms.shared.math_utils import log10p, exp10p
from src.remote_service_algorithms.shared.schedule import Schedule
from src.shared.logger import logger_factory
from src.shared.losses import symmetric_mean_absolute_percentage_error_200, \
    symmetric_mean_absolute_percentage_error_100

MINIMUM_DAYS = 15
ONLINE_LIMIT = 70
logger = logger_factory(__name__)


def generate_feature_dataset(adf_df: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """
    This function generates a feature dataset from the given DataFrame. It performs several checks and transformations
    on the data, including checking for matching active heating circuits, availability of consumption data, and
    availability of weather temperature data. It also pivots the DataFrame and prepares the data for further processing.

    Parameters:
    adf_df (pd.DataFrame): The input DataFrame containing the data to be processed.

    Returns:
    pd.DataFrame or None: The processed DataFrame, or None if any of the checks fail.
    """

    if len(adf_df) == 0:
        return

    active_hcs_current, adf_df = keep_days_with_matching_active_hcs(adf_df)

    consumption_available, adf_df = consumption_data_available(adf_df)
    if not consumption_available:
        return

    weather_data_available, adf_df = weather_temperature_available(adf_df)

    df = pivot_dataframe(adf_df)

    if "average_outside_temperature" not in df.columns:
        return

    df = df[~df["average_outside_temperature"].isna()]

    if len(df) == 0:
        return

    # check whether the installation is weather controlled
    if (
            FeatureName.CONFIG_WEATHER_CONTROLLED_REGULATION.value in df.columns.tolist()
            and df[FeatureName.CONFIG_WEATHER_CONTROLLED_REGULATION.value].iloc[-1] == 0
    ):
        weather_controlled = False
    else:
        weather_controlled = True

    # temperature data sanity check
    if df["average_outside_temperature"].rolling(10).var().min() < 0.001:
        return

    df = prepare_data(df, weather_controlled=weather_controlled)

    if len(df[df["consumption_heating"] > 0]) < MINIMUM_DAYS:
        return  # we need at least 15 days of data

    # if e3 gas device then the consumption is rescaled by 10 from kWh to ccm due to ViCare units
    df_e3_gas_device = adf_df[adf_df["feature_name"] == FeatureName.E3_GAS_DEVICE.value]
    if len(df_e3_gas_device) and df_e3_gas_device["feature_value"].max() > 0:
        df["consumption_heating"] = df["consumption_heating"] / 10.0

    df.reset_index(drop=True, inplace=True)
    return df


def train_model(df: pd.DataFrame, device_uuid: str) -> dict | None:
    """
    This function trains a model for predicting heating consumption. It first extracts the feature columns from the DataFrame,
    then creates an enhanced model using these features. The function returns the result of the model creation process.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be used for training the model.
    device_uuid (str): The unique identifier of the device for which the model is being trained.

    Returns:
    dict or None: A dictionary containing the model parameters and metrics, or None if the model creation process fails.
    """

    feature_names = get_feature_columns(df)
    df = df[feature_names + ["consumption_heating", "date"]]
    result = create_enhanced_model(df, device_uuid)

    if type(result) is not dict or "model_params" not in result:
        return

    weather_data_available = FeatureName.AVERAGE_OUTSIDE_TEMPERATURE_WEATHER_SERVICE.value in df.columns
    active_hcs_current = set(df.columns[df.columns.str.contains("mean_target_supply_hc_")])
    temperature_data_source = "weather_service" if weather_data_available else "outside_sensor"
    result["model_params"]["temperature_data_source"] = temperature_data_source
    result["model_params"]["active_heating_circuits"] = "".join(sorted(list(active_hcs_current)))

    return result


def create_enhanced_model(df: pd.DataFrame, device_uuid: str, random_seed: int = None) -> dict:
    """
    This function creates an enhanced model for predicting heating consumption. It performs cross-validation on multiple models,
    selects the best model based on the mean absolute error, and then retrains the best model on the full dataset.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be used for training the model.
    device_uuid (str): The unique identifier of the device for which the model is being trained.
    random_seed (int, optional): The seed for the random number generator. If not provided, a random seed is generated.

    Returns:
    dict: A dictionary containing the model parameters and metrics.
    """

    splits = 5
    df = df.reset_index()

    if random_seed is None:
        random_seed, rng = generate_random_seed()
    else:
        rng = np.random.RandomState(random_seed)
    dict_models = {
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=rng),
        "ExtraTreesRegressor": ExtraTreesRegressor(random_state=rng, n_estimators=50, max_depth=10),
    }
    feature_list = get_feature_columns(df)

    # for model evaluation and comparison in retraining, heating_train_test_split is used
    list_df_test, list_df_train = heating_train_test_split(
        df,
        test_fraction=0.19,
        heating_fraction=0.3,
        num_splits=splits,
        allow_overlap=False,
        num_seasons=3,
        random_seed=random_seed,
    )

    if len(list_df_test) == 0 or len(list_df_train) == 0:
        logger.error(f"Sizes of train and test index are {len(list_df_train)} and {len(list_df_test)}")
        raise Exception("Train or test index is empty")

    best_score = -9999
    best_model = None
    best_model_name = None
    model = None

    sc = StandardScaler()
    sc.fit(df[feature_list])

    for k, v in dict_models.items():
        cv_scores = []
        for train_index, test_index in zip(list_df_train, list_df_test):
            X_train = sc.transform(df.loc[df.index.isin(train_index), feature_list])
            y_train = df.loc[df.index.isin(train_index), "consumption_heating"]
            X_test = sc.transform(df.loc[df.index.isin(test_index), feature_list])
            y_test = df.loc[df.index.isin(test_index), "consumption_heating"]

            model = make_pipeline(
                TransformedTargetRegressor(regressor=v, func=log10p, inverse_func=exp10p),
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = -mean_absolute_error(y_pred, y_test)
            # In rare cases the error explodes into billions for individual splits. In such cases the score is
            # set to -9999 otherwise even the mean score below -9999 and the model is not selected
            if score > -9999:
                cv_scores.append(score)
            else:
                cv_scores.append(-9999)

        score = np.mean(cv_scores)

        if (score > best_score) and model is not None:
            best_score = score
            best_model = model
            best_model_name = k

    # once the best model is found we can estimate the uncertainty of the model prediction
    tscv = TimeSeriesSplit(n_splits=splits)  # for uncertainty estimation TimeSeriesSplit is used
    ts_scores = []
    for train_index, test_index in tscv.split(df):
        X_train = sc.transform(df.loc[df.index.isin(train_index), feature_list])
        X_test = sc.transform(df.loc[df.index.isin(test_index), feature_list])
        y_train = df.loc[df.index.isin(train_index), "consumption_heating"]
        y_test = df.loc[df.index.isin(test_index), "consumption_heating"]
        if best_model is not None:
            model2 = deepcopy(best_model)
            model2.fit(X_train, y_train)
            y_pred = model2.predict(X_test)
            ts_scores.append(mean_absolute_error(y_pred, y_test))

    # the model is retrained based on the full set of data and a holdout set is used for uncertainty estimation
    test_index, train_index = np.array(list_df_test)[-1], np.array(list_df_train)[-1]

    if len(train_index) == 0 or len(test_index) == 0:
        logger.error(f"Sizes of train and test index are {train_index.shape} and {test_index.shape}")
        raise Exception("Train or test index is empty")
    logger.debug(f"Train index: {str(train_index.shape)}, Test index: {str(test_index.shape)}")

    X_train = df.loc[df.index.isin(train_index), feature_list]
    y_train = df.loc[df.index.isin(train_index), "consumption_heating"]
    X_test = df.loc[df.index.isin(test_index), feature_list]
    y_test = df.loc[df.index.isin(test_index), "consumption_heating"]

    best_model = make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(
            regressor=dict_models[best_model_name], func=log10p, inverse_func=exp10p
        ),
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    uncertainty = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    smape_200 = symmetric_mean_absolute_percentage_error_200(y_test, y_pred)
    smape_100 = symmetric_mean_absolute_percentage_error_100(y_test, y_pred)

    logger.info(
        f"Device_uuid: {device_uuid}, Best model: {best_model_name}, score: {best_score}, uncertainty: {uncertainty}")

    return {
        "model_metrics": {
            "score": best_score,
            "uncertainty": uncertainty,
            "mae": uncertainty,
            "mse": mse,
            "rmse": rmse,
            "smape": smape_200,
            "smape_200": smape_200,
            "smape_100": smape_100,
        },
        "model_version": 1,
        "model_params": {
            "features": feature_list,
            "coefficients": np.nan,
            "intercept": np.nan,
            "train_index": train_index,
            "test_index": test_index,
            "seed": random_seed,
            "device_uuid": device_uuid,
            "enhanced": True,
            "algo_name": best_model_name,
        },
        "model": best_model,
    }


def weather_temperature_available(adf_df: pd.DataFrame) -> (bool, pd.DataFrame):
    """
    This function checks if weather temperature data is available in the DataFrame. If both weather service data and outside sensor data
    are available, it prioritizes the weather service data. If only outside sensor data is available, it uses that. If neither are available,
    it returns False.

    Parameters:
    adf_df (pd.DataFrame): The DataFrame containing the data to be checked.

    Returns:
    bool: True if weather temperature data is available, False otherwise.
    pd.DataFrame: The DataFrame after processing.
    """

    feature_weather = FeatureName.AVERAGE_OUTSIDE_TEMPERATURE_WEATHER_SERVICE.value
    feature_sensor = FeatureName.AVERAGE_OUTSIDE_TEMPERATURE.value
    temperature_list = [feature_weather, feature_sensor]

    if adf_df[adf_df["feature_name"].isin(temperature_list)]["feature_name"].nunique() == len(
            temperature_list
    ):
        weather_service_data = adf_df[
            (adf_df["feature_name"] == feature_weather) & (~adf_df["feature_value"].isna())
            ]
        outside_sensor_data = adf_df[
            (adf_df["feature_name"] == feature_sensor) & (~adf_df["feature_value"].isna())
            ]

        if (len(weather_service_data) > 300) or (
                len(weather_service_data) >= 0.9 * len(outside_sensor_data)
        ):
            adf_df = adf_df[adf_df["feature_name"] != feature_sensor]
            adf_df.loc[adf_df["feature_name"] == feature_weather, "feature_name"] = feature_sensor
            weather_data_available = True
        else:
            adf_df = adf_df[adf_df["feature_name"] != feature_weather]
            weather_data_available = False
    elif feature_sensor in adf_df["feature_name"].unique():
        weather_data_available = False
    else:
        adf_df.loc[adf_df["feature_name"] == feature_weather, "feature_name"] = feature_sensor
        weather_data_available = True

    return weather_data_available, adf_df


def consumption_data_available(adf_df: pd.DataFrame) -> (bool, pd.DataFrame):
    """
    This function checks if consumption data is available in the DataFrame. If both power and gas consumption data are available, it prioritizes
    the power consumption data. If only gas consumption data is available, it uses that. If neither are available, it returns False.

    Parameters:
    adf_df (pd.DataFrame): The DataFrame containing the data to be checked.

    Returns:
    bool: True if consumption data is available, False otherwise.
    pd.DataFrame: The DataFrame after processing.
    """

    consumption_list = [
        FeatureName.POWER_CONSUMPTION_HEATING.value,
        FeatureName.GAS_CONSUMPTION_HEATING.value,
    ]

    if adf_df[adf_df["feature_name"].isin(consumption_list)]["feature_name"].nunique() == len(
            consumption_list
    ):
        if (
                adf_df[
                    (adf_df["feature_name"] == FeatureName.POWER_CONSUMPTION_HEATING.value)
                    & (adf_df["feature_value"] != 0)
                ].count()[0]
                >= adf_df[
            (adf_df["feature_name"] == FeatureName.GAS_CONSUMPTION_HEATING.value)
            & (adf_df["feature_value"] != 0)
        ].count()[0]
        ):
            adf_df = adf_df[adf_df["feature_name"] != FeatureName.GAS_CONSUMPTION_HEATING.value]
        else:
            adf_df = adf_df[adf_df["feature_name"] != FeatureName.POWER_CONSUMPTION_HEATING.value]

    # in case gas consumption from the cloud is available use it in case of no other consumption
    if len(adf_df[adf_df["feature_name"].isin(consumption_list)]) > 0:
        adf_df = adf_df[adf_df["feature_name"] != FeatureName.GAS_CONSUMPTION_HEATING_CLOUD.value]
        return True, adf_df
    elif len(adf_df[adf_df["feature_name"] == FeatureName.GAS_CONSUMPTION_HEATING_CLOUD.value]) > 0:
        return True, adf_df
    else:
        return False, adf_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    This function extracts the feature columns from the DataFrame. The feature columns are those that are used for training the model.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to extract the feature columns.

    Returns:
    list: The list of feature column names.
    """

    feature_list = [
        "average_outside_temperature",
        "diff_outside_temperature_1day",
        "diff_outside_temperature_2day",
        "consumption_estimation",
        "week_y",
    ]

    for n in range(0, 4):
        if "mean_target_supply_hc_" + str(n) in df.columns:
            feature_list.append("mean_target_supply_hc_" + str(n))

    return feature_list


def pivot_dataframe(adf_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function pivots the DataFrame so that each row corresponds to a single date and each column corresponds to a feature. It also renames
    the columns to more descriptive names.

    Parameters:
    adf_df (pd.DataFrame): The DataFrame to be pivoted.

    Returns:
    pd.DataFrame: The pivoted DataFrame.
    """

    df = adf_df.pivot_table(index="feature_date", values="feature_value", columns="feature_name")
    df = df.reset_index()
    df = df.rename(
        columns={
            "feature_date": "date",
            FeatureName.AVERAGE_OUTSIDE_TEMPERATURE_WEATHER_SERVICE: "average_outside_temperature",
            FeatureName.POWER_CONSUMPTION_HEATING.value: "consumption_heating",
            FeatureName.GAS_CONSUMPTION_HEATING.value: "consumption_heating",
            FeatureName.GAS_CONSUMPTION_HEATING_CLOUD.value: "consumption_heating",
        }
    )
    return df.sort_values("date")


def prepare_data(
        df: pd.DataFrame, weather_controlled: bool, predict_mode: bool = False
) -> pd.DataFrame:
    """
    This function prepares the data for model training or prediction. It generates heating curves and indoor temperatures, calculates temperature
    differences, checks online availability, and creates additional features.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be prepared.
    weather_controlled (bool): Whether the heating is controlled by the weather.
    predict_mode (bool, optional): Whether the function is being called in prediction mode. If True, certain checks and transformations are skipped.

    Returns:
    pd.DataFrame: The DataFrame after preparation.
    """

    # create features: heating curves
    df = get_heating_curves_and_indoor_temperatures(df, predict_mode)

    # create features: temperature differences
    df = get_outside_temperature_differences(df)

    # moved this to the end so as many temperature differences as possible can be calculated
    if not predict_mode:
        df = df[df["consumption_heating"].notna()]

    df = df[df["diff_outside_temperature_2day"].notna()]
    df = df[df["diff_outside_temperature_1day"].notna()]

    # check online availability
    if not predict_mode:
        df = check_online_availability(df)
        df = df.drop(columns=FeatureName.ONLINE_SHARE.value)

    column_names = df.columns.tolist()

    if weather_controlled:
        mean_target_supply_columns = [c for c in column_names if "mean_target_supply_hc" in c]
        df["mean_target_supply"] = df[mean_target_supply_columns].mean(axis=1)
        for c in mean_target_supply_columns:
            df = df[df[c].notna()]

        avg_indoor_temp_columns = [c for c in column_names if "average_indoor_temperature_hc" in c]
        df["average_indoor_temperature"] = df[avg_indoor_temp_columns].mean(axis=1)
        df = df.drop(columns=avg_indoor_temp_columns)

        # if heatingrod heating consumption data is available use it to clean data
        heatingrod_consumption = FeatureName.HEATINGROD_POWER_CONSUMPTION_HEATING.value
        if heatingrod_consumption in df.columns:
            df[heatingrod_consumption] = df[heatingrod_consumption].fillna(0.)
            df = df[df["consumption_heating"] > df[heatingrod_consumption]]
            df = df.drop(columns=heatingrod_consumption)

    else:
        # in a case the installation is not weather controlled,
        # then average indoor temperature columns actually contain target supply temperatures
        for n in range(4):
            avg_indoor_temp_column = "average_indoor_temperature_hc_" + str(n)
            if avg_indoor_temp_column in column_names:
                df["mean_target_supply_hc_" + str(n)] = df[avg_indoor_temp_column]
                df = df.drop(columns=avg_indoor_temp_column)
        mean_target_supply_columns = [c for c in column_names if "mean_target_supply_hc" in c]
        df["mean_target_supply"] = df[mean_target_supply_columns].mean(axis=1)
        for c in mean_target_supply_columns:
            df = df[df[c].notna()]
        df["average_indoor_temperature"] = 20

    df["consumption_estimation"] = (
            (df["average_indoor_temperature"] - df["average_outside_temperature"])
            * (df["mean_target_supply"] - df["average_outside_temperature"])
            / df["mean_target_supply"]
    )
    df["consumption_estimation"] = df["consumption_estimation"].fillna(0)
    df.loc[df["consumption_estimation"] <= 0, "consumption_estimation"] = 0
    df.loc[df["mean_target_supply"] <= 0, "consumption_estimation"] = 0

    # remove cols for calculation of consumption_estimation
    df = df.drop(columns=["average_indoor_temperature", "mean_target_supply"])

    df["week_y"] = df["date"].apply(get_cosine_calender_week)

    return df


def get_heating_curves_and_indoor_temperatures(df: pd.DataFrame,
                                               predict_mode: bool) -> pd.DataFrame:
    """
    This function calculates the heating curves and indoor temperatures for each heating circuit. It also removes unnecessary columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be processed.
    predict_mode (bool): Whether the function is being called in prediction mode. If True, certain checks and transformations are skipped.

    Returns:
    pd.DataFrame: The DataFrame after processing.
    """

    column_names = df.columns.tolist()
    for n in range(0, 4):
        heating_circuit_list = [
            "operating_programs_reduced_temperature_hc_",
            "operating_programs_normal_temperature_hc_",
            "operating_programs_comfort_temperature_hc_",
            "config_heating_schedule_duration_reduced_hc",
            "config_heating_schedule_duration_normal_hc",
            "config_heating_schedule_duration_comfort_hc",
            "heating_curve_slope_hc_",
            "heating_curve_shift_hc_",
        ]
        heating_circuit_list = [v + str(n) for v in heating_circuit_list]

        # check whether all required columns are there
        if len(list(set(heating_circuit_list) & set(column_names))) != len(heating_circuit_list):
            continue

        df = calc_heating_curve(df, n)
        df = df.drop(
            columns=[
                "target_supply_reduced_hc_" + str(n),
                "target_supply_normal_hc_" + str(n),
                "target_supply_comfort_hc_" + str(n),
            ]
        )

        df = calc_average_indoor_temperature(df, n)
        df = df.drop(columns=heating_circuit_list)

        if ((df["mean_target_supply_hc_" + str(n)] == 0).all() or df[
            "mean_target_supply_hc_" + str(n)
        ].isna().all()) and not predict_mode:
            # Original
            # df = df.drop(columns="mean_target_supply_hc_" + str(n))
            df["mean_target_supply_hc_" + str(n)] = 0

        if ((df["average_indoor_temperature_hc_" + str(n)] == 0).all() or df[
            "average_indoor_temperature_hc_" + str(n)
        ].isna().all()) and not predict_mode:
            # Original
            # df = df.drop(columns="average_indoor_temperature_hc_" + str(n))
            df["average_indoor_temperature_hc_" + str(n)] = 0

    df = df.drop(
        columns=[FeatureName.CONFIG_VACATION.value, FeatureName.CONFIG_HEATING_ACTIVE.value]
    )

    return df


def calculate_target_supply_temperature(
        t_out: pd.Series, t_in: pd.Series, slope: pd.Series, t_shift: pd.Series
) -> pd.Series:
    """
    This function calculates the target supply temperature for a heating circuit based on the outside temperature, the indoor temperature, the slope of the
    heating curve, and the shift of the heating curve.

    Parameters:
    t_out (pd.Series): The outside temperature.
    t_in (pd.Series): The indoor temperature.
    slope (pd.Series): The slope of the heating curve.
    t_shift (pd.Series): The shift of the heating curve.

    Returns:
    pd.Series: The target supply temperature.
    """

    delta = t_out - t_in
    t_target = -slope * delta * (1.4347 + 0.021 * delta + 0.0002479 * delta ** 2) + t_in + t_shift
    return t_target


def calc_heating_curve(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    This function calculates the heating curve for a specific heating circuit. It also calculates the mean target supply temperature for each day.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be processed.
    n (int): The number of the heating circuit.

    Returns:
    pd.DataFrame: The DataFrame after processing.
    """
    df = df.copy()

    df.loc[:, "target_supply_reduced_hc_" + str(n)] = calculate_target_supply_temperature(
        df.loc[:, "average_outside_temperature"],
        df.loc[:, "operating_programs_reduced_temperature_hc_" + str(n)],
        df.loc[:, "heating_curve_slope_hc_" + str(n)],
        df.loc[:, "heating_curve_shift_hc_" + str(n)],
    )
    df.loc[:, "target_supply_normal_hc_" + str(n)] = calculate_target_supply_temperature(
        df.loc[:, "average_outside_temperature"],
        df.loc[:, "operating_programs_normal_temperature_hc_" + str(n)],
        df.loc[:, "heating_curve_slope_hc_" + str(n)],
        df.loc[:, "heating_curve_shift_hc_" + str(n)],
    )
    df.loc[:, "target_supply_comfort_hc_" + str(n)] = calculate_target_supply_temperature(
        df.loc[:, "average_outside_temperature"],
        df.loc[:, "operating_programs_comfort_temperature_hc_" + str(n)],
        df.loc[:, "heating_curve_slope_hc_" + str(n)],
        df.loc[:, "heating_curve_shift_hc_" + str(n)],
    )
    df.loc[:, "mean_target_supply_hc_" + str(n)] = (
                                                           df.loc[:, "target_supply_normal_hc_" + str(n)]
                                                           * df.loc[:,
                                                             "config_heating_schedule_duration_normal_hc" + str(n)]
                                                           + df.loc[:, "target_supply_reduced_hc_" + str(n)]
                                                           * df.loc[:,
                                                             "config_heating_schedule_duration_reduced_hc" + str(n)]
                                                           + df.loc[:, "target_supply_comfort_hc_" + str(n)]
                                                           * df.loc[:,
                                                             "config_heating_schedule_duration_comfort_hc" + str(n)]
                                                   ) / 1440

    # set to reduced when vacation mode is active
    df.loc[df["config_vacation"] == 1, "mean_target_supply_hc_" + str(n)] = df.loc[
        df["config_vacation"] == 1, "target_supply_reduced_hc_" + str(n)]

    # set to 0 when heating is off
    df.loc[df["config_heating_active"] == 0, "mean_target_supply_hc_" + str(n)] = 0

    return df


def calc_average_indoor_temperature(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    This function calculates the average indoor temperature for a specific heating circuit.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be processed.
    n (int): The number of the heating circuit.

    Returns:
    pd.DataFrame: The DataFrame after processing.
    """
    df["average_indoor_temperature_hc_" + str(n)] = (
                                                            df["operating_programs_reduced_temperature_hc_" + str(n)]
                                                            * df["config_heating_schedule_duration_reduced_hc" + str(n)]
                                                            + df["operating_programs_normal_temperature_hc_" + str(n)]
                                                            * df["config_heating_schedule_duration_normal_hc" + str(n)]
                                                            + df["operating_programs_comfort_temperature_hc_" + str(n)]
                                                            * df["config_heating_schedule_duration_comfort_hc" + str(n)]
                                                    ) / 1440

    # set to reduced when vacation mode is active
    df.loc[df["config_vacation"] == 1, "average_indoor_temperature_hc_" + str(n)] = df[
        "operating_programs_reduced_temperature_hc_" + str(n)
        ]

    # set to 0 when heating is off
    df.loc[df["config_heating_active"] == 0, "average_indoor_temperature_hc_" + str(n)] = 0

    return df


def get_temperature_difference(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    This function calculates the difference in outside temperature from a certain number of days ago.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be processed.
    days (int): The number of days ago to compare the temperature with.

    Returns:
    pd.DataFrame: The DataFrame after processing.
    """

    df1 = df[["average_outside_temperature"]].copy()
    df1 = df1.shift(periods=days, freq="D")
    df[f"diff_outside_temperature_{days}day"] = (
            df["average_outside_temperature"] - df1["average_outside_temperature"]
    )
    return df


def get_outside_temperature_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the difference in outside temperature from one and two days ago.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be processed.

    Returns:
    pd.DataFrame: The DataFrame after processing.
    """

    df["date"] = df["date"].astype('datetime64[ns]')
    df = df.sort_values("date")
    df = df.set_index("date")

    df = get_temperature_difference(df, days=1)
    df = get_temperature_difference(df, days=2)

    df = df.reset_index()
    return df


def check_online_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function checks the online availability of the data. It removes days where the online share is less than a certain limit.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be checked.

    Returns:
    pd.DataFrame: The DataFrame after checking.
    """

    if FeatureName.ONLINE_SHARE.value not in df.columns:
        df[FeatureName.ONLINE_SHARE.value] = 0

    dates = df.loc[df[FeatureName.ONLINE_SHARE.value] < ONLINE_LIMIT, "date"].to_list()
    dates2 = [d + timedelta(days=1) for d in dates]
    dates = dates + dates2
    df = df[~df["date"].isin(dates)]

    return df


def get_cosine_calender_week(x) -> float:
    """
    This function calculates the cosine of the calendar week of a date. This is used as a feature in the model.

    Parameters:
    x (datetime): The date.

    Returns:
    float: The cosine of the calendar week.
    """

    week = float(x.strftime("%V"))
    return np.round(np.cos((2.0 * np.pi / 52.0 * week)), 4)


def heating_train_test_split(
        df, test_fraction=0.2, heating_fraction=0.3, num_splits=5, num_seasons=3, allow_overlap=False,
        random_seed=1
):
    """
    This function splits the data into training and test sets. It ensures that the test set contains a certain fraction of heating and non-heating days.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be split.
    test_fraction (float, optional): The fraction of the data to be used as the test set. Default is 0.2.
    heating_fraction (float, optional): The fraction of the test set to be heating days. Default is 0.3.
    num_splits (int, optional): The number of splits to make. Default is 5.
    num_seasons (int, optional): The number of seasons to consider. Default is 3.
    allow_overlap (bool, optional): Whether to allow overlap between the training and test sets. Default is False.
    random_seed (int, optional): The seed for the random number generator. Default is 1.

    Returns:
    list: A list of test indices.
    list: A list of train indices.
    """

    if test_fraction * num_splits >= 1 and not allow_overlap:
        raise ValueError("Not enough data. test_fraction * num_splits >= 1!")

    if num_seasons > 2 and heating_fraction * num_seasons > 1:
        raise ValueError("Not enough data. heating_fraction * num_seasons > 1!")

    if num_splits < 1:
        raise ValueError("num_splits < 1 but must be at least 1!")

    if heating_fraction == 0 or heating_fraction > 1:
        raise ValueError("heating_fraction must be in the range (0, 1]!")

    if test_fraction == 0 or test_fraction > 0.5:
        raise ValueError("test_fraction must be in the range (0, 0.5]!")

    if num_seasons > 3 or num_seasons < 1:
        raise ValueError("num_seasons must be in the range [1, 3]!")

    num_data = len(df)
    num_test_data = round(test_fraction * num_data)

    if num_splits >= num_test_data:
        raise ValueError("Please reduce num_splits or increase test_fraction!")

    if num_seasons > 2:
        num_heating_test_data = round(heating_fraction * num_test_data)
        num_non_heating_test_data = num_heating_test_data
        num_between_test_data = num_test_data - num_non_heating_test_data - num_heating_test_data
    elif num_seasons == 2:
        num_heating_test_data = round(heating_fraction * num_test_data)
        num_non_heating_test_data = num_test_data - num_heating_test_data
        num_between_test_data = 0
    else:
        num_heating_test_data = num_test_data
        num_non_heating_test_data = 0
        num_between_test_data = 0

    df = df.sort_values("consumption_heating")
    seed(random_seed)

    non_heating_indices = df[
                          0: num_non_heating_test_data * num_splits  # noqa: E203
                          ].index.to_list()
    between_indices = df[
                      num_non_heating_test_data
                      * num_splits: (num_non_heating_test_data + num_between_test_data)  # noqa: E203
                                    * num_splits
                      ].index.to_list()
    heating_indices = df[
                      (num_non_heating_test_data + num_between_test_data) * num_splits:  # noqa: E203
                      ].index.to_list()

    df = df.sort_values("date")

    if not allow_overlap:
        heating_test_all_indices = sample(heating_indices, num_heating_test_data * num_splits)
        heating_test_indices = [
            heating_test_all_indices[
            n * num_heating_test_data: (n + 1) * num_heating_test_data  # noqa: E203
            ]
            for n in range(num_splits)
        ]

        non_heating_test_all_indices = sample(
            non_heating_indices, num_non_heating_test_data * num_splits
        )
        non_heating_test_indices = [
            non_heating_test_all_indices[
            n * num_non_heating_test_data: (n + 1) * num_non_heating_test_data  # noqa: E203
            ]
            for n in range(num_splits)
        ]

        between_test_all_indices = sample(between_indices, num_between_test_data * num_splits)
        between_test_indices = [
            between_test_all_indices[
            n * num_between_test_data: (n + 1) * num_between_test_data  # noqa: E203
            ]
            for n in range(num_splits)
        ]
    else:
        heating_test_indices = []
        non_heating_test_indices = []
        between_test_indices = []
        for n in range(num_splits):
            heating_test_indices.append(sample(heating_indices, num_heating_test_data))
            non_heating_test_indices.append(sample(non_heating_indices, num_non_heating_test_data))
            between_test_indices.append(sample(between_indices, num_between_test_data))

    # now generate the num_splits training sets via the indices
    list_test_indices = []
    list_train_indices = []
    for n in range(num_splits):
        list_test_indices.append(
            heating_test_indices[n] + non_heating_test_indices[n] + between_test_indices[n]
        )
        list_train_indices.append(list(set(df.index.to_list()) ^ set(list_test_indices[-1])))

    return list_test_indices, list_train_indices


def predict(df_temperatures: pd.DataFrame, config: dict, models: Union[str, pd.DataFrame]):
    """
    allows to predict the consumption based on input data which is supplied in form of a config
    dictionary containing the device information about the heating circuits and a dataframe which
    holds the average outside temperatures of the prediction phase and the two days prior to it.

    Parameters:
    df_temperatures (pd.DataFrame): The DataFrame containing the temperatures for the prediction period.
    config (dict): The configuration dictionary containing the device information.
    models (Union[str, pd.DataFrame]): The location of the parquet file containing the models, or a DataFrame containing the models.

    Returns:
    pd.Series: The predicted consumption.
    """

    if len(df_temperatures) < 1 or len(config) < 10:
        raise ValueError("config or df_temperatures have too few entries!")

    config = config.copy()

    if config[FeatureName.CONFIG_HEATING_ACTIVE.value] == 0:
        dti = pd.date_range(df_temperatures["date"].iloc[2], periods=len(df_temperatures) - 2,
                            freq="D")
        ps_result = pd.Series(np.zeros(len(dti)), index=dti).round(decimals=1)
        return ps_result

    # infer the number of active HCs in the config
    config_active_hcs = set()
    hcs = {
        "0": FeatureName.CONFIG_HEATING_ACTIVE_HC0.value,
        "1": FeatureName.CONFIG_HEATING_ACTIVE_HC1.value,
        "2": FeatureName.CONFIG_HEATING_ACTIVE_HC2.value,
        "3": FeatureName.CONFIG_HEATING_ACTIVE_HC3.value,
    }
    for k, v in hcs.items():
        if v in config.keys() and config[v] == 1:
            config_active_hcs.add(k)

    # load the model data for the device
    if isinstance(models, str):
        if ".parquet" in models:
            model_df = pd.read_parquet(models)
        else:
            device_uuid = config["device_uuid"]
            model_df = pd.read_parquet(models + f"/device_uuid={device_uuid}")
    else:
        model_df = models

    if len(model_df) == 0:
        raise ValueError("The loaded model_df is empty for the requested device_uuid!")

    # get the latest model for a given set of active HCs
    model_df = model_df[
        model_df["model_name"] == ModelName.TRAIN_PREDICT_CONSUMPTION_FOR_HEATING.value
        ]

    model_df["active_heating_circuits"] = model_df["model_params"].apply(get_active_hcs)
    model_df = model_df[model_df["active_heating_circuits"] == config_active_hcs]

    if len(model_df) == 0:
        raise Exception("The loaded model_df does not contain a model for the device UUID and "
                        "active heating circuits set provided in the config!")

    last_version = model_df["model_date"].max()
    model_df = model_df[model_df["model_date"] == last_version]
    model_bytes = model_df["model"].iloc[0]

    if model_bytes == b"":
        error = model_df["model_params"].iloc[0]
        raise Exception("The last version of the model is unable to provide the prediction. "
                        f"The error encountered in the training phase is: {error}")

    model = model_from_bytes(model_bytes)

    # create dataframe with features
    df_features = df_temperatures.sort_values("date").copy()
    df_features["date"] = df_features["date"].astype(np.datetime64)
    for k, v in config.items():
        df_features[k] = v

    # get durations from schedules
    schedules = {}
    for n in range(4):
        k = "heating_schedule_hc" + str(n)
        if k not in config:
            continue
        schedules[n] = Schedule(
            config[k],
            type="heating",
            offset=config["time_offset"],
        )
        config.pop(k)
        df_features["config_heating_schedule_duration_reduced_hc" + str(n)] = df_features[
            "date"
        ].apply(lambda x: schedules[n].get_duration(x.weekday(), mode="reduced"))
        df_features["config_heating_schedule_duration_normal_hc" + str(n)] = df_features[
            "date"
        ].apply(lambda x: schedules[n].get_duration(x.weekday(), mode="normal"))
        df_features["config_heating_schedule_duration_comfort_hc" + str(n)] = df_features[
            "date"
        ].apply(lambda x: schedules[n].get_duration(x.weekday(), mode="comfort"))
    config.pop("time_offset")

    # get the list of features
    feature_list = convert_json(model_df["model_params"].iloc[0])["features"]
    sensor = True
    if "consumption_estimation" not in feature_list:
        sensor = False

    # prepare the data
    df_features = prepare_data(df_features, weather_controlled=sensor, predict_mode=True)

    # check whether the features match
    if len(list(set(feature_list) & set(list(df_features.columns)))) != len(feature_list):
        raise ValueError(
            "Number of features prepared from config input does not match the number "
            "of features used when training the model!"
        )
    df_features = df_features[feature_list]

    # get prediction
    prediction = model.predict(df_features)
    dti = pd.date_range(df_temperatures["date"].iloc[2], periods=len(df_features), freq="D")
    ps_result = pd.Series(prediction, index=dti).round(decimals=1)
    ps_result.loc[ps_result < 0] = 0

    return ps_result


def get_active_hcs(model_params: str) -> set[str]:
    """
    This function extracts the active heating circuits from the model parameters.

    Parameters:
    model_params (str): The model parameters in JSON format.

    Returns:
    set: The set of active heating circuits.
    """

    json = convert_json(model_params)
    return set() if "active_heating_circuits" not in json else set(json["active_heating_circuits"])


def keep_days_with_matching_active_hcs(df: pd.DataFrame) -> (set, pd.DataFrame):
    """
    This function keeps only the days where the active heating circuits match the current configuration.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be processed.

    Returns:
    set: The set of active heating circuits.
    pd.DataFrame: The DataFrame after processing.
    """

    active_hcs_current = set()
    hcs = {
        "0": FeatureName.CONFIG_HEATING_ACTIVE_HC0.value,
        "1": FeatureName.CONFIG_HEATING_ACTIVE_HC1.value,
        "2": FeatureName.CONFIG_HEATING_ACTIVE_HC2.value,
        "3": FeatureName.CONFIG_HEATING_ACTIVE_HC3.value,
    }

    matching_days = set(df["feature_date"].unique())

    df_current = df[df["feature_date"] == df["feature_date"].max()]

    for hc in hcs:
        hc_active = df_current[df_current["feature_name"] == hcs[hc]]
        if len(hc_active) == 0:
            continue
        hc_active = hc_active.iloc[0]["feature_value"]

        # keep only days with the same state of the heating circuit
        df_tmp = df[(df["feature_name"] == hcs[hc]) & (df["feature_value"] == hc_active)]
        matching_days = matching_days.intersection(set(df_tmp["feature_date"].unique()))

        if hc_active:
            active_hcs_current.add(hc)
        else:
            df = df[~df["feature_name"].str.contains(f"hc{hc}")]
            df = df[~df["feature_name"].str.contains(f"hc_{hc}")]

    return active_hcs_current, df[df["feature_date"].isin(matching_days)]


def generate_random_seed():
    """
    This function generates a random seed for the random number generator.

    Returns:
    int: The random seed.
    np.random.RandomState: The random number generator.
    """

    random_seed = np.random.randint(9999)
    rng = np.random.RandomState(random_seed)
    return random_seed, rng
