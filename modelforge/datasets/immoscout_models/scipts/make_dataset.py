import os
import shutil
from logging import Logger
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from distributed import Client, LocalCluster
from dotenv import find_dotenv, load_dotenv
from geopy import Nominatim, distance
from joblib import Memory
from pyproj import Transformer

from modelforge.datasets.immoscout_models.scipts.constants import (
    BASE_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATA_TYPES,
    NUMERIC_COLUMNS,
    STATS_COLUMNS,
)
from modelforge.datasets.immoscout_models.scipts.create_additional_stats import (
    get_additional_stats,
)
from modelforge.datasets.immoscout_models.scipts.immoscout_housing_model_trainer import (
    ImmoscoutHousingPriceModelTrainer,
)
from modelforge.shared.logger import logger_factory


@click.command()
@click.argument("raw_filepath", type=click.Path(exists=True))
@click.argument("processed_filepath", type=click.Path())
def main(raw_filepath: str, processed_filepath: str):
    logger = logger_factory(__name__)
    logger.info("Clearing previous point and models")
    shutil.rmtree(f"{processed_filepath}", ignore_errors=True)
    os.makedirs(f"{processed_filepath}/point", exist_ok=True)

    # Ignore point before 2015, as e.g. no ev value is reported
    logger.info("Reading point files")
    if not os.path.exists(f"{raw_filepath}/house_price.parquet"):
        logger.info("Reading raw point")
        df = pd.concat(
            [
                pd.read_csv(
                    f"{raw_filepath}/house_price/HKSUF{i}.csv", dtype=DATA_TYPES
                )
                for i in range(2, 15)
            ]
        )
        df = prepare_raw_data(df, logger, raw_filepath)
        df.to_parquet(f"{raw_filepath}/house_price.parquet")
    else:
        logger.info("Reading cached point")
        df = pd.read_parquet(f"{raw_filepath}/house_price.parquet")

    entity_ids = []
    dfs = []
    for kid, frame in df.groupby("kid2015"):
        dfs.append(frame)
        entity_ids.append(str(kid))

    client = LocalCluster().get_client()

    trained_entities, global_error = prepare_data_set_and_train(
        processed_filepath,
        client,
        entity_ids,
        dfs,
        logger,
    )
    error = np.mean(
        [
            entity.metadata["model_metrics"]["test_mean_absolute_percentage_error"]
            for entity in trained_entities
        ]
    )
    plt.hist(
        [
            entity.metadata["model_metrics"]["test_mean_absolute_percentage_error"]
            for entity in trained_entities
        ],
        bins=20,
    )
    plt.axvline(global_error, color="r", linestyle="dashed", linewidth=1)
    plt.savefig(f"{raw_filepath}/error_hist.png")
    logger.info(
        f"Trained {len(trained_entities)} models. Mean absolute percentage error is {error}"
    )
    client.close()


def prepare_data_set_and_train(
    processed_filepath: str,
    client: Client,
    entity_ids: list[str],
    dfs: list[pd.DataFrame],
    logger,
):
    trainer = ImmoscoutHousingPriceModelTrainer(logger, processed_filepath)
    global_model = trainer.train("global", pd.concat(dfs))
    global_error = global_model.metadata["model_metrics"][
        "test_mean_absolute_percentage_error"
    ]
    logger.info("Trained global set. Error: " + str(global_error))
    housing_train_entities = client.map(trainer.train, entity_ids, dfs)
    logger.info(f"Submitted {len(housing_train_entities)} training tasks")
    housing_train_entities = client.gather(housing_train_entities)
    return [
        entity for entity in housing_train_entities if entity is not None
    ], global_error


def prepare_raw_data(
    df: pd.DataFrame, logger: Logger, raw_file_path: str
) -> pd.DataFrame:
    logger.info("Convert some columns to numeric")
    df = convert_to_numeric(df, NUMERIC_COLUMNS)
    logger.info("Clean point")
    df = clean_data(df)
    logger.info("Fill some missing values")
    df = fill_missing_values(df)
    logger.info("Craft some features")
    df = feature_engineering(df)
    logger.info("Replace string values to bool")
    df = replace_with_bool_values(df)
    logger.info("Normalize platform metadata column")
    df = normalize_columns(df)
    logger.info("Encode categorical features")
    df = encode_features(df)
    logger.info("Transform coordinate grid")
    df = transform_coordinates(df)
    logger.info("Add geocoordinates")
    df = add_geocoordinates(df)
    logger.info("Calculate distance to kid center")
    df = calculate_distance(df)
    logger.info("Enrich with additional stats")
    df = merge_additional_stats(df, raw_file_path)
    logger.info("Select relevant columns")
    df = select_final_columns(df)
    return df


def clean_data(df):
    df = df[df["plz"] != "Other missing"]
    df = df.dropna(subset=["plz", "gid2019", "kid2019"])
    df = df[(df["kaufpreis"] > 50_000) & (df["kaufpreis"] <= 1_000_000)]
    return df


def convert_to_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fill_missing_values(df):
    df["mieteinnahmenpromonat"] = df["mieteinnahmenpromonat"].fillna(0)
    df["letzte_modernisierung"] = df["letzte_modernisierung"].fillna(np.nan)
    df["parkplatzpreis"] = df["parkplatzpreis"].fillna(0)
    df["nutzflaeche"] = df["nutzflaeche"].fillna(0)
    return df


def find_parkplatz(x):
    if x["parkplatz"] == "Other missing":
        return x["parkplatzpreis"] != 0
    return x["parkplatz"] == "Yes"


def feature_engineering(df):
    df["parkplatz"] = df.apply(find_parkplatz, axis=1)
    df["baujahr"] = (
        df.groupby("objektzustand")["baujahr"]
        .transform(lambda x: x.fillna(x.mean()))
        .round(0)
        .astype(int)
    )
    df["alter"] = 2024 - df["baujahr"]
    df["alter_letzte_modernisierung"] = 2024 - df["letzte_modernisierung"].fillna(-1)
    df = df.drop(columns=["letzte_modernisierung", "baujahr"])
    return df


def replace_with_bool_values(df):
    df["ev_kennwert"] = (
        df["ev_kennwert"]
        .replace(["Other missing", "Implausible value"], -1)
        .astype(float)
    )
    df["anzahletagen"] = (
        df["anzahletagen"]
        .replace(["Other missing", "Implausible value"], -1)
        .astype(int)
    )
    bool_cols = [
        "denkmalobjekt",
        "einliegerwohnung",
        "ferienhaus",
        "gaestewc",
        "kaufvermietet",
        "keller",
        "rollstuhlgerecht",
    ]
    for col in bool_cols:
        df[col] = df[col].apply(lambda x: x == "Yes")
    return df


def normalize_columns(df):
    for col in ["hits", "click_schnellkontakte", "click_url", "click_weitersagen"]:
        df[col + "_norm"] = df[col] / df["laufzeittage"]
        df = df.drop(columns=[col])
    return df


def encode_features(df):
    df["ad_duration_months"] = (df["ejahr"] - df["ajahr"]) * 12 + (
        df["emonat"] - df["amonat"]
    )
    df["month_sin"] = np.sin(2 * np.pi * (df["amonat"] - 1) / 12)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    df.columns = [
        str(col).replace("[", "").replace("]", "").replace("<", "").replace(">", "")
        for col in df.columns
    ]
    return df


def transform_coordinates(df):
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326")

    def map_coordinates(row):
        if (
            pd.isna(row["ergg_1km"])
            or not isinstance(row["ergg_1km"], str)
            or row["ergg_1km"].startswith("-")
            or row["ergg_1km"] == "Other missing"
        ):
            return np.nan, np.nan
        codes = row["ergg_1km"].split("_")
        if len(codes) != 2:
            return np.nan, np.nan
        x_code, y_code = int(codes[0]) * 1000, int(codes[1]) * 1000
        return transformer.transform(y_code, x_code)

    df[["latitude", "longitude"]] = df.apply(
        map_coordinates, axis=1, result_type="expand"
    )
    return df


def find_geocoordinates(loc):
    def search(erg_amd):
        if pd.isna(erg_amd) or erg_amd == "Other missing":
            return np.nan, np.nan
        if "," in erg_amd:
            erg_amd = erg_amd.split(",")[0]
        print(erg_amd)
        getLoc = loc.geocode(erg_amd)
        if getLoc is None:
            return np.nan, np.nan
        return getLoc.raw["lat"], getLoc.raw["lon"]

    return search


def add_geocoordinates(df):
    loc = Nominatim(user_agent="GetLoc")
    memory = Memory(".cache", verbose=0)
    find_geocoordinates_cached = memory.cache(find_geocoordinates(loc))
    df[["latitude_amd", "longitude_amd"]] = df.apply(
        lambda x: (
            find_geocoordinates_cached(x["erg_amd"])
            if x["erg_amd"] != "Other missing"
            else find_geocoordinates_cached(x["kid2015"])
        ),
        axis=1,
        result_type="expand",
    )
    df = df.drop(columns=["ergg_1km", "erg_amd"])
    return df


def haversine_distance(row):
    if (
        pd.isna(row["latitude"])
        or pd.isna(row["longitude"])
        or pd.isna(row["latitude_amd"])
        or pd.isna(row["longitude_amd"])
    ):
        return np.nan
    return distance.distance(
        (row["latitude"], row["longitude"]), (row["latitude_amd"], row["longitude_amd"])
    ).km


def calculate_distance(df):
    df["distance_erg_center"] = df.apply(haversine_distance, axis=1)
    return df


def merge_additional_stats(df, raw_file_path: str):
    stats_df = get_additional_stats(raw_file_path)
    num_samples = len(df)
    df = df.merge(
        stats_df, left_on=["ajahr", "amonat"], right_on=["year", "month"], how="left"
    )
    assert num_samples == len(
        df
    ), "The number of samples should not change after merging the stats_df."
    df = df.drop(columns=["year", "month"])
    return df


def get_pd_dummy_cols(cols):
    return [
        col
        for col in cols
        if any(
            cat in col
            for cat in [
                "energieeffizienzklasse",
                "energieausweistyp",
                "heizungsart",
                "kategorie_Haus",
                "objektzustand",
                "ausstattung",
            ]
        )
    ]


def select_final_columns(df):
    # Get dummy columns from the DataFrame
    dummy_cols = get_pd_dummy_cols(df.columns)

    # Combine base columns, dummy columns, and stats columns
    selected_cols = BASE_COLUMNS + dummy_cols + STATS_COLUMNS

    # Select the columns from the DataFrame
    df = df[selected_cols]

    # Drop the unnecessary columns in one step
    df.drop(
        columns=[
            "gid2015",
            "plz",
            "amonat",
            "emonat",
            "ejahr",
            "latitude",
            "longitude",
            "latitude_amd",
            "longitude_amd",
        ],
        inplace=True,
    )

    return df


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
