# -*- coding: utf-8 -*-
import itertools
import logging
import os
import shutil
from pathlib import Path

import click
import joblib
import pandas as pd
from dask import dataframe as dd
from dask.distributed import LocalCluster
from distributed import Client
from dotenv import find_dotenv, load_dotenv

from src.data.model_trainer import ModelTrainer
from src.shared.dask import apply_to_partition
from src.shared.logger import logger_factory
from src.shared.persistence.data_path_factory import DataPathFactory
from src.shared.persistence.model_data_factory import ModelDataFactory


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--n-models', type=click.INT, default=-1, help='Number of models to train. -1 means all models')
def main(input_filepath: str, output_filepath: str, n_models: int):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logger_factory(__name__)
    logger.info('Clearing previous data and models')
    shutil.rmtree(f"{output_filepath}/data", ignore_errors=True)
    shutil.rmtree(f"{output_filepath}/models", ignore_errors=True)
    shutil.rmtree(f"{output_filepath}/pairwise_predictions", ignore_errors=True)
    os.makedirs(f"{output_filepath}/data", exist_ok=True)
    os.makedirs(f"{output_filepath}/models", exist_ok=True)
    os.makedirs(f"{output_filepath}/pairwise_predictions", exist_ok=True)

    data_factory = ModelDataFactory(
        DataPathFactory(input_filepath, output_filepath)
    )

    filenames = [f for f in os.listdir(input_filepath) if f.startswith('device_uuid=')]
    if n_models > 0:
        filenames = filenames[:n_models]

    client = LocalCluster().get_client()

    trained_devices = prepare_data_set_and_train(client, filenames, logger, data_factory)

    index_pairwise_predictions(client, trained_devices, logger, data_factory)


def index_pairwise_predictions(client: Client, device_uuids: list[str], logger: logging.Logger,
                               data_factory: ModelDataFactory):
    device_uuids.sort()
    logger.info('Indexing pairwise predictions')

    # Create a function to calculate the pairwise distance
    def calculate_pairwise_distance(row):
        # After setting the index, the row name is the model_device_uuid
        device_uuid_a, device_uuid_b = row.name, row['test_set_device_uuid']
        feature_list = data_factory.feature_list(device_uuid_a)
        model = data_factory.get_model(device_uuid_a)
        X_test = data_factory.get_test_data(device_uuid_b)[feature_list]
        return model.predict(X_test)

    def index_for_model_device_uuid(model_device_uuid: str, device_uuids: list[str]):
        prediction_df = pd.DataFrame({
            'model_device_uuid': [model_device_uuid] * len(device_uuids),
            'test_set_device_uuid': device_uuids
        })
        prediction_df = prediction_df.set_index('model_device_uuid')
        prediction_df['y_pred'] = prediction_df.apply(calculate_pairwise_distance, axis=1)
        prediction_df.to_parquet(f"{data_factory.data_path_factory.output_path}/pairwise_predictions/{model_device_uuid}.parquet")
        logger.info(f'Indexed pairwise predictions for {model_device_uuid}')

    # Submit work to happen in parallel
    results = []
    i = 1
    for device_uuid in device_uuids:
        result = client.submit(index_for_model_device_uuid, device_uuid, device_uuids)
        results.append(result)
        logger.info(f'{i}/{len(device_uuids)} Submitted work for {device_uuid}')
        i += 1
    # Gather results back to local computer
    client.gather(results)

    # Create a dataframe with the actual values
    df = dd.from_pandas(pd.DataFrame({'device_uuid': device_uuids}), npartitions=1)

    def get_actual_consumption(row):
        # After setting the index, the row name is the device_uuid
        return data_factory.get_test_data(row['device_uuid'])['consumption_heating'].to_numpy()

    df['actual'] = df.map_partitions(apply_to_partition(get_actual_consumption), meta=(None, 'object'), )
    df = df.set_index('device_uuid', sort=True)
    df.to_parquet(f"{data_factory.data_path_factory.output_path}/actual_values")


def prepare_data_set_and_train(client: Client, filenames: list[str], logger: logging.Logger,
                               data_factory: ModelDataFactory) -> list[str]:
    logger.info('Preparing final data set from raw data and training models')
    # Submit work to happen in parallel
    results = []
    i = 1
    model_trainer = ModelTrainer(data_factory)
    for filename in filenames:
        device_uuid = filename.split('=')[1]
        data = client.submit(model_trainer.process, filename, device_uuid)
        result = client.submit(model_trainer.train, data, device_uuid)
        results.append(result)
        logger.info(f'{i}/{len(filenames)} Submitted work for {device_uuid}')
        i += 1
    # Gather results back to local computer
    devices = client.gather(results)
    # Filter out the devices that failed to train
    devices = [device for device in devices if device is not None]
    logger.info(f"Trained {len(devices)} models")
    return devices


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
