import glob
import os
import shutil
from logging import Logger
from pathlib import Path

import click
from distributed import Client, LocalCluster
from dotenv import find_dotenv, load_dotenv

from modelforge.datasets.anomaly_models.scripts.anomaly_model_trainer import (
    AnomalyModelTrainer,
)
from modelforge.shared.logger import logger_factory


@click.command()
@click.argument("raw_filepath", type=click.Path(exists=True))
@click.argument("processed_filepath", type=click.Path())
@click.option(
    "--n-models",
    type=click.INT,
    default=-1,
    help="Number of models to train. -1 means all models",
)
def main(raw_filepath: str, processed_filepath: str, n_models: int):
    logger = logger_factory(__name__)
    logger.info("Clearing previous point and models")
    shutil.rmtree(f"{processed_filepath}", ignore_errors=True)
    os.makedirs(f"{processed_filepath}/point", exist_ok=True)
    pattern = f"{raw_filepath}/**/*.test.csv"

    # Get a list of all files that match the pattern
    filenames = glob.glob(pattern, recursive=True)
    if n_models > 0:
        filenames = filenames[:n_models]
    client = LocalCluster().get_client()
    success_filenames = prepare_data_set_and_train(
        raw_filepath, processed_filepath, client, filenames, logger
    )
    logger.info(f"Finished training models. Trained {len(success_filenames)} models.")

    # Shutdown the client
    client.shutdown()


def prepare_data_set_and_train(
    raw_path: str,
    processed_filepath: str,
    client: Client,
    filenames: list[str],
    logger: Logger,
):
    logger.info("Preparing point set and training models")
    results = []
    i = 1
    trainer = AnomalyModelTrainer(logger, raw_path, processed_filepath)
    for filename in filenames:
        logger.info(f"{i}/{len(filenames)} Processing {filename}")
        result = client.submit(trainer.train, filename)
        results.append(result)
        logger.info(f"{i}/{len(filenames)} Submitted work for {filename}")
        i += 1

    results = client.gather(results)
    return [
        filename for filename, result in zip(filenames, results) if result is not None
    ]


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
