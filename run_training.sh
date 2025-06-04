#!/usr/bin/env bash
python -m modelforge.datasets.weather_models.scripts.make_dataset ./data/raw/weather_models/ ./data/processed/weather_models/
python -m modelforge.datasets.immoscout_models.scipts.make_dataset ./data/raw/immoscout24_models/ ./data/processed/immoscout24_models/
python -m modelforge.datasets.anomaly_models.scripts.make_dataset ./data/raw/anomaly_models/univariate/ ./data/processed/anomaly_models/