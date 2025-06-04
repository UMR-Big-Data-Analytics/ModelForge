#!/usr/bin/env bash

LOG_FILE="logs/experiment_log_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "WARNING: This script takes a lot of time to run. It is recommended to run it in the background on a server."
echo ""
echo "Running experiments: Method ranking, Embedding dimensionality and Consolidation strategy."
echo "Depending on your environment you might want to adjust the DASK_N_WORKERS environment variable."

echo "Running Method ranking experiment"
python -m modelforge.experiments.method_ranking.experiment_definitions

echo "Running Embedding dimensionality experiment"
python -m modelforge.experiments.embedding_dimensionality.experiment_definitions

echo "Running Consolidation strategy experiment"
python -m modelforge.experiments.consolidation_strategy.experiment_definitions