# 🔄 Reproducing ModelForge Experiments

<div align="center">

**Step-by-step guide to reproduce ModelForge experiments**

</div>

## 🛠️ Environment Setup

Follow these steps to set up your environment:

1. Install Python 3.11 for your system.
2. Install [poetry](https://python-poetry.org/docs/#installation) for your system.
3. Install packages:
   ```bash
   poetry install
   ```

## 📊 Dataset Preparation

### Prepare Directories

First, ensure you have the proper directory structure in place for the datasets.

### Housing Dataset

1. Download [RWI-GEO-RED](https://www.rwi-essen.de/forschung-beratung/weitere/forschungsdatenzentrum-ruhr/datenangebot/rwi-geo-red-real-estate-data) by placing a data access request (see [here](https://www.rwi-essen.de/forschung-beratung/weitere/forschungsdatenzentrum-ruhr/datenantrag)).
2. Unzip the downloaded file.
3. Place the `HKSUF*.csv` file in the `data/raw/immoscout24_models/house_price` folder.

### Anomaly Dataset

1. Download IOPS, KDD-TSAD, NASA-MSL, and NASA-SMAP datasets from [here](https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html).
2. Unzip the downloaded files.
3. Place the `IOPS`, `KDD-TSAD`, `NASA_MSL` and `NASA-SMAP` folders in the `data/raw/anomaly_models/univariate` folder.

### Weather Dataset

1. Download the ECMWF dataset from [here](https://doi.org/10.6084/m9.figshare.13516301.v1).
2. Place the `data_RL18.feather` in `data/raw/weather_models/` folder.

### Train Models

To train the models, run the following command:
```bash
./run_training.sh
```

## 🧪 Run Experiments

To run the experiments, it is recommended to use background processing because the computation takes a lot of time. We recommend to set `N_DASK_WORKERS` accordingly to your setup.

To run the experiment, use the following command:
```bash
./run_experiments.sh
```

The raw results will be saved in the `data/results` folder. The results are saved in the following format:
```
data/results/<experiment_name>/<dataset_name>/<embedding_category>_<date>_<time>/<embedding_pipeline_name>/<num_clusters>/
```

Each of the folders contains the following files:
- `clustering.csv`: The cluster labels of the models.
- `embedding.csv`: The computed embeddings of the models.
- `scoring.csv`: The consolidation scores for each cluster.
- `scores.json`: The computed scores for each clustering.
- `params.json`: Additional parameters of the pipelines.

## Create plots

To create the plots, run the following command:
```bash
./run_plots.sh
```

The plots will be saved in the `data/reports` folder.
