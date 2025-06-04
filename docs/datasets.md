# 📊 Evaluation Datasets

<div align="center">

**Overview of datasets used for evaluating ModelForge**

</div>

The following table provides an overview of the datasets used for ModelForge evaluation. For creating the actual datasets and reproducibility, see [Reproducibility](reproducibility.md).

| Dataset Name | Loss Metric | Total Models | Features | Min Samples | Mean Samples | Max Samples |
|:-----------:|:----------:|:------------:|:--------:|:-----------:|:------------:|:-----------:|
| Energy Consumption | MAE | 919 | 6 | 58 | 293 | 937 |
| Anomaly Detection | ROC AUC | 357 | 50 | 1,435 | 85,604 | 1,149,900 |
| Weather | CRPS | 499 | 34 | 515 | 3,522 | 3,651 |
| House Price | MAPE | 389 | 109 | 1,197 | 32,757 | 6,371,308 |

## 🔥 Heating Dataset

The heating dataset consists of time series data from 919 heating devices, including oil/gas boilers and heat pumps. The data contains daily aggregated energy consumption used for residential heating, along with additional sensor readings and calculated features. Note that the dataset is proprietary and as such not publicly available.

**Features:**

| Feature | Description |
|:-------:|:------------|
| out_temp | Average outside temperature |
| diff_temp_1day | Difference between current outside temperature and previous day |
| diff_temp_2day | Difference between current outside temperature and 2 days ago |
| consumption_estimation | Physical estimation of consumption |
| week_y | Cosine encode calendar week |
| supply_temp | Average supply temperature |

Models trained on this dataset are Random Forests or Gradient Boosting Trees, depending on validation set performance. Features were z-normalized before training.

## 🚨 Anomaly Dataset

This dataset comprises 357 univariate time series collected from multiple sources including NASA-MSL, NASA-SMAP, IOPS, and KDD-TSDA. The datasets can be downloaded [here](https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html). Each datapoint is labeled with a binary indicator (1 for anomalous points, 0 for normal observations).

**Features:**

| Dataset | Origin | Mean Length | Mean Contamination | Count |
|:-------:|:------:|:-----------:|:-----------------:|:-----:|
| KDD-TSAD | synthetic | 77,415 | 0.6% | 250 |
| IOPS | real | 100,649 | 1.8% | 29 |
| NASA-MSL | real | 2,730 | 12.0% | 27 |
| NASA-SMAP | real | 8,070 | 12.4% | 54 |

XGBoost models were trained for each time series using a sliding window technique with a window size of 50 observations to capture temporal patterns. Performance is evaluated using ROC-AUC.

## ☁️ Weather Dataset

This dataset is used for postprocessing 2-m temperature forecasts and contains 48-hour lead time predictions from the European Centre for Medium-Range Weather Forecasts (ECMWF) 50-member ensemble. The dataset can be download [here](https://doi.org/10.6084/m9.figshare.13516301.v1). The data was obtained from the THORPEX Interactive Grand Global Ensemble (TIGGE) archive, with observed data from weather stations provided by the Deutscher Wetterdienst (DWD).

**Features:**

| Feature | Description |
|:-------:|:------------|
| t2m | 2-m temperature |
| cape | Convective available potential energy |
| sp | Surface pressure |
| tcc | Total cloud cover |
| sshf | Sensible heat flux |
| slhf | Latent heat flux |
| u10 | 10-m U wind |
| v10 | 10-m V wind |
| d2m | 2-m dewpoint temperature |
| ssr | Shortwave radiation flux |
| str | Longwave radiation flux |
| sm | Soil moisture |
| u_pl500 | U wind at 500 hPa |
| v_pl500 | V wind at 500 hPa |
| u_pl850 | U wind at 850 hPa |
| v_pl850 | V wind at 850 hPa |
| gh_pl500 | Geopotential at 500 hPa |
| q_pl850 | Specific humidity at 850 hPa |

The dataset spans from 2007-2016, with training data from 2007-2015 and validation using 2016 data. Models were trained for 499 weather stations across Germany that met data availability criteria. Each station has its own [probabilistic regression model](https://github.com/CDonnerer/xgboost-distribution) using Natural Gradient Boosting.

## 🏠 Housing Dataset

The RWI-GEO-RED (housing) dataset originates from Immoscout24 in cooperation with the Forschungsdatenzentrum Ruhr (FDZ Ruhr). It includes over 17.5 million residential properties published on the platform between January 2007 and June 2024. We used the scientifc use-file (SUF) version of the dataset. Further overview is given here: [RWI-GEO-RED](https://www.rwi-essen.de/forschung-beratung/weitere/forschungsdatenzentrum-ruhr/datenangebot/rwi-geo-red-real-estate-data). For a data access request see [here](https://www.rwi-essen.de/forschung-beratung/weitere/forschungsdatenzentrum-ruhr/datenantrag).

**Features:**

| Feature | Description |
|:-------:|:------------|
| population | Total population |
| Pop_under_25 | Population under 25 |
| Pop_25_30 | Population aged 25-30 |
| Pop_30_35 | Population aged 30-35 |
| Pop_35_40 | Population aged 35-40 |
| Pop_40_45 | Population aged 40-45 |
| Pop_45_50 | Population aged 45-50 |
| Pop_50_55 | Population aged 50-55 |
| Pop_55_60 | Population aged 55-60 |
| Pop_over_60 | Population over 60 |
| build_land_price_index_existing_buildings | Price index for existing buildings |
| build_land_price_index_flat | Price index for flats |
| build_land_price_index_new_constructed | Price index for new constructions |
| build_land_price_index_total | Total building & land price index |
| construction_industry_employees | Number of employees in construction |
| construction_industry_employees_change | Change in construction employees |
| construction_industry_turnover | Turnover in construction industry |
| construction_industry_turnover_change | Change in construction turnover |
| construction_industry_turnover_per_employee | Turnover per employee in construction |
| consumption_index | Consumer spending index |
| gdp_change | Change in GDP |
| gdp_per_citizen | GDP per citizen |
| gdp | Gross Domestic Product |
| gross_value_creation | Total value of goods and services produced |
| house_price_index | Index of house prices |
| housing_interest_rate | Interest rate for mortgages |
| num_new_buildings | Number of new buildings |
| num_new_flats | Number of new flats |
| nominallohnindex | Nominal wage index |
| reallohnindex | Real wage index (adjusted for inflation) |
| employed_mio | Number of employed (millions) |
| unemployed_mio | Number of unemployed (millions) |
| unemployed_rate | Unemployment rate |

See the [documentation](https://www.rwi-essen.de/fileadmin/user_upload/RWI/FDZ/FDZ_Datensatzbeschreibung_RED_v11.pdf) of the FDZ Ruhr for more details about the features.

For each of 389 counties, an XGBoost regression model was trained to predict house prices. Geographic information specific to individual counties was removed before training.
