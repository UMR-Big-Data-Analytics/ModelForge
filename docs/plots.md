# 📈 Experimental Results

<div align="center">

**Visualizations of ModelForge embedding and clustering performance**

</div>

See the paper for more details on the experiments and explanation. Further interpretation of the embedding space, which had to be obmitted in the paper due to space constraints, is provided [below](#inspection-of-embedding-space).

## 🎯 Effectiveness of embedding strategies

<div align="center">
<img src="paper_plots/agg_performance.png" alt="Aggregated Performance Scores" width="700px" />
<p><em>Aggregated performance scores across different embedding strategies</em></p>
</div>

<div align="center">
<img src="paper_plots/dataset_agg_performance.png" alt="Aggregated Performance Scores per dataset" width="700px" />
<p><em>Aggregated performance scores broken down by dataset</em></p>
</div>

<div align="center">
<img src="paper_plots/overall_scaled.png" alt="Scores over number of clusters" width="700px" />
<p><em>Performance scores as the number of clusters increases</em></p>
</div>

## 🧩 Influence of embedding size

<div align="center">
<img src="paper_plots/embedding_size_top_3.png" alt="Influence of embedding size" width="700px" />
<p><em>Effect of embedding dimensionality on performance</em></p>
</div>

<div align="center">
<img src="paper_plots/dataset_nmi.png" alt="NMI vs. embedding size" width="700px" />
<p><em>Normalized Mutual Information scores at different embedding dimensions</em></p>
</div>

<div align="center">
<img src="paper_plots/knn_graph_dataset.png" alt="NN(p) distance graph" width="700px" />
<p><em>k-Nearest Neighbor distance analysis</em></p>
</div>

## 🔬 Inspection of embedding space

<div align="center">
<img src="paper_plots/2d_embedding.png" alt="2D UMAP" width="700px" />
<p><em>2D UMAP projection of the model embedding space</em></p>
</div>

### Prediction error of models in the embedding space

To further understand the embedding space, we inspect the ends of the "U" shape by getting 5 models for each "end" of the U shape. We then plot the prediction error of these models against the training sets, which have been used for constructing the embedding space. The results indicate that there are two major behavioral paths: Some overestimate the prediction error, while others underestimate it.

#### Heating Dataset

<div align="center">
<img src="paper_plots/heating_prediction_error.png" alt="Heating Arms" width="700px" />
<p><em>Prediction error analysis for models in the Heating dataset</em></p>
</div>

#### Weather Dataset

<div align="center">
<img src="paper_plots/weather_prediction_error.png" alt="Weather Arms" width="700px" />
<p><em>Prediction error analysis for models in the Weather dataset</em></p>
</div>

#### Housing Dataset

<div align="center">
<img src="paper_plots/house_price_prediction_error.png" alt="Housing Arms" width="700px" />
<p><em>Prediction error analysis for models in the Housing dataset</em></p>
</div>
