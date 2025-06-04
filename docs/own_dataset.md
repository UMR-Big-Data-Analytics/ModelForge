# 📊 Using Your Own Datasets

<div align="center">

**A guide to integrating custom datasets with ModelForge**

</div>

## 🔍 Overview

ModelForge is centered around the concept of a `ModelEntity` encapsulating a machine learning model, its metadata and associated training and test data. This section provides guidance on how to create your own datasets and integrate them into ModelForge. 

> 💡 For a quick start see this [demo notebook](demo.ipynb).

## 🛠️ Creating a Dataset

To create a dataset, you need to instantiate a `ModelEntity` object for each model. A collection of `ModelEntity` objects can be stored in a `ModelDataSet`. The following code snippet demonstrates how to create a dataset with multiple models.

### Step 1: Create a Model Wrapper

First, create a wrapper for your models to ensure they work seamlessly with ModelForge (here shown for the `scikit-learn` models):

```python
class ModelWrapper(Pipeline, BaseEstimator):
    def __init__(self, model: BaseEstimator):
        super().__init__(steps=[])
        self.model = model
        self._is_fitted = False

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series = None, **kwargs):
        self.model.fit(x_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, x: pd.DataFrame, **kwargs) -> pd.Series:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        return self.model.predict(x)
```

### Step 2: Generate Data and Train Models

Create utility functions for generating data and training models:

```python
def generate_random_data() -> pd.DataFrame:
    data = {
        "a": np.random.random(100),
        "b": np.random.random(100),
        "c": np.random.random(100),
    }
    return pd.DataFrame(data)


def train_model(x: pd.DataFrame, y: pd.Series) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor()
    model.fit(x, y)
    return model
```

### Step 3: Create Model Entities

Define a function to create individual model entities:

```python
def create_model_entity() -> ModelEntity:
    # Generate random data
    x = generate_random_data()
    y = pd.Series(np.random.random(100))

    # Train test split
    x_train = x[:80]
    y_train = y[:80]
    x_test = x[80:]
    y_test = y[80:]

    model = train_model(x_train, y_train)
    model_entity = ModelEntity(
        path="save_dir",
        id=str(uuid4()),
        pipeline=model,
        train_x=x_train,
        train_y=y_train,
        test_x=x_test,
        test_y=y_test,
        loss=mean_absolute_error,
        feature_list=["a", "b", "c"],
        metadata={"description": "Test model"},
    )

    return model_entity
```

## 📦 Building a Model Dataset

Create a dataset by combining multiple model entities:

```python
dataset = ModelDataSet.from_iterable([create_model_entity() for _ in range(100)])
```

## 🚀 Setting Up Distributed Processing

Set up a Dask cluster for distributed processing:

```python
client = LocalCluster(n_workers=1, threads_per_worker=1).get_client()
```

## 🔍 Defining and Executing a Consolidation Pipeline

Define a consolidation pipeline and execute it to analyze your models:

```python
pipeline_factory = PipelineFactory(
    dataset,
    client,
    RetrainingConsolidationStrategy(ModelWrapper(ExtraTreesRegressor()), PandasModelEntityDataMerger()),
)
pipeline = PredictionLossSetPipelineBuilder(
    pipeline_factory,
    KMeans(n_clusters=10),
    UniformSetSampler(3, TargetSelector(), Entropy()),
).build_pipeline()
pipeline.fit(dataset)

score: dict = pipeline.score(dataset)
consolidation_score = ModelConsolidationScore.from_dict(score)
```