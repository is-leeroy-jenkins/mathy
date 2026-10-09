###### Mathy-Py

<p align="center">
  <img src="resources/Mathy.png" alt="Mathy logo" width="800">
</p>
<p align="left">
  <a href="#-overview">Overview</a> ·
  <a href="#-application-modes">Modes</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-data-sources">Sources</a> ·
  <a href="#-data-profile">Profile</a> ·
  <a href="#-statistics">Statistics</a> ·
  <a href="#-anomaly-detection">Anomalies</a> ·
  <a href="#-feature-engineering">Features</a> ·
  <a href="#-classification-models">Classification</a> ·
  <a href="#-regression-models">Regression</a> ·
  <a href="#-clustering-models">Clustering</a> ·
  <a href="#-time-series-models">Time-Series</a> ·
  <a href="#-data-management">Data</a> ·
  <a href="#-framework-modules">Modules</a> ·
  <a href="#-requirements">Requirements</a> ·
  <a href="#-quickstart-example">Quickstart</a> ·
  <a href="#-license">License</a>
</p>

___

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0078FC?style=for-the-badge&logo=github)](https://is-leeroy-jenkins.github.io/mathy/)


Mathy-Py is a machine-learning, statistical-analysis, data-transformation, and
SQLite data-management workspace. It combines a modular Python wrapper framework with an
interactive application for profiling datasets, computing descriptive and inferential statistics,
detecting anomalies, engineering features, training supervised and unsupervised models, forecasting
time series, and administering local data tables.

Mathy is built for repeatable analytical workflows. Users can load default data, database tables, or
custom spreadsheets; inspect and edit data; profile schema and distributions; apply preprocessing
operations; train classification, regression, clustering, and forecasting models; and manage local
SQLite data through a guarded SQL/data-administration interface.

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/Mathy/blob/main/resources/mathy-demo.gif)
___
## ☁️ Cloud

<table>
<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://mathy.happyground-bf8f32a5.centralus.azurecontainerapps.io">
<img src="https://img.shields.io/badge/Docker-App-2496ED?logo=docker&logoColor=white" alt="Docker App">
</a>
</td>


<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://mathy-py.streamlit.app/">
<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit App">
</a>
</td>


<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://drive.google.com/file/d/1GJg9U-xR0YpyqZxraZR-NpzDLEo16lKq/view?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>
</td>

<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://dbc-a0c21f80-7bb3.cloud.databricks.com/browse/folders/3169291152435631?o=7474645703081351">
<img src="https://img.shields.io/badge/Databricks%20Repo-Mathy%20Py-FF3621?logo=databricks&logoColor=white" alt="Databricks Notebook">
</a>
</td>


<td align="center">
<img width="190" height="1" alt=""><br>
<a href="https://leeroy.usw-16.palantirfoundry.com/shares/links/fahcjyqmbp552">
<img src="https://img.shields.io/badge/Palantir%20-Repo-101113?logo=palantir&logoColor=white" alt="Palantir Repo">
</a>
</td>
</tr>
</table>

## 🧠 Overview

| Capability             | Description                                                                                                                                                      |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data loading           | Load the default Excel workbook, SQLite database tables, or user-uploaded `.xlsx`, `.xls`, and `.csv` files.                                                     |
| Data profiling         | Infer schema, edit records, inspect missingness, cardinality, labels, numeric distributions, and export edited datasets.                                         |
| Descriptive statistics | Generate summary statistics, percentiles, missingness, zeros, skewness, kurtosis, distributions, Q-Q plots, correlation matrices, heatmaps, and PCA diagnostics. |
| Inferential statistics | Run normality tests, group comparisons, correlation analysis, categorical association, ANOVA, Kruskal-Wallis, chi-square, and effect-size style metrics.         |
| Anomaly detection      | Detect outliers through Z-score, modified Z-score, IQR fences, Mahalanobis distance, Isolation Forest, and Local Outlier Factor.                                 |
| Feature engineering    | Apply scalers, imputers, encoders, transformers, text vectorizers, feature hashing, dimensionality reduction, and feature selection.                             |
| Classification         | Train and evaluate categorical/discrete target models with scoring and visualization outputs.                                                                    |
| Regression             | Train and evaluate continuous target models with scoring and visualization outputs.                                                                              |
| Clustering             | Train unsupervised clustering models and inspect assignments, counts, metrics, centroids, and scatter visualizations.                                            |
| Forecasting            | Build time-series selections and run lag/ARIMA/SARIMA-style forecasting workflows.                                                                               |
| SQLite administration  | Import spreadsheets, browse tables, perform CRUD, explore, filter, aggregate, visualize, alter schema, index columns, and run guarded SQL queries.               |

## 🧩 Framework Modules

| Module                                                                                     | Purpose                                                         |
|--------------------------------------------------------------------------------------------| --------------------------------------------------------------- |
| [classifications](https://github.com/is-leeroy-jenkins/Mathy/blob/main/classifications.py) | Classification model wrappers.                                  |
| [regressions](https://github.com/is-leeroy-jenkins/Mathy/blob/main/regressions.py)                                                                            | Regression model wrappers.                                      |
| [clusters](https://github.com/is-leeroy-jenkins/Mathy/blob/main/clusters.py)                                                                               | Clustering wrappers.                                            |
| [forecasting](https://github.com/is-leeroy-jenkins/Mathy/blob/main/forecasting.py)                                                                            | Time-series forecasting wrappers.                               |
| [outliers](https://github.com/is-leeroy-jenkins/Mathy/blob/main/outliers.py)                                                                                  | Outlier and novelty detection wrappers.                         |
| [encoders](https://github.com/is-leeroy-jenkins/Mathy/blob/main/encoders.py)                                                                                  | Label, categorical, target, and polynomial encoders.            |
| [scalers](https://github.com/is-leeroy-jenkins/Mathy/blob/main/scalers.py)                                                                                   | Scaling wrappers.                                               |
| [imputers](https://github.com/is-leeroy-jenkins/Mathy/blob/main/imputers.py)                                                                                  | Missing-value imputation wrappers.                              |
| [transformers](https://github.com/is-leeroy-jenkins/Mathy/blob/main/transformers.py)                                                                              | Binarizers, vectorizers, hashers, and column transformers.      |
| [features](https://github.com/is-leeroy-jenkins/Mathy/blob/main/features.py)                                                                                  | Feature selection and dimensionality-reduction utilities.       |
| [boogr](https://github.com/is-leeroy-jenkins/Mathy/blob/main/boogr.py)                                                                                     | Error handling and diagnostics.                                 |
| [config](https://github.com/is-leeroy-jenkins/Mathy/blob/main/config.py)                                                                                    | App constants, mode labels, help text, paths, and style assets. |

## 🧭 Application Modes



| Mode                       | Purpose                                                                          | Primary Outputs                                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Data Profile**           | Load, inspect, edit, and profile the working dataset.                            | Schema inference, data editor, type counts, missingness charts, cardinality charts, numeric distributions, exportable dataset.    |
| **Descriptive Statistics** | Summarize numeric variables and inspect distribution/correlation structure.      | Summary table, histograms, KDE overlays, Q-Q plots, Shapiro metrics, correlation matrix, heatmap, PCA explained variance.         |
| **Inferential Statistics** | Run hypothesis-oriented statistical checks and association tests.                | Normality results, ANOVA, Kruskal-Wallis, Pearson/Spearman correlations, chi-square, Cramér's V, contingency heatmap.             |
| **Anomaly Detection**      | Identify outliers and anomalous observations through statistical and ML methods. | Flagged rows, consensus counts, ECDF plots, violin/box summaries, bivariate anomaly scatter, CSV export.                          |
| **Classification Models**  | Build preprocessing pipelines and train categorical/discrete target models.      | Working dataset, processed dataset, trained classifier, predictions, scores, confusion matrix/ROC-style outputs where applicable. |
| **Regression Models**      | Build preprocessing pipelines and train continuous target models.                | Working dataset, processed dataset, trained regressor, predictions, scores, residual/fit diagnostics where applicable.            |
| **Clustering Models**      | Prepare numeric feature spaces and train unsupervised clustering models.         | Cluster labels, counts, metrics, centroids, detail tables, scatter plots.                                                         |
| **Time-Series Models**     | Select time-series fields and run forecasting wrappers.                          | Time-series splits, lagged features, forecast outputs, model diagnostics.                                                         |
| **Data Management**        | Manage the local SQLite database and imported tables.                            | Imported tables, browsed records, CRUD changes, profiles, filters, aggregates, visualizations, schema changes, SQL results.       |

## 🏛 Architecture

![](https://github.com/is-leeroy-jenkins/Mathy/blob/main/resources/mathy-architecture.png)

___


```text
Data Source
    │
    ├── Default Excel Data
    ├── SQLite Database Tables
    └── Uploaded XLSX / XLS / CSV
            │
            ▼
      Shared Session State
            │
            ├── df_dataset      # active loaded dataset
            ├── df_original     # original/base copy
            ├── df_working      # user-selected modeling subset
            ├── df_processed    # transformed/model-ready data
            ├── df_features     # active feature matrix
            └── df_targets      # active target matrix
            │
            ├── Data Profile / Statistics / Anomaly Detection
            ├── Classification / Regression / Clustering / Forecasting
            └── Data Management / SQLite Administration
```

## 🗂 Layout

```text
mathy/
├── app.py                  # Main Streamlit application
├── config.py               # App constants, paths, labels, modes, help text, and styling
├── classifications.py      # Classification wrappers
├── regressions.py          # Regression wrappers
├── clusters.py             # Clustering wrappers
├── forecasting.py          # Time-series forecasting wrappers
├── outliers.py             # Outlier and novelty detection wrappers
├── encoders.py             # Label, categorical, target, and polynomial encoders
├── scalers.py              # Feature scaling wrappers
├── imputers.py             # Missing-value imputation wrappers
├── transformers.py         # Binarizers, text vectorizers, hashers, and column transformers
├── features.py             # Feature selection and dimensionality-reduction utilities
├── boogr.py                # Error handling and diagnostics
├── minion.py               # Utility helpers
├── stores/
│   └── sqlite/             # SQLite database storage
├── resources/              # Images, notebooks, and supporting assets
└── README.md
```

## 📥 Data Sources

The sidebar supports three source modes.

| Source            | Description                                                                   | File / Storage Path                       |
| ----------------- | ----------------------------------------------------------------------------- | ----------------------------------------- |
| **Default Data**  | Loads the configured default Excel workbook from `cfg.DEFAULT_DATA`.          | Excel workbook configured in `config.py`. |
| **Database Data** | Lists local SQLite tables and loads a selected table into the active dataset. | `cfg.DB_PATH`.                            |
| **Custom Data**   | Uploads a spreadsheet from the browser.                                       | `.xlsx`, `.xls`, `.csv`.                  |

Once loaded, Mathy stores the dataset into shared session state as `df_dataset`, `df_original`, and
`raw_df`, allowing later modes to reuse the same working data without reloading it.

## 🧾 Data Profile

| Section               | Description                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| Data                  | Displays the active dataset in an editable table.                                               |
| Types                 | Infers columns as numeric, ordinal/identifier, categorical, or datetime.                        |
| Records               | Allows row-level editing with type-aware widgets.                                               |
| Diagnostics           | Shows column type distribution and top columns by missing percentage.                           |
| Cardinality           | Shows top columns by unique-value count.                                                        |
| Labels                | Supports column drops, column renames, reset to original, and CSV export.                       |
| Numeric Distributions | Shows histograms, optional KDE overlays, distribution metrics, and mean/median reference lines. |

## 📊 Statistics

### Descriptive Statistics

| Component                    | Description                                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Summary table                | Computes count, mean, standard deviation, variance, min, max, optional percentiles, missing values, zero counts, skewness, and kurtosis. |
| Distribution view            | Renders histogram/KDE and Q-Q plot diagnostics for selected numeric variables.                                                           |
| Correlation structure        | Computes Pearson or Spearman correlation tables and heatmaps.                                                                            |
| Principal Component Analysis | Standardizes selected variables and displays explained variance by component.                                                            |

### Inferential Statistics

| Component               | Description                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| Summary                 | Computes a compact table of normality, group comparison, association, and categorical association statistics. |
| Normality test          | Runs Shapiro-Wilk and renders Q-Q plots for selected numeric variables.                                       |
| Group comparison        | Runs one-way ANOVA and Kruskal-Wallis over a selected grouping variable.                                      |
| Correlation analysis    | Runs Pearson and Spearman correlation between paired numeric variables.                                       |
| Categorical association | Runs chi-square tests and Cramér's V over two categorical variables.                                          |

## 🚨 Anomaly Detection

Mathy supports both univariate and multivariate anomaly detection.

| Method               | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| Z-Score              | Flags observations outside a selected standard-deviation threshold.     |
| Modified Z-Score     | Uses median absolute deviation for robust univariate outlier detection. |
| IQR Fence            | Uses interquartile-range lower/upper fences.                            |
| Mahalanobis Distance | Detects multivariate distance outliers when covariance is invertible.   |
| Isolation Forest     | Uses tree-based isolation for multivariate anomaly detection.           |
| Local Outlier Factor | Uses local density deviation to flag outliers.                          |

Outputs include a flagged-observation table, consensus-strength distribution, empirical cumulative
distribution function plots, violin/box summaries, bivariate scatter views, and CSV export.

## 🧰 Feature Engineering

The Classification, Regression, and Clustering workflows expose a shared feature-engineering surface.
Transformations are applied to `df_processed` after the first processing step; the original user
selection remains preserved as `df_working`.

| Group                                | Tools                                                                                                                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data Scaling                         | Standard Scaler, Min-Max Scaler, Robust Scaler, Normal Scaler, Max-Absolute Scaler.                                                                                          |
| Data Imputation                      | Mean Imputer, Nearest Neighbor Imputer, Iterative Imputer, Simple Imputer.                                                                                                   |
| Data Encoding                        | One-Hot Encoder, Ordinal Encoder, Label Encoder, Target Encoder, Polynomial Features.                                                                                        |
| Data Transformation                  | Binarizer, Label Binarizer, Multi-Label Binarizer, TF-IDF Transformer, Column Transformer.                                                                                   |
| Feature Extraction                   | TF-IDF Vectorizer, Count Vectorizer, Hash Vectorizer, Dictionary Vectorizer, Feature Hasher.                                                                                 |
| Dimensionality Reduction / Selection | Variance Threshold, Canonical Correlation Analysis, Principal Component Analysis, Select-Best, Select-Percent, Sequential Backward Selection, Recursive Feature Elimination. |

## 🧪 Classification Models

Classification mode trains supervised models for categorical or discrete targets. The workflow is:
select features and target, create a working dataset, apply feature engineering, split train/test
sets, train a model, and inspect scores/predictions.

| Model Family                | Models / Wrappers                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------- |
| Linear models               | Perceptron, Ordinary Least Squares, Logistic Regression.                                       |
| Regularized / margin models | Support Vector Classification and related regularized classification controls.                 |
| Tree models                 | Decision Tree.                                                                                 |
| Ensemble models             | Random Forest, Bagging, Adaptive Boosting, Gradient Boosting, optional XGBoost when installed. |
| Instance-based models       | Nearest Neighbor / K-Nearest Neighbors.                                                        |
| Probabilistic baselines     | Gaussian Naive Bayes where supported.                                                          |

Common classification outputs include elapsed training time, predictions, score tables, confusion
matrix, accuracy, precision, recall, F1 score, and ROC/AUC paths where applicable.

## 📈 Regression Models

Regression mode trains supervised models for continuous targets. It mirrors the classification
workflow but targets numeric dependent variables.

| Model Family                  | Models / Wrappers                                                                                   |
| ----------------------------- | --------------------------------------------------------------------------------------------------- |
| Linear models                 | Ordinary Least Squares, Ridge Regression, Lasso Regression, Elastic Net, Bayesian Ridge.            |
| Stochastic / iterative models | Gradient Descent / SGD-style regression controls.                                                   |
| Instance-based models         | Nearest Neighbor regression.                                                                        |
| Kernel models                 | Gaussian Process and support-vector style regression where wrappers are available.                  |
| Tree models                   | Decision Tree regression.                                                                           |
| Ensemble models               | Random Forest, Gradient Boosting, AdaBoost, Bagging, Voting, Stacking where wrappers are available. |

Outputs include predictions, score tables, elapsed time, train/test split settings, and model-specific
fit diagnostics.

## 🧭 Clustering Models

Clustering mode prepares unsupervised numeric feature spaces and trains clustering models.

| Model                | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| K-Means              | Centroid-based clustering with cluster labels, counts, and centroid tables. |
| DBSCAN               | Density-based clustering with noise/outlier handling.                       |
| Agglomerative        | Hierarchical clustering.                                                    |
| Spectral             | Graph/spectral clustering.                                                  |
| OPTICS               | Density ordering and reachability-style clustering.                         |
| Mean Shift           | Mode-seeking clustering.                                                    |
| Affinity Propagation | Message-passing exemplar clustering.                                        |
| Birch                | Incremental clustering using clustering-feature trees.                      |

The app tracks cluster results, counts, metrics, centroids, detail records, selected plot features, and
cluster signatures in session state.

## ⏱ Time-Series Models

Time-Series mode supports time-indexed forecasting and split diagnostics.

| Component             | Description                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| Time-Series Selection | Select a series/date/index field and target series for modeling.                                |
| Model Selection       | Run lag-based and ARIMA/SARIMA-style models where configured.                                   |
| Split Diagnostics     | Display time-series splits through expandable controls.                                         |
| Forecasting Wrappers  | LaggingSeries, LagBoostingSeries, ARIMA, SARIMA, and TimeSeriesSpliter are imported by the app. |

## 🗄 Data Management

Data Management provides a SQLite administration workspace for local data operations.

| Tab           | Purpose                                                                                                                      |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Import**    | Import Excel workbooks into SQLite with transaction handling and optional overwrite.                                         |
| **Browse**    | Select and browse local SQLite tables.                                                                                       |
| **CRUD**      | Insert, update, and delete rows using schema-aware controls.                                                                 |
| **Explore**   | Page through records with configurable page size.                                                                            |
| **Filter**    | Apply column-value substring filters.                                                                                        |
| **Aggregate** | Compute count, sum, average, minimum, maximum, or median over numeric columns.                                               |
| **Visualize** | Render histogram, bar, line, scatter, box, pie, and correlation charts.                                                      |
| **Admin**     | Profile data, drop tables, create indexes, create custom tables, inspect schema, add/rename/drop columns, and rename tables. |
| **SQL**       | Run guarded read-only SQL queries and inspect result timing.                                                                 |

The SQL console blocks mutation statements and accepts read-only query patterns such as `SELECT`,
`WITH`, `EXPLAIN`, and read-only `PRAGMA` usage.

## ⚙️ Quickstart Example

```python
import pandas as pd

from encoders import LabelEncoder
from scalers import StandardScaler
from classifications import LogisticRegression


df = pd.read_csv("data.csv")
y = df.pop("Label").values
X = df.values

y_enc = LabelEncoder().train_transform(y)
X_scaled = StandardScaler().train_transform(X)

model = LogisticRegression().train(X_scaled, y_enc)
print("Accuracy:", model.score(X_scaled, y_enc))
```
## 🧮 Example Usage

Mathy provides a consistent class-based interface for preparing data, transforming features, training models, generating predictions, and evaluating analytical results. Most modeling classes follow the same general workflow:

1. Create the model.
2. Split the data when appropriate.
3. Train the model.
4. Project or predict results.
5. Score and analyze the model.

The following examples assume that the Mathy modules are available from the project directory.

## 📊 Profile and Partition a Dataset

- The `DataSource` class examines a pandas dataframe, identifies numeric and categorical columns, calculates descriptive statistics, and creates reproducible training and testing partitions.

```python
from sklearn.datasets import load_iris
from data import DataSource


iris = load_iris( as_frame=True )
df_iris = iris.frame.rename( columns={ 'target': 'species' } )

source = DataSource(
	df=df_iris,
	target='species',
	size=0.20,
	rando=42
)

print( f'Samples: {source.n_samples}' )
print( f'Features: {source.n_features}' )
print( f'Feature names: {source.feature_names}' )
print( f'Target values: {source.target_names}' )
print( source.numeric_metrics )
print( source.covariance )
```

- The prepared partitions are available directly from the object:

```python
X_train = source.X_training
X_test = source.X_testing
y_train = source.y_training
y_test = source.y_testing

print( X_train.shape )
print( X_test.shape )
```

- `DataSource` also exposes distribution and relationship visualizations:

```python
source.create_histogram( )
source.create_heatmap( numeric=True )
```

## 🧹 Impute, Encode, and Scale Features

- Mathy preprocessing classes share a `train`, `transform`, and `train_transform` interface. The following example prepares numeric and categorical data for a downstream model.

```python
import numpy as np
from encoders import OneHotEncoder
from imputers import SimpleImputer
from scalers import StandardScaler


numeric_data = np.array(
	[
		[ 42.0, 72000.0 ],
		[ 35.0, np.nan ],
		[ np.nan, 81000.0 ],
		[ 51.0, 94000.0 ],
		[ 29.0, 61000.0 ]
	],
	dtype=float
)

categorical_data = np.array(
	[
		[ 'East' ],
		[ 'West' ],
		[ 'East' ],
		[ 'South' ],
		[ 'West' ]
	]
)

imputer = SimpleImputer( strategy='median' )
imputed_data = imputer.train_transform( numeric_data )

scaler = StandardScaler( )
scaled_data = scaler.train_transform( imputed_data )

encoder = OneHotEncoder( sparse=False, unknown='ignore' )
encoded_data = encoder.train_transform( categorical_data )

X_prepared = np.hstack( (scaled_data, encoded_data) )

print( X_prepared )
print( f'Prepared shape: {X_prepared.shape}' )
print( f'Categories: {encoder.categories}' )
```

- For production modeling, fit the preprocessing classes to the training partition and use `transform` for validation, testing, and future data:

```python
imputer = SimpleImputer( strategy='median' )
imputer.train( X_train )

X_train_imputed = imputer.transform( X_train )
X_test_imputed = imputer.transform( X_test )

scaler = StandardScaler( )
scaler.train( X_train_imputed )

X_train_scaled = scaler.transform( X_train_imputed )
X_test_scaled = scaler.transform( X_test_imputed )
```

- This prevents information from the testing partition from influencing preprocessing statistics.

## 🔍 Reduce Features with Principal Component Analysis

- The `PCA` class reduces a numeric feature matrix to a smaller set of principal components while reporting how much variance is retained.

```python
from sklearn.datasets import load_wine
from features import PCA
from scalers import StandardScaler


wine = load_wine( )
X = wine.data
y = wine.target

scaler = StandardScaler( )
X_scaled = scaler.train_transform( X )

selector = PCA( num=3, solver='auto' )
X_components = selector.train_transform( X_scaled )
df_pca_metrics = selector.score( X_scaled )

print( f'Original shape: {X.shape}' )
print( f'Reduced shape: {X_components.shape}' )
print( selector.explained_variance_ratio )
print( df_pca_metrics.to_string( index=False ) )
```

- The fitted selector can project additional observations into the same component space:

```python
X_new = X_scaled[ :5 ]
X_new_components = selector.project( X_new )

print( X_new_components )
```

## 🎯 Train and Evaluate a Classification Model

- The classification wrappers provide reproducible splitting, training, prediction, scoring, and analysis. This example uses multinomial logistic regression to classify the Iris dataset.

```python
from sklearn.datasets import load_iris
from classifications import LogisticRegression
from scalers import StandardScaler


iris = load_iris( )
X = iris.data
y = iris.target

classifier = LogisticRegression(
	C=1.0,
	penalty='l2',
	iters=1000,
	multiclass='multinomial',
	solver='lbfgs',
	random=42
)

X_train, X_test, y_train, y_test = classifier.split_data(
	X,
	y,
	size=0.20,
	random=42
)

scaler = StandardScaler( )
scaler.train( X_train )

X_train_scaled = scaler.transform( X_train )
X_test_scaled = scaler.transform( X_test )

classifier.train( X_train_scaled, y_train )

predictions = classifier.project( X_test_scaled )
df_classification_metrics = classifier.analyze( X_test_scaled, y_test )

print( predictions )
print( df_classification_metrics.to_string( index=False ) )
print( classifier.confusion_matrix_values )
```

- The returned metrics include training score, testing score, misclassifications, precision, accuracy, recall, balanced accuracy, and weighted F-score.

## 📈 Analyze a Regression Model

- The regression classes use the same split, train, project, score, and analyze pattern. The following example uses a random-forest regressor to model the scikit-learn diabetes dataset.

```python
from sklearn.datasets import load_diabetes
from regressions import RandomForest


diabetes = load_diabetes( )
X = diabetes.data
y = diabetes.target

regressor = RandomForest(
	estimators=300,
	criterion='squared_error',
	depth=8,
	jobs=-1,
	rando=42
)

X_train, X_test, y_train, y_test = regressor.split_data(
	X,
	y,
	size=0.20,
	random=42
)

regressor.train( X_train, y_train )

predictions = regressor.project( X_test )
df_regression_metrics = regressor.analyze( X_test, y_test )

print( predictions[ :10 ] )
print( df_regression_metrics.to_string( index=False ) )
```

The analysis includes:

* Training and testing scores
* R-squared
* Mean absolute error
* Mean squared error
* Root mean squared error
* Explained variance
* Median absolute error
* Maximum error

## 🧩 Discover Natural Groups with K-Means

- The `KMeans` class can identify groups in unlabeled numeric data. If reference labels are available, Mathy can also calculate external clustering metrics.

```python
from sklearn.datasets import make_blobs
from clusters import KMeans
from scalers import StandardScaler


X, reference_labels = make_blobs(
	n_samples=300,
	centers=4,
	cluster_std=0.75,
	random_state=42
)

scaler = StandardScaler( )
X_scaled = scaler.train_transform( X )

clusterer = KMeans(
	clusters=4,
	n_init='auto',
	rando=42,
	max_iter=300
)

clusterer.train( X_scaled )

cluster_labels = clusterer.project( X_scaled )
df_cluster_scores = clusterer.score( X_scaled, reference_labels )
cluster_analysis = clusterer.analyze( X_scaled, reference_labels )

print( cluster_labels[ :20 ] )
print( clusterer.centroids_ )
print( df_cluster_scores.to_string( index=False ) )
print( cluster_analysis )
```

The clustering results can include silhouette score, inertia, iterations, cluster count, homogeneity, completeness, mutual information, and V-measure.

Reference labels are optional:

```python
df_intrinsic_scores = clusterer.score( X_scaled )

print( df_intrinsic_scores.to_string( index=False ) )
```

## 🚨 Detecting Anomalies

- The `IsolationForest` class identifies unusual records without requiring a labeled target. Predictions use `1` for an inlier and `-1` for an outlier.

```python
import numpy as np
from outliers import IsolationForest
from scalers import StandardScaler


rng = np.random.default_rng( 42 )

normal_data = rng.normal(
	loc=0.0,
	scale=1.0,
	size=(200, 2)
)

unusual_data = np.array(
	[
		[ 6.0, 6.0 ],
		[ -7.0, 5.0 ],
		[ 8.0, -6.0 ],
		[ -6.0, -7.0 ]
	]
)

X = np.vstack( (normal_data, unusual_data) )

scaler = StandardScaler( )
X_scaled = scaler.train_transform( X )

detector = IsolationForest( contamination=0.02 )
detector.train( X_scaled )

labels = detector.project( X_scaled )
df_anomaly_scores = detector.score( X_scaled )
df_anomaly_summary = detector.analyze( X_scaled )

outlier_rows = np.where( labels == -1 )[ 0 ]

print( f'Detected outlier rows: {outlier_rows}' )
print( df_anomaly_scores.iloc[ outlier_rows ] )
print( df_anomaly_summary.to_string( index=False ) )
```

- The row-level score output contains the predicted class, anomaly score, inlier flag, and outlier flag. The analysis method returns aggregate counts and displays an inlier-versus-outlier chart.

## ⏳ Forecast a Time Series

- `LagBoostingSeries` converts an ordered series into lagged predictors and fits a histogram gradient-boosting regressor. Forecasts are produced recursively by feeding each predicted value into the next lag window.

```python
import numpy as np
from forecasting import LagBoostingSeries


rng = np.random.default_rng( 42 )
periods = np.arange( 96 )

trend = 100.0 + (periods * 1.25)
seasonality = 15.0 * np.sin( 2.0 * np.pi * periods / 12.0 )
noise = rng.normal( loc=0.0, scale=2.0, size=len( periods ) )

series = trend + seasonality + noise

forecaster = LagBoostingSeries(
	lag=12,
	loss='squared_error',
	rate=0.05,
	iters=300,
	leaf=8,
	rando=42
)

forecaster.train( series )

forecast = forecaster.project( n_steps=12 )
training_score = forecaster.score( )
forecast_metrics = forecaster.analyze( )

print( f'Next 12 periods: {forecast}' )
print( f'Training R-squared: {training_score:.4f}' )
print( forecast_metrics )
```

- The forecast analysis reports mean absolute error, mean squared error, root mean squared error, R-squared, explained variance, median absolute error, and maximum error.

## 🔄 Common Mathy Workflow

- Across classification, regression, clustering, outlier detection, and forecasting, Mathy uses a predictable analytical pattern:

```python
model = ModelClass( )
model.train( training_data, training_targets )

predictions = model.project( testing_data )
scores = model.score( testing_data, testing_targets )
analysis = model.analyze( testing_data, testing_targets )
```

- The exact arguments differ for unsupervised and time-series models:

```python
clusterer.train( feature_data )
cluster_labels = clusterer.project( feature_data )
cluster_metrics = clusterer.analyze( feature_data )

forecaster.train( time_series )
future_values = forecaster.project( n_steps=12 )
forecast_metrics = forecaster.analyze( )
```

- This shared interface makes it straightforward to exchange estimators while preserving the surrounding data-preparation and evaluation workflow.

## 🔧 Configuration

| Configuration Item  | Purpose                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `cfg.FAVICON`       | Streamlit page icon.                                                                              |
| `cfg.LOGO`          | Application logo used by `st.logo`.                                                               |
| `cfg.REPO_URL`      | Repository link for the Streamlit logo.                                                           |
| `cfg.DEFAULT_DATA`  | Default Excel dataset loaded from the sidebar.                                                    |
| `cfg.DB_PATH`       | SQLite database path used by Data Management and persistence helpers.                             |
| `cfg.MODE`          | Mode-label mapping rendered by the sidebar radio selector and page headers.                       |
| `cfg.BLUE_DIVIDER`  | Shared divider styling.                                                                           |
| Help-text constants | Mode, scaler, imputer, encoder, transformer, statistics, anomaly, and model-control descriptions. |
| Plot constants      | Color palettes and markers used for styled charts.                                                |

## 🔒 Workflow Notes

![](https://github.com/is-leeroy-jenkins/Mathy/blob/main/resources/mathy-workflows.png)

| Topic                    | Note                                                                                                            |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Working data             | User-selected modeling data is stored in `df_working`.                                                          |
| Processed data           | Feature-engineering operations write to `df_processed`; subsequent operations continue from `df_processed`.     |
| Original data            | `df_original` and `raw_df` preserve the loaded source dataset.                                                  |
| Mode reset               | Classification and Regression mode state is reset when switching into those modes to prevent stale model state. |
| SQL safety               | The SQL console blocks mutating SQL statements and multiple-statement execution.                                |
| Optional XGBoost         | XGBoost is used only when import succeeds.                                                                      |
| Streamlit display safety | The app includes dataframe display fallbacks for serialization-sensitive values.                                |


## 📦 Requirements

The table below reflects the active imports and runtime features used by the current `app.py`. Use
`requirements.txt` as the installation source of truth when version pins are present.

| Requirement           | Package / Import                           | Purpose                                                                                                                  | Used By                                                         |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| Python                | `python>=3.10`                             | Runtime for modern type hints and Streamlit execution.                                                                   | Entire application.                                             |
| Streamlit             | `streamlit`                                | Web UI framework, sidebar controls, data editors, charts, expanders, tabs, metrics, and uploaders.                       | All modes.                                                      |
| Pandas                | `pandas`                                   | Dataframes, spreadsheet loading, SQL query results, profiling, modeling datasets, exports.                               | All data workflows.                                             |
| NumPy                 | `numpy`                                    | Numeric arrays, vector operations, statistics, matrix operations, model inputs.                                          | Statistics, anomaly detection, modeling.                        |
| SciPy                 | `scipy.stats`                              | Normality tests, probability plots, ANOVA, Kruskal-Wallis, chi-square, correlation, distribution utilities.              | Descriptive and inferential statistics, anomaly detection.      |
| Matplotlib            | `matplotlib.pyplot`                        | Static charts, histograms, Q-Q plots, scatter plots, box/violin overlays, PCA and anomaly visuals.                       | Data Profile, statistics, anomaly detection, model diagnostics. |
| Seaborn               | `seaborn`                                  | Statistical plots such as histograms/KDE, heatmaps, boxplots, stripplots, violin plots.                                  | Statistics and anomaly detection.                               |
| Plotly Express        | `plotly.express`                           | Interactive charts over tabular data.                                                                                    | Data Management visualization.                                  |
| Plotly Graph Objects  | `plotly.graph_objects`                     | Safe lower-level Plotly charts that avoid problematic dataframe serialization paths.                                     | Data Management visualization.                                  |
| scikit-learn          | `sklearn`                                  | Train/test split, preprocessing, feature selection, classifiers, clustering, neighbors, SVM, metrics, anomaly models.    | Feature engineering and model modes.                            |
| Statsmodels           | `statsmodels`                              | Statistical modeling and time-series components, including power and ARIMA/SARIMA-style wrappers.                        | Inferential/statistical and forecasting workflows.              |
| XGBoost               | `xgboost`                                  | Optional gradient-boosted classification when installed.                                                                 | Classification Models.                                          |
| SQLite                | `sqlite3`                                  | Local database storage, SQL console, imported tables, prompt/chat/embedding tables.                                      | Data Management and local persistence.                          |
| OpenPyXL              | `openpyxl`                                 | Excel `.xlsx` read support through pandas.                                                                               | Sidebar data loading and Data Management import.                |
| pathlib               | `pathlib`                                  | Filesystem path creation and management.                                                                                 | SQLite store setup.                                             |
| regular expressions   | `re`                                       | Identifier sanitization, SQL safety checks, and text/column validation.                                                  | Data Management and utilities.                                  |
| typing                | `List`, `Dict`, `Optional`, `Tuple`, `Any` | Type annotations and interface clarity.                                                                                  | Application utilities and wrappers.                             |
| Local scalers         | `scalers.py`                               | Standard, Min-Max, Robust, Normal, and MaxAbs scaling wrappers.                                                          | Feature Engineering.                                            |
| Local imputers        | `imputers.py`                              | Mean, nearest-neighbor, iterative, and simple imputation wrappers.                                                       | Feature Engineering.                                            |
| Local encoders        | `encoders.py`                              | One-hot, ordinal, label, target, and polynomial encoders.                                                                | Feature Engineering.                                            |
| Local transformers    | `transformers.py`                          | Binarizers, text vectorizers, TF-IDF, count/hash/dictionary vectorizers, feature hasher, column transformer.             | Feature Engineering.                                            |
| Local clusters        | `clusters.py`                              | KMeans, DBSCAN, Agglomerative, Spectral, OPTICS, MeanShift, AffinityPropagation, Birch wrappers.                         | Clustering Models.                                              |
| Local features        | `features.py`                              | VarianceThreshold, CCA, PCA, SelectBest, SelectPercent, SBS, RFE.                                                        | Feature Engineering and dimensionality reduction.               |
| Local classifications | `classifications.py`                       | Perceptron, logistic, decision tree, support vector, random forest, nearest neighbor, bagging, AdaBoost, gradient boost. | Classification Models.                                          |
| Local regressions     | `regressions.py`                           | Regression model wrappers.                                                                                               | Regression Models.                                              |
| Local forecasting     | `forecasting.py`                           | LaggingSeries, LagBoostingSeries, ARIMA, SARIMA, TimeSeriesSpliter.                                                      | Time-Series Models.                                             |


## 📄 License

- License: [MIT License © 2022–2026](https://github.com/is-leeroy-jenkins/Mathy/blob/main/LICENSE.txt)
- Contact: [Terry Eppler](mailto:terryeppler@gmail.com)



