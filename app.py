'''
	******************************************************************************************
	  Assembly:                mathy
	  Filename:                app.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022
	
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        08-13-2026
	******************************************************************************************
	<copyright file="capp.py" company="Terry D. Eppler">
	
		 mathy app
	
	 Permission is hereby granted, free of charge, to any person obtaining a copy
	 of this software and associated documentation files (the “Software”),
	 to deal in the Software without restriction,
	 including without limitation the rights to use,
	 copy, modify, merge, publish, distribute, sublicense,
	 and/or sell copies of the Software,
	 and to permit persons to whom the Software is furnished to do so,
	 subject to the following conditions:
	
	 The above copyright notice and this permission notice shall be included in all
	 copies or substantial portions of the Software.
	
	 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	 FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
	 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
	 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
	 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	 DEALINGS IN THE SOFTWARE.
	
	 You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov
	
	</copyright>
	<summary>
		app.py
	</summary>
	******************************************************************************************
'''

from __future__ import annotations

import base64
from mathy.boogr import Error, Logger
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy import stats
from typing import List, Dict, Optional, Tuple, Any

# Mathy
import mathy.config as cfg

# sklearn / statsmodels
import numpy as np
import pandas as pd
import re
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, roc_curve, auc, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score)

try:
	from xgboost import XGBClassifier
	
	has_xgb = True
except Exception:
	has_xgb = False

import time
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from statsmodels.stats.power import TTestPower
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
import sklearn.feature_selection as sf
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split as split
from mathy.scalers import (StandardScaler, MinMaxScaler, RobustScaler, NormalScaler, MaxAbsScaler)
from mathy.imputers import (MeanImputer, NearestImputer, IterativeImputer, SimpleImputer)
from mathy.encoders import (OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder,
                      PolynomialFeatures)

from mathy.transformers import (Binarizer, LabelBinarizer, MultiLabelBinarizer, TfidfTransformer,
                          ColumnTransformer, TfidfVectorizer, CountVectorizer, HashVectorizer,
                          DictVectorizer, FeatureHasher)

from mathy.clusters import (KMeans, DBSCAN, Agglomerative, Spectral, OPTICS, MeanShift,
                      AffinityPropagation, Birch)

from mathy.features import (VarianceThreshold, CCA, PCA, SelectBest, SelectPercent, SBS, RFE)
import mathy.classifications as classification_model
import mathy.regressions as regression_model

from mathy.classifications import (Perceptron, LogisticRegression, DecisionTree, SupportVector,
                             RandomForest, NearestNeighbor, BaggingModel, AdaptiveBoost,
                             GradientBoost)

from mathy.encoders import (OneHotEncoder, OrdinalEncoder, TargetEncoder)
from mathy.imputers import (MeanImputer, SimpleImputer, NearestImputer, IterativeImputer)
from mathy.forecasting import (LaggingSeries, LagQuantileSeries, LagBoostingSeries, ARIMA, SARIMA,
                         TimeSeriesSpliter)

# ============================================
# Session State
# ============================================

if 'df_original' not in st.session_state or st.session_state[ 'df_original' ] is None:
	st.session_state[ 'df_original' ] = pd.DataFrame( )

if 'df_profile' not in st.session_state or st.session_state[ 'df_profile' ] is None:
	st.session_state[ 'df_profile' ] = pd.DataFrame( )

if 'df_features' not in st.session_state:
	st.session_state[ 'df_features' ] = pd.DataFrame( )

if 'df_targets' not in st.session_state:
	st.session_state[ 'df_targets' ] = pd.DataFrame( )

if 'df_processed' not in st.session_state or st.session_state[ 'df_processed' ] is None:
	st.session_state[ 'df_processed' ] = pd.DataFrame( )

if 'df_dataset' not in st.session_state or st.session_state[ 'df_dataset' ] is None:
	st.session_state[ 'df_dataset' ] = pd.DataFrame( )

if 'df_working' not in st.session_state or st.session_state[ 'df_working' ] is None:
	st.session_state[ 'df_working' ] = pd.DataFrame( )

if 'df_scores' not in st.session_state:
	st.session_state[ 'df_scores' ] = pd.DataFrame( )

if 'df_model' not in st.session_state or st.session_state[ 'df_model' ] is None:
	st.session_state[ 'df_model' ] = pd.DataFrame( )

if 'numeric_columns' not in st.session_state:
	st.session_state[ 'numeric_columns' ] = [ ]

if 'categorical_columns' not in st.session_state:
	st.session_state[ 'categorical_columns' ] = [ ]

if 'ordinal_columns' not in st.session_state:
	st.session_state[ 'ordinal_columns' ] = [ ]

if 'identifier_columns' not in st.session_state:
	st.session_state[ 'identifier_columns' ] = [ ]

if 'datetime_columns' not in st.session_state:
	st.session_state[ 'datetime_columns' ] = [ ]

if 'boolean_columns' not in st.session_state:
	st.session_state[ 'boolean_columns' ] = [ ]

if 'column_profiles' not in st.session_state:
	st.session_state[ 'column_profiles' ] = { }

if 'column_type_overrides' not in st.session_state:
	st.session_state[ 'column_type_overrides' ] = { }

if 'df_analysis' not in st.session_state:
	st.session_state[ 'df_analysis' ] = pd.DataFrame( )

if 'active_features' not in st.session_state:
	st.session_state[ 'active_features' ] = [ ]

if 'active_targets' not in st.session_state:
	st.session_state[ 'active_targets' ] = [ ]

if 'features' not in st.session_state:
	st.session_state[ 'features' ] = [ ]

if 'targets' not in st.session_state:
	st.session_state[ 'targets' ] = [ ]

if 'X_data' not in st.session_state:
	st.session_state[ 'X_data' ]=None

if 'X_train' not in st.session_state:
	st.session_state[ 'X_train' ]=None

if 'y_train' not in st.session_state:
	st.session_state[ 'y_train' ]=None

if 'X_test' not in st.session_state:
	st.session_state[ 'X_test' ]=None

if 'y_test' not in st.session_state:
	st.session_state[ 'y_test' ]=None

if 'y_prediction' not in st.session_state:
	st.session_state[ 'y_prediction' ]=None

if 'y_series' not in st.session_state:
	st.session_state[ 'y_series' ]=None

if 'target_count' not in st.session_state:
	st.session_state[ 'target_count' ] = 0.0

if 'elapsed_seconds' not in st.session_state:
	st.session_state[ 'elapsed_seconds' ] = 0.0

# ----------- Classfication Members

if 'df_classification' not in st.session_state or st.session_state[ 'df_classification' ] is None:
	st.session_state[ 'df_classification' ] = pd.DataFrame( )

if 'df_classification_scores' not in st.session_state or st.session_state[ 'df_classification_scores' ] is None:
	st.session_state[ 'df_classification_scores' ] = pd.DataFrame( )

# ----------- Regression Members

if 'df_regression' not in st.session_state or st.session_state[ 'df_regression' ] is None:
	st.session_state[ 'df_regression' ] = pd.DataFrame( )

if 'df_regression_scores' not in st.session_state or st.session_state[
	'df_regression_scores' ] is None:
	st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )

# ----------- Clustering Members

if 'df_cluster' not in st.session_state or st.session_state[ 'df_cluster' ] is None:
	st.session_state[ 'df_cluster' ] = pd.DataFrame( )

if 'df_cluster_results' not in st.session_state:
	st.session_state[ 'df_cluster_results' ] = pd.DataFrame( )

if 'df_cluster_counts' not in st.session_state:
	st.session_state[ 'df_cluster_counts' ] = pd.DataFrame( )

if 'df_cluster_metrics' not in st.session_state:
	st.session_state[ 'df_cluster_metrics' ] = pd.DataFrame( )

if 'df_cluster_centroids' not in st.session_state:
	st.session_state[ 'df_cluster_centroids' ] = pd.DataFrame( )

if 'df_cluster_details' not in st.session_state:
	st.session_state[ 'df_cluster_details' ] = pd.DataFrame( )

if 'cluster_plot_features' not in st.session_state:
	st.session_state[ 'cluster_plot_features' ] = [ ]

if 'cluster_signature' not in st.session_state:
	st.session_state[ 'cluster_signature' ]=None

# ----------- Data Management Members

if 'data_upload_signature' not in st.session_state:
	st.session_state[ 'data_upload_signature' ]=None

if 'data_upload_tables' not in st.session_state:
	st.session_state[ 'data_upload_tables' ] = { }

if 'data_upload_sheet_names' not in st.session_state:
	st.session_state[ 'data_upload_sheet_names' ] = { }

if 'data_upload_filename' not in st.session_state:
	st.session_state[ 'data_upload_filename' ]=None

if 'active_source_kind' not in st.session_state:
	st.session_state[ 'active_source_kind' ] = ''

if 'active_database_table' not in st.session_state:
	st.session_state[ 'active_database_table' ] = ''

if 'active_declared_schema' not in st.session_state:
	st.session_state[ 'active_declared_schema' ] = { }

if 'custom_upload_signature' not in st.session_state:
	st.session_state[ 'custom_upload_signature' ]=None

if 'custom_upload_tables' not in st.session_state:
	st.session_state[ 'custom_upload_tables' ] = { }

if 'custom_upload_filename' not in st.session_state:
	st.session_state[ 'custom_upload_filename' ]=None

if 'database_operation_result' not in st.session_state:
	st.session_state[ 'database_operation_result' ]=None

if 'crud_next_table' not in st.session_state:
	st.session_state[ 'crud_next_table' ] = ''

# ============================================
# Session State
# ============================================

def throw_if( name: str, value: object ) -> None:
	"""Validate a required clustering argument.
	
	Purpose:
	    Enforces the presence of required clustering inputs before estimator execution. The
	    validation accepts populated NumPy arrays and standard Python containers while
	    rejecting null values and empty collections that would otherwise cause downstream
	    operations to fail or produce undefined clustering results.
	
	Args:
	    name (str): Argument name used in the validation error message.
	    value (object): Argument value checked for a null or empty state.
	
	Returns:
	    None: This function performs its work through side effects and does not return a
	          value.
	
	Raises:
	    ValueError: Raised when `value` is None or empty."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, np.ndarray ) and value.size == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (str, list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def init_state( ) -> None:
	"""Initialize core application session state.

	Purpose:
	    Creates the baseline dataset, column-classification, feature-selection, target-selection,
	    and pipeline-log keys required by downstream application workflows. Existing values are
	    preserved so Streamlit reruns do not overwrite active user state.

	Returns:
	    None: This function initializes missing ``st.session_state`` keys in place.
	"""
	defaults = { 'df_dataset': None, 'df_original': None, 'df_processed': None,
		'numeric_columns': [ ], 'categorical_columns': [ ], 'features': [ ], 'targets': [ ],
		'pipeline_log': [ ] }
	for k, v in defaults.items( ):
		if k not in st.session_state:
			st.session_state[ k ] = v

init_state( )

def has_loaded_dataset( df_frame: object ) -> bool:
	"""Determine whether an object contains a usable loaded dataset.

	Purpose:
	    Validates that the supplied object is a non-empty pandas dataframe containing at least one
	    column before it is used by data-loading, preprocessing, visualization, or modeling
	    workflows.

	Args:
	    df_frame (object): Candidate object evaluated as the currently loaded dataset.

	Returns:
	    bool: ``True`` when the object is a non-empty dataframe with at least one column; otherwise
	    ``False``.
	"""
	return (isinstance( df_frame, pd.DataFrame ) and not df_frame.empty and len(
		df_frame.columns ) > 0)

def get_loaded_dataset( ) -> pd.DataFrame | None:
	"""Return a defensive copy of the currently loaded dataset.

	Purpose:
	    Retrieves ``df_dataset`` from Streamlit session state and returns an isolated dataframe copy
	    when the stored value satisfies the loaded-dataset validation contract.

	Returns:
	    pd.DataFrame | None: Copy of the loaded dataset, or ``None`` when no valid dataset is stored.
	"""
	df_frame = st.session_state.get( 'df_dataset', None )
	if not has_loaded_dataset( df_frame ):
		return None
	
	return df_frame.copy( )

def store_loaded_dataset( df_dataset: pd.DataFrame, df_original: pd.DataFrame=None ) -> None:
	"""Persist a validated dataset and its original baseline in session state.

	Purpose:
	    Stores independent copies of the loaded dataset under the raw, original, and active dataset
	    session-state keys while preserving an explicitly supplied original dataframe when
	    available.

	Args:
	    df_dataset (pd.DataFrame): Loaded dataframe persisted as the active dataset.
	    df_original (pd.DataFrame): Optional original dataframe retained as the baseline dataset.

	Returns:
	    None: This function updates Streamlit session state and does not return a value.
	"""
	if not has_loaded_dataset( df_dataset ):
		return
	
	df_source = df_dataset.copy( )
	df_base = df_original.copy( ) if isinstance( df_original, pd.DataFrame ) else df_source.copy( )
	
	st.session_state[ 'raw_df' ] = df_source.copy( )
	st.session_state[ 'df_original' ] = df_base.copy( )
	st.session_state[ 'df_dataset' ] = df_source.copy( )
	synchronize_dataset_columns( df_source )

def synchronize_dataset_columns( df_dataset: pd.DataFrame ) -> tuple[ List[ str ], List[ str ] ]:
	"""Synchronize analytical column classifications with the active dataset.

	Purpose:
	    Recomputes numeric and categorical analytical roles from the current dataframe and stores
	    the resulting column collections in session state. This keeps analytical modes aligned with
	    source changes and column-management operations instead of relying on classifications left
	    behind by a previously rendered mode.

	Args:
	    df_dataset (pd.DataFrame): Active dataframe whose columns are classified.

	Returns:
	    tuple[List[str], List[str]]: Numeric column names followed by categorical column names.
	"""
	if not has_loaded_dataset( df_dataset ):
		st.session_state[ 'numeric_columns' ] = [ ]
		st.session_state[ 'categorical_columns' ] = [ ]
		st.session_state[ 'ordinal_columns' ] = [ ]
		st.session_state[ 'identifier_columns' ] = [ ]
		st.session_state[ 'datetime_columns' ] = [ ]
		st.session_state[ 'boolean_columns' ] = [ ]
		st.session_state[ 'column_profiles' ] = { }
		st.session_state[ 'df_analysis' ] = pd.DataFrame( )
		return [ ], [ ]

	declared_schema = st.session_state.get( 'active_declared_schema', { } ) if (
		st.session_state.get( 'active_source_kind', '' ) == 'database') else { }
	profiles = profile_dataframe_schema( df_dataset, declared_schema )
	schema = { column: str( profile[ 'analytical_role' ] ) for column, profile in
		profiles.items( ) }
	numeric_columns = [ column for column, role in schema.items( ) if role == 'numeric' ]
	ordinal_columns = [ column for column, role in schema.items( ) if role == 'ordinal' ]
	identifier_columns = [ column for column, role in schema.items( ) if role == 'identifier' ]
	datetime_columns = [ column for column, role in schema.items( ) if role == 'datetime' ]
	boolean_columns = [ column for column, profile in profiles.items( ) if
		profile[ 'inferred_dtype' ] == 'boolean' ]
	categorical_columns = [ column for column, role in schema.items( ) if role in (
		'categorical', 'ordinal' ) ]
	st.session_state[ 'numeric_columns' ] = numeric_columns.copy( )
	st.session_state[ 'categorical_columns' ] = categorical_columns.copy( )
	st.session_state[ 'ordinal_columns' ] = ordinal_columns.copy( )
	st.session_state[ 'identifier_columns' ] = identifier_columns.copy( )
	st.session_state[ 'datetime_columns' ] = datetime_columns.copy( )
	st.session_state[ 'boolean_columns' ] = boolean_columns.copy( )
	st.session_state[ 'column_profiles' ] = profiles.copy( )
	st.session_state[ 'column_schema' ] = schema.copy( )
	st.session_state[ 'df_analysis' ] = create_typed_analysis_dataframe( df_dataset, profiles )
	return numeric_columns, categorical_columns

def reset_inferential_selection_state( df_dataset: pd.DataFrame,
	numeric_columns: List[ str ], categorical_columns: List[ str ] ) -> None:
	"""Reset inferential selections that do not belong to the active dataset schema.

	Purpose:
	    Compares the active dataframe schema with the schema previously used by Inferential
	    Statistics and removes dataset-dependent widget state before the corresponding controls are
	    instantiated. Selections are also validated individually so invalid legacy values cannot be
	    used to index the current dataframe.

	Args:
	    df_dataset (pd.DataFrame): Active dataframe used by Inferential Statistics.
	    numeric_columns (List[str]): Current numeric columns available to inferential controls.
	    categorical_columns (List[str]): Current categorical columns available to inferential
	        controls.

	Returns:
	    None: This function updates Streamlit session state in place.
	"""
	throw_if( 'df_dataset', df_dataset )
	schema_signature = (tuple( df_dataset.columns.tolist( ) ),
		tuple( str( dtype ) for dtype in df_dataset.dtypes.tolist( ) ))
	selection_keys = [ 'infer_summary_y', 'infer_summary_x', 'infer_summary_group',
		'infer_summary_cat1', 'infer_summary_cat2' ]
	previous_signature = st.session_state.get( 'inferential_schema_signature', None )

	if previous_signature != schema_signature:
		clear_keys( selection_keys )
	else:
		numeric_options = set( numeric_columns )
		categorical_options = set( categorical_columns )
		if st.session_state.get( 'infer_summary_y', None ) not in numeric_options:
			clear_keys( [ 'infer_summary_y' ] )
		if st.session_state.get( 'infer_summary_x', '<None>' ) not in numeric_options | {
				'<None>' }:
			clear_keys( [ 'infer_summary_x' ] )
		if st.session_state.get( 'infer_summary_group', '<None>' ) not in categorical_options | {
				'<None>' }:
			clear_keys( [ 'infer_summary_group' ] )
		if st.session_state.get( 'infer_summary_cat1', None ) not in categorical_options:
			clear_keys( [ 'infer_summary_cat1' ] )
		if st.session_state.get( 'infer_summary_cat2', None ) not in categorical_options:
			clear_keys( [ 'infer_summary_cat2' ] )
		if st.session_state.get( 'infer_summary_cat1', None ) == st.session_state.get(
				'infer_summary_cat2', None ):
			clear_keys( [ 'infer_summary_cat2' ] )

	st.session_state[ 'inferential_schema_signature' ] = schema_signature

def clear_keys( keys: List[ str ] ) -> None:
	"""Remove specified keys from Streamlit session state.

	Purpose:
	    Deletes each supplied session-state key when present so mode-specific workflows can reset
	    their owned state without raising errors for keys that have not been initialized.

	Args:
	    keys (List[str]): Session-state keys to remove.

	Returns:
	    None: This function mutates ``st.session_state`` in place.
	"""
	for key in keys:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_session_keys( keys: List[ str ] ) -> None:
	"""Reset Streamlit widget state before controls are recreated.

	Purpose:
	    Removes an explicit collection of widget-owned session-state keys from a Streamlit callback
	    so reset operations occur before the affected controls are instantiated on the subsequent
	    application rerun.

	Args:
	    keys (List[str]): Widget session-state keys removed by the reset callback.

	Returns:
	    None: This function delegates session-state removal to ``clear_keys``.
	"""
	clear_keys( keys )

def reset_session_prefixes( prefixes: Tuple[ str, ... ] ) -> None:
	"""Reset Streamlit widget state matching one or more key prefixes.

	Purpose:
	    Finds dynamic CRUD widget keys owned by the supplied prefixes and clears them from a
	    Streamlit callback before the corresponding controls are recreated.

	Args:
	    prefixes (Tuple[str, ...]): Session-state key prefixes identifying controls to reset.

	Returns:
	    None: This function removes matching session-state keys in place.
	"""
	throw_if( 'prefixes', prefixes )
	keys = [ key for key in st.session_state.keys( ) if str( key ).startswith( prefixes ) ]
	clear_keys( keys )

def reset_classification_mode_state( ) -> None:
	"""Reset state owned by the classification workflow.

	Purpose:
	    Clears classification inputs, outputs, selected columns, train-test partitions, predictions,
	    model references, metrics, and elapsed-time values, then recreates the required keys with
	    their established empty defaults.

	Returns:
	    None: This function resets classification-related values in Streamlit session state.
	"""
	classification_keys = [ 'df_working', 'df_processed', 'df_model', 'df_scores',
		'df_predictions', 'df_classification', 'df_classification_scores', 'df_features',
		'df_targets', 'features', 'targets', 'selected_all', 'active_features', 'active_targets',
		'X_data', 'X_train', 'X_test', 'y_train', 'y_test', 'y_series', 'y_prediction', 'model',
		'elapsed_seconds', 'target_count' ]
	
	clear_keys( classification_keys )
	st.session_state[ 'df_working' ] = pd.DataFrame( )
	st.session_state[ 'df_processed' ] = pd.DataFrame( )
	st.session_state[ 'df_model' ] = pd.DataFrame( )
	st.session_state[ 'df_scores' ] = pd.DataFrame( )
	st.session_state[ 'df_predictions' ] = pd.DataFrame( )
	st.session_state[ 'df_classification' ] = pd.DataFrame( )
	st.session_state[ 'df_classification_scores' ] = pd.DataFrame( )
	st.session_state[ 'df_features' ] = pd.DataFrame( )
	st.session_state[ 'df_targets' ] = pd.DataFrame( )
	st.session_state[ 'features' ] = [ ]
	st.session_state[ 'targets' ] = [ ]
	st.session_state[ 'selected_all' ] = [ ]
	st.session_state[ 'active_features' ] = [ ]
	st.session_state[ 'active_targets' ] = [ ]
	st.session_state[ 'X_data' ]=None
	st.session_state[ 'X_train' ]=None
	st.session_state[ 'X_test' ]=None
	st.session_state[ 'y_train' ]=None
	st.session_state[ 'y_test' ]=None
	st.session_state[ 'y_series' ]=None
	st.session_state[ 'y_prediction' ]=None
	st.session_state[ 'model' ]=None
	st.session_state[ 'elapsed_seconds' ] = 0.0
	st.session_state[ 'target_count' ] = 0.0

def reset_regression_mode_state( ) -> None:
	"""Reset state owned by the regression workflow.

	Purpose:
	    Clears regression inputs, outputs, selected columns, train-test partitions, predictions,
	    model references, metrics, and elapsed-time values, then recreates the required keys with
	    their established empty defaults.

	Returns:
	    None: This function resets regression-related values in Streamlit session state.
	"""
	regression_keys = [ 'df_working', 'df_processed', 'df_model', 'df_scores', 'df_predictions',
		'df_regression', 'df_regression_scores', 'df_features', 'df_targets', 'features',
		'targets', 'selected_all', 'active_features', 'active_targets', 'X_data', 'X_train', 'X_test',
		'y_train', 'y_test', 'y_series', 'y_prediction', 'model', 'elapsed_seconds',
		'target_count' ]
	
	clear_keys( regression_keys )
	st.session_state[ 'df_working' ] = pd.DataFrame( )
	st.session_state[ 'df_processed' ] = pd.DataFrame( )
	st.session_state[ 'df_model' ] = pd.DataFrame( )
	st.session_state[ 'df_scores' ] = pd.DataFrame( )
	st.session_state[ 'df_predictions' ] = pd.DataFrame( )
	st.session_state[ 'df_regression' ] = pd.DataFrame( )
	st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
	st.session_state[ 'df_features' ] = pd.DataFrame( )
	st.session_state[ 'df_targets' ] = pd.DataFrame( )
	st.session_state[ 'features' ] = [ ]
	st.session_state[ 'targets' ] = [ ]
	st.session_state[ 'selected_all' ] = [ ]
	st.session_state[ 'active_features' ] = [ ]
	st.session_state[ 'active_targets' ] = [ ]
	st.session_state[ 'X_data' ]=None
	st.session_state[ 'X_train' ]=None
	st.session_state[ 'X_test' ]=None
	st.session_state[ 'y_train' ]=None
	st.session_state[ 'y_test' ]=None
	st.session_state[ 'y_series' ]=None
	st.session_state[ 'y_prediction' ]=None
	st.session_state[ 'model' ]=None
	st.session_state[ 'elapsed_seconds' ] = 0.0
	st.session_state[ 'target_count' ] = 0.0

# ============================================
# Utilities
# ============================================
def inferential_plot( title: str, subtitle: str | None=None, figsize: tuple[ int, int ]=(6, 4),
	grid: bool=True, ref_line: float | None=None, legend: bool=True ):
	"""Create a standardized inferential-analysis figure.

	Purpose:
	    Creates a Plotly figure configured with consistent titles, optional grid styling, an
	    optional horizontal reference line, and configurable legend behavior for inferential-
	    statistics visualizations.

	Args:
	    title (str): Primary title displayed above the plot.
	    subtitle (str | None): Optional contextual subtitle displayed beneath the primary title.
	    figsize (tuple[int, int]): Width and height of the figure in inches.
	    grid (bool): Flag indicating whether the axes display a background grid.
	    ref_line (float | None): Optional y-axis value used to draw a horizontal reference line.
	    legend (bool): Flag indicating whether the axes display a legend.

	Returns:
	    go.Figure: Configured Plotly figure used to render the inferential plot.
	"""
	figure = go.Figure( )
	figure.update_layout( title=title, width=int( figsize[ 0 ] * 100 ),
		height=int( figsize[ 1 ] * 100 ), showlegend=legend )
	figure.update_xaxes( showgrid=grid )
	figure.update_yaxes( showgrid=grid )
	if subtitle:
		figure.add_annotation( x=0.5, y=1.08, xref='paper', yref='paper', text=subtitle,
			showarrow=False, font={ 'size': 12, 'color': '#94A3B8' } )
	
	if ref_line is not None:
		figure.add_hline( y=ref_line, line_dash='dash', line_color='#94A3B8' )
	
	return figure

def blue_divider( ) -> None:
	"""Render the configured blue section divider.

	Purpose:
	    Inserts the application-standard HTML divider into the Streamlit interface using the
	    divider markup defined in the configuration module.

	Returns:
	    None: This function renders Streamlit markup and does not return a value.
	"""
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

def log_step( msg: str ) -> None:
	"""Append a processing message to the pipeline log.

	Purpose:
	    Records a workflow status message in the session-state pipeline log for later display,
	    diagnostics, or processing-history review.

	Args:
	    msg (str): Message appended to the active pipeline log.

	Returns:
	    None: This function updates Streamlit session state in place.
	"""
	st.session_state.pipeline_log.append( msg )

def detect_column_types( df: pd.DataFrame ) -> tuple[ List[ str ], List[ str ] ]:
	"""Classify dataframe columns as numeric or categorical.

	Purpose:
	    Examines column names and pandas data types to divide dataframe columns into numeric and
	    categorical collections. Name-based hints take precedence over inferred pandas data types
	    so budgeting identifiers and fiscal-year fields retain their intended analytical roles.

	Args:
	    df (pd.DataFrame): Dataframe whose columns are classified.

	Returns:
	    tuple[List[str], List[str]]: Numeric column names followed by categorical column names.
	"""
	profiles = profile_dataframe_schema( df )
	numeric = [ column for column, profile in profiles.items( ) if
		profile[ 'analytical_role' ] == 'numeric' ]
	categorical = [ column for column, profile in profiles.items( ) if
		profile[ 'analytical_role' ] in ('categorical', 'ordinal') ]
	return numeric, categorical

def styled_scatter( figure: go.Figure, x: np.ndarray, y: np.ndarray, series_index: int=0,
	label: Optional[ str ]=None, size: int=30, ) -> None:
	"""Render a consistently styled scatter series.

	Purpose:
	    Draws a scatter series on the supplied Plotly figure using the configured application
	    color and marker palettes, visible point boundaries, controlled transparency, and grid
	    styling.

	Args:
	    figure (go.Figure): Plotly figure receiving the scatter series.
	    x (np.ndarray): Horizontal coordinates for the plotted observations.
	    y (np.ndarray): Vertical coordinates aligned to ``x``.
	    series_index (int): Palette index used to select the series color and marker.
	    label (Optional[str]): Optional label associated with the plotted series.
	    size (int): Marker size applied to each plotted observation.

	Returns:
	    None: This function modifies the supplied figure in place.
	"""
	color = cfg.PALETTE[ series_index % len( cfg.PALETTE ) ]
	figure.add_trace( go.Scatter( x=x, y=y, mode='markers', name=label,
		marker={ 'color': color, 'size': max( 4, int( np.sqrt( size ) ) ), 'opacity': 0.9,
			'line': { 'color': '#020617', 'width': 0.6 } }, showlegend=label is not None ) )
	figure.update_xaxes( showgrid=True )
	figure.update_yaxes( showgrid=True )

def auto_float_format( series: pd.Series, max_decimals: int=4 ) -> str:
	"""Select a numeric display format from the magnitude of a series.

	Purpose:
	    Converts the supplied series to numeric values, evaluates its ninety-fifth-percentile
	    absolute magnitude, and returns a comma-separated floating-point format with a bounded
	    number of decimal places.

	Args:
	    series (pd.Series): Series used to determine the numeric display scale.
	    max_decimals (int): Maximum number of decimal places permitted in the returned format.

	Returns:
	    str: Python numeric format string appropriate for the observed data magnitude.
	"""
	s = pd.to_numeric( series, errors='coerce' )
	s = s.replace( [ np.inf, -np.inf ], np.nan ).dropna( )
	if s.empty:
		return '{:,.2f}'
	
	mag = float( np.nanpercentile( np.abs( s.values ), 95 ) )
	
	if mag >= 1e9:
		decimals = 0
	elif mag >= 1e6:
		decimals = 1
	elif mag >= 1e3:
		decimals = 2
	elif mag >= 1:
		decimals = 3
	else:
		decimals = 4
	
	decimals = min( decimals, max_decimals )
	return f"{{:,.{decimals}f}}"

def clean_numeric( df: pd.DataFrame ) -> pd.DataFrame:
	"""Normalize a dataframe for numeric analysis.

	Purpose:
	    Replaces infinite values with missing values, coerces every column to numeric form, removes
	    columns containing only missing values, and removes columns without more than one distinct
	    non-null value.

	Args:
	    df (pd.DataFrame): Dataframe prepared for numeric analysis.

	Returns:
	    pd.DataFrame: Numeric dataframe containing only populated, non-constant columns.
	"""
	out = df.replace( [ np.inf, -np.inf ], np.nan )
	for c in out.columns:
		out[ c ] = pd.to_numeric( out[ c ], errors="coerce" )
	out = out.dropna( axis=1, how="all" )
	out = out.loc[ :, out.nunique( dropna=True ) > 1 ]
	return out

def analysis_fillna_mean( df: pd.DataFrame ) -> pd.DataFrame:
	"""Fill missing floating-point and complex values with column means.

	Purpose:
	    Applies mean imputation to columns whose NumPy dtype kind represents floating-point or
	    complex numeric data while leaving all other columns unchanged.

	Args:
	    df (pd.DataFrame): Dataframe containing values prepared for analytical processing.

	Returns:
	    pd.DataFrame: Dataframe with eligible numeric missing values replaced by column means.
	"""
	return df.apply( lambda c: c.fillna( c.mean( ) ) if c.dtype.kind in "fc" else c )

def default_pick( items: List[ str ], k: int=2 ) -> List[ str ]:
	"""Return the default leading items from a sequence.

	Purpose:
	    Selects up to the requested number of leading strings for use as default Streamlit control
	    selections while safely returning an empty list for an empty input collection.

	Args:
	    items (List[str]): Ordered string values available for default selection.
	    k (int): Maximum number of leading values to return.

	Returns:
	    List[str]: Selected leading values, limited by the input length and requested count.
	"""
	return items[ : min( k, len( items ) ) ] if items else [ ]

def style_subheaders( ) -> None:
	"""Apply application styling to Streamlit subheaders.

	Purpose:
	    Injects CSS that applies the configured blue foreground color to selected heading levels in
	    standard Streamlit markdown containers and chat-message markdown containers.

	Returns:
	    None: This function renders CSS markup in the Streamlit interface.
	"""
	st.markdown( """
		<style>
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stMarkdownContainer"] h4,
		div[data-testid="stMarkdownContainer"] h6,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h4 {
			color: rgb(053, 149, 252) !important;
		}
		</style>
		""", unsafe_allow_html=True, )

# ----------- Data Plumbing Utilities

def get_numeric_columns( df_frame: pd.DataFrame ) -> list[ str ]:
	"""Return numeric columns from a dataframe.

	Purpose:
	    Identifies dataframe columns whose pandas data types are recognized as numeric for use by
	    preprocessing, feature engineering, visualization, and model-selection workflows.

	Args:
	    df_frame (pd.DataFrame): Dataframe whose columns are evaluated.

	Returns:
	    list[str]: Names of columns containing numeric data.
	"""
	profiles = profile_dataframe_schema( df_frame )
	return [ column for column, profile in profiles.items( ) if
		profile[ 'analytical_role' ] == 'numeric' ]

def get_categorical_columns( df_frame: pd.DataFrame ) -> list[ str ]:
	"""Return non-numeric columns from a dataframe.

	Purpose:
	    Identifies dataframe columns whose pandas data types are not recognized as numeric for use
	    by categorical preprocessing, encoding, selection, and visualization workflows.

	Args:
	    df_frame (pd.DataFrame): Dataframe whose columns are evaluated.

	Returns:
	    list[str]: Names of columns containing non-numeric data.
	"""
	profiles = profile_dataframe_schema( df_frame )
	return [ column for column, profile in profiles.items( ) if
		profile[ 'analytical_role' ] in ('categorical', 'ordinal') ]

def get_working_frame( ) -> pd.DataFrame:
	"""Return the active dataframe used by Data Plumbing workflows.

	Purpose:
	    Retrieves the processed dataframe from Streamlit session state when it contains data.
	    Otherwise, returns an independent copy of the original loaded dataframe as the current
	    working frame.

	Returns:
	    pd.DataFrame: Copy of the active processed dataframe or original dataframe.
	"""
	df_working = st.session_state.get( 'df_processed' )
	if df_working is None or df_working.empty:
		return st.session_state[ 'df_original' ].copy( )
	return df_working.copy( )

def get_feature_columns( df_frame: pd.DataFrame ) -> list[ str ]:
	"""Return active feature columns that remain available.

	Purpose:
	    Filters the feature-column names stored in Streamlit session state so only columns that
	    still exist in the supplied dataframe are returned after preprocessing or transformation.

	Args:
	    df_frame (pd.DataFrame): Dataframe used to validate active feature-column names.

	Returns:
	    list[str]: Active feature columns that exist in ``df_frame``.
	"""
	return [ c for c in st.session_state.get( 'features', [ ] ) if c in df_frame.columns ]

def get_target_columns( df_frame: pd.DataFrame ) -> list[ str ]:
	"""Return active target columns that remain available.

	Purpose:
	    Filters the target-column names stored in Streamlit session state so only columns that
	    still exist in the supplied dataframe are returned after preprocessing or transformation.

	Args:
	    df_frame (pd.DataFrame): Dataframe used to validate active target-column names.

	Returns:
	    list[str]: Active target columns that exist in ``df_frame``.
	"""
	return [ c for c in st.session_state.get( 'targets', [ ] ) if c in df_frame.columns ]

def commit_frame( df_frame: pd.DataFrame ) -> None:
	"""Commit active feature and target frames to session state.

	Purpose:
	    Rebuilds the feature and target dataframes from the supplied working dataframe using the
	    currently selected feature and target columns. Empty dataframes with the working index are
	    stored when no valid selections remain.

	Args:
	    df_frame (pd.DataFrame): Updated working dataframe containing processed columns.

	Returns:
	    None: This function updates ``df_features`` and ``df_targets`` in Streamlit session state.
	"""
	feature_columns = get_feature_columns( df_frame )
	target_columns = get_target_columns( df_frame )
	
	if feature_columns:
		st.session_state[ 'df_features' ] = df_frame[ feature_columns ].copy( )
	else:
		st.session_state[ 'df_features' ] = pd.DataFrame( index=df_frame.index )
	
	if target_columns:
		st.session_state[ 'df_targets' ] = df_frame[ target_columns ].copy( )
	else:
		st.session_state[ 'df_targets' ] = pd.DataFrame( index=df_frame.index )

def working_to_original( ) -> None:
	"""Reset the working dataframe to the original loaded dataset.

	Purpose:
	    Copies the original dataframe into ``df_working`` and refreshes the active feature and target
	    dataframes so Data Plumbing operations restart from the unprocessed dataset baseline.

	Returns:
	    None: This function updates Data Plumbing values in Streamlit session state.
	"""
	df_reset = st.session_state[ 'df_original' ].copy( )
	st.session_state[ 'df_working' ] = df_reset.copy( )
	commit_frame( df_reset )

def processed_to_working( ) -> None:
	"""Promote the working dataframe to the processed-data state.

	Purpose:
	    Copies the current working dataframe into ``df_processed`` and refreshes the active feature
	    and target dataframes from that committed processed representation.

	Returns:
	    None: This function updates Data Plumbing values in Streamlit session state.
	"""
	df_reset = st.session_state[ 'df_working' ].copy( )
	st.session_state[ 'df_processed' ] = df_reset.copy( )
	commit_frame( df_reset )

def normalize_result_frame( result: object, index: pd.Index, prefix: str,
	columns: list[ str ] ) -> pd.DataFrame:
	"""Normalize transformation output into a dataframe.

	Purpose:
	    Converts dataframe, tuple, sparse-matrix, one-dimensional array, and multidimensional array
	    transformation results into a consistently indexed pandas dataframe with supplied or
	    generated column names.

	Args:
	    result (object): Transformation output converted into dataframe form.
	    index (pd.Index): Index assigned to the normalized dataframe.
	    prefix (str): Prefix used when generated output-column names are required.
	    columns (list[str]): Preferred output-column names when their count matches the result.

	Returns:
	    pd.DataFrame: Normalized transformation result aligned to ``index``.
	"""
	if isinstance( result, pd.DataFrame ):
		df_result = result.copy( )
		df_result.index = index
		return df_result
	
	if isinstance( result, tuple ):
		parts = [ ]
		for i, item in enumerate( result ):
			df_part = normalize_result_frame( item, index=index, prefix=f'{prefix}_{i + 1}',
				columns=None )
			parts.append( df_part )
		return pd.concat( parts, axis=1 )
	
	if hasattr( result, 'toarray' ):
		result = result.toarray( )
	
	arr = np.asarray( result )
	if arr.ndim == 1:
		col_name = columns[ 0 ] if columns and len( columns ) == 1 else prefix
		return pd.DataFrame( arr, index=index, columns=[ col_name ] )
	
	if columns and len( columns ) == arr.shape[ 1 ]:
		return pd.DataFrame( arr, index=index, columns=columns )
	
	generated = [ f'{prefix}_{i + 1}' for i in range( arr.shape[ 1 ] ) ]
	return pd.DataFrame( arr, index=index, columns=generated )

def replace_columns( df_frame: pd.DataFrame, column_names: list[ str ], result: object,
	prefix: str ) -> pd.DataFrame:
	"""Replace selected dataframe columns with transformed output.

	Purpose:
	    Normalizes transformation output, removes the source columns from the input dataframe, and
	    appends the transformed columns while preserving the original row index.

	Args:
	    df_frame (pd.DataFrame): Input dataframe containing the source columns.
	    column_names (list[str]): Source columns removed from the dataframe.
	    result (object): Transformation result inserted into the dataframe.
	    prefix (str): Prefix used for generated transformed-column names.

	Returns:
	    pd.DataFrame: Updated dataframe containing the unmodified columns and transformed output.
	"""
	df_result = normalize_result_frame( result=result, index=df_frame.index, prefix=prefix,
		columns=column_names )
	
	df_updated = df_frame.drop( columns=column_names, errors='ignore' )
	df_updated = pd.concat( [ df_updated, df_result ], axis=1 )
	return df_updated

def apply_text_vectorizer( df_frame: pd.DataFrame, column_names: list[ str ], vectorizer: object,
	prefix: str ) -> pd.DataFrame:
	"""Apply a text vectorizer to selected dataframe columns.

	Purpose:
	    Combines selected text columns row by row, applies the supplied vectorizer's training and
	    transformation operation, removes the original text columns, and appends the normalized
	    vectorized features.

	Args:
	    df_frame (pd.DataFrame): Input dataframe containing text columns.
	    column_names (list[str]): Text columns combined and vectorized.
	    vectorizer (object): Wrapper exposing a ``train_transform`` method.
	    prefix (str): Prefix used for generated vectorized-column names.

	Returns:
	    pd.DataFrame: Updated dataframe containing vectorized features in place of source text
	    columns.
	"""
	text = df_frame[ column_names ].fillna( '' ).astype( str ).agg( ' '.join, axis=1 ).tolist( )
	
	result = vectorizer.train_transform( text )
	df_result = normalize_result_frame( result=result, index=df_frame.index, prefix=prefix,
		columns=None )
	
	df_updated = df_frame.drop( columns=column_names, errors='ignore' )
	df_updated = pd.concat( [ df_updated, df_result ], axis=1 )
	return df_updated

def apply_dict_transform( df_frame: pd.DataFrame, column_names: list[ str ], transformer: object,
	prefix: str ) -> pd.DataFrame:
	"""Apply a dictionary-based transformer to selected columns.

	Purpose:
	    Converts selected dataframe columns into row-oriented dictionaries, applies the supplied
	    transformer's training and transformation operation, removes the source columns, and appends
	    the normalized transformed features.

	Args:
	    df_frame (pd.DataFrame): Input dataframe containing columns converted to dictionaries.
	    column_names (list[str]): Columns included in each row dictionary.
	    transformer (object): Wrapper exposing a ``train_transform`` method.
	    prefix (str): Prefix used for generated transformed-column names.

	Returns:
	    pd.DataFrame: Updated dataframe containing dictionary-derived transformed features.
	"""
	records = df_frame[ column_names ].fillna( '' ).to_dict( orient='records' )
	result = transformer.train_transform( records )
	df_result = normalize_result_frame( result=result, index=df_frame.index, prefix=prefix,
		columns=None )
	
	df_updated = df_frame.drop( columns=column_names, errors='ignore' )
	df_updated = pd.concat( [ df_updated, df_result ], axis=1 )
	return df_updated

def parse_multilabel_series( series: pd.Series, delimiter: str ) -> np.ndarray:
	"""Parse delimited text values into multilabel collections.

	Purpose:
	    Replaces missing values with empty strings, separates each row by the supplied delimiter,
	    removes surrounding whitespace and empty labels, and returns the parsed label collections
	    as a NumPy array.

	Args:
	    series (pd.Series): Series containing delimited label values.
	    delimiter (str): Text delimiter separating labels within each row.

	Returns:
	    np.ndarray: Array whose elements contain the parsed labels for each source row.
	"""
	values = series.fillna( '' ).astype( str ).apply(
		lambda s: [ item.strip( ) for item in s.split( delimiter ) if item.strip( ) ] )
	return values.to_numpy( )

def score_function_from_name( name: str ) -> object:
	"""Resolve a feature-selection score function by display name.

	Purpose:
	    Maps a supported user-interface score-function name to its corresponding sklearn feature-
	    selection callable for univariate feature ranking and selection workflows.

	Args:
	    name (str): Supported score-function name.

	Returns:
	    object: Sklearn feature-selection scoring callable associated with ``name``.

	Raises:
	    KeyError: Raised when ``name`` is not present in the supported score-function mapping.
	"""
	mapper = { 'chi2': sf.chi2, 'f_classif': sf.f_classif, 'f_regression': sf.f_regression,
		'mutual_info_classif': sf.mutual_info_classif,
		'mutual_info_regression': sf.mutual_info_regression }
	return mapper[ name ]

# ----------  Database Utilities ----------

def initialize_database( ) -> None:
	"""Initialize the application SQLite database.

	Purpose:
	    Creates the configured SQLite storage directory, ensures the chat-history, embeddings, and
	    prompts tables exist, verifies that the prompts table contains the required caption column,
	    and commits all schema changes.

	Returns:
	    None: This function creates or updates SQLite schema objects and does not return a value.
	"""
	Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS chat_history
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          role
                          TEXT,
                          content
                          TEXT
                      )
		              """ )
		
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS embeddings
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          chunk
                          TEXT,
                          vector
                          BLOB
                      )
		              """ )
		
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS Prompts
                      (
                          PromptsId
                          INTEGER
                          NOT
                          NULL
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          Caption
                          TEXT,
                          Name
                          TEXT
                      (
                          80
                      ),
                          Text TEXT,
                          Version TEXT
                      (
                          80
                      ),
                          ID TEXT
                      (
                          80
                      )
                          )
		              """ )
		
		prompt_columns = [ row[ 1 ] for row in
			conn.execute( 'PRAGMA table_info("Prompts");' ).fetchall( ) ]
		if 'Caption' not in prompt_columns:
			conn.execute( 'ALTER TABLE "Prompts" ADD COLUMN "Caption" TEXT;' )
		
		conn.commit( )

def get_column_name_tokens( column_name: str ) -> List[ str ]:
	"""Return normalized semantic tokens from a dataframe column name.

	Purpose:
	    Splits camel-case, whitespace-delimited, and punctuation-delimited column names into exact
	    lowercase tokens so schema inference can use semantic evidence without unsafe substring
	    matches.

	Args:
	    column_name (str): Source dataframe column name.

	Returns:
	    List[str]: Ordered normalized name tokens.
	"""
	throw_if( 'column_name', column_name )
	spaced_name = re.sub( r'([a-z0-9])([A-Z])', r'\1 \2', str( column_name ) )
	return [ token for token in re.split( r'[^A-Za-z0-9]+', spaced_name.lower( ) ) if token ]

def get_populated_mask( series: pd.Series ) -> pd.Series:
	"""Return the populated-value mask for a series.

	Purpose:
	    Identifies values that are neither null nor blank text so conversion-success ratios use the
	    number of populated observations rather than the dataframe row count.

	Args:
	    series (pd.Series): Series evaluated for populated values.

	Returns:
	    pd.Series: Boolean mask aligned to the source series.
	"""
	throw_if( 'series', series )
	text_values = series.astype( 'string' ).str.strip( )
	return series.notna( ) & text_values.ne( '' )

def parse_numeric_series( series: pd.Series ) -> pd.Series:
	"""Parse numeric values without modifying the source series.

	Purpose:
	    Converts native numbers and common database or spreadsheet numeric text into numeric values.
	    Thousands separators, currency symbols, percentage suffixes, and accounting parentheses are
	    handled once in a dedicated conversion path while unparseable values become missing.

	Args:
	    series (pd.Series): Source values evaluated as a numeric candidate.

	Returns:
	    pd.Series: Numeric values aligned to the source index.
	"""
	throw_if( 'series', series )
	if pd.api.types.is_numeric_dtype( series ) and not pd.api.types.is_bool_dtype( series ):
		return pd.to_numeric( series, errors='coerce' )

	text_values = series.astype( 'string' ).str.strip( )
	accounting_mask = text_values.str.match( r'^\(.*\)$', na=False )
	cleaned_values = text_values.str.replace( r'^\((.*)\)$', r'-\1', regex=True )
	cleaned_values = cleaned_values.str.replace( r'[$£€¥,]', '', regex=True )
	cleaned_values = cleaned_values.str.replace( r'\s+', '', regex=True )
	cleaned_values = cleaned_values.str.replace( r'%$', '', regex=True )
	parsed_values = pd.to_numeric( cleaned_values, errors='coerce' )
	parsed_values.loc[ accounting_mask & parsed_values.gt( 0 ) ] *= -1
	return parsed_values

def parse_datetime_series( series: pd.Series ) -> pd.Series:
	"""Parse datetime values without modifying the source series.

	Purpose:
	    Converts native datetimes and populated date-like text into pandas datetime values while
	    rejecting unparseable values. Mixed-format parsing is used when supported by pandas, with a
	    compatibility fallback for earlier versions.

	Args:
	    series (pd.Series): Source values evaluated as a datetime candidate.

	Returns:
	    pd.Series: Datetime values aligned to the source index.
	"""
	throw_if( 'series', series )
	if pd.api.types.is_datetime64_any_dtype( series ):
		return pd.to_datetime( series, errors='coerce' )

	try:
		text_values = series.astype( 'string' ).str.strip( )
		parsed_values = pd.to_datetime( text_values, errors='coerce', format='mixed' )
		if parsed_values.notna( ).any( ) or not get_populated_mask( series ).any( ):
			return parsed_values
		return pd.to_datetime( text_values, errors='coerce' )
	except (TypeError, ValueError):
		return pd.to_datetime( series.astype( 'string' ).str.strip( ), errors='coerce' )

def get_declared_type_family( declared_type: str ) -> str:
	"""Return the physical type family for a declared SQLite column type.

	Purpose:
	    Maps SQLite declared type names into stable numeric, datetime, boolean, text, or unknown
	    families so database-backed datasets use authoritative schema metadata before analytical
	    column-name semantics are applied.

	Args:
	    declared_type (str): SQLite declared type returned by ``PRAGMA table_info``.

	Returns:
	    str: Normalized physical type family used by the schema classifier.
	"""
	if not declared_type:
		return 'unknown'

	type_name = declared_type.strip( ).upper( )
	if any( token in type_name for token in ( 'DATE', 'TIME', 'TIMESTAMP' ) ):
		return 'datetime'
	if 'BOOL' in type_name:
		return 'boolean'
	if any( token in type_name for token in (
			'INT', 'REAL', 'FLOA', 'DOUB', 'NUMERIC', 'DECIMAL' ) ):
		return 'numeric'
	if any( token in type_name for token in (
			'CHAR', 'CLOB', 'TEXT', 'VARCHAR', 'NVARCHAR' ) ):
		return 'text'
	return 'unknown'

def create_declared_schema_map( schema_rows: List[ Tuple ] ) -> Dict[ str, str ]:
	"""Create a declared-type mapping from SQLite schema metadata.

	Purpose:
	    Converts ``PRAGMA table_info`` rows into a column-to-declared-type mapping that can be
	    passed to the dataframe schema profiler without coupling the profiler to database I/O.

	Args:
	    schema_rows (List[Tuple]): SQLite ``PRAGMA table_info`` rows.

	Returns:
	    Dict[str, str]: Declared SQLite type keyed by column name.
	"""
	if not schema_rows:
		return { }
	return { str( row[ 1 ] ): str( row[ 2 ] or '' ) for row in schema_rows }

def profile_column( column_name: str, series: pd.Series, override_role: str='',
	declared_type: str='' ) -> Dict[ str, object ]:
	"""Infer the physical type and analytical role of one dataframe column.

	Purpose:
	    Uses an authoritative SQLite declared type when available, then applies exact column-name
	    semantics to distinguish identifiers, categories, ordered variables, measures, booleans,
	    and datetimes. Dataframe value inference remains the fallback for non-database sources.
	    Explicit user overrides retain highest priority for the analytical role.

	Args:
	    column_name (str): Column name associated with the supplied series.
	    series (pd.Series): Column values evaluated by the profiler.
	    override_role (str): Optional user-selected analytical role.
	    declared_type (str): Optional SQLite declared type for database-backed data.

	Returns:
	    Dict[str, object]: Detailed profile containing storage type, inferred dtype, analytical role,
	        confidence, evidence, and cardinality statistics.
	"""
	throw_if( 'column_name', column_name )
	throw_if( 'series', series )
	valid_roles = { 'numeric', 'categorical', 'ordinal', 'identifier', 'datetime' }
	tokens = set( get_column_name_tokens( column_name ) )
	populated_mask = get_populated_mask( series )
	populated_count = int( populated_mask.sum( ) )
	distinct_count = int( series[ populated_mask ].nunique( dropna=True ) )
	unique_ratio = distinct_count / max( 1, populated_count )
	text_values = series.astype( 'string' ).str.strip( )
	numeric_values = parse_numeric_series( series )
	numeric_success = float( numeric_values[ populated_mask ].notna( ).mean( ) ) if (
		populated_count > 0) else 0.0
	integer_like = bool( numeric_values[ populated_mask ].dropna( ).mod( 1 ).eq( 0 ).all( ) ) if (
		numeric_values[ populated_mask ].notna( ).any( )) else False
	leading_zero_ratio = float( text_values[ populated_mask ].str.match(
		r'^[+-]?0\d+$', na=False ).mean( ) ) if populated_count > 0 else 0.0

	date_tokens = { 'date', 'datetime', 'timestamp', 'time', 'effective', 'expiration', 'expiry' }
	temporal_context_tokens = { 'approved', 'approval', 'created', 'modified', 'updated',
		'submitted', 'posted', 'received', 'processed', 'opened', 'closed' }
	identifier_tokens = { 'id', 'identifier', 'key', 'uuid', 'guid', 'index', 'sequence' }
	ordinal_tokens = { 'rank', 'grade', 'level', 'rating', 'priority', 'stage', 'tier', 'fy' }
	category_tokens = { 'category', 'categories', 'class', 'type', 'status', 'agency', 'program',
		'symbol', 'split', 'authority', 'footnote', 'footnotes', 'code' }
	name_tokens = { 'name', 'title', 'description', 'label', 'caption' }
	boolean_values = { 'yes', 'no', 'true', 'false', 'y', 'n', '0', '1' }
	normalized_values = set( text_values[ populated_mask ].str.lower( ).unique( ).tolist( ) )
	boolean_candidate = bool( normalized_values ) and normalized_values.issubset( boolean_values )
	identifier_evidence = bool( tokens & identifier_tokens )
	ordinal_evidence = bool( tokens & ordinal_tokens ) or ({ 'fiscal', 'year' }.issubset( tokens )) or (
		{ 'calendar', 'year' }.issubset( tokens ))
	date_name_evidence = bool( tokens & date_tokens ) or ({ 'start', 'date' }.issubset( tokens )) or (
		{ 'end', 'date' }.issubset( tokens ))
	temporal_context_evidence = bool( tokens & temporal_context_tokens )
	category_evidence = bool( tokens & category_tokens )
	name_evidence = bool( tokens & name_tokens )

	date_lexical_mask = text_values[ populated_mask ].str.contains(
		r'[-/:T]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',
		case=False, regex=True, na=False )
	date_lexical_ratio = float( date_lexical_mask.mean( ) ) if populated_count > 0 else 0.0
	datetime_values = parse_datetime_series( series ) if (
		pd.api.types.is_datetime64_any_dtype( series ) or date_name_evidence or
		temporal_context_evidence or date_lexical_ratio > 0.0) else pd.Series( pd.NaT,
		index=series.index, dtype='datetime64[ns]' )
	datetime_success = float( datetime_values[ populated_mask ].notna( ).mean( ) ) if (
		populated_count > 0) else 0.0
	datetime_value_evidence = bool( populated_count > 0 and datetime_success == 1.0 and (
		date_name_evidence or temporal_context_evidence or date_lexical_ratio > 0.0) )

	declared_family = get_declared_type_family( declared_type )
	storage_dtype = declared_type if declared_type else str( series.dtype )
	analytical_role = 'categorical'
	inferred_dtype = 'text'
	confidence = 0.60
	reason = 'Text values default to a categorical analytical role.'

	if override_role in valid_roles:
		analytical_role = override_role
		inferred_dtype = 'overridden'
		confidence = 1.0
		reason = f'User override selected the {override_role} analytical role.'
	elif declared_family != 'unknown':
		if declared_family == 'datetime':
			analytical_role = 'datetime'
			inferred_dtype = 'datetime'
			confidence = 1.0
			reason = f'SQLite declares the column as {declared_type}.'
		elif declared_family == 'boolean':
			analytical_role = 'categorical'
			inferred_dtype = 'boolean'
			confidence = 1.0
			reason = f'SQLite declares the column as {declared_type}; booleans are categorical.'
		elif declared_family == 'numeric':
			inferred_dtype = 'integer' if 'INT' in declared_type.upper( ) else 'float'
			if identifier_evidence:
				analytical_role = 'identifier'
				confidence = 0.99
				reason = 'SQLite declares a numeric type and the column name identifies a key or ID.'
			elif ordinal_evidence:
				analytical_role = 'ordinal'
				confidence = 0.99
				reason = 'SQLite declares a numeric type and the column name indicates an ordered variable.'
			elif category_evidence:
				analytical_role = 'categorical'
				confidence = 0.99
				reason = 'SQLite declares a numeric type but the column name indicates a category or code.'
			else:
				analytical_role = 'numeric'
				confidence = 1.0
				reason = f'SQLite declares the column as numeric type {declared_type}.'
		elif declared_family == 'text':
			inferred_dtype = 'text'
			if date_name_evidence and datetime_success > 0.0:
				analytical_role = 'datetime'
				inferred_dtype = 'datetime'
				confidence = 0.99
				reason = (
					'SQLite declares text storage, the column name indicates a date or time, '
					'and populated values parse as datetimes.' )
			elif datetime_value_evidence:
				analytical_role = 'datetime'
				inferred_dtype = 'datetime'
				confidence = 0.98
				reason = (
					'SQLite declares text storage and all populated values contain date-like '
					'structure and parse as datetimes.' )
			elif identifier_evidence:
				analytical_role = 'identifier'
				confidence = 0.98
				reason = 'SQLite declares text storage and the column name identifies a key or ID.'
			elif ordinal_evidence:
				analytical_role = 'ordinal'
				confidence = 0.98
				reason = 'SQLite declares text storage and the column name indicates an ordered variable.'
			elif category_evidence or name_evidence:
				analytical_role = 'categorical'
				confidence = 0.98
				reason = 'SQLite declares text storage and the column name indicates categorical text.'
			else:
				analytical_role = 'categorical'
				confidence = 0.95
				reason = f'SQLite declares the column as text type {declared_type}.'
	elif pd.api.types.is_datetime64_any_dtype( series ):
		analytical_role = 'datetime'
		inferred_dtype = 'datetime'
		confidence = 1.0
		reason = 'The pandas storage dtype is datetime.'
	elif pd.api.types.is_bool_dtype( series ) or boolean_candidate:
		analytical_role = 'categorical'
		inferred_dtype = 'boolean'
		confidence = 0.98
		reason = 'The values form a boolean or two-state category.'
	elif date_name_evidence and datetime_success > 0.0:
		analytical_role = 'datetime'
		inferred_dtype = 'datetime'
		confidence = 0.96
		reason = 'The column name indicates a date or time and populated values parse as datetimes.'
	elif datetime_value_evidence:
		analytical_role = 'datetime'
		inferred_dtype = 'datetime'
		confidence = 0.95
		reason = 'All populated values contain date-like structure and parse as datetimes.'
	elif identifier_evidence:
		analytical_role = 'identifier'
		inferred_dtype = 'integer' if integer_like else 'text'
		confidence = 0.95
		reason = 'The column name identifies a key or ID.'
	elif numeric_success == 1.0 and populated_count > 0:
		inferred_dtype = 'integer' if integer_like else 'float'
		if ordinal_evidence:
			analytical_role = 'ordinal'
			confidence = 0.95
			reason = 'Numeric values and the column name indicate an ordered variable.'
		elif category_evidence:
			analytical_role = 'categorical'
			confidence = 0.95
			reason = 'Numeric values are used by a column whose name indicates a category or code.'
		else:
			analytical_role = 'numeric'
			confidence = 0.95
			reason = 'All populated values parse as numeric values.'
	elif isinstance( series.dtype, pd.CategoricalDtype ):
		analytical_role = 'categorical'
		inferred_dtype = 'category'
		confidence = 1.0
		reason = 'The pandas storage dtype is categorical.'

	return { 'column': column_name, 'storage_dtype': storage_dtype,
		'inferred_dtype': inferred_dtype, 'analytical_role': analytical_role,
		'non_null_count': populated_count, 'distinct_count': distinct_count,
		'unique_ratio': unique_ratio, 'numeric_success_ratio': numeric_success,
		'datetime_success_ratio': datetime_success, 'leading_zero_ratio': leading_zero_ratio,
		'is_integer_like': integer_like, 'confidence': confidence, 'reason': reason }

def profile_dataframe_schema( df: pd.DataFrame,
	declared_schema: Dict[ str, str ]=None ) -> Dict[ str, Dict[ str, object ] ]:
	"""Profile every dataframe column using the authoritative schema classifier.

	Purpose:
	    Profiles dataframe columns with optional SQLite declared-type metadata while preserving the
	    established detailed profile contract and dataset-specific analytical-role overrides.

	Args:
	    df (pd.DataFrame): Dataframe whose storage types and analytical roles are profiled.
	    declared_schema (Dict[str, str]): Optional SQLite declared types keyed by column name.

	Returns:
	    Dict[str, Dict[str, object]]: Detailed profile keyed by dataframe column name.
	"""
	throw_if( 'df', df )
	schema_signature = repr( (tuple( df.columns.tolist( ) ),
		tuple( str( dtype ) for dtype in df.dtypes.tolist( ) )) )
	overrides_by_schema = st.session_state.get( 'column_type_overrides', { } )
	legacy_override_values = list( overrides_by_schema.values( ) ) if isinstance(
		overrides_by_schema, dict ) else [ ]
	if legacy_override_values and all( isinstance( value, str ) for value in
			legacy_override_values ):
		overrides = overrides_by_schema
	else:
		overrides = overrides_by_schema.get( schema_signature, { } ) if isinstance(
			overrides_by_schema, dict ) else { }
	declared_types = declared_schema if isinstance( declared_schema, dict ) else { }
	return { column: profile_column( str( column ), df[ column ], str( overrides.get(
		column, '' ) ), str( declared_types.get( column, '' ) ) ) for column in df.columns }

def infer_schema( df: pd.DataFrame ) -> Dict[ str, str ]:
	"""Return backward-compatible analytical roles for dataframe columns.

	Purpose:
	    Preserves the established ``infer_schema`` mapping contract while delegating all decisions
	    to the authoritative detailed dataframe profiler.

	Args:
	    df (pd.DataFrame): Dataframe whose columns are classified.

	Returns:
	    Dict[str, str]: Mapping of column names to numeric, categorical, ordinal, identifier, or
	        datetime analytical roles.
	"""
	profiles = profile_dataframe_schema( df )
	return { column: str( profile[ 'analytical_role' ] ) for column, profile in profiles.items( ) }

def create_typed_analysis_dataframe( df: pd.DataFrame,
	profiles: Dict[ str, Dict[ str, object ] ] ) -> pd.DataFrame:
	"""Create a typed analytical copy of a dataframe.

	Purpose:
	    Converts inferred numeric measures and datetimes in an independent dataframe while
	    preserving categories, identifiers, leading-zero codes, source values, database content,
	    and export representations in the active dataset.

	Args:
	    df (pd.DataFrame): Source dataframe retained without mutation.
	    profiles (Dict[str, Dict[str, object]]): Authoritative column profiles for the dataframe.

	Returns:
	    pd.DataFrame: Typed dataframe copy for statistics, visualization, and modeling.
	"""
	throw_if( 'df', df )
	throw_if( 'profiles', profiles )
	df_analysis = df.copy( )
	for column, profile in profiles.items( ):
		role = str( profile[ 'analytical_role' ] )
		if role == 'numeric':
			df_analysis[ column ] = parse_numeric_series( df_analysis[ column ] )
		elif role == 'datetime':
			df_analysis[ column ] = parse_datetime_series( df_analysis[ column ] )
	return df_analysis

def get_analysis_dataset( ) -> pd.DataFrame | None:
	"""Return a synchronized typed dataframe for analytical workflows.

	Purpose:
	    Retrieves the active dataset, synchronizes its authoritative schema metadata, and returns an
	    isolated typed copy without changing source data or database-backed representations.

	Returns:
	    pd.DataFrame | None: Typed analytical dataframe or ``None`` when no dataset is loaded.
	"""
	df_dataset = get_loaded_dataset( )
	if df_dataset is None:
		return None
	synchronize_dataset_columns( df_dataset )
	return st.session_state[ 'df_analysis' ].copy( )

def create_connection( ) -> sqlite3.Connection:
	"""Create a connection to the configured SQLite database.

	Purpose:
	    Opens a new SQLite connection using the database path defined by the application
	    configuration.

	Returns:
	    sqlite3.Connection: Open SQLite database connection.
	"""
	return sqlite3.connect( cfg.DB_PATH )

def table_supports_rowid( table: str ) -> bool:
	"""Determine whether a SQLite table exposes the implicit rowid.

	Purpose:
	    Detects ``WITHOUT ROWID`` table definitions before CRUD workflows issue rowid-based
	    selection, update, delete, or verification statements.

	Args:
	    table (str): Existing SQLite table inspected for rowid support.

	Returns:
	    bool: ``True`` when the table exposes SQLite rowid semantics; otherwise ``False``.
	"""
	throw_if( 'table', table )
	with create_connection( ) as conn:
		row = conn.execute(
			"SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,) ).fetchone( )
	if row is None or not row[ 0 ]:
		raise ValueError( f'Table definition for "{table}" could not be found.' )
	return 'WITHOUT ROWID' not in str( row[ 0 ] ).upper( )

def read_table_with_rowid( table: str ) -> pd.DataFrame:
	"""Read a SQLite table with its internal row identifier.

	Purpose:
	    Reads the selected SQLite table while exposing the internal ``rowid`` under an application-
	    owned column name. The internal identifier is used only by CRUD workflows so dataframe row
	    positions are never assumed to be SQLite row identifiers.

	Args:
	    table (str): SQLite table read into dataframe form.

	Returns:
	    pd.DataFrame: Table rows containing ``__mathy_rowid__`` followed by the stored columns.
	"""
	throw_if( 'table', table )
	with create_connection( ) as conn:
		return pd.read_sql_query( f'SELECT rowid AS "__mathy_rowid__", * FROM "{table}";', conn )

def read_database_record( table: str, rowid: int ) -> Dict[ str, object ] | None:
	"""Read one SQLite record by internal row identifier.

	Purpose:
	    Retrieves the current values of a selected database record so CRUD controls are populated
	    from authoritative SQLite state instead of placeholder values or dataframe row positions.

	Args:
	    table (str): SQLite table containing the requested record.
	    rowid (int): SQLite internal row identifier.

	Returns:
	    Dict[str, object] | None: Column/value mapping for the record, or ``None`` when the row does
	        not exist.
	"""
	throw_if( 'table', table )
	throw_if( 'rowid', rowid )
	with create_connection( ) as conn:
		cursor = conn.execute( f'SELECT * FROM "{table}" WHERE rowid=?;', (int( rowid ),) )
		row = cursor.fetchone( )
		if row is None:
			return None
		columns = [ description[ 0 ] for description in cursor.description or [ ] ]
		return { column: row[ index ] for index, column in enumerate( columns ) }

def coerce_sqlite_value( value: object, declared_type: str ) -> object:
	"""Coerce an edited control value to its declared SQLite storage family.

	Purpose:
	    Converts text-oriented CRUD inputs to stable Python values compatible with the selected
	    column's SQLite declared type. Empty non-text inputs become ``None`` while text values retain
	    an intentionally entered empty string.

	Args:
	    value (object): User-edited value prepared for persistence.
	    declared_type (str): SQLite declared type associated with the target column.

	Returns:
	    object: Converted SQLite-compatible scalar value.
	"""
	family = get_declared_type_family( declared_type )
	if value is None:
		return None
	if isinstance( value, str ):
		text = value.strip( )
		if family != 'text' and text == '':
			return None
	else:
		text = str( value )

	if family == 'numeric':
		parsed = pd.to_numeric( pd.Series( [ text ] ), errors='coerce' ).iloc[ 0 ]
		if pd.isna( parsed ):
			raise ValueError( f'Value "{value}" is not valid for SQLite type {declared_type}.' )
		if 'INT' in declared_type.upper( ):
			if float( parsed ) % 1 != 0:
				raise ValueError( f'Value "{value}" is not a whole number.' )
			return int( parsed )
		return float( parsed )
	if family == 'boolean':
		if isinstance( value, bool ):
			return int( value )
		normalized = text.lower( )
		if normalized in ( '1', 'true', 'yes', 'y', 'on' ):
			return 1
		if normalized in ( '0', 'false', 'no', 'n', 'off' ):
			return 0
		raise ValueError( f'Value "{value}" is not valid for SQLite type {declared_type}.' )
	if family == 'datetime':
		parsed = pd.to_datetime( text, errors='coerce' )
		if pd.isna( parsed ):
			raise ValueError( f'Value "{value}" is not a valid datetime.' )
		return parsed.isoformat( )
	if 'BLOB' in declared_type.upper( ):
		return value if isinstance( value, bytes ) else text.encode( 'utf-8' )
	return str( value )

def create_crud_database_table( table_name: str,
	columns: List[ Dict[ str, str ] ] ) -> Tuple[ str, bool ]:
	"""Create a SQLite table from CRUD Ops schema controls.

	Purpose:
	    Creates a new user-defined SQLite table with an automatic ``Id INTEGER PRIMARY KEY
	    AUTOINCREMENT`` column, validates and normalizes each requested column identifier, preserves
	    the selected SQLite data types, and optionally inserts one initial row when at least one
	    initial field value is supplied.

	Args:
	    table_name (str): User-entered table name normalized to a valid SQLite identifier.
	    columns (List[Dict[str, str]]): User-defined column specifications containing ``name``,
	        ``type``, and optional ``initial_value`` values.

	Returns:
	    Tuple[str, bool]: Created SQLite table name followed by a flag indicating whether an initial
	        row was inserted.
	"""
	throw_if( 'table_name', table_name )
	throw_if( 'columns', columns )
	safe_table_name = create_identifier( table_name )
	existing_tables = { name.lower( ) for name in list_tables( ) }
	if safe_table_name.lower( ) in existing_tables:
		raise ValueError( f'Table "{safe_table_name}" already exists.' )

	supported_types = { 'INTEGER', 'REAL', 'NUMERIC', 'TEXT', 'BLOB' }
	column_definitions: List[ str ] = [ '"Id" INTEGER PRIMARY KEY AUTOINCREMENT' ]
	column_names: List[ str ] = [ ]
	initial_values: List[ object ] = [ ]
	has_initial_values = False
	seen_columns = { 'id' }

	for index, column in enumerate( columns ):
		column_name = str( column.get( 'name', '' ) )
		throw_if( f'column_{index + 1}_name', column_name )
		safe_column_name = create_identifier( column_name )
		if safe_column_name.lower( ) in seen_columns:
			raise ValueError( f'Column name "{safe_column_name}" is duplicated or reserved.' )
		seen_columns.add( safe_column_name.lower( ) )

		declared_type = str( column.get( 'type', 'TEXT' ) ).upper( )
		if declared_type not in supported_types:
			raise ValueError( f'Unsupported SQLite type: {declared_type}' )

		raw_initial_value = column.get( 'initial_value', '' )
		initial_text = str( raw_initial_value ).strip( ) if raw_initial_value is not None else ''
		if initial_text:
			initial_value = coerce_sqlite_value( raw_initial_value, declared_type )
			has_initial_values = True
		else:
			initial_value = None

		column_definitions.append( f'"{safe_column_name}" {declared_type}' )
		column_names.append( safe_column_name )
		initial_values.append( initial_value )

	create_statement = (f'CREATE TABLE "{safe_table_name}" '
		f'({", ".join( column_definitions )});')

	with create_connection( ) as conn:
		try:
			conn.execute( 'BEGIN' )
			conn.execute( create_statement )
			if has_initial_values:
				column_list = ', '.join( f'"{name}"' for name in column_names )
				placeholders = ', '.join( [ '?' ] * len( column_names ) )
				conn.execute( f'INSERT INTO "{safe_table_name}" ({column_list}) '
					f'VALUES ({placeholders});', initial_values )
			conn.commit( )
		except Exception:
			conn.rollback( )
			raise

	created_schema = create_schema( safe_table_name )
	created_columns = [ str( row[ 1 ] ) for row in created_schema ]
	expected_columns = [ 'Id' ] + column_names
	if created_columns != expected_columns:
		raise RuntimeError( 'Created table schema could not be verified.' )
	if has_initial_values:
		df_created = read_table( safe_table_name )
		if len( df_created ) != 1:
			raise RuntimeError( 'Initial table row could not be verified.' )

	return safe_table_name, has_initial_values

def database_record_exists( table: str, values: Dict[ str, object ],
	exclude_rowid: int=0 ) -> bool:
	"""Determine whether an equivalent complete SQLite record already exists.

	Purpose:
	    Compares every supplied persisted field against existing rows so Add Row can prevent exact
	    duplicate records without incorrectly prohibiting legitimate reuse of individual categorical
	    or dimensional values.

	Args:
	    table (str): SQLite table checked for a duplicate record.
	    values (Dict[str, object]): Persisted column/value mapping for the proposed record.
	    exclude_rowid (int): Optional row identifier excluded from duplicate detection during edits.

	Returns:
	    bool: ``True`` when an equivalent record exists; otherwise ``False``.
	"""
	throw_if( 'table', table )
	throw_if( 'values', values )
	clauses = [ f'"{column}" IS ?' for column in values.keys( ) ]
	parameters = list( values.values( ) )
	if exclude_rowid > 0:
		clauses.append( 'rowid <> ?' )
		parameters.append( int( exclude_rowid ) )
	query = f'SELECT 1 FROM "{table}" WHERE {" AND ".join( clauses )} LIMIT 1;'
	with create_connection( ) as conn:
		return conn.execute( query, parameters ).fetchone( ) is not None

def database_record_matches( table: str, rowid: int, values: Dict[ str, object ] ) -> bool:
	"""Verify persisted values for one SQLite record.

	Purpose:
	    Confirms that a committed row contains the exact SQLite values submitted by a CRUD update or
	    insert operation so success messages are emitted only after database readback verification.

	Args:
	    table (str): SQLite table containing the record.
	    rowid (int): SQLite internal row identifier of the record.
	    values (Dict[str, object]): Expected persisted column/value mapping.

	Returns:
	    bool: ``True`` when the selected row contains all expected values; otherwise ``False``.
	"""
	throw_if( 'table', table )
	throw_if( 'rowid', rowid )
	throw_if( 'values', values )
	clauses = [ 'rowid=?' ] + [ f'"{column}" IS ?' for column in values.keys( ) ]
	parameters = [ int( rowid ) ] + list( values.values( ) )
	query = f'SELECT 1 FROM "{table}" WHERE {" AND ".join( clauses )} LIMIT 1;'
	with create_connection( ) as conn:
		return conn.execute( query, parameters ).fetchone( ) is not None

def refresh_crud_database_state( table: str ) -> pd.DataFrame:
	"""Refresh CRUD and active analytical state after a committed SQLite mutation.

	Purpose:
	    Re-reads the selected SQLite table after a successful mutation. When the table is also the
	    active Mathy database source, the loaded dataset, original baseline, declared schema, and
	    analytical column classifications are synchronized to the committed database state.

	Args:
	    table (str): SQLite table whose committed state is refreshed.

	Returns:
	    pd.DataFrame: Fresh dataframe read from the selected table.
	"""
	throw_if( 'table', table )
	df_database = read_table( table )
	if (st.session_state.get( 'active_source_kind', '' ) == 'database' and
			st.session_state.get( 'active_database_table', '' ) == table):
		st.session_state[ 'active_declared_schema' ] = create_declared_schema_map(
			create_schema( table ) )
		if has_loaded_dataset( df_database ):
			store_loaded_dataset( df_database, df_database )
		else:
			st.session_state[ 'raw_df' ] = df_database.copy( )
			st.session_state[ 'df_original' ] = df_database.copy( )
			st.session_state[ 'df_dataset' ] = df_database.copy( )
			synchronize_dataset_columns( df_database )
	return df_database

def queue_database_success( message: str ) -> None:
	"""Queue a database success message for the next Streamlit render.

	Purpose:
	    Stores one verified database-operation result in session state so a success dialog can be
	    rendered after the operation-triggered rerun without losing the acknowledgement.

	Args:
	    message (str): Success message displayed in the database operation dialog.

	Returns:
	    None: This function updates Streamlit session state in place.
	"""
	throw_if( 'message', message )
	st.session_state[ 'database_operation_result' ] = str( message )

def activate_database_table( table: str ) -> pd.DataFrame:
	"""Activate one persisted SQLite table as the Mathy dataset.

	Purpose:
	    Reads the selected table and declared SQLite schema, records the database-backed source
	    identity, and synchronizes the active, original, raw, typed, and analytical dataset state.

	Args:
	    table (str): Existing SQLite table activated as the current Mathy dataset.

	Returns:
	    pd.DataFrame: Fresh dataframe read from the activated SQLite table.
	"""
	throw_if( 'table', table )
	if table not in list_tables( ):
		raise ValueError( f'Table "{table}" does not exist.' )
	df_database = read_table( table )
	st.session_state[ 'active_source_kind' ] = 'database'
	st.session_state[ 'active_database_table' ] = table
	st.session_state[ 'active_declared_schema' ] = create_declared_schema_map(
		create_schema( table ) )
	st.session_state[ 'active_source_signature' ] = ('Database Data', table)
	if has_loaded_dataset( df_database ):
		store_loaded_dataset( df_database, df_database )
	else:
		st.session_state[ 'raw_df' ] = df_database.copy( )
		st.session_state[ 'df_original' ] = df_database.copy( )
		st.session_state[ 'df_dataset' ] = df_database.copy( )
		synchronize_dataset_columns( df_database )
	return df_database

def clear_active_database_table( ) -> None:
	"""Clear active database-backed dataset state.

	Purpose:
	    Removes a deleted database table from the active source contract and resets loaded dataset
	    state when no replacement table is available.

	Returns:
	    None: This function updates Streamlit session state in place.
	"""
	st.session_state[ 'active_source_kind' ] = ''
	st.session_state[ 'active_database_table' ] = ''
	st.session_state[ 'active_declared_schema' ] = { }
	st.session_state[ 'active_source_signature' ]=None
	st.session_state[ 'raw_df' ] = pd.DataFrame( )
	st.session_state[ 'df_original' ] = pd.DataFrame( )
	st.session_state[ 'df_dataset' ] = pd.DataFrame( )
	synchronize_dataset_columns( pd.DataFrame( ) )

def create_available_table_names( requested_names: List[ str ],
	existing_tables: List[ str ] ) -> List[ str ]:
	"""Create collision-safe SQLite table names.

	Purpose:
	    Normalizes requested names to valid identifiers and appends numeric suffixes when a requested
	    name conflicts with an existing or already-reserved table name.

	Args:
	    requested_names (List[str]): Source names converted to database table identifiers.
	    existing_tables (List[str]): Current database tables reserved against collisions.

	Returns:
	    List[str]: Unique SQLite-safe table names in source order.
	"""
	throw_if( 'requested_names', requested_names )
	reserved = { str( table ).lower( ) for table in existing_tables }
	available: List[ str ] = [ ]
	for requested_name in requested_names:
		base_name = create_identifier( requested_name )
		candidate = base_name
		suffix = 2
		while candidate.lower( ) in reserved:
			candidate = f'{base_name}_{suffix}'
			suffix += 1
		reserved.add( candidate.lower( ) )
		available.append( candidate )
	return available

def write_dataframe_tables_to_database( df_tables: Dict[ str, pd.DataFrame ],
	overwrite: bool=False ) -> Dict[ str, pd.DataFrame ]:
	"""Persist dataframe tables to SQLite in one transaction.

	Purpose:
	    Normalizes uploaded dataframe column identifiers, creates each requested SQLite table, bulk
	    inserts normalized values, and rolls back the entire operation when any table fails.

	Args:
	    df_tables (Dict[str, pd.DataFrame]): Desired table names mapped to source dataframes.
	    overwrite (bool): Whether existing tables with matching names may be replaced.

	Returns:
	    Dict[str, pd.DataFrame]: Fresh persisted dataframes keyed by created table name.
	"""
	throw_if( 'df_tables', df_tables )
	with create_connection( ) as conn:
		try:
			conn.execute( 'BEGIN' )
			for requested_name, df_source in df_tables.items( ):
				throw_if( 'requested_name', requested_name )
				throw_if( 'df_source', df_source )
				if len( df_source.columns ) == 0:
					raise ValueError( f'Table "{requested_name}" does not contain any columns.' )
				table_name = create_identifier( requested_name )
				df_table = df_source.copy( )
				df_table.columns = create_unique_upload_identifiers( df_table.columns.tolist( ) )
				if overwrite:
					conn.execute( f'DROP TABLE IF EXISTS "{table_name}";' )
				elif table_name in list_tables( ):
					raise ValueError( f'Table "{table_name}" already exists.' )
				columns: List[ str ] = [ ]
				for column in df_table.columns:
					sql_type = get_sqlite_type( df_table[ column ].dtype )
					columns.append( f'"{column}" {sql_type}' )
				conn.execute( f'CREATE TABLE "{table_name}" ({", ".join( columns )});' )
				if not df_table.empty:
					placeholders = ', '.join( [ '?' ] * len( df_table.columns ) )
					rows = [ tuple( normalize_upload_sql_value( value ) for value in row )
						for row in df_table.itertuples( index=False, name=None ) ]
					conn.executemany( f'INSERT INTO "{table_name}" VALUES ({placeholders});', rows )
			conn.commit( )
		except Exception:
			conn.rollback( )
			raise

	persisted_tables: Dict[ str, pd.DataFrame ] = { }
	for table_name, df_source in df_tables.items( ):
		created_name = create_identifier( table_name )
		df_persisted = read_table( created_name )
		if len( df_persisted ) != len( df_source ) or len( df_persisted.columns ) != len( df_source.columns ):
			raise RuntimeError( f'Table "{created_name}" could not be verified after import.' )
		persisted_tables[ created_name ] = df_persisted
	return persisted_tables

def reconcile_import_table_metadata( old_table: str, new_table: str='' ) -> None:
	"""Reconcile upload metadata after a table rename or deletion.

	Purpose:
	    Updates sidebar and Data Upload table mappings so renamed or deleted SQLite tables do not
	    remain as stale session-state references.

	Args:
	    old_table (str): Previous table name removed from metadata mappings.
	    new_table (str): Replacement table name for a rename; empty for deletion.

	Returns:
	    None: This function updates upload-related session state in place.
	"""
	throw_if( 'old_table', old_table )
	for key in ( 'data_upload_tables', 'custom_upload_tables' ):
		table_map = st.session_state.get( key, { } )
		if not isinstance( table_map, dict ) or old_table not in table_map:
			continue
		value = table_map.pop( old_table )
		if new_table:
			table_map[ new_table ] = read_table( new_table ) if new_table in list_tables( ) else value
		st.session_state[ key ] = table_map
		if key == 'data_upload_tables':
			if st.session_state.get( 'data_upload_selected_table', None ) == old_table:
				if new_table:
					st.session_state[ 'data_upload_selected_table' ] = new_table
				else:
					clear_keys( [ 'data_upload_selected_table' ] )
			if not table_map and not new_table:
				st.session_state[ 'data_upload_signature' ]=None
		elif key == 'custom_upload_tables' and not table_map and not new_table:
			st.session_state[ 'custom_upload_signature' ]=None

	sheet_names = st.session_state.get( 'data_upload_sheet_names', { } )
	if isinstance( sheet_names, dict ) and old_table in sheet_names:
		value = sheet_names.pop( old_table )
		if new_table:
			sheet_names[ new_table ] = value
		st.session_state[ 'data_upload_sheet_names' ] = sheet_names

def split_sqlite_definitions( sql_text: str ) -> List[ str ]:
	"""Split a SQLite table-definition body at top-level commas.

	Purpose:
	    Separates column and table-constraint definitions while preserving commas contained inside
	    quoted text, parentheses, default expressions, and check constraints.

	Args:
	    sql_text (str): Text contained inside a SQLite ``CREATE TABLE`` parenthesized definition.

	Returns:
	    List[str]: Ordered top-level SQL definitions.
	"""
	throw_if( 'sql_text', sql_text )
	definitions: List[ str ] = [ ]
	buffer: List[ str ] = [ ]
	depth = 0
	quote = ''
	for char in sql_text:
		if quote:
			buffer.append( char )
			if char == quote:
				quote = ''
			continue
		if char in ( '"', "'", '`' ):
			quote = char
			buffer.append( char )
		elif char == '(':
			depth += 1
			buffer.append( char )
		elif char == ')':
			depth = max( 0, depth - 1 )
			buffer.append( char )
		elif char == ',' and depth == 0:
			definitions.append( ''.join( buffer ).strip( ) )
			buffer = [ ]
		else:
			buffer.append( char )
	if buffer:
		definitions.append( ''.join( buffer ).strip( ) )
	return definitions

def replace_sqlite_column_type( create_sql: str, table: str, column: str,
	new_type: str, temp_table: str ) -> str:
	"""Build a temporary CREATE TABLE statement with one revised declared type.

	Purpose:
	    Preserves the original SQLite column definitions and table-level constraints while replacing
	    only the selected column's declared type and temporary table name for an atomic schema rebuild.

	Args:
	    create_sql (str): Original SQLite ``CREATE TABLE`` statement.
	    table (str): Existing table name.
	    column (str): Column whose declared type is changed.
	    new_type (str): Replacement SQLite declared type.
	    temp_table (str): Temporary table name used during reconstruction.

	Returns:
	    str: Temporary ``CREATE TABLE`` statement containing the revised declared type.
	"""
	throw_if( 'create_sql', create_sql )
	throw_if( 'table', table )
	throw_if( 'column', column )
	throw_if( 'new_type', new_type )
	throw_if( 'temp_table', temp_table )
	open_paren = create_sql.find( '(' )
	close_paren = create_sql.rfind( ')' )
	if open_paren < 0 or close_paren <= open_paren:
		raise ValueError( 'Malformed CREATE TABLE statement.' )

	definitions = split_sqlite_definitions( create_sql[ open_paren + 1: close_paren ] )
	constraint_tokens = { 'PRIMARY', 'NOT', 'UNIQUE', 'CHECK', 'DEFAULT', 'COLLATE',
		'REFERENCES', 'GENERATED' }
	updated = False
	for index, definition in enumerate( definitions ):
		match = re.match( r'^\s*(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([^\s]+))\s+(.*)$',
			definition, flags=re.DOTALL )
		if not match:
			continue
		name = next( (group for group in match.groups( )[ :4 ] if group is not None), '' )
		if name != column:
			continue
		remainder = match.group( 5 ).strip( )
		parts = remainder.split( )
		constraint_index = next( (position for position, token in enumerate( parts )
			if token.upper( ) in constraint_tokens), len( parts ) )
		constraints = ' '.join( parts[ constraint_index: ] )
		quoted_name = f'"{column}"'
		definitions[ index ] = f'{quoted_name} {new_type}' + (
			f' {constraints}' if constraints else '' )
		updated = True
		break
	if not updated:
		raise ValueError( f'Column "{column}" was not found in the table definition.' )

	suffix = create_sql[ close_paren + 1: ].strip( ).rstrip( ';' )
	return f'CREATE TABLE "{temp_table}" ({", ".join( definitions )})' + (
		f' {suffix};' if suffix else ';' )

def change_column_type( table: str, column: str, new_type: str ) -> None:
	"""Change a SQLite column's declared type with an atomic table rebuild.

	Purpose:
	    Validates convertibility of populated values, recreates the table using its original schema
	    and constraints with one revised declared type, casts the selected column during data copy,
	    recreates explicit indexes and triggers, and commits the change as one transaction.

	Args:
	    table (str): Existing SQLite table containing the target column.
	    column (str): Column whose declared type is changed.
	    new_type (str): Supported replacement SQLite type.

	Returns:
	    None: This function changes the SQLite schema and stored values in place.
	"""
	throw_if( 'table', table )
	throw_if( 'column', column )
	throw_if( 'new_type', new_type )
	supported_types = { 'INTEGER', 'REAL', 'NUMERIC', 'TEXT', 'BLOB' }
	target_type = new_type.upper( )
	if target_type not in supported_types:
		raise ValueError( f'Unsupported SQLite type: {new_type}' )

	df_table = read_table( table )
	if column not in df_table.columns:
		raise ValueError( f'Column "{column}" does not exist.' )
	series = df_table[ column ]
	populated = get_populated_mask( series )
	if target_type in ( 'INTEGER', 'REAL', 'NUMERIC' ) and populated.any( ):
		parsed = parse_numeric_series( series )
		if not parsed[ populated ].notna( ).all( ):
			raise ValueError( f'Column "{column}" contains values that cannot be converted to {target_type}.' )
		if target_type == 'INTEGER' and not parsed[ populated ].dropna( ).mod( 1 ).eq( 0 ).all( ):
			raise ValueError( f'Column "{column}" contains non-integer numeric values.' )

	temp_table = f'{table}__mathy_type_rebuild'
	with create_connection( ) as conn:
		row = conn.execute( "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
			(table,) ).fetchone( )
		if row is None or not row[ 0 ]:
			raise ValueError( 'Table definition not found.' )
		create_sql = str( row[ 0 ] )
		index_sql = [ item[ 0 ] for item in conn.execute(
			"SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL;",
			(table,) ).fetchall( ) if item[ 0 ] ]
		trigger_sql = [ item[ 0 ] for item in conn.execute(
			"SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=? AND sql IS NOT NULL;",
			(table,) ).fetchall( ) if item[ 0 ] ]
		columns = [ str( item[ 1 ] ) for item in conn.execute(
			f'PRAGMA table_info("{table}");' ).fetchall( ) ]
		new_create_sql = replace_sqlite_column_type( create_sql, table, column, target_type,
			temp_table )
		select_expressions = [ (
			f"""CASE WHEN "{name}" IS NULL OR TRIM(CAST("{name}" AS TEXT)) = '' THEN NULL """
			f'ELSE CAST("{name}" AS {target_type}) END'
			if name == column and target_type in ( 'INTEGER', 'REAL', 'NUMERIC' ) else
			f'CAST("{name}" AS {target_type})' if name == column else f'"{name}"' )
			for name in columns ]
		column_list = ', '.join( f'"{name}"' for name in columns )
		select_list = ', '.join( select_expressions )

		try:
			conn.execute( 'BEGIN' )
			conn.execute( f'DROP TABLE IF EXISTS "{temp_table}";' )
			conn.execute( new_create_sql )
			conn.execute( f'INSERT INTO "{temp_table}" ({column_list}) SELECT {select_list} FROM "{table}";' )
			conn.execute( f'DROP TABLE "{table}";' )
			conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table}";' )
			for statement in index_sql:
				conn.execute( statement )
			for statement in trigger_sql:
				conn.execute( statement )
			conn.commit( )
		except Exception:
			conn.rollback( )
			raise

def list_tables( ) -> List[ str ]:
	"""Return SQLite table names from the configured database.

	Purpose:
	    Queries the SQLite schema catalog and returns table names in ascending lexical order for
	    database browsing, schema management, and user-interface selection controls.

	Returns:
	    List[str]: Ordered SQLite table names.
	"""
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	"""Return schema metadata for a SQLite table.

	Purpose:
	    Executes SQLite table-information introspection for the supplied table and returns the
	    resulting column metadata used by schema-aware database workflows.

	Args:
	    table (str): Table whose schema metadata is requested.

	Returns:
	    List[Tuple]: SQLite ``PRAGMA table_info`` rows describing the table columns.
	"""
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int=None, offset: int=0 ) -> pd.DataFrame:
	"""Read a SQLite table into a normalized dataframe.

	Purpose:
	    Executes a table query with optional row limiting, preserves duplicate result-column names
	    by generating unique suffixes, converts non-scalar values into display-safe scalar
	    representations, and returns the normalized rows as a pandas dataframe.

	Args:
	    table (str): SQLite table read into dataframe form.
	    limit (int): Optional maximum number of rows returned.
	    offset (int): Number of rows skipped when ``limit`` is applied.

	Returns:
	    pd.DataFrame: Table rows containing plain Python scalar or string values.
	"""
	if not table:
		return pd.DataFrame( )
	
	query = f'SELECT * FROM "{table}"'
	if limit:
		query += f' LIMIT {int( limit )} OFFSET {int( offset )}'
	
	with create_connection( ) as conn:
		cur = conn.cursor( )
		cur.execute( query )
		
		raw_columns = [ d[ 0 ] for d in (cur.description or [ ]) ]
		rows = cur.fetchall( )
	
	seen: Dict[ str, int ] = { }
	columns: List[ str ] = [ ]
	
	for col in raw_columns:
		name = str( col )
		if name not in seen:
			seen[ name ] = 0
			columns.append( name )
		else:
			seen[ name ] += 1
			columns.append( f'{name}_{seen[ name ]}' )
	
	def _scalarize( value: Any ) -> Any:
		"""Convert a database value into a display-safe scalar.

		Purpose:
		    Preserves null and primitive scalar values, decodes byte values when possible, converts
		    collections and model objects to strings, and provides a final string representation for
		    unsupported database-returned objects.

		Args:
		    value (Any): Database value normalized for dataframe storage and Streamlit display.

		Returns:
		    Any: Primitive scalar, decoded text, hexadecimal text, or string representation.
		"""
		if value is None or isinstance( value, (str, int, float, bool) ):
			return value
		
		if isinstance( value, bytes ):
			try:
				return value.decode( 'utf-8' )
			except Exception:
				return value.hex( )
		
		if isinstance( value, (list, tuple, set, dict) ):
			return str( value )
		
		if hasattr( value, 'model_dump' ):
			try:
				return str( value.model_dump( ) )
			except Exception:
				return str( value )
		
		return str( value )
	
	normalized_rows: List[ Dict[ str, Any ] ] = [ ]
	for row in rows:
		record: Dict[ str, Any ] = { }
		for idx, col in enumerate( columns ):
			record[ col ] = _scalarize( row[ idx ] )
		normalized_rows.append( record )
	
	return pd.DataFrame( normalized_rows, columns=columns )

def get_data_editor_column_config( df: pd.DataFrame,
	existing_config: Dict[ str, object ]=None ) -> Dict[ str, object ]:
	"""Build numeric display configuration for a dataframe editor.

	Purpose:
	    Adds locale-aware thousands separators and two-decimal precision for floating-point and
	    double-precision columns while retaining whole-number formatting for integer columns.
	    Explicit caller configurations take precedence over generated defaults.

	Args:
	    df (pd.DataFrame): Dataframe displayed by the editor.
	    existing_config (Dict[str, object]): Optional specialized caller configuration preserved in
	        the returned mapping.

	Returns:
	    Dict[str, object]: Streamlit column configuration keyed by dataframe column name.
	"""
	throw_if( 'df', df )
	column_config = dict( existing_config ) if existing_config else { }
	profiles = profile_dataframe_schema( df )
	for column in df.columns:
		if column in column_config:
			continue
		series = df[ column ]
		if profiles[ column ][ 'analytical_role' ] != 'numeric':
			continue
		if pd.api.types.is_float_dtype( series ):
			column_config[ column ] = st.column_config.NumberColumn( str( column ),
				format='localized', step=0.01 )
		elif pd.api.types.is_integer_dtype( series ) and not pd.api.types.is_bool_dtype( series ):
			column_config[ column ] = st.column_config.NumberColumn( str( column ),
				format='localized', step=1 )
	return column_config

_streamlit_data_editor = st.data_editor

def render_data_editor( df: pd.DataFrame, use_container_width: bool | None = None,
	key: str='', height: int | str='auto', hide_index: bool | None=None,
	disabled: bool | List[ str ]=False, column_config: Dict[ str, object ]=None,
	num_rows: str='fixed' ) -> pd.DataFrame:
	"""Render a consistently formatted Streamlit dataframe editor.

	Purpose:
	    Preserves the established data-editor arguments while merging automatic numeric formatting
	    with any specialized caller configuration. Floating-point values use the locale equivalent
	    of ``#,##0.00`` whenever Streamlit supports numeric column formatting.

	Args:
	    df (pd.DataFrame): Dataframe rendered by Streamlit.
	    use_container_width (bool | None): Existing Streamlit container-width setting.
	    key (str): Optional unique widget key.
	    height (int | str): Editor height setting.
	    hide_index (bool | None): Index visibility setting.
	    disabled (bool | List[str]): Editor or column disabled state.
	    column_config (Dict[str, object]): Optional specialized column configuration.
	    num_rows (str): Established fixed or dynamic row mode.

	Returns:
	    pd.DataFrame: Dataframe returned by ``st.data_editor``.
	"""
	throw_if( 'df', df )
	formatted_config = get_data_editor_column_config( df, column_config )
	widget_key = key if key else None
	return _streamlit_data_editor( df, use_container_width=use_container_width, key=widget_key,
		height=height, hide_index=hide_index, disabled=disabled,
		column_config=formatted_config, num_rows=num_rows )

def render_table( df: pd.DataFrame ) -> None:
	"""Render a dataframe safely in Streamlit.

	Purpose:
	    Displays the supplied dataframe in an interactive Streamlit data editor and falls back to
	    escaped HTML table output when Streamlit or PyArrow serialization cannot represent one or
	    more values.

	Args:
	    df (pd.DataFrame): Dataframe rendered in the application interface.

	Returns:
	    None: This function renders Streamlit output and does not return a value.
	"""
	if df is None:
		st.info( 'No data available.' )
		return
	
	try:
		render_data_editor( df, use_container_width=True )
		return
	except Exception:
		pass
	
	fallback_df = df.copy( )
	fallback_df = fallback_df.where( pd.notnull( fallback_df ), '' )
	
	for col in fallback_df.columns:
		fallback_df[ col ] = fallback_df[ col ].map(
			lambda x: x if isinstance( x, (str, int, float, bool) ) or x == '' else str( x ) )
	
	st.markdown( fallback_df.to_html( index=False, escape=True ), unsafe_allow_html=True )

def make_display_safe( df: pd.DataFrame ) -> pd.DataFrame:
	"""Convert dataframe values into display-safe text.

	Purpose:
	    Creates an independent dataframe copy, replaces null object values with empty strings, and
	    converts all remaining values to text for reliable rendering or export.

	Args:
	    df (pd.DataFrame): Dataframe normalized for display.

	Returns:
	    pd.DataFrame: Copy containing empty strings for null values and strings for all other values.
	"""
	display_df = df.copy( )
	
	for col in display_df.columns:
		display_df[ col ] = display_df[ col ].map( lambda x: '' if x is None else str( x ) )
	
	return display_df

def drop_table( table: str ) -> None:
	"""Drop a SQLite table when it exists.

	Purpose:
	    Removes the specified table from the configured SQLite database using a guarded
	    ``DROP TABLE IF EXISTS`` statement and commits the schema change.

	Args:
	    table (str): Table removed from the database.

	Returns:
	    None: This function modifies the SQLite schema and does not return a value.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""Create an index for a validated SQLite table column.

	Purpose:
	    Verifies that the supplied table and column exist, generates a sanitized index identifier,
	    creates the index when it is not already present, and commits the schema change.

	Args:
	    table (str): Existing SQLite table receiving the index.
	    column (str): Existing table column included in the index.

	Returns:
	    None: This function creates a SQLite index and does not return a value.

	Raises:
	    ValueError: Raised when the table or column does not exist.
	"""
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( 'Invalid table name.' )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( 'Invalid column name.' )
	
	# ------------------------------------------------------------------
	# Sanitize index name (identifier only)
	# ------------------------------------------------------------------
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ------------------------------------------------------------------
	# Create index safely (quote identifiers)
	# ------------------------------------------------------------------
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
	"""Filter a dataframe from user-selected conditions.

	Purpose:
	    Renders Streamlit controls for selecting a dataframe column, comparison operator, and filter
	    value, then applies the selected condition to the dataframe when a value is supplied.

	Args:
	    df (pd.DataFrame): Dataframe filtered by the selected user-interface condition.

	Returns:
	    pd.DataFrame: Filtered dataframe when a filter value is supplied; otherwise the original
	    dataframe.
	"""
	st.subheader( 'Advanced Filters' )
	conditions = [ ]
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=', '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: pd.DataFrame ):
	"""Render an interactive dataframe aggregation result.

	Purpose:
	    Identifies numeric dataframe columns, renders controls for selecting a column and aggregation
	    operation, calculates the requested count, sum, average, minimum, maximum, or median, and
	    displays the result as a Streamlit metric.

	Args:
	    df (pd.DataFrame): Dataframe containing numeric values available for aggregation.

	Returns:
	    None: This function renders Streamlit controls and an aggregation metric.
	"""
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: pd.DataFrame ) -> None:
	"""Render an interactive Plotly visualization.

	Purpose:
	    Prepares dataframe values for Plotly rendering, identifies columns containing usable numeric
	    values, presents supported chart selections, validates each chart's column requirements, and
	    renders histogram, bar, line, scatter, box, pie, or correlation visualizations without
	    passing pandas objects directly into Plotly traces.

	Args:
	    df (pd.DataFrame): Dataframe containing values available for visualization.

	Returns:
	    None: This function renders Streamlit controls, informational messages, and Plotly figures.
	"""
	st.subheader( 'Visualization Engine' )
	
	if df is None or df.empty:
		st.info( 'No data available.' )
		return
	
	df_plot = df.copy( )
	
	for col in df_plot.columns:
		if df_plot[ col ].dtype == object:
			df_plot[ col ] = df_plot[ col ].map( lambda x: '' if x is None else str( x ) )
	
	numeric_cols: List[ str ] = [ ]
	for col in df_plot.columns:
		series_num = pd.to_numeric( df_plot[ col ], errors='coerce' )
		if series_num.notna( ).any( ):
			numeric_cols.append( col )
	
	categorical_columns: List[ str ] = [ col for col in df_plot.columns if col not in numeric_cols ]
	
	chart = st.selectbox( 'Chart Type',
		[ 'Histogram', 'Bar', 'Line', 'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Histogram( x=values ) ] )
		fig.update_layout( xaxis_title=col, yaxis_title='Count' )
		render_mathy_plotly_chart( fig, 'data_visualization_histogram_chart',
			'mathy_data_histogram', f'Histogram — {col}', 500 )
	
	elif chart == 'Bar':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Bar( x=x_values, y=y_values ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		render_mathy_plotly_chart( fig, 'data_visualization_bar_chart',
			'mathy_data_bar', f'{y} by {x}', 500 )
	
	elif chart == 'Line':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='lines' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		render_mathy_plotly_chart( fig, 'data_visualization_line_chart',
			'mathy_data_line', f'{y} by {x}', 500 )
	
	elif chart == 'Scatter':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		x = st.selectbox( 'X', numeric_cols, key='viz_scatter_x' )
		y = st.selectbox( 'Y', numeric_cols, key='viz_scatter_y' )
		
		x_series = pd.to_numeric( df_plot[ x ], errors='coerce' )
		y_series = pd.to_numeric( df_plot[ y ], errors='coerce' )
		mask = x_series.notna( ) & y_series.notna( )
		
		x_values = x_series[ mask ].tolist( )
		y_values = y_series[ mask ].tolist( )
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='markers' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		render_mathy_plotly_chart( fig, 'data_visualization_scatter_chart',
			'mathy_data_scatter', f'{y} vs {x}', 500 )
	
	elif chart == 'Box':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols, key='viz_box_col' )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Box( y=values, name=col ) ] )
		fig.update_layout( yaxis_title=col )
		render_mathy_plotly_chart( fig, 'data_visualization_box_chart',
			'mathy_data_box', f'Box Plot — {col}', 500 )
	
	elif chart == 'Pie':
		if not categorical_columns:
			st.info( 'No categorical columns available.' )
			return
		
		col = st.selectbox( 'Category Column', categorical_columns )
		counts = df_plot[ col ].astype( str ).value_counts( )
		
		fig = go.Figure(
			data=[ go.Pie( labels=counts.index.tolist( ), values=counts.values.tolist( ) ) ] )
		render_mathy_plotly_chart( fig, 'data_visualization_pie_chart',
			'mathy_data_pie', f'Category Share — {col}', 500 )
	
	elif chart == 'Correlation':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		corr_df = pd.DataFrame( )
		for col in numeric_cols:
			corr_df[ col ] = pd.to_numeric( df_plot[ col ], errors='coerce' )
		
		corr = corr_df.corr( )
		
		fig = go.Figure( data=[ go.Heatmap( z=corr.values.tolist( ), x=corr.columns.tolist( ),
			y=corr.index.tolist( ) ) ] )
		render_mathy_plotly_chart( fig, 'data_visualization_correlation_chart',
			'mathy_data_correlation', 'Correlation Heatmap', 550 )

def convert_dataframe( table_name: str, df: pd.DataFrame ):
	"""Create a SQLite table from a dataframe schema.

	Purpose:
	    Maps each dataframe column type to a SQLite storage type, replaces spaces in column names
	    with underscores, builds a table-creation statement, and creates the table when it does not
	    already exist.

	Args:
	    table_name (str): Name assigned to the SQLite table.
	    df (pd.DataFrame): Dataframe whose columns define the SQLite table schema.

	Returns:
	    None: This function creates a SQLite table and does not return a value.
	"""
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}' )
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: pd.DataFrame ):
	"""Insert dataframe rows into a SQLite table.

	Purpose:
	    Copies the supplied dataframe, normalizes column names by replacing spaces with underscores,
	    builds a positional parameter statement, inserts all dataframe rows into the specified
	    table, and commits the transaction.

	Args:
	    table_name (str): Existing SQLite table receiving the dataframe rows.
	    df (pd.DataFrame): Dataframe containing rows inserted into the table.

	Returns:
	    None: This function inserts records into SQLite and does not return a value.
	"""
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""Map a pandas data type to a SQLite storage class.

	Purpose:
	    Converts a pandas or NumPy dtype representation into the SQLite storage class used when
	    generating table schemas. Integer and Boolean values map to ``INTEGER``, floating-point
	    values map to ``REAL``, and datetime, categorical, and unsupported values map to ``TEXT``.

	Args:
	    dtype (object): Pandas or NumPy data type evaluated for SQLite storage.

	Returns:
	    str: SQLite storage class associated with the supplied data type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if 'int' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if 'float' in dtype_str:
		return 'REAL'
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if 'bool' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if 'datetime' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if 'category' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return 'TEXT'

def create_custom_table( table_name: str, columns: list ) -> None:
	"""Create a SQLite table from explicit column definitions.

	Purpose:
	    Validates the table and column identifiers, builds SQLite column definitions from the
	    supplied metadata, applies primary-key, autoincrement, and not-null constraints, creates
	    the table when it does not already exist, and commits the schema change.

	Args:
	    table_name (str): Valid SQLite table identifier.
	    columns (list): Column-definition dictionaries containing ``name``, ``type``,
	        ``not_null``, ``primary_key``, and ``auto_increment`` values.

	Returns:
	    None: This function creates a SQLite table and does not return a value.

	Raises:
	    ValueError: Raised when the table name is missing or when a table or column identifier is
	        invalid.
	"""
	if not table_name:
		raise ValueError( 'Table name required.' )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( 'Invalid table name.' )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ 'name' ]
		col_type = col[ 'type' ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ 'primary_key' ]:
			definition += ' PRIMARY KEY'
			if col[ 'auto_increment' ] and col_type == 'INTEGER':
				definition += ' AUTOINCREMENT'
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""Determine whether a SQL statement is read-only.

	Purpose:
	    Normalizes and inspects a SQL statement, rejects multiple statements and write-oriented
	    keywords, and permits statements beginning with ``SELECT``, ``WITH``, ``EXPLAIN``, or
	    ``PRAGMA``.

	Args:
	    query (str): SQL text evaluated for read-only execution.

	Returns:
	    bool: ``True`` when the statement satisfies the configured read-only checks; otherwise
	    ``False``.
	"""
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ------------------------------------------------------------------
	# Block multiple statements
	# ------------------------------------------------------------------
	if ';' in q[ :-1 ]:
		return False
	
	# ------------------------------------------------------------------
	# Remove SQL comments
	# ------------------------------------------------------------------
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ------------------------------------------------------------------
	# Allowed starting keywords
	# ------------------------------------------------------------------
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ------------------------------------------------------------------
	# Block dangerous keywords anywhere
	# ------------------------------------------------------------------
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ', 'create ', 'attach ',
		'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""Convert text into a valid SQLite identifier.

	Purpose:
	    Replaces unsupported identifier characters with underscores, prefixes values that do not
	    begin with a letter or underscore, and rejects missing or unusable input.

	Args:
	    name (str): Source text converted into a SQLite-safe identifier.

	Returns:
	    str: Sanitized identifier containing letters, digits, and underscores.

	Raises:
	    ValueError: Raised when ``name`` is missing, is not a string, or cannot produce a usable
	        identifier.
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )
	
	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'
	
	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	"""Return index metadata for a SQLite table.

	Purpose:
	    Executes SQLite index-list introspection for the specified table and returns the index
	    metadata rows used by schema-management and user-interface workflows.

	Args:
	    table (str): SQLite table whose indexes are requested.

	Returns:
	    list: SQLite ``PRAGMA index_list`` result rows.
	"""
	with create_connection( ) as conn:
		rows = conn.execute( f'PRAGMA index_list("{table}");' ).fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	"""Add a column to an existing SQLite table.

	Purpose:
	    Sanitizes the requested column identifier, normalizes the supplied SQLite data type, adds
	    the column to the specified table, and commits the schema change.

	Args:
	    table (str): Existing SQLite table receiving the new column.
	    column (str): Column name sanitized before execution.
	    col_type (str): SQLite column type normalized to uppercase.

	Returns:
	    None: This function alters the SQLite table and does not return a value.

	Raises:
	    ValueError: Raised by ``create_identifier`` when the column name is invalid.
	"""
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute( f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	"""Rename a column in an existing SQLite table.

	Purpose:
	    Attempts to rename the selected column with SQLite's native ``ALTER TABLE`` operation.
	    When native renaming fails, rebuilds the table from schema metadata, preserves column
	    order and stored records, maps the renamed column into the rebuilt schema, and recreates
	    existing indexes with the updated column reference.

	Args:
	    table_name (str): Existing SQLite table containing the column.
	    old_name (str): Current column name.
	    new_name (str): Replacement column name.

	Returns:
	    None: This function modifies the SQLite schema and does not return a value.

	Raises:
	    ValueError: Raised when the table definition is unavailable or the source column does not
	        exist.
	"""
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute(
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (table_name,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (table_name,) ).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute(
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";' )
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def create_profile_table( table: str ):
	"""Create a column-level profile for a SQLite table.

	Purpose:
	    Reads the specified SQLite table into a dataframe and calculates each column's pandas data
	    type, null percentage, distinct-value percentage, and numeric minimum, maximum, and mean
	    values when applicable.

	Args:
	    table (str): SQLite table profiled through the configured database connection.

	Returns:
	    pd.DataFrame: Dataframe containing one profile record for each source column.
	"""
	df = read_table( table )
	declared_schema = create_declared_schema_map( create_schema( table ) )
	profiles = profile_dataframe_schema( df, declared_schema )
	df_analysis = create_typed_analysis_dataframe( df, profiles )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		profile = profiles[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = { 'column': col, 'dtype': str( series.dtype ),
			'inferred_type': profile[ 'inferred_dtype' ], 'role': profile[ 'analytical_role' ],
			'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
			'distinct_%': round( (distinct_count / total_rows) * 100, 2 ) if total_rows else 0, }
		
		if profile[ 'analytical_role' ] == 'numeric':
			numeric_series = df_analysis[ col ].dropna( )
			row[ 'min' ] = numeric_series.min( )
			row[ 'max' ] = numeric_series.max( )
			row[ 'mean' ] = numeric_series.mean( )
		else:
			row[ 'min' ]=None
			row[ 'max' ]=None
			row[ 'mean' ]=None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ) -> None:
	"""Remove a column from a SQLite table while preserving valid schema structure.

	Purpose:
	    Uses SQLite's native ``DROP COLUMN`` operation when supported. When native removal is not
	    available, rebuilds the table with a top-level SQL-definition parser so commas inside
	    constraints, defaults, and expressions do not corrupt the schema. Compatible indexes and
	    triggers are recreated after the transaction.

	Args:
	    table (str): Existing SQLite table containing the column.
	    column (str): Column removed from the table schema.

	Returns:
	    None: This function modifies the SQLite schema and does not return a value.
	"""
	throw_if( 'table', table )
	throw_if( 'column', column )

	with create_connection( ) as conn:
		schema = conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )
		columns = [ str( row[ 1 ] ) for row in schema ]
		if column not in columns:
			raise ValueError( f'Column "{column}" does not exist.' )
		if len( columns ) <= 1:
			raise ValueError( 'The final remaining table column cannot be removed.' )

		try:
			conn.execute( 'BEGIN' )
			conn.execute( f'ALTER TABLE "{table}" DROP COLUMN "{column}";' )
			conn.commit( )
			return
		except sqlite3.OperationalError:
			conn.rollback( )

		row = conn.execute(
			"SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,) ).fetchone( )
		if row is None or not row[ 0 ]:
			raise ValueError( 'Table definition not found.' )

		create_sql = str( row[ 0 ] )
		open_paren = create_sql.find( '(' )
		close_paren = create_sql.rfind( ')' )
		if open_paren < 0 or close_paren <= open_paren:
			raise ValueError( 'Malformed CREATE TABLE statement.' )

		definitions = split_sqlite_definitions( create_sql[ open_paren + 1: close_paren ] )
		table_constraint_tokens = { 'CONSTRAINT', 'PRIMARY', 'UNIQUE', 'CHECK', 'FOREIGN' }
		remaining_definitions: List[ str ] = [ ]
		column_removed = False

		for definition in definitions:
			stripped = definition.strip( )
			first_token = stripped.split( None, 1 )[ 0 ].strip( '"`[]' ).upper( ) if stripped else ''
			if first_token in table_constraint_tokens:
				if re.search( rf'(?i)(?:"{re.escape( column )}"|\b{re.escape( column )}\b)',
						stripped ):
					raise ValueError(
						f'Column "{column}" participates in a table-level constraint and cannot be '
						'dropped safely.' )
				remaining_definitions.append( definition )
				continue

			match = re.match( r'^\s*(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([^\s]+))',
				definition )
			definition_name = next(
				(group for group in match.groups( ) if group is not None), '' ) if match else ''
			if definition_name == column:
				column_removed = True
				continue
			remaining_definitions.append( definition )

		if not column_removed:
			raise ValueError( f'Column "{column}" was not found in the table definition.' )

		remaining_columns = [ name for name in columns if name != column ]
		temp_table = f'{table}__mathy_drop_rebuild'
		suffix = create_sql[ close_paren + 1: ].strip( ).rstrip( ';' )
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( remaining_definitions )})' + (
			f' {suffix};' if suffix else ';' )
		column_list = ', '.join( f'"{name}"' for name in remaining_columns )
		index_sql = [ item[ 0 ] for item in conn.execute(
			"SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL;",
			(table,) ).fetchall( ) if item[ 0 ] ]
		trigger_sql = [ item[ 0 ] for item in conn.execute(
			"SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=? AND sql IS NOT NULL;",
			(table,) ).fetchall( ) if item[ 0 ] ]

		try:
			conn.execute( 'BEGIN' )
			conn.execute( f'DROP TABLE IF EXISTS "{temp_table}";' )
			conn.execute( new_create_sql )
			conn.execute( f'INSERT INTO "{temp_table}" ({column_list}) '
				f'SELECT {column_list} FROM "{table}";' )
			conn.execute( f'DROP TABLE "{table}";' )
			conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table}";' )
			for statement in index_sql:
				if not re.search( rf'(?i)(?:"{re.escape( column )}"|\b{re.escape( column )}\b)',
						statement ):
					conn.execute( statement )
			for statement in trigger_sql:
				if not re.search( rf'(?i)(?:"{re.escape( column )}"|\b{re.escape( column )}\b)',
						statement ):
					conn.execute( statement )
			conn.commit( )
		except Exception:
			conn.rollback( )
			raise


def rename_table( old_name: str, new_name: str ) -> None:
	"""Rename an existing SQLite table.

	Purpose:
	    Attempts to rename the table with SQLite's native ``ALTER TABLE`` operation. When native
	    renaming fails, rebuilds the table from its original creation statement, copies all column
	    values, replaces the source table, and recreates existing indexes against the new table name.

	Args:
	    old_name (str): Current SQLite table name.
	    new_name (str): Replacement SQLite table name.

	Returns:
	    None: This function modifies the SQLite schema and does not return a value.

	Raises:
	    ValueError: Raised when the source table definition is unavailable or malformed.
	"""
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute( """
                            SELECT sql
                            FROM sqlite_master
                            WHERE type ='table' AND name =?
		                    """, (old_name,) ).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute( """
                                SELECT sql
                                FROM sqlite_master
                                WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
		                        """, (old_name,) ).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";' )
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

# -------- Data Management Utilities ---------------

def create_unique_upload_identifiers( values: list[ object ] ) -> list[ str ]:
	"""Create unique SQLite identifiers for uploaded names.

	Purpose:
	    Converts worksheet or dataframe-column names into valid SQLite identifiers and
	    appends numeric suffixes when multiple source names normalize to the same
	    identifier.

	Args:
	    values (list[object]): Source worksheet or dataframe-column names.

	Returns:
	    list[str]: Unique SQLite-safe identifiers in source order.
	"""
	identifiers: list[ str ] = [ ]
	identifier_counts: Dict[ str, int ] = { }
	
	for value in values:
		base_identifier = create_identifier( str( value ) )
		identifier_count = identifier_counts.get( base_identifier, 0 )
		identifier = base_identifier
		
		while identifier in identifiers:
			identifier_count += 1
			identifier = f'{base_identifier}_{identifier_count + 1}'
		
		identifier_counts[ base_identifier ] = identifier_count
		identifiers.append( identifier )
	
	return identifiers

def normalize_upload_sql_value( value: Any ) -> Any:
	"""Normalize an uploaded dataframe value for SQLite insertion.

	Purpose:
	    Converts pandas and NumPy scalar values into SQLite-compatible Python values,
	    serializes timestamps and timedeltas as text, converts missing values to ``None``,
	    and converts unsupported objects to strings.

	Args:
	    value (Any): Dataframe cell value prepared for SQLite insertion.

	Returns:
	    Any: SQLite-compatible scalar value or ``None``.
	"""
	if value is None:
		return None
	
	if isinstance( value, np.generic ):
		value = value.item( )
	
	if isinstance( value, pd.Timestamp ):
		return value.isoformat( )
	
	if isinstance( value, pd.Timedelta ):
		return str( value )
	
	try:
		if pd.isna( value ):
			return None
	except Exception:
		pass
	
	if isinstance( value, (str, int, float, bool, bytes) ):
		return value
	
	return str( value )

def get_visualization_dataframe( ) -> pd.DataFrame:
	"""Return the active dataframe for visualization.

	Purpose:
	    Retrieves the currently loaded dataset through the application's established data-access
	    helper and returns an independent copy for read-only visualization workflows.

	Returns:
	    pd.DataFrame: Copy of the active dataset, or an empty dataframe when no data is loaded.
	"""
	df_dataset = get_analysis_dataset( )
	if df_dataset is None or df_dataset.empty:
		return pd.DataFrame( )

	return df_dataset.copy( )

def get_visualization_columns( df_frame: pd.DataFrame ) -> Dict[ str, List[ str ] ]:
	"""Classify dataframe columns for visualization.

	Purpose:
	    Identifies numeric, categorical, datetime, boolean, and missing-value columns using the
	    authoritative analytical profiles synchronized from the active source dataframe. Stored
	    profiles are reused for a matching analytical copy so conversions do not change roles or
	    detach dataset-specific overrides.

	Args:
	    df_frame (pd.DataFrame): Dataframe whose columns are classified.

	Returns:
	    Dict[str, List[str]]: Column names grouped by visualization-compatible data type.
	"""
	throw_if( 'df_frame', df_frame )
	stored_profiles = st.session_state.get( 'column_profiles', { } )
	if isinstance( stored_profiles, dict ) and list( stored_profiles.keys( ) ) == list(
			df_frame.columns ):
		profiles = stored_profiles
	else:
		profiles = profile_dataframe_schema( df_frame )
	numeric_columns = [ column for column, profile in profiles.items( ) if
		profile[ 'analytical_role' ] == 'numeric' ]
	datetime_columns = [ column for column, profile in profiles.items( ) if
		profile[ 'analytical_role' ] == 'datetime' ]
	boolean_columns = [ column for column, profile in profiles.items( ) if
		profile[ 'inferred_dtype' ] == 'boolean' ]
	categorical_columns = [ column for column, profile in profiles.items( ) if
		profile[ 'analytical_role' ] in ('categorical', 'ordinal') ]

	missing_columns = [ column for column in df_frame.columns
		if df_frame[ column ].isna( ).any( ) ]

	return {
		'numeric': numeric_columns,
		'categorical': categorical_columns,
		'datetime': datetime_columns,
		'boolean': boolean_columns,
		'missing': missing_columns
	}

def get_usable_categorical_columns( df_frame: pd.DataFrame,
	categorical_columns: List[ str ]=None ) -> List[ str ]:
	"""Return categorical columns containing comparative information.

	Purpose:
	    Filters categorical and ordinal visualization fields so columns containing only one
	    populated category are not offered for comparisons, distributions, hierarchies, flows,
	    or heatmaps. The underlying analytical schema remains unchanged for non-visualization
	    workflows.

	Args:
	    df_frame (pd.DataFrame): Dataframe containing categorical values evaluated for variation.
	    categorical_columns (List[str]): Optional categorical columns to evaluate. When omitted,
	        the authoritative visualization classification is used.

	Returns:
	    List[str]: Categorical columns containing at least two distinct populated values.
	"""
	throw_if( 'df_frame', df_frame )
	columns = categorical_columns if categorical_columns is not None else \
		get_visualization_columns( df_frame )[ 'categorical' ]
	return [ column for column in columns if column in df_frame.columns and
		df_frame[ column ].nunique( dropna=True ) > 1 ]

def aggregate_visualization_dataframe( df_frame: pd.DataFrame, group_columns: List[ str ],
	value_column: str, aggregation: str ) -> pd.DataFrame:
	"""Aggregate a dataframe for visualization.

	Purpose:
	    Converts the selected measure to numeric form, removes unusable observations, groups values
	    by the requested dimensions, and applies one supported aggregation consistently across the
	    advanced visualization modes.

	Args:
	    df_frame (pd.DataFrame): Source dataframe containing grouping dimensions and a numeric measure.
	    group_columns (List[str]): Ordered categorical, ordinal, or datetime grouping columns.
	    value_column (str): Numeric measure aggregated for visualization.
	    aggregation (str): Supported aggregation display name.

	Returns:
	    pd.DataFrame: Aggregated dataframe containing the grouping columns and numeric measure.
	"""
	throw_if( 'df_frame', df_frame )
	throw_if( 'group_columns', group_columns )
	throw_if( 'value_column', value_column )
	throw_if( 'aggregation', aggregation )
	aggregation_map = { 'Sum': 'sum', 'Mean': 'mean', 'Median': 'median',
		'Minimum': 'min', 'Maximum': 'max', 'Count': 'count' }
	if aggregation not in aggregation_map:
		raise ValueError( f'Unsupported aggregation: {aggregation}' )
	columns = list( dict.fromkeys( group_columns + [ value_column ] ) )
	df_result = df_frame[ columns ].copy( )
	df_result[ value_column ] = pd.to_numeric( df_result[ value_column ], errors='coerce' )
	df_result = df_result.dropna( subset=group_columns + [ value_column ] )
	if df_result.empty:
		return pd.DataFrame( columns=columns )
	df_result = df_result.groupby( group_columns, observed=True, dropna=False )[ value_column ].agg(
		aggregation_map[ aggregation ] ).reset_index( )
	return df_result

def create_sankey_from_stages( df_frame: pd.DataFrame, stage_columns: List[ str ],
	value_column: str='', aggregation: str='Count' ) -> go.Figure:
	"""Create a Sankey diagram from ordered categorical stages.

	Purpose:
	    Aggregates transitions between adjacent categorical stages and creates one Sankey diagram
	    whose link widths represent observation counts or an explicitly selected numeric measure.

	Args:
	    df_frame (pd.DataFrame): Source dataframe containing stage fields and optional measure values.
	    stage_columns (List[str]): Ordered categorical fields defining adjacent lifecycle stages.
	    value_column (str): Optional numeric measure used as link weight.
	    aggregation (str): Aggregation applied when a numeric measure is supplied.

	Returns:
	    go.Figure: Sankey diagram representing transitions across selected stages.
	"""
	throw_if( 'df_frame', df_frame )
	throw_if( 'stage_columns', stage_columns )
	if len( stage_columns ) < 2:
		raise ValueError( 'At least two stage columns are required.' )

	labels: List[ str ] = [ ]
	label_index: Dict[ str, int ] = { }
	sources: List[ int ] = [ ]
	targets: List[ int ] = [ ]
	values: List[ float ] = [ ]

	for stage_index in range( len( stage_columns ) - 1 ):
		left_column = stage_columns[ stage_index ]
		right_column = stage_columns[ stage_index + 1 ]
		columns = [ left_column, right_column ] + ([ value_column ] if value_column else [ ])
		df_stage = df_frame[ columns ].copy( ).dropna( subset=[ left_column, right_column ] )
		if df_stage.empty:
			continue
		if value_column:
			df_stage[ value_column ] = pd.to_numeric( df_stage[ value_column ], errors='coerce' )
			df_stage = df_stage.dropna( subset=[ value_column ] )
			if df_stage.empty:
				continue
			df_links = aggregate_visualization_dataframe( df_stage, [ left_column, right_column ],
				value_column, aggregation )
			weight_column = value_column
		else:
			df_links = df_stage.groupby( [ left_column, right_column ], observed=True,
				dropna=False ).size( ).reset_index( name='Weight' )
			weight_column = 'Weight'

		for _, row in df_links.iterrows( ):
			left_label = f'{left_column}: {row[ left_column ]}'
			right_label = f'{right_column}: {row[ right_column ]}'
			for label in (left_label, right_label):
				if label not in label_index:
					label_index[ label ] = len( labels )
					labels.append( label )
			sources.append( label_index[ left_label ] )
			targets.append( label_index[ right_label ] )
			values.append( abs( float( row[ weight_column ] ) ) )

	figure = go.Figure( data=[ go.Sankey(
		node={ 'label': labels, 'pad': 15, 'thickness': 18 },
		link={ 'source': sources, 'target': targets, 'value': values } ) ] )
	return figure

def render_visualization_metric_styles( ) -> None:
	"""Render the established Mathy metric typography.

	Purpose:
	    Applies the metric label, value, and vertical-padding settings used by the original
	    analytical modes to the newer Plotly visualization modes.

	Returns:
	    None: This function renders scoped page styling and does not return a value.
	"""
	st.markdown( """
		<style>
		[data-testid="stMetricLabel"] p {
			font-size: 0.85rem;
		}

		[data-testid="stMetricValue"] {
			font-size: 1.05rem;
		}

		[data-testid="stMetric"] {
			padding-top: 0.10rem;
			padding-bottom: 0.10rem;
		}
		</style>
		""", unsafe_allow_html=True )

def apply_mathy_plotly_theme( figure: go.Figure, title: str='',
		height: int=500 ) -> go.Figure:
	"""Apply the Mathy Plotly presentation theme.

	Purpose:
	    Applies consistent typography, spacing, transparent backgrounds, hover styling, legends,
	    axes, and responsive dimensions to a Plotly figure before Streamlit renders it.

	Args:
	    figure (go.Figure): Plotly figure receiving the shared presentation settings.
	    title (str): Optional chart title displayed above the plotting area.
	    height (int): Figure height in pixels.

	Returns:
	    go.Figure: The themed Plotly figure.
	"""
	throw_if( 'figure', figure )
	figure.update_layout(
		template='plotly_dark',
		title={ 'text': title, 'x': 0.01, 'xanchor': 'left' },
		height=height,
		autosize=True,
		paper_bgcolor='#171A1F',
		plot_bgcolor='#1B2028',
		font={ 'family': 'Arial, sans-serif', 'size': 13, 'color': '#E2E8F0' },
		margin={ 'l': 55, 'r': 30, 't': 70 if title else 35, 'b': 55 },
		legend={ 'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02,
			'xanchor': 'right', 'x': 1.0 },
		hoverlabel={ 'bgcolor': '#0F172A', 'font_size': 13,
			'font_family': 'Arial, sans-serif' },
		colorway=[ '#38BDF8', '#A78BFA', '#2DD4BF', '#F59E0B', '#F472B6', '#60A5FA',
			'#34D399', '#FB7185' ] )

	figure.update_xaxes( showgrid=True, gridcolor='rgba(148,163,184,0.16)', zeroline=False,
		showline=True, linecolor='rgba(148,163,184,0.35)' )
	figure.update_yaxes( showgrid=True, gridcolor='rgba(148,163,184,0.16)', zeroline=False,
		showline=True, linecolor='rgba(148,163,184,0.35)' )
	return figure

def add_statistical_reference_lines( figure: go.Figure, mean_value: float,
		median_value: float ) -> go.Figure:
	"""Add non-overlapping mean and median references to a Plotly figure.

	Purpose:
	    Draws mean and median reference lines without placing collision-prone text annotations
	    inside the plotting area. The formatted values are exposed through distinct legend entries.

	Args:
	    figure (go.Figure): Plotly figure receiving the statistical reference lines.
	    mean_value (float): Arithmetic mean represented by the dashed amber line.
	    median_value (float): Median represented by the dotted teal line.

	Returns:
	    go.Figure: Plotly figure containing mean and median lines and legend entries.
	"""
	throw_if( 'figure', figure )
	figure.add_vline( x=mean_value, line_dash='dash', line_color='#F59E0B' )
	figure.add_vline( x=median_value, line_dash='dot', line_color='#2DD4BF' )
	figure.add_trace( go.Scatter( x=[ None ], y=[ None ], mode='lines',
		name=f'Mean: {mean_value:,.4g}', line={ 'color': '#F59E0B', 'dash': 'dash',
			'width': 2 }, hoverinfo='skip' ) )
	figure.add_trace( go.Scatter( x=[ None ], y=[ None ], mode='lines',
		name=f'Median: {median_value:,.4g}', line={ 'color': '#2DD4BF', 'dash': 'dot',
			'width': 2 }, hoverinfo='skip' ) )
	return figure

def render_mathy_plotly_chart( figure: go.Figure, key: str, filename: str,
		title: str='', height: int=500 ) -> None:
	"""Render a themed Plotly chart.

	Purpose:
	    Applies the shared Mathy Plotly theme and renders the figure with a responsive toolbar,
	    high-resolution PNG export, zooming, panning, autoscaling, and no Plotly branding.

	Args:
	    figure (go.Figure): Plotly figure rendered in Streamlit.
	    key (str): Unique Streamlit element key assigned to the chart.
	    filename (str): Base filename used by the Plotly image-download control.
	    title (str): Optional chart title.
	    height (int): Figure height in pixels.

	Returns:
	    None: This function renders the chart and does not return a value.
	"""
	throw_if( 'figure', figure )
	throw_if( 'key', key )
	throw_if( 'filename', filename )
	apply_mathy_plotly_theme( figure, title, height )
	chart_config = {
		'displaylogo': False,
		'responsive': True,
		'scrollZoom': True,
		'toImageButtonOptions': { 'format': 'png', 'filename': filename, 'scale': 2 }
	}
	st.plotly_chart( figure, use_container_width=True, key=key, config=chart_config )

def create_correlation_pairs( df_correlation: pd.DataFrame ) -> pd.DataFrame:
	"""Create a ranked table of unique correlation pairs.

	Purpose:
	    Converts a square correlation matrix into unique variable pairs, removes diagonal and
	    symmetric duplicates, and classifies each relationship by absolute strength.

	Args:
	    df_correlation (pd.DataFrame): Square correlation matrix.

	Returns:
	    pd.DataFrame: Ranked correlation pairs with signed and absolute correlation values.
	"""
	throw_if( 'df_correlation', df_correlation )
	rows: List[ Dict[ str, Any ] ] = [ ]
	columns = df_correlation.columns.tolist( )
	for left_index, left_column in enumerate( columns ):
		for right_column in columns[ left_index + 1: ]:
			correlation = float( df_correlation.loc[ left_column, right_column ] )
			absolute = abs( correlation )
			if absolute >= 0.70:
				strength = 'Strong'
			elif absolute >= 0.40:
				strength = 'Moderate'
			elif absolute >= 0.20:
				strength = 'Weak'
			else:
				strength = 'Very Weak'

			rows.append( { 'Variable 1': left_column, 'Variable 2': right_column,
				'Correlation': correlation, 'Absolute Correlation': absolute,
				'Strength': strength } )

	df_pairs = pd.DataFrame( rows )
	if not df_pairs.empty:
		df_pairs = df_pairs.sort_values( 'Absolute Correlation', ascending=False,
			ignore_index=True )
	return df_pairs

def create_qq_figure( values: np.ndarray, variable: str ) -> go.Figure:
	"""Create a Plotly quantile-quantile figure.

	Purpose:
	    Calculates theoretical normal quantiles and ordered sample values, then constructs an
	    interactive Q-Q plot with the fitted reference line used by Mathy's normality views.

	Args:
	    values (np.ndarray): One-dimensional numeric observations.
	    variable (str): Variable name used in chart labels and hover content.

	Returns:
	    go.Figure: Plotly Q-Q figure containing observed quantiles and the fitted reference line.
	"""
	throw_if( 'values', values )
	throw_if( 'variable', variable )
	(theoretical, ordered), (slope, intercept, _) = stats.probplot(
		np.asarray( values, dtype=float ), dist='norm' )
	figure = go.Figure( )
	figure.add_trace( go.Scatter( x=theoretical, y=ordered, mode='markers', name='Observed',
		marker={ 'color': '#38BDF8', 'size': 8, 'opacity': 0.78 },
		hovertemplate='Theoretical: %{x:.4f}<br>Observed: %{y:.4f}<extra></extra>' ) )
	figure.add_trace( go.Scatter( x=theoretical, y=(slope * theoretical) + intercept,
		mode='lines', name='Normal Reference', line={ 'color': '#F59E0B', 'width': 3 } ) )
	figure.update_xaxes( title_text='Theoretical Quantiles' )
	figure.update_yaxes( title_text=f'Ordered Values — {variable}' )
	return figure

def create_cluster_figure( df_results: pd.DataFrame, feature_columns: List[ str ],
		title: str, df_centroids: Optional[ pd.DataFrame ]=None,
		centroid_label: str='Centroids' ) -> go.Figure:
	"""Create an interactive cluster-assignment figure.

	Purpose:
	    Plots two selected clustering features, colors observations by assigned cluster, and
	    optionally overlays estimator centroids, exemplars, or subcluster centers.

	Args:
	    df_results (pd.DataFrame): Clustered observations containing a `Cluster` column.
	    feature_columns (List[str]): Two numeric feature columns plotted on the axes.
	    title (str): Chart title describing the clustering estimator.
	    df_centroids (Optional[pd.DataFrame]): Optional center coordinates to overlay.
	    centroid_label (str): Legend label assigned to the optional center trace.

	Returns:
	    go.Figure: Plotly cluster-assignment figure.
	"""
	throw_if( 'df_results', df_results )
	throw_if( 'feature_columns', feature_columns )
	throw_if( 'title', title )
	x_column, y_column = feature_columns[ 0 ], feature_columns[ 1 ]
	df_plot = df_results.copy( )
	df_plot[ 'Cluster Label' ] = df_plot[ 'Cluster' ].astype( str )
	figure = px.scatter( df_plot, x=x_column, y=y_column, color='Cluster Label',
		hover_data=df_plot.columns.tolist( ), opacity=0.78 )
	figure.update_traces( marker={ 'size': 9 } )
	if df_centroids is not None and not df_centroids.empty and \
			x_column in df_centroids.columns and y_column in df_centroids.columns:
		figure.add_trace( go.Scatter( x=df_centroids[ x_column ], y=df_centroids[ y_column ],
			mode='markers', name=centroid_label, marker={ 'symbol': 'x', 'size': 16,
				'color': '#F59E0B', 'line': { 'width': 2, 'color': '#F8FAFC' } },
			hovertemplate=f'{centroid_label}<br>{x_column}: %{{x:.4f}}<br>'
				f'{y_column}: %{{y:.4f}}<extra></extra>' ) )
	figure.update_layout( title={ 'text': title } )
	return figure

def create_forecast_figure( series: np.ndarray, forecast: np.ndarray, variable: str,
		title: str, forecast_label: str='Forecast' ) -> go.Figure:
	"""Create an observed-versus-forecast Plotly figure.

	Purpose:
	    Plots the complete observed series and appends a forward forecast trace beginning after
	    the final observed period while preserving the original period-based axis convention.

	Args:
	    series (np.ndarray): Historical ordered observations.
	    forecast (np.ndarray): Forward projected values.
	    variable (str): Series name used as the y-axis label.
	    title (str): Forecast chart title.
	    forecast_label (str): Legend label assigned to the projected trace.

	Returns:
	    go.Figure: Plotly figure containing observed and forecast traces.
	"""
	throw_if( 'series', series )
	throw_if( 'forecast', forecast )
	throw_if( 'variable', variable )
	throw_if( 'title', title )
	observed_values = np.asarray( series, dtype=float ).reshape( -1 )
	forecast_values = np.asarray( forecast, dtype=float ).reshape( -1 )
	observed_index = np.arange( len( observed_values ) )
	forecast_index = np.arange( len( observed_values ),
		len( observed_values ) + len( forecast_values ) )
	figure = go.Figure( )
	figure.add_trace( go.Scatter( x=observed_index, y=observed_values,
		mode='lines', name='Observed', line={ 'color': '#38BDF8', 'width': 2.5 } ) )
	figure.add_trace( go.Scatter( x=forecast_index, y=forecast_values,
		mode='lines+markers', name=forecast_label,
		line={ 'color': '#F472B6', 'width': 3, 'dash': 'dash' },
		marker={ 'size': 7 } ) )
	figure.update_xaxes( title_text='Period' )
	figure.update_yaxes( title_text=variable )
	figure.update_layout( title={ 'text': title } )
	return figure

def create_split_figure( df_split_results: pd.DataFrame ) -> go.Figure:
	"""Create an interactive time-series validation-window figure.

	Purpose:
	    Visualizes each generated split on its own horizontal lane, distinguishing training and
	    testing periods while retaining period values in hover details.

	Args:
	    df_split_results (pd.DataFrame): Long-form split assignments containing split, period,
	        value, and partition columns.

	Returns:
	    go.Figure: Plotly validation-window assignment figure.
	"""
	throw_if( 'df_split_results', df_split_results )
	figure = px.scatter( df_split_results, x='Period', y='Split', color='Partition',
		hover_data=[ 'Value' ], color_discrete_map={ 'Train': '#38BDF8', 'Test': '#F472B6' },
		category_orders={ 'Partition': [ 'Train', 'Test' ] } )
	figure.update_traces( marker={ 'symbol': 'square', 'size': 10 } )
	figure.update_yaxes( autorange='reversed', dtick=1 )
	return figure

# ============================================
# Page Configuration
# ============================================
st.set_page_config( page_title='Mathy', layout='wide', page_icon=cfg.FAVICON,
	initial_sidebar_state='expanded' )

st.logo( image=cfg.LOGO, size='large', link=cfg.REPO_URL )
pd.options.display.float_format = '{:,.2f}'.format


@st.dialog( 'Operation Successful' )
def render_database_success_dialog( message: str ) -> None:
	"""Render a queued database-operation success dialog.

	Purpose:
	    Displays one verified database success message after a rerun while preserving the operation
	    context that triggered the acknowledgement.

	Args:
	    message (str): Database operation result displayed to the user.

	Returns:
	    None: This function renders a Streamlit dialog.
	"""
	throw_if( 'message', message )
	st.success( message )


pending_database_result = st.session_state.get( 'database_operation_result', None )
if pending_database_result:
	st.session_state[ 'database_operation_result' ]=None
	render_database_success_dialog( str( pending_database_result ) )

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
	st.sidebar.divider( )
	st.subheader( 'Data Source' )
	with st.expander( 'Select Source', expanded=False ):
		src = [ 'Default Data', 'Database Data', 'Custom Data' ]
		source = st.selectbox( label='Select Source', options=src, key='source_selectbox' )
		df_loaded: pd.DataFrame | None = None
		loaded_original: pd.DataFrame | None = None
		loaded_source_signature: tuple | None = None
		loaded_source_message = ''
		loaded_source_kind = ''
		loaded_database_table = ''
		loaded_declared_schema: Dict[ str, str ] = { }
		if source == 'Default Data':
			df_loaded = pd.read_excel( cfg.DEFAULT_DATA )
			loaded_original = df_loaded.copy( )
			loaded_source_signature = ('Default Data', str( cfg.DEFAULT_DATA ))
			loaded_source_message = 'Loaded Default Dataset'
			loaded_source_kind = 'default'
		
		elif source == 'Database Data':
			try:
				table_options = [ table for table in list_tables( ) if not table.startswith( 'sqlite_' ) ]
				if table_options:
					active_table = st.session_state.get( 'active_database_table', '' )
					if active_table in table_options and st.session_state.get(
							'database_table_selectbox', None ) not in table_options:
						st.session_state[ 'database_table_selectbox' ] = active_table
					selected_table = st.selectbox( label='Select Database Table',
						options=table_options, key='database_table_selectbox' )
					if selected_table:
						df_loaded = read_table( selected_table )
						loaded_original = df_loaded.copy( )
						loaded_source_signature = ('Database Data', selected_table)
						loaded_source_message = f'Loaded Database Table: {selected_table}'
						loaded_source_kind = 'database'
						loaded_database_table = selected_table
						loaded_declared_schema = create_declared_schema_map(
							create_schema( selected_table ) )
				else:
					st.warning( 'No tables were found in the database.' )
			except Exception as ex:
				st.error( f'Error loading database data: {ex}' )
		
		elif source == 'Custom Data':
			uploaded = st.file_uploader( label='Upload Spreadsheet', type=[ 'xlsx', 'xls', 'csv' ],
				key='source_uploader' )
			if uploaded is not None:
				try:
					file_bytes = uploaded.getvalue( )
					upload_signature = (uploaded.name, len( file_bytes ), hash( file_bytes ))
					if st.session_state.get( 'custom_upload_signature', None ) != upload_signature:
						uploaded.seek( 0 )
						file_stem = Path( uploaded.name ).stem
						if uploaded.name.lower( ).endswith( ('.xlsx', '.xls') ):
							sheets = pd.read_excel( uploaded, sheet_name=None )
							if not sheets:
								raise ValueError( 'The uploaded workbook does not contain any worksheets.' )
							source_names = list( sheets.keys( ) )
							requested_names = ([ file_stem ] if len( source_names ) == 1 else
								[ f'{file_stem}_{sheet_name}' for sheet_name in source_names ])
							table_names = create_available_table_names( requested_names, list_tables( ) )
							df_tables = { table_name: sheets[ sheet_name ] for sheet_name, table_name
								in zip( source_names, table_names ) }
						else:
							df_csv = pd.read_csv( uploaded )
							table_name = create_available_table_names( [ file_stem ], list_tables( ) )[ 0 ]
							df_tables = { table_name: df_csv }

						persisted_tables = write_dataframe_tables_to_database( df_tables, overwrite=False )
						first_table = next( iter( persisted_tables ) )
						st.session_state[ 'custom_upload_signature' ] = upload_signature
						st.session_state[ 'custom_upload_tables' ] = persisted_tables
						st.session_state[ 'custom_upload_filename' ] = uploaded.name
						activate_database_table( first_table )
						queue_database_success(
							f'Imported {len( persisted_tables ):,} table(s) from "{uploaded.name}". '
							f'"{first_table}" is now the active table.' )
						st.rerun( )

					persisted_tables = st.session_state.get( 'custom_upload_tables', { } )
					available_tables = [ table for table in persisted_tables if table in list_tables( ) ]
					if available_tables:
						active_custom_table = st.session_state.get( 'active_database_table', '' )
						selected_custom_table = active_custom_table if active_custom_table in available_tables else available_tables[ 0 ]
						df_loaded = read_table( selected_custom_table )
						loaded_original = df_loaded.copy( )
						loaded_source_signature = ('Database Data', selected_custom_table)
						loaded_source_message = f'Loaded Database Table: {selected_custom_table}'
						loaded_source_kind = 'database'
						loaded_database_table = selected_custom_table
						loaded_declared_schema = create_declared_schema_map(
							create_schema( selected_custom_table ) )
				except Exception as ex:
					st.session_state[ 'custom_upload_signature' ]=None
					st.error( f'Unable to import spreadsheet: {ex}' )
			else:
				st.info( 'Upload a spreadsheet to load data.' )

		active_source_signature = st.session_state.get( 'active_source_signature', None )
		if has_loaded_dataset( df_loaded ) and (not has_loaded_dataset(
				st.session_state.get( 'df_dataset', None ) ) or
				loaded_source_signature != active_source_signature):
			st.session_state[ 'active_source_kind' ] = loaded_source_kind
			st.session_state[ 'active_database_table' ] = loaded_database_table
			st.session_state[ 'active_declared_schema' ] = loaded_declared_schema.copy( )
			store_loaded_dataset( df_loaded, loaded_original )
			st.session_state[ 'active_source_signature' ] = loaded_source_signature
			log_step( loaded_source_message )
	
	def get_visualization_modes( df_frame: pd.DataFrame | None ) -> list[ str ]:
		"""Return visualization modes supported by the loaded dataframe.

		Purpose:
		    Examines authoritative visualization column groups and actual categorical variation,
		    then exposes only visualization modes whose minimum data requirements are satisfied.
		    Single-valued categorical fields remain part of the analytical schema but are excluded
		    from comparative visualization capability checks.

		Args:
		    df_frame (pd.DataFrame | None): Currently loaded dataframe evaluated for visualization
		        compatibility.

		Returns:
		    list[str]: Visualization modes supported by the available dataframe schema and values.
		"""
		if df_frame is None or df_frame.empty:
			return [ ]

		column_groups = get_visualization_columns( df_frame )
		numeric_columns = column_groups[ 'numeric' ]
		datetime_columns = column_groups[ 'datetime' ]
		categorical_columns = column_groups[ 'categorical' ]
		usable_categorical_columns = get_usable_categorical_columns( df_frame,
			categorical_columns )

		visualization_modes = [ 'Data Overview' ]

		if numeric_columns:
			visualization_modes.append( 'Numeric Distributions' )
			visualization_modes.append( 'KPI Summary' )

		if len( numeric_columns ) >= 2:
			visualization_modes.append( 'Correlation Analysis' )
			visualization_modes.append( 'Scatter Analysis' )

		if usable_categorical_columns:
			visualization_modes.append( 'Categorical Distributions' )

		if numeric_columns and usable_categorical_columns:
			visualization_modes.append( 'Category Comparisons' )

		if len( numeric_columns ) >= 2 or (numeric_columns and usable_categorical_columns):
			visualization_modes.append( 'Variance & Decomposition' )

		if datetime_columns and numeric_columns:
			visualization_modes.append( 'Time-Series Visualization' )

		if len( usable_categorical_columns ) >= 2 and numeric_columns:
			visualization_modes.append( 'Composition & Allocation' )

		if len( usable_categorical_columns ) >= 2 or (
				usable_categorical_columns and datetime_columns and numeric_columns):
			visualization_modes.append( 'Pattern Heatmaps' )

		if len( usable_categorical_columns ) >= 2 or (
				len( datetime_columns ) >= 2 and usable_categorical_columns):
			visualization_modes.append( 'Lifecycle & Flow' )

		if df_frame.isna( ).any( ).any( ):
			visualization_modes.append( 'Missing Data Visualization' )

		return visualization_modes

	def handle_ml_mode_change( ) -> None:
		"""Activate the selected machine-learning mode.

		Purpose:
		    Reads the selected machine-learning mode from Streamlit session state, clears inactive
		    data-management and visualization selections, updates the active application mode, and
		    resets classification or regression state when those workflows are selected.

		Returns:
		    None: This function updates mode-selection and model-workflow values in Streamlit
		    session state.
		"""
		selected_mode = st.session_state.get( 'ml_mode_radio', None )
		
		if selected_mode is None:
			return
		
		st.session_state[ 'db_mode_radio' ]=None
		st.session_state[ 'visualization_mode_radio' ]=None
		st.session_state[ 'active_mode' ] = selected_mode
		
		if selected_mode == 'Classification Models':
			reset_classification_mode_state( )
		elif selected_mode == 'Regression Models':
			reset_regression_mode_state( )
	
	def handle_db_mode_change( ) -> None:
		"""Activate the selected data-management mode.

		Purpose:
		    Reads the selected data-management mode from Streamlit session state, clears inactive
		    machine-learning and visualization selections, and updates the active application mode.

		Returns:
		    None: This function updates mode-selection values in Streamlit session state.
		"""
		selected_mode = st.session_state.get( 'db_mode_radio', None )
		
		if selected_mode is None:
			return
		
		st.session_state[ 'ml_mode_radio' ]=None
		st.session_state[ 'visualization_mode_radio' ]=None
		st.session_state[ 'active_mode' ] = selected_mode
	
	def handle_visualization_mode_change( ) -> None:
		"""Activate the selected visualization mode.

		Purpose:
		    Reads the selected visualization mode from Streamlit session state, clears inactive
		    machine-learning and data-management selections, and updates the active application
		    mode.

		Returns:
		    None: This function updates mode-selection values in Streamlit session state.
		"""
		selected_mode = st.session_state.get( 'visualization_mode_radio', None )
		
		if selected_mode is None:
			return
		
		st.session_state[ 'ml_mode_radio' ]=None
		st.session_state[ 'db_mode_radio' ]=None
		st.session_state[ 'active_mode' ] = selected_mode
		
	# ------- Available Modes
	ml_modes = list( dict.fromkeys( cfg.ML_MODE ) )
	db_modes = list( cfg.DB_MODE )
	df_loaded = st.session_state.get( 'df_dataset', None )
	visualization_modes = get_visualization_modes( df_loaded )
	
	# ------- Mode State Initialization
	if 'active_mode' not in st.session_state:
		st.session_state[ 'active_mode' ] = ml_modes[ 0 ] if ml_modes else None
	
	if 'ml_mode_radio' not in st.session_state:
		st.session_state[ 'ml_mode_radio' ] = st.session_state[ 'active_mode' ]
	
	if 'db_mode_radio' not in st.session_state:
		st.session_state[ 'db_mode_radio' ]=None
	
	if 'visualization_mode_radio' not in st.session_state:
		st.session_state[ 'visualization_mode_radio' ]=None
	
	available_modes = ml_modes + db_modes + visualization_modes
	if st.session_state[ 'active_mode' ] not in available_modes:
		st.session_state[ 'active_mode' ] = ml_modes[ 0 ] if ml_modes else None
		st.session_state[ 'ml_mode_radio' ] = st.session_state[ 'active_mode' ]
		st.session_state[ 'db_mode_radio' ]=None
		st.session_state[ 'visualization_mode_radio' ]=None
	
	if st.session_state.get( 'ml_mode_radio', None ) not in ml_modes:
		st.session_state[ 'ml_mode_radio' ]=None
	
	if st.session_state.get( 'db_mode_radio', None ) not in db_modes:
		st.session_state[ 'db_mode_radio' ]=None
	
	if st.session_state.get( 'visualization_mode_radio', None ) not in visualization_modes:
		st.session_state[ 'visualization_mode_radio' ]=None
	
	# ------- Data Visualization Selection Mode
	st.sidebar.divider( )
	st.subheader( 'Data Visualization' )
	with st.expander( 'Select Method', expanded=True ):
		if not visualization_modes:
			st.info( 'Load a dataset to enable visualization modes.' )
		else:
			st.radio( label='Select', options=visualization_modes, index=None,
				key='visualization_mode_radio', on_change=handle_visualization_mode_change )
	
	# ------- Machine Learning Selection Mode
	st.sidebar.divider( )
	st.subheader( 'Machine Learning' )
	with st.expander( 'Select Model', expanded=True ):
		st.radio( label='Select', options=ml_modes, index=None,
			key='ml_mode_radio', on_change=handle_ml_mode_change )
	
	# ------- Data Management Selection Mode
	st.sidebar.divider( )
	st.subheader( 'Data Management' )
	with st.expander( 'Select Mode', expanded=True ):
		st.radio( label='Select', options=db_modes, index=None, key='db_mode_radio',
			on_change=handle_db_mode_change )
	
	# ------- Active Application Mode
	mode = st.session_state[ 'active_mode' ]

style_subheaders( )

# ============================================
# DATA PROFILING MODE
# ============================================
if mode == 'Data Profile':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Profile' ] )
		st.divider( )
		
		df_dataset = get_loaded_dataset( )
		if df_dataset is None:
			st.info( 'No data loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		numeric_columns, categorical_columns = synchronize_dataset_columns( df_dataset )
		schema = st.session_state[ 'column_schema' ].copy( )
		df_analysis = st.session_state[ 'df_analysis' ].copy( )
		
		# -------------------------------------------------------------------------------------
		# DATASET DISPLAY
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Data' )
		render_table( df_analysis )
		
		# -------------------------------------------------------------------------------------
		# SCHEMA METRICS
		# -------------------------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Types' )
		type_counts = pd.Series( schema ).value_counts( )
		m1, m2, m3, m4, m5 = st.columns( 5, border=True )
		m1.metric( 'Rows', len( df_dataset ) )
		m2.metric( 'Numeric', type_counts.get( 'numeric', 0 ) )
		m3.metric( 'Ordinal / ID',
			type_counts.get( 'ordinal', 0 ) + type_counts.get( 'identifier', 0 ) )
		m4.metric( 'Categorical', type_counts.get( 'categorical', 0 ) )
		m5.metric( 'Datetime', type_counts.get( 'datetime', 0 ) )
		
		# =====================================================================================
		# DIAGNOSTIC VISUALIZATIONS (TAB-1 APPROPRIATE)
		# =====================================================================================
		blue_divider( )
		st.markdown( '##### Diagnostics' )
		
		v1, v2 = st.columns( 2, border=True )
		with v1:
			df_type_counts = type_counts.sort_values( ascending=False ).rename_axis(
				'Type' ).reset_index( name='Count' )
			figure = px.bar( df_type_counts, x='Type', y='Count', color='Type', text='Count' )
			render_mathy_plotly_chart( figure, 'profile_type_distribution_chart',
				'mathy_profile_types', 'Column Type Distribution', 450 )
		
		with v2:
			missing_pct = (df_dataset.isna( ).mean( ) * 100).sort_values( ascending=False )
			missing_pct = missing_pct[ missing_pct > 0 ].head( 10 )
			if not missing_pct.empty:
				df_missing_pct = missing_pct.sort_values( ascending=True ).rename_axis(
					'Column' ).reset_index( name='Percent Missing' )
				figure = px.bar( df_missing_pct, x='Percent Missing', y='Column', orientation='h',
					color='Percent Missing', color_continuous_scale='Reds', text='Percent Missing' )
				figure.update_traces( texttemplate='%{text:.1f}%' )
				render_mathy_plotly_chart( figure, 'profile_missing_percentage_chart',
					'mathy_profile_missing', 'Top Columns by Missing %', 450 )
			else:
				st.info( 'No Missing Values Detected.' )
		
		blue_divider( )
		st.markdown( '##### Cardinality', help=cfg.DATA_CARDINALITY )
		
		v3, v4 = st.columns( 2, border=True )
		with v3:
			cardinality = df_dataset.nunique( dropna=True ).sort_values( ascending=False ).head(
				10 )
			df_cardinality = cardinality.sort_values( ascending=True ).rename_axis(
				'Column' ).reset_index( name='Unique Values' )
			figure = px.bar( df_cardinality, x='Unique Values', y='Column', orientation='h',
				color='Unique Values', color_continuous_scale='Blues', text='Unique Values' )
			render_mathy_plotly_chart( figure, 'profile_cardinality_chart',
				'mathy_profile_cardinality', 'Top Columns by Cardinality', 450 )
		
		with v4:
			st.caption( 'Column cardinality is calculated from populated source values.' )
		
		# -------------------------------------------------------------------------------------
		# Probability Distributions
		# -------------------------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Numeric Distributions' )
		
		numeric_dist_cols = numeric_columns.copy( )
		
		if not numeric_dist_cols:
			st.info( 'No numeric columns detected.' )
		else:
			st.caption( f'{len( numeric_dist_cols )} numeric column(s) detected.' )
			
			ctrl1, ctrl2, ctrl3 = st.columns( 3, border=True )
			with ctrl1:
				dist_bins = st.slider( 'Bins', min_value=10, max_value=60, value=30, step=5,
					key='profile_numeric_dist_bins' )
			
			with ctrl2:
				show_kde = st.checkbox( 'Show KDE Overlay', value=True,
					key='profile_numeric_dist_kde' )
			
			with ctrl3:
				dist_mode = st.radio( 'Display', options=[ 'Density', 'Frequency' ],
					horizontal=True, key='profile_numeric_dist_mode' )
			
			st.markdown( """
				<style>
				[data-testid="stMetricLabel"] p {
					font-size: 0.85rem;
				}
				
				[data-testid="stMetricValue"] {
					font-size: 1.05rem;
				}
				
				[data-testid="stMetric"] {
					padding-top: 0.10rem;
					padding-bottom: 0.10rem;
				}
				</style>
				""", unsafe_allow_html=True )
			
			stat_mode = 'density' if dist_mode == 'Density' else 'count'
			grid_cols = st.columns( 2, border=True )
			for i, col in enumerate( numeric_dist_cols ):
				with grid_cols[ i % 2 ]:
					s = pd.to_numeric( df_analysis[ col ], errors='coerce' )
					s = s.replace( [ np.inf, -np.inf ], np.nan ).dropna( )
					
					if s.empty:
						st.warning( f'{col}: no plottable numeric values.' )
						continue
					
					mean_val = float( s.mean( ) )
					median_val = float( s.median( ) )
					histnorm = 'probability density' if stat_mode == 'density' else None
					figure = go.Figure( )
					figure.add_trace( go.Histogram( x=s.tolist( ), nbinsx=dist_bins,
						histnorm=histnorm, name=col, marker_color='#38BDF8', opacity=0.82 ) )
					if show_kde and len( s ) > 1 and float( s.std( ddof=1 ) ) > 0:
						x_values = np.linspace( float( s.min( ) ), float( s.max( ) ), 250 )
						kde_values = stats.gaussian_kde( s.to_numpy( ) )( x_values )
						if stat_mode == 'count':
							bin_width = (float( s.max( ) ) - float( s.min( ) )) / dist_bins
							kde_values = kde_values * len( s ) * bin_width
						figure.add_trace( go.Scatter( x=x_values, y=kde_values, mode='lines',
							name='KDE', line={ 'color': '#F472B6', 'width': 3 } ) )
					add_statistical_reference_lines( figure, mean_val, median_val )
					figure.update_xaxes( title_text=col )
					figure.update_yaxes( title_text='Density' if stat_mode == 'density' else 'Frequency' )
					render_mathy_plotly_chart( figure, f'profile_numeric_distribution_chart_{i}',
						f'mathy_profile_distribution_{i}', f'Distribution — {col}', 470 )
					m1, m2, m3, m4 = st.columns( 4, border=True )
					m1.metric( 'Count', f'{len( s ):,}' )
					m2.metric( 'Mean', f'{mean_val:,.2f}' )
					m3.metric( 'Median', f'{median_val:,.2f}' )
					m4.metric( 'Std',
						f'{float( s.std( ddof=1 ) ):,.2f}' if len( s ) > 1 else '0.00' )
		
# ============================================
#  DESCRIPTIVE STATISTICS MODE
# ============================================
elif mode == 'Descriptive Statistics':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Descriptive Statistics' ], help=cfg.DESCRIPTIVE_STATISTICS )
		st.divider( )
		df_dataset = get_analysis_dataset( )
		if df_dataset is None:
			st.info( 'No data available.' )
			st.stop( )
		df_numeric = clean_numeric( df_dataset.select_dtypes( include=[ np.number ] ) )
		if df_numeric.empty:
			st.info( 'No numeric variables available for descriptive analysis.' )
			st.stop( )
		
		all_num_cols = df_numeric.columns.tolist( )
		
		st.markdown( """
			<style>
			[data-testid="stMetricLabel"] p {
				font-size: 0.85rem;
			}
			
			[data-testid="stMetricValue"] {
				font-size: 1.05rem;
			}
			
			[data-testid="stMetric"] {
				padding-top: 0.10rem;
				padding-bottom: 0.10rem;
			}
			</style>
			""", unsafe_allow_html=True )
		
		st.markdown( '##### Summary' )
		
		num_c1, num_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with num_c1:
			vars_sel = st.multiselect( 'Select Numeric Variables', all_num_cols,
				default=default_pick( all_num_cols, 3 ) )
		
		sum_c1, sum_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with sum_c1:
			summary_vars = st.multiselect( 'Variables for Summary Table', all_num_cols,
				default=all_num_cols[ : min( 8, len( all_num_cols ) ) ], key='desc_summary_vars' )
		
		with sum_c2:
			show_percentiles = st.checkbox( 'Include Percentiles', value=True,
				key='desc_summary_percentiles' )
		
		if summary_vars:
			df_summary_source = df_numeric[ summary_vars ].copy( )
			percentiles = [ 0.05, 0.25, 0.50, 0.75, 0.95 ] if show_percentiles else None
			df_descriptive = df_summary_source.describe( percentiles=percentiles ).T.reset_index( )
			df_descriptive = df_descriptive.rename( columns={ 'index': 'Variable' } )
			df_descriptive[ 'Variance' ] = df_summary_source.var( ddof=1 ).values
			df_descriptive[ 'Missing' ] = df_dataset[ summary_vars ].isna( ).sum( ).values
			df_descriptive[ 'Missing %' ] = (
					df_dataset[ summary_vars ].isna( ).mean( ).values * 100.0)
			
			df_descriptive[ 'Skew' ] = df_summary_source.skew( ).values
			df_descriptive[ 'Kurtosis' ] = df_summary_source.kurtosis( ).values
			df_descriptive[ 'Zeros' ] = (df_summary_source == 0).sum( ).values
			df_descriptive[ 'Zeros %' ] = ((df_summary_source == 0).mean( ).values * 100.0)
			ordered_cols = [ 'Variable', 'count', 'mean', 'std', 'Variance', 'min' ]
			if show_percentiles:
				for pcol in [ '5%', '25%', '50%', '75%', '95%' ]:
					if pcol in df_descriptive.columns:
						ordered_cols.append( pcol )
			ordered_cols += [ 'max', 'Missing', 'Missing %', 'Zeros', 'Zeros %', 'Skew',
				'Kurtosis' ]
			ordered_cols = [ c for c in ordered_cols if c in df_descriptive.columns ]
			df_descriptive = df_descriptive[ ordered_cols ]
			
			for c in df_descriptive.columns:
				if c != 'Variable':
					df_descriptive[ c ] = pd.to_numeric( df_descriptive[ c ], errors='coerce' )
			
			column_config = { 'Variable': st.column_config.TextColumn( 'Variable',
				width='medium' ),
				'count': st.column_config.NumberColumn( 'Count', format='%.0f' ),
				'mean': st.column_config.NumberColumn( 'Mean', format='%.2f' ),
				'std': st.column_config.NumberColumn( 'Std', format='%.2f' ),
				'Variance': st.column_config.NumberColumn( 'Variance', format='%.2f' ),
				'min': st.column_config.NumberColumn( 'Min', format='%.2f' ),
				'5%': st.column_config.NumberColumn( 'P5', format='%.2f' ),
				'25%': st.column_config.NumberColumn( 'P25', format='%.2f' ),
				'50%': st.column_config.NumberColumn( 'Median', format='%.2f' ),
				'75%': st.column_config.NumberColumn( 'P75', format='%.2f' ),
				'95%': st.column_config.NumberColumn( 'P95', format='%.2f' ),
				'max': st.column_config.NumberColumn( 'Max', format='%.2f' ),
				'Missing': st.column_config.NumberColumn( 'Missing', format='%.0f' ),
				'Missing %': st.column_config.NumberColumn( 'Missing %', format='%.2f' ),
				'Zeros': st.column_config.NumberColumn( 'Zeros', format='%.0f' ),
				'Zeros %': st.column_config.NumberColumn( 'Zeros %', format='%.2f' ),
				'Skew': st.column_config.NumberColumn( 'Skew', format='%.4f' ),
				'Kurtosis': st.column_config.NumberColumn( 'Kurtosis', format='%.4f' ) }
			
			column_config = { k: v for k, v in column_config.items( ) if
				k in df_descriptive.columns }
			render_data_editor( df_descriptive, use_container_width=True, hide_index=True,
				disabled=True, column_config=column_config, key='desc_summary_editor' )
		else:
			st.info( 'Select one or more numeric variables to display descriptive statistics.' )
		
		with num_c2:
			dist_bins = st.slider( 'Distribution Bins', min_value=10, max_value=60, value=30,
				step=5, key='desc_dist_bins' )
		
		for col in vars_sel:
			s = pd.to_numeric( df_numeric[ col ], errors='coerce' )
			s = s.replace( [ np.inf, -np.inf ], np.nan ).dropna( )
			
			if s.empty:
				st.warning( f'{col}: no plottable numeric values.' )
				continue
			
			blue_divider( )
			st.markdown( f'##### Distribution & Shape — {col}' )
			c1, c2 = st.columns( 2, border=True )
			with c1:
				mean_val = float( s.mean( ) )
				median_val = float( s.median( ) )
				figure = go.Figure( )
				figure.add_trace( go.Histogram( x=s.tolist( ), nbinsx=dist_bins, name=col,
					marker_color='#38BDF8', opacity=0.82 ) )
				if len( s ) > 1 and float( s.std( ddof=1 ) ) > 0:
					x_values = np.linspace( float( s.min( ) ), float( s.max( ) ), 250 )
					bin_width = (float( s.max( ) ) - float( s.min( ) )) / dist_bins
					kde_values = stats.gaussian_kde( s.to_numpy( ) )( x_values ) * len( s ) * bin_width
					figure.add_trace( go.Scatter( x=x_values, y=kde_values, mode='lines',
						name='KDE', line={ 'color': '#F472B6', 'width': 3 } ) )
				add_statistical_reference_lines( figure, mean_val, median_val )
				figure.update_xaxes( title_text=col )
				figure.update_yaxes( title_text='Frequency' )
				render_mathy_plotly_chart( figure, f'descriptive_histogram_chart_{col}',
					f'mathy_descriptive_histogram_{col}', f'Histogram — {col}', 480 )
				m1, m2, m3, m4 = st.columns( 4, border=True )
				m1.metric( 'Count', f'{len( s ):,}' )
				m2.metric( 'Mean', f'{mean_val:,.2f}' )
				m3.metric( 'Median', f'{median_val:,.2f}' )
				m4.metric( 'Deviation',
					f'{float( s.std( ddof=1 ) ):,.2f}' if len( s ) > 1 else '0.000' )
			
			with c2:
				figure = create_qq_figure( s.to_numpy( ), col )
				render_mathy_plotly_chart( figure, f'descriptive_qq_chart_{col}',
					f'mathy_descriptive_qq_{col}', f'Q–Q Plot — {col}', 480 )
				
				try:
					if 3 <= len( s ) <= 5000:
						shapiro_stat, shapiro_p = stats.shapiro( s )
						q1, q2, q3 = st.columns( 3, border=True )
						q1.metric( 'Skew',
							f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
						q2.metric( 'Kurtosis',
							f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
						q3.metric( 'Shapiro P', f'{shapiro_p:,.3f}' )
					else:
						q1, q2, q3 = st.columns( 3, border=True )
						q1.metric( 'Skew',
							f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
						q2.metric( 'Kurtosis',
							f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
						q3.metric( 'Shapiro P', 'n/a' )
				except Exception:
					q1, q2, q3 = st.columns( 3, border=True )
					q1.metric( 'Skew',
						f'{float( stats.skew( s ) ):,.3f}' if len( s ) >= 3 else '0.000' )
					q2.metric( 'Kurtosis',
						f'{float( stats.kurtosis( s ) ):,.3f}' if len( s ) >= 4 else '0.000' )
					q3.metric( 'Shapiro P', 'n/a' )
		
		blue_divider( )
		st.markdown( '##### Correlation Structure', help=cfg.CORRELATION_STRUCTURE )
		cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with cor_c1:
			corr_vars = st.multiselect( 'Variables for Correlation', all_num_cols,
				default=default_pick( all_num_cols, 4 ) )
		
		with cor_c2:
			corr_method = st.radio( 'Correlation Method', options=[ 'Pearson', 'Spearman' ],
				horizontal=True, key='desc_corr_method', help=cfg.CORRELATION_HEATMAP )
		
		c3, c4 = st.columns( 2, border=True )
		if len( corr_vars ) >= 2:
			df_correlation = analysis_fillna_mean( df_numeric[ corr_vars ] )
			corr = df_correlation.corr( method=corr_method.lower( ) )
			
			with c3:
				render_table( corr )
			
			with c4:
				figure = go.Figure( data=go.Heatmap( z=corr.values, x=corr.columns,
					y=corr.index, zmin=-1, zmax=1, colorscale='RdBu', reversescale=True,
					text=np.round( corr.values, 2 ), texttemplate='%{text}',
					hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.4f}<extra></extra>' ) )
				render_mathy_plotly_chart( figure, 'descriptive_correlation_heatmap_chart',
					'mathy_descriptive_correlation', f'Correlation Heatmap — {corr_method}', 580 )
		else:
			with c3:
				st.info( 'Select at least two numeric variables.' )
			with c4:
				st.caption( 'Heatmap will appear here once at least two variables are selected.' )
		
		blue_divider( )
		st.markdown( '##### Principal Component Analysis', help=cfg.PCA )
		
		pca_c1, pca_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with pca_c1:
			pca_vars = st.multiselect( 'Select Components', all_num_cols,
				default=default_pick( all_num_cols, 4 ) )
		
		with pca_c2:
			max_components = min( 6, len( pca_vars ) )
			if max_components < 2:
				n_comp = 2
				st.caption( 'Select at least two variables to configure PCA components.' )
			elif max_components == 2:
				n_comp = st.selectbox( 'Components', [ 2 ], key='desc_pca_components_fixed' )
			else:
				n_comp = st.slider( 'Components', 2, max_components, min( 3, max_components ),
					key='desc_pca_components_slider' )
		
		c5, c6 = st.columns( 2, border=True )
		if len( pca_vars ) >= 2:
			X = analysis_fillna_mean( df_numeric[ pca_vars ] )
			Xs = SKStandardScaler( ).fit_transform( X )
			pca = PCA( num=n_comp ).train( Xs )
			df_explained = pd.DataFrame( { 'Component': [ f'PC{i + 1}' for i in range( n_comp ) ],
				'Explained Variance (%)': pca.explained_variance_ratio * 100 } )
			
			with c5:
				render_table( df_explained )
			
			with c6:
				figure = px.bar( df_explained, x='Component', y='Explained Variance (%)',
					color='Explained Variance (%)', color_continuous_scale='Blues',
					text='Explained Variance (%)' )
				figure.update_traces( texttemplate='%{text:.1f}%' )
				render_mathy_plotly_chart( figure, 'descriptive_pca_variance_chart',
					'mathy_pca_variance', 'PCA Variance Explained', 500 )
		else:
			with c5:
				st.info( 'Select at least two numeric variables for PCA.' )
			with c6:
				st.caption( 'Explained variance chart will appear here once at least two variables are '
					'selected.' )

# ============================================
# INFERENTIAL STATISTICS MODE
# ============================================
elif mode == 'Inferential Statistics':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Inferential Statistics' ], help=cfg.INFERENTIAL_STATISTICS )
		st.divider( )
		
		df_source = get_loaded_dataset( )
		df_dataset = df_source
		if df_dataset is None or df_dataset.empty:
			st.info( 'No data available.' )
			st.stop( )
		
		numeric_columns, categorical_columns = synchronize_dataset_columns( df_source )
		df_dataset = st.session_state[ 'df_analysis' ].copy( )
		reset_inferential_selection_state( df_dataset, numeric_columns, categorical_columns )
		if not numeric_columns:
			st.info( 'No numeric variables available for inferential analysis.' )
			st.stop( )
		
		st.markdown( """
			<style>
			[data-testid="stMetricLabel"] p {
				font-size: 0.85rem;
			}
			
			[data-testid="stMetricValue"] {
				font-size: 1.05rem;
			}
			
			[data-testid="stMetric"] {
				padding-top: 0.10rem;
				padding-bottom: 0.10rem;
			}
			</style>
			""", unsafe_allow_html=True )
		
		# -------------------------------------------------------------------------------------
		# INFERENTIAL SUMMARY
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Summary' )
		
		sum_r1c1, sum_r1c2, sum_r1c3 = st.columns( 3, border=True )
		with sum_r1c1:
			summary_y = st.selectbox( 'Summary Outcome Variable', numeric_columns,
				key='infer_summary_y' )
		
		with sum_r1c2:
			summary_x = st.selectbox( 'Summary Second Numeric Variable',
				[ '<None>' ] + [ c for c in numeric_columns if c != summary_y ],
				key='infer_summary_x' )
			if summary_x == '<None>':
				summary_x = None
		
		with sum_r1c3:
			if categorical_columns:
				summary_group = st.selectbox( 'Summary Grouping Variable',
					[ '<None>' ] + categorical_columns, key='infer_summary_group' )
				if summary_group == '<None>':
					summary_group = None
			else:
				summary_group = None
				st.caption( 'No categorical grouping variables available.' )
		
		sum_r2c1, sum_r2c2 = st.columns( 2, border=True )
		with sum_r2c1:
			if len( categorical_columns ) >= 2:
				summary_cat1 = st.selectbox( 'Summary First Categorical Variable',
					categorical_columns, key='infer_summary_cat1' )
			else:
				summary_cat1 = None
				st.caption( 'At least two categorical variables are required.' )
		
		with sum_r2c2:
			if summary_cat1 and len( categorical_columns ) >= 2:
				summary_cat2 = st.selectbox( 'Summary Second Categorical Variable',
					[ c for c in categorical_columns if c != summary_cat1 ],
					key='infer_summary_cat2' )
			else:
				summary_cat2 = None
		
		infer_rows = [ ]
		
		# -----------------------------------------------------------------
		# Normality Summary
		# -----------------------------------------------------------------
		summary_series = pd.to_numeric( df_dataset[ summary_y ], errors='coerce' ).dropna( )
		if len( summary_series ) >= 3:
			try:
				shapiro_stat, shapiro_p = stats.shapiro( summary_series )
				infer_rows.append( { 'Analysis': 'Outcome Distribution', 'Test': 'Shapiro-Wilk',
					'Statistic': shapiro_stat, 'P-Value': shapiro_p, 'DoF': np.nan,
					'Effect Size': np.nan, 'N': float( len( summary_series ) ),
					'Notes': 'Normality Assessment' } )
			except Exception:
				pass
		
		# -----------------------------------------------------------------
		# Group Comparison Summary
		# -----------------------------------------------------------------
		if summary_group:
			df_group_summary = df_dataset[ [ summary_group, summary_y ] ].copy( )
			df_group_summary[ summary_y ] = pd.to_numeric( df_group_summary[ summary_y ],
				errors='coerce' )
			
			df_group_summary = df_group_summary.dropna( subset=[ summary_group, summary_y ] )
			
			group_arrays = [ grp[ summary_y ].values for _, grp in
				df_group_summary.groupby( summary_group ) ]
			
			valid_group_arrays = [ g for g in group_arrays if len( g ) >= 2 ]
			if len( valid_group_arrays ) >= 2:
				try:
					f_stat, p_anova = stats.f_oneway( *valid_group_arrays )
					infer_rows.append( { 'Analysis': 'Group Comparison', 'Test': 'One-Way ANOVA',
						'Statistic': f_stat, 'P-Value': p_anova,
						'DoF': float( len( valid_group_arrays ) - 1 ), 'Effect Size': np.nan,
						'N': float( sum( len( g ) for g in valid_group_arrays ) ),
						'Notes': f'{summary_y} by {summary_group}' } )
				except Exception:
					pass
				
				try:
					h_stat, p_kw = stats.kruskal( *valid_group_arrays )
					infer_rows.append( { 'Analysis': 'Group Comparison', 'Test': 'Kruskal-Wallis',
						'Statistic': h_stat, 'P-Value': p_kw,
						'DoF': float( len( valid_group_arrays ) - 1 ), 'Effect Size': np.nan,
						'N': float( sum( len( g ) for g in valid_group_arrays ) ),
						'Notes': f'{summary_y} by {summary_group}' } )
				except Exception:
					pass
		
		# -----------------------------------------------------------------
		# Correlation Summary
		# -----------------------------------------------------------------
		if summary_x:
			x_summary = pd.to_numeric( df_dataset[ summary_x ], errors='coerce' )
			y_summary = pd.to_numeric( df_dataset[ summary_y ], errors='coerce' )
			pair_mask = x_summary.notna( ) & y_summary.notna( )
			if pair_mask.sum( ) >= 3:
				try:
					pearson_r, pearson_p = stats.pearsonr( x_summary[ pair_mask ],
						y_summary[ pair_mask ] )
					
					infer_rows.append( { 'Analysis': 'Association', 'Test': 'Pearson Correlation',
						'Statistic': pearson_r, 'P-Value': pearson_p,
						'DoF': float( pair_mask.sum( ) - 2 ), 'Effect Size': abs( pearson_r ),
						'N': float( pair_mask.sum( ) ), 'Notes': f'{summary_y} vs {summary_x}' } )
				except Exception:
					pass
				
				try:
					spearman_rho, spearman_p = stats.spearmanr( x_summary[ pair_mask ],
						y_summary[ pair_mask ] )
					
					infer_rows.append( { 'Analysis': 'Association', 'Test': 'Spearman Correlation',
						'Statistic': spearman_rho, 'P-Value': spearman_p, 'DoF': np.nan,
						'Effect Size': abs( spearman_rho ), 'N': float( pair_mask.sum( ) ),
						'Notes': f'{summary_y} vs {summary_x}' } )
				except Exception:
					pass
		
		# -----------------------------------------------------------------
		# Categorical Association Summary
		# -----------------------------------------------------------------
		if summary_cat1 and summary_cat2:
			contingency_summary = pd.crosstab( df_dataset[ summary_cat1 ],
				df_dataset[ summary_cat2 ] )
			
			if not contingency_summary.empty and contingency_summary.shape[ 0 ] >= 2 and \
					contingency_summary.shape[ 1 ] >= 2:
				try:
					chi2_stat, chi2_p, chi2_dof, expected = stats.chi2_contingency(
						contingency_summary )
					
					n_total = contingency_summary.to_numpy( ).sum( )
					phi2 = chi2_stat / n_total if n_total > 0 else np.nan
					r_dim, c_dim = contingency_summary.shape
					cramers_v = (np.sqrt( phi2 / min( c_dim - 1, r_dim - 1 ) ) if min( c_dim - 1,
						r_dim - 1 ) > 0 else np.nan)
					
					infer_rows.append(
						{ 'Analysis': 'Categorical Association', 'Test': 'Chi-Square',
							'Statistic': chi2_stat, 'P-Value': chi2_p, 'DoF': float( chi2_dof ),
							'Effect Size': cramers_v, 'N': float( n_total ),
							'Notes': f'{summary_cat1} vs {summary_cat2}' } )
				except Exception:
					pass
		
		if infer_rows:
			df_infer_summary = pd.DataFrame( infer_rows )
			
			for c in [ 'Statistic', 'P-Value', 'DoF', 'Effect Size', 'N' ]:
				if c in df_infer_summary.columns:
					df_infer_summary[ c ] = pd.to_numeric( df_infer_summary[ c ], errors='coerce' )
			
			infer_column_config = {
				'Analysis': st.column_config.TextColumn( 'Analysis', width='medium' ),
				'Test': st.column_config.TextColumn( 'Test', width='medium' ),
				'Statistic': st.column_config.NumberColumn( 'Statistic', format='%.2f' ),
				'P-Value': st.column_config.NumberColumn( 'P Value', format='%.2g' ),
				'DoF': st.column_config.NumberColumn( 'DoF', format='%.0f' ),
				'Effect Size': st.column_config.NumberColumn( 'Effect Size', format='%.2f' ),
				'N': st.column_config.NumberColumn( 'N', format='%.0f' ),
				'Notes': st.column_config.TextColumn( 'Notes', width='large' ) }
			
			render_data_editor( df_infer_summary, use_container_width=True, hide_index=True,
				disabled=True, column_config=infer_column_config, key='infer_summary_editor' )
		else:
			st.info( 'Unable to compute inferential summary for the current selections.' )
		
		blue_divider( )
		
		# -------------------------------------------------------------------------------------
		# NORMALITY + GROUP COMPARISON
		# -------------------------------------------------------------------------------------
		nml_c1, nml_c2 = st.columns( [ 0.5, 0.5 ], border=True, gap='medium' )
		col_group = None
		
		with nml_c1:
			st.markdown( '##### Normality Test', help=cfg.NORMALITY_TESTING )
			col_y = st.selectbox( 'Select Numeric Outcome Variable', numeric_columns )
			y = pd.to_numeric( df_dataset[ col_y ], errors='coerce' ).dropna( )
			if len( y ) >= 3:
				stat, p_value = stats.shapiro( y )
				figure = create_qq_figure( y.to_numpy( ), col_y )
				render_mathy_plotly_chart( figure, 'inferential_normality_qq_chart',
					'mathy_inferential_qq', f'Q–Q Plot — {col_y}', 525 )
				m1, m2, m3 = st.columns( 3 )
				m1.metric( 'Count', f'{len( y ):,}' )
				m2.metric( 'Shapiro-W', f'{stat:,.2f}' )
				m3.metric( 'Shapiro-P', f'{p_value:,.2g}' )
				
				if p_value < 0.05:
					st.caption( 'Distribution departs from normality at α = 0.05.' )
				else:
					st.caption(
						'Distribution does not significantly depart from normality at α=0.05' )
			else:
				st.info( 'Not enough observations for normality testing.' )
		
		with nml_c2:
			st.markdown( '##### Group Comparison' )
			if categorical_columns:
				col_group = st.selectbox( 'Select Grouping Variable (optional)',
					[ '<None>' ] + categorical_columns )
				if col_group == '<None>':
					col_group = None
			
			if col_group:
				df_group = df_dataset[ [ col_group, col_y ] ].copy( )
				df_group[ col_y ] = pd.to_numeric( df_group[ col_y ], errors='coerce' )
				df_group = df_group.dropna( subset=[ col_group, col_y ] )
				grouped = [ grp[ col_y ].values for _, grp in df_group.groupby( col_group ) ]
				valid_groups = [ g for g in grouped if len( g ) >= 2 ]
				
				if len( valid_groups ) >= 2:
					f_stat, p_anova = stats.f_oneway( *valid_groups )
					h_stat, p_kw = stats.kruskal( *valid_groups )
					figure = px.box( df_group, x=col_group, y=col_y, color=col_group,
						points='all' )
					render_mathy_plotly_chart( figure, 'inferential_group_comparison_chart',
						'mathy_group_comparison',
						f'Group Comparison — {col_y} by {col_group}', 525 )
					g1, g2, g3, g4 = st.columns( 4 )
					g1.metric( 'Groups', f'{len( valid_groups ):,}' )
					g2.metric( 'ANOVA F', f'{f_stat:,.2f}' )
					g3.metric( 'ANOVA P', f'{p_anova:,.2g}' )
					g4.metric( 'Kruskal P', f'{p_kw:,.2g}' )
					st.caption( f'Kruskal–Wallis H = {h_stat:.4f}. '
					            f'Use the nonparametric result when normality or homoscedasticity '
					            f'is doubtful.' )
				else:
					st.info( 'Not enough valid groups for group comparison.' )
			else:
				st.info( 'Select a grouping variable to compare groups.' )
		
		blue_divider( )
		
		# -------------------------------------------------------------------------------------
		# CORRELATION ANALYSIS
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Correlation Analysis' )
		cor_c1, cor_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with cor_c1:
			candidate_x = [ c for c in numeric_columns if c != col_y ]
			if not candidate_x:
				st.info( 'A second numeric variable is required for Correlation Analysis.' )
				col_x2 = None
			else:
				col_x2 = st.selectbox( 'Select Second Numeric Variable', candidate_x )
		
		with cor_c2:
			if col_x2:
				x = pd.to_numeric( df_dataset[ col_x2 ], errors='coerce' )
				y2 = pd.to_numeric( df_dataset[ col_y ], errors='coerce' )
				mask = x.notna( ) & y2.notna( )
				if mask.sum( ) >= 3:
					r_p, p_p = stats.pearsonr( x[ mask ], y2[ mask ] )
					r_s, p_s = stats.spearmanr( x[ mask ], y2[ mask ] )
					r1, r2, r3, r4 = st.columns( 4 )
					r1.metric( 'Pairs', f'{int( mask.sum( ) ):,}' )
					r2.metric( 'Pearson R', f'{r_p:,.2f}' )
					r3.metric( 'Pearson P', f'{p_p:,.2g}' )
					r4.metric( 'Spearman R', f'{r_s:,.2f}' )
					st.caption( f'Spearman P = {p_s:.2g}' )
				else:
					st.info( 'Not enough paired observations for correlation.' )
		
		if col_x2:
			mask = x.notna( ) & y2.notna( )
			if mask.sum( ) >= 3:
				df_correlation_plot = pd.DataFrame(
					{ col_x2: x[ mask ].to_numpy( ), col_y: y2[ mask ].to_numpy( ) } )
				figure = px.scatter( df_correlation_plot, x=col_x2, y=col_y, opacity=0.70 )
				figure.update_traces( marker={ 'size': 9 } )
				if mask.sum( ) >= 2:
					try:
						m, b = np.polyfit( x[ mask ], y2[ mask ], 1 )
						xline = np.linspace( float( x[ mask ].min( ) ), float( x[ mask ].max( ) ),
							100 )
						figure.add_trace( go.Scatter( x=xline, y=(m * xline) + b,
							mode='lines', name='Linear Trend',
							line={ 'color': '#F59E0B', 'width': 3, 'dash': 'dash' } ) )
					except Exception:
						pass
				render_mathy_plotly_chart( figure, 'inferential_correlation_scatter_chart',
					'mathy_inferential_correlation', f'Correlation — {col_y} vs {col_x2}', 525 )
		
		blue_divider( )
		
		# -------------------------------------------------------------------------------------
		# CATEGORICAL ASSOCIATION
		# -------------------------------------------------------------------------------------
		st.markdown( '##### Categorical Association' )
		if not categorical_columns or len( categorical_columns ) < 2:
			st.info(
				'At least two categorical variables are required for categorical association.' )
		else:
			cat_c1, cat_c2 = st.columns( [ 0.5, 0.5 ] )
			with cat_c1:
				col_cat1 = st.selectbox( 'Select First Categorical Variable', categorical_columns )
			
			with cat_c2:
				col_cat2 = st.selectbox( 'Select Second Categorical Variable',
					[ c for c in categorical_columns if c != col_cat1 ] )
			
			contingency = pd.crosstab( df_dataset[ col_cat1 ], df_dataset[ col_cat2 ] )
			
			if contingency.empty or contingency.shape[ 0 ] < 2 or contingency.shape[ 1 ] < 2:
				st.info( 'Not enough categorical variation for Chi-Square Analysis.' )
			else:
				chi2, p_chi, dof, expected = stats.chi2_contingency( contingency )
				n = contingency.to_numpy( ).sum( )
				phi2 = chi2 / n if n > 0 else np.nan
				r, k = contingency.shape
				cramers_v = np.sqrt( phi2 / min( k - 1, r - 1 ) ) if min( k - 1,
					r - 1 ) > 0 else np.nan
				
				ca1, ca2 = st.columns( 2, border=True )
				with ca1:
					render_data_editor( contingency, key='inference_data', height='stretch',
						num_rows='dynamic' )
				
				with ca2:
					figure = go.Figure( data=go.Heatmap( z=contingency.values,
						x=contingency.columns.astype( str ), y=contingency.index.astype( str ),
						colorscale='Blues', text=contingency.values, texttemplate='%{text}',
						hovertemplate=f'{col_cat1}: %{{y}}<br>{col_cat2}: %{{x}}'
							'<br>Count: %{z}<extra></extra>' ) )
					figure.update_xaxes( title_text=col_cat2 )
					figure.update_yaxes( title_text=col_cat1 )
					render_mathy_plotly_chart( figure, 'inferential_contingency_heatmap_chart',
						'mathy_contingency_heatmap',
						f'Contingency Heatmap — {col_cat1} vs {col_cat2}', 550 )
				
				cm1, cm2, cm3, cm4 = st.columns( 4, border=True )
				cm1.metric( 'Chi-Square', f'{chi2:,.2f}' )
				cm2.metric( 'P Value', f'{p_chi:,.2g}' )
				cm3.metric( 'DoF', f'{dof:,}' )
				cm4.metric( "Cramér's V",
					f'{cramers_v:,.2f}' if np.isfinite( cramers_v ) else 'n/a' )

# ============================================
# ANOMALY DETECTION MODE
# ============================================
elif mode == 'Anomaly Detection':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Anomaly Detection' ] )
		st.divider( )
		if st.session_state.df_dataset is None:
			st.info( 'No data loaded.' )
			st.stop( )
		
		df_dataset = get_analysis_dataset( )
		if df_dataset is None:
			st.info( 'No data loaded.' )
			st.stop( )
		df_numeric = clean_numeric( df_dataset.select_dtypes( include=[ np.number ] ) )
		if df_numeric.empty:
			st.info( 'No usable numeric columns available for anomaly detection.' )
			st.stop( )
		
		st.markdown( """
			<style>
			[data-testid="stMetricLabel"] p {
				font-size: 0.85rem;
			}
			
			[data-testid="stMetricValue"] {
				font-size: 1.05rem;
			}
			
			[data-testid="stMetric"] {
				padding-top: 0.10rem;
				padding-bottom: 0.10rem;
			}
			</style>
			""", unsafe_allow_html=True )
		
		msg = 'Select fields/columns containing numerical data to analyze outliers'
		st.markdown( '##### Feature Selection', help=msg )
		aml_c1, aml_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with aml_c1:
			all_num_cols = df_numeric.columns.tolist( )
			preferred = [ c for c in all_num_cols if c.lower( ) in ('py', 'cy', 'by') ]
			default_vars = preferred if preferred else default_pick( all_num_cols, 2 )
			vars_sel = st.multiselect( 'Variables to Analyze', all_num_cols, default=default_vars )
			
			if not vars_sel:
				st.info( 'Select at least one numeric variable to run anomaly detection.' )
				st.stop( )
		
		with aml_c2:
			analysis_scale = st.checkbox( 'Use Analysis-Only Standardization', value=False )
			df_analysis = df_numeric[ vars_sel ].astype( float ).copy( )
			
			if analysis_scale and len( vars_sel ) > 1:
				df_analysis = pd.DataFrame( SKStandardScaler( ).fit_transform(
					df_analysis.values ),
					columns=df_analysis.columns, index=df_analysis.index )
			
			if analysis_scale and len( vars_sel ) > 1:
				df_analysis[ : ] = SKStandardScaler( ).fit_transform( df_analysis.values )
		
		# -------------------------------------------------------------------------
		# Method Selection
		# -------------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Detection Methods' )
		
		c_m1, c_m2 = st.columns( 2, border=True )
		with c_m1:
			use_z = st.checkbox( 'Z-Score', value=True, help=cfg.Z_SCORE )
			use_mz = st.checkbox( 'Modified Z-Score (MAD)', value=True, help=cfg.MODIFIED_Z )
			use_iqr = st.checkbox( 'IQR Fence', value=True, help=cfg.IQR )
		
		with c_m2:
			use_mahal = st.checkbox( 'Mahalanobis Distance', value=True, help=cfg.MAHALANOBIS )
			use_iforest = st.checkbox( 'Isolation Forest', value=True, help=cfg.ISOLATION_FOREST )
			use_lof = st.checkbox( 'Local Outlier Factor (LOF)', value=False, help=cfg.LOF )
		
		# -------------------------------------------------------------------------
		# Threshold Controls
		# -------------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Thresholds' )
		
		c_t1, c_t2 = st.columns( 2, border=True )
		with c_t1:
			z_thresh = st.slider( 'Z / Modified Z threshold', 2.0, 5.0, 3.0, 0.1,
				help=cfg.MODIFIED_Z )
			iqr_mult = st.slider( 'IQR Multiplier', 1.0, 3.0, 1.5, 0.1, help=cfg.IQR_MULTIPLIER )
		
		with c_t2:
			lof_k = st.slider( 'LOF Neighbors (k)', 5, 50, 20, 1, help=cfg.LOF_K )
			min_methods = st.slider( 'Consensus: minimum methods flagging a row', 1, 4, 1, 1,
				help=cfg.MIN_METHODS )
		
		# -------------------------------------------------------------------------
		# Run Detection
		# -------------------------------------------------------------------------
		df_anamolies = pd.DataFrame( index=df_analysis.index )
		for col in vars_sel:
			s = df_analysis[ col ].dropna( )
			if s.empty:
				continue
			
			if use_z:
				z = (s - s.mean( )) / s.std( ) if s.std( ) else pd.Series( 0.0, index=s.index )
				df_anamolies[ f'{col}_z' ] = z.abs( ) >= z_thresh
			
			if use_mz:
				med = s.median( )
				mad = np.median( np.abs( s - med ) )
				if mad == 0:
					mz = pd.Series( 0.0, index=s.index )
				else:
					mz = 0.6745 * (s - med) / mad
				df_anamolies[ f'{col}_mz' ] = mz.abs( ) >= z_thresh
			
			if use_iqr:
				q1, q3 = s.quantile( 0.25 ), s.quantile( 0.75 )
				iqr = q3 - q1
				lo = q1 - iqr_mult * iqr
				hi = q3 + iqr_mult * iqr
				df_anamolies[ f'{col}_iqr' ] = (s < lo) | (s > hi)
		
		df_muliti = df_analysis.dropna( axis=0 )
		if df_muliti.shape[ 0 ] >= 10 and df_muliti.shape[ 1 ] >= 2:
			if use_mahal:
				cov = np.cov( df_muliti.values, rowvar=False )
				if np.linalg.det( cov ) != 0:
					inv_cov = np.linalg.inv( cov )
					mean = df_muliti.mean( ).values
					diffs = df_muliti.values - mean
					md = np.sqrt( np.einsum( 'ij,jk,ik->i', diffs, inv_cov, diffs ) )
					cutoff = np.sqrt( stats.chi2.ppf( 0.975, df_muliti.shape[ 1 ] ) )
					df_anamolies.loc[ df_muliti.index, 'mahal' ] = md > cutoff
			
			if use_iforest:
				from sklearn.ensemble import IsolationForest
				
				iso = IsolationForest( contamination='auto', random_state=42 )
				preds = iso.fit_predict( df_muliti.values )
				df_anamolies.loc[ df_muliti.index, 'iforest' ] = preds == -1
			
			if use_lof:
				from sklearn.neighbors import LocalOutlierFactor
				
				lof = LocalOutlierFactor( n_neighbors=lof_k )
				preds = lof.fit_predict( df_muliti.values )
				df_anamolies.loc[ df_muliti.index, 'lof' ] = preds == -1
		
		# -------------------------------------------------------------------------
		# Consensus & Output
		# -------------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Outlier Summary' )
		
		if df_anamolies.empty:
			st.info( 'No anomalies detected under the selected methods and thresholds.' )
			st.stop( )
		
		df_anamolies = df_anamolies.fillna( False )
		df_anamolies[ 'methods_flagged' ] = df_anamolies.sum( axis=1 )
		anomalies = df_anamolies[ df_anamolies[ 'methods_flagged' ] >= min_methods ].copy( )
		
		m1, m2, m3, m4 = st.columns( 4, border=True )
		m1.metric( 'Rows Analyzed', f'{len( df_analysis ):,}' )
		m2.metric( 'Flagged Rows', f'{len( anomalies ):,}' )
		m3.metric( 'Flag Rate %',
			f'{(100.0 * len( anomalies ) / max( 1, len( df_analysis ) )):,.2f}' )
		m4.metric( 'Min Methods', f'{min_methods:,}' )
		
		c_o1, c_o2 = st.columns( 2, border=True )
		with c_o1:
			st.markdown( '##### Flagged Observations' )
			render_table( anomalies.sort_values( 'methods_flagged', ascending=False ) )
		
		with c_o2:
			st.markdown( '##### Flag Count Distribution' )
			if anomalies.empty:
				st.info( 'No rows met the current consensus threshold.' )
			else:
				vc = anomalies[ 'methods_flagged' ].value_counts( ).sort_index( )
				df_consensus = vc.rename_axis( 'Methods Flagging' ).reset_index(
					name='Observation Count' )
				df_consensus[ 'Methods Flagging' ] = df_consensus[ 'Methods Flagging' ].astype( str )
				figure = px.bar( df_consensus, x='Methods Flagging', y='Observation Count',
					color='Observation Count', color_continuous_scale='Reds',
					text='Observation Count' )
				render_mathy_plotly_chart( figure, 'anomaly_consensus_strength_chart',
					'mathy_anomaly_consensus', 'Consensus Strength', 500 )
		
		# -------------------------------------------------------------------------
		# Visualization — Distribution with Anomalies
		# -------------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Empirical Cumulative Distribution Function (ECDF)', help=cfg.ECDF )
		
		for col in vars_sel:
			if col not in df_analysis.columns:
				continue
			
			s = pd.to_numeric( df_analysis[ col ], errors='coerce' ).replace( [ np.inf, -np.inf ],
				np.nan )
			
			s_clean = s.dropna( )
			if s_clean.empty:
				continue
			
			flagged_idx = anomalies.index.intersection( s_clean.index )
			flagged_vals = s_clean.loc[ flagged_idx ] if not flagged_idx.empty else pd.Series(
				dtype=float )
			
			c_v1, c_v2 = st.columns( 2, border=True )
			with c_v1:
				s_sorted = np.sort( s_clean.values.astype( float ) )
				n_vals = len( s_sorted )
				y_ecdf = np.arange( 1, n_vals + 1 ) / n_vals
				figure = go.Figure( )
				figure.add_trace( go.Scatter( x=s_sorted, y=y_ecdf, mode='lines', name='ECDF',
					line={ 'color': '#38BDF8', 'width': 3, 'shape': 'hv' } ) )
				
				if not flagged_vals.empty:
					flagged_array = flagged_vals.values.astype( float )
					flagged_y = np.searchsorted( s_sorted, flagged_array, side='right' ) / n_vals
					figure.add_trace( go.Scatter( x=flagged_array, y=flagged_y, mode='markers',
						name='Flagged', marker={ 'color': '#EF4444', 'size': 10,
							'symbol': 'x' } ) )
				
				mean_val = float( s_clean.mean( ) )
				median_val = float( s_clean.median( ) )
				add_statistical_reference_lines( figure, mean_val, median_val )
				figure.update_xaxes( title_text=col )
				figure.update_yaxes( title_text='Cumulative Probability', range=[ 0.0, 1.02 ] )
				render_mathy_plotly_chart( figure, f'anomaly_ecdf_chart_{col}',
					f'mathy_anomaly_ecdf_{col}', f'{col} — ECDF with Anomalies', 500 )
			
			with c_v2:
				figure = go.Figure( )
				figure.add_trace( go.Violin( x=s_clean.values, name=col, orientation='h',
					box_visible=True, meanline_visible=True, points='outliers',
					line_color='#38BDF8', fillcolor='rgba(56,189,248,0.45)' ) )
				if not flagged_vals.empty:
					figure.add_trace( go.Scatter( x=flagged_vals.values,
						y=[ col ] * len( flagged_vals ), mode='markers', name='Flagged',
						marker={ 'color': '#EF4444', 'size': 10, 'symbol': 'x' } ) )
				figure.add_vline( x=float( s_clean.mean( ) ), line_dash='dash',
					line_color='#F59E0B' )
				figure.add_vline( x=float( s_clean.median( ) ), line_dash='dot',
					line_color='#2DD4BF' )
				figure.update_xaxes( title_text=col )
				render_mathy_plotly_chart( figure, f'anomaly_violin_chart_{col}',
					f'mathy_anomaly_violin_{col}', f'{col} — Violin / Box Summary', 500 )
		
		blue_divider( )
		
		# -------------------------------------------------------------------------
		# Bivariate View
		# -------------------------------------------------------------------------
		if len( vars_sel ) >= 2:
			st.markdown( '##### Bivariate View' )
			
			x_col = vars_sel[ 0 ]
			y_col = vars_sel[ 1 ]
			df_scatter = df_analysis[ [ x_col, y_col ] ].copy( ).dropna( )
			
			if not df_scatter.empty:
				flag_mask = df_scatter.index.isin( anomalies.index )
				figure = go.Figure( )
				figure.add_trace( go.Scatter( x=df_scatter.loc[ ~flag_mask, x_col ],
					y=df_scatter.loc[ ~flag_mask, y_col ], mode='markers', name='Inliers',
					marker={ 'color': '#38BDF8', 'size': 8, 'opacity': 0.70 } ) )
				
				if flag_mask.any( ):
					figure.add_trace( go.Scatter( x=df_scatter.loc[ flag_mask, x_col ],
						y=df_scatter.loc[ flag_mask, y_col ], mode='markers', name='Flagged',
						marker={ 'color': '#EF4444', 'size': 12, 'symbol': 'x' } ) )
				figure.update_xaxes( title_text=x_col )
				figure.update_yaxes( title_text=y_col )
				render_mathy_plotly_chart( figure, 'anomaly_bivariate_scatter_chart',
					'mathy_anomaly_scatter', f'Anomaly Scatter — {x_col} vs {y_col}', 550 )
		
		# -------------------------------------------------------------------------
		# Export
		# -------------------------------------------------------------------------
		st.download_button( "Export Anomaly Table (CSV)", anomalies.to_csv( ), "anomalies.csv",
			"text/csv" )

# ============================================
# CLASSIFICATION MODE
# ============================================
elif mode == 'Classification Models':
	df_original = get_analysis_dataset( )
	df_dataset = df_original.copy( ) if isinstance( df_original, pd.DataFrame ) else None
	df_working = st.session_state.get( 'df_working', None )
	df_processed = st.session_state.get( 'df_processed', None )
	df_classification = st.session_state.get( 'df_classification', None )
	df_model = st.session_state.get( 'df_model', None )
	df_scores = st.session_state.get( 'df_scores', None )
	df_predictions = st.session_state.get( 'df_predictions', None )
	numeric_columns = st.session_state.get( 'numeric_columns', [ ] )
	categorical_columns = st.session_state.get( 'categorical_columns', [ ] )
	features = st.session_state.get( 'features', [ ] )
	targets = st.session_state.get( 'targets', [ ] )
	selected_all = st.session_state.get( 'selected_all', [ ] )
	active_features = st.session_state.get( 'active_features', [ ] )
	active_targets = st.session_state.get( 'active_targets', [ ] )
	model = st.session_state.get( 'model', None )
	X_data = st.session_state.get( 'X_data', None )
	X_train = st.session_state.get( 'X_train', None )
	X_test = st.session_state.get( 'X_test', None )
	y_train = st.session_state.get( 'y_train', None )
	y_test = st.session_state.get( 'y_test', None )
	y_series = st.session_state.get( 'y_series', None )
	y_prediction = st.session_state.get( 'y_prediction', None )
	elapsed_seconds = st.session_state.get( 'elapsed_seconds', 0.0 )
	
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Classification Models' ] )
		st.caption( 'Predictive Models for Categorical, Discrete-Values' )
		st.divider( )
		
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		st.session_state[ 'df_original' ] = df_original.copy( )
		numeric_columns = st.session_state.get( 'numeric_columns', [ ] ).copy( )
		categorical_columns = st.session_state.get( 'categorical_columns', [ ] ).copy( )
		
		st.session_state[ 'numeric_columns' ] = numeric_columns
		st.session_state[ 'categorical_columns' ] = categorical_columns
		if len( df_original.columns ) < 2:
			warn = '⚠️ Classification requires at least one feature column and one target column.'
			st.warning( warn )
			st.stop( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Samples: {len( df_original ):,} | Features: {len( df_original.columns ):,}' )
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_original.columns.tolist( ),
				key='classification_features' )
		
		with col_c2:
			target_options = [ column for column in df_original.columns if column not in features ]
			if target_options:
				targets = st.selectbox( 'Select Target', options=target_options,
					key='classification_target' )
			else:
				targets = None
				st.info( 'At least one column must remain available as the classification target.' )
		
		# Create Button
		sel_b1, sel_b2 = st.columns( [ 0.5, 0.5 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', icon='➕', key='classification_create_dataset',
					use_container_width=True, disabled=targets is None ):
				selected_all = features.copy( )
				if targets and targets not in selected_all:
					selected_all.append( targets )
				
				if selected_all:
					df_working = df_original[ selected_all ].copy( )
				else:
					df_working = df_original.copy( )
				
				st.session_state[ 'features' ] = features.copy( )
				st.session_state[ 'targets' ] = [ targets ] if targets else [ ]
				st.session_state[ 'df_working' ] = df_working.copy( )
				commit_frame( df_working )
				st.success( 'Working Dataset Created!' )
		
		# Reset Button
		with sel_b2:
			if st.button( 'Reset Working Dataset', icon='🔁', key='classification_reset_to_original',
					use_container_width=True ):
				st.session_state[ 'features' ] = [ ]
				st.session_state[ 'targets' ] = [ ]
				st.session_state[ 'df_working' ] = pd.DataFrame( )
				df_working = None
				df_processed = None
				st.success( 'Reset to Original' )
				st.rerun( )
		
		blue_divider( )
		if df_working is None:
			st.stop( )
		st.markdown( '##### Working Data' )
		
		st.caption( f'Samples: {len( df_working ):,} | Feautres: {len( df_working.columns ):,}' )
		render_data_editor( df_working, key='classification_working_data' )
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Feature-Engineering' )
		
		def has_classification_frame( df_frame: object ) -> bool:
			"""Determine whether an object contains usable classification data.

			Purpose:
			    Validates that the supplied object is a non-empty pandas dataframe containing at
			    least one column before it is used by the classification preprocessing pipeline.

			Args:
			    df_frame (object): Candidate object evaluated as a classification dataframe.

			Returns:
			    bool: ``True`` when the object is a non-empty dataframe with at least one column;
			    otherwise ``False``.
			"""
			return (isinstance( df_frame, pd.DataFrame ) and not df_frame.empty and len(
				df_frame.columns ) > 0)
				
		def get_classification_source_signature( df_frame: pd.DataFrame ) -> tuple:
			"""Create a signature for a classification working dataframe.

			Purpose:
			    Builds a stable comparison tuple from the dataframe column order, row count, and
			    first ten index values so the preprocessing workflow can detect changes to its
			    source dataset.

			Args:
			    df_frame (pd.DataFrame): Current classification working dataframe.

			Returns:
			    tuple: Source signature containing columns, row count, and sampled index values.
			"""
			index_sample = tuple( str( value ) for value in df_frame.index[ :10 ].tolist( ) )
			return (tuple( df_frame.columns.tolist( ) ), int( len( df_frame ) ), index_sample)
		
		def clear_classification_processed_state( ) -> None:
			"""Clear classification preprocessing outputs.

			Purpose:
			    Resets processed dataframes, active column selections, model inputs, train-test
			    partitions, predictions, model references, scores, and elapsed-time values without
			    modifying the classification working dataframe.

			Returns:
			    None: This function resets classification preprocessing values in Streamlit session
			    state.
			"""
			st.session_state[ 'df_processed' ] = pd.DataFrame( )
			st.session_state[ 'df_features' ] = pd.DataFrame( )
			st.session_state[ 'df_targets' ] = pd.DataFrame( )
			st.session_state[ 'active_features' ] = [ ]
			st.session_state[ 'active_targets' ] = [ ]
			st.session_state[ 'df_model' ] = pd.DataFrame( )
			st.session_state[ 'df_scores' ] = pd.DataFrame( )
			st.session_state[ 'df_predictions' ] = pd.DataFrame( )
			st.session_state[ 'X_data' ]=None
			st.session_state[ 'X_train' ]=None
			st.session_state[ 'X_test' ]=None
			st.session_state[ 'y_train' ]=None
			st.session_state[ 'y_test' ]=None
			st.session_state[ 'y_series' ]=None
			st.session_state[ 'y_prediction' ]=None
			st.session_state[ 'model' ]=None
			st.session_state[ 'elapsed_seconds' ] = 0.0
			
		def get_classification_pipeline_frame( ) -> pd.DataFrame:
			"""Return the active classification preprocessing dataframe.

			Purpose:
			    Returns an independent copy of the processed dataframe when valid processed data
			    exists. Otherwise, returns a copy of the current classification working dataframe.

			Returns:
			    pd.DataFrame: Active dataframe used as input to the next preprocessing operation.
			"""
			df_current = st.session_state.get( 'df_processed', pd.DataFrame( ) )
			if has_classification_frame( df_current ):
				return df_current.copy( )
			
			return st.session_state[ 'df_working' ].copy( )
			
		def get_classification_columns( numeric_only: bool = False,
			categorical_only: bool = False ) -> list[ str ]:
			"""Return selectable classification preprocessing columns.

			Purpose:
			    Retrieves column names from the active classification pipeline dataframe and
			    optionally restricts the result to numeric or non-numeric columns.

			Args:
			    numeric_only (bool): Flag indicating whether only numeric columns are returned.
			    categorical_only (bool): Flag indicating whether only non-numeric columns are
			        returned.

			Returns:
			    list[str]: Column names available for the requested preprocessing operation.
			"""
			df_input = get_classification_pipeline_frame( )
			if numeric_only:
				return [ col for col in df_input.columns if
					pd.api.types.is_numeric_dtype( df_input[ col ] ) ]
			
			if categorical_only:
				return [ col for col in df_input.columns if
					not pd.api.types.is_numeric_dtype( df_input[ col ] ) ]
			
			return df_input.columns.tolist( )
			
		def prune_classification_multiselect( key: str, options: list[ str ] ) -> None:
			"""Remove invalid values from a classification multiselect state key.

			Purpose:
			    Filters an existing list-valued Streamlit session-state entry so it contains only
			    values that remain available in the current widget options before the multiselect
			    control is instantiated.

			Args:
			    key (str): Streamlit session-state key associated with the multiselect widget.
			    options (list[str]): Current valid values accepted by the widget.

			Returns:
			    None: This function updates the specified session-state value in place.
			"""
			if key in st.session_state and isinstance( st.session_state[ key ], list ):
				st.session_state[ key ] = [ value for value in st.session_state[ key ] if
					value in options ]
		
		def prune_classification_selectbox( key: str, options: list[ str ] ) -> None:
			"""Remove an invalid classification selectbox state value.

			Purpose:
			    Deletes an existing Streamlit session-state entry when its selected value is no
			    longer present in the current selectbox options, allowing the widget to initialize
			    with a valid selection.

			Args:
			    key (str): Streamlit session-state key associated with the selectbox widget.
			    options (list[str]): Current valid values accepted by the widget.

			Returns:
			    None: This function may remove the specified session-state key.
			"""
			if key in st.session_state and st.session_state[ key ] not in options:
				del st.session_state[ key ]
		
		def get_valid_classification_columns( column_names: list[ str ],
			df_frame: pd.DataFrame ) -> list[ str ]:
			"""Return selected columns that remain available.

			Purpose:
			    Filters a requested column collection against the current classification dataframe
			    so downstream preprocessing operations receive only existing columns.

			Args:
			    column_names (list[str]): Selected column names evaluated for availability.
			    df_frame (pd.DataFrame): Current classification dataframe.

			Returns:
			    list[str]: Selected column names that exist in ``df_frame``.
			"""
			if not column_names:
				return [ ]
			
			return [ col for col in column_names if col in df_frame.columns ]
			
		def get_numeric_classification_subset( df_frame: pd.DataFrame,
			column_names: list[ str ] ) -> pd.DataFrame:
			"""Return a numeric classification dataframe subset.

			Purpose:
			    Selects the requested columns, coerces their values to numeric representations,
			    converts invalid values to missing values, and replaces resulting missing values
			    with zero for estimators that require fully numeric input.

			Args:
			    df_frame (pd.DataFrame): Classification dataframe containing the selected columns.
			    column_names (list[str]): Columns included in the numeric subset.

			Returns:
			    pd.DataFrame: Numeric dataframe containing the requested columns with invalid and
			    missing values replaced by zero.
			"""
			return df_frame[ column_names ].apply( pd.to_numeric, errors='coerce' ).fillna( 0.0 )
		
		def require_classification_columns( column_names: list[ str ], label: str ) -> bool:
			"""Validate that a classification operation has selected columns.

			Purpose:
			    Confirms that one or more valid columns are available for a preprocessing operation
			    and renders a Streamlit warning containing the operation label when no columns are
			    selected.

			Args:
			    column_names (list[str]): Selected column names evaluated for presence.
			    label (str): Operation label included in the warning message.

			Returns:
			    bool: ``True`` when at least one column is selected; otherwise ``False``.
			"""
			if not column_names:
				st.warning( f'Select at least one available column for {label}.' )
				return False
			
			return True
			
		def update_classification_pipeline_contract( df_before: pd.DataFrame, df_after: pd.DataFrame,
			source_columns: list[ str ], preserve_source_names: bool ) -> None:
			"""Update classification feature and target column contracts.

			Purpose:
			    Reconciles active feature and target selections after a preprocessing operation.
			    Existing names are retained when transformations preserve source columns; otherwise,
			    consumed source columns are replaced with newly generated output columns.

			Args:
			    df_before (pd.DataFrame): Classification dataframe before transformation.
			    df_after (pd.DataFrame): Classification dataframe after transformation.
			    source_columns (list[str]): Source columns consumed by the transformation.
			    preserve_source_names (bool): Flag indicating whether transformed columns retain
			        their original names.

			Returns:
			    None: This function updates feature and target selections in Streamlit session state.
			"""
			source_columns = source_columns or [ ]
			features_current = st.session_state.get( 'features', [ ] )
			targets_current = st.session_state.get( 'targets', [ ] )
			
			if preserve_source_names:
				st.session_state[ 'features' ] = [ col for col in features_current if
					col in df_after.columns ]
				st.session_state[ 'targets' ] = [ col for col in targets_current if
					col in df_after.columns ]
				return
			
			replacement_columns = [ col for col in df_after.columns if
				col not in df_before.columns ]
			
			if not replacement_columns:
				replacement_columns = [ col for col in df_after.columns if
					col not in [ old_col for old_col in df_before.columns if
						old_col not in source_columns ] ]
			
			features_next = [ ]
			for col in features_current:
				if col in source_columns:
					for new_col in replacement_columns:
						if new_col in df_after.columns and new_col not in features_next:
							features_next.append( new_col )
				elif col in df_after.columns and col not in features_next:
					features_next.append( col )
			
			targets_next = [ ]
			for col in targets_current:
				if col in source_columns:
					for new_col in replacement_columns:
						if new_col in df_after.columns and new_col not in targets_next:
							targets_next.append( new_col )
				elif col in df_after.columns and col not in targets_next:
					targets_next.append( col )
			
			st.session_state[ 'features' ] = features_next
			st.session_state[ 'targets' ] = targets_next
			
		def commit_classification_processed_frame( df_before: pd.DataFrame, df_after: pd.DataFrame,
			source_columns: list[ str ] | None = None, preserve_source_names: bool = True ) -> None:
			"""Commit a classification preprocessing result.

			Purpose:
			    Updates feature and target column contracts for the completed transformation,
			    stores an independent copy of the resulting dataframe as ``df_processed``, and
			    rebuilds the active feature and target dataframes without modifying ``df_working``.

			Args:
			    df_before (pd.DataFrame): Classification dataframe before transformation.
			    df_after (pd.DataFrame): Classification dataframe after transformation.
			    source_columns (list[str] | None): Source columns consumed by the transformation.
			    preserve_source_names (bool): Flag indicating whether transformed columns retain
			        their original names.

			Returns:
			    None: This function commits processed classification data and related session-state
			    contracts.
			"""
			update_classification_pipeline_contract( df_before=df_before, df_after=df_after,
				source_columns=source_columns or [ ], preserve_source_names=preserve_source_names )
			
			st.session_state[ 'df_processed' ] = df_after.copy( )
			commit_frame( df_after )
		
		df_working_current = st.session_state[ 'df_working' ].copy( )
		source_signature = get_classification_source_signature( df_working_current )
		prior_signature = st.session_state.get( 'classification_processing_source_signature',
			None )
		
		if prior_signature is not None and prior_signature != source_signature:
			clear_classification_processed_state( )
		
		st.session_state[ 'classification_processing_source_signature' ] = source_signature
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			# -----------------------------------------------------------------
			#
			# -----------------------------------------------------------------
			with st.expander( label='Data Scaling', icon='⚖️', key='classification_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.STANDARD_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_standard_scaler_cols',
						options )
					columns = st.multiselect( 'Columns', options=options,
						key='classification_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', use_container_width=True,
								key='classification_standard_scaler_apply' ):
							df_input = get_classification_pipeline_frame( )
							columns = get_valid_classification_columns( columns, df_input )
							if require_classification_columns( columns, 'Standard Scaler' ):
								scaler = StandardScaler( )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										columns ).to_numpy( ) )
								df_output[ columns ] = result
								commit_classification_processed_frame( df_input, df_output,
									columns,
									True )
								st.success( 'Standard Scaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_standard_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MINMAX_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_minmax_scaler_cols',
						options )
					scale_cols = st.multiselect( 'Columns', options=options,
						key='classification_minmax_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_minmax_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							scale_cols = get_valid_classification_columns( scale_cols, df_input )
							if require_classification_columns( scale_cols, 'Min-Max Scaler' ):
								scaler = MinMaxScaler( )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										scale_cols ).to_numpy( ) )
								df_output[ scale_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									scale_cols, True )
								st.success( 'Min-Max Scaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔄',
								key='classification_minmax_scaler_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.ROBUST_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_robust_scaler_cols',
						options )
					scale_cols = st.multiselect( 'Columns', options=options,
						key='classification_robust_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_robust_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							scale_cols = get_valid_classification_columns( scale_cols, df_input )
							if require_classification_columns( scale_cols, 'Robust Scaler' ):
								scaler = RobustScaler( )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										scale_cols ).to_numpy( ) )
								df_output[ scale_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									scale_cols, True )
								st.success( 'Robust Scaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_robust_scaler_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.NORMAL_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_normal_scaler_cols',
						options )
					scale_cols = st.multiselect( 'Columns', options=options,
						key='classification_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ], index=1,
						key='classification_normal_scaler_norm' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_normal_scaler_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							scale_cols = get_valid_classification_columns( scale_cols, df_input )
							if require_classification_columns( scale_cols, 'Normal Scaler' ):
								scaler = NormalScaler( norm=norm )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										scale_cols ).to_numpy( ) )
								df_output[ scale_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									scale_cols, True )
								st.success( 'NormalScaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_normal_scaler_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MAXABS_SCALER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_maxabs_scaler_cols',
						options )
					scale_cols = st.multiselect( 'Columns', options=options,
						key='classification_maxabs_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_maxabs_scaler_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							scale_cols = get_valid_classification_columns( scale_cols, df_input )
							if require_classification_columns( scale_cols, 'Max-Absolute Scaler' ):
								scaler = MaxAbsScaler( )
								df_output = df_input.copy( )
								result = scaler.train_transform(
									get_numeric_classification_subset( df_input,
										scale_cols ).to_numpy( ) )
								df_output[ scale_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									scale_cols, True )
								st.success( 'MaxAbsScaler applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_maxabs_scaler_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )

			# -----------------------------------------------------------------
			#
			# -----------------------------------------------------------------
			with st.expander( label='Data Imputation', icon='🧹', key='classification_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.MEAN_IMPUTER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_mean_imputer_cols', options )
					impute_cols = st.multiselect( 'Columns', options=options,
						key='classification_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='classification_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_mean_imputer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							impute_cols = get_valid_classification_columns( impute_cols, df_input )
							if require_classification_columns( impute_cols, 'Mean Imputer' ):
								imputer = MeanImputer( strategy='mean',
									add_indicator=add_indicator )
								result = imputer.train_transform(
									get_numeric_classification_subset( df_input,
										impute_cols ).to_numpy( ) )
								
								df_output = replace_columns( df_input, impute_cols, result,
									'mean_imputer' )
								commit_classification_processed_frame( df_input, df_output,
									impute_cols, not bool( add_indicator ) )
								st.success( 'MeanImputer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_mean_imputer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_nearest_imputer_cols',
						options )
					impute_cols = st.multiselect( 'Columns', options=options,
						key='classification_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1, value=5, step=1,
						key='classification_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_nearest_imputer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							impute_cols = get_valid_classification_columns( impute_cols, df_input )
							if require_classification_columns( impute_cols,
									'Nearest Neighbor Imputer' ):
								imputer = NearestImputer( neighbors=int( neighbors ) )
								result = imputer.train_transform(
									get_numeric_classification_subset( df_input,
										impute_cols ).to_numpy( ) )
								
								df_output = replace_columns( df_input, impute_cols, result,
									'nearest_imputer' )
								commit_classification_processed_frame( df_input, df_output,
									impute_cols, True )
								st.success( 'Nearest Imputer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_nearest_imputer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.ITERATIVE_IMPUTER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_iterative_imputer_cols',
						options )
					impute_cols = st.multiselect( 'Columns', options=options,
						key='classification_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=10, step=1,
						key='classification_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=0, step=1,
						key='classification_iterative_imputer_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer', icon='✔️',
								key='classification_iterative_imputer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							impute_cols = get_valid_classification_columns( impute_cols, df_input )
							if require_classification_columns( impute_cols, 'Iterative Imputer' ):
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									get_numeric_classification_subset( df_input,
										impute_cols ).to_numpy( ) )
								
								df_output = replace_columns( df_input, impute_cols, result,
									'iterative_imputer' )
								commit_classification_processed_frame( df_input, df_output,
									impute_cols, True )
								st.success( 'Iterative Imputer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_iterative_imputer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.SIMPLE_IMPUTER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_simple_imputer_cols',
						options )
					impute_cols = st.multiselect( 'Columns', options=options,
						key='classification_simple_imputer_cols' )
					
					strategy = st.selectbox( 'Strategy',
						options=[ 'mean', 'median', 'most_frequent', 'constant' ],
						key='classification_simple_imputer_strategy' )
					
					fill_value = st.text_input( 'Fill Value', value='0.0',
						key='classification_simple_imputer_fill_value' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='classification_simple_imputer_indicator' )
					
					keep_empty_features = st.checkbox( 'Keep Empty Features', value=False,
						key='classification_simple_imputer_keep_empty' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', icon='✔️',
								key='classification_simpleimputer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							impute_cols = get_valid_classification_columns( impute_cols, df_input )
							if require_classification_columns( impute_cols, 'Simple Imputer' ):
								if strategy in [ 'mean', 'median' ]:
									df_values = get_numeric_classification_subset( df_input,
										impute_cols )
									fill_object: object = 0.0
								else:
									df_values = df_input[ impute_cols ].copy( )
									fill_object = fill_value
								
								imputer = SimpleImputer( strategy=strategy, fill_value=fill_object,
									add_indicator=add_indicator,
									keep_empty_features=keep_empty_features )
								
								result = imputer.train_transform( df_values.to_numpy( ) )
								df_output = replace_columns( df_input, impute_cols, result,
									'simple_imputer' )
								commit_classification_processed_frame( df_input, df_output,
									impute_cols, not bool( add_indicator ) )
								st.success( 'Simple Imputer Applied' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_simple_imputer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )

			# -----------------------------------------------------------------
			#
			# -----------------------------------------------------------------
			with st.expander( label='Data Encoding', icon='🔣', key='classification_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ONEHOT_ENCODER )
					
					options = get_classification_columns( categorical_only=True )
					prune_classification_multiselect( 'classification_onehot_cols', options )
					encode_cols = st.multiselect( 'Columns', options=options,
						key='classification_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='classification_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0,
						key='classification_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_onehot_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							encode_cols = get_valid_classification_columns( encode_cols, df_input )
							if require_classification_columns( encode_cols, 'One-Hot Encoder' ):
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_input[ encode_cols ].fillna( '' ).astype( str ).to_numpy(
									
									) )
								df_output = replace_columns( df_input, encode_cols, result,
									'onehot' )
								commit_classification_processed_frame( df_input, df_output,
									encode_cols, False )
								st.success( 'One-Hot Encoder applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_onehot_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ORDINAL_ENCODER )
					
					options = get_classification_columns( categorical_only=True )
					prune_classification_multiselect( 'classification_ordinal_cols', options )
					encode_cols = st.multiselect( 'Columns', options=options,
						key='classification_ordinal_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_ordinal_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							encode_cols = get_valid_classification_columns( encode_cols, df_input )
							if require_classification_columns( encode_cols, 'Ordinal Encoder' ):
								encoder = OrdinalEncoder( )
								df_output = df_input.copy( )
								result = encoder.train_transform(
									df_output[ encode_cols ].fillna( '' ).astype(
										str ).to_numpy( ) )
								df_output[ encode_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									encode_cols, True )
								st.success( 'Ordinal Encoder Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_ordinal_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_ENCODER )
					
					options = get_classification_columns( )
					prune_classification_selectbox( 'classification_label_encoder_col', options )
					target_col = st.selectbox( 'Column', options=options,
						key='classification_label_encoder_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_label_encoder_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							if target_col and target_col in df_input.columns:
								encoder = LabelEncoder( )
								df_output = df_input.copy( )
								result = encoder.train_transform(
									df_output[ target_col ].fillna( '' ).astype( str ).to_numpy(
									
									) )
								df_output[ target_col ] = result
								commit_classification_processed_frame( df_input, df_output,
									[ target_col ], True )
								st.success( 'Label Encoder Applied.' )
								st.rerun( )
							else:
								st.warning( 'Select an available column for Label Encoder.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_label_encoder_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.TARGET_ENCODER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_target_encoder_cols',
						options )
					encode_cols = st.multiselect( 'Categorical Feature Columns', options=options,
						key='classification_target_encoder_cols' )
					
					prune_classification_selectbox( 'classification_target_encoder_target_col',
						options )
					target_col = st.selectbox( 'Target Column', options=options,
						key='classification_target_encoder_target_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_target_encoder_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							encode_cols = get_valid_classification_columns( encode_cols, df_input )
							if require_classification_columns( encode_cols, 'Target Encoder' ):
								if target_col and target_col in df_input.columns:
									encoder = TargetEncoder( )
									X_enc = df_input[ encode_cols ].fillna( '' ).astype(
										str ).to_numpy( )
									y_enc = df_input[ target_col ].to_numpy( )
									result = encoder.train_transform( X_enc, y_enc )
									df_output = replace_columns( df_input, encode_cols, result,
										'target_encoder' )
									commit_classification_processed_frame( df_input, df_output,
										encode_cols, False )
									st.success( 'Target Encoder Applied.' )
									st.rerun( )
								else:
									st.warning(
										'Select an available target column for Target Encoder.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_target_encoder_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.POLYNOMIAL_FEATURES )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_polynomial_cols', options )
					poly_cols = st.multiselect( 'Columns', options=options,
						key='classification_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4, value=2,
						key='classification_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='classification_polynomial_interaction' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_polynomial_apply', use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							poly_cols = get_valid_classification_columns( poly_cols, df_input )
							if require_classification_columns( poly_cols, 'Polynomial Features' ):
								encoder = PolynomialFeatures( degree=int( degree ),
									interaction=bool( interaction ) )
								result = encoder.train_transform(
									get_numeric_classification_subset( df_input,
										poly_cols ).to_numpy( ) )
								df_output = replace_columns( df_input, poly_cols, result,
									'polynomial' )
								commit_classification_processed_frame( df_input, df_output,
									poly_cols, False )
								st.success( 'PolynomialFeatures applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_polynomial_reset', use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
		
		with feature_c2:

			# -----------------------------------------------------------------
			#
			# -----------------------------------------------------------------
			with st.expander( label='Data Transformation', icon='⚡',
					key='classification_transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.BINARIZER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_binarizer_cols', options )
					transform_cols = st.multiselect( 'Columns', options=options,
						key='classification_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='classification_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='classification_binarizer_copy' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer', key='classification_binarizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							transform_cols = get_valid_classification_columns( transform_cols,
								df_input )
							if require_classification_columns( transform_cols, 'Binarizer' ):
								transformer = Binarizer( threshold=float( threshold ),
									copy=bool( copy ) )
								df_output = df_input.copy( )
								result = transformer.train_transform(
									get_numeric_classification_subset( df_input,
										transform_cols ).to_numpy( ) )
								df_output[ transform_cols ] = result
								commit_classification_processed_frame( df_input, df_output,
									transform_cols, True )
								st.success( 'Binarizer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_binarizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_BINARIZER )
					
					options = get_classification_columns( )
					prune_classification_selectbox( 'classification_label_binarizer_col', options )
					target_col = st.selectbox( 'Column', options=options,
						key='classification_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='classification_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='classification_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='classification_label_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer',
								key='classification_lblbinarizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							if target_col and target_col in df_input.columns:
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ),
									sparse_output=bool( sparse_output ) )
								result = transformer.train_transform(
									df_input[ target_col ].fillna( '' ).astype( str ).to_numpy( ) )
								df_output = replace_columns( df_input, [ target_col ], result,
									'label_binarizer' )
								preserve_name = np.asarray( result ).ndim == 1
								commit_classification_processed_frame( df_input, df_output,
									[ target_col ], preserve_name )
								st.success( 'Label Binarizer Applied.' )
								st.rerun( )
							else:
								st.warning( 'Select an available column for Label Binarizer.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_lblbinarizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.MULTILABEL_BINARIZER )
					
					options = get_classification_columns( )
					prune_classification_selectbox( 'classification_multilabel_binarizer_col',
						options )
					target_col = st.selectbox( 'Column', options=options,
						key='classification_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='classification_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='classification_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_multilabel_binarizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							if target_col and target_col in df_input.columns:
								y_multi = parse_multilabel_series( df_input[ target_col ],
									delimiter=delimiter )
								transformer = MultiLabelBinarizer( classes=None,
									sparse_output=bool( sparse_output ) )
								result = transformer.train_transform( y_multi )
								df_output = replace_columns( df_input, [ target_col ], result,
									'multilabel_binarizer' )
								commit_classification_processed_frame( df_input, df_output,
									[ target_col ], False )
								st.success( 'Multi-Label Binarizer Applied.' )
								st.rerun( )
							else:
								st.warning(
									'Select an available column for Multi-Label Binarizer.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_multilabel_binarizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_TRANSFORMER )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_tfidf_transformer_cols',
						options )
					text_count_cols = st.multiselect( 'Count Matrix Columns', options=options,
						key='classification_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ], index=1,
						key='classification_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='classification_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='classification_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='classification_tfidf_transformer_sublinear' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_tfidf_transformer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							text_count_cols = get_valid_classification_columns( text_count_cols,
								df_input )
							if require_classification_columns( text_count_cols,
									'TF-IDF Transformer' ):
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ),
									sublinear_tf=bool( sublinear_tf ) )
								result = transformer.train_transform(
									get_numeric_classification_subset( df_input,
										text_count_cols ).to_numpy( ) )
								df_output = replace_columns( df_input, text_count_cols, result,
									'tfidf_transformer' )
								commit_classification_processed_frame( df_input, df_output,
									text_count_cols, True )
								st.success( 'TFIDF Transformer Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_tfidf_transformer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Column Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COLUMN_TRANSFORMER )
					
					all_options = get_classification_columns( )
					df_column_input = get_classification_pipeline_frame( )
					numeric_options = [ col for col in all_options if
						pd.api.types.is_numeric_dtype( df_column_input[ col ] ) ]
					categorical_options = [ col for col in all_options if
						col not in numeric_options ]
					
					prune_classification_multiselect(
						'classification_column_transformer_numeric_columns', numeric_options )
					numeric_columns = st.multiselect( 'Numeric Columns', options=numeric_options,
						key='classification_column_transformer_numeric_columns' )
					
					prune_classification_multiselect(
						'classification_column_transformer_categorical_columns',
						categorical_options )
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=categorical_options,
						key='classification_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
							'Binarizer', 'None' ],
						key='classification_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='classification_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='classification_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0, max_value=1.0,
						value=0.3, key='classification_column_transformer_sparse_threshold' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer',
								key='classification_column_transformer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							numeric_columns = get_valid_classification_columns( numeric_columns,
								df_input )
							categorical_columns = get_valid_classification_columns(
								categorical_columns, df_input )
							
							transformers = [ ]
							if numeric_columns and numeric_transform != 'None':
								if numeric_transform == 'StandardScaler':
									numeric_model = StandardScaler( ).model
								elif numeric_transform == 'MinMaxScaler':
									numeric_model = MinMaxScaler( ).model
								elif numeric_transform == 'RobustScaler':
									numeric_model = RobustScaler( ).model
								elif numeric_transform == 'MaxAbsScaler':
									numeric_model = MaxAbsScaler( ).model
								else:
									numeric_model = Binarizer( ).model
								
								transformers.append( ('numeric', numeric_model, numeric_columns) )
							
							if categorical_columns and categorical_transform != 'None':
								if categorical_transform == 'OneHotEncoder':
									categorical_model = OneHotEncoder( sparse=False,
										unknown='ignore' ).model
								else:
									categorical_model = OrdinalEncoder( ).model
								
								transformers.append(
									('categorical', categorical_model, categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float(
										sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								result = transformer.train_transform( df_input )
								df_output = normalize_result_frame( result=result,
									index=df_input.index, prefix='column_transformer',
									columns=None )
								commit_classification_processed_frame( df_input, df_output,
									numeric_columns + categorical_columns, False )
								st.success( 'ColumnTransformer applied.' )
								st.rerun( )
							else:
								st.warning( 'Select at least one transformer and matching '
								            'column.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_column_transformer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )

			# -----------------------------------------------------------------
			#
			# -----------------------------------------------------------------
			with st.expander( label='Feature Extration', icon='⛏️',
					key='classification_extractors' ):
				
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_VECTORIZER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_tfidf_vectorizer_cols',
						options )
					text_cols = st.multiselect( 'Text Columns', options=options,
						key='classification_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='classification_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='classification_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_tfidf_vectorizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							text_cols = get_valid_classification_columns( text_cols, df_input )
							if require_classification_columns( text_cols, 'TF-IDF Vectorizer' ):
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), use_idf=bool( use_idf ) )
								df_output = apply_text_vectorizer( df_input, text_cols,
									transformer,
									'tfidf_vectorizer' )
								commit_classification_processed_frame( df_input, df_output,
									text_cols, False )
								st.success( 'TFIDF Vectorizer Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_tfidf_vectorizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COUNT_VECTORIZER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_count_vectorizer_cols',
						options )
					text_cols = st.multiselect( 'Text Columns', options=options,
						key='classification_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='classification_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='classification_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_count_vectorizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							text_cols = get_valid_classification_columns( text_cols, df_input )
							if require_classification_columns( text_cols, 'Count Vectorizer' ):
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), binary=bool( binary ) )
								df_output = apply_text_vectorizer( df_input, text_cols,
									transformer,
									'count_vectorizer' )
								commit_classification_processed_frame( df_input, df_output,
									text_cols, False )
								st.success( 'Count Vectorizer Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_count_vectorizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.HASH_VECTORIZER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_hash_vectorizer_cols',
						options )
					text_cols = st.multiselect( 'Text Columns', options=options,
						key='classification_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='classification_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='classification_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='classification_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='classification_hash_vectorizer_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_hash_vectorizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							text_cols = get_valid_classification_columns( text_cols, df_input )
							if require_classification_columns( text_cols, 'Hash Vectorizer' ):
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_output = apply_text_vectorizer( df_input, text_cols,
									transformer,
									'hash_vectorizer' )
								commit_classification_processed_frame( df_input, df_output,
									text_cols, False )
								st.success( 'HashVectorizer Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_hash_vectorizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.DICT_VECTORIZER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_dict_vectorizer_cols',
						options )
					dict_cols = st.multiselect( 'Columns', options=options,
						key='classification_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='classification_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='classification_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='classification_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_dict_vectorizer_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							dict_cols = get_valid_classification_columns( dict_cols, df_input )
							if require_classification_columns( dict_cols, 'Dictionary '
							                                              'Vectorizer' ):
								transformer = DictVectorizer( dtype=np.float64,
									separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								df_output = apply_dict_transform( df_input, dict_cols, transformer,
									'dict_vectorizer' )
								commit_classification_processed_frame( df_input, df_output,
									dict_cols, False )
								st.success( 'DictVectorizer applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_dict_vectorizer_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.FEATURE_HASHER )
					
					options = get_classification_columns( )
					prune_classification_multiselect( 'classification_feature_hasher_cols',
						options )
					hash_cols = st.multiselect( 'Columns', options=options,
						key='classification_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='classification_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='classification_feature_hasher_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_feature_hasher_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							hash_cols = get_valid_classification_columns( hash_cols, df_input )
							if require_classification_columns( hash_cols, 'Feature Hasher' ):
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								df_output = apply_dict_transform( df_input, hash_cols, transformer,
									'feature_hasher' )
								commit_classification_processed_frame( df_input, df_output,
									hash_cols, False )
								st.success( 'FeatureHasher applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_feature_hasher_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )

			# -----------------------------------------------------------------
			#
			# -----------------------------------------------------------------
			with st.expander( label='Dimensionality Reduction', icon='🎚️',
					key='classification_selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.VARIANCE_THRESHOLD )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_variance_threshold_cols',
						options )
					select_cols = st.multiselect( 'Columns', options=options,
						key='classification_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0, step=0.01,
						key='classification_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_variance_threshold_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							select_cols = get_valid_classification_columns( select_cols, df_input )
							if require_classification_columns( select_cols, 'Variance Threshold' ):
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									get_numeric_classification_subset( df_input,
										select_cols ).to_numpy( ) )
								df_output = replace_columns( df_input, select_cols, result,
									'variance_threshold' )
								preserve_name = np.asarray( result ).ndim == 2 and \
								                np.asarray( result ).shape[ 1 ] == len(
									select_cols )
								commit_classification_processed_frame( df_input, df_output,
									select_cols, preserve_name )
								st.success( 'VarianceThreshold applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_variance_threshold_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Canonical Correlation Analysis (CCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.CCA )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_cca_x_cols', options )
					X_cols = st.multiselect( 'Predictor Columns', options=options,
						key='classification_cca_x_cols' )
					
					prune_classification_multiselect( 'classification_cca_y_cols', options )
					y_cols = st.multiselect( 'Target Columns', options=options,
						key='classification_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='classification_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True, key='classification_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500, step=1,
						key='classification_cca_max_iter' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_cca_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							y_cols = get_valid_classification_columns( y_cols, df_input )
							if require_classification_columns( X_cols,
									'CCA predictors' ) and require_classification_columns( y_cols,
								'CCA targets' ):
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								result = selector.train_transform(
									get_numeric_classification_subset( df_input,
										X_cols ).to_numpy( ),
									get_numeric_classification_subset( df_input,
										y_cols ).to_numpy( ) )
								df_result = normalize_result_frame( result=result,
									index=df_input.index, prefix='cca', columns=None )
								df_output = pd.concat(
									[ df_input.drop( columns=X_cols + y_cols, errors='ignore' ),
										df_result ], axis=1 )
								commit_classification_processed_frame( df_input, df_output,
									X_cols + y_cols, False )
								st.success( 'CCA Applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_cca_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Principle Component Analysis (PCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.PCA )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_pca_cols', options )
					select_cols = st.multiselect( 'Columns', options=options,
						key='classification_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='classification_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='classification_pca_solver' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_pca_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							select_cols = get_valid_classification_columns( select_cols, df_input )
							if require_classification_columns( select_cols, 'PCA' ):
								selector = PCA( num=int( n_components ), solver=solver )
								result = selector.train_transform(
									get_numeric_classification_subset( df_input,
										select_cols ).to_numpy( ) )
								df_output = replace_columns( df_input, select_cols, result, 'pca' )
								commit_classification_processed_frame( df_input, df_output,
									select_cols, False )
								st.success( 'PCA applied.' )
								st.rerun( )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_pca_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Select-Best', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_BEST )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_selectbest_x_cols', options )
					X_cols = st.multiselect( 'Feature Columns', options=options,
						key='classification_selectbest_x_cols' )
					
					all_options = get_classification_columns( )
					prune_classification_selectbox( 'classification_selectbest_target_col',
						all_options )
					target_col = st.selectbox( 'Target Column', options=all_options,
						key='classification_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ],
						key='classification_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='classification_selectbest_k' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='classification_selectbest_apply', use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							if require_classification_columns( X_cols, 'Select Best' ):
								if target_col and target_col in df_input.columns:
									selector = SelectBest(
										score_func=score_function_from_name( score_name ),
										num=int( min( k_best, len( X_cols ) ) ) )
									result = selector.train_transform(
										get_numeric_classification_subset( df_input,
											X_cols ).to_numpy( ),
										df_input[ target_col ].to_numpy( ) )
									df_output = replace_columns( df_input, X_cols, result,
										'select_best' )
									commit_classification_processed_frame( df_input, df_output,
										X_cols, False )
									st.success( 'Select Best Applied.' )
									st.rerun( )
								else:
									st.warning(
										'Select an available target column for Select Best.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_selectbest_reset', use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Select-Percent', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_PERCENT )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_selectpercent_x_cols',
						options )
					X_cols = st.multiselect( 'Feature Columns', options=options,
						key='classification_selectpercent_x_cols' )
					
					all_options = get_classification_columns( )
					prune_classification_selectbox( 'classification_selectpercent_target_col',
						all_options )
					target_col = st.selectbox( 'Target Column', options=all_options,
						key='classification_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ],
						key='classification_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='classification_selectpercent_percentile' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent',
								key='classification_selectpercent_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							if require_classification_columns( X_cols, 'Select Percent' ):
								if target_col and target_col in df_input.columns:
									selector = SelectPercent(
										score_func=score_function_from_name( score_name ),
										pct=int( percentile ) )
									result = selector.train_transform(
										get_numeric_classification_subset( df_input,
											X_cols ).to_numpy( ),
										df_input[ target_col ].to_numpy( ) )
									df_output = replace_columns( df_input, X_cols, result,
										'select_percent' )
									commit_classification_processed_frame( df_input, df_output,
										X_cols, False )
									st.success( 'SelectPercent applied.' )
									st.rerun( )
								else:
									st.warning(
										'Select an available target column for Select Percent.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='classification_selectpercent_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Sequential Back Selection (SBS)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SBS )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_sbs_x_cols', options )
					X_cols = st.multiselect( 'Feature Columns', options=options,
						key='classification_sbs_x_cols' )
					
					all_options = get_classification_columns( )
					prune_classification_selectbox( 'classification_sbs_target_col', all_options )
					target_col = st.selectbox( 'Target Column', options=all_options,
						key='classification_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='classification_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='classification_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1, step=1,
						key='classification_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_sbs_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							if require_classification_columns( X_cols, 'SBS' ):
								if target_col and target_col in df_input.columns:
									selector = SBS( classifier=None,
										k_features=int( min( k_features, len( X_cols ) ) ),
										test_size=float( test_size ),
										random_state=int( random_state ) )
									X_input = get_numeric_classification_subset( df_input,
										X_cols ).to_numpy( )
									y_input = df_input[ target_col ].to_numpy( )
									selector.train( X_input, y_input )
									result = selector.transform( X_input )
									df_output = replace_columns( df_input, X_cols, result, 'sbs' )
									commit_classification_processed_frame( df_input, df_output,
										X_cols, False )
									st.success( 'SBS applied.' )
									st.rerun( )
								else:
									st.warning( 'Select an available target column for SBS.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_sbs_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
				
				with st.expander( 'Recursive Feature Elimination (RFA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.RFE )
					
					options = get_classification_columns( numeric_only=True )
					prune_classification_multiselect( 'classification_rfe_x_cols', options )
					X_cols = st.multiselect( 'Feature Columns', options=options,
						key='classification_rfe_x_cols' )
					
					all_options = get_classification_columns( )
					prune_classification_selectbox( 'classification_rfe_target_col', all_options )
					target_col = st.selectbox( 'Target Column', options=all_options,
						key='classification_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='classification_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0, step=1,
						key='classification_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='classification_rfe_apply',
								use_container_width=True ):
							df_input = get_classification_pipeline_frame( )
							X_cols = get_valid_classification_columns( X_cols, df_input )
							if require_classification_columns( X_cols, 'RFE' ):
								if target_col and target_col in df_input.columns:
									selector = RFE(
										k_features=int( min( k_features, len( X_cols ) ) ),
										verbose=int( verbose ) )
									X_input = get_numeric_classification_subset( df_input,
										X_cols ).to_numpy( )
									y_input = df_input[ target_col ].to_numpy( )
									selector.train( X_input, y_input )
									result = selector.transform( X_input )
									df_output = replace_columns( df_input, X_cols, result, 'rfe' )
									commit_classification_processed_frame( df_input, df_output,
										X_cols, False )
									st.success( 'RFE applied.' )
									st.rerun( )
								else:
									st.warning( 'Select an available target column for RFE.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='classification_rfe_reset',
								use_container_width=True ):
							clear_classification_processed_state( )
							st.success( 'Processed data cleared.' )
							st.rerun( )
		
		blue_divider( )
		df_processed = st.session_state.get( 'df_processed', pd.DataFrame( ) )
		if not has_classification_frame( df_processed ):
			st.info( 'No processed classification data has been created.' )
			st.stop( )
		
		st.markdown( '##### Processed Data' )
		st.caption(
			f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		render_data_editor( df_processed, key='classification_processed_data' )
		
		# ------------------------------------------------------------------
		# MODEL TRAINING
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Model Training', help=cfg.CLASSIFICATION_MODELS )
		
		active_features = [ ftr for ftr in st.session_state.get( 'features', [ ] ) if
			ftr in df_processed.columns ]
		
		active_targets = [ tgt for tgt in st.session_state.get( 'targets', [ ] ) if
			tgt in df_processed.columns ]
		
		st.session_state[ 'active_features' ] = active_features
		st.session_state[ 'active_targets' ] = active_targets
		if not active_features:
			warn = '⚠️ Classification training requires at least one processed feature column.'
			st.warning( warn )
			st.stop( )
		
		if len( active_targets ) != 1:
			st.warning( '⚠️ Classification models requires exactly one processed target column.' )
			st.stop( )
		
		target_name = active_targets[ 0 ]
		df_model = df_processed[ active_features + [ target_name ] ].copy( )
		df_model = df_model.dropna( subset=active_features + [ target_name ] ).copy( )
		st.session_state[ 'df_model' ] = df_model.copy( )
		X_data = df_model[ active_features ].copy( )
		st.session_state[ 'X_data' ] = X_data.copy( )
		for col in X_data.columns:
			X_data[ col ] = pd.to_numeric( X_data[ col ], errors='coerce' )
		
		if X_data.isna( ).any( ).any( ):
			st.warning( '⚠️ One or more feature columns are still non-numeric after preprocessing.'
			            'Apply the appropriate encoder/transformer before training.' )
			st.stop( )
		
		# Create training matrix and target vector
		X = X_data.to_numpy( dtype=float )
		y_series = df_model[ target_name ].copy( )
		st.session_state[ 'y_series' ] = y_series.copy( )
		if isinstance( y_series, pd.DataFrame ):
			st.warning( '⚠️ The processed target must resolve to a single column.' )
			st.stop( )
		
		y = y_series.to_numpy( )
		if y.ndim != 1:
			y = np.ravel( y )
		
		if len( y ) != len( X ):
			st.warning( '⚠️ Feature and target row counts do not match.' )
			st.stop( )
		
		if pd.api.types.is_numeric_dtype( y_series ):
			y_num = pd.to_numeric( y_series, errors='coerce' ).dropna( )
			if len( y_num ) == 0:
				st.warning( '⚠️ The processed target contains no valid values.' )
				st.stop( )
			
			unique_count = int( y_num.nunique( ) )
			unique_ratio = float( unique_count / max( 1, len( y_num ) ) )
			
			if unique_count > 20 and unique_ratio > 0.20:
				st.warning( '⚠️ The processed target appears continuous. '
				            'Use a label encoder/binarizer for classification targets, '
				            'not a scaler.' )
				
				st.stop( )
		
		class_counts = pd.Series( y ).value_counts( dropna=False )
		if len( class_counts ) < 2:
			st.warning( '⚠️ Classification requires at least two classes.' )
			st.stop( )
		
		df_classification = df_model.copy( )
		st.session_state[ 'df_classification' ] = df_classification.copy( )
		
		# ------------------------------------------------------------------
		# Classification Models
		# ------------------------------------------------------------------
		with st.expander( 'Linear Models', expanded=True ):
			
			with st.expander( 'Perceptron', expanded=True ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.PERCEPTRON_CLASSIFIER )
				
				perceptron_defaults = { 'classification_perceptron_alpha': 0.001000,
					'classification_perceptron_eta': 1.000000,
					'classification_perceptron_iters': 1000,
					'classification_perceptron_shuffle': False,
					'classification_perceptron_penalty': None,
					'classification_perceptron_test_size': 20,
					'classification_perceptron_random_state': 1 }
				
				def reset_perceptron_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Perceptron widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in perceptron_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in perceptron_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				per_c1, per_c2, per_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with per_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					perceptron_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_perceptron_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_perceptron_alpha' )
					
					perceptron_eta = st.number_input( 'Eta', min_value=0.000001,
						value=float( st.session_state[ 'classification_perceptron_eta' ] ),
						step=0.100000, format='%.6f', key='classification_perceptron_eta' )
					
					perceptron_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_perceptron_iters' ] ), step=1,
						key='classification_perceptron_iters' )
				
				with per_c2:
					st.markdown( '###### 🚦 Regularization / Split' )
					perceptron_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'classification_perceptron_shuffle' ] ),
						key='classification_perceptron_shuffle' )
					
					perceptron_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_perceptron_penalty' ] ),
						format_func=lambda value: 'None' if value is None else str( value ),
						key='classification_perceptron_penalty' )
					
					perceptron_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_perceptron_test_size' ] ),
						step=1, key='classification_perceptron_test_size' ) / 100.0
				
				with per_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					perceptron_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_perceptron_random_state' ] ),
						step=1, key='classification_perceptron_random_state' )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				per_btn_1, per_btn_2 = st.columns( 2 )
				with per_btn_1:
					train_perceptron = st.button( '🚂 Train Perceptron',
						key='classification_perceptron_train', use_container_width=True )
				
				with per_btn_2:
					st.button( '🔁 Reset Perceptron', key='classification_perceptron_reset',
						use_container_width=True, on_click=reset_perceptron_state )
				
				if train_perceptron:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Perceptron requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Perceptron requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Perceptron requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.Perceptron( alpha=float( perceptron_alpha ),
							eta=float( perceptron_eta ), iters=int( perceptron_iters ),
							shuffle=bool( perceptron_shuffle ), penalty=perceptron_penalty,
							random=int( perceptron_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( perceptron_test_size ),
							random=int( perceptron_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Perceptron training completed.' )
					except Exception as ex:
						st.error( f'Perceptron training failed: {ex}' )
			
			with st.expander( 'Ordinary Least Squares', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LEAST_SQUARES_CLASSIFIER )
				
				least_squares_defaults = { 'classification_least_squares_alpha': 0.000100,
					'classification_least_squares_eta': 0.010000,
					'classification_least_squares_iters': 1000,
					'classification_least_squares_shuffle': False,
					'classification_least_squares_penalty': 'l2',
					'classification_least_squares_test_size': 20,
					'classification_least_squares_random_state': 42 }
				
				def reset_least_squares_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Ordinary Least Squares widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in least_squares_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in least_squares_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ls_c1, ls_c2, ls_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ls_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					least_squares_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_least_squares_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_leastsquares_alpha' )
					
					least_squares_eta = st.number_input( 'Eta', min_value=0.000001,
						value=float( st.session_state[ 'classification_least_squares_eta' ] ),
						step=0.010000, format='%.6f', key='classification_leastsquares_eta' )
					
					least_squares_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_least_squares_iters' ] ),
						step=1, key='classification_leastsquares_iters' )
				
				with ls_c2:
					st.markdown( '###### 🚦 Regularization / Split' )
					least_squares_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'classification_least_squares_shuffle' ] ),
						key='classification_leastsquares_shuffle' )
					
					least_squares_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_least_squares_penalty' ] ),
						format_func=lambda value: 'None' if value is None else str( value ),
						key='classification_leastsquares_penalty' )
					
					least_squares_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30, step=1,
						value=int( st.session_state[ 'classification_least_squares_test_size' ] ),
						key='classification_leastsquares_test_size' ) / 100.0
				
				with ls_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					least_squares_random_state = st.number_input( 'Random State', min_value=0,
						value=int(
							st.session_state[ 'classification_least_squares_random_state' ] ),
						step=1, key='classification_leastsquares_random_state' )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				ls_btn_1, ls_btn_2 = st.columns( 2 )
				with ls_btn_1:
					train_least_squares = st.button( '🚂 Train Least Squares',
						key='classification_leastsquares_train', use_container_width=True )
				
				with ls_btn_2:
					st.button( '🔁 Reset Least Squares', key='classification_leastsquares_reset',
						use_container_width=True, on_click=reset_least_squares_state )
				
				if train_least_squares:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Least Squares requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Least Squares requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Least Squares requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.LeastSquares(
							alpha=float( least_squares_alpha ), eta=float( least_squares_eta ),
							iters=int( least_squares_iters ), shuffle=bool(
								least_squares_shuffle ),
							penalty=least_squares_penalty,
							random=int( least_squares_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( least_squares_test_size ),
							random=int( least_squares_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Least Squares training completed.' )
					
					except Exception as ex:
						st.error( f'Least Squares training failed: {ex}' )
			
			with st.expander( 'Logistic Regression', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LOGISTIC_REGRESSION )
				
				logistic_defaults = { 'classification_logistic_c': 1.000000,
					'classification_logistic_penalty': 'l2', 'classification_logistic_iters': 1000,
					'classification_logistic_multiclass': 'multinomial',
					'classification_logistic_solver': 'lbfgs',
					'classification_logistic_test_size': 20,
					'classification_logistic_random_state': 42 }
				
				for key, value in logistic_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				log_c1, log_c2, log_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with log_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					logistic_c = st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'classification_logistic_c' ] ),
						step=0.100000, format='%.6f', key='classification_logistic_c' )
					
					logistic_penalty = st.selectbox( 'Penalty',
						options=[ 'l2', 'l1', 'elasticnet', 'none', None ],
						index=[ 'l2', 'l1', 'elasticnet', 'none', None ].index(
							st.session_state[ 'classification_logistic_penalty' ] ),
						format_func=lambda v: 'None' if v is None else str( v ),
						key='classification_logistic_penalty' )
					
					logistic_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_logistic_iters' ] ), step=1,
						key='classification_logistic_iters' )
				
				with log_c2:
					st.markdown( '###### Strategy / Solver' )
					logistic_multiclass = st.selectbox( 'Multiclass',
						options=[ 'multinomial', 'ovr', 'auto' ],
						index=[ 'multinomial', 'ovr', 'auto' ].index(
							st.session_state[ 'classification_logistic_multiclass' ] ),
						key='classification_logistic_multiclass' )
					
					logistic_solver = st.selectbox( 'Solver',
						options=[ 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag',
							'saga' ],
						index=[ 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag',
							'saga' ].index( st.session_state[ 'classification_logistic_solver' ] ),
						key='classification_logistic_solver' )
					
					logistic_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_logistic_test_size' ] ),
						step=1, key='classification_logistic_test_size' ) / 100.0
				
				with log_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					logistic_random_state = st.number_input( 'Random State',
						value=int( st.session_state[ 'classification_logistic_random_state' ] ),
						step=1, key='classification_logistic_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# Model Training
				log_btn_1, log_btn_2 = st.columns( 2 )
				with log_btn_1:
					train_logistic = st.button( '🚂 Train Logistic Regression',
						key='classification_logistic_train', use_container_width=True )
				
				with log_btn_2:
					reset_logistic = st.button( '🔁 Reset Logistic Regression',
						key='classification_logistic_reset', use_container_width=True )
				
				if reset_logistic:
					for key, value in logistic_defaults.items( ):
						st.session_state[ key ] = value
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_model' ]=None
					st.session_state[ 'df_scores' ]=None
					st.session_state[ 'df_predictions' ]=None
					st.rerun( )
				
				if train_logistic:
					try:
						start_time = time.perf_counter( )
						model = classification_model.LogisticRegression( C=float( logistic_c ),
							penalty=logistic_penalty, iters=int( logistic_iters ),
							multiclass=str( logistic_multiclass ), solver=str( logistic_solver ),
							random=int( logistic_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( logistic_test_size ), random=int( logistic_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test ).copy( )
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': y_test, 'Predicted': y_prediction } )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
					except Exception as ex:
						st.error( f'Logistic Regression training failed: {ex}' )
			
			with st.expander( 'Ridge Classification', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.RIDGE_CLASSIFIER )
				
				ridge_defaults = { 'classification_ridge_alpha': 1.000000,
					'classification_ridge_solver': 'auto', 'classification_ridge_iters': 1000,
					'classification_ridge_test_size': 20, 'classification_ridge_random_state': 42 }
				
				def reset_ridge_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Ridge Classification widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in ridge_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in ridge_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ridge_c1, ridge_c2, ridge_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ridge_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					ridge_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_ridge_alpha' ] ),
						step=0.100000, format='%.6f', key='classification_ridge_alpha' )
					
					ridge_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_ridge_iters' ] ), step=1,
						key='classification_ridge_iters' )
				
				with ridge_c2:
					st.markdown( '###### Solver / Split' )
					ridge_solver = st.selectbox( 'Solver',
						options=[ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga',
							'lbfgs' ],
						index=[ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga',
							'lbfgs' ].index( st.session_state[ 'classification_ridge_solver' ] ),
						key='classification_ridge_solver' )
					
					ridge_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_ridge_test_size' ] ), step=1,
						key='classification_ridge_test_size' ) / 100.0
				
				with ridge_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					ridge_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_ridge_random_state' ] ),
						step=1, key='classification_ridge_random_state' )
					
					st.caption( f'Samples: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				ridge_btn_1, ridge_btn_2 = st.columns( 2 )
				with ridge_btn_1:
					train_ridge = st.button( '🚂 Train Ridge', key='classification_ridge_train',
						use_container_width=True )
				
				with ridge_btn_2:
					st.button( '🔁 Reset Ridge', key='classification_ridge_reset',
						use_container_width=True, on_click=reset_ridge_state )
				
				if train_ridge:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Ridge Classification requires prepared feature and target '
								'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning(
								'⚠️ Ridge Classification requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Ridge Classification requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.Ridge( alpha=float( ridge_alpha ),
							solver=str( ridge_solver ), iters=int( ridge_iters ),
							rando=int( ridge_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( ridge_test_size ), random=int( ridge_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Ridge Classification training completed.' )
					
					except Exception as ex:
						st.error( f'Ridge training failed: {ex}' )
			
			with st.expander( 'Lasso Classification', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.LASSO_CLASSIFIER )
				
				lasso_defaults = { 'classification_lasso_alpha': 1.000000,
					'classification_lasso_iters': 500, 'classification_lasso_threshold': 0.500000,
					'classification_lasso_selection': 'random',
					'classification_lasso_test_size': 20, 'classification_lasso_random_state': 42 }
				
				def reset_lasso_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Lasso Classification widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in lasso_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in lasso_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				lasso_c1, lasso_c2, lasso_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with lasso_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					lasso_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_lasso_alpha' ] ),
						step=0.100000, format='%.6f', key='classification_lasso_alpha' )
					
					lasso_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_lasso_iters' ] ), step=1,
						key='classification_lasso_iters' )
					
					lasso_threshold = st.number_input( 'Threshold', min_value=0.000000,
						max_value=1.000000,
						value=float( st.session_state[ 'classification_lasso_threshold' ] ),
						step=0.050000, format='%.6f', key='classification_lasso_threshold' )
				
				with lasso_c2:
					st.markdown( '###### ↔️ Selection / Split' )
					lasso_selection = st.selectbox( 'Selection', options=[ 'cyclic', 'random' ],
						index=[ 'cyclic', 'random' ].index(
							st.session_state[ 'classification_lasso_selection' ] ),
						key='classification_lasso_selection' )
					
					lasso_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_lasso_test_size' ] ), step=1,
						key='classification_lasso_test_size' ) / 100.0
				
				with lasso_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					lasso_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_lasso_random_state' ] ),
						step=1, key='classification_lasso_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				lasso_btn_1, lasso_btn_2 = st.columns( 2 )
				with lasso_btn_1:
					train_lasso = st.button( '🚂 Train Lasso', key='classification_lasso_train',
						use_container_width=True )
				
				with lasso_btn_2:
					st.button( '🔁 Reset Lasso', key='classification_lasso_reset',
						use_container_width=True, on_click=reset_lasso_state )
				
				if train_lasso:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Lasso Classification requires prepared feature and target '
								'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning(
								'⚠️ Lasso Classification requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Lasso Classification requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.Lasso( alpha=float( lasso_alpha ),
							iters=int( lasso_iters ), rando=int( lasso_random_state ),
							threshold=float( lasso_threshold ), selection=str( lasso_selection ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( lasso_test_size ), random=int( lasso_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Lasso Classification training completed.' )
					
					except Exception as ex:
						st.error( f'Lasso training failed: {ex}' )
			
			with st.expander( 'Gradient Descent', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.GRADIENT_DESCENT_CLASSIFIER )
				
				gradient_defaults = { 'classification_gradient_loss': 'hinge',
					'classification_gradient_penalty': 'l2',
					'classification_gradient_alpha': 0.000100,
					'classification_gradient_iters': 1000, 'classification_gradient_shuffle': True,
					'classification_gradient_eta': 0.010000,
					'classification_gradient_learning': 'optimal',
					'classification_gradient_power': 0.500000,
					'classification_gradient_epsilon': 0.100000,
					'classification_gradient_test_size': 20,
					'classification_gradient_random_state': 42 }
				
				def reset_gradient_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Gradient Descent widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in gradient_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in gradient_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				gd_c1, gd_c2, gd_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with gd_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					gradient_loss = st.selectbox( 'Loss',
						options=[ 'hinge', 'log_loss', 'modified_huber', 'squared_hinge',
							'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',
							'squared_epsilon_insensitive' ],
						index=[ 'hinge', 'log_loss', 'modified_huber', 'squared_hinge',
							'perceptron', 'squared_error', 'huber', 'epsilon_insensitive',
							'squared_epsilon_insensitive' ].index(
							st.session_state[ 'classification_gradient_loss' ] ),
						key='classification_gradient_loss' )
					
					gradient_penalty = st.selectbox( 'Penalty',
						options=[ None, 'l2', 'l1', 'elasticnet' ],
						index=[ None, 'l2', 'l1', 'elasticnet' ].index(
							st.session_state[ 'classification_gradient_penalty' ] ),
						format_func=lambda value: 'None' if value is None else str( value ),
						key='classification_gradient_penalty' )
					
					gradient_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_gradient_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_gradient_alpha' )
					
					gradient_iters = st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'classification_gradient_iters' ] ), step=1,
						key='classification_gradient_iters' )
				
				with gd_c2:
					st.markdown( '###### Learning / Split' )
					gradient_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'classification_gradient_shuffle' ] ),
						key='classification_gradient_shuffle' )
					
					gradient_eta = st.number_input( 'Eta', min_value=0.000001,
						value=float( st.session_state[ 'classification_gradient_eta' ] ),
						step=0.010000, format='%.6f', key='classification_gradient_eta' )
					
					gradient_learning = st.selectbox( 'Learning Rate Schedule',
						options=[ 'constant', 'optimal', 'invscaling', 'adaptive' ],
						index=[ 'constant', 'optimal', 'invscaling', 'adaptive' ].index(
							st.session_state[ 'classification_gradient_learning' ] ),
						key='classification_gradient_learning' )
					
					gradient_power = st.number_input( 'Power T', min_value=0.000000,
						value=float( st.session_state[ 'classification_gradient_power' ] ),
						step=0.100000, format='%.6f', key='classification_gradient_power' )
					
					gradient_epsilon = st.number_input( 'Epsilon', min_value=0.000000,
						value=float( st.session_state[ 'classification_gradient_epsilon' ] ),
						step=0.010000, format='%.6f', key='classification_gradient_epsilon' )
				
				with gd_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					gradient_test_size = st.slider( 'Test Set Size (%)', min_value=10,
						max_value=30,
						value=int( st.session_state[ 'classification_gradient_test_size' ] ),
						step=1, key='classification_gradient_test_size' ) / 100.0
					
					gradient_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_gradient_random_state' ] ),
						step=1, key='classification_gradient_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				gd_btn_1, gd_btn_2 = st.columns( 2 )
				with gd_btn_1:
					train_gradient = st.button( '🚂 Train Gradient Descent',
						key='classification_gradient_train', use_container_width=True )
				
				with gd_btn_2:
					st.button( '🔁 Reset Gradient Descent', key='classification_gradient_reset',
						use_container_width=True, on_click=reset_gradient_state )
				
				if train_gradient:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Gradient Descent requires prepared feature and target '
							            'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning(
								'⚠️ Gradient Descent requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Gradient Descent requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.GradientDescent( loss=str( gradient_loss ),
							penalty=gradient_penalty, alpha=float( gradient_alpha ),
							iters=int( gradient_iters ), shuffle=bool( gradient_shuffle ),
							eta=float( gradient_eta ), learning=str( gradient_learning ),
							power=float( gradient_power ), epsilon=float( gradient_epsilon ),
							rando=int( gradient_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( gradient_test_size ), random=int( gradient_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Gradient Descent training completed.' )
					
					except Exception as ex:
						st.error( f'Gradient Descent training failed: {ex}' )
		
		# -----------------------------------------------------------------
		# Instance Models
		# -----------------------------------------------------------------
		with st.expander( 'Instance Models', expanded=True ):
			
			with st.expander( 'Nearest Neighbor', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.NEAREST_NEIGHBOR_CLASSFIER )
				
				nearest_defaults = { 'classification_nearest_num': 5,
					'classification_nearest_algorithm': 'auto', 'classification_nearest_power': 2,
					'classification_nearest_metric': 'minkowski',
					'classification_nearest_leafs': 30, 'classification_nearest_test_size': 20,
					'classification_nearest_random_state': 42 }
				
				def reset_nearest_state( ) -> None:
					"""
						
							Purpose:
							--------
							Reset Nearest Neighbor widget values and model outputs before
							widget instantiation on the next Streamlit run.
						
							Parameters:
							-----------
							None
						
							Returns:
							--------
							None
						
						"""
					for reset_key, reset_value in nearest_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in nearest_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				nn_c1, nn_c2, nn_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with nn_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					nearest_num = st.number_input( 'Neighbors', min_value=1,
						value=int( st.session_state[ 'classification_nearest_num' ] ), step=1,
						key='classification_nearest_num' )
					
					nearest_power = st.number_input( 'Power', min_value=1,
						value=int( st.session_state[ 'classification_nearest_power' ] ), step=1,
						key='classification_nearest_power' )
					
					nearest_leafs = st.number_input( 'Leaf Size', min_value=1,
						value=int( st.session_state[ 'classification_nearest_leafs' ] ), step=1,
						key='classification_nearest_leafs' )
				
				with nn_c2:
					st.markdown( '###### Distance / Search' )
					nearest_algorithm = st.selectbox( 'Algorithm',
						options=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ],
						index=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ].index(
							st.session_state[ 'classification_nearest_algorithm' ] ),
						key='classification_nearest_algorithm' )
					
					nearest_metric = st.selectbox( 'Metric',
						options=[ 'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'hamming',
							'canberra', 'braycurtis', 'cityblock', 'cosine', 'l1', 'l2',
							'nan_euclidean', 'mahalanobis', 'seuclidean' ],
						index=[ 'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'hamming',
							'canberra', 'braycurtis', 'cityblock', 'cosine', 'l1', 'l2',
							'nan_euclidean', 'mahalanobis', 'seuclidean' ].index(
							st.session_state[ 'classification_nearest_metric' ] ),
						key='classification_nearest_metric' )
					
					nearest_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_nearest_test_size' ] ),
						step=1,
						key='classification_nearest_test_size' ) / 100.0
				
				with nn_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					nearest_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_nearest_random_state' ] ),
						step=1, key='classification_nearest_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				nn_btn_1, nn_btn_2 = st.columns( 2, border=True )
				with nn_btn_1:
					train_nearest = st.button( '🚂 Train Nearest Neighbor',
						key='classification_nearest_train', use_container_width=True )
				
				with nn_btn_2:
					st.button( '🔁 Reset Nearest Neighbor', key='classification_nearest_reset',
						use_container_width=True, on_click=reset_nearest_state )
				
				if train_nearest:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Nearest Neighbor requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Nearest Neighbor requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Nearest Neighbor requires at least two target classes.' )
							st.stop( )
						
						if int( nearest_num ) > len( X ):
							st.warning( '⚠️ Neighbors cannot exceed the number of available samples.' )
							st.stop( )
						
						if str( nearest_metric ) in [ 'mahalanobis', 'seuclidean' ]:
							st.warning( '⚠️ The selected metric requires additional parameters. '
								'Use minkowski, euclidean, manhattan, cityblock, l1, l2, '
								'cosine, chebyshev, hamming, canberra, braycurtis, or '
								'nan_euclidean for this UI path.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.NearestNeighbor( num=int( nearest_num ),
							algorithm=str( nearest_algorithm ), power=int( nearest_power ),
							metric=str( nearest_metric ), leafs=int( nearest_leafs ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( nearest_test_size ), random=int( nearest_random_state ) )
						
						if int( nearest_num ) > len( X_train ):
							st.warning( '⚠️ Neighbors cannot exceed the number of training '
							            'samples after the train/test split.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Nearest Neighbor training completed.' )
					except Exception as ex:
						st.error( f'Nearest Neighbor training failed: {ex}' )
			
			with st.expander( 'Support Vector Machine', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.SUPPORT_VECTOR_CLASSIFIER )
				
				svm_defaults = { 'classification_svm_c': 1.000000,
					'classification_svm_kernel': 'rbf', 'classification_svm_degree': 3,
					'classification_svm_test_size': 20, 'classification_svm_random_state': 42 }
				
				def reset_svm_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Support Vector Machine widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in svm_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in svm_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				svm_c1, svm_c2, svm_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with svm_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					svm_c = st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'classification_svm_c' ] ), step=0.100000,
						format='%.6f', key='classification_svm_c' )
					
					svm_kernel = st.selectbox( 'Kernel',
						options=[ 'linear', 'poly', 'rbf', 'sigmoid' ],
						index=[ 'linear', 'poly', 'rbf', 'sigmoid' ].index(
							st.session_state[ 'classification_svm_kernel' ] ) if \
							st.session_state[ 'classification_svm_kernel' ] in [ 'linear',  'poly',
								'rbf', 'sigmoid' ] else 2, key='classification_svm_kernel' )
					
					svm_degree = st.number_input( 'Degree', min_value=1,
						value=int( st.session_state[ 'classification_svm_degree' ] ), step=1,
						key='classification_svm_degree' )
				
				with svm_c2:
					st.markdown( '###### ↔️ Split' )
					svm_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_svm_test_size' ] ), step=1,
						key='classification_svm_test_size' ) / 100.0
					
					if svm_kernel != 'poly':
						st.caption( 'Degree is only used when kernel = poly.' )
				
				with svm_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					svm_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_svm_random_state' ] ), step=1,
						key='classification_svm_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				svm_btn_1, svm_btn_2 = st.columns( 2 )
				with svm_btn_1:
					train_svm = st.button( '🚂 Train Support Vector',
						key='classification_svm_train',
						use_container_width=True )
				
				with svm_btn_2:
					st.button( '🔁 Reset Support Vector', key='classification_svm_reset',
						use_container_width=True, on_click=reset_svm_state )
				
				if train_svm:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Support Vector Machine requires prepared feature and target '
								'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Support Vector Machine requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Support Vector Machine requires at least two target classes.' )
							st.stop( )
						
						if str( svm_kernel ) == 'precomputed':
							st.warning(
								'⚠️ The precomputed kernel requires a square kernel matrix and is '
								'not supported by this Classification UI path.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.SupportVector( C=float( svm_c ),
							kernel=str( svm_kernel ), degree=int( svm_degree ),
							random=int( svm_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( svm_test_size ), random=int( svm_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Support Vector Machine training completed.' )
					
					except Exception as ex:
						st.error( f'Support Vector training failed: {ex}' )
		
		# -----------------------------------------------------------------
		# Tree Models
		# -----------------------------------------------------------------
		with st.expander( 'Tree Models', expanded=True ):
			
			with st.expander( 'Decision Tree', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.DESICION_TREE_CLASSIFIER )
				
				tree_defaults = { 'classification_tree_criterion': 'gini',
					'classification_tree_splitter': 'best', 'classification_tree_depth': 0,
					'classification_tree_min_split': 2, 'classification_tree_min_leaf': 1,
					'classification_tree_test_size': 20, 'classification_tree_random_state': 42 }
				
				def reset_tree_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Decision Tree widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in tree_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'y_predictions' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in tree_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				tree_c1, tree_c2, tree_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with tree_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					tree_criterion = st.selectbox( 'Criterion',
						options=[ 'gini', 'entropy', 'log_loss' ],
						index=[ 'gini', 'entropy', 'log_loss' ].index(
							st.session_state[ 'classification_tree_criterion' ] ),
						key='classification_tree_criterion' )
					
					tree_splitter = st.selectbox( 'Splitter', options=[ 'best', 'random' ],
						index=[ 'best', 'random' ].index(
							st.session_state[ 'classification_tree_splitter' ] ),
						key='classification_tree_splitter' )
					
					tree_depth = st.number_input( 'Max Depth (0 = None)', min_value=0,
						value=int( st.session_state[ 'classification_tree_depth' ] ), step=1,
						key='classification_tree_depth' )
				
				with tree_c2:
					st.markdown( '###### Node Constraints' )
					tree_min_split = st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'classification_tree_min_split' ] ), step=1,
						key='classification_tree_min_split' )
					
					tree_min_leaf = st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'classification_tree_min_leaf' ] ), step=1,
						key='classification_tree_min_leaf' )
					
					tree_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_tree_test_size' ] ), step=1,
						key='classification_tree_test_size' ) / 100.0
				
				with tree_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					tree_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_tree_random_state' ] ),
						step=1,
						key='classification_tree_random_state' )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				tree_btn_1, tree_btn_2 = st.columns( 2 )
				with tree_btn_1:
					train_tree = st.button( '🚂 Train Decision Tree',
						key='classification_tree_train', use_container_width=True )
				
				with tree_btn_2:
					st.button( '🔁 Reset Decision Tree', key='classification_tree_reset',
						use_container_width=True, on_click=reset_tree_state )
				
				if train_tree:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Decision Tree requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Decision Tree requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Decision Tree requires at least two target classes.' )
							st.stop( )
						
						if int( tree_min_leaf ) > int( tree_min_split ):
							st.warning( '⚠️ Min Samples Leaf should not exceed Min Samples Split '
							            'for this UI path.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.DecisionTree( criterion=str( tree_criterion ),
							splitter=str( tree_splitter ),
							depth=None if int( tree_depth ) == 0 else int( tree_depth ),
							min_split=int( tree_min_split ), min_leaf=int( tree_min_leaf ),
							random=int( tree_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( tree_test_size ), random=int( tree_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Decision Tree training completed.' )
					
					except Exception as ex:
						st.error( f'Decision Tree training failed: {ex}' )
			
			with st.expander( 'Random Forest', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.RANDOM_FOREST_CLASSIFIER )
				
				forest_defaults = { 'classification_forest_estimators': 100,
					'classification_forest_criterion': 'gini', 'classification_forest_depth': 0,
					'classification_forest_min_split': 2, 'classification_forest_min_leaf': 1,
					'classification_forest_test_size': 20,
					'classification_forest_random_state': 42 }
				
				def reset_forest_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Random Forest widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in forest_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in forest_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				forest_c1, forest_c2, forest_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with forest_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					forest_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_forest_estimators' ] ),
						step=1,
						key='classification_forest_estimators' )
					
					forest_criterion = st.selectbox( 'Criterion',
						options=[ 'gini', 'entropy', 'log_loss' ],
						index=[ 'gini', 'entropy', 'log_loss' ].index(
							st.session_state[ 'classification_forest_criterion' ] ),
						key='classification_forest_criterion' )
					
					forest_depth = st.number_input( 'Max Depth (0 = None)', min_value=0,
						value=int( st.session_state[ 'classification_forest_depth' ] ), step=1,
						key='classification_forest_depth' )
				
				with forest_c2:
					st.markdown( '###### Node Constraints' )
					forest_min_split = st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'classification_forest_min_split' ] ), step=1,
						key='classification_forest_min_split' )
					
					forest_min_leaf = st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'classification_forest_min_leaf' ] ), step=1,
						key='classification_forest_min_leaf' )
					
					forest_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_forest_test_size' ] ), step=1,
						key='classification_forest_test_size' ) / 100.0
				
				with forest_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					forest_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_forest_random_state' ] ),
						step=1, key='classification_forest_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				forest_btn_1, forest_btn_2 = st.columns( 2 )
				with forest_btn_1:
					train_forest = st.button( '🚂 Train Random Forest',
						key='classification_forest_train', use_container_width=True )
				
				with forest_btn_2:
					st.button( '🔁 Reset Random Forest', key='classification_forest_reset',
						use_container_width=True, on_click=reset_forest_state )
				
				if train_forest:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Random Forest requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Random Forest requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Random Forest requires at least two target classes.' )
							st.stop( )
						
						if int( forest_min_leaf ) > int( forest_min_split ):
							st.warning( '⚠️ Min Samples Leaf should not exceed Min Samples Split '
							            'for this UI path.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.RandomForest(
							n_estimators=int( forest_estimators ),
							criterion=str( forest_criterion ),
							depth=None if int( forest_depth ) == 0 else int( forest_depth ),
							min_split=int( forest_min_split ), min_leaf=int( forest_min_leaf ),
							random=int( forest_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( forest_test_size ), random=int( forest_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Random Forest training completed.' )
					
					except Exception as ex:
						st.error( f'Random Forest training failed: {ex}' )
		
		# -----------------------------------------------------------------
		# Ensemble Models
		# -----------------------------------------------------------------
		with st.expander( 'Ensemble Models', expanded=False ):
			
			with st.expander( 'Gradient Boost', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.GRADIENT_BOOST_CLASSIFIER )
				
				gb_defaults = { 'classification_gb_estimators': 100,
					'classification_gb_rate': 0.100000, 'classification_gb_depth': 3,
					'classification_gb_criterion': 'friedman_mse',
					'classification_gb_test_size': 20, 'classification_gb_random_state': 42 }
				
				def reset_gb_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Gradient Boost widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in gb_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'y_predictions' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in gb_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				gb_c1, gb_c2, gb_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with gb_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					gb_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_gb_estimators' ] ), step=1,
						key='classification_gb_estimators' )
					
					gb_rate = st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'classification_gb_rate' ] ), step=0.010000,
						format='%.6f', key='classification_gb_rate' )
					
					gb_depth = st.number_input( 'Max Depth', min_value=1,
						value=int( st.session_state[ 'classification_gb_depth' ] ), step=1,
						key='classification_gb_depth' )
				
				with gb_c2:
					st.markdown( '###### Criterion / Split' )
					gb_criterion = st.selectbox( 'Criterion',
						options=[ 'friedman_mse', 'squared_error' ],
						index=[ 'friedman_mse', 'squared_error' ].index(
							st.session_state[ 'classification_gb_criterion' ] ),
						key='classification_gb_criterion' )
					
					gb_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_gb_test_size' ] ), step=1,
						key='classification_gb_test_size' ) / 100.0
				
				with gb_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					gb_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_gb_random_state' ] ), step=1,
						key='classification_gb_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				gb_btn_1, gb_btn_2 = st.columns( 2 )
				with gb_btn_1:
					train_gb = st.button( '🚂 Train Gradient Boost', key='classification_gb_train',
						use_container_width=True )
				
				with gb_btn_2:
					st.button( '🔁 Reset Gradient Boost', key='classification_gb_reset',
						use_container_width=True, on_click=reset_gb_state )
				
				if train_gb:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Gradient Boost requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Gradient Boost requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Gradient Boost requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.GradientBoost( estimators=int(
							gb_estimators ),
							rate=float( gb_rate ), depth=int( gb_depth ),
							criterion=str( gb_criterion ), random=int( gb_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( gb_test_size ), random=int( gb_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Gradient Boost training completed.' )
					except Exception as ex:
						st.error( f'Gradient Boost training failed: {ex}' )
			
			with st.expander( 'Adaptive Boost', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.ADAPTIVE_BOOST_CLASSIFIER )
				
				ab_defaults = { 'classification_ab_estimators': 50,
					'classification_ab_rate': 1.000000, 'classification_ab_algorithm': 'SAMME',
					'classification_ab_test_size': 20, 'classification_ab_random_state': 42 }
				
				def reset_ab_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Adaptive Boost widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in ab_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'y_predictions' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in ab_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				ab_c1, ab_c2, ab_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ab_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					ab_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_ab_estimators' ] ), step=1,
						key='classification_ab_estimators' )
					
					ab_rate = st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'classification_ab_rate' ] ), step=0.010000,
						format='%.6f', key='classification_ab_rate' )
				
				with ab_c2:
					st.markdown( '###### ♟️ Algorithm / Split' )
					ab_algorithm = st.selectbox( 'Algorithm', options=[ 'SAMME' ], index=0,
						key='classification_ab_algorithm' )
					
					ab_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_ab_test_size' ] ), step=1,
						key='classification_ab_test_size' ) / 100.0
				
				with ab_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					ab_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_ab_random_state' ] ), step=1,
						key='classification_ab_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				ab_btn_1, ab_btn_2 = st.columns( 2 )
				with ab_btn_1:
					train_ab = st.button( '🚂 Train Adaptive Boost', key='classification_ab_train',
						use_container_width=True )
				
				with ab_btn_2:
					st.button( '🔁 Reset Adaptive Boost', key='classification_ab_reset',
						use_container_width=True, on_click=reset_ab_state )
				
				if train_ab:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Adaptive Boost requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Adaptive Boost requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Adaptive Boost requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = classification_model.AdaptiveBoost( base=None,
							estimators=int( ab_estimators ), rate=float( ab_rate ),
							algorithm=str( ab_algorithm ), random=int( ab_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( ab_test_size ), random=int( ab_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Adaptive Boost training completed.' )
					
					except Exception as ex:
						st.error( f'Adaptive Boost training failed: {ex}' )
			
			with st.expander( 'Bagging Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.BAGGING_CLASSIFIER )
				
				bag_defaults = { 'classification_bag_estimators': 50,
					'classification_bag_test_size': 20, 'classification_bag_random_state': 42 }
				
				def reset_bag_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Bagging Model widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in bag_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in bag_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				bag_c1, bag_c2, bag_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with bag_c1:
					st.markdown( '###### 🎚️ Hyper-Parameters' )
					bag_estimators = st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'classification_bag_estimators' ] ), step=1,
						key='classification_bag_estimators' )
				
				with bag_c2:
					st.markdown( '###### ↔️ Split' )
					bag_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_bag_test_size' ] ), step=1,
						key='classification_bag_test_size' ) / 100.0
				
				with bag_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					bag_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_bag_random_state' ] ), step=1,
						key='classification_bag_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				bag_btn_1, bag_btn_2 = st.columns( 2 )
				with bag_btn_1:
					train_bag = st.button( '🚂 Train Bagging Model',
						key='classification_bag_train',
						use_container_width=True )
				
				with bag_btn_2:
					st.button( '🔁 Reset Bagging Model', key='classification_bag_reset',
						use_container_width=True, on_click=reset_bag_state )
				
				if train_bag:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Bagging Model requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Bagging Model requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Bagging Model requires at least two target classes.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.BaggingModel(
							estimators=int( bag_estimators ), random=int( bag_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( bag_test_size ), random=int( bag_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Bagging Model training completed.' )
					except Exception as ex:
						st.error( f'Bagging Model training failed: {ex}' )
			
			with st.expander( 'Voting Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.VOTING_CLASSFIER )
				
				vote_defaults = { 'classification_vote_mode': 'hard',
					'classification_vote_include_logistic': True,
					'classification_vote_include_tree': True,
					'classification_vote_include_knn': True,
					'classification_vote_include_forest': False,
					'classification_vote_include_nb': False, 'classification_vote_test_size': 20,
					'classification_vote_random_state': 42 }
				
				def reset_vote_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Voting Model widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in vote_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'y_predictions' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in vote_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				vote_c1, vote_c2, vote_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with vote_c1:
					st.markdown( '###### ♟️ Voting Strategy' )
					vote_mode = st.selectbox( 'Vote', options=[ 'hard', 'soft' ],
						index=[ 'hard', 'soft' ].index(
							st.session_state[ 'classification_vote_mode' ] ),
						key='classification_vote_mode' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with vote_c2:
					st.markdown( '###### 📐 Base Estimators' )
					vote_include_logistic = st.checkbox( 'Logistic Regression',
						value=bool( st.session_state[ 'classification_vote_include_logistic' ] ),
						key='classification_vote_include_logistic' )
					
					vote_include_tree = st.checkbox( 'Decision Tree',
						value=bool( st.session_state[ 'classification_vote_include_tree' ] ),
						key='classification_vote_include_tree' )
					
					vote_include_knn = st.checkbox( 'k-Nearest Neighbors',
						value=bool( st.session_state[ 'classification_vote_include_knn' ] ),
						key='classification_vote_include_knn' )
					
					vote_include_forest = st.checkbox( 'Random Forest',
						value=bool( st.session_state[ 'classification_vote_include_forest' ] ),
						key='classification_vote_include_forest' )
					
					vote_include_nb = st.checkbox( 'Gaussian Naive Bayes',
						value=bool( st.session_state[ 'classification_vote_include_nb' ] ),
						key='classification_vote_include_nb' )
				
				with vote_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					vote_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_vote_test_size' ] ), step=1,
						key='classification_vote_test_size' ) / 100.0
					
					vote_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_vote_random_state' ] ),
						step=1,
						key='classification_vote_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				vote_btn_1, vote_btn_2 = st.columns( 2 )
				with vote_btn_1:
					train_vote = st.button( '🚂 Train Voting Model',
						key='classification_vote_train',
						use_container_width=True )
				
				with vote_btn_2:
					st.button( '🔁 Reset Voting Model', key='classification_vote_reset',
						use_container_width=True, on_click=reset_vote_state )
				
				if train_vote:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Voting Model requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Voting Model requires at least one numeric feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Voting Model requires at least two target classes.' )
							st.stop( )
						
						estimators = [ ]
						
						if vote_include_logistic:
							estimators.append( ('logistic', LogisticRegression( max_iter=1000,
								random_state=int( vote_random_state ) )) )
						
						if vote_include_tree:
							estimators.append( ('tree',
								DecisionTreeClassifier( random_state=int( vote_random_state ) )) )
						
						if vote_include_knn:
							estimators.append( ('knn', KNeighborsClassifier( )) )
						
						if vote_include_forest:
							estimators.append( ('forest',
								RandomForestClassifier( random_state=int( vote_random_state ) )) )
						
						if vote_include_nb:
							estimators.append( ('naive_bayes', GaussianNB( )) )
						
						if len( estimators ) < 2:
							st.warning( '⚠️ Voting Model requires at least two base estimators.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						model = classification_model.VotingModel( estimators=estimators,
							vote=str( vote_mode ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( vote_test_size ), random=int( vote_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Estimator Count',
							int( len( estimators ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Vote Mode', str( vote_mode ) )
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Voting Model training completed.' )
					except Exception as ex:
						st.error( f'Voting Model training failed: {ex}' )
			
			with st.expander( 'Stacking Model', expanded=False ):
				st.caption( 'Description', width='stretch', text_alignment='left',
					help=cfg.STACKING_CLASSIFIER )
				
				stack_defaults = { 'classification_stack_include_logistic': True,
					'classification_stack_include_tree': True,
					'classification_stack_include_knn': True,
					'classification_stack_include_forest': False,
					'classification_stack_include_nb': False,
					'classification_stack_final': 'logistic', 'classification_stack_test_size': 20,
					'classification_stack_random_state': 42 }
				
				def reset_stack_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Stacking Model widget values and model outputs before widget
						instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in stack_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'y_predictions' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in stack_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				stack_c1, stack_c2, stack_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with stack_c1:
					st.markdown( '###### 🛑 Final Estimator' )
					stack_final = st.selectbox( 'Final Estimator', options=[ 'logistic', 'tree' ],
						index=[ 'logistic', 'tree' ].index(
							st.session_state[ 'classification_stack_final' ] ),
						key='classification_stack_final' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with stack_c2:
					st.markdown( '###### 📐 Base Estimators' )
					stack_include_logistic = st.checkbox( 'Logistic Regression',
						value=bool( st.session_state[ 'classification_stack_include_logistic' ] ),
						key='classification_stack_include_logistic' )
					
					stack_include_tree = st.checkbox( 'Decision Tree',
						value=bool( st.session_state[ 'classification_stack_include_tree' ] ),
						key='classification_stack_include_tree' )
					
					stack_include_knn = st.checkbox( 'k-Nearest Neighbors',
						value=bool( st.session_state[ 'classification_stack_include_knn' ] ),
						key='classification_stack_include_knn' )
					
					stack_include_forest = st.checkbox( 'Random Forest',
						value=bool( st.session_state[ 'classification_stack_include_forest' ] ),
						key='classification_stack_include_forest' )
					
					stack_include_nb = st.checkbox( 'Gaussian Naive Bayes',
						value=bool( st.session_state[ 'classification_stack_include_nb' ] ),
						key='classification_stack_include_nb' )
				
				with stack_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					stack_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_stack_test_size' ] ), step=1,
						key='classification_stack_test_size' ) / 100.0
					
					stack_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_stack_random_state' ] ),
						step=1, key='classification_stack_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				stack_btn_1, stack_btn_2 = st.columns( 2 )
				with stack_btn_1:
					train_stack = st.button( '🚂 Train Stacking Model',
						key='classification_stack_train', use_container_width=True )
				
				with stack_btn_2:
					st.button( '🔁 Reset Stacking Model', key='classification_stack_reset',
						use_container_width=True, on_click=reset_stack_state )
				
				if train_stack:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Stacking Model requires prepared feature and target arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Stacking Model requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ Stacking Model requires at least two target classes.' )
							st.stop( )
						
						estimators = [ ]
						if stack_include_logistic:
							estimators.append( ('logistic', LogisticRegression( max_iter=1000,
								random_state=int( stack_random_state ) )) )
						
						if stack_include_tree:
							estimators.append( ('tree',
								DecisionTreeClassifier( random_state=int( stack_random_state ) )) )
						
						if stack_include_knn:
							estimators.append( ('knn', KNeighborsClassifier( )) )
						
						if stack_include_forest:
							estimators.append( ('forest',
								RandomForestClassifier( random_state=int( stack_random_state ) )) )
						
						if stack_include_nb:
							estimators.append( ('naive_bayes', GaussianNB( )) )
						
						if len( estimators ) < 2:
							st.warning( '⚠️ Stacking Model requires at least two base estimators.' )
							st.stop( )
						
						if stack_final == 'logistic':
							final_estimator = LogisticRegression( max_iter=1000,
								random_state=int( stack_random_state ) )
						else:
							final_estimator = DecisionTreeClassifier(
								random_state=int( stack_random_state ) )
						
						start_time = time.perf_counter( )
						
						model = classification_model.StackingModel( est=estimators,
							final=final_estimator )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( stack_test_size ), random=int( stack_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Estimator Count',
							int( len( estimators ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Final Estimator',
							str( stack_final ) )
						
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Stacking Model training completed.' )
					except Exception as ex:
						st.error( f'Stacking Model training failed: {ex}' )
		
		# -----------------------------------------------------------------
		# Neural Models
		# -----------------------------------------------------------------
		with st.expander( 'Neural Models', expanded=True ):
			
			with st.expander( 'Multi-Layer Perceptron', expanded=False ):
				mlp_defaults = { 'classification_mlp_hidden_1': 100,
					'classification_mlp_hidden_2': 0, 'classification_mlp_activation': 'logistic',
					'classification_mlp_solver': 'lbfgs', 'classification_mlp_alpha': 0.000100,
					'classification_mlp_learning': 'constant', 'classification_mlp_test_size': 20,
					'classification_mlp_random_state': 42 }
				
				def reset_mlp_state( ) -> None:
					"""
					
						Purpose:
						--------
						Reset Multi-Layer Perceptron widget values and model outputs before
						widget instantiation on the next Streamlit run.
					
						Parameters:
						-----------
						None
					
						Returns:
						--------
						None
					
					"""
					for reset_key, reset_value in mlp_defaults.items( ):
						st.session_state[ reset_key ] = reset_value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'y_predictions' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				
				for key, value in mlp_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				mlp_c1, mlp_c2, mlp_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with mlp_c1:
					st.markdown( '###### 🛡️ Network Structure' )
					mlp_hidden_1 = st.number_input( 'Hidden Layer 1', min_value=1,
						value=int( st.session_state[ 'classification_mlp_hidden_1' ] ), step=1,
						key='classification_mlp_hidden_1' )
					
					mlp_hidden_2 = st.number_input( 'Hidden Layer 2 (0 = none)', min_value=0,
						value=int( st.session_state[ 'classification_mlp_hidden_2' ] ), step=1,
						key='classification_mlp_hidden_2' )
					
					mlp_activation = st.selectbox( 'Activation',
						options=[ 'identity', 'logistic', 'tanh', 'relu' ],
						index=[ 'identity', 'logistic', 'tanh', 'relu' ].index(
							st.session_state[ 'classification_mlp_activation' ] ),
						key='classification_mlp_activation' )
				
				with mlp_c2:
					st.markdown( '###### 🎚️ Optimization' )
					mlp_solver = st.selectbox( 'Solver', options=[ 'lbfgs', 'sgd', 'adam' ],
						index=[ 'lbfgs', 'sgd', 'adam' ].index(
							st.session_state[ 'classification_mlp_solver' ] ),
						key='classification_mlp_solver' )
					
					mlp_alpha = st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'classification_mlp_alpha' ] ),
						step=0.000100, format='%.6f', key='classification_mlp_alpha' )
					
					mlp_learning = st.selectbox( 'Learning Rate Schedule',
						options=[ 'constant', 'invscaling', 'adaptive' ],
						index=[ 'constant', 'invscaling', 'adaptive' ].index(
							st.session_state[ 'classification_mlp_learning' ] ),
						key='classification_mlp_learning' )
					
					mlp_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'classification_mlp_test_size' ] ), step=1,
						key='classification_mlp_test_size' ) / 100.0
				
				with mlp_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					mlp_random_state = st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'classification_mlp_random_state' ] ), step=1,
						key='classification_mlp_random_state' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Classes: {len( class_counts ):,}' )
					
					st.caption( f'Target: {target_name}' )
				
				# ------------------------------------------------------------------
				# Model Training
				# ------------------------------------------------------------------
				mlp_btn_1, mlp_btn_2 = st.columns( 2 )
				with mlp_btn_1:
					train_mlp = st.button( '🚂 Train Multi-Layer Perceptron',
						key='classification_mlp_train', use_container_width=True )
				
				with mlp_btn_2:
					st.button( '🔁 Reset Multi-Layer Perceptron', key='classification_mlp_reset',
						use_container_width=True, on_click=reset_mlp_state )
				
				if train_mlp:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Multi-Layer Perceptron requires prepared feature and target '
								'arrays.' )
							st.stop( )
						
						if np.asarray( X ).ndim != 2 or np.asarray( X ).shape[ 1 ] < 1:
							st.warning( '⚠️ Multi-Layer Perceptron requires at least one numeric '
							            'feature.' )
							st.stop( )
						
						y = np.ravel( y ) if np.asarray( y ).ndim != 1 else y
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ Multi-Layer Perceptron requires at least two target classes.' )
							st.stop( )
						
						if int( mlp_hidden_2 ) > 0:
							hidden_layers = (int( mlp_hidden_1 ), int( mlp_hidden_2 ))
						else:
							hidden_layers = (int( mlp_hidden_1 ),)
						
						start_time = time.perf_counter( )
						model = classification_model.MultiLayerPerceptron( hidden=hidden_layers,
							activation=str( mlp_activation ), solver=str( mlp_solver ),
							alpha=float( mlp_alpha ), learning=str( mlp_learning ),
							rando=int( mlp_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( mlp_test_size ), random=int( mlp_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.score( X_test, y_test )
						df_scores = df_scores.copy( ) if isinstance( df_scores,
							pd.DataFrame ) else pd.DataFrame( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						df_scores.insert( len( df_scores.columns ), 'Hidden Layers',
							str( hidden_layers ) )
						
						df_scores.insert( len( df_scores.columns ), 'Activation',
							str( mlp_activation ) )
						
						df_scores.insert( len( df_scores.columns ), 'Solver', str( mlp_solver ) )
						y_prediction = np.asarray( y_prediction )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test ), 'Predicted': y_prediction } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'y_predictions' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Multi-Layer Perceptron training completed.' )
					except Exception as ex:
						st.error( f'Multi-Layer Perceptron training failed: {ex}' )
		
		if st.session_state.get( 'model', None ) is None:
			st.stop( )
		
		# ------------------------------------------------------------------
		# Performance Metrics & Visualizations
		# ------------------------------------------------------------------
		target_count = int( st.session_state.get( 'target_count', 0 ) )
		model = st.session_state.get( 'model', None )
		X_test = st.session_state.get( 'X_test', None )
		y_test = st.session_state.get( 'y_test', None )
		y_prediction = st.session_state.get( 'y_prediction', None )
		if y_prediction is None:
			y_prediction = st.session_state.get( 'y_prediction', None )
		
		df_scores = st.session_state.get( 'df_scores', pd.DataFrame( ) )
		df_predictions = st.session_state.get( 'df_predictions', pd.DataFrame( ) )
		elapsed_seconds = float( st.session_state.get( 'elapsed_seconds', 0.0 ) )
		has_metric_frame = (isinstance( df_scores,
			pd.DataFrame ) and not df_scores.empty and 'Accuracy Score' in df_scores.columns and
		                    'Mis-Classifications' in df_scores.columns)
		
		has_prediction_frame = (isinstance( df_predictions,
			pd.DataFrame ) and not df_predictions.empty and 'Actual' in df_predictions.columns and
		                        'Predicted' in df_predictions.columns)
		
		has_visual_context = (
				model is not None and X_test is not None and y_test is not None and y_prediction
				is not None)
		
		blue_divider( )
		st.markdown( '##### Model Performance' )
		
		if has_metric_frame:
			m1, m2, m3 = st.columns( 3, border=True )
			with m1:
				st.metric( 'Accuracy', f"{float( df_scores.at[ 0, 'Accuracy Score' ] ):0.2f}" )
			
			with m2:
				st.metric( 'Mis-Classifications',
					f"{int( df_scores.at[ 0, 'Mis-Classifications' ] ):,}" )
			
			with m3:
				st.metric( 'Processing Time', f'{elapsed_seconds:0.2f} sec' )
			
			render_data_editor( df_scores, use_container_width=True,
				key='classification_performance_scores' )
		else:
			st.info( 'No classification performance metrics are available yet. '
			         'Train a classification model to populate this section.' )
		
		blue_divider( )
		st.markdown( '##### Predictions' )
		
		if has_prediction_frame:
			render_data_editor( df_predictions, use_container_width=True,
				key='classification_performance_predictions' )
		else:
			st.info( 'No predictions are available for the current classification result.' )
		
		blue_divider( )
		st.markdown( '##### Confusion Matrix', help=cfg.CONFUSION_MATRIX )
		
		if has_visual_context:
			try:
				class_labels = sorted( set( pd.Series( y_test ).astype( str ) ).union(
					set( pd.Series( y_prediction ).astype( str ) ) ) )
				matrix = confusion_matrix( pd.Series( y_test ).astype( str ),
					pd.Series( y_prediction ).astype( str ), labels=class_labels )
				figure = go.Figure( data=go.Heatmap( z=matrix, x=class_labels,
					y=class_labels, colorscale='Blues', text=matrix, texttemplate='%{text}',
					hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>' ) )
				figure.update_layout( xaxis_title='Predicted Class', yaxis_title='Actual Class' )
				render_mathy_plotly_chart( figure, 'classification_confusion_matrix_chart',
					'mathy_classification_confusion_matrix', 'Confusion Matrix', 500 )
			except Exception as e:
				st.info( f'Confusion Matrix skipped: {e}' )
		else:
			st.info( 'Confusion Matrix is unavailable until a classification model is trained.' )
		
		# ------------------------------------------------------------------
		# Actual vs Predicted Class Counts
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Actual vs Predicted Counts' )
		
		if has_visual_context:
			try:
				actual_counts = pd.Series( y_test ).value_counts( ).sort_index( )
				pred_counts = pd.Series( y_prediction ).value_counts( ).sort_index( )
				df_counts = pd.DataFrame(
					{ 'Actual': actual_counts, 'Predicted': pred_counts } ).fillna( 0 )
				
				df_counts = df_counts.rename_axis( 'Class' ).reset_index( ).melt(
					id_vars='Class', var_name='Series', value_name='Count' )
				figure = px.bar( df_counts, x='Class', y='Count', color='Series', barmode='group' )
				render_mathy_plotly_chart( figure, 'classification_actual_predicted_counts_chart',
					'mathy_classification_class_counts', 'Actual vs Predicted Class Counts', 500 )
			except Exception as e:
				st.info( f'Actual vs Predicted Counts skipped: {e}' )
		else:
			st.info( 'Actual vs Predicted Counts are unavailable until a model is trained.' )
		
		# ------------------------------------------------------------------
		# Per-Class Accuracy
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Per-Class Accuracy', help=cfg.PERCLASS_ACCURACY )
		
		if has_visual_context:
			try:
				df_evaluation = pd.DataFrame( { 'Actual': y_test, 'Predicted': y_prediction } )
				df_evaluation[ 'Correct' ] = (
						df_evaluation[ 'Actual' ] == df_evaluation[ 'Predicted' ]).astype( int )
				
				df_class_acc = df_evaluation.groupby( 'Actual', dropna=False )[
					'Correct' ].mean( ).sort_index( )
				
				df_class_acc = df_class_acc.rename( 'Accuracy' ).rename_axis( 'Class' ).reset_index( )
				df_class_acc[ 'Class' ] = df_class_acc[ 'Class' ].astype( str )
				figure = px.bar( df_class_acc, x='Class', y='Accuracy', range_y=[ 0.0, 1.05 ] )
				render_mathy_plotly_chart( figure, 'classification_per_class_accuracy_chart',
					'mathy_classification_per_class_accuracy', 'Per-Class Accuracy', 500 )
			except Exception as e:
				st.info( f'Per-Class Accuracy skipped: {e}' )
		else:
			st.info( 'Per-Class Accuracy is unavailable until a model is trained.' )
		
		# ------------------------------------------------------------------
		# Prediction Confidence
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Prediction Confidence', help=cfg.PREDICTION_CONFIDENCE )
		
		if has_visual_context and hasattr( model, 'predict_probability' ):
			try:
				proba = model.predict_probability( X_test )
				if (isinstance( proba, np.ndarray ) and proba.ndim == 2 and proba.shape[ 1 ] > 1):
					max_conf = np.max( proba, axis=1 )
					figure = px.histogram( x=max_conf, nbins=20,
						labels={ 'x': 'Maximum Predicted Probability', 'y': 'Frequency' } )
					render_mathy_plotly_chart( figure, 'classification_confidence_chart',
						'mathy_classification_confidence', 'Prediction Confidence Distribution', 500 )
				else:
					st.info( 'Prediction confidence is unavailable for this model output.' )
			except Exception as e:
				st.info( f'Prediction Confidence skipped: {e}' )
		else:
			st.info( 'Prediction Confidence is unavailable until a model is trained.' )
		
		# ------------------------------------------------------------------
		# Observed vs Predicted
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Observed vs Predicted' )
		
		if has_visual_context and target_count == 2:
			try:
				df_observed = pd.DataFrame( { 'Actual': pd.Series( y_test ).astype( str ),
					'Predicted': pd.Series( y_prediction ).astype( str ) } )
				figure = px.scatter( df_observed, x='Actual', y='Predicted', color='Actual',
					hover_data=df_observed.columns )
				render_mathy_plotly_chart( figure, 'classification_observed_predicted_chart',
					'mathy_classification_observed_predicted', 'Observed vs Predicted', 500 )
			except Exception as e:
				st.info( f'Observed vs Predicted plot skipped: {e}' )
		else:
			st.info( 'Observed vs Predicted is shown only for binary classification targets.' )
		
		# ------------------------------------------------------------------
		# ROC Curve
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### ROC Curve', help=cfg.ROC_CURVE )
		
		if has_visual_context and target_count == 2 and hasattr( model, 'predict_probability' ):
			try:
				proba = np.asarray( model.predict_probability( X_test ) )
				classes = np.unique( np.asarray( y_test ) )
				positive_class = classes[ -1 ]
				y_binary = (np.asarray( y_test ) == positive_class).astype( int )
				scores = proba[ :, -1 ] if proba.ndim == 2 else proba.reshape( -1 )
				false_positive_rate, true_positive_rate, _ = roc_curve( y_binary, scores )
				roc_auc = auc( false_positive_rate, true_positive_rate )
				figure = go.Figure( )
				figure.add_trace( go.Scatter( x=false_positive_rate, y=true_positive_rate,
					mode='lines', name=f'ROC (AUC = {roc_auc:0.3f})' ) )
				figure.add_trace( go.Scatter( x=[ 0, 1 ], y=[ 0, 1 ], mode='lines',
					name='Chance', line={ 'dash': 'dash', 'color': '#94A3B8' } ) )
				figure.update_layout( xaxis_title='False Positive Rate',
					yaxis_title='True Positive Rate' )
				render_mathy_plotly_chart( figure, 'classification_roc_curve_chart',
					'mathy_classification_roc_curve', 'ROC Curve', 500 )
			except Exception as e:
				st.info( f'ROC Curve Skipped: {e}' )
		else:
			st.info( 'ROC curve is available only for binary classification targets.' )

# ============================================
# REGRESSION MODE
# ============================================
elif mode == 'Regression Models':
	df_dataset = get_analysis_dataset( )
	df_original = st.session_state.get( 'df_original', None )
	df_working = st.session_state.get( 'df_working', pd.DataFrame( ) )
	df_processed = st.session_state.get( 'df_processed', pd.DataFrame( ) )
	df_regression = st.session_state.get( 'df_regression', pd.DataFrame( ) )
	df_model = st.session_state.get( 'df_model', pd.DataFrame( ) )
	df_scores = st.session_state.get( 'df_scores', pd.DataFrame( ) )
	df_predictions = st.session_state.get( 'df_predictions', pd.DataFrame( ) )
	numeric_columns = st.session_state.get( 'numeric_columns', [ ] )
	categorical_columns = st.session_state.get( 'categorical_columns', [ ] )
	features = st.session_state.get( 'features', [ ] )
	targets = st.session_state.get( 'targets', [ ] )
	active_features = st.session_state.get( 'active_features', [ ] )
	active_targets = st.session_state.get( 'active_targets', [ ] )
	X_data = st.session_state.get( 'X_data', None )
	X_train = st.session_state.get( 'X_train', None )
	X_test = st.session_state.get( 'X_test', None )
	y_train = st.session_state.get( 'y_train', None )
	y_test = st.session_state.get( 'y_test', None )
	y_series = st.session_state.get( 'y_series', None )
	elapsed_seconds = st.session_state.get( 'elapsed_seconds', 0.0 )
	
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Regression Models' ] )
		st.caption( 'Predictive Models for Continuous Values' )
		st.divider( )
		
		if not has_loaded_dataset( df_dataset ):
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		st.session_state[ 'df_original' ] = df_original.copy( )
		
		numeric_columns = st.session_state.get( 'numeric_columns', [ ] ).copy( )
		categorical_columns = st.session_state.get( 'categorical_columns', [ ] ).copy( )
		
		st.session_state[ 'numeric_columns' ] = numeric_columns.copy( )
		st.session_state[ 'categorical_columns' ] = categorical_columns.copy( )
		if len( df_original.columns ) < 2:
			st.warning(
				'⚠️ Regression requires at least one feature column and one target column.' )
			st.stop( )
		
		if not numeric_columns:
			st.warning( '⚠️ Regression requires at least one numeric target column.' )
			st.stop( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Samples: {len( df_original ):,} | '
		            f'Features: {len( df_original.columns ):,}' )
		
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_original.columns.tolist( ),
				key='regression_features' )
		
		with col_c2:
			target_options = [ column for column in numeric_columns if column not in features ]
			if target_options:
				target = st.selectbox( 'Select Target', options=target_options,
					key='regression_target' )
			else:
				target = None
				st.info(
					'At least one numeric column must remain available as the regression target.' )
		
		sel_b1, sel_b2 = st.columns( [ 0.5, 0.5 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', icon='➕', key='regression_create_dataset',
					use_container_width=True ):
				if not features:
					st.warning( '⚠️ Select at least one feature column.' )
					st.stop( )
				
				if target is None:
					st.warning( '⚠️ Select one numeric target column.' )
					st.stop( )
				
				if target in features:
					st.warning( '⚠️ The regression target cannot also be selected as a feature.' )
					st.stop( )
				
				selected_all = features.copy( )
				selected_all.append( target )
				df_working = df_original[ selected_all ].copy( )
				if df_working.empty:
					st.warning( '⚠️ The selected feature and target columns contain no observations.' )
					st.stop( )
				
				st.session_state[ 'features' ] = features.copy( )
				st.session_state[ 'targets' ] = [ target ]
				st.session_state[ 'selected_all' ] = selected_all.copy( )
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = pd.DataFrame( )
				st.session_state[ 'df_model' ] = pd.DataFrame( )
				st.session_state[ 'df_scores' ] = pd.DataFrame( )
				st.session_state[ 'df_predictions' ] = pd.DataFrame( )
				st.session_state[ 'df_regression' ] = pd.DataFrame( )
				st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
				st.session_state[ 'active_features' ] = [ ]
				st.session_state[ 'active_targets' ] = [ ]
				st.session_state[ 'X_data' ]=None
				st.session_state[ 'X_train' ]=None
				st.session_state[ 'X_test' ]=None
				st.session_state[ 'y_train' ]=None
				st.session_state[ 'y_test' ]=None
				st.session_state[ 'y_series' ]=None
				st.session_state[ 'y_prediction' ]=None
				st.session_state[ 'model' ]=None
				st.session_state[ 'elapsed_seconds' ] = 0.0
				st.session_state[ 'target_count' ] = 1.0
				commit_frame( df_working )
				st.success( 'Working Dataset Created!' )
				st.rerun( )
		
		with sel_b2:
			if st.button( 'Reset Working Dataset', icon='🔁', key='regression_reset_to_original',
					use_container_width=True ):
				reset_regression_mode_state( )
				st.success( 'Regression working data reset.' )
				st.rerun( )
		
		blue_divider( )
		df_working = st.session_state.get( 'df_working', pd.DataFrame( ) )
		if not has_loaded_dataset( df_working ):
			i = ('Select regression features and one numeric target, then create the working dataset.')
			st.info( i )
			st.stop( )
		
		features = [ column for column in st.session_state.get( 'features', [ ] ) if
			column in df_working.columns ]
		
		targets = [ column for column in st.session_state.get( 'targets', [ ] ) if
			column in df_working.columns ]
		
		if not features:
			st.warning( '⚠️ The working dataset does not contain any selected feature columns.' )
			st.stop( )
		
		if len( targets ) != 1:
			st.warning( '⚠️ Regression requires exactly one numeric target column.' )
			st.stop( )
		
		target_name = targets[ 0 ]
		if not pd.api.types.is_numeric_dtype( df_working[ target_name ] ):
			st.warning( '⚠️ The selected regression target must be numeric.' )
			st.stop( )
		
		st.markdown( '##### Working Data' )
		st.caption( f'Samples: {len( df_working ):,} | '
		            f'Features: {len( features ):,} | Target: {target_name}' )
		
		render_data_editor( df_working, key='regression_working_data', use_container_width=True )
		
		# ----- Feature-Engineering
		blue_divider( )
		st.markdown( '##### Feature-Engineering' )
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			
			# -----------------------------------------------------------------
			# Data Scaling
			# -----------------------------------------------------------------
			with st.expander( label='Data Scaling', icon='⚖️', key='regression_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.STANDARD_SCALER )
					
					columns = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_standard_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', use_container_width=True,
								key='regression_standard_scaler_apply' ):
							if columns:
								scaler = StandardScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ columns ].to_numpy( ) )
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Standard Scaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_standard_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MINMAX_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_minmax_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_minmax_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MinMaxScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Min-Max Scaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔄',
								key='regression_minmax_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ]=None
							df_processed = None
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.ROBUST_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_robust_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_robust_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = RobustScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Robust Scaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_robust_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ]=None
							df_processed = None
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.NORMAL_SCALER )
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ], index=1,
						key='regression_normal_scaler_norm' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_normal_scaler_apply', use_container_width=True ):
							if scale_cols:
								scaler = NormalScaler( norm=norm )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_normal_scaler_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Scaler Description', width='stretch', text_alignment='left',
						help=cfg.MAXABS_SCALER )
					
					scale_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_maxabs_scaler_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_maxabs_scaler_apply', use_container_width=True ):
							if scale_cols:
								scaler = MaxAbsScaler( )
								df_processed = df_working.copy( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ columns ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_maxabs_scaler_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )

			# -----------------------------------------------------------------
			# Data Imputation
			# -----------------------------------------------------------------
			with st.expander( label='Data Imputation', icon='🧹', key='regression_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.MEAN_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='regression_mean_imputer_indicator' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_mean_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = MeanImputer( strategy='mean',
									add_indicator=add_indicator )
								
								df_processed = df_working.copy( )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols, result,
									'mean_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'MeanImputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_mean_imputer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1, value=5, step=1,
						key='regression_nearest_imputer_neighbors' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_nearest_imputer_apply', use_container_width=True ):
							if impute_cols:
								imputer = NearestImputer( neighbors=int( neighbors ) )
								df_processed = df_working.copy( )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols, result,
									'nearest_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Nearest Imputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_nearest_imputer_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.ITERATIVE_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=10, step=1,
						key='regression_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=0, step=1,
						key='regression_iterative_imputer_random_state' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer', icon='✔️',
								key='regression_iterative_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, impute_cols, result,
									'iterative_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Iterative Imputer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_iterative_imputer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Imputer Description', width='stretch', text_alignment='left',
						help=cfg.SIMPLE_IMPUTER )
					
					impute_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_simple_imputer_cols' )
					
					strategy = st.selectbox( 'Strategy',
						options=[ 'mean', 'median', 'most_frequent', 'constant' ],
						key='regression_simple_imputer_strategy' )
					
					fill_value = st.text_input( 'Fill Value', value='0.0',
						key='regression_simple_imputer_fill_value' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='regression_simple_imputer_indicator' )
					
					keep_empty_features = st.checkbox( 'Keep Empty Features', value=False,
						key='regression_simple_imputer_keep_empty' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', icon='✔️',
								key='regression_simpleimputer_apply', use_container_width=True ):
							if impute_cols:
								if strategy in [ 'mean', 'median' ]:
									df_input = df_processed[ impute_cols ].apply( pd.to_numeric,
										errors='coerce' )
									fill_object: object = 0.0
								elif strategy == 'constant':
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								else:
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								
								imputer = SimpleImputer( strategy=strategy, fill_value=fill_object,
									add_indicator=add_indicator,
									keep_empty_features=keep_empty_features )
								
								result = imputer.train_transform( df_input.to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols, result,
									'simple_imputer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Simple Imputer Applied' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_simple_imputer_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )

			# -----------------------------------------------------------------
			# Data Encoding
			# -----------------------------------------------------------------
			with st.expander( label='Data Encoding', icon='🔣', key='regression_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ONEHOT_ENCODER )
					
					encode_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='regression_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0, key='regression_onehot_unknown' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_onehot_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'onehot' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'One-Hot Encoder applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_onehot_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.ORDINAL_ENCODER )
					
					encode_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_ordinal_cols' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_ordinal_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OrdinalEncoder( )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								df_processed[ encode_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Ordinal Encoder Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_ordinal_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_ENCODER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='regression_label_encoder_col' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_label_encoder_apply', use_container_width=True ):
							if target_col:
								encoder = LabelEncoder( )
								result = encoder.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed[ target_col ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Label Encoder Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_label_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
					
					st.session_state[ 'df_processed' ] = df_processed
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.TARGET_ENCODER )
					
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=df_working.columns, key='regression_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_target_encoder_target_col' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_target_encoder_apply', use_container_width=True ):
							if encode_cols and target_col:
								df_processed = df_working.copy( )
								encoder = TargetEncoder( )
								X_enc = df_processed[ encode_cols ].astype( str ).to_numpy( )
								y_enc = df_processed[ target_col ].to_numpy( )
								result = encoder.train_transform( X_enc, y_enc )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'target_encoder' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Target Encoder Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_target_encoder_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Encoder Description', width='stretch', text_alignment='left',
						help=cfg.POLYNOMIAL_FEATURES )
					
					poly_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4, value=2,
						key='regression_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='regression_polynomial_interaction' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_polynomial_apply',
								use_container_width=True ):
							if poly_cols:
								df_processed = df_working.copy( )
								encoder = PolynomialFeatures( degree=int( degree ),
									interaction=bool( interaction ) )
								
								result = encoder.train_transform(
									df_processed[ poly_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, poly_cols, result,
									'polynomial' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'PolynomialFeatures applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_polynomial_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
		
		with feature_c2:

			# -----------------------------------------------------------------
			# Data Transformation
			# -----------------------------------------------------------------
			with st.expander( label='Data Transformation', icon='⚡',
					key='regression_transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.BINARIZER )
					
					transform_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='regression_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='regression_binarizer_copy' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer', key='regression_binarizer_apply',
								use_container_width=True ):
							if transform_cols:
								df_processed = df_working.copy( )
								transformer = Binarizer( threshold=float( threshold ),
									copy=bool( copy ) )
								
								result = transformer.train_transform(
									df_processed[ transform_cols ].to_numpy( ) )
								
								df_processed[ transform_cols ] = result
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Binarizer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_binarizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_BINARIZER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='regression_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='regression_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='regression_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='regression_label_binarizer_sparse' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer', key='regression_lblbinarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ),
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'label_binarizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_lblbinarizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.MULTILABEL_BINARIZER )
					
					target_col = st.selectbox( 'Column', options=df_working.columns,
						key='regression_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='regression_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='regression_multilabel_binarizer_sparse' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_multilabel_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								y_multi = parse_multilabel_series( df_processed[ target_col ],
									delimiter=delimiter )
								
								transformer = MultiLabelBinarizer( classes=None,
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform( y_multi )
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'multilabel_binarizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Multi-Label Binarizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_multilabel_binarizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_TRANSFORMER )
					
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=df_working.columns, key='regression_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ], index=1,
						key='regression_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='regression_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='regression_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='regression_tfidf_transformer_sublinear' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_tfidf_transformer_apply',
								use_container_width=True ):
							if text_count_cols:
								df_processed = df_working.copy( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ),
									sublinear_tf=bool( sublinear_tf ) )
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply( pd.to_numeric,
										errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols,
									result, 'tfidf_transformer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_tfidf_transformer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Column Transformer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COLUMN_TRANSFORMER )
					
					numeric_columns = st.multiselect( 'Numeric Columns',
						options=df_working.columns,
						key='regression_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=df_working.columns,
						key='regression_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
							'Binarizer', 'None' ],
						key='regression_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='regression_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='regression_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0, max_value=1.0,
						value=0.3, key='regression_column_transformer_sparse_threshold' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer',
								key='regression_column_transformer_apply',
								use_container_width=True ):
							df_processed = df_working.copy( )
							transformers = [ ]
							
							if numeric_columns and numeric_transform != 'None':
								if numeric_transform == 'StandardScaler':
									numeric_model = StandardScaler( ).model
								elif numeric_transform == 'MinMaxScaler':
									numeric_model = MinMaxScaler( ).model
								elif numeric_transform == 'RobustScaler':
									numeric_model = RobustScaler( ).model
								elif numeric_transform == 'MaxAbsScaler':
									numeric_model = MaxAbsScaler( ).model
								else:
									numeric_model = Binarizer( ).model
								
								transformers.append( ('numeric', numeric_model, numeric_columns) )
							
							if categorical_columns and categorical_transform != 'None':
								if categorical_transform == 'OneHotEncoder':
									categorical_model = OneHotEncoder( sparse=False,
										unknown='ignore' ).model
								else:
									categorical_model = OrdinalEncoder( ).model
								
								transformers.append(
									('categorical', categorical_model, categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float(
										sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								
								result = transformer.train_transform( df_processed )
								df_processed = normalize_result_frame( result=result,
									index=df_processed.index, prefix='column_transformer',
									columns=None )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_column_transformer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )

			# -----------------------------------------------------------------
			# Feature Extraction
			# -----------------------------------------------------------------
			with st.expander( label='Feature Extration', icon='⛏️', key='regression_extractors' ):
				
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.TDIDF_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='regression_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='regression_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='regression_tfidf_vectorizer_use_idf' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_tfidf_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'tfidf_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_tfidf_vectorizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.COUNT_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='regression_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='regression_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='regression_count_vectorizer_binary' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_count_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), binary=bool( binary ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'count_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_count_vectorizer_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.HASH_VECTORIZER )
					
					text_cols = st.multiselect( 'Text Columns', options=df_working.columns,
						key='regression_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='regression_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='regression_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='regression_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='regression_hash_vectorizer_alternate_sign' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_hash_vectorizer_apply', use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'hash_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_hash_vectorizer_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.DICT_VECTORIZER )
					
					dict_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='regression_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='regression_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='regression_dict_vectorizer_sort' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_dict_vectorizer_apply', use_container_width=True ):
							if dict_cols:
								df_processed = df_working.copy( )
								transformer = DictVectorizer( dtype=np.float64,
									separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_dict_vectorizer_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					st.caption( 'Transformer Description', width='stretch', text_alignment='left',
						help=cfg.FEATURE_HASHER )
					
					hash_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='regression_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='regression_feature_hasher_alternate_sign' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_feature_hasher_apply', use_container_width=True ):
							if hash_cols:
								df_processed = df_working.copy( )
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								
								df_processed = apply_dict_transform( df_processed, hash_cols,
									transformer, 'feature_hasher' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'FeatureHasher applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_feature_hasher_reset', use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )

			# -----------------------------------------------------------------
			# Dimensionality Reduction
			# -----------------------------------------------------------------
			with st.expander( label='Dimensionality Reduction', icon='🎚️',
					key='regression_selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.VARIANCE_THRESHOLD )
					
					select_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0, step=0.01,
						key='regression_variance_threshold_value' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='regression_variance_threshold_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'variance_threshold' )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'VarianceThreshold applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_variance_threshold_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Canonical Correlation Analysis (CCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.CCA )
					
					X_cols = st.multiselect( 'Predictor Columns', options=df_working.columns,
						key='regression_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=df_working.columns,
						key='regression_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='regression_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True, key='regression_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500, step=1,
						key='regression_cca_max_iter' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_cca_apply',
								use_container_width=True ):
							if X_cols and y_cols:
								df_processed = df_working.copy( )
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								
								result = selector.train_transform(
									df_processed[ X_cols ].to_numpy( ),
									df_processed[ y_cols ].to_numpy( ) )
								
								df_result = normalize_result_frame( result=result,
									index=df_processed.index, prefix='cca', columns=None )
								
								df_processed = pd.concat(
									[ df_processed.drop( columns=X_cols + y_cols,
										errors='ignore' ),
										df_result ], axis=1 )
								
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_cca_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Principle Component Analysis (PCA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.PCA )
					
					select_cols = st.multiselect( 'Columns', options=df_working.columns,
						key='regression_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='regression_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='regression_pca_solver' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_pca_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = PCA( num=int( n_components ), solver=solver )
								
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'pca' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_pca_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Select-Best', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_BEST )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='regression_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ], key='regression_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='regression_selectbest_k' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_selectbest_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectBest(
									score_func=score_function_from_name( score_name ),
									num=int( k_best ) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_best' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'Select Best Applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_selectbest_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Select-Percent', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SELECT_PERCENT )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='regression_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ], key='regression_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='regression_selectpercent_percentile' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent', key='regression_selectpercent_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectPercent(
									score_func=score_function_from_name( score_name ),
									pct=int( percentile ) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_percent' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='regression_selectpercent_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Sequential Back Selection (SBS)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.SBS )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='regression_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='regression_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='regression_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1, step=1,
						key='regression_sbs_random_state' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_sbs_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state
									) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'sbs' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'SBS applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_sbs_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
				
				with st.expander( 'Recursive Feature Elimination (RFA)', expanded=False ):
					st.caption( 'Reducer Description', width='stretch', text_alignment='left',
						help=cfg.RFE )
					
					X_cols = st.multiselect( 'Feature Columns', options=df_working.columns,
						key='regression_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=df_working.columns,
						key='regression_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='regression_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0, step=1,
						key='regression_rfe_verbose' )
					
					# Apply Button
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='regression_rfe_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = RFE( k_features=int( k_features ),
									verbose=int( verbose ) )
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'rfe' )
								st.session_state[ 'df_processed' ] = df_processed.copy( )
								commit_frame( df_processed )
								st.success( 'RFE applied.' )
					
					# Reset Button
					with a2:
						if st.button( label='Reset', icon='🔁', key='regression_rfe_reset',
								use_container_width=True ):
							df_processed = None
							st.session_state[ 'df_processed' ]=None
							st.success( 'Reset Processed Data.' )
							st.rerun( )
		
		blue_divider( )
		if df_processed is None:
			st.stop( )
		
		st.markdown( '##### Processed Data' )
		st.caption(
			f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		render_data_editor( df_processed, key='regression_processed_data' )
		
		# ------------------------------------------------------------------
		# MODEL TRAINING
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Model Training' )
		active_features = list( st.session_state.get( 'df_features', pd.DataFrame( ) ).columns )
		active_targets = list( st.session_state.get( 'df_targets', pd.DataFrame( ) ).columns )
		st.session_state[ 'active_features' ] = active_features.copy( )
		st.session_state[ 'active_targets' ] = active_targets.copy( )
		
		if not active_features:
			st.warning( '⚠️ No processed feature columns are available for training.' )
			st.stop( )
		
		if not active_targets:
			st.warning( '⚠️ Regression training requires exactly one processed target column.' )
			st.stop( )
		
		target_name = active_targets[ 0 ]
		missing_columns = [ c for c in active_features + [ target_name ] if
			c not in df_processed.columns ]
		if missing_columns:
			st.warning(
				f'⚠️ The processed dataframe is missing required columns: {missing_columns}. '
				f'Apply preprocessing again or reset the working dataset.' )
			st.stop( )
		
		active_features = list( st.session_state.get( 'df_features', pd.DataFrame( ) ).columns )
		active_targets = list( st.session_state.get( 'df_targets', pd.DataFrame( ) ).columns )
		st.session_state[ 'active_features' ] = active_features.copy( )
		st.session_state[ 'active_targets' ] = active_targets.copy( )
		if not active_features:
			st.warning( '⚠️ Regression training requires at least one processed feature column.' )
			st.stop( )
		
		if len( active_targets ) != 1:
			st.warning( '⚠️ Regression training requires exactly one processed target column.' )
			st.stop( )
		
		target_name = active_targets[ 0 ]
		if target_name in active_features:
			st.warning( '⚠️ The regression target cannot also be used as a feature.' )
			st.stop( )
		
		required_columns = active_features + [ target_name ]
		missing_columns = [ column for column in required_columns if
			column not in df_processed.columns ]
		
		if missing_columns:
			st.warning( f'⚠️ The processed dataframe is missing required columns: '
			            f'{missing_columns}. Apply preprocessing again or reset the '
			            f'working dataset.' )
			st.stop( )
		
		df_model = df_processed[ required_columns ].copy( )
		for column in required_columns:
			df_model[ column ] = pd.to_numeric( df_model[ column ], errors='coerce' )
		
		df_model = df_model.replace( [ np.inf, -np.inf ], np.nan )
		df_model = df_model.dropna( subset=required_columns ).copy( )
		
		if df_model.empty:
			st.warning( '⚠️ No complete numeric observations remain after model-input '
			            'validation.' )
			st.stop( )
		
		if len( df_model ) < 2:
			st.warning( '⚠️ Regression training requires at least two complete observations.' )
			st.stop( )
		
		X_data = df_model[ active_features ].copy( )
		y_series = df_model[ target_name ].copy( )
		if X_data.empty or len( X_data.columns ) == 0:
			st.warning( '⚠️ Regression training requires at least one numeric feature.' )
			st.stop( )
		
		if X_data.isna( ).any( ).any( ):
			st.warning( '⚠️ One or more feature columns contain invalid numeric values.' )
			st.stop( )
		
		if y_series.isna( ).any( ):
			st.warning( '⚠️ The regression target contains invalid numeric values.' )
			st.stop( )
		
		if int( y_series.nunique( dropna=True ) ) < 2:
			st.warning( '⚠️ The regression target must contain at least two distinct values.' )
			st.stop( )
		
		X = X_data.to_numpy( dtype=float )
		y = y_series.to_numpy( dtype=float ).reshape( -1 )
		if X.ndim != 2 or X.shape[ 1 ] < 1:
			st.warning( '⚠️ Regression training requires a two-dimensional feature matrix.' )
			st.stop( )
		
		if y.ndim != 1:
			st.warning( '⚠️ The regression target must resolve to a one-dimensional vector.' )
			st.stop( )
		
		if len( X ) != len( y ):
			st.warning( '⚠️ Feature and target row counts do not match.' )
			st.stop( )
		
		if not np.isfinite( X ).all( ):
			st.warning( '⚠️ The regression feature matrix contains non-finite values.' )
			st.stop( )
		
		if not np.isfinite( y ).all( ):
			st.warning( '⚠️ The regression target contains non-finite values.' )
			st.stop( )
		
		df_regression = df_model.copy( )
		st.session_state[ 'df_model' ] = df_model.copy( )
		st.session_state[ 'df_regression' ] = df_regression.copy( )
		st.session_state[ 'X_data' ] = X_data.copy( )
		st.session_state[ 'y_series' ] = y_series.copy( )
		st.session_state[ 'X_train' ]=None
		st.session_state[ 'X_test' ]=None
		st.session_state[ 'y_train' ]=None
		st.session_state[ 'y_test' ]=None
		st.session_state[ 'y_prediction' ]=None
		
		# ------------------------------------------------------------------
		# REGRESSION MODELS
		# ------------------------------------------------------------------
		with st.expander( 'Linear Models', expanded=False ):
			
			with st.expander( 'Ordinary Least Squares', expanded=False ):
				ols_defaults = { 'regression_ols_test_size': 0.20,
					'regression_ols_random_state': 42, 'regression_ols_fit_intercept': True,
					'regression_ols_copy_x': True, 'regression_ols_tol': 1e-6,
					'regression_ols_n_jobs': 1, 'regression_ols_positive': False }
				
				for key, value in ols_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Linear regression for continuous targets.' )
				ols_c1, ols_c2, ols_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ols_c1:
					st.markdown( '###### ↔️ Data Split' )
					ols_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_ols_test_size' ] * 100 ), step=5,
						key='regression_ols_test_size_slider' ) / 100.0
					
					ols_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_ols_random_state' ] ), step=1,
						key='regression_ols_random_state_input' ) )
				
				with ols_c2:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					ols_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_ols_fit_intercept' ] ),
						key='regression_ols_fit_intercept_check' )
					
					ols_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_ols_copy_x' ] ),
						key='regression_ols_copy_x_check' )
					
					ols_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_ols_positive' ] ),
						key='regression_ols_positive_check' )
				
				with ols_c3:
					st.markdown( '###### ♟️ Solver Settings' )
					ols_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_ols_tol' ] ), step=0.000001,
						format='%.6f', key='regression_ols_tol_input' ) )
					
					ols_n_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_ols_n_jobs' ] ), step=1,
						key='regression_ols_n_jobs_input' ) )
				
				ols_btn_1, ols_btn_2 = st.columns( 2 )
				with ols_btn_1:
					train_ols = st.button( '🚂 Train Ordinary Least Squares',
						key='regression_ols_train', use_container_width=True )
				
				with ols_btn_2:
					reset_ols = st.button( '🔄 Reset Ordinary Least Squares',
						key='regression_ols_reset', use_container_width=True )
				
				if reset_ols:
					for key, value in ols_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_ols_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_ols:
					try:
						if X is None or y is None:
							st.warning( '⚠️ OLS requires feature and target arrays.' )
							st.stop( )
						
						X_ols = np.asarray( X, dtype=float )
						y_ols = np.asarray( y, dtype=float ).reshape( -1 )
						if X_ols.ndim != 2 or X_ols.shape[ 1 ] < 1:
							st.warning( '⚠️ OLSrequires at least one numeric feature.' )
							st.stop( )
						
						if y_ols.ndim != 1:
							st.warning( '⚠️ The Ordinary Least Squares target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_ols ) != len( y_ols ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_ols ) < 2:
							st.warning( '⚠️ Ordinary Least Squares requires at least '
							            'two observations.' )
							st.stop( )
						
						if not np.isfinite( X_ols ).all( ):
							st.warning( '⚠️ The Ordinary Least Squares feature matrix '
							            'contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_ols ).all( ):
							st.warning( '⚠️ The Ordinary Least Squares target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_ols ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						st.session_state[ 'regression_ols_test_size' ] = float( ols_test_size )
						st.session_state[ 'regression_ols_random_state' ] = int( ols_random_state )
						st.session_state[ 'regression_ols_fit_intercept' ] = bool(
							ols_fit_intercept )
						st.session_state[ 'regression_ols_copy_x' ] = bool( ols_copy_x )
						st.session_state[ 'regression_ols_tol' ] = float( ols_tol )
						st.session_state[ 'regression_ols_n_jobs' ] = int( ols_n_jobs )
						st.session_state[ 'regression_ols_positive' ] = bool( ols_positive )
						start_time = time.perf_counter( )
						
						model = regression_model.LeastSquares( fit=bool( ols_fit_intercept ),
							copy=bool( ols_copy_x ), tol=float( ols_tol ), jobs=int( ols_n_jobs ),
							positive=bool( ols_positive ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_ols, y_ols,
							size=float( ols_test_size ), random=int( ols_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( )
						else:
							df_scores = df_scores.copy( )
						
						df_scores.insert( len( df_scores.columns ), 'Processing Time (Seconds)',
							round( elapsed_seconds, 4 ) )
						
						df_scores.insert( len( df_scores.columns ), 'Training Rows',
							int( len( X_train ) ) )
						df_scores.insert( len( df_scores.columns ), 'Testing Rows',
							int( len( X_test ) ) )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_ols_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.success( 'Ordinary Least Squares training completed.' )
					except Exception as ex:
						st.error( f'Ordinary Least Squares training failed: {ex}' )
			
			with st.expander( 'Ridge Regression', expanded=False ):
				ridge_defaults = { 'regression_ridge_alpha': 1.0,
					'regression_ridge_fit_intercept': True, 'regression_ridge_copy_x': True,
					'regression_ridge_max_iter': 0, 'regression_ridge_tol': 0.0001,
					'regression_ridge_solver': 'auto', 'regression_ridge_positive': False,
					'regression_ridge_test_size': 0.20, 'regression_ridge_random_state': 42 }
				
				for key, value in ridge_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'L2-regularized linear regression for continuous targets.' )
				ridge_c1, ridge_c2, ridge_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with ridge_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					ridge_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_ridge_alpha' ] ), step=0.100000,
						format='%.6f', key='regression_ridge_alpha_input' ) )
					
					ridge_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_ridge_fit_intercept' ] ),
						key='regression_ridge_fit_intercept_check' )
					
					ridge_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_ridge_copy_x' ] ),
						key='regression_ridge_copy_x_check' )
				
				with ridge_c2:
					st.markdown( '###### ♟️ Solver / Iteration' )
					ridge_solvers = [ 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag',
						'saga',
						'lbfgs' ]
					
					ridge_solver_value = st.session_state.get( 'regression_ridge_solver', 'auto' )
					if ridge_solver_value not in ridge_solvers:
						ridge_solver_value = 'auto'
					
					ridge_solver = st.selectbox( 'Solver', options=ridge_solvers,
						index=ridge_solvers.index( ridge_solver_value ),
						key='regression_ridge_solver_select' )
					
					ridge_max_iter_raw = int(
						st.number_input( 'Max Iterations (0 = Auto)', min_value=0,
							value=int( st.session_state[ 'regression_ridge_max_iter' ] ), step=1,
							key='regression_ridge_max_iter_input' ) )
					
					ridge_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_ridge_tol' ] ), step=0.000100,
						format='%.6f', key='regression_ridge_tol_input' ) )
					
					ridge_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_ridge_positive' ] ),
						key='regression_ridge_positive_check' )
					
					if ridge_positive and ridge_solver != 'lbfgs':
						st.info( "Positive coefficients require the 'lbfgs' solver. "
						         "The solver will be set to 'lbfgs' during training." )
					
					if not ridge_positive and ridge_solver == 'lbfgs':
						st.warning( "The 'lbfgs' solver requires Positive Coefficients." )
				
				with ridge_c3:
					st.markdown( '###### ↔️ Data Split' )
					ridge_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_ridge_test_size' ] * 100 ),
						step=5,
						key='regression_ridge_test_size_slider' ) / 100.0
					
					ridge_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_ridge_random_state' ] ), step=1,
						key='regression_ridge_random_state_input' ) )
				
				ridge_btn_1, ridge_btn_2 = st.columns( 2 )
				with ridge_btn_1:
					train_ridge = st.button( '🚂 Train Ridge Regression',
						key='regression_ridge_train', use_container_width=True )
				
				with ridge_btn_2:
					reset_ridge = st.button( '🔄 Reset Ridge Regression',
						key='regression_ridge_reset', use_container_width=True )
				
				if reset_ridge:
					for key, value in ridge_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_ridge_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_ridge:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Ridge Regression requires prepared feature & target arrays.' )
							st.stop( )
						
						X_ridge = np.asarray( X, dtype=float )
						y_ridge = np.asarray( y, dtype=float ).reshape( -1 )
						if X_ridge.ndim != 2 or X_ridge.shape[ 1 ] < 1:
							st.warning(
								'⚠️ Ridge Regression requires at least one numeric feature.' )
							st.stop( )
						
						if y_ridge.ndim != 1:
							st.warning( '⚠️ The Ridge Regression target must be one-dimensional.' )
							st.stop( )
						
						if len( X_ridge ) != len( y_ridge ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_ridge ) < 2:
							st.warning( '⚠️ Ridge Regression requires at least two observations.' )
							st.stop( )
						
						if not np.isfinite( X_ridge ).all( ):
							st.warning( '⚠️ The Ridge feature matrix contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_ridge ).all( ):
							st.warning( '⚠️ The Ridge target contains non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_ridge ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if ridge_alpha <= 0.0:
							st.warning( '⚠️ Ridge alpha must be greater than zero.' )
							st.stop( )
						
						if ridge_tol < 0.0:
							st.warning( '⚠️ Ridge tolerance cannot be negative.' )
							st.stop( )
						
						if (not ridge_positive and ridge_solver == 'lbfgs'):
							st.warning( "⚠️ The 'lbfgs' solver requires Positive Coefficients." )
							st.stop( )
						
						effective_solver = ('lbfgs' if ridge_positive else str( ridge_solver ))
						effective_max_iter = (
							None if ridge_max_iter_raw == 0 else int( ridge_max_iter_raw ))
						
						st.session_state[ 'regression_ridge_alpha' ] = float( ridge_alpha )
						st.session_state[ 'regression_ridge_fit_intercept' ] = bool(
							ridge_fit_intercept )
						st.session_state[ 'regression_ridge_copy_x' ] = bool( ridge_copy_x )
						st.session_state[ 'regression_ridge_max_iter' ] = int( ridge_max_iter_raw )
						st.session_state[ 'regression_ridge_tol' ] = float( ridge_tol )
						st.session_state[ 'regression_ridge_solver' ] = str( ridge_solver )
						st.session_state[ 'regression_ridge_positive' ] = bool( ridge_positive )
						st.session_state[ 'regression_ridge_test_size' ] = float( ridge_test_size )
						st.session_state[ 'regression_ridge_random_state' ] = int(
							ridge_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.Ridge( alpha=float( ridge_alpha ),
							fit=bool( ridge_fit_intercept ), copy=bool( ridge_copy_x ),
							iters=effective_max_iter, tol=float( ridge_tol ),
							solver=effective_solver, positive=bool( ridge_positive ),
							rando=int( ridge_random_state ) )
						
						X_train, X_test, y_train, y_test = (
							model.split_data( X_ridge, y_ridge, size=float( ridge_test_size ),
								random=int( ridge_random_state ) ))
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Alpha', 'Solver', 'Maximum Iterations',
								'Positive Coefficients' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), float( ridge_alpha ),
								str( effective_solver ),
								('Auto' if effective_max_iter is None else int(
									effective_max_iter )), bool( ridge_positive ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						coefficient_values = np.asarray( model.weights, dtype=float ).reshape( -1 )
						if len( coefficient_values ) != len( active_features ):
							coefficient_names = [ f'Feature {index + 1}' for index in
								range( len( coefficient_values ) ) ]
						else:
							coefficient_names = active_features.copy( )
						
						df_coefficients = pd.DataFrame(
							{ 'Feature': coefficient_names, 'Coefficient': coefficient_values } )
						
						st.session_state[ 'regression_ridge_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.session_state[ 'df_coefficients' ] = df_coefficients.copy( )
						st.success( 'Ridge Regression training completed.' )
					except Exception as ex:
						st.error( f'Ridge Regression training failed: {ex}' )
			
			with st.expander( 'Lasso Regression', expanded=False ):
				lasso_defaults = { 'regression_lasso_alpha': 1.0,
					'regression_lasso_fit_intercept': True, 'regression_lasso_precompute': False,
					'regression_lasso_copy_x': True, 'regression_lasso_max_iter': 1000,
					'regression_lasso_tol': 0.0001, 'regression_lasso_warm_start': False,
					'regression_lasso_positive': False, 'regression_lasso_random_state': 42,
					'regression_lasso_selection': 'cyclic', 'regression_lasso_test_size': 0.20 }
				
				for key, value in lasso_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'L1-regularized linear regression for continuous targets.' )
				lasso_c1, lasso_c2, lasso_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with lasso_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					lasso_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_lasso_alpha' ] ), step=0.100000,
						format='%.6f', key='regression_lasso_alpha_input' ) )
					
					lasso_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_lasso_fit_intercept' ] ),
						key='regression_lasso_fit_intercept_check' )
					
					lasso_precompute = st.checkbox( 'Precompute',
						value=bool( st.session_state[ 'regression_lasso_precompute' ] ),
						key='regression_lasso_precompute_check' )
					
					lasso_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_lasso_copy_x' ] ),
						key='regression_lasso_copy_x_check' )
				
				with lasso_c2:
					st.markdown( '###### Solver / Iteration' )
					
					lasso_max_iter = int( st.number_input( 'Max Iterations', min_value=1,
						value=int( st.session_state[ 'regression_lasso_max_iter' ] ), step=1,
						key='regression_lasso_max_iter_input' ) )
					
					lasso_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_lasso_tol' ] ), step=0.000100,
						format='%.6f', key='regression_lasso_tol_input' ) )
					
					lasso_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_lasso_warm_start' ] ),
						key='regression_lasso_warm_start_check' )
					
					lasso_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_lasso_positive' ] ),
						key='regression_lasso_positive_check' )
					
					lasso_selection_options = [ 'cyclic', 'random' ]
					
					lasso_selection_value = st.session_state.get( 'regression_lasso_selection',
						'cyclic' )
					
					if lasso_selection_value not in lasso_selection_options:
						lasso_selection_value = 'cyclic'
					
					lasso_selection = st.selectbox( 'Selection', options=lasso_selection_options,
						index=lasso_selection_options.index( lasso_selection_value ),
						key='regression_lasso_selection_select' )
				
				with lasso_c3:
					st.markdown( '###### Data Split' )
					lasso_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_lasso_test_size' ] * 100 ),
						step=5,
						key='regression_lasso_test_size_slider' ) / 100.0
					
					lasso_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_lasso_random_state' ] ), step=1,
						key='regression_lasso_random_state_input' ) )
				
				lasso_btn_1, lasso_btn_2 = st.columns( 2 )
				with lasso_btn_1:
					train_lasso = st.button( '🚂 Train Lasso Regression',
						key='regression_lasso_train', use_container_width=True )
				
				with lasso_btn_2:
					reset_lasso = st.button( '🔄 Reset Lasso Regression',
						key='regression_lasso_reset', use_container_width=True )
				
				if reset_lasso:
					for key, value in lasso_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_lasso_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_lasso:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Lasso Regression requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_lasso = np.asarray( X, dtype=float )
						y_lasso = np.asarray( y, dtype=float ).reshape( -1 )
						if X_lasso.ndim != 2 or X_lasso.shape[ 1 ] < 1:
							st.warning( '⚠️ Lasso Regression requires at least one '
							            'numeric feature.' )
							st.stop( )
						
						if y_lasso.ndim != 1:
							st.warning( '⚠️ The Lasso Regression target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_lasso ) != len( y_lasso ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_lasso ) < 2:
							st.warning( '⚠️ Lasso Regression requires at least two '
							            'observations.' )
							st.stop( )
						
						if not np.isfinite( X_lasso ).all( ):
							st.warning( '⚠️ The Lasso feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_lasso ).all( ):
							st.warning( '⚠️ The Lasso target contains non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_lasso ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if lasso_alpha <= 0.0:
							st.warning( '⚠️ Lasso alpha must be greater than zero.' )
							st.stop( )
						
						if lasso_max_iter < 1:
							st.warning( '⚠️ Maximum iterations must be greater than zero.' )
							st.stop( )
						
						if lasso_tol < 0.0:
							st.warning( '⚠️ Lasso tolerance cannot be negative.' )
							st.stop( )
						
						st.session_state[ 'regression_lasso_alpha' ] = float( lasso_alpha )
						st.session_state[ 'regression_lasso_fit_intercept' ] = bool(
							lasso_fit_intercept )
						st.session_state[ 'regression_lasso_precompute' ] = bool(
							lasso_precompute )
						st.session_state[ 'regression_lasso_copy_x' ] = bool( lasso_copy_x )
						st.session_state[ 'regression_lasso_max_iter' ] = int( lasso_max_iter )
						st.session_state[ 'regression_lasso_tol' ] = float( lasso_tol )
						st.session_state[ 'regression_lasso_warm_start' ] = bool(
							lasso_warm_start )
						st.session_state[ 'regression_lasso_positive' ] = bool( lasso_positive )
						st.session_state[ 'regression_lasso_random_state' ] = int(
							lasso_random_state )
						st.session_state[ 'regression_lasso_selection' ] = str( lasso_selection )
						st.session_state[ 'regression_lasso_test_size' ] = float( lasso_test_size )
						start_time = time.perf_counter( )
						model = regression_model.Lasso( alpha=float( lasso_alpha ),
							fit=bool( lasso_fit_intercept ), precompute=bool( lasso_precompute ),
							copy=bool( lasso_copy_x ), iters=int( lasso_max_iter ),
							tol=float( lasso_tol ), warm=bool( lasso_warm_start ),
							positive=bool( lasso_positive ), rando=int( lasso_random_state ),
							selection=str( lasso_selection ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_lasso, y_lasso,
							size=float( lasso_test_size ), random=int( lasso_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Alpha', 'Selection', 'Maximum Iterations',
								'Positive Coefficients' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), float( lasso_alpha ), str( lasso_selection ),
								int( lasso_max_iter ), bool( lasso_positive ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						coefficient_values = np.asarray( model.weights, dtype=float ).reshape( -1 )
						if len( coefficient_values ) == len( active_features ):
							coefficient_names = active_features.copy( )
						else:
							coefficient_names = [ f'Feature {index + 1}' for index in
								range( len( coefficient_values ) ) ]
						
						df_coefficients = pd.DataFrame(
							{ 'Feature': coefficient_names, 'Coefficient': coefficient_values } )
						
						st.session_state[ 'regression_lasso_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.session_state[ 'df_coefficients' ] = df_coefficients.copy( )
						
						st.success( 'Lasso Regression training completed.' )
					
					except Exception as ex:
						st.error( f'Lasso Regression training failed: {ex}' )
			
			with st.expander( 'Elastic Net', expanded=False ):
				elastic_defaults = { 'regression_elastic_alpha': 1.0,
					'regression_elastic_ratio': 0.5, 'regression_elastic_fit_intercept': True,
					'regression_elastic_precompute': False, 'regression_elastic_copy_x': True,
					'regression_elastic_max_iter': 1000, 'regression_elastic_tol': 0.0001,
					'regression_elastic_warm_start': False, 'regression_elastic_positive': False,
					'regression_elastic_random_state': 42, 'regression_elastic_selection':
						'cyclic',
					'regression_elastic_test_size': 0.20 }
				
				for key, value in elastic_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Combined L1/L2-regularized linear regression for continuous '
				            'targets.' )
				
				elastic_c1, elastic_c2, elastic_c3 = st.columns( [ 0.34, 0.33, 0.33 ],
					border=True )
				
				with elastic_c1:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					elastic_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_elastic_alpha' ] ),
						step=0.100000, format='%.6f', key='regression_elastic_alpha_input' ) )
					
					elastic_ratio = float( st.slider( 'L1 Ratio', min_value=0.0, max_value=1.0,
						value=float( st.session_state[ 'regression_elastic_ratio' ] ), step=0.05,
						key='regression_elastic_ratio_slider' ) )
					
					elastic_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_elastic_fit_intercept' ] ),
						key='regression_elastic_fit_intercept_check' )
					
					elastic_precompute = st.checkbox( 'Precompute',
						value=bool( st.session_state[ 'regression_elastic_precompute' ] ),
						key='regression_elastic_precompute_check' )
				
				with elastic_c2:
					st.markdown( '###### Solver / Iteration' )
					elastic_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_elastic_copy_x' ] ),
						key='regression_elastic_copy_x_check' )
					
					elastic_max_iter = int( st.number_input( 'Max Iterations', min_value=1,
						value=int( st.session_state[ 'regression_elastic_max_iter' ] ), step=1,
						key='regression_elastic_max_iter_input' ) )
					
					elastic_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_elastic_tol' ] ), step=0.000100,
						format='%.6f', key='regression_elastic_tol_input' ) )
					
					elastic_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_elastic_warm_start' ] ),
						key='regression_elastic_warm_start_check' )
					
					elastic_positive = st.checkbox( 'Positive Coefficients',
						value=bool( st.session_state[ 'regression_elastic_positive' ] ),
						key='regression_elastic_positive_check' )
					
					elastic_selection_options = [ 'cyclic', 'random' ]
					elastic_selection_value = st.session_state.get( 'regression_elastic_selection',
						'cyclic' )
					
					if elastic_selection_value not in elastic_selection_options:
						elastic_selection_value = 'cyclic'
					
					elastic_selection = st.selectbox( 'Selection',
						options=elastic_selection_options,
						index=elastic_selection_options.index( elastic_selection_value ),
						key='regression_elastic_selection_select' )
				
				with elastic_c3:
					st.markdown( '###### Data Split' )
					elastic_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_elastic_test_size' ] * 100 ),
						step=5, key='regression_elastic_test_size_slider' ) / 100.0
					
					elastic_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_elastic_random_state' ] ), step=1,
						key='regression_elastic_random_state_input' ) )
					
					if elastic_ratio == 0.0:
						st.info( 'l1_ratio = 0.0 uses only L2 regularization.' )
					
					if elastic_ratio == 1.0:
						st.info( 'l1_ratio = 1.0 is equivalent to Lasso.' )
				
				elastic_btn_1, elastic_btn_2 = st.columns( 2 )
				with elastic_btn_1:
					train_elastic = st.button( '🚂 Train Elastic Net',
						key='regression_elastic_train', use_container_width=True )
				
				with elastic_btn_2:
					reset_elastic = st.button( '🔄 Reset Elastic Net',
						key='regression_elastic_reset', use_container_width=True )
				
				if reset_elastic:
					for key, value in elastic_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_elastic_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_elastic:
					try:
						if X is None or y is None:
							st.warning(
								'⚠️ Elastic Net requires prepared feature and target arrays.' )
							st.stop( )
						
						X_elastic = np.asarray( X, dtype=float )
						
						y_elastic = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_elastic.ndim != 2 or X_elastic.shape[ 1 ] < 1:
							st.warning( '⚠️ Elastic Net requires at least one numeric feature.' )
							st.stop( )
						
						if y_elastic.ndim != 1:
							st.warning( '⚠️ The Elastic Net target must be one-dimensional.' )
							st.stop( )
						
						if len( X_elastic ) != len( y_elastic ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_elastic ) < 2:
							st.warning( '⚠️ Elastic Net requires at least two observations.' )
							st.stop( )
						
						if not np.isfinite( X_elastic ).all( ):
							st.warning(
								'⚠️ The Elastic Net feature matrix contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_elastic ).all( ):
							st.warning( '⚠️ The Elastic Net target contains non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_elastic ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if elastic_alpha <= 0.0:
							st.warning( '⚠️ Elastic Net alpha must be greater than zero.' )
							st.stop( )
						
						if elastic_ratio < 0.0 or elastic_ratio > 1.0:
							st.warning( '⚠️ The Elastic Net L1 ratio must be between '
							            '0.0 and 1.0.' )
							st.stop( )
						
						if elastic_max_iter < 1:
							st.warning( '⚠️ Maximum iterations must be greater than zero.' )
							st.stop( )
						
						if elastic_tol < 0.0:
							st.warning( '⚠️ Elastic Net tolerance cannot be negative.' )
							st.stop( )
						
						st.session_state[ 'regression_elastic_alpha' ] = float( elastic_alpha )
						st.session_state[ 'regression_elastic_ratio' ] = float( elastic_ratio )
						st.session_state[ 'regression_elastic_fit_intercept' ] = bool(
							elastic_fit_intercept )
						st.session_state[ 'regression_elastic_precompute' ] = bool(
							elastic_precompute )
						st.session_state[ 'regression_elastic_copy_x' ] = bool( elastic_copy_x )
						st.session_state[ 'regression_elastic_max_iter' ] = int( elastic_max_iter )
						st.session_state[ 'regression_elastic_tol' ] = float( elastic_tol )
						st.session_state[ 'regression_elastic_warm_start' ] = bool(
							elastic_warm_start )
						st.session_state[ 'regression_elastic_positive' ] = bool(
							elastic_positive )
						st.session_state[ 'regression_elastic_random_state' ] = int(
							elastic_random_state )
						st.session_state[ 'regression_elastic_selection' ] = str(
							elastic_selection )
						st.session_state[ 'regression_elastic_test_size' ] = float(
							elastic_test_size )
						start_time = time.perf_counter( )
						model = regression_model.ElasticNet( alpha=float( elastic_alpha ),
							ratio=float( elastic_ratio ), fit=bool( elastic_fit_intercept ),
							precompute=bool( elastic_precompute ), iters=int( elastic_max_iter ),
							copy=bool( elastic_copy_x ), tol=float( elastic_tol ),
							warm=bool( elastic_warm_start ), positive=bool( elastic_positive ),
							rando=int( elastic_random_state ), select=str( elastic_selection ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_elastic, y_elastic,
							size=float( elastic_test_size ), random=int( elastic_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Alpha', 'L1 Ratio', 'Selection',
								'Maximum Iterations', 'Positive Coefficients' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), float( elastic_alpha ),
								float( elastic_ratio ), str( elastic_selection ),
								int( elastic_max_iter ), bool( elastic_positive ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						coefficient_values = np.asarray( model.weights, dtype=float ).reshape( -1 )
						if len( coefficient_values ) == len( active_features ):
							coefficient_names = active_features.copy( )
						else:
							coefficient_names = [ f'Feature {index + 1}' for index in
								range( len( coefficient_values ) ) ]
						
						df_coefficients = pd.DataFrame(
							{ 'Feature': coefficient_names, 'Coefficient': coefficient_values } )
						
						st.session_state[ 'regression_elastic_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.session_state[ 'df_coefficients' ] = df_coefficients.copy( )
						st.success( 'Elastic Net training completed.' )
					except Exception as ex:
						st.error( f'Elastic Net training failed: {ex}' )
			
			with st.expander( 'Bayesian Ridge', expanded=False ):
				bayes_defaults = { 'regression_bayes_max_iter': 300,
					'regression_bayes_shape_alpha': 0.000001,
					'regression_bayes_scale_alpha': 0.000001,
					'regression_bayes_shape_lambda': 0.000001,
					'regression_bayes_scale_lambda': 0.000001, 'regression_bayes_tol': 0.001000,
					'regression_bayes_alpha_init': 0.0, 'regression_bayes_lambda_init': 0.0,
					'regression_bayes_compute_score': False, 'regression_bayes_fit_intercept':
						True,
					'regression_bayes_copy_x': True, 'regression_bayes_verbose': False,
					'regression_bayes_test_size': 0.20, 'regression_bayes_random_state': 42 }
				
				for key, value in bayes_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Bayesian linear regression with automatic regularization '
				            'estimation.' )
				
				bayes_c1, bayes_c2, bayes_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with bayes_c1:
					st.markdown( '###### Prior / Precision Parameters' )
					
					bayes_shape_alpha = float( st.number_input( 'Alpha Shape', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_shape_alpha' ] ),
						step=0.000001, format='%.6f', key='regression_bayes_shape_alpha_input' ) )
					
					bayes_scale_alpha = float( st.number_input( 'Alpha Scale', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_scale_alpha' ] ),
						step=0.000001, format='%.6f', key='regression_bayes_scale_alpha_input' ) )
					
					bayes_shape_lambda = float( st.number_input( 'Lambda Shape', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_shape_lambda' ] ),
						step=0.000001, format='%.6f', key='regression_bayes_shape_lambda_input' ) )
					
					bayes_scale_lambda = float( st.number_input( 'Lambda Scale', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_scale_lambda' ] ),
						step=0.000001, format='%.6f', key='regression_bayes_scale_lambda_input' ) )
				
				with bayes_c2:
					st.markdown( '###### 🎚️ Hyper Parameters' )
					
					bayes_max_iter = int( st.number_input( 'Max Iterations', min_value=1,
						value=int( st.session_state[ 'regression_bayes_max_iter' ] ), step=1,
						key='regression_bayes_max_iter_input' ) )
					
					bayes_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_bayes_tol' ] ), step=0.000100,
						format='%.6f', key='regression_bayes_tol_input' ) )
					
					bayes_alpha_init_raw = float(
						st.number_input( 'Alpha Init (0 = None)', min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_alpha_init' ] ),
							step=0.000100, format='%.6f',
							key='regression_bayes_alpha_init_input' ) )
					
					bayes_lambda_init_raw = float(
						st.number_input( 'Lambda Init (0 = None)', min_value=0.0,
							value=float( st.session_state[ 'regression_bayes_lambda_init' ] ),
							step=0.000100, format='%.6f',
							key='regression_bayes_lambda_init_input' ) )
					
					bayes_compute_score = st.checkbox( 'Compute Marginal Log Likelihood',
						value=bool( st.session_state[ 'regression_bayes_compute_score' ] ),
						key='regression_bayes_compute_score_check' )
					
					bayes_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_bayes_fit_intercept' ] ),
						key='regression_bayes_fit_intercept_check' )
					
					bayes_copy_x = st.checkbox( 'Copy X',
						value=bool( st.session_state[ 'regression_bayes_copy_x' ] ),
						key='regression_bayes_copy_x_check' )
					
					bayes_verbose = st.checkbox( 'Verbose',
						value=bool( st.session_state[ 'regression_bayes_verbose' ] ),
						key='regression_bayes_verbose_check' )
				
				with bayes_c3:
					st.markdown( '###### Data Split' )
					
					bayes_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_bayes_test_size' ] * 100 ),
						step=5,
						key='regression_bayes_test_size_slider' ) / 100.0
					
					bayes_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_bayes_random_state' ] ), step=1,
						key='regression_bayes_random_state_input' ) )
				
				bayes_btn_1, bayes_btn_2 = st.columns( 2 )
				with bayes_btn_1:
					train_bayes = st.button( '🚂 Train Bayesian Ridge',
						key='regression_bayes_train',
						use_container_width=True )
				
				with bayes_btn_2:
					reset_bayes = st.button( '🔄 Reset Bayesian Ridge',
						key='regression_bayes_reset',
						use_container_width=True )
				
				if reset_bayes:
					for key, value in bayes_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.session_state[ 'regression_bayes_elapsed_seconds' ]=None
					st.rerun( )
				
				if train_bayes:
					try:
						st.session_state[ 'regression_bayes_max_iter' ] = int( bayes_max_iter )
						st.session_state[ 'regression_bayes_shape_alpha' ] = float(
							bayes_shape_alpha )
						st.session_state[ 'regression_bayes_scale_alpha' ] = float(
							bayes_scale_alpha )
						st.session_state[ 'regression_bayes_shape_lambda' ] = float(
							bayes_shape_lambda )
						st.session_state[ 'regression_bayes_scale_lambda' ] = float(
							bayes_scale_lambda )
						st.session_state[ 'regression_bayes_tol' ] = float( bayes_tol )
						st.session_state[ 'regression_bayes_alpha_init' ] = float(
							bayes_alpha_init_raw )
						st.session_state[ 'regression_bayes_lambda_init' ] = float(
							bayes_lambda_init_raw )
						st.session_state[ 'regression_bayes_compute_score' ] = bool(
							bayes_compute_score )
						st.session_state[ 'regression_bayes_fit_intercept' ] = bool(
							bayes_fit_intercept )
						st.session_state[ 'regression_bayes_copy_x' ] = bool( bayes_copy_x )
						st.session_state[ 'regression_bayes_verbose' ] = bool( bayes_verbose )
						st.session_state[ 'regression_bayes_test_size' ] = float( bayes_test_size )
						st.session_state[ 'regression_bayes_random_state' ] = int(
							bayes_random_state )
						df_training = df_model.copy( )
						X = df_training[ active_features ].apply( pd.to_numeric,
							errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning( '⚠️ The target must contain at least two distinct '
							            'values.' )
							st.stop( )
						
						bayes_alpha_init = None if bayes_alpha_init_raw == 0.0 else float(
							bayes_alpha_init_raw )
						
						bayes_lambda_init = None if bayes_lambda_init_raw == 0.0 else float(
							bayes_lambda_init_raw )
						
						start_time = time.perf_counter( )
						
						model = regression_model.BayesianRidge( max=int( bayes_max_iter ),
							shape_alpha=float( bayes_shape_alpha ),
							scale_alpha=float( bayes_scale_alpha ),
							shape_lambda=float( bayes_shape_lambda ),
							scale_lambda=float( bayes_scale_lambda ), tol=float( bayes_tol ),
							alpha_init=bayes_alpha_init, lambda_init=bayes_lambda_init,
							compute_score=bool( bayes_compute_score ),
							fit=bool( bayes_fit_intercept ), copy=bool( bayes_copy_x ),
							verbose=bool( bayes_verbose ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( bayes_test_size ), random=int( bayes_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						st.session_state[ 'regression_bayes_elapsed_seconds' ] = elapsed_seconds
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame( {
								'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
									'Testing Rows', 'Max Iterations' ],
								'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
									int( len( X_test ) ), int( bayes_max_iter ) ] } )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': y_test, 'Predicted': y_prediction } )
						
						df_coefficients = pd.DataFrame( { 'Feature': active_features,
							'Coefficient': np.asarray( model.model.coef_ ).reshape( -1 ) } )
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_regression' ] = df_training.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Bayesian Ridge training failed: {ex}' )
			
			with st.expander( 'Stochastic Gradient Descent', expanded=False ):
				sgd_defaults = { 'regression_sgd_loss': 'squared_error',
					'regression_sgd_penalty': 'l2', 'regression_sgd_alpha': 0.0001,
					'regression_sgd_iters': 1000, 'regression_sgd_shuffle': True,
					'regression_sgd_learning_rate': 'invscaling', 'regression_sgd_l1_ratio': 0.15,
					'regression_sgd_fit_intercept': True, 'regression_sgd_tol': 0.001,
					'regression_sgd_verbose': 0, 'regression_sgd_epsilon': 0.1,
					'regression_sgd_eta0': 0.01, 'regression_sgd_power_t': 0.25,
					'regression_sgd_early_stopping': False,
					'regression_sgd_validation_fraction': 0.1,
					'regression_sgd_n_iter_no_change': 5,
					'regression_sgd_warm_start': False, 'regression_sgd_average': False,
					'regression_sgd_test_size': 0.20, 'regression_sgd_random_state': 42 }
				
				for key, value in sgd_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption(
					'Linear regression trained with SGD for large-scale continuous targets.' )
				
				sgd_c1, sgd_c2, sgd_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with sgd_c1:
					st.markdown( '###### Loss / Penalty' )
					
					sgd_loss_options = [ 'squared_error', 'huber', 'epsilon_insensitive',
						'squared_epsilon_insensitive' ]
					
					sgd_loss_value = st.session_state.get( 'regression_sgd_loss', 'squared_error' )
					
					if sgd_loss_value not in sgd_loss_options:
						sgd_loss_value = 'squared_error'
					
					sgd_loss = st.selectbox( 'Loss', options=sgd_loss_options,
						index=sgd_loss_options.index( sgd_loss_value ),
						key='regression_sgd_loss_select' )
					
					sgd_penalty_options = [ None, 'l2', 'l1', 'elasticnet' ]
					sgd_penalty_value = st.session_state.get( 'regression_sgd_penalty', 'l2' )
					if sgd_penalty_value not in sgd_penalty_options:
						sgd_penalty_value = 'l2'
					
					sgd_penalty = st.selectbox( 'Penalty', options=sgd_penalty_options,
						index=sgd_penalty_options.index( sgd_penalty_value ),
						format_func=lambda value: 'None' if value is None else str( value ),
						key='regression_sgd_penalty_select' )
					
					sgd_alpha = float( st.number_input( 'Alpha', min_value=0.000001,
						value=float( st.session_state[ 'regression_sgd_alpha' ] ), step=0.0001,
						format='%.6f', key='regression_sgd_alpha_input' ) )
					
					sgd_iters = int( st.number_input( 'Iterations', min_value=1,
						value=int( st.session_state[ 'regression_sgd_iters' ] ), step=1,
						key='regression_sgd_iters_input' ) )
					
					sgd_l1_ratio = float( st.slider( 'L1 Ratio', min_value=0.0, max_value=1.0,
						value=float( st.session_state[ 'regression_sgd_l1_ratio' ] ), step=0.05,
						key='regression_sgd_l1_ratio_slider' ) )
				
				with sgd_c2:
					st.markdown( '###### Learning Controls' )
					
					sgd_shuffle = st.checkbox( 'Shuffle',
						value=bool( st.session_state[ 'regression_sgd_shuffle' ] ),
						key='regression_sgd_shuffle_check' )
					
					sgd_learning_rate_options = [ 'constant', 'optimal', 'invscaling', 'adaptive' ]
					
					sgd_learning_rate_value = st.session_state.get( 'regression_sgd_learning_rate',
						'invscaling' )
					
					if sgd_learning_rate_value not in sgd_learning_rate_options:
						sgd_learning_rate_value = 'invscaling'
					
					sgd_learning_rate = st.selectbox( 'Learning Rate Schedule',
						options=sgd_learning_rate_options,
						index=sgd_learning_rate_options.index( sgd_learning_rate_value ),
						key='regression_sgd_learning_rate_select' )
					
					sgd_eta0 = float( st.number_input( 'Eta0', min_value=0.000001,
						value=float( st.session_state[ 'regression_sgd_eta0' ] ), step=0.01,
						format='%.6f', key='regression_sgd_eta0_input' ) )
					
					sgd_power_t = float( st.number_input( 'Power T', min_value=0.0,
						value=float( st.session_state[ 'regression_sgd_power_t' ] ), step=0.1,
						format='%.6f', key='regression_sgd_power_t_input' ) )
					
					sgd_epsilon = float( st.number_input( 'Epsilon', min_value=0.0,
						value=float( st.session_state[ 'regression_sgd_epsilon' ] ), step=0.01,
						format='%.6f', key='regression_sgd_epsilon_input' ) )
					
					sgd_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_sgd_tol' ] ), step=0.0001,
						format='%.6f', key='regression_sgd_tol_input' ) )
					
					sgd_fit_intercept = st.checkbox( 'Fit Intercept',
						value=bool( st.session_state[ 'regression_sgd_fit_intercept' ] ),
						key='regression_sgd_fit_intercept_check' )
				
				with sgd_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					sgd_early_stopping = st.checkbox( 'Early Stopping',
						value=bool( st.session_state[ 'regression_sgd_early_stopping' ] ),
						key='regression_sgd_early_stopping_check' )
					
					sgd_validation_fraction = float(
						st.slider( 'Validation Fraction', min_value=0.05, max_value=0.40,
							value=float( st.session_state[ 'regression_sgd_validation_fraction'
							] ),
							step=0.05, key='regression_sgd_validation_fraction_slider' ) )
					
					sgd_n_iter_no_change = int( st.number_input( 'N Iter No Change', min_value=1,
						value=int( st.session_state[ 'regression_sgd_n_iter_no_change' ] ), step=1,
						key='regression_sgd_n_iter_no_change_input' ) )
					
					sgd_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_sgd_warm_start' ] ),
						key='regression_sgd_warm_start_check' )
					
					sgd_average = st.checkbox( 'Average Weights',
						value=bool( st.session_state[ 'regression_sgd_average' ] ),
						key='regression_sgd_average_check' )
					
					sgd_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_sgd_verbose' ] ), step=1,
						key='regression_sgd_verbose_input' ) )
					
					sgd_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_sgd_test_size' ] * 100 ), step=5,
						key='regression_sgd_test_size_slider' ) / 100.0
					
					sgd_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_sgd_random_state' ] ), step=1,
						key='regression_sgd_random_state_input' ) )
					
					if sgd_penalty != 'elasticnet':
						st.caption( 'L1 Ratio is only used when Penalty = elasticnet.' )
					
					epsilon_losses = [ 'huber', 'epsilon_insensitive',
						'squared_epsilon_insensitive' ]
					
					if sgd_loss not in epsilon_losses:
						st.caption( 'Epsilon is only used by Huber and '
						            'epsilon-insensitive losses.' )
				
				sgd_btn_1, sgd_btn_2 = st.columns( 2 )
				
				with sgd_btn_1:
					train_sgd = st.button( '🚂 Train Stochastic Gradient Descent',
						key='regression_sgd_train', use_container_width=True )
				
				with sgd_btn_2:
					reset_sgd = st.button( '🔄 Reset Stochastic Gradient Descent',
						key='regression_sgd_reset', use_container_width=True )
				
				if reset_sgd:
					for key, value in sgd_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_sgd_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'df_coefficients' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_sgd:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Stochastic Gradient Descent requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_sgd = np.asarray( X, dtype=float )
						y_sgd = np.asarray( y, dtype=float ).reshape( -1 )
						if X_sgd.ndim != 2 or X_sgd.shape[ 1 ] < 1:
							st.warning( '⚠️ Stochastic Gradient Descent requires at least '
							            'one numeric feature.' )
							st.stop( )
						
						if y_sgd.ndim != 1:
							st.warning( '⚠️ The Stochastic Gradient Descent target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_sgd ) != len( y_sgd ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_sgd ) < 2:
							st.warning( '⚠️ Stochastic Gradient Descent requires at least '
							            'two observations.' )
							st.stop( )
						
						if not np.isfinite( X_sgd ).all( ):
							st.warning( '⚠️ The Stochastic Gradient Descent feature matrix '
							            'contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_sgd ).all( ):
							st.warning( '⚠️ The Stochastic Gradient Descent target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_sgd ) ) < 2:
							st.warning( '⚠️ The regression target must contain at least '
							            'two distinct values.' )
							st.stop( )
						
						if sgd_alpha <= 0.0:
							st.warning( '⚠️ SGD alpha must be greater than zero.' )
							st.stop( )
						
						if sgd_iters < 1:
							st.warning( '⚠️ SGD iterations must be greater than zero.' )
							st.stop( )
						
						if sgd_tol < 0.0:
							st.warning( '⚠️ SGD tolerance cannot be negative.' )
							st.stop( )
						
						if sgd_eta0 <= 0.0:
							st.warning( '⚠️ SGD Eta0 must be greater than zero.' )
							st.stop( )
						
						if sgd_power_t < 0.0:
							st.warning( '⚠️ SGD Power T cannot be negative.' )
							st.stop( )
						
						if sgd_epsilon < 0.0:
							st.warning( '⚠️ SGD epsilon cannot be negative.' )
							st.stop( )
						
						if sgd_l1_ratio < 0.0 or sgd_l1_ratio > 1.0:
							st.warning( '⚠️ SGD L1 Ratio must be between 0.0 and 1.0.' )
							st.stop( )
						
						if (sgd_validation_fraction <= 0.0 or sgd_validation_fraction >= 1.0):
							st.warning( '⚠️ Validation Fraction must be greater than zero '
							            'and less than one.' )
							st.stop( )
						
						if sgd_n_iter_no_change < 1:
							st.warning( '⚠️ N Iter No Change must be greater than zero.' )
							st.stop( )
						
						st.session_state[ 'regression_sgd_loss' ] = str( sgd_loss )
						st.session_state[ 'regression_sgd_penalty' ] = sgd_penalty
						st.session_state[ 'regression_sgd_alpha' ] = float( sgd_alpha )
						st.session_state[ 'regression_sgd_iters' ] = int( sgd_iters )
						st.session_state[ 'regression_sgd_shuffle' ] = bool( sgd_shuffle )
						st.session_state[ 'regression_sgd_learning_rate' ] = str(
							sgd_learning_rate )
						st.session_state[ 'regression_sgd_l1_ratio' ] = float( sgd_l1_ratio )
						st.session_state[ 'regression_sgd_fit_intercept' ] = bool(
							sgd_fit_intercept )
						st.session_state[ 'regression_sgd_tol' ] = float( sgd_tol )
						st.session_state[ 'regression_sgd_verbose' ] = int( sgd_verbose )
						st.session_state[ 'regression_sgd_epsilon' ] = float( sgd_epsilon )
						st.session_state[ 'regression_sgd_eta0' ] = float( sgd_eta0 )
						st.session_state[ 'regression_sgd_power_t' ] = float( sgd_power_t )
						st.session_state[ 'regression_sgd_early_stopping' ] = bool(
							sgd_early_stopping )
						st.session_state[ 'regression_sgd_validation_fraction' ] = float(
							sgd_validation_fraction )
						st.session_state[ 'regression_sgd_n_iter_no_change' ] = int(
							sgd_n_iter_no_change )
						st.session_state[ 'regression_sgd_warm_start' ] = bool( sgd_warm_start )
						st.session_state[ 'regression_sgd_average' ] = bool( sgd_average )
						st.session_state[ 'regression_sgd_test_size' ] = float( sgd_test_size )
						st.session_state[ 'regression_sgd_random_state' ] = int( sgd_random_state )
						start_time = time.perf_counter( )
						
						model = regression_model.GradientDescent( loss=str( sgd_loss ),
							iters=int( sgd_iters ), penalty=sgd_penalty, alpha=float( sgd_alpha ),
							rando=int( sgd_random_state ), learning_rate=str( sgd_learning_rate ),
							l1_ratio=float( sgd_l1_ratio ), fit=bool( sgd_fit_intercept ),
							tol=float( sgd_tol ), shuffle=bool( sgd_shuffle ),
							verbose=int( sgd_verbose ), epsilon=float( sgd_epsilon ),
							eta0=float( sgd_eta0 ), power_t=float( sgd_power_t ),
							early_stopping=bool( sgd_early_stopping ),
							validation_fraction=float( sgd_validation_fraction ),
							n_iter_no_change=int( sgd_n_iter_no_change ),
							warm=bool( sgd_warm_start ), average=bool( sgd_average ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_sgd, y_sgd,
							size=float( sgd_test_size ), random=int( sgd_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						if sgd_early_stopping and len( X_train ) < 10:
							st.warning( '⚠️ Early stopping requires a larger training set. '
							            'Disable Early Stopping or use more observations.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Loss', 'Penalty', 'Learning Rate', 'Iterations',
								'Alpha' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), str( sgd_loss ),
								('None' if sgd_penalty is None else str( sgd_penalty )),
								str( sgd_learning_rate ), int( sgd_iters ), float( sgd_alpha ) ]
						} )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						coefficient_values = np.asarray( model.weights, dtype=float ).reshape( -1 )
						if len( coefficient_values ) == len( active_features ):
							coefficient_names = active_features.copy( )
						else:
							coefficient_names = [ f'Feature {index + 1}' for index in
								range( len( coefficient_values ) ) ]
						
						df_coefficients = pd.DataFrame(
							{ 'Feature': coefficient_names, 'Coefficient': coefficient_values } )
						
						st.session_state[ 'regression_sgd_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						st.session_state[ 'df_coefficients' ] = df_coefficients.copy( )
						
						st.success( 'Stochastic Gradient Descent training completed.' )
					
					except Exception as ex:
						st.error( f'Stochastic Gradient Descent training failed: {ex}' )

		# ------------------------------------------------------------------
		# Instance Models
		# ------------------------------------------------------------------
		with st.expander( 'Instance Models', expanded=False ):
			
			with st.expander( 'k-Nearest Neighbors', expanded=False ):
				knn_defaults = { 'regression_knn_neighbors': 5, 'regression_knn_weights':
					'uniform',
					'regression_knn_algorithm': 'auto', 'regression_knn_leaf_size': 30,
					'regression_knn_power': 2.0, 'regression_knn_metric': 'minkowski',
					'regression_knn_jobs': 1, 'regression_knn_test_size': 0.20,
					'regression_knn_random_state': 42 }
				
				for key, value in knn_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Instance-based regression using nearby observations.' )
				
				knn_c1, knn_c2, knn_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with knn_c1:
					st.markdown( '###### Neighbor Parameters' )
					
					knn_neighbors = int( st.number_input( 'Neighbors', min_value=1,
						value=int( st.session_state[ 'regression_knn_neighbors' ] ), step=1,
						key='regression_knn_neighbors_input' ) )
					
					knn_weights_options = [ 'uniform', 'distance' ]
					
					knn_weights_value = st.session_state.get( 'regression_knn_weights', 'uniform' )
					
					if knn_weights_value not in knn_weights_options:
						knn_weights_value = 'uniform'
					
					knn_weights = st.selectbox( 'Weights', options=knn_weights_options,
						index=knn_weights_options.index( knn_weights_value ),
						key='regression_knn_weights_select' )
					
					knn_power = float( st.number_input( 'Power', min_value=1.0,
						value=float( st.session_state[ 'regression_knn_power' ] ), step=1.0,
						format='%.1f', key='regression_knn_power_input' ) )
					
					knn_leaf_size = int( st.number_input( 'Leaf Size', min_value=1,
						value=int( st.session_state[ 'regression_knn_leaf_size' ] ), step=1,
						key='regression_knn_leaf_size_input' ) )
				
				with knn_c2:
					st.markdown( '###### Distance / Search' )
					
					knn_algorithm_options = [ 'auto', 'ball_tree', 'kd_tree', 'brute' ]
					
					knn_algorithm_value = st.session_state.get( 'regression_knn_algorithm',
						'auto' )
					
					if knn_algorithm_value not in knn_algorithm_options:
						knn_algorithm_value = 'auto'
					
					knn_algorithm = st.selectbox( 'Algorithm', options=knn_algorithm_options,
						index=knn_algorithm_options.index( knn_algorithm_value ),
						key='regression_knn_algorithm_select' )
					
					knn_metric_options = [ 'minkowski', 'euclidean', 'manhattan', 'chebyshev',
						'canberra', 'braycurtis', 'cityblock', 'cosine', 'l1', 'l2',
						'nan_euclidean', 'hamming' ]
					
					knn_metric_value = st.session_state.get( 'regression_knn_metric', 'minkowski' )
					
					if knn_metric_value not in knn_metric_options:
						knn_metric_value = 'minkowski'
					
					knn_metric = st.selectbox( 'Metric', options=knn_metric_options,
						index=knn_metric_options.index( knn_metric_value ),
						key='regression_knn_metric_select' )
					
					knn_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_knn_jobs' ] ), step=1,
						key='regression_knn_jobs_input' ) )
					
					if knn_metric != 'minkowski':
						st.caption( 'Power is primarily used with the Minkowski metric.' )
				
				with knn_c3:
					st.markdown( '###### Data Split' )
					
					knn_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_knn_test_size' ] * 100 ), step=5,
						key='regression_knn_test_size_slider' ) / 100.0
					
					knn_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_knn_random_state' ] ), step=1,
						key='regression_knn_random_state_input' ) )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Target: {target_name}' )
				
				knn_btn_1, knn_btn_2 = st.columns( 2 )
				
				with knn_btn_1:
					train_knn = st.button( '🚂 Train k-Nearest Neighbors',
						key='regression_knn_train', use_container_width=True )
				
				with knn_btn_2:
					reset_knn = st.button( '🔄 Reset k-Nearest Neighbors',
						key='regression_knn_reset', use_container_width=True )
				
				if reset_knn:
					for key, value in knn_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_knn_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_knn:
					try:
						if X is None or y is None:
							st.warning( '⚠️ k-Nearest Neighbors requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_knn = np.asarray( X, dtype=float )
						
						y_knn = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_knn.ndim != 2 or X_knn.shape[ 1 ] < 1:
							st.warning( '⚠️ k-Nearest Neighbors requires at least one '
							            'numeric feature.' )
							st.stop( )
						
						if y_knn.ndim != 1:
							st.warning( '⚠️ The k-Nearest Neighbors target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_knn ) != len( y_knn ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_knn ) < 2:
							st.warning( '⚠️ k-Nearest Neighbors requires at least two '
							            'observations.' )
							st.stop( )
						
						if not np.isfinite( X_knn ).all( ):
							st.warning( '⚠️ The k-Nearest Neighbors feature matrix '
							            'contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_knn ).all( ):
							st.warning( '⚠️ The k-Nearest Neighbors target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_knn ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if knn_neighbors < 1:
							st.warning( '⚠️ The number of neighbors must be greater '
							            'than zero.' )
							st.stop( )
						
						if knn_leaf_size < 1:
							st.warning( '⚠️ Leaf Size must be greater than zero.' )
							st.stop( )
						
						if knn_power < 1.0:
							st.warning( '⚠️ Minkowski Power must be at least 1.0.' )
							st.stop( )
						
						if knn_jobs < 1:
							st.warning( '⚠️ Parallel Jobs must be greater than zero.' )
							st.stop( )
						
						st.session_state[ 'regression_knn_neighbors' ] = int( knn_neighbors )
						st.session_state[ 'regression_knn_weights' ] = str( knn_weights )
						st.session_state[ 'regression_knn_algorithm' ] = str( knn_algorithm )
						st.session_state[ 'regression_knn_leaf_size' ] = int( knn_leaf_size )
						st.session_state[ 'regression_knn_power' ] = float( knn_power )
						st.session_state[ 'regression_knn_metric' ] = str( knn_metric )
						st.session_state[ 'regression_knn_jobs' ] = int( knn_jobs )
						st.session_state[ 'regression_knn_test_size' ] = float( knn_test_size )
						st.session_state[ 'regression_knn_random_state' ] = int( knn_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.NearestNeighbor( num=int( knn_neighbors ),
							weight=str( knn_weights ), algo=str( knn_algorithm ),
							leaf=int( knn_leaf_size ), power=float( knn_power ),
							metric=str( knn_metric ), metric_params=None, jobs=int( knn_jobs ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_knn, y_knn,
							size=float( knn_test_size ), random=int( knn_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						if knn_neighbors > len( X_train ):
							st.warning( '⚠️ Neighbors cannot exceed the number of '
							            'training observations.' )
							st.stop( )
						
						model.train( X_train, y_train )
						
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Neighbors', 'Weights', 'Algorithm', 'Metric',
								'Power' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( knn_neighbors ), str( knn_weights ),
								str( knn_algorithm ), str( knn_metric ), float( knn_power ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_knn_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'k-Nearest Neighbors training completed.' )
					
					except Exception as ex:
						st.error( f'k-Nearest Neighbors training failed: {ex}' )
			
			with st.expander( 'Support Vector Machine', expanded=False ):
				svr_defaults = { 'regression_svr_kernel': 'rbf', 'regression_svr_degree': 3,
					'regression_svr_gamma_mode': 'scale', 'regression_svr_gamma_value': 0.1,
					'regression_svr_coef0': 0.0, 'regression_svr_tol': 0.001,
					'regression_svr_c': 1.0, 'regression_svr_epsilon': 0.1,
					'regression_svr_shrinking': True, 'regression_svr_cache_size': 200.0,
					'regression_svr_verbose': False, 'regression_svr_max_iter': -1,
					'regression_svr_test_size': 0.20, 'regression_svr_random_state': 42 }
				
				for key, value in svr_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Support Vector Regression for continuous targets.' )
				
				svr_c1, svr_c2, svr_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with svr_c1:
					st.markdown( '###### Kernel Parameters' )
					
					svr_kernel_options = [ 'linear', 'poly', 'rbf', 'sigmoid' ]
					
					svr_kernel_value = st.session_state.get( 'regression_svr_kernel', 'rbf' )
					
					if svr_kernel_value not in svr_kernel_options:
						svr_kernel_value = 'rbf'
					
					svr_kernel = st.selectbox( 'Kernel', options=svr_kernel_options,
						index=svr_kernel_options.index( svr_kernel_value ),
						key='regression_svr_kernel_select' )
					
					svr_degree = int( st.number_input( 'Degree', min_value=1,
						value=int( st.session_state[ 'regression_svr_degree' ] ), step=1,
						key='regression_svr_degree_input' ) )
					
					svr_gamma_options = [ 'scale', 'auto', 'custom' ]
					
					svr_gamma_mode_value = st.session_state.get( 'regression_svr_gamma_mode',
						'scale' )
					
					if svr_gamma_mode_value not in svr_gamma_options:
						svr_gamma_mode_value = 'scale'
					
					svr_gamma_mode = st.selectbox( 'Gamma', options=svr_gamma_options,
						index=svr_gamma_options.index( svr_gamma_mode_value ),
						key='regression_svr_gamma_mode_select' )
					
					svr_gamma_value = float( st.number_input( 'Gamma Value', min_value=0.000001,
						value=float( st.session_state[ 'regression_svr_gamma_value' ] ),
						step=0.010000, format='%.6f', key='regression_svr_gamma_value_input' ) )
					
					svr_coef0 = float( st.number_input( 'Coef0',
						value=float( st.session_state[ 'regression_svr_coef0' ] ), step=0.100000,
						format='%.6f', key='regression_svr_coef0_input' ) )
				
				with svr_c2:
					st.markdown( '###### Regularization / Solver' )
					
					svr_c = float( st.number_input( 'C', min_value=0.000001,
						value=float( st.session_state[ 'regression_svr_c' ] ), step=0.100000,
						format='%.6f', key='regression_svr_c_input' ) )
					
					svr_epsilon = float( st.number_input( 'Epsilon', min_value=0.0,
						value=float( st.session_state[ 'regression_svr_epsilon' ] ), step=0.010000,
						format='%.6f', key='regression_svr_epsilon_input' ) )
					
					svr_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_svr_tol' ] ), step=0.000100,
						format='%.6f', key='regression_svr_tol_input' ) )
					
					svr_shrinking = st.checkbox( 'Shrinking Heuristic',
						value=bool( st.session_state[ 'regression_svr_shrinking' ] ),
						key='regression_svr_shrinking_check' )
					
					svr_cache_size = float( st.number_input( 'Cache Size (MB)', min_value=1.0,
						value=float( st.session_state[ 'regression_svr_cache_size' ] ), step=10.0,
						format='%.1f', key='regression_svr_cache_size_input' ) )
					
					svr_verbose = st.checkbox( 'Verbose',
						value=bool( st.session_state[ 'regression_svr_verbose' ] ),
						key='regression_svr_verbose_check' )
					
					svr_max_iter = int(
						st.number_input( 'Max Iterations (-1 = No Limit)', min_value=-1,
							value=int( st.session_state[ 'regression_svr_max_iter' ] ), step=1,
							key='regression_svr_max_iter_input' ) )
				
				with svr_c3:
					st.markdown( '###### Data Split' )
					
					svr_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_svr_test_size' ] * 100 ), step=5,
						key='regression_svr_test_size_slider' ) / 100.0
					
					svr_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_svr_random_state' ] ), step=1,
						key='regression_svr_random_state_input' ) )
					
					if svr_kernel != 'poly':
						st.caption( 'Degree is only used when Kernel = poly.' )
					
					if svr_kernel not in [ 'poly', 'sigmoid' ]:
						st.caption( 'Coef0 is mainly used by poly and sigmoid kernels.' )
					
					if svr_gamma_mode != 'custom':
						st.caption( 'Gamma Value is only used when Gamma = custom.' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Target: {target_name}' )
				
				svr_btn_1, svr_btn_2 = st.columns( 2 )
				
				with svr_btn_1:
					train_svr = st.button( '🚂 Train Support Vector', key='regression_svr_train',
						use_container_width=True )
				
				with svr_btn_2:
					reset_svr = st.button( '🔄 Reset Support Vector', key='regression_svr_reset',
						use_container_width=True )
				
				if reset_svr:
					for key, value in svr_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_svr_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_svr:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Support Vector Regression requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_svr = np.asarray( X, dtype=float )
						y_svr = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_svr.ndim != 2 or X_svr.shape[ 1 ] < 1:
							st.warning( '⚠️ Support Vector Regression requires at least '
							            'one numeric feature.' )
							st.stop( )
						
						if y_svr.ndim != 1:
							st.warning( '⚠️ The Support Vector Regression target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_svr ) != len( y_svr ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_svr ) < 2:
							st.warning( '⚠️ Support Vector Regression requires at least '
							            'two observations.' )
							st.stop( )
						
						if not np.isfinite( X_svr ).all( ):
							st.warning( '⚠️ The Support Vector Regression feature matrix '
							            'contains non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_svr ).all( ):
							st.warning( '⚠️ The Support Vector Regression target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_svr ) ) < 2:
							st.warning( '⚠️ The regression target must contain at least '
							            'two distinct values.' )
							st.stop( )
						
						if svr_c <= 0.0:
							st.warning( '⚠️ Support Vector C must be greater than zero.' )
							st.stop( )
						
						if svr_epsilon < 0.0:
							st.warning( '⚠️ Support Vector epsilon cannot be negative.' )
							st.stop( )
						
						if svr_tol < 0.0:
							st.warning( '⚠️ Support Vector tolerance cannot be negative.' )
							st.stop( )
						
						if svr_cache_size <= 0.0:
							st.warning( '⚠️ Support Vector cache size must be greater '
							            'than zero.' )
							st.stop( )
						
						if svr_max_iter == 0 or svr_max_iter < -1:
							st.warning( '⚠️ Max Iterations must be -1 or greater than zero.' )
							st.stop( )
						
						if svr_gamma_mode == 'custom' and svr_gamma_value <= 0.0:
							st.warning( '⚠️ Custom gamma must be greater than zero.' )
							st.stop( )
						
						effective_gamma = (
							float( svr_gamma_value ) if svr_gamma_mode == 'custom' else str(
								svr_gamma_mode ))
						
						st.session_state[ 'regression_svr_kernel' ] = str( svr_kernel )
						st.session_state[ 'regression_svr_degree' ] = int( svr_degree )
						st.session_state[ 'regression_svr_gamma_mode' ] = str( svr_gamma_mode )
						st.session_state[ 'regression_svr_gamma_value' ] = float( svr_gamma_value )
						st.session_state[ 'regression_svr_coef0' ] = float( svr_coef0 )
						st.session_state[ 'regression_svr_tol' ] = float( svr_tol )
						st.session_state[ 'regression_svr_c' ] = float( svr_c )
						st.session_state[ 'regression_svr_epsilon' ] = float( svr_epsilon )
						st.session_state[ 'regression_svr_shrinking' ] = bool( svr_shrinking )
						st.session_state[ 'regression_svr_cache_size' ] = float( svr_cache_size )
						st.session_state[ 'regression_svr_verbose' ] = bool( svr_verbose )
						st.session_state[ 'regression_svr_max_iter' ] = int( svr_max_iter )
						st.session_state[ 'regression_svr_test_size' ] = float( svr_test_size )
						st.session_state[ 'regression_svr_random_state' ] = int( svr_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.SupportVector( kernel=str( svr_kernel ),
							degree=int( svr_degree ), gamma=effective_gamma,
							coef0=float( svr_coef0 ), tol=float( svr_tol ), penalty=float( svr_c ),
							epsilon=float( svr_epsilon ), shrinking=bool( svr_shrinking ),
							cache=float( svr_cache_size ), verbose=bool( svr_verbose ),
							iters=int( svr_max_iter ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_svr, y_svr,
							size=float( svr_test_size ), random=int( svr_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Kernel', 'C', 'Epsilon', 'Gamma',
								'Maximum Iterations' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), str( svr_kernel ), float( svr_c ),
								float( svr_epsilon ), effective_gamma, int( svr_max_iter ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_svr_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Support Vector Regression training completed.' )
					
					except Exception as ex:
						st.error( f'Support Vector Regression training failed: {ex}' )

		# ------------------------------------------------------------------
		# Tree Models
		# ------------------------------------------------------------------
		with st.expander( 'Tree Models', expanded=False ):
			
			with st.expander( 'Extra Trees Regressor', expanded=False ):
				extra_defaults = { 'regression_extra_estimators': 100,
					'regression_extra_criterion': 'squared_error',
					'regression_extra_max_depth_mode': 'none',
					'regression_extra_max_depth_value': 10,
					'regression_extra_max_features_mode': 'all',
					'regression_extra_max_features_value': 1.0, 'regression_extra_bootstrap':
						False,
					'regression_extra_oob_score': False, 'regression_extra_warm_start': False,
					'regression_extra_jobs': 1, 'regression_extra_verbose': 0,
					'regression_extra_test_size': 0.20, 'regression_extra_random_state': 42 }
				
				for key, value in extra_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Highly randomized tree ensemble for continuous targets.' )
				
				extra_c1, extra_c2, extra_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with extra_c1:
					st.markdown( '###### Forest Parameters' )
					
					extra_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_extra_estimators' ] ), step=1,
						key='regression_extra_estimators_input' ) )
					
					extra_criterion_options = [ 'squared_error', 'absolute_error', 'friedman_mse',
						'poisson' ]
					
					extra_criterion_value = st.session_state.get( 'regression_extra_criterion',
						'squared_error' )
					
					if extra_criterion_value not in extra_criterion_options:
						extra_criterion_value = 'squared_error'
					
					extra_criterion = st.selectbox( 'Criterion', options=extra_criterion_options,
						index=extra_criterion_options.index( extra_criterion_value ),
						key='regression_extra_criterion_select' )
					
					extra_depth_options = [ 'none', 'custom' ]
					
					extra_depth_value = st.session_state.get( 'regression_extra_max_depth_mode',
						'none' )
					
					if extra_depth_value not in extra_depth_options:
						extra_depth_value = 'none'
					
					extra_max_depth_mode = st.selectbox( 'Max Depth', options=extra_depth_options,
						index=extra_depth_options.index( extra_depth_value ),
						key='regression_extra_max_depth_mode_select' )
					
					extra_max_depth_value = int( st.number_input( 'Max Depth Value', min_value=1,
						value=int( st.session_state[ 'regression_extra_max_depth_value' ] ),
						step=1,
						key='regression_extra_max_depth_value_input' ) )
				
				with extra_c2:
					st.markdown( '###### Feature / Run Settings' )
					
					extra_feature_options = [ 'all', 'sqrt', 'log2', 'fraction' ]
					
					extra_feature_value = st.session_state.get(
						'regression_extra_max_features_mode', 'all' )
					
					if extra_feature_value not in extra_feature_options:
						extra_feature_value = 'all'
					
					extra_max_features_mode = st.selectbox( 'Max Features',
						options=extra_feature_options,
						index=extra_feature_options.index( extra_feature_value ),
						key='regression_extra_max_features_mode_select' )
					
					extra_max_features_value = float(
						st.slider( 'Max Features Fraction', min_value=0.10, max_value=1.00,
							value=float(
								st.session_state[ 'regression_extra_max_features_value' ] ),
							step=0.05, key='regression_extra_max_features_value_slider' ) )
					
					extra_bootstrap = st.checkbox( 'Bootstrap Samples',
						value=bool( st.session_state[ 'regression_extra_bootstrap' ] ),
						key='regression_extra_bootstrap_check' )
					
					extra_oob_score = st.checkbox( 'Out-of-Bag Score',
						value=bool( st.session_state[ 'regression_extra_oob_score' ] ),
						key='regression_extra_oob_score_check' )
					
					extra_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_extra_warm_start' ] ),
						key='regression_extra_warm_start_check' )
					
					extra_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_extra_jobs' ] ), step=1,
						key='regression_extra_jobs_input' ) )
					
					extra_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_extra_verbose' ] ), step=1,
						key='regression_extra_verbose_input' ) )
				
				with extra_c3:
					st.markdown( '###### Data Split' )
					
					extra_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_extra_test_size' ] * 100 ),
						step=5,
						key='regression_extra_test_size_slider' ) / 100.0
					
					extra_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_extra_random_state' ] ), step=1,
						key='regression_extra_random_state_input' ) )
					
					if extra_oob_score and not extra_bootstrap:
						st.info( 'Out-of-bag scoring requires Bootstrap Samples.' )
					
					if extra_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when '
						            'Max Features = fraction.' )
					
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Target: {target_name}' )
				
				extra_btn_1, extra_btn_2 = st.columns( 2 )
				
				with extra_btn_1:
					train_extra = st.button( '🚂 Train Extra Trees Regressor',
						key='regression_extra_train', use_container_width=True )
				
				with extra_btn_2:
					reset_extra = st.button( '🔄 Reset Extra Trees Regressor',
						key='regression_extra_reset', use_container_width=True )
				
				if reset_extra:
					for key, value in extra_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_extra_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_extra:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Extra Trees Regressor requires prepared '
							            'feature and target arrays.' )
							st.stop( )
						
						X_extra = np.asarray( X, dtype=float )
						
						y_extra = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_extra.ndim != 2 or X_extra.shape[ 1 ] < 1:
							st.warning( '⚠️ Extra Trees Regressor requires at least '
							            'one numeric feature.' )
							st.stop( )
						
						if y_extra.ndim != 1:
							st.warning( '⚠️ The Extra Trees target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_extra ) != len( y_extra ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_extra ) < 2:
							st.warning( '⚠️ Extra Trees Regressor requires at least '
							            'two observations.' )
							st.stop( )
						
						if not np.isfinite( X_extra ).all( ):
							st.warning( '⚠️ The Extra Trees feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_extra ).all( ):
							st.warning( '⚠️ The Extra Trees target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_extra ) ) < 2:
							st.warning( '⚠️ The regression target must contain at '
							            'least two distinct values.' )
							st.stop( )
						
						if extra_estimators < 1:
							st.warning( '⚠️ Estimators must be greater than zero.' )
							st.stop( )
						
						if extra_jobs < 1:
							st.warning( '⚠️ Parallel Jobs must be greater than zero.' )
							st.stop( )
						
						if (extra_max_features_value <= 0.0 or extra_max_features_value > 1.0):
							st.warning( '⚠️ Max Features Fraction must be greater '
							            'than zero and no greater than one.' )
							st.stop( )
						
						if extra_criterion == 'poisson' and np.any( y_extra < 0.0 ):
							st.warning( '⚠️ Poisson criterion requires a '
							            'non-negative target.' )
							st.stop( )
						
						if extra_max_depth_mode == 'none':
							effective_depth = None
						else:
							effective_depth = int( extra_max_depth_value )
						
						if extra_max_features_mode == 'all':
							effective_features = 1.0
						elif extra_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif extra_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( extra_max_features_value )
						
						effective_oob = (bool( extra_oob_score ) and bool( extra_bootstrap ))
						
						st.session_state[ 'regression_extra_estimators' ] = int( extra_estimators )
						st.session_state[ 'regression_extra_criterion' ] = str( extra_criterion )
						st.session_state[ 'regression_extra_max_depth_mode' ] = str(
							extra_max_depth_mode )
						st.session_state[ 'regression_extra_max_depth_value' ] = int(
							extra_max_depth_value )
						st.session_state[ 'regression_extra_max_features_mode' ] = str(
							extra_max_features_mode )
						st.session_state[ 'regression_extra_max_features_value' ] = float(
							extra_max_features_value )
						st.session_state[ 'regression_extra_bootstrap' ] = bool( extra_bootstrap )
						st.session_state[ 'regression_extra_oob_score' ] = bool( extra_oob_score )
						st.session_state[ 'regression_extra_warm_start' ] = bool(
							extra_warm_start )
						st.session_state[ 'regression_extra_jobs' ] = int( extra_jobs )
						st.session_state[ 'regression_extra_verbose' ] = int( extra_verbose )
						st.session_state[ 'regression_extra_test_size' ] = float( extra_test_size )
						st.session_state[ 'regression_extra_random_state' ] = int(
							extra_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.ExtraTreesModel(
							estimators=int( extra_estimators ), criterion=str( extra_criterion ),
							depth=effective_depth, features=effective_features,
							bootstrap=bool( extra_bootstrap ), oob_score=bool( effective_oob ),
							jobs=int( extra_jobs ), rando=int( extra_random_state ),
							verbose=int( extra_verbose ), warm=bool( extra_warm_start ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_extra, y_extra,
							size=float( extra_test_size ), random=int( extra_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce '
							            'valid training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Estimators', 'Criterion', 'Max Depth',
								'Max Features', 'Bootstrap' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( extra_estimators ),
								str( extra_criterion ),
								('None' if effective_depth is None else int( effective_depth )),
								str( effective_features ), bool( extra_bootstrap ) ] } )
						
						if (effective_oob and hasattr( model.model, 'oob_score_' )):
							df_oob = pd.DataFrame( { 'Metric': [ 'OOB Score' ],
								'Value': [ float( model.model.oob_score_ ) ] } )
							
							df_metadata = pd.concat( [ df_metadata, df_oob ], ignore_index=True )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_extra_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Extra Trees Regressor training completed.' )
					
					except Exception as ex:
						st.error( f'Extra Trees Regressor training failed: {ex}' )
			
			with st.expander( 'Random Forest', expanded=False ):
				rf_defaults = { 'regression_rf_estimators': 100,
					'regression_rf_criterion': 'squared_error',
					'regression_rf_max_depth_mode': 'none', 'regression_rf_max_depth_value': 10,
					'regression_rf_min_samples_split': 2, 'regression_rf_min_samples_leaf': 1,
					'regression_rf_min_weight_fraction_leaf': 0.0,
					'regression_rf_max_features_mode': 'all',
					'regression_rf_max_features_value': 1.0,
					'regression_rf_max_leaf_nodes_mode': 'none',
					'regression_rf_max_leaf_nodes_value': 31,
					'regression_rf_min_impurity_decrease': 0.0, 'regression_rf_bootstrap': True,
					'regression_rf_oob_score': False, 'regression_rf_jobs': 1,
					'regression_rf_verbose': 0, 'regression_rf_warm_start': False,
					'regression_rf_ccp_alpha': 0.0, 'regression_rf_max_samples_mode': 'all',
					'regression_rf_max_samples_value': 1.0, 'regression_rf_test_size': 0.20,
					'regression_rf_random_state': 42 }
				
				for key, value in rf_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Bootstrap random forest for continuous targets.' )
				
				rf_c1, rf_c2, rf_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with rf_c1:
					st.markdown( '###### Forest Parameters' )
					
					rf_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_rf_estimators' ] ), step=1,
						key='regression_rf_estimators_input' ) )
					
					rf_criterion_options = [ 'squared_error', 'absolute_error', 'friedman_mse',
						'poisson' ]
					
					rf_criterion_value = st.session_state.get( 'regression_rf_criterion',
						'squared_error' )
					
					if rf_criterion_value not in rf_criterion_options:
						rf_criterion_value = 'squared_error'
					
					rf_criterion = st.selectbox( 'Criterion', options=rf_criterion_options,
						index=rf_criterion_options.index( rf_criterion_value ),
						key='regression_rf_criterion_select' )
					
					rf_depth_options = [ 'none', 'custom' ]
					
					rf_depth_value = st.session_state.get( 'regression_rf_max_depth_mode', 'none' )
					
					if rf_depth_value not in rf_depth_options:
						rf_depth_value = 'none'
					
					rf_max_depth_mode = st.selectbox( 'Max Depth', options=rf_depth_options,
						index=rf_depth_options.index( rf_depth_value ),
						key='regression_rf_max_depth_mode_select' )
					
					rf_max_depth_value = int( st.number_input( 'Max Depth Value', min_value=1,
						value=int( st.session_state[ 'regression_rf_max_depth_value' ] ), step=1,
						key='regression_rf_max_depth_value_input' ) )
					
					rf_min_samples_split = int( st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'regression_rf_min_samples_split' ] ), step=1,
						key='regression_rf_min_samples_split_input' ) )
					
					rf_min_samples_leaf = int( st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'regression_rf_min_samples_leaf' ] ), step=1,
						key='regression_rf_min_samples_leaf_input' ) )
				
				with rf_c2:
					st.markdown( '###### Node / Feature Controls' )
					
					rf_min_weight_fraction_leaf = float(
						st.number_input( 'Min Weight Fraction Leaf', min_value=0.0, max_value=0.5,
							value=float(
								st.session_state[ 'regression_rf_min_weight_fraction_leaf' ] ),
							step=0.010000, format='%.6f',
							key='regression_rf_min_weight_fraction_leaf_input' ) )
					
					rf_feature_options = [ 'all', 'sqrt', 'log2', 'fraction' ]
					
					rf_feature_value = st.session_state.get( 'regression_rf_max_features_mode',
						'all' )
					
					if rf_feature_value not in rf_feature_options:
						rf_feature_value = 'all'
					
					rf_max_features_mode = st.selectbox( 'Max Features',
						options=rf_feature_options,
						index=rf_feature_options.index( rf_feature_value ),
						key='regression_rf_max_features_mode_select' )
					
					rf_max_features_value = float(
						st.slider( 'Max Features Fraction', min_value=0.10, max_value=1.00,
							value=float( st.session_state[ 'regression_rf_max_features_value' ] ),
							step=0.05, key='regression_rf_max_features_value_slider' ) )
					
					rf_leaf_node_options = [ 'none', 'custom' ]
					
					rf_leaf_node_value = st.session_state.get( 'regression_rf_max_leaf_nodes_mode',
						'none' )
					
					if rf_leaf_node_value not in rf_leaf_node_options:
						rf_leaf_node_value = 'none'
					
					rf_max_leaf_nodes_mode = st.selectbox( 'Max Leaf Nodes',
						options=rf_leaf_node_options,
						index=rf_leaf_node_options.index( rf_leaf_node_value ),
						key='regression_rf_max_leaf_nodes_mode_select' )
					
					rf_max_leaf_nodes_value = int(
						st.number_input( 'Max Leaf Nodes Value', min_value=2,
							value=int( st.session_state[ 'regression_rf_max_leaf_nodes_value' ] ),
							step=1, key='regression_rf_max_leaf_nodes_value_input' ) )
					
					rf_min_impurity_decrease = float(
						st.number_input( 'Min Impurity Decrease', min_value=0.0, value=float(
							st.session_state[ 'regression_rf_min_impurity_decrease' ] ),
							step=0.000100, format='%.6f',
							key='regression_rf_min_impurity_decrease_input' ) )
				
				with rf_c3:
					st.markdown( '###### Sampling / Run Configuration' )
					
					rf_bootstrap = st.checkbox( 'Bootstrap Samples',
						value=bool( st.session_state[ 'regression_rf_bootstrap' ] ),
						key='regression_rf_bootstrap_check' )
					
					rf_oob_score = st.checkbox( 'Out-of-Bag Score',
						value=bool( st.session_state[ 'regression_rf_oob_score' ] ),
						key='regression_rf_oob_score_check' )
					
					rf_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_rf_jobs' ] ), step=1,
						key='regression_rf_jobs_input' ) )
					
					rf_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_rf_verbose' ] ), step=1,
						key='regression_rf_verbose_input' ) )
					
					rf_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_rf_warm_start' ] ),
						key='regression_rf_warm_start_check' )
					
					rf_ccp_alpha = float( st.number_input( 'CCP Alpha', min_value=0.0,
						value=float( st.session_state[ 'regression_rf_ccp_alpha' ] ),
						step=0.000100,
						format='%.6f', key='regression_rf_ccp_alpha_input' ) )
					
					rf_sample_options = [ 'all', 'fraction' ]
					
					rf_sample_value = st.session_state.get( 'regression_rf_max_samples_mode',
						'all' )
					
					if rf_sample_value not in rf_sample_options:
						rf_sample_value = 'all'
					
					rf_max_samples_mode = st.selectbox( 'Max Samples', options=rf_sample_options,
						index=rf_sample_options.index( rf_sample_value ),
						key='regression_rf_max_samples_mode_select' )
					
					rf_max_samples_value = float(
						st.slider( 'Max Samples Fraction', min_value=0.10, max_value=1.00,
							value=float( st.session_state[ 'regression_rf_max_samples_value' ] ),
							step=0.05, key='regression_rf_max_samples_value_slider' ) )
					
					rf_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_rf_test_size' ] * 100 ), step=5,
						key='regression_rf_test_size_slider' ) / 100.0
					
					rf_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_rf_random_state' ] ), step=1,
						key='regression_rf_random_state_input' ) )
					
					if rf_oob_score and not rf_bootstrap:
						st.info( 'Out-of-bag scoring requires Bootstrap Samples.' )
					
					if rf_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when '
						            'Max Features = fraction.' )
					
					if rf_max_samples_mode != 'fraction':
						st.caption( 'Max Samples Fraction is only used when '
						            'Max Samples = fraction.' )
				
				rf_btn_1, rf_btn_2 = st.columns( 2 )
				
				with rf_btn_1:
					train_rf = st.button( '🚂 Train Random Forest', key='regression_rf_train',
						use_container_width=True )
				
				with rf_btn_2:
					reset_rf = st.button( '🔄 Reset Random Forest', key='regression_rf_reset',
						use_container_width=True )
				
				if reset_rf:
					for key, value in rf_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_rf_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_rf:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Random Forest requires prepared feature and '
							            'target arrays.' )
							st.stop( )
						
						X_rf = np.asarray( X, dtype=float )
						y_rf = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_rf.ndim != 2 or X_rf.shape[ 1 ] < 1:
							st.warning( '⚠️ Random Forest requires at least one numeric feature.' )
							st.stop( )
						
						if len( X_rf ) != len( y_rf ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_rf ) < 2:
							st.warning( '⚠️ Random Forest requires at least two observations.' )
							st.stop( )
						
						if not np.isfinite( X_rf ).all( ):
							st.warning( '⚠️ The Random Forest feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_rf ).all( ):
							st.warning( '⚠️ The Random Forest target contains non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_rf ) ) < 2:
							st.warning( '⚠️ The target must contain at least two distinct '
							            'values.' )
							st.stop( )
						
						if rf_criterion == 'poisson' and np.any( y_rf < 0.0 ):
							st.warning( '⚠️ Poisson criterion requires a non-negative target.' )
							st.stop( )
						
						effective_depth = (
							None if rf_max_depth_mode == 'none' else int( rf_max_depth_value ))
						
						if rf_max_features_mode == 'all':
							effective_features = 1.0
						elif rf_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif rf_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( rf_max_features_value )
						
						effective_leaf_nodes = (None if rf_max_leaf_nodes_mode == 'none' else int(
							rf_max_leaf_nodes_value ))
						
						effective_samples = (
							None if rf_max_samples_mode == 'all' else float(
								rf_max_samples_value ))
						
						effective_oob = bool( rf_oob_score ) and bool( rf_bootstrap )
						
						if not rf_bootstrap:
							effective_samples = None
						
						st.session_state[ 'regression_rf_estimators' ] = int( rf_estimators )
						st.session_state[ 'regression_rf_criterion' ] = str( rf_criterion )
						st.session_state[ 'regression_rf_max_depth_mode' ] = str(
							rf_max_depth_mode )
						st.session_state[ 'regression_rf_max_depth_value' ] = int(
							rf_max_depth_value )
						st.session_state[ 'regression_rf_min_samples_split' ] = int(
							rf_min_samples_split )
						st.session_state[ 'regression_rf_min_samples_leaf' ] = int(
							rf_min_samples_leaf )
						st.session_state[ 'regression_rf_min_weight_fraction_leaf' ] = float(
							rf_min_weight_fraction_leaf )
						st.session_state[ 'regression_rf_max_features_mode' ] = str(
							rf_max_features_mode )
						st.session_state[ 'regression_rf_max_features_value' ] = float(
							rf_max_features_value )
						st.session_state[ 'regression_rf_max_leaf_nodes_mode' ] = str(
							rf_max_leaf_nodes_mode )
						st.session_state[ 'regression_rf_max_leaf_nodes_value' ] = int(
							rf_max_leaf_nodes_value )
						st.session_state[ 'regression_rf_min_impurity_decrease' ] = float(
							rf_min_impurity_decrease )
						st.session_state[ 'regression_rf_bootstrap' ] = bool( rf_bootstrap )
						st.session_state[ 'regression_rf_oob_score' ] = bool( rf_oob_score )
						st.session_state[ 'regression_rf_jobs' ] = int( rf_jobs )
						st.session_state[ 'regression_rf_verbose' ] = int( rf_verbose )
						st.session_state[ 'regression_rf_warm_start' ] = bool( rf_warm_start )
						st.session_state[ 'regression_rf_ccp_alpha' ] = float( rf_ccp_alpha )
						st.session_state[ 'regression_rf_max_samples_mode' ] = str(
							rf_max_samples_mode )
						st.session_state[ 'regression_rf_max_samples_value' ] = float(
							rf_max_samples_value )
						st.session_state[ 'regression_rf_test_size' ] = float( rf_test_size )
						st.session_state[ 'regression_rf_random_state' ] = int( rf_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.RandomForest( estimators=int( rf_estimators ),
							criterion=str( rf_criterion ), depth=effective_depth,
							split=int( rf_min_samples_split ), leaf=int( rf_min_samples_leaf ),
							weight_fraction=float( rf_min_weight_fraction_leaf ),
							features=effective_features, leaf_nodes=effective_leaf_nodes,
							impurity=float( rf_min_impurity_decrease ),
							bootstrap=bool( rf_bootstrap ), oob_score=bool( effective_oob ),
							jobs=int( rf_jobs ), rando=int( rf_random_state ),
							verbose=int( rf_verbose ), warm=bool( rf_warm_start ),
							ccp_alpha=float( rf_ccp_alpha ), samples=effective_samples,
							monotonic=None )
						
						X_train, X_test, y_train, y_test = model.split_data( X_rf, y_rf,
							size=float( rf_test_size ), random=int( rf_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce valid '
							            'training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Estimators', 'Criterion', 'Max Depth',
								'Max Features', 'Bootstrap' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( rf_estimators ), str( rf_criterion ),
								('None' if effective_depth is None else int( effective_depth )),
								str( effective_features ), bool( rf_bootstrap ) ] } )
						
						if effective_oob and hasattr( model.model, 'oob_score_' ):
							df_oob = pd.DataFrame( { 'Metric': [ 'OOB Score' ],
								'Value': [ float( model.model.oob_score_ ) ] } )
							
							df_metadata = pd.concat( [ df_metadata, df_oob ], ignore_index=True )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_rf_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Random Forest training completed.' )
					
					except Exception as ex:
						st.error( f'Random Forest training failed: {ex}' )

		# ------------------------------------------------------------------
		# Ensemble Models
		# ------------------------------------------------------------------
		with st.expander( 'Ensemble Models', expanded=False ):
			
			with st.expander( 'Adaptive Boosting', expanded=False ):
				ada_defaults = { 'regression_ada_estimators': 50,
					'regression_ada_learning_rate': 1.0, 'regression_ada_loss': 'linear',
					'regression_ada_test_size': 0.20, 'regression_ada_random_state': 42 }
				
				for key, value in ada_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'AdaBoost ensemble for continuous targets.' )
				
				ada_c1, ada_c2, ada_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with ada_c1:
					st.markdown( '###### Ensemble Parameters' )
					
					ada_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_ada_estimators' ] ), step=1,
						key='regression_ada_estimators_input' ) )
					
					ada_learning_rate = float( st.number_input( 'Learning Rate',
						min_value=0.000001,
						value=float( st.session_state[ 'regression_ada_learning_rate' ] ),
						step=0.100000, format='%.6f', key='regression_ada_learning_rate_input' ) )
					
					ada_loss_options = [ 'linear', 'square', 'exponential' ]
					
					ada_loss_value = st.session_state.get( 'regression_ada_loss', 'linear' )
					
					if ada_loss_value not in ada_loss_options:
						ada_loss_value = 'linear'
					
					ada_loss = st.selectbox( 'Loss', options=ada_loss_options,
						index=ada_loss_options.index( ada_loss_value ),
						key='regression_ada_loss_select' )
				
				with ada_c2:
					st.markdown( '###### Data Split' )
					
					ada_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_ada_test_size' ] * 100 ), step=5,
						key='regression_ada_test_size_slider' ) / 100.0
					
					ada_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_ada_random_state' ] ), step=1,
						key='regression_ada_random_state_input' ) )
				
				with ada_c3:
					st.markdown( '###### Context' )
					st.caption( f'Rows: {len( df_model ):,} | '
					            f'Features: {len( active_features ):,} | '
					            f'Target: {target_name}' )
				
				ada_btn_1, ada_btn_2 = st.columns( 2 )
				
				with ada_btn_1:
					train_ada = st.button( '🚂 Train Adaptive Boosting',
						key='regression_ada_train',
						use_container_width=True )
				
				with ada_btn_2:
					reset_ada = st.button( '🔄 Reset Adaptive Boosting',
						key='regression_ada_reset',
						use_container_width=True )
				
				if reset_ada:
					for key, value in ada_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_ada_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_ada:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Adaptive Boosting requires prepared feature '
							            'and target arrays.' )
							st.stop( )
						
						X_ada = np.asarray( X, dtype=float )
						y_ada = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_ada.ndim != 2 or X_ada.shape[ 1 ] < 1:
							st.warning( '⚠️ Adaptive Boosting requires at least one '
							            'numeric feature.' )
							st.stop( )
						
						if y_ada.ndim != 1:
							st.warning( '⚠️ The Adaptive Boosting target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_ada ) != len( y_ada ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_ada ) < 2:
							st.warning( '⚠️ Adaptive Boosting requires at least two '
							            'observations.' )
							st.stop( )
						
						if not np.isfinite( X_ada ).all( ):
							st.warning( '⚠️ The Adaptive Boosting feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_ada ).all( ):
							st.warning( '⚠️ The Adaptive Boosting target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_ada ) ) < 2:
							st.warning( '⚠️ The regression target must contain at least '
							            'two distinct values.' )
							st.stop( )
						
						if ada_estimators < 1:
							st.warning( '⚠️ Estimators must be greater than zero.' )
							st.stop( )
						
						if ada_learning_rate <= 0.0:
							st.warning( '⚠️ Learning Rate must be greater than zero.' )
							st.stop( )
						
						st.session_state[ 'regression_ada_estimators' ] = int( ada_estimators )
						st.session_state[ 'regression_ada_learning_rate' ] = float(
							ada_learning_rate )
						st.session_state[ 'regression_ada_loss' ] = str( ada_loss )
						st.session_state[ 'regression_ada_test_size' ] = float( ada_test_size )
						st.session_state[ 'regression_ada_random_state' ] = int( ada_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.AdaptiveBoost( estimators=int( ada_estimators ),
							rate=float( ada_learning_rate ), loss=str( ada_loss ),
							rando=int( ada_random_state ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_ada, y_ada,
							size=float( ada_test_size ), random=int( ada_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce valid '
							            'training and testing partitions.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Estimators', 'Learning Rate', 'Loss' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( ada_estimators ),
								float( ada_learning_rate ), str( ada_loss ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_ada_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Adaptive Boosting training completed.' )
					
					except Exception as ex:
						st.error( f'Adaptive Boosting training failed: {ex}' )
			
			with st.expander( 'Gradient Boosting', expanded=False ):
				gb_defaults = { 'regression_gb_loss': 'squared_error',
					'regression_gb_learning_rate': 0.100000, 'regression_gb_estimators': 100,
					'regression_gb_subsample': 1.0, 'regression_gb_criterion': 'friedman_mse',
					'regression_gb_min_samples_split': 2, 'regression_gb_min_samples_leaf': 1,
					'regression_gb_min_weight_fraction_leaf': 0.0,
					'regression_gb_max_depth_mode': 'custom', 'regression_gb_max_depth_value': 3,
					'regression_gb_min_impurity_decrease': 0.0,
					'regression_gb_max_features_mode': 'none',
					'regression_gb_max_features_value': 1.0, 'regression_gb_alpha': 0.9,
					'regression_gb_verbose': 0, 'regression_gb_max_leaf_nodes_mode': 'none',
					'regression_gb_max_leaf_nodes_value': 31, 'regression_gb_warm_start': False,
					'regression_gb_validation_fraction': 0.1,
					'regression_gb_n_iter_no_change_mode': 'none',
					'regression_gb_n_iter_no_change_value': 5, 'regression_gb_tol': 0.000100,
					'regression_gb_ccp_alpha': 0.0, 'regression_gb_test_size': 0.20,
					'regression_gb_random_state': 42 }
				
				for key, value in gb_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Sequential tree boosting for continuous targets.' )
				
				gb_c1, gb_c2, gb_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				
				with gb_c1:
					st.markdown( '###### Boosting Parameters' )
					
					gb_loss_options = [ 'squared_error', 'absolute_error', 'huber', 'quantile' ]
					
					gb_loss_value = st.session_state.get( 'regression_gb_loss', 'squared_error' )
					
					if gb_loss_value not in gb_loss_options:
						gb_loss_value = 'squared_error'
					
					gb_loss = st.selectbox( 'Loss', options=gb_loss_options,
						index=gb_loss_options.index( gb_loss_value ),
						key='regression_gb_loss_select' )
					
					gb_learning_rate = float( st.number_input( 'Learning Rate', min_value=0.000001,
						value=float( st.session_state[ 'regression_gb_learning_rate' ] ),
						step=0.010000, format='%.6f', key='regression_gb_learning_rate_input' ) )
					
					gb_estimators = int( st.number_input( 'Estimators', min_value=1,
						value=int( st.session_state[ 'regression_gb_estimators' ] ), step=1,
						key='regression_gb_estimators_input' ) )
					
					gb_subsample = float( st.slider( 'Subsample', min_value=0.10, max_value=1.00,
						value=float( st.session_state[ 'regression_gb_subsample' ] ), step=0.05,
						key='regression_gb_subsample_slider' ) )
					
					gb_alpha = float( st.slider( 'Alpha', min_value=0.01, max_value=0.99,
						value=float( st.session_state[ 'regression_gb_alpha' ] ), step=0.01,
						key='regression_gb_alpha_slider' ) )
				
				with gb_c2:
					st.markdown( '###### Tree Parameters' )
					
					gb_criterion_options = [ 'friedman_mse', 'squared_error' ]
					
					gb_criterion_value = st.session_state.get( 'regression_gb_criterion',
						'friedman_mse' )
					
					if gb_criterion_value not in gb_criterion_options:
						gb_criterion_value = 'friedman_mse'
					
					gb_criterion = st.selectbox( 'Criterion', options=gb_criterion_options,
						index=gb_criterion_options.index( gb_criterion_value ),
						key='regression_gb_criterion_select' )
					
					gb_min_samples_split = int( st.number_input( 'Min Samples Split', min_value=2,
						value=int( st.session_state[ 'regression_gb_min_samples_split' ] ), step=1,
						key='regression_gb_min_samples_split_input' ) )
					
					gb_min_samples_leaf = int( st.number_input( 'Min Samples Leaf', min_value=1,
						value=int( st.session_state[ 'regression_gb_min_samples_leaf' ] ), step=1,
						key='regression_gb_min_samples_leaf_input' ) )
					
					gb_min_weight_fraction_leaf = float(
						st.number_input( 'Min Weight Fraction Leaf', min_value=0.0, max_value=0.5,
							value=float(
								st.session_state[ 'regression_gb_min_weight_fraction_leaf' ] ),
							step=0.010000, format='%.6f',
							key='regression_gb_min_weight_fraction_leaf_input' ) )
					
					gb_depth_options = [ 'none', 'custom' ]
					
					gb_depth_value = st.session_state.get( 'regression_gb_max_depth_mode',
						'custom' )
					
					if gb_depth_value not in gb_depth_options:
						gb_depth_value = 'custom'
					
					gb_max_depth_mode = st.selectbox( 'Max Depth', options=gb_depth_options,
						index=gb_depth_options.index( gb_depth_value ),
						key='regression_gb_max_depth_mode_select' )
					
					gb_max_depth_value = int( st.number_input( 'Max Depth Value', min_value=1,
						value=int( st.session_state[ 'regression_gb_max_depth_value' ] ), step=1,
						key='regression_gb_max_depth_value_input' ) )
					
					gb_min_impurity_decrease = float(
						st.number_input( 'Min Impurity Decrease', min_value=0.0, value=float(
							st.session_state[ 'regression_gb_min_impurity_decrease' ] ),
							step=0.000100, format='%.6f',
							key='regression_gb_min_impurity_decrease_input' ) )
				
				with gb_c3:
					st.markdown( '###### Feature / Run Controls' )
					
					gb_feature_options = [ 'none', 'sqrt', 'log2', 'fraction' ]
					
					gb_feature_value = st.session_state.get( 'regression_gb_max_features_mode',
						'none' )
					
					if gb_feature_value not in gb_feature_options:
						gb_feature_value = 'none'
					
					gb_max_features_mode = st.selectbox( 'Max Features',
						options=gb_feature_options,
						index=gb_feature_options.index( gb_feature_value ),
						key='regression_gb_max_features_mode_select' )
					
					gb_max_features_value = float(
						st.slider( 'Max Features Fraction', min_value=0.10, max_value=1.00,
							value=float( st.session_state[ 'regression_gb_max_features_value' ] ),
							step=0.05, key='regression_gb_max_features_value_slider' ) )
					
					gb_leaf_node_options = [ 'none', 'custom' ]
					
					gb_leaf_node_value = st.session_state.get( 'regression_gb_max_leaf_nodes_mode',
						'none' )
					
					if gb_leaf_node_value not in gb_leaf_node_options:
						gb_leaf_node_value = 'none'
					
					gb_max_leaf_nodes_mode = st.selectbox( 'Max Leaf Nodes',
						options=gb_leaf_node_options,
						index=gb_leaf_node_options.index( gb_leaf_node_value ),
						key='regression_gb_max_leaf_nodes_mode_select' )
					
					gb_max_leaf_nodes_value = int(
						st.number_input( 'Max Leaf Nodes Value', min_value=2,
							value=int( st.session_state[ 'regression_gb_max_leaf_nodes_value' ] ),
							step=1, key='regression_gb_max_leaf_nodes_value_input' ) )
					
					gb_warm_start = st.checkbox( 'Warm Start',
						value=bool( st.session_state[ 'regression_gb_warm_start' ] ),
						key='regression_gb_warm_start_check' )
					
					gb_validation_fraction = float(
						st.slider( 'Validation Fraction', min_value=0.05, max_value=0.40,
							value=float( st.session_state[ 'regression_gb_validation_fraction' ] ),
							step=0.05, key='regression_gb_validation_fraction_slider' ) )
					
					gb_no_change_options = [ 'none', 'custom' ]
					
					gb_no_change_value = st.session_state.get(
						'regression_gb_n_iter_no_change_mode', 'none' )
					
					if gb_no_change_value not in gb_no_change_options:
						gb_no_change_value = 'none'
					
					gb_n_iter_no_change_mode = st.selectbox( 'N Iter No Change',
						options=gb_no_change_options,
						index=gb_no_change_options.index( gb_no_change_value ),
						key='regression_gb_n_iter_no_change_mode_select' )
					
					gb_n_iter_no_change_value = int(
						st.number_input( 'N Iter No Change Value', min_value=1,
							value=int( st.session_state[ 'regression_gb_n_iter_no_change_value'
							] ),
							step=1, key='regression_gb_n_iter_no_change_value_input' ) )
					
					gb_tol = float( st.number_input( 'Tolerance', min_value=0.0,
						value=float( st.session_state[ 'regression_gb_tol' ] ), step=0.000100,
						format='%.6f', key='regression_gb_tol_input' ) )
					
					gb_ccp_alpha = float( st.number_input( 'CCP Alpha', min_value=0.0,
						value=float( st.session_state[ 'regression_gb_ccp_alpha' ] ),
						step=0.000100,
						format='%.6f', key='regression_gb_ccp_alpha_input' ) )
					
					gb_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_gb_verbose' ] ), step=1,
						key='regression_gb_verbose_input' ) )
					
					gb_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=40,
						value=int( st.session_state[ 'regression_gb_test_size' ] * 100 ), step=5,
						key='regression_gb_test_size_slider' ) / 100.0
					
					gb_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_gb_random_state' ] ), step=1,
						key='regression_gb_random_state_input' ) )
					
					if gb_loss not in [ 'huber', 'quantile' ]:
						st.caption( 'Alpha is only used by huber and quantile losses.' )
					
					if gb_max_features_mode != 'fraction':
						st.caption( 'Max Features Fraction is only used when '
						            'Max Features = fraction.' )
				
				gb_btn_1, gb_btn_2 = st.columns( 2 )
				
				with gb_btn_1:
					train_gb = st.button( '🚂 Train Gradient Boosting', key='regression_gb_train',
						use_container_width=True )
				
				with gb_btn_2:
					reset_gb = st.button( '🔄 Reset Gradient Boosting', key='regression_gb_reset',
						use_container_width=True )
				
				if reset_gb:
					for key, value in gb_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'elapsed_seconds' ] = 0.0
					st.session_state[ 'regression_gb_elapsed_seconds' ] = 0.0
					st.session_state[ 'model' ]=None
					st.session_state[ 'X_train' ]=None
					st.session_state[ 'X_test' ]=None
					st.session_state[ 'y_train' ]=None
					st.session_state[ 'y_test' ]=None
					st.session_state[ 'y_prediction' ]=None
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_regression_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.rerun( )
				
				if train_gb:
					try:
						if X is None or y is None:
							st.warning( '⚠️ Gradient Boosting requires prepared feature '
							            'and target arrays.' )
							st.stop( )
						
						X_gb = np.asarray( X, dtype=float )
						y_gb = np.asarray( y, dtype=float ).reshape( -1 )
						
						if X_gb.ndim != 2 or X_gb.shape[ 1 ] < 1:
							st.warning( '⚠️ Gradient Boosting requires at least one '
							            'numeric feature.' )
							st.stop( )
						
						if y_gb.ndim != 1:
							st.warning( '⚠️ The Gradient Boosting target must be '
							            'one-dimensional.' )
							st.stop( )
						
						if len( X_gb ) != len( y_gb ):
							st.warning( '⚠️ Feature and target row counts do not match.' )
							st.stop( )
						
						if len( X_gb ) < 2:
							st.warning( '⚠️ Gradient Boosting requires at least two '
							            'observations.' )
							st.stop( )
						
						if not np.isfinite( X_gb ).all( ):
							st.warning( '⚠️ The Gradient Boosting feature matrix contains '
							            'non-finite values.' )
							st.stop( )
						
						if not np.isfinite( y_gb ).all( ):
							st.warning( '⚠️ The Gradient Boosting target contains '
							            'non-finite values.' )
							st.stop( )
						
						if len( np.unique( y_gb ) ) < 2:
							st.warning( '⚠️ The regression target must contain at least '
							            'two distinct values.' )
							st.stop( )
						
						if gb_learning_rate <= 0.0:
							st.warning( '⚠️ Learning Rate must be greater than zero.' )
							st.stop( )
						
						if gb_estimators < 1:
							st.warning( '⚠️ Estimators must be greater than zero.' )
							st.stop( )
						
						if gb_subsample <= 0.0 or gb_subsample > 1.0:
							st.warning( '⚠️ Subsample must be greater than zero and '
							            'no greater than one.' )
							st.stop( )
						
						if gb_tol < 0.0:
							st.warning( '⚠️ Tolerance cannot be negative.' )
							st.stop( )
						
						effective_depth = (
							None if gb_max_depth_mode == 'none' else int( gb_max_depth_value ))
						
						if gb_max_features_mode == 'none':
							effective_features = None
						elif gb_max_features_mode == 'sqrt':
							effective_features = 'sqrt'
						elif gb_max_features_mode == 'log2':
							effective_features = 'log2'
						else:
							effective_features = float( gb_max_features_value )
						
						effective_leaf_nodes = (None if gb_max_leaf_nodes_mode == 'none' else int(
							gb_max_leaf_nodes_value ))
						
						effective_no_change = (None if gb_n_iter_no_change_mode == 'none' else int(
							gb_n_iter_no_change_value ))
						
						st.session_state[ 'regression_gb_loss' ] = str( gb_loss )
						st.session_state[ 'regression_gb_learning_rate' ] = float(
							gb_learning_rate )
						st.session_state[ 'regression_gb_estimators' ] = int( gb_estimators )
						st.session_state[ 'regression_gb_subsample' ] = float( gb_subsample )
						st.session_state[ 'regression_gb_criterion' ] = str( gb_criterion )
						st.session_state[ 'regression_gb_min_samples_split' ] = int(
							gb_min_samples_split )
						st.session_state[ 'regression_gb_min_samples_leaf' ] = int(
							gb_min_samples_leaf )
						st.session_state[ 'regression_gb_min_weight_fraction_leaf' ] = float(
							gb_min_weight_fraction_leaf )
						st.session_state[ 'regression_gb_max_depth_mode' ] = str(
							gb_max_depth_mode )
						st.session_state[ 'regression_gb_max_depth_value' ] = int(
							gb_max_depth_value )
						st.session_state[ 'regression_gb_min_impurity_decrease' ] = float(
							gb_min_impurity_decrease )
						st.session_state[ 'regression_gb_max_features_mode' ] = str(
							gb_max_features_mode )
						st.session_state[ 'regression_gb_max_features_value' ] = float(
							gb_max_features_value )
						st.session_state[ 'regression_gb_alpha' ] = float( gb_alpha )
						st.session_state[ 'regression_gb_verbose' ] = int( gb_verbose )
						st.session_state[ 'regression_gb_max_leaf_nodes_mode' ] = str(
							gb_max_leaf_nodes_mode )
						st.session_state[ 'regression_gb_max_leaf_nodes_value' ] = int(
							gb_max_leaf_nodes_value )
						st.session_state[ 'regression_gb_warm_start' ] = bool( gb_warm_start )
						st.session_state[ 'regression_gb_validation_fraction' ] = float(
							gb_validation_fraction )
						st.session_state[ 'regression_gb_n_iter_no_change_mode' ] = str(
							gb_n_iter_no_change_mode )
						st.session_state[ 'regression_gb_n_iter_no_change_value' ] = int(
							gb_n_iter_no_change_value )
						st.session_state[ 'regression_gb_tol' ] = float( gb_tol )
						st.session_state[ 'regression_gb_ccp_alpha' ] = float( gb_ccp_alpha )
						st.session_state[ 'regression_gb_test_size' ] = float( gb_test_size )
						st.session_state[ 'regression_gb_random_state' ] = int( gb_random_state )
						
						start_time = time.perf_counter( )
						
						model = regression_model.GradientBoost( loss=str( gb_loss ),
							rate=float( gb_learning_rate ), estimators=int( gb_estimators ),
							subsample=float( gb_subsample ), criterion=str( gb_criterion ),
							split=int( gb_min_samples_split ), leaf=int( gb_min_samples_leaf ),
							weight_fraction=float( gb_min_weight_fraction_leaf ),
							depth=effective_depth, impurity=float( gb_min_impurity_decrease ),
							init=None, rando=int( gb_random_state ), features=effective_features,
							alpha=float( gb_alpha ), verbose=int( gb_verbose ),
							leaf_nodes=effective_leaf_nodes, warm=bool( gb_warm_start ),
							validation_fraction=float( gb_validation_fraction ),
							no_change=effective_no_change, tol=float( gb_tol ),
							ccp_alpha=float( gb_ccp_alpha ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X_gb, y_gb,
							size=float( gb_test_size ), random=int( gb_random_state ) )
						
						if len( X_train ) < 1 or len( X_test ) < 1:
							st.warning( '⚠️ The selected test size does not produce valid '
							            'training and testing partitions.' )
							st.stop( )
						
						if effective_no_change is not None and len( X_train ) < 10:
							st.warning( '⚠️ Early stopping requires a larger training set. '
							            'Disable N Iter No Change or use more observations.' )
							st.stop( )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if not isinstance( df_scores, pd.DataFrame ):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						else:
							df_scores = df_scores.copy( )
						
						if ('Metric' not in df_scores.columns or 'Value' not in df_scores.columns):
							df_scores = pd.DataFrame( columns=[ 'Metric', 'Value' ] )
						
						df_metadata = pd.DataFrame( {
							'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
								'Testing Rows', 'Estimators', 'Learning Rate', 'Loss', 'Subsample',
								'Criterion' ],
							'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
								int( len( X_test ) ), int( gb_estimators ),
								float( gb_learning_rate ), str( gb_loss ), float( gb_subsample ),
								str( gb_criterion ) ] } )
						
						df_scores = pd.concat( [ df_scores, df_metadata ], ignore_index=True )
						
						y_prediction = np.asarray( y_prediction, dtype=float ).reshape( -1 )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_gb_elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Gradient Boosting training completed.' )
					
					except Exception as ex:
						st.error( f'Gradient Boosting training failed: {ex}' )
			
			with st.expander( 'Voting Regressor', expanded=False ):
				vote_defaults = { 'regression_vote_include_ols': True,
					'regression_vote_include_ridge': True, 'regression_vote_include_knn': True,
					'regression_vote_use_weights': False, 'regression_vote_weight_ols': 1.0,
					'regression_vote_weight_ridge': 1.0, 'regression_vote_weight_knn': 1.0,
					'regression_vote_jobs': 1, 'regression_vote_verbose': False,
					'regression_vote_test_percent': 20, 'regression_vote_test_size': 0.20,
					'regression_vote_random_state': 42 }
				
				for key, value in vote_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				def reset_voting_regressor( ) -> None:
					"""Reset Voting Regressor state.

						Purpose:
							Restores Voting Regressor controls and output dataframes to their
							default
							state without modifying dataset, feature, or target selections.

						Returns:
							None
					"""
					for key, value in vote_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_vote_elapsed_seconds' ]=None
				
				st.caption(
					'Average predictions from multiple regressors fit on the full dataset.' )
				
				vote_c1, vote_c2, vote_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with vote_c1:
					st.markdown( '###### Base Estimators' )
					
					vote_include_ols = st.checkbox( 'Ordinary Least Squares',
						key='regression_vote_include_ols' )
					
					vote_include_ridge = st.checkbox( 'Ridge Regression',
						key='regression_vote_include_ridge' )
					
					vote_include_knn = st.checkbox( 'k-Nearest Neighbors',
						key='regression_vote_include_knn' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with vote_c2:
					st.markdown( '###### Weighting' )
					
					vote_use_weights = st.checkbox( 'Use Custom Weights',
						key='regression_vote_use_weights' )
					
					vote_weight_ols = float(
						st.number_input( 'OLS Weight', min_value=0.0, step=0.10, format='%.2f',
							key='regression_vote_weight_ols' ) )
					
					vote_weight_ridge = float(
						st.number_input( 'Ridge Weight', min_value=0.0, step=0.10, format='%.2f',
							key='regression_vote_weight_ridge' ) )
					
					vote_weight_knn = float(
						st.number_input( 'kNN Weight', min_value=0.0, step=0.10, format='%.2f',
							key='regression_vote_weight_knn' ) )
					
					st.caption( 'Weights are only applied when custom weighting is enabled.' )
				
				with vote_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					
					vote_jobs = int( st.number_input( 'Parallel Jobs', min_value=1, step=1,
						key='regression_vote_jobs' ) )
					
					vote_verbose = st.checkbox( 'Verbose', key='regression_vote_verbose' )
					
					vote_test_percent = int(
						st.slider( 'Test Set Size (%)', min_value=10, max_value=30, step=1,
							key='regression_vote_test_percent' ) )
					vote_test_size = float( vote_test_percent / 100.0 )
					
					vote_random_state = int( st.number_input( 'Random State', min_value=0, step=1,
						key='regression_vote_random_state' ) )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}' )
				
				vote_btn_1, vote_btn_2 = st.columns( 2 )
				with vote_btn_1:
					train_vote = st.button( '🚂 Train Voting Regressor',
						key='regression_vote_train',
						use_container_width=True )
				
				with vote_btn_2:
					st.button( '🔄 Reset Voting Regressor', key='regression_vote_reset',
						use_container_width=True, on_click=reset_voting_regressor )
				
				if train_vote:
					try:
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply( pd.to_numeric,
							errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( X ) < 3:
							st.warning( '⚠️ Voting Regressor requires at least three rows.' )
							st.stop( )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two '
								'distinct '
								'values.' )
							st.stop( )
						
						estimated_training_rows = max( 1,
							int( np.floor( len( X ) * (1.0 - vote_test_size) ) ) )
						vote_neighbors = min( 5, estimated_training_rows )
						
						estimators = [ ]
						weights = [ ]
						
						if vote_include_ols:
							estimators.append(
								('least_squares', regression_model.skl.LinearRegression( )) )
							weights.append( float( vote_weight_ols ) )
						
						if vote_include_ridge:
							estimators.append( ('ridge', regression_model.skl.Ridge(
								random_state=int( vote_random_state ) )) )
							weights.append( float( vote_weight_ridge ) )
						
						if vote_include_knn:
							estimators.append( ('nearest_neighbor',
								regression_model.skn.KNeighborsRegressor(
									n_neighbors=vote_neighbors )) )
							weights.append( float( vote_weight_knn ) )
						
						if len( estimators ) < 2:
							st.warning(
								'⚠️ Voting Regressor requires at least two base estimators.' )
							st.stop( )
						
						if vote_use_weights and all( weight == 0.0 for weight in weights ):
							st.warning( '⚠️ At least one voting weight must be greater than '
							            'zero.' )
							st.stop( )
						
						start_time = time.perf_counter( )
						
						model = regression_model.VotingModel( est=estimators,
							weights=weights if vote_use_weights else None, jobs=int( vote_jobs ),
							verbose=bool( vote_verbose ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( vote_test_size ), random=int( vote_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = np.asarray( model.project( X_test ), dtype=float ).reshape(
							-1 )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test )
						
						if df_scores is None:
							df_scores = pd.DataFrame( )
						else:
							df_scores = df_scores.copy( )
						
						if not df_scores.empty:
							df_extra = pd.DataFrame( {
								'Metric': [ 'Processing Time (Seconds)', 'Training Rows',
									'Testing Rows', 'Estimator Count', 'Weighted Voting' ],
								'Value': [ round( elapsed_seconds, 4 ), int( len( X_train ) ),
									int( len( X_test ) ), int( len( estimators ) ),
									bool( vote_use_weights ) ] } )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': np.asarray( y_test, dtype=float ).reshape( -1 ),
								'Predicted': y_prediction } )
						
						st.session_state[ 'regression_vote_test_size' ] = vote_test_size
						
						st.session_state[ 'regression_vote_elapsed_seconds' ] = elapsed_seconds
						
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = np.asarray( X_train ).copy( )
						st.session_state[ 'X_test' ] = np.asarray( X_test ).copy( )
						st.session_state[ 'y_train' ] = np.asarray( y_train ).copy( )
						st.session_state[ 'y_test' ] = np.asarray( y_test ).copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_regression' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_regression_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
						
						st.success( 'Voting Regressor training completed.' )
					except Exception as ex:
						st.error( f'Voting Regressor training failed: {ex}' )
			
			with st.expander( 'Stacking Regressor', expanded=False ):
				stack_defaults = { 'regression_stack_include_ols': True,
					'regression_stack_include_ridge': True, 'regression_stack_include_knn': True,
					'regression_stack_final_estimator': 'ridge',
					'regression_stack_cv_mode': 'default', 'regression_stack_cv_value': 5,
					'regression_stack_jobs': 1, 'regression_stack_passthrough': False,
					'regression_stack_verbose': 0, 'regression_stack_test_size': 20,
					'regression_stack_random_state': 42 }
				
				for key, value in stack_defaults.items( ):
					if key not in st.session_state:
						st.session_state[ key ] = value
				
				st.caption( 'Stacked generalization ensemble for continuous targets.' )
				stack_c1, stack_c2, stack_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
				with stack_c1:
					st.markdown( '###### Base Estimators' )
					
					stack_include_ols = st.checkbox( 'Ordinary Least Squares',
						value=bool( st.session_state[ 'regression_stack_include_ols' ] ),
						key='regression_stack_include_ols' )
					
					stack_include_ridge = st.checkbox( 'Ridge Regression',
						value=bool( st.session_state[ 'regression_stack_include_ridge' ] ),
						key='regression_stack_include_ridge' )
					
					stack_include_knn = st.checkbox( 'k-Nearest Neighbors',
						value=bool( st.session_state[ 'regression_stack_include_knn' ] ),
						key='regression_stack_include_knn' )
					
					st.caption( 'Select at least two base estimators.' )
				
				with stack_c2:
					st.markdown( '###### Meta-Estimator' )
					
					stack_final_estimator = st.selectbox( 'Final Estimator',
						options=[ 'linear_regression', 'ridge', 'knn' ],
						index=[ 'linear_regression', 'ridge', 'knn' ].index(
							st.session_state[ 'regression_stack_final_estimator' ] ),
						key='regression_stack_final_estimator' )
					
					stack_cv_mode = st.selectbox( 'Cross-Validation',
						options=[ 'default', 'custom' ], index=[ 'default', 'custom' ].index(
							st.session_state[ 'regression_stack_cv_mode' ] ),
						key='regression_stack_cv_mode' )
					
					stack_cv_value = int( st.number_input( 'CV Folds', min_value=2,
						value=int( st.session_state[ 'regression_stack_cv_value' ] ), step=1,
						key='regression_stack_cv_value' ) )
					
					stack_passthrough = st.checkbox( 'Passthrough Original Features',
						value=bool( st.session_state[ 'regression_stack_passthrough' ] ),
						key='regression_stack_passthrough' )
				
				with stack_c3:
					st.markdown( '###### 🏃 Run Configuration' )
					
					stack_jobs = int( st.number_input( 'Parallel Jobs', min_value=1,
						value=int( st.session_state[ 'regression_stack_jobs' ] ), step=1,
						key='regression_stack_jobs' ) )
					
					stack_verbose = int( st.number_input( 'Verbose', min_value=0,
						value=int( st.session_state[ 'regression_stack_verbose' ] ), step=1,
						key='regression_stack_verbose' ) )
					
					stack_test_size = st.slider( 'Test Set Size (%)', min_value=10, max_value=30,
						value=int( st.session_state[ 'regression_stack_test_size' ] ), step=1,
						key='regression_stack_test_size' ) / 100.0
					
					stack_random_state = int( st.number_input( 'Random State', min_value=0,
						value=int( st.session_state[ 'regression_stack_random_state' ] ), step=1,
						key='regression_stack_random_state' ) )
					
					st.caption(
						f'Rows: {len( df_model ):,} | Features: {len( active_features ):,} | '
						f'Target: {target_name}' )
				
				stack_btn_1, stack_btn_2 = st.columns( 2 )
				
				with stack_btn_1:
					train_stack = st.button( '🚂 Train Stacking Regressor',
						key='regression_stack_train', use_container_width=True )
				
				with stack_btn_2:
					reset_stack = st.button( '🔄 Reset Stacking Regressor',
						key='regression_stack_reset', use_container_width=True )
				
				if reset_stack:
					for key, value in stack_defaults.items( ):
						st.session_state[ key ] = value
					
					st.session_state[ 'df_regression' ] = df_model.copy( )
					st.session_state[ 'df_scores' ] = pd.DataFrame( )
					st.session_state[ 'df_predictions' ] = pd.DataFrame( )
					st.session_state[ 'regression_stack_elapsed_seconds' ]=None
					st.rerun( )
				
				if train_stack:
					try:
						df_training = df_model.copy( )
						
						X = df_training[ active_features ].apply( pd.to_numeric,
							errors='coerce' ).fillna( 0.0 ).to_numpy( )
						
						y = pd.to_numeric( df_training[ target_name ], errors='coerce' ).fillna(
							0.0 ).to_numpy( ).reshape( -1 )
						
						if len( np.unique( y ) ) < 2:
							st.warning(
								'⚠️ The selected numeric target must contain at least two distinct'
								' values.' )
							st.stop( )
						
						estimators = [ ]
						
						if stack_include_ols:
							estimators.append(
								('least_squares', regression_model.skl.LinearRegression( )) )
						
						if stack_include_ridge:
							estimators.append( ('ridge', regression_model.skl.Ridge(
								random_state=int( stack_random_state ) )) )
						
						if stack_include_knn:
							estimators.append(
								('nearest_neighbor', regression_model.skn.KNeighborsRegressor( )) )
						
						if len( estimators ) < 2:
							st.warning(
								'⚠️ Stacking Regressor requires at least two base estimators.' )
							st.stop( )
						
						if stack_final_estimator == 'linear_regression':
							final_estimator = regression_model.skl.LinearRegression( )
						elif stack_final_estimator == 'ridge':
							final_estimator = regression_model.skl.Ridge(
								random_state=int( stack_random_state ) )
						else:
							final_estimator = regression_model.skn.KNeighborsRegressor( )
						
						effective_cv = None if stack_cv_mode == 'default' else int(
							stack_cv_value )
						
						start_time = time.perf_counter( )
						
						model = regression_model.StackingModel( est=estimators,
							final=final_estimator, cv=effective_cv, jobs=int( stack_jobs ),
							passthrough=bool( stack_passthrough ), verbose=int( stack_verbose ) )
						
						X_train, X_test, y_train, y_test = model.split_data( X, y,
							size=float( stack_test_size ), random=int( stack_random_state ) )
						
						model.train( X_train, y_train )
						y_prediction = model.project( X_test )
						
						elapsed_seconds = float( time.perf_counter( ) - start_time )
						
						df_scores = model.analyze( X_test, y_test ).copy( )
						
						if df_scores is not None and not df_scores.empty:
							if 'Metric' in df_scores.columns and 'Value' in df_scores.columns:
								df_scores = df_scores.copy( )
							elif df_scores.shape[ 1 ] == 1:
								df_scores = df_scores.reset_index( )
								df_scores.columns = [ 'Metric', 'Value' ]
							
							df_extra = pd.DataFrame( {
								'Metric': [ 'Training Score', 'Testing Score',
									'Processing Time (Seconds)', 'Training Rows', 'Testing Rows',
									'Estimator Count', 'Passthrough' ],
								'Value': [ float( model.training_score ),
									float( model.testing_score ), round( elapsed_seconds, 4 ),
									int( len( X_train ) ), int( len( X_test ) ),
									int( len( estimators ) ), bool( stack_passthrough ) ] } )
							
							df_scores = pd.concat( [ df_scores, df_extra ], ignore_index=True )
						
						df_predictions = pd.DataFrame(
							{ 'Actual': y_test, 'Predicted': y_prediction } )
						st.session_state[ 'elapsed_seconds' ] = elapsed_seconds
						st.session_state[ 'model' ] = model
						st.session_state[ 'y_prediction' ] = y_prediction.copy( )
						st.session_state[ 'X_train' ] = X_train.copy( )
						st.session_state[ 'X_test' ] = X_test.copy( )
						st.session_state[ 'y_train' ] = y_train.copy( )
						st.session_state[ 'y_test' ] = y_test.copy( )
						st.session_state[ 'df_model' ] = df_model.copy( )
						st.session_state[ 'df_scores' ] = df_scores.copy( )
						st.session_state[ 'df_predictions' ] = df_predictions.copy( )
					except Exception as ex:
						st.error( f'Stacking Regressor training failed: {ex}' )
		
		model = st.session_state.get( 'model', None )
		X_test = st.session_state.get( 'X_test', None )
		y_test = st.session_state.get( 'y_test', None )
		
		if model is None or X_test is None or y_test is None:
			st.stop( )
		
		# ------------------------------------------------------------------
		# PREDICTIONS
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Predictions' )
		y_prediction = model.project( X_test )
		df_predictions = pd.DataFrame(
			{ 'Observed': y_test, 'Predicted': y_prediction, 'Residual': y_test - y_prediction } )
		
		st.session_state[ 'y_prediction' ] = y_prediction.copy( )
		st.session_state[ 'df_predictions' ] = df_predictions.copy( )
		
		render_data_editor( df_predictions, use_container_width=True )
		
		# ------------------------------------------------------------------
		# MODEL DETAILS
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Model Details' )
		detail_rows = [ ]
		
		if hasattr( model, 'features' ):
			try:
				detail_rows.append( { 'Property': 'Features', 'Value': model.features } )
			except Exception:
				pass
		
		if hasattr( model, 'training_score' ):
			try:
				detail_rows.append(
					{ 'Property': 'Training Score', 'Value': model.training_score } )
			except Exception:
				pass
		
		if hasattr( model, 'testing_score' ):
			try:
				detail_rows.append( { 'Property': 'Testing Score', 'Value': model.testing_score } )
			except Exception:
				pass
		
		if hasattr( model, 'weights' ):
			try:
				weights = model.weights
				if weights is not None:
					df_weights = pd.DataFrame(
						{ 'Feature': features, 'Weight': np.asarray( weights ).reshape( -1 ) } )
					st.caption( 'Coefficients' )
					render_data_editor( df_weights, use_container_width=True )
			except Exception:
				pass
		
		if hasattr( model, 'intercept' ):
			try:
				intercept = model.intercept
				if intercept is not None:
					detail_rows.append( { 'Property': 'Intercept', 'Value': intercept } )
			except Exception:
				pass
		
		if detail_rows:
			df_details = pd.DataFrame( detail_rows )
			render_data_editor( df_details, use_container_width=True )
		else:
			st.info( 'No additional model details are exposed for this regressor.' )
		
		# ------------------------------------------------------------------
		# SCATTER PLOT
		# ------------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Observed vs Predicted' )
		df_scatter = pd.DataFrame( { 'Observed': np.asarray( y_test ).reshape( -1 ),
			'Predicted': np.asarray( y_prediction ).reshape( -1 ) } )
		figure = px.scatter( df_scatter, x='Observed', y='Predicted' )
		minimum_value = float( df_scatter[ [ 'Observed', 'Predicted' ] ].min( ).min( ) )
		maximum_value = float( df_scatter[ [ 'Observed', 'Predicted' ] ].max( ).max( ) )
		figure.add_trace( go.Scatter( x=[ minimum_value, maximum_value ],
			y=[ minimum_value, maximum_value ], mode='lines', name='Ideal',
			line={ 'dash': 'dash', 'color': '#94A3B8' } ) )
		render_mathy_plotly_chart( figure, 'regression_observed_predicted_chart',
			'mathy_regression_observed_predicted', 'Observed vs Predicted', 540 )

# ============================================
# CLUSTERING MODELS MODE
# ============================================
elif mode == 'Clustering Models':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	df_dataset = get_analysis_dataset( )
	with center:
		st.subheader( cfg.MODE[ 'Clustering Models' ] )
		st.divider( )
		
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		df_original = df_dataset.copy( )
		numeric_columns = st.session_state.get( 'numeric_columns', [ ] ).copy( )
		categorical_columns = st.session_state.get( 'categorical_columns', [ ] ).copy( )
		
		if not numeric_columns:
			st.warning( '⚠️ Clustering requires at least one numeric feature.' )
			st.stop( )
		
		cluster_source_signature = (tuple( df_original.columns.tolist( ) ),
			tuple( str( dtype ) for dtype in df_original.dtypes.tolist( ) ),
			int( len( df_original ) ),
			int( pd.util.hash_pandas_object( df_original, index=True ).sum( ) ))
		
		prior_cluster_source_signature = st.session_state.get( 'clustering_source_signature',
			None )
		
		if prior_cluster_source_signature != cluster_source_signature:
			cluster_state_keys = [ key for key in list( st.session_state.keys( ) ) if
				key.startswith( 'cluster_' ) or key in [ 'clusters_working_data', 'df_cluster',
					'df_cluster_results', 'df_cluster_counts', 'df_cluster_metrics',
					'df_cluster_centroids', 'df_cluster_details', 'df_cluster_ordering',
					'cluster_plot_features', 'cluster_signature' ] ]
			clear_keys( cluster_state_keys )
			st.session_state[ 'df_working' ] = df_original.copy( )
			st.session_state[ 'df_processed' ] = df_original.copy( )
			st.session_state[ 'features' ] = [ ]
			st.session_state[ 'targets' ] = [ ]
			st.session_state[ 'active_features' ] = [ ]
			st.session_state[ 'active_targets' ] = [ ]
		
		st.session_state[ 'clustering_source_signature' ] = cluster_source_signature
		st.session_state[ 'df_original' ] = df_original.copy( )
		st.session_state[ 'numeric_columns' ] = numeric_columns.copy( )
		st.session_state[ 'categorical_columns' ] = categorical_columns.copy( )
		df_working = st.session_state.get( 'df_working', pd.DataFrame( ) )
		df_processed = st.session_state.get( 'df_processed', pd.DataFrame( ) )
		df_cluster = st.session_state.get( 'df_cluster', pd.DataFrame( ) )
		features = st.session_state.get( 'features', [ ] )
		targets = st.session_state.get( 'targets', [ ] )
		if df_working is None or df_working.empty:
			df_working = df_original.copy( )
			st.session_state[ 'df_working' ] = df_working.copy( )
		
		if df_processed is None or df_processed.empty:
			df_processed = df_working.copy( )
			st.session_state[ 'df_processed' ] = df_processed.copy( )
		
		if df_cluster is None or df_cluster.empty:
			df_cluster = df_original.copy( )
			st.session_state[ 'df_cluster' ] = df_cluster.copy( )
		
		# ======================================================================================
		# Data Selection
		# ======================================================================================
		st.markdown( '##### Feature Selection' )
		st.caption( f'Inputs: {len( df_original ):,} | Features: {len( df_original.columns ):,}' )
		col_c1, col_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with col_c1:
			features = st.multiselect( 'Select Features', options=df_working.columns )
		
		with col_c2:
			target_options = [ c for c in numeric_columns if c not in features ]
			targets = st.multiselect( 'Select Targets', options=target_options )
		
		sel_b1, sel_b2 = st.columns( [ 0.5, 0.5 ] )
		with sel_b1:
			if st.button( 'Create Working Dataset', icon='➕', key='cluster_create_dataset',
					use_container_width=True ):
				selected_all = features + [ c for c in targets if c not in features ]
				
				if selected_all:
					df_working = df_original[ selected_all ].copy( )
				else:
					df_working = df_original.copy( )
				
				df_processed = df_working.copy( )
				st.session_state[ 'features' ] = features.copy( )
				st.session_state[ 'targets' ] = targets.copy( )
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = df_processed.copy( )
				commit_frame( df_working )
				st.success( 'Working Dataset Created!' )
		
		with sel_b2:
			if st.button( 'Reset To Original', icon='🔁', key='cluster_reset_to_original',
					use_container_width=True ):
				df_original = df_dataset.copy( )
				df_working = df_original.copy( )
				df_processed = df_working.copy( )
				st.session_state[ 'features' ] = [ ]
				st.session_state[ 'targets' ] = [ ]
				st.session_state[ 'df_working' ] = df_working.copy( )
				st.session_state[ 'df_processed' ] = df_processed.copy( )
				commit_frame( df_working )
				st.success( 'Reset to Original' )
		
		blue_divider( )
		st.markdown( '##### Working Data' )
		st.caption( f'Samples: {len( df_working ):,} | Features: {len( df_working.columns ):,}' )
		render_data_editor( df_working, key='clusters_working_data', disabled=True )
		
		# ------------------------------------------------------------------
		# Training Features
		# ------------------------------------------------------------------
		if df_working.empty:
			st.warning( '⚠️ No complete rows remain after preprocessing and feature selection.' )
			st.stop( )
		
		# -----------------------------------------------------------------
		# Data Processing
		# -----------------------------------------------------------------
		blue_divider( )
		st.markdown( '##### Feature-Engineering' )
		
		feature_c1, feature_c2 = st.columns( [ 0.50, 0.50 ], border=True )
		with feature_c1:
			with st.expander( label='Data Scaling', icon='⚖️', key='cluster_scalers' ):
				
				with st.expander( 'Standard Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.STANDARD_SCALER )
					scale_cols = st.multiselect( 'Columns', options=targets,
						key='cluster_standard_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_standard_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = StandardScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'Standard Scaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_standard_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Min-Max Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.MINMAX_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_minmax_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_minmax_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MinMaxScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'Min-Max Scaler applied.' )
					
					with a2:
						if st.button( 'Reset', key='cluster_minmax_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Robust Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.ROBUST_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_robust_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_robust_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = RobustScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'RobustScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_robust_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Normal Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.NORMAL_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_normal_scaler_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', 'max' ], index=1,
						key='cluster_normal_scaler_norm' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_normal_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = NormalScaler( norm=norm )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'NormalScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_normal_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Max-Absolute Scaler', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.MAXABS_SCALER )
					scale_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_maxabs_scaler_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_maxabs_scaler_apply',
								use_container_width=True ):
							if scale_cols:
								scaler = MaxAbsScaler( )
								result = scaler.train_transform(
									df_processed[ scale_cols ].to_numpy( ) )
								df_processed[ scale_cols ] = result
								commit_frame( df_processed )
								st.success( 'MaxAbsScaler applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_maxabs_scaler_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Data Imputation', icon='🧹', key='cluster_imputers' ):
				
				with st.expander( 'Mean Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.MEAN_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_mean_imputer_cols' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='cluster_mean_imputer_indicator' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_mean_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = MeanImputer( strategy='mean',
									add_indicator=add_indicator )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols, result,
									'mean_imputer' )
								
								commit_frame( df_processed )
								st.success( 'MeanImputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_mean_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Nearest Neighbor Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.NEAREST_NEIGHBOR_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_nearest_imputer_cols' )
					
					neighbors = st.number_input( 'Neighbors', min_value=1, value=5, step=1,
						key='cluster_nearest_imputer_neighbors' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_nearest_imputer_apply',
								use_container_width=True ):
							if impute_cols:
								imputer = NearestImputer( neighbors=int( neighbors ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols, result,
									'nearest_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Nearest Imputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_nearest_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Iterative Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.ITERATIVE_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_iterative_imputer_cols' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=10, step=1,
						key='cluster_iterative_imputer_max_iter' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=0, step=1,
						key='cluster_iterative_imputer_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Iterative Imputer',
								key='cluster_iterative_imputer_apply', use_container_width=True ):
							if impute_cols:
								imputer = IterativeImputer( max_iter=int( max_iter ),
									random_state=int( random_state ) )
								result = imputer.train_transform(
									df_processed[ impute_cols ].to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols, result,
									'iterative_imputer' )
								commit_frame( df_processed )
								st.success( 'Iterative Imputer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_iterative_imputer_reset', use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Simple Imputer', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.SIMPLE_IMPUTER )
					impute_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_simple_imputer_cols' )
					
					strategy = st.selectbox( 'Strategy',
						options=[ 'mean', 'median', 'most_frequent', 'constant' ],
						key='cluster_simple_imputer_strategy' )
					
					fill_value = st.text_input( 'Fill Value', value='0.0',
						key='cluster_simple_imputer_fill_value' )
					
					add_indicator = st.checkbox( 'Add Indicator Columns', value=False,
						key='cluster_simple_imputer_indicator' )
					
					keep_empty_features = st.checkbox( 'Keep Empty Features', value=False,
						key='cluster_simple_imputer_keep_empty' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SimpleImputer', key='cluster_simpleimputer_apply',
								use_container_width=True ):
							if impute_cols:
								if strategy in [ 'mean', 'median' ]:
									df_input = df_processed[ impute_cols ].apply( pd.to_numeric,
										errors='coerce' )
									fill_object: object = 0.0
								elif strategy == 'constant':
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								else:
									df_input = df_processed[ impute_cols ].copy( )
									fill_object = fill_value
								
								imputer = SimpleImputer( strategy=strategy, fill_value=fill_object,
									add_indicator=add_indicator,
									keep_empty_features=keep_empty_features )
								
								result = imputer.train_transform( df_input.to_numpy( ) )
								df_processed = replace_columns( df_processed, impute_cols, result,
									'simple_imputer' )
								
								commit_frame( df_processed )
								st.success( 'Simple Imputer Applied' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_simple_imputer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Data Encoding', icon='🔣', key='cluster_encoders' ):
				
				with st.expander( 'One-Hot Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.ONEHOT_ENCODER )
					encode_cols = st.multiselect( 'Columns', options=features,
						key='cluster_onehot_cols' )
					
					sparse = st.checkbox( 'Sparse Output', value=False,
						key='cluster_onehot_sparse' )
					
					unknown = st.selectbox( 'Unknown Category Handling',
						options=[ 'ignore', 'error' ], index=0, key='cluster_onehot_unknown' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_onehot_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OneHotEncoder( sparse=bool( sparse ), unknown=unknown )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'onehot' )
								commit_frame( df_processed )
								st.success( 'OneHotEncoder applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_onehot_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Ordinal Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.ORDINAL_ENCODER )
					encode_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='cluster_ordinal_cols' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_ordinal_apply',
								use_container_width=True ):
							if encode_cols:
								encoder = OrdinalEncoder( )
								result = encoder.train_transform(
									df_processed[ encode_cols ].astype( str ).to_numpy( ) )
								df_processed[ encode_cols ] = result
								commit_frame( df_processed )
								st.success( 'Ordinal Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_ordinal_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.LABEL_ENCODER )
					target_col = st.selectbox( 'Column', options=categorical_columns,
						key='cluster_label_encoder_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_label_encoder_apply',
								use_container_width=True ):
							if target_col:
								encoder = LabelEncoder( )
								result = encoder.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed[ target_col ] = result
								commit_frame( df_processed )
								st.success( 'Label Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_label_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Target Encoder', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.TARGET_ENCODER )
					encode_cols = st.multiselect( 'Categorical Feature Columns',
						options=categorical_columns, key='cluster_target_encoder_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_target_encoder_target_col' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_target_encoder_apply',
								use_container_width=True ):
							if encode_cols and target_col:
								df_processed = df_working.copy( )
								encoder = TargetEncoder( )
								X_enc = df_processed[ encode_cols ].astype( str ).to_numpy( )
								y_enc = df_processed[ target_col ].to_numpy( )
								result = encoder.train_transform( X_enc, y_enc )
								
								df_processed = replace_columns( df_processed, encode_cols, result,
									'target_encoder' )
								
								commit_frame( df_processed )
								st.success( 'Target Encoder Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_target_encoder_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Polynomial Features', expanded=False ):
					st.caption( 'Description', width='stretch', text_alignment='left',
						help=cfg.POLYNOMIAL_FEATURES )
					poly_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_polynomial_cols' )
					
					degree = st.slider( 'Degree', min_value=2, max_value=4, value=2,
						key='cluster_polynomial_degree' )
					
					interaction = st.checkbox( 'Interaction Only', value=True,
						key='cluster_polynomial_interaction' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_polynomial_apply',
								use_container_width=True ):
							if poly_cols:
								df_processed = df_working.copy( )
								
								encoder = PolynomialFeatures( degree=int( degree ),
									interaction=bool( interaction ) )
								
								result = encoder.train_transform(
									df_processed[ poly_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, poly_cols, result,
									'polynomial' )
								
								commit_frame( df_processed )
								st.success( 'PolynomialFeatures applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_polynomial_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
		
		with feature_c2:
			with st.expander( label='Data Transformation', icon='⚡', key='cluster_transformers' ):
				
				with st.expander( 'Binarizer', expanded=False ):
					transform_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_binarizer_cols' )
					
					threshold = st.number_input( 'Threshold', value=0.0, step=0.1,
						key='cluster_binarizer_threshold' )
					
					copy = st.checkbox( 'Copy', value=True, key='cluster_binarizer_copy' )
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Binarizer', key='cluster_binarizer_apply',
								use_container_width=True ):
							if transform_cols:
								df_processed = df_working.copy( )
								transformer = Binarizer( threshold=float( threshold ),
									copy=bool( copy ) )
								result = transformer.train_transform(
									df_processed[ transform_cols ].to_numpy( ) )
								
								df_processed[ transform_cols ] = result
								commit_frame( df_processed )
								st.success( 'Binarizer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column', options=categorical_columns,
						key='cluster_label_binarizer_col' )
					
					pos_label = st.number_input( 'Positive Label', value=1, step=1,
						key='cluster_label_binarizer_pos' )
					
					neg_label = st.number_input( 'Negative Label', value=0, step=1,
						key='cluster_label_binarizer_neg' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='cluster_label_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply LabelBinarizer', key='cluster_label_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								transformer = LabelBinarizer( pos_label=int( pos_label ),
									neg_label=int( neg_label ),
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform(
									df_processed[ target_col ].astype( str ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'label_binarizer' )
								commit_frame( df_processed )
								st.success( 'Label Binarizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_label_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Multi-Label Binarizer', expanded=False ):
					target_col = st.selectbox( 'Column', options=categorical_columns,
						key='cluster_multilabel_binarizer_col' )
					
					delimiter = st.text_input( 'Delimiter', value=',',
						key='cluster_multilabel_binarizer_delimiter' )
					
					sparse_output = st.checkbox( 'Sparse Output', value=False,
						key='cluster_multilabel_binarizer_sparse' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_multilabel_binarizer_apply',
								use_container_width=True ):
							if target_col:
								df_processed = df_working.copy( )
								y_multi = parse_multilabel_series( df_processed[ target_col ],
									delimiter=delimiter )
								
								transformer = MultiLabelBinarizer( classes=None,
									sparse_output=bool( sparse_output ) )
								
								result = transformer.train_transform( y_multi )
								df_processed = replace_columns( df_processed, [ target_col ],
									result, 'multilabel_binarizer' )
								
								commit_frame( df_processed )
								st.success( 'Multi-Label Binarizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_multilabel_binarizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'TF-IDF Transformer', expanded=False ):
					text_count_cols = st.multiselect( 'Count Matrix Columns',
						options=numeric_columns, key='cluster_tfidf_transformer_cols' )
					
					norm = st.selectbox( 'Norm', options=[ 'l1', 'l2', None ], index=1,
						key='cluster_tfidf_transformer_norm' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='cluster_tfidf_transformer_use_idf' )
					
					smooth_idf = st.checkbox( 'Smooth IDF', value=True,
						key='cluster_tfidf_transformer_smooth_idf' )
					
					sublinear_tf = st.checkbox( 'Sublinear TF', value=False,
						key='cluster_tfidf_transformer_sublinear' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_tfidf_transformer_apply', use_container_width=True ):
							if text_count_cols:
								df_processed = df_working.copy( )
								transformer = TfidfTransformer( norm=norm, use_idf=bool( use_idf ),
									smooth_idf=bool( smooth_idf ),
									sublinear_tf=bool( sublinear_tf ) )
								
								result = transformer.train_transform(
									df_processed[ text_count_cols ].apply( pd.to_numeric,
										errors='coerce' ).fillna( 0.0 ).to_numpy( ) )
								
								df_processed = replace_columns( df_processed, text_count_cols,
									result, 'tfidf_transformer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Transformer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_tfidf_transformer_reset', use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Column Transformer', expanded=False ):
					numeric_columns = st.multiselect( 'Numeric Columns', options=numeric_columns,
						key='cluster_column_transformer_numeric_columns' )
					
					categorical_columns = st.multiselect( 'Categorical Columns',
						options=categorical_columns,
						key='cluster_column_transformer_categorical_columns' )
					
					numeric_transform = st.selectbox( 'Numeric Transformer',
						options=[ 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
							'Binarizer', 'None' ],
						key='cluster_column_transformer_numeric_transform' )
					
					categorical_transform = st.selectbox( 'Categorical Transformer',
						options=[ 'OneHotEncoder', 'OrdinalEncoder', 'None' ],
						key='cluster_column_transformer_categorical_transform' )
					
					remainder = st.selectbox( 'Remainder', options=[ 'drop', 'passthrough' ],
						key='cluster_column_transformer_remainder' )
					
					sparse_threshold = st.slider( 'Sparse Threshold', min_value=0.0, max_value=1.0,
						value=0.3, key='cluster_column_transformer_sparse_threshold' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply Column Transformer',
								key='cluster_column_transformer_apply', use_container_width=True ):
							df_processed = df_working.copy( )
							transformers = [ ]
							
							if numeric_columns and numeric_transform != 'None':
								if numeric_transform == 'StandardScaler':
									numeric_model = StandardScaler( ).model
								elif numeric_transform == 'MinMaxScaler':
									numeric_model = MinMaxScaler( ).model
								elif numeric_transform == 'RobustScaler':
									numeric_model = RobustScaler( ).model
								elif numeric_transform == 'MaxAbsScaler':
									numeric_model = MaxAbsScaler( ).model
								else:
									numeric_model = Binarizer( ).model
								
								transformers.append( ('numeric', numeric_model, numeric_columns) )
							
							if categorical_columns and categorical_transform != 'None':
								if categorical_transform == 'OneHotEncoder':
									categorical_model = OneHotEncoder( sparse=False,
										unknown='ignore' ).model
								else:
									categorical_model = OrdinalEncoder( ).model
								
								transformers.append(
									('categorical', categorical_model, categorical_columns) )
							
							if transformers:
								transformer = ColumnTransformer( transformers=transformers,
									remainder=remainder, sparse_threshold=float(
										sparse_threshold ),
									n_jobs=None, transformer_weights=None, verbose=False )
								
								result = transformer.train_transform( df_processed )
								df_processed = normalize_result_frame( result=result,
									index=df_processed.index, prefix='column_transformer',
									columns=None )
								
								commit_frame( df_processed )
								st.success( 'ColumnTransformer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_column_transformer_reset', use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Feature Extration', icon='⛏️', key='cluster_extractors' ):
				
				with st.expander( 'TF-IDF Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=categorical_columns,
						key='cluster_tfidf_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='cluster_tfidf_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='cluster_tfidf_vectorizer_max_features' )
					
					use_idf = st.checkbox( 'Use IDF', value=True,
						key='cluster_tfidf_vectorizer_use_idf' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_tfidf_vectorizer_apply', use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = TfidfVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), use_idf=bool( use_idf ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'tfidf_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'TFIDF Vectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_tfidf_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Count Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=categorical_columns,
						key='cluster_count_vectorizer_cols' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='cluster_count_vectorizer_ngram_max' )
					
					max_features = st.number_input( 'Max Features', min_value=0, value=0, step=1,
						key='cluster_count_vectorizer_max_features' )
					
					binary = st.checkbox( 'Binary Counts', value=False,
						key='cluster_count_vectorizer_binary' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_count_vectorizer_apply', use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = CountVectorizer( ngram_range=(1, int( ngram_max )),
									max_features=None if int( max_features ) == 0 else int(
										max_features ), binary=bool( binary ) )
								
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'count_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'Count Vectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_count_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Hash Vectorizer', expanded=False ):
					text_cols = st.multiselect( 'Text Columns', options=categorical_columns,
						key='cluster_hash_vectorizer_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='cluster_hash_vectorizer_n_features' )
					
					ngram_max = st.slider( 'Max N-Gram', min_value=1, max_value=3, value=1,
						key='cluster_hash_vectorizer_ngram_max' )
					
					binary = st.checkbox( 'Binary', value=False,
						key='cluster_hash_vectorizer_binary' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='cluster_hash_vectorizer_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_hash_vectorizer_apply',
								use_container_width=True ):
							if text_cols:
								df_processed = df_working.copy( )
								transformer = HashVectorizer( num=int( n_features ),
									ngram_range=(1, int( ngram_max )), binary=bool( binary ),
									alternate_sign=bool( alternate_sign ) )
								df_processed = apply_text_vectorizer( df_processed, text_cols,
									transformer, 'hash_vectorizer' )
								commit_frame( df_processed )
								st.success( 'HashVectorizer Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_hash_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Dictionary Vectorizer', expanded=False ):
					dict_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='cluster_dict_vectorizer_cols' )
					
					separator = st.text_input( 'Separator', value='=',
						key='cluster_dict_vectorizer_separator' )
					
					sparse = st.checkbox( 'Sparse Output', value=True,
						key='cluster_dict_vectorizer_sparse' )
					
					sort = st.checkbox( 'Sort Feature Names', value=True,
						key='cluster_dict_vectorizer_sort' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_dict_vectorizer_apply',
								use_container_width=True ):
							if dict_cols:
								df_processed = df_working.copy( )
								transformer = DictVectorizer( dtype=np.float64,
									separator=separator,
									sparse=bool( sparse ), sort=bool( sort ) )
								
								df_processed = apply_dict_transform( df_processed, dict_cols,
									transformer, 'dict_vectorizer' )
								
								commit_frame( df_processed )
								st.success( 'DictVectorizer applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_dict_vectorizer_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Feature Hasher', expanded=False ):
					hash_cols = st.multiselect( 'Columns', options=categorical_columns,
						key='cluster_feature_hasher_cols' )
					
					n_features = st.number_input( 'Number of Features', min_value=8, value=1024,
						step=8, key='cluster_feature_hasher_n_features' )
					
					alternate_sign = st.checkbox( 'Alternate Sign', value=True,
						key='cluster_feature_hasher_alternate_sign' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_feature_hasher_apply',
								use_container_width=True ):
							if hash_cols:
								df_processed = df_working.copy( )
								transformer = FeatureHasher( n_features=int( n_features ),
									input_type='dict', dtype=np.float64,
									alternate_sign=bool( alternate_sign ) )
								
								df_processed = apply_dict_transform( df_processed, hash_cols,
									transformer, 'feature_hasher' )
								commit_frame( df_processed )
								st.success( 'FeatureHasher applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_feature_hasher_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
			
			with st.expander( label='Dimensionality Reduction', icon='🎚️',
					key='cluster_selectors' ):
				
				with st.expander( 'Variance Threshold', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_variance_threshold_cols' )
					
					threshold = st.number_input( 'Threshold', min_value=0.0, value=0.0, step=0.01,
						key='cluster_variance_threshold_value' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️',
								key='cluster_variance_threshold_apply', use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = VarianceThreshold( thresh=float( threshold ) )
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'variance_threshold' )
								
								commit_frame( df_processed )
								st.success( 'VarianceThreshold applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁',
								key='cluster_variance_threshold_reset', use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Canonical Correlation Analysis', expanded=False ):
					X_cols = st.multiselect( 'Predictor Columns', options=numeric_columns,
						key='cluster_cca_x_cols' )
					
					y_cols = st.multiselect( 'Target Columns', options=numeric_columns,
						key='cluster_cca_y_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='cluster_cca_components' )
					
					scale = st.checkbox( 'Scale', value=True, key='cluster_cca_scale' )
					
					max_iter = st.number_input( 'Max Iterations', min_value=1, value=500, step=1,
						key='cluster_cca_max_iter' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_cca_apply',
								use_container_width=True ):
							if X_cols and y_cols:
								df_processed = df_working.copy( )
								selector = CCA( num=int( n_components ), scale=bool( scale ),
									size=int( max_iter ) )
								
								result = selector.train_transform(
									df_processed[ X_cols ].to_numpy( ),
									df_processed[ y_cols ].to_numpy( ) )
								
								df_result = normalize_result_frame( result=result,
									index=df_processed.index, prefix='cca', columns=None )
								
								df_processed = pd.concat(
									[ df_processed.drop( columns=X_cols + y_cols,
										errors='ignore' ),
										df_result ], axis=1 )
								
								commit_frame( df_processed )
								st.success( 'CCA Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_cca_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Principle Component Analysis', expanded=False ):
					select_cols = st.multiselect( 'Columns', options=numeric_columns,
						key='cluster_pca_cols' )
					
					n_components = st.number_input( 'Components', min_value=1, value=2, step=1,
						key='cluster_pca_components' )
					
					solver = st.selectbox( 'SVD Solver',
						options=[ 'auto', 'full', 'randomized', 'covariance_eigh', 'arpack' ],
						key='cluster_pca_solver' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_pca_apply',
								use_container_width=True ):
							if select_cols:
								df_processed = df_working.copy( )
								selector = PCA( num=int( n_components ), solver=solver )
								
								result = selector.train_transform(
									df_processed[ select_cols ].to_numpy( ) )
								
								df_processed = replace_columns( df_processed, select_cols, result,
									'pca' )
								commit_frame( df_processed )
								st.success( 'PCA applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_pca_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Select-Best', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_selectbest_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_selectbest_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ], key='cluster_selectbest_score_name' )
					
					k_best = st.number_input( 'K', min_value=1, value=5, step=1,
						key='cluster_selectbest_k' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_selectbest_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectBest(
									score_func=score_function_from_name( score_name ),
									num=int( k_best ) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_best' )
								commit_frame( df_processed )
								st.success( 'Select Best Applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_selectbest_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Select-Percent', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_selectpercent_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_selectpercent_target_col' )
					
					score_name = st.selectbox( 'Score Function',
						options=[ 'chi2', 'f_classif', 'f_regression', 'mutual_info_classif',
							'mutual_info_regression' ], key='cluster_selectpercent_score_name' )
					
					percentile = st.slider( 'Percentile', min_value=1, max_value=100, value=10,
						key='cluster_selectpercent_percentile' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( 'Apply SelectPercent', key='cluster_selectpercent_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SelectPercent(
									score_func=score_function_from_name( score_name ),
									pct=int( percentile ) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								result = selector.train_transform( X_input, y_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'select_percent' )
								commit_frame( df_processed )
								st.success( 'SelectPercent applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_selectpercent_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Sequential Back Selection', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_sbs_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_sbs_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='cluster_sbs_k_features' )
					
					test_size = st.slider( 'Validation Split', min_value=0.10, max_value=0.50,
						value=0.25, step=0.05, key='cluster_sbs_test_size' )
					
					random_state = st.number_input( 'Random State', min_value=0, value=1, step=1,
						key='cluster_sbs_random_state' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_sbs_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = SBS( classifier=None, k_features=int( k_features ),
									test_size=float( test_size ), random_state=int( random_state
									) )
								
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'sbs' )
								
								commit_frame( df_processed )
								st.success( 'SBS applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_sbs_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
				
				with st.expander( 'Recursive Feature Elimination', expanded=False ):
					X_cols = st.multiselect( 'Feature Columns', options=numeric_columns,
						key='cluster_rfe_x_cols' )
					
					target_col = st.selectbox( 'Target Column', options=categorical_columns,
						key='cluster_rfe_target_col' )
					
					k_features = st.number_input( 'Features To Retain', min_value=1, value=1,
						step=1, key='cluster_rfe_k_features' )
					
					verbose = st.number_input( 'Verbose', min_value=0, value=0, step=1,
						key='cluster_rfe_verbose' )
					
					a1, a2 = st.columns( 2 )
					with a1:
						if st.button( label='Apply', icon='✔️', key='cluster_rfe_apply',
								use_container_width=True ):
							if X_cols and target_col:
								df_processed = df_working.copy( )
								selector = RFE( k_features=int( k_features ),
									verbose=int( verbose ) )
								X_input = df_processed[ X_cols ].apply( pd.to_numeric,
									errors='coerce' ).fillna( 0.0 ).to_numpy( )
								
								y_input = df_processed[ target_col ].to_numpy( )
								selector.train( X_input, y_input )
								result = selector.transform( X_input )
								df_processed = replace_columns( df_processed, X_cols, result,
									'rfe' )
								commit_frame( df_processed )
								st.success( 'RFE applied.' )
					
					with a2:
						if st.button( label='Reset', icon='🔁', key='cluster_rfe_reset',
								use_container_width=True ):
							st.session_state[ 'df_processed' ] = df_working.copy( )
							df_processed = st.session_state.get( 'df_processed',
								df_working.copy( ) )
							commit_frame( df_processed )
							st.success( 'Reset to Working.' )
		
		blue_divider( )
		st.markdown( '##### Processed Data' )
		st.caption(
			f'Samples: {len( df_processed ):,} | Features: {len( df_processed.columns ):,}' )
		render_data_editor( df_processed, key='cluster_processed_data' )
		
		# ------------------------------------------------------------------
		# MODEL INPUT
		# ------------------------------------------------------------------
		active_features = [ feature for feature in st.session_state.get( 'features', [ ] ) if
			feature in df_processed.columns and pd.api.types.is_numeric_dtype(
				df_processed[ feature ] ) ]
		
		if not active_features:
			active_features = [ column for column in df_processed.columns if
				pd.api.types.is_numeric_dtype(
					df_processed[ column ] ) and column not in st.session_state.get( 'targets',
					[ ] ) ]
		
		if not active_features:
			st.warning( '⚠️ Clustering requires at least one processed numeric feature column.' )
			st.stop( )
		
		feature_columns = active_features.copy( )
		df_cluster_input = df_processed[ feature_columns ].copy( )
		for column in feature_columns:
			df_cluster_input[ column ] = pd.to_numeric( df_cluster_input[ column ],
				errors='coerce' )
		
		df_cluster_input = df_cluster_input.replace( [ np.inf, -np.inf ], np.nan )
		if df_cluster_input.isna( ).any( ).any( ):
			st.warning(
				'⚠️ One or more clustering feature columns contain invalid numeric values.' )
			st.stop( )
		
		if len( df_cluster_input ) < 2:
			st.warning( '⚠️ Clustering requires at least two complete rows.' )
			st.stop( )
		
		X = df_cluster_input.to_numpy( dtype=float )
		cluster_data_signature = int( pd.util.hash_pandas_object( df_cluster_input, index=True ).sum( ) )
		st.session_state[ 'active_features' ] = active_features.copy( )
		st.session_state[ 'X_data' ] = df_cluster_input.copy( )
		
		# ------------------------------------------------------------------
		# MODEL SELECTION
		# ------------------------------------------------------------------
		st.markdown( '##### Clustering Model' )
		
		if 'cluster_kmeans_n_clusters' not in st.session_state:
			st.session_state[ 'cluster_kmeans_n_clusters' ] = 3
		
		if 'cluster_kmeans_init' not in st.session_state:
			st.session_state[ 'cluster_kmeans_init' ] = 'k-means++'
		
		if 'cluster_kmeans_n_init_mode' not in st.session_state:
			st.session_state[ 'cluster_kmeans_n_init_mode' ] = 'auto'
		
		if 'cluster_kmeans_n_init_value' not in st.session_state:
			st.session_state[ 'cluster_kmeans_n_init_value' ] = 10
		
		if 'cluster_kmeans_max_iter' not in st.session_state:
			st.session_state[ 'cluster_kmeans_max_iter' ] = 300
		
		if 'cluster_kmeans_tol' not in st.session_state:
			st.session_state[ 'cluster_kmeans_tol' ] = 0.0001
		
		if 'cluster_kmeans_random_state' not in st.session_state:
			st.session_state[ 'cluster_kmeans_random_state' ] = 42
		
		if 'cluster_kmeans_verbose' not in st.session_state:
			st.session_state[ 'cluster_kmeans_verbose' ] = 0
		
		if 'cluster_kmeans_copy_x' not in st.session_state:
			st.session_state[ 'cluster_kmeans_copy_x' ] = True
		
		if 'cluster_kmeans_algorithm' not in st.session_state:
			st.session_state[ 'cluster_kmeans_algorithm' ] = 'lloyd'
		
		cluster_max_clusters = max( 2, len( df_cluster_input ) - 1 )
		st.session_state[ 'cluster_kmeans_n_clusters' ] = min(
			max( 2, int( st.session_state[ 'cluster_kmeans_n_clusters' ] ) ),
			cluster_max_clusters )
		df_results = st.session_state.get( 'df_cluster_kmeans_results', pd.DataFrame( ) )
		df_counts = st.session_state.get( 'df_cluster_kmeans_counts', pd.DataFrame( ) )
		df_metrics = st.session_state.get( 'df_cluster_kmeans_metrics', pd.DataFrame( ) )
		df_centroids = st.session_state.get( 'df_cluster_kmeans_centroids', pd.DataFrame( ) )
		df_details = st.session_state.get( 'df_cluster_kmeans_details', pd.DataFrame( ) )
		
		with st.expander( 'K-Means', expanded=True ):
			st.caption( 'Prototype-based clustering using centroid minimization.' )
			
			km_c1, km_c2, km_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with km_c1:
				n_clusters = st.number_input( 'Number of Clusters (K)', min_value=2,
					max_value=cluster_max_clusters, step=1, key='cluster_kmeans_n_clusters' )
				
				init = st.selectbox( 'Initialization', options=[ 'k-means++', 'random' ],
					key='cluster_kmeans_init' )
				
				algorithm = st.selectbox( 'Algorithm', options=[ 'lloyd', 'elkan' ],
					key='cluster_kmeans_algorithm' )
			
			with km_c2:
				n_init_mode = st.selectbox( 'Initialization Runs', options=[ 'auto', 'manual' ],
					key='cluster_kmeans_n_init_mode' )
				
				if n_init_mode == 'manual':
					n_init = int( st.number_input( 'Number of Initializations', min_value=1,
						step=1,
						key='cluster_kmeans_n_init_value' ) )
				else:
					n_init = 'auto'
					st.text_input( 'Number of Initializations', value='auto', disabled=True,
						key='cluster_kmeans_n_init_display' )
				
				max_iter = st.number_input( 'Maximum Iterations', min_value=1, step=1,
					key='cluster_kmeans_max_iter' )
			
			with km_c3:
				tol = st.number_input( 'Tolerance', min_value=0.0, step=0.0001, format='%.4f',
					key='cluster_kmeans_tol' )
				
				random_state = st.number_input( 'Random State', step=1,
					key='cluster_kmeans_random_state' )
				
				verbose = st.number_input( 'Verbose', min_value=0, step=1,
					key='cluster_kmeans_verbose' )
				
				copy_x = st.checkbox( 'Copy Input Data', key='cluster_kmeans_copy_x' )
			
			model = KMeans( clusters=int( n_clusters ), init=init, n_init=n_init, tol=float( tol ),
				rando=int( random_state ), max_iter=int( max_iter ), verbose=int( verbose ),
				copy_x=bool( copy_x ), algorithm=algorithm )
			
			model_parameters = { 'Model': 'K-Means', 'n_clusters': int( n_clusters ), 'init': init,
				'n_init': n_init, 'max_iter': int( max_iter ), 'tol': float( tol ),
				'random_state': int( random_state ), 'verbose': int( verbose ),
				'copy_x': bool( copy_x ), 'algorithm': algorithm }
			
			km_b1, km_b2 = st.columns( 2 )
			with km_b1:
				if st.button( 'Run K-Means', icon='🏃', key='cluster_kmeans_run',
						use_container_width=True ):
					cluster_kmeans_signature = ((tuple( active_features ), cluster_data_signature),
						'K-Means', tuple( (k, str( v )) for k, v in model_parameters.items( ) ))
					
					try:
						start_time = time.time( )
						labels = model.project( X )
						elapsed_seconds = time.time( ) - start_time
						df_results = df_cluster_input.copy( )
						df_results[ 'Cluster' ] = labels
						
						df_counts = ( df_results[ 'Cluster' ].value_counts(
							dropna=False ).rename_axis( 'Cluster' ).reset_index(
							name='Count' ).sort_values( by='Cluster' ).reset_index( drop=True ))
						
						try:
							df_metrics = model.score( X )
							if df_metrics is None:
								df_metrics = pd.DataFrame( )
						except Exception:
							df_metrics = pd.DataFrame( )
						
						if df_metrics is None or df_metrics.empty:
							df_metrics = pd.DataFrame(
								[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
						else:
							df_metrics = df_metrics.copy( )
							df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
						
						detail_rows = [ ]
						for prop in [ 'features', 'inertia', 'iterations', 'metric', 'algorithm',
							'n_clusters', 'n_init', 'max_iter', 'tolerance', 'random_state' ]:
							if hasattr( model, prop ):
								try:
									value = getattr( model, prop )
									if value is not None and not isinstance( value,
											(np.ndarray, pd.DataFrame) ):
										detail_rows.append( { 'Property': prop, 'Value': value } )
								except Exception:
									pass
						
						df_details = (
							pd.DataFrame( detail_rows ) if detail_rows else pd.DataFrame( ))
						df_centroids = pd.DataFrame( )
						if hasattr( model, 'centroids_' ):
							try:
								centroids = model.centroids_
								if centroids is not None:
									df_centroids = pd.DataFrame( centroids,
										columns=feature_columns )
									df_centroids.insert( 0, 'Cluster',
										range( len( df_centroids ) ) )
							except Exception:
								df_centroids = pd.DataFrame( )
						
						st.session_state[ 'df_cluster_kmeans_results' ] = df_results
						st.session_state[ 'df_cluster_kmeans_counts' ] = df_counts
						st.session_state[ 'df_cluster_kmeans_metrics' ] = df_metrics
						st.session_state[ 'df_cluster_kmeans_centroids' ] = df_centroids
						st.session_state[ 'df_cluster_kmeans_details' ] = df_details
						st.session_state[ 'cluster_kmeans_plot_features' ] = feature_columns.copy( )
						st.session_state[ 'cluster_kmeans_signature' ] = cluster_kmeans_signature
						st.success( 'K-Means clustering complete.' )
					except Exception as ex:
						st.session_state[ 'df_cluster_kmeans_results' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_kmeans_counts' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_kmeans_metrics' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_kmeans_centroids' ] = pd.DataFrame( )
						st.session_state[ 'df_cluster_kmeans_details' ] = pd.DataFrame( )
						st.session_state[ 'cluster_kmeans_plot_features' ] = [ ]
						st.session_state[ 'cluster_kmeans_signature' ]=None
						st.error( f'K-Means clustering failed: {ex}' )
			
			kmeans_reset_keys = [ 'cluster_kmeans_n_clusters', 'cluster_kmeans_init',
				'cluster_kmeans_n_init_mode', 'cluster_kmeans_n_init_value',
				'cluster_kmeans_n_init_display', 'cluster_kmeans_max_iter', 'cluster_kmeans_tol',
				'cluster_kmeans_random_state', 'cluster_kmeans_verbose', 'cluster_kmeans_copy_x',
				'cluster_kmeans_algorithm', 'df_cluster_kmeans_results',
				'df_cluster_kmeans_counts',
				'df_cluster_kmeans_metrics', 'df_cluster_kmeans_centroids',
				'df_cluster_kmeans_details', 'cluster_kmeans_plot_features',
				'cluster_kmeans_signature' ]
			
			with km_b2:
				st.button( 'Reset K-Means', icon='🔁', key='cluster_kmeans_reset',
					use_container_width=True, on_click=clear_keys, args=(kmeans_reset_keys,) )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if df_counts is not None and not df_counts.empty:
				render_data_editor( df_counts, use_container_width=True )
				
				if df_metrics is not None and not df_metrics.empty:
					st.caption( 'Metrics' )
					render_data_editor( df_metrics, use_container_width=True )
				
				if df_details is not None and not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run clustering to view cluster counts and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			if df_results is not None and not df_results.empty:
				if len( feature_columns ) == 2:
					figure = create_cluster_figure( df_results, feature_columns,
						'Cluster Assignments', df_centroids )
					render_mathy_plotly_chart( figure, 'cluster_kmeans_assignment_chart',
						'mathy_kmeans_clusters', 'Cluster Assignments', 560 )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run clustering to view the scatter plot.' )
			
			# ------------------------------------------------------------------
			# CENTROIDS (IF AVAILABLE)
			# ------------------------------------------------------------------
			if df_centroids is not None and not df_centroids.empty:
				blue_divider( )
				st.markdown( '##### Cluster Centroids' )
				render_data_editor( df_centroids, use_container_width=True )
		
		if 'cluster_dbscan_eps' not in st.session_state:
			st.session_state[ 'cluster_dbscan_eps' ] = 0.5
		
		if 'cluster_dbscan_min_samples' not in st.session_state:
			st.session_state[ 'cluster_dbscan_min_samples' ] = 5
		
		if 'cluster_dbscan_metric' not in st.session_state:
			st.session_state[ 'cluster_dbscan_metric' ] = 'euclidean'
		
		if 'cluster_dbscan_algorithm' not in st.session_state:
			st.session_state[ 'cluster_dbscan_algorithm' ] = 'auto'
		
		if 'cluster_dbscan_leaf_size' not in st.session_state:
			st.session_state[ 'cluster_dbscan_leaf_size' ] = 30
		
		if 'cluster_dbscan_p' not in st.session_state:
			st.session_state[ 'cluster_dbscan_p' ] = 2.0
		
		if 'cluster_dbscan_n_jobs' not in st.session_state:
			st.session_state[ 'cluster_dbscan_n_jobs' ] = 1
		
		with st.expander( 'DBSCAN', expanded=False ):
			st.caption(
				'Density-based clustering that identifies dense regions and isolates noise.' )
			
			db_c1, db_c2, db_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with db_c1:
				epsilon = st.number_input( 'Neighborhood Radius', min_value=0.0001, step=0.1,
					format='%.4f', key='cluster_dbscan_eps' )
				
				min_samples = st.number_input( 'Minimum Samples', min_value=1, step=1,
					key='cluster_dbscan_min_samples' )
			
			with db_c2:
				metric = st.selectbox( 'Distance Metric',
					options=[ 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine' ],
					key='cluster_dbscan_metric' )
				
				algorithm = st.selectbox( 'Neighbor Algorithm',
					options=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ],
					key='cluster_dbscan_algorithm' )
			
			with db_c3:
				leaf_size = st.number_input( 'Leaf Size', min_value=1, step=1,
					key='cluster_dbscan_leaf_size' )
				
				p_value = st.number_input( 'Minkowski Power', min_value=1.0, step=1.0,
					format='%.1f', key='cluster_dbscan_p' )
				
				n_jobs = st.number_input( 'Parallel Jobs', min_value=-1, step=1,
					key='cluster_dbscan_n_jobs' )
			
			db_b1, db_b2 = st.columns( 2 )
			with db_b1:
				run_dbscan = st.button( 'Run DBSCAN', icon='🏃', key='cluster_dbscan_run',
					use_container_width=True )
			
			dbscan_reset_keys = [ 'cluster_dbscan_eps', 'cluster_dbscan_min_samples',
				'cluster_dbscan_metric', 'cluster_dbscan_algorithm', 'cluster_dbscan_leaf_size',
				'cluster_dbscan_p', 'cluster_dbscan_n_jobs', 'df_cluster_dbscan_results',
				'df_cluster_dbscan_counts', 'df_cluster_dbscan_metrics',
				'df_cluster_dbscan_centroids', 'df_cluster_dbscan_details',
				'cluster_dbscan_plot_features', 'cluster_dbscan_signature' ]
			
			with db_b2:
				st.button( 'Reset DBSCAN', icon='🔁', key='cluster_dbscan_reset',
					use_container_width=True, on_click=clear_keys, args=(dbscan_reset_keys,) )
			
			if run_dbscan:
				model_parameters = { 'Model': 'DBSCAN', 'eps': float( epsilon ),
					'min_samples': int( min_samples ), 'metric': metric, 'algorithm': algorithm,
					'leaf_size': int( leaf_size ), 'p': float( p_value ), 'n_jobs': int( n_jobs ) }
				
				cluster_dbscan_signature = ((tuple( active_features ), cluster_data_signature),
					'DBSCAN', tuple( (k, str( value )) for k, value in model_parameters.items( ) ))
				
				try:
					model = DBSCAN( eps=float( epsilon ), samples=int( min_samples ),
						metric=metric,
						metric_params=None, algorithm=algorithm, leaf_size=int( leaf_size ),
						p=float( p_value ), n_jobs=int( n_jobs ) )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					unique_labels = np.unique( labels )
					cluster_labels = unique_labels[ unique_labels != -1 ]
					noise_count = int( np.sum( labels == -1 ) )
					detail_rows = [ { 'Property': 'features', 'Value': int( model.features ) },
						{ 'Property': 'clusters', 'Value': int( len( cluster_labels ) ) },
						{ 'Property': 'noise_samples', 'Value': noise_count },
						{ 'Property': 'core_samples', 'Value': int( len( model.core_samples ) ) },
						{ 'Property': 'epsilon', 'Value': float( model.epsilon ) },
						{ 'Property': 'min_samples', 'Value': int( model.min_samples ) },
						{ 'Property': 'metric', 'Value': model.metric },
						{ 'Property': 'algorithm', 'Value': model.algorithm },
						{ 'Property': 'leaf_size', 'Value': int( model.leaf_size ) },
						{ 'Property': 'p', 'Value': model.p },
						{ 'Property': 'n_jobs', 'Value': model.n_jobs } ]
					
					df_details = pd.DataFrame( detail_rows )
					df_centroids = pd.DataFrame( )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_dbscan_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_dbscan_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_dbscan_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_dbscan_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_dbscan_details' ] = df_details.copy( )
					st.session_state[ 'cluster_dbscan_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_dbscan_signature' ] = cluster_dbscan_signature
					st.success( 'DBSCAN clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_dbscan_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_dbscan_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_dbscan_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_dbscan_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_dbscan_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_dbscan_plot_features' ] = [ ]
					st.session_state[ 'cluster_dbscan_signature' ]=None
					st.error( f'DBSCAN clustering failed: {ex}' )
			
			dbscan_signature = st.session_state.get( 'cluster_dbscan_signature', None )
			if (isinstance( dbscan_signature, tuple ) and len( dbscan_signature ) > 1 and
					dbscan_signature[ 1 ] == 'DBSCAN'):
				df_results = st.session_state.get( 'df_cluster_dbscan_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_dbscan_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_dbscan_metrics', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_dbscan_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				render_data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					render_data_editor( df_metrics, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run DBSCAN to view cluster counts, noise counts, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					figure = create_cluster_figure( df_results, feature_columns,
						'DBSCAN Cluster Assignments' )
					render_mathy_plotly_chart( figure, 'cluster_dbscan_assignment_chart',
						'mathy_dbscan_clusters', 'DBSCAN Cluster Assignments', 560 )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run DBSCAN to view the scatter plot.' )
		
		if 'cluster_agglomerative_n_clusters' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_n_clusters' ] = 2
		
		if 'cluster_agglomerative_metric' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_metric' ] = 'euclidean'
		
		if 'cluster_agglomerative_linkage' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_linkage' ] = 'ward'
		
		if 'cluster_agglomerative_compute_full_tree' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_compute_full_tree' ] = 'auto'
		
		if 'cluster_agglomerative_use_distance_threshold' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_use_distance_threshold' ] = False
		
		if 'cluster_agglomerative_distance_threshold' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_distance_threshold' ] = 0.0
		
		if 'cluster_agglomerative_compute_distances' not in st.session_state:
			st.session_state[ 'cluster_agglomerative_compute_distances' ] = False
		
		cluster_max_clusters = max( 2, len( df_cluster_input ) - 1 )
		if 'cluster_agglomerative_n_clusters' in st.session_state:
			st.session_state[ 'cluster_agglomerative_n_clusters' ] = min(
				max( 2, int( st.session_state[ 'cluster_agglomerative_n_clusters' ] ) ),
				cluster_max_clusters )
		
		with st.expander( 'Agglomerative Clustering', expanded=False ):
			st.caption( 'Hierarchical bottom-up clustering that successively merges related '
			            'observations.' )
			
			ag_c1, ag_c2, ag_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with ag_c1:
				agglomerative_clusters = int( st.number_input( 'Number of Clusters', min_value=2,
					max_value=max( 2, len( df_cluster_input ) - 1 ), step=1,
					key='cluster_agglomerative_n_clusters' ) )
				
				agglomerative_linkage = st.selectbox( 'Linkage',
					options=[ 'ward', 'complete', 'average', 'single' ],
					key='cluster_agglomerative_linkage' )
			
			with ag_c2:
				agglomerative_metric = st.selectbox( 'Distance Metric',
					options=[ 'euclidean', 'manhattan', 'cosine', 'l1', 'l2' ],
					key='cluster_agglomerative_metric' )
				
				agglomerative_compute_full_tree = st.selectbox( 'Compute Full Tree',
					options=[ 'auto', True, False ],
					key='cluster_agglomerative_compute_full_tree' )
			
			with ag_c3:
				agglomerative_use_distance_threshold = st.checkbox( 'Use Distance Threshold',
					key='cluster_agglomerative_use_distance_threshold' )
				
				agglomerative_distance_threshold = float(
					st.number_input( 'Distance Threshold', min_value=0.0, step=0.1, format='%.4f',
						disabled=not agglomerative_use_distance_threshold,
						key='cluster_agglomerative_distance_threshold' ) )
				
				agglomerative_compute_distances = st.checkbox( 'Compute Merge Distances',
					key='cluster_agglomerative_compute_distances' )
			
			ag_b1, ag_b2 = st.columns( 2 )
			
			with ag_b1:
				run_agglomerative = st.button( 'Run Agglomerative Clustering', icon='🏃',
					key='cluster_agglomerative_run', use_container_width=True )
			
			agglomerative_reset_keys = [ 'cluster_agglomerative_n_clusters',
				'cluster_agglomerative_metric', 'cluster_agglomerative_linkage',
				'cluster_agglomerative_compute_full_tree',
				'cluster_agglomerative_use_distance_threshold',
				'cluster_agglomerative_distance_threshold',
				'cluster_agglomerative_compute_distances', 'df_cluster_agglomerative_results',
				'df_cluster_agglomerative_counts', 'df_cluster_agglomerative_metrics',
				'df_cluster_agglomerative_centroids', 'df_cluster_agglomerative_details',
				'cluster_agglomerative_plot_features', 'cluster_agglomerative_signature' ]
			
			with ag_b2:
				st.button( 'Reset Agglomerative Clustering', icon='🔁',
					key='cluster_agglomerative_reset', use_container_width=True,
					on_click=clear_keys, args=(agglomerative_reset_keys,) )
			
			if run_agglomerative:
				try:
					if (agglomerative_linkage == 'ward' and agglomerative_metric != 'euclidean'):
						st.warning( '⚠️ Ward linkage requires the Euclidean distance metric.' )
						st.stop( )
					
					if ( agglomerative_use_distance_threshold and
							agglomerative_distance_threshold <= 0.0):
						st.warning(
							'⚠️ Distance Threshold must be greater than zero when enabled.' )
						st.stop( )
					
					distance_threshold = (float(
						agglomerative_distance_threshold ) if agglomerative_use_distance_threshold
					                      else None)
					
					model_parameters = { 'Model': 'Agglomerative', 'n_clusters': (
						None if distance_threshold is not None else int( agglomerative_clusters )),
						'metric': agglomerative_metric, 'linkage': agglomerative_linkage,
						'compute_full_tree': agglomerative_compute_full_tree,
						'distance_threshold': distance_threshold,
						'compute_distances': bool( agglomerative_compute_distances ) }
					
					cluster_agglomerative_signature = (
						(tuple( active_features ), cluster_data_signature), 'Agglomerative',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = Agglomerative( clusters=int( agglomerative_clusters ),
						affinity=agglomerative_metric, linkage=agglomerative_linkage,
						metric=agglomerative_metric, memory=None, connectivity=None,
						compute_full_tree=agglomerative_compute_full_tree,
						distance_threshold=distance_threshold,
						compute_distances=bool( agglomerative_compute_distances ), n_clusters=(
							None if distance_threshold is not None else int(
								agglomerative_clusters )) )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					detail_rows = [ ]
					for property_name in [ 'features', 'leaves', 'connected_components',
						'n_clusters', 'metric', 'linkage', 'compute_full_tree',
						'distance_threshold', 'compute_distances' ]:
						if hasattr( model, property_name ):
							try:
								property_value = getattr( model, property_name )
								
								if (property_value is not None and not isinstance( property_value,
										(np.ndarray, pd.DataFrame) )):
									detail_rows.append(
										{ 'Property': property_name, 'Value': property_value } )
							except Exception:
								pass
					
					df_details = (pd.DataFrame( detail_rows ) if detail_rows else pd.DataFrame( ))
					df_centroids = pd.DataFrame( )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_agglomerative_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_agglomerative_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_agglomerative_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_agglomerative_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_agglomerative_details' ] = df_details.copy( )
					st.session_state[ 'cluster_agglomerative_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_agglomerative_signature' ] = cluster_agglomerative_signature
					st.success( 'Agglomerative clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_agglomerative_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_agglomerative_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_agglomerative_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_agglomerative_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_agglomerative_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_agglomerative_plot_features' ] = [ ]
					st.session_state[ 'cluster_agglomerative_signature' ]=None
					st.error( f'Agglomerative clustering failed: {ex}' )
			
			agglomerative_signature = st.session_state.get( 'cluster_agglomerative_signature',
				None )
			
			if (isinstance( agglomerative_signature, tuple ) and len(
					agglomerative_signature ) > 1 and agglomerative_signature[ 1 ] == 'Agglomerative'):
				df_results = st.session_state.get( 'df_cluster_agglomerative_results',
					pd.DataFrame( ) )
				
				df_counts = st.session_state.get( 'df_cluster_agglomerative_counts',
					pd.DataFrame( ) )
				
				df_metrics = st.session_state.get( 'df_cluster_agglomerative_metrics',
					pd.DataFrame( ) )
				
				df_details = st.session_state.get( 'df_cluster_agglomerative_details',
					pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				render_data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					render_data_editor( df_metrics, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run Agglomerative Clustering to view cluster counts and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					figure = create_cluster_figure( df_results, feature_columns,
						'Agglomerative Cluster Assignments' )
					render_mathy_plotly_chart( figure, 'cluster_agglomerative_assignment_chart',
						'mathy_agglomerative_clusters', 'Agglomerative Cluster Assignments', 560 )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Agglomerative Clustering to view the scatter plot.' )
		
		if 'cluster_spectral_n_clusters' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_clusters' ] = 2
		
		if 'cluster_spectral_affinity' not in st.session_state:
			st.session_state[ 'cluster_spectral_affinity' ] = 'rbf'
		
		if 'cluster_spectral_gamma' not in st.session_state:
			st.session_state[ 'cluster_spectral_gamma' ] = 1.0
		
		if 'cluster_spectral_n_neighbors' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_neighbors' ] = 10
		
		if 'cluster_spectral_eigen_solver' not in st.session_state:
			st.session_state[ 'cluster_spectral_eigen_solver' ] = 'None'
		
		if 'cluster_spectral_n_components' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_components' ] = 0
		
		if 'cluster_spectral_n_init' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_init' ] = 10
		
		if 'cluster_spectral_eigen_tolerance' not in st.session_state:
			st.session_state[ 'cluster_spectral_eigen_tolerance' ] = 'auto'
		
		if 'cluster_spectral_assign_labels' not in st.session_state:
			st.session_state[ 'cluster_spectral_assign_labels' ] = 'kmeans'
		
		if 'cluster_spectral_degree' not in st.session_state:
			st.session_state[ 'cluster_spectral_degree' ] = 3.0
		
		if 'cluster_spectral_coef0' not in st.session_state:
			st.session_state[ 'cluster_spectral_coef0' ] = 1.0
		
		if 'cluster_spectral_n_jobs' not in st.session_state:
			st.session_state[ 'cluster_spectral_n_jobs' ] = 1
		
		if 'cluster_spectral_verbose' not in st.session_state:
			st.session_state[ 'cluster_spectral_verbose' ] = False
		
		if 'cluster_spectral_random_state' not in st.session_state:
			st.session_state[ 'cluster_spectral_random_state' ] = 42
		
		cluster_max_clusters = max( 2, len( df_cluster_input ) - 1 )
		if 'cluster_spectral_n_clusters' in st.session_state:
			st.session_state[ 'cluster_spectral_n_clusters' ] = min(
				max( 2, int( st.session_state[ 'cluster_spectral_n_clusters' ] ) ),
				cluster_max_clusters )
		
		with st.expander( 'Spectral Clustering', expanded=False ):
			st.caption( 'Graph-based clustering for identifying nonlinear cluster structures.' )
			
			sp_c1, sp_c2, sp_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with sp_c1:
				spectral_clusters = int( st.number_input( 'Number of Clusters', min_value=2,
					max_value=max( 2, len( df_cluster_input ) - 1 ), step=1,
					key='cluster_spectral_n_clusters' ) )
				
				spectral_affinity = st.selectbox( 'Affinity',
					options=[ 'rbf', 'nearest_neighbors', 'polynomial', 'sigmoid', 'cosine' ],
					key='cluster_spectral_affinity' )
				
				spectral_gamma = float(
					st.number_input( 'Gamma', min_value=0.0001, step=0.1, format='%.4f',
						key='cluster_spectral_gamma' ) )
				
				spectral_neighbors = int( st.number_input( 'Nearest Neighbors', min_value=1,
					max_value=max( 1, len( df_cluster_input ) - 1 ), step=1,
					key='cluster_spectral_n_neighbors' ) )
			
			with sp_c2:
				spectral_eigen_solver_display = st.selectbox( 'Eigen Solver',
					options=[ 'None', 'arpack', 'lobpcg' ], key='cluster_spectral_eigen_solver' )
				
				spectral_components = int(
					st.number_input( 'Embedding Components (0 = Default)', min_value=0,
						max_value=max( 1, len( df_cluster_input ) - 1 ), step=1,
						key='cluster_spectral_n_components' ) )
				
				spectral_n_init = int(
					st.number_input( 'K-Means Initializations', min_value=1, step=1,
						key='cluster_spectral_n_init' ) )
				
				spectral_eigen_tolerance = st.selectbox( 'Eigen Tolerance',
					options=[ 'auto', 0.0, 0.0001, 0.001, 0.01 ],
					key='cluster_spectral_eigen_tolerance' )
			
			with sp_c3:
				spectral_assign_labels = st.selectbox( 'Label Assignment',
					options=[ 'kmeans', 'discretize', 'cluster_qr' ],
					key='cluster_spectral_assign_labels' )
				
				spectral_degree = float(
					st.number_input( 'Polynomial Degree', min_value=1.0, step=1.0, format='%.1f',
						key='cluster_spectral_degree' ) )
				
				spectral_coef0 = float(
					st.number_input( 'Kernel Coefficient', step=0.1, format='%.4f',
						key='cluster_spectral_coef0' ) )
				
				spectral_jobs = int( st.number_input( 'Parallel Jobs', min_value=-1, step=1,
					key='cluster_spectral_n_jobs' ) )
				
				spectral_verbose = st.checkbox( 'Verbose', key='cluster_spectral_verbose' )
				
				spectral_random_state = int( st.number_input( 'Random State', min_value=0, step=1,
					key='cluster_spectral_random_state' ) )
			
			sp_b1, sp_b2 = st.columns( 2 )
			with sp_b1:
				run_spectral = st.button( 'Run Spectral Clustering', icon='🏃',
					key='cluster_spectral_run', use_container_width=True )
			
			spectral_reset_keys = [ 'cluster_spectral_n_clusters', 'cluster_spectral_affinity',
				'cluster_spectral_gamma', 'cluster_spectral_n_neighbors',
				'cluster_spectral_eigen_solver', 'cluster_spectral_n_components',
				'cluster_spectral_n_init', 'cluster_spectral_eigen_tolerance',
				'cluster_spectral_assign_labels', 'cluster_spectral_degree',
				'cluster_spectral_coef0', 'cluster_spectral_n_jobs', 'cluster_spectral_verbose',
				'cluster_spectral_random_state', 'df_cluster_spectral_results',
				'df_cluster_spectral_counts', 'df_cluster_spectral_metrics',
				'df_cluster_spectral_centroids', 'df_cluster_spectral_details',
				'cluster_spectral_plot_features', 'cluster_spectral_signature' ]
			
			with sp_b2:
				st.button( 'Reset Spectral Clustering', icon='🔁', key='cluster_spectral_reset',
					use_container_width=True, on_click=clear_keys, args=(spectral_reset_keys,) )
			
			if run_spectral:
				try:
					if spectral_clusters >= len( df_cluster_input ):
						st.warning(
							'⚠️ Number of Clusters must be less than the number of samples.' )
						st.stop( )
					
					if (spectral_affinity == 'nearest_neighbors' and spectral_neighbors >= len(
							df_cluster_input )):
						st.warning(
							'⚠️ Nearest Neighbors must be less than the number of samples.' )
						st.stop( )
					
					if (spectral_components > 0 and spectral_components >= len( df_cluster_input
					)):
						st.warning(
							'⚠️ Embedding Components must be less than the number of samples.' )
						st.stop( )
					
					spectral_eigen_solver = (
						None if spectral_eigen_solver_display == 'None' else
						spectral_eigen_solver_display)
					
					spectral_n_components = (
						None if spectral_components == 0 else spectral_components)
					
					model_parameters = { 'Model': 'Spectral', 'n_clusters': spectral_clusters,
						'affinity': spectral_affinity, 'gamma': spectral_gamma,
						'n_neighbors': spectral_neighbors, 'eigen_solver': spectral_eigen_solver,
						'n_components': spectral_n_components, 'n_init': spectral_n_init,
						'eigen_tol': spectral_eigen_tolerance,
						'assign_labels': spectral_assign_labels, 'degree': spectral_degree,
						'coef0': spectral_coef0, 'n_jobs': spectral_jobs,
						'verbose': spectral_verbose, 'random_state': spectral_random_state }
					
					cluster_spectral_signature = (
						(tuple( active_features ), cluster_data_signature), 'Spectral',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = Spectral( clusters=spectral_clusters,
						random_state=spectral_random_state, n_init=spectral_n_init,
						gama=spectral_gamma, distance=spectral_affinity,
						neighbors=spectral_neighbors, tolerance=spectral_eigen_tolerance,
						assign=spectral_assign_labels, eigen_solver=spectral_eigen_solver,
						n_components=spectral_n_components, degree=spectral_degree,
						coef0=spectral_coef0, kernel_params=None, n_jobs=spectral_jobs,
						verbose=spectral_verbose )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					detail_rows = [ { 'Property': 'features', 'Value': int( model.features ) },
						{ 'Property': 'n_clusters', 'Value': int( model.n_clusters ) },
						{ 'Property': 'affinity', 'Value': model.affinity },
						{ 'Property': 'gamma', 'Value': float( model.gamma ) },
						{ 'Property': 'n_neighbors', 'Value': int( model.n_neighbors ) },
						{ 'Property': 'eigen_solver', 'Value': model.eigen_solver },
						{ 'Property': 'n_components', 'Value': model.n_components },
						{ 'Property': 'n_init', 'Value': int( model.n_init ) },
						{ 'Property': 'eigen_tolerance', 'Value': model.eigen_tolerance },
						{ 'Property': 'assign_labels', 'Value': model.assign_labels },
						{ 'Property': 'degree', 'Value': model.degree },
						{ 'Property': 'coef0', 'Value': model.coef0 },
						{ 'Property': 'n_jobs', 'Value': model.n_jobs },
						{ 'Property': 'verbose', 'Value': model.verbose },
						{ 'Property': 'random_state', 'Value': model.random_state } ]
					
					df_details = pd.DataFrame( detail_rows )
					df_centroids = pd.DataFrame( )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_spectral_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_spectral_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_spectral_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_spectral_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_spectral_details' ] = df_details.copy( )
					st.session_state[ 'cluster_spectral_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_spectral_signature' ] = cluster_spectral_signature
					st.success( 'Spectral clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_spectral_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_spectral_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_spectral_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_spectral_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_spectral_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_spectral_plot_features' ] = [ ]
					st.session_state[ 'cluster_spectral_signature' ]=None
					st.error( f'Spectral clustering failed: {ex}' )
			
			spectral_signature = st.session_state.get( 'cluster_spectral_signature', None )
			if (isinstance( spectral_signature, tuple ) and len( spectral_signature ) > 1 and
					spectral_signature[ 1 ] == 'Spectral'):
				df_results = st.session_state.get( 'df_cluster_spectral_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_spectral_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_spectral_metrics', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_spectral_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				render_data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					render_data_editor( df_metrics, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run Spectral Clustering to view cluster counts and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					figure = create_cluster_figure( df_results, feature_columns,
						'Spectral Cluster Assignments' )
					render_mathy_plotly_chart( figure, 'cluster_spectral_assignment_chart',
						'mathy_spectral_clusters', 'Spectral Cluster Assignments', 560 )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Spectral Clustering to view the scatter plot.' )
		
		if 'cluster_optics_min_samples' not in st.session_state:
			st.session_state[ 'cluster_optics_min_samples' ] = 5
		
		if 'cluster_optics_max_eps' not in st.session_state:
			st.session_state[ 'cluster_optics_max_eps' ] = 0.0
		
		if 'cluster_optics_metric' not in st.session_state:
			st.session_state[ 'cluster_optics_metric' ] = 'minkowski'
		
		if 'cluster_optics_p' not in st.session_state:
			st.session_state[ 'cluster_optics_p' ] = 2.0
		
		if 'cluster_optics_cluster_method' not in st.session_state:
			st.session_state[ 'cluster_optics_cluster_method' ] = 'xi'
		
		if 'cluster_optics_eps' not in st.session_state:
			st.session_state[ 'cluster_optics_eps' ] = 0.5
		
		if 'cluster_optics_xi' not in st.session_state:
			st.session_state[ 'cluster_optics_xi' ] = 0.05
		
		if 'cluster_optics_predecessor_correction' not in st.session_state:
			st.session_state[ 'cluster_optics_predecessor_correction' ] = True
		
		if 'cluster_optics_min_cluster_size' not in st.session_state:
			st.session_state[ 'cluster_optics_min_cluster_size' ] = 0
		
		if 'cluster_optics_algorithm' not in st.session_state:
			st.session_state[ 'cluster_optics_algorithm' ] = 'auto'
		
		if 'cluster_optics_leaf_size' not in st.session_state:
			st.session_state[ 'cluster_optics_leaf_size' ] = 30
		
		if 'cluster_optics_n_jobs' not in st.session_state:
			st.session_state[ 'cluster_optics_n_jobs' ] = 1
		
		with st.expander( 'OPTICS Clustering', expanded=False ):
			st.caption(
				'Density-based clustering across varying neighborhood radii with explicit noise '
				'detection.' )
			
			op_c1, op_c2, op_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with op_c1:
				optics_min_samples = int( st.number_input( 'Minimum Samples', min_value=2,
					max_value=max( 2, len( df_cluster_input ) ), step=1,
					key='cluster_optics_min_samples' ) )
				
				optics_max_eps_input = float(
					st.number_input( 'Maximum Epsilon (0 = Unlimited)', min_value=0.0, step=0.1,
						format='%.4f', key='cluster_optics_max_eps' ) )
				
				optics_metric = st.selectbox( 'Distance Metric',
					options=[ 'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine' ],
					key='cluster_optics_metric' )
				
				optics_p = float(
					st.number_input( 'Minkowski Power', min_value=1.0, step=1.0, format='%.1f',
						key='cluster_optics_p' ) )
			
			with op_c2:
				optics_cluster_method = st.selectbox( 'Cluster Extraction Method',
					options=[ 'xi', 'dbscan' ], key='cluster_optics_cluster_method' )
				
				optics_eps = float(
					st.number_input( 'DBSCAN Epsilon', min_value=0.0001, step=0.1, format='%.4f',
						disabled=optics_cluster_method != 'dbscan', key='cluster_optics_eps' ) )
				
				optics_xi = float(
					st.number_input( 'Xi', min_value=0.0001, max_value=1.0, step=0.01,
						format='%.4f', disabled=optics_cluster_method != 'xi',
						key='cluster_optics_xi' ) )
				
				optics_predecessor_correction = st.checkbox( 'Predecessor Correction',
					disabled=optics_cluster_method != 'xi',
					key='cluster_optics_predecessor_correction' )
			
			with op_c3:
				optics_min_cluster_size_input = int(
					st.number_input( 'Minimum Cluster Size (0 = Default)', min_value=0,
						max_value=max( 1, len( df_cluster_input ) ), step=1,
						key='cluster_optics_min_cluster_size' ) )
				
				optics_algorithm = st.selectbox( 'Neighbor Algorithm',
					options=[ 'auto', 'ball_tree', 'kd_tree', 'brute' ],
					key='cluster_optics_algorithm' )
				
				optics_leaf_size = int( st.number_input( 'Leaf Size', min_value=1, step=1,
					key='cluster_optics_leaf_size' ) )
				
				optics_n_jobs = int( st.number_input( 'Parallel Jobs', min_value=-1, step=1,
					key='cluster_optics_n_jobs' ) )
			
			op_b1, op_b2 = st.columns( 2 )
			with op_b1:
				run_optics = st.button( 'Run OPTICS Clustering', icon='🏃',
					key='cluster_optics_run', use_container_width=True )
			
			optics_reset_keys = [ 'cluster_optics_min_samples', 'cluster_optics_max_eps',
				'cluster_optics_metric', 'cluster_optics_p', 'cluster_optics_cluster_method',
				'cluster_optics_eps', 'cluster_optics_xi', 'cluster_optics_predecessor_correction',
				'cluster_optics_min_cluster_size', 'cluster_optics_algorithm',
				'cluster_optics_leaf_size', 'cluster_optics_n_jobs', 'df_cluster_optics_results',
				'df_cluster_optics_counts', 'df_cluster_optics_metrics',
				'df_cluster_optics_centroids', 'df_cluster_optics_details',
				'df_cluster_optics_ordering', 'cluster_optics_plot_features',
				'cluster_optics_signature' ]
			
			with op_b2:
				st.button( 'Reset OPTICS Clustering', icon='🔁', key='cluster_optics_reset',
					use_container_width=True, on_click=clear_keys, args=(optics_reset_keys,) )
			
			if run_optics:
				try:
					if optics_min_samples > len( df_cluster_input ):
						st.warning( '⚠️ Minimum Samples cannot exceed the number of samples.' )
						st.stop( )
					
					if (optics_min_cluster_size_input > 0 and optics_min_cluster_size_input > len(
							df_cluster_input )):
						st.warning( '⚠️ Minimum Cluster Size cannot exceed the number of '
						            'samples.' )
						st.stop( )
					
					optics_max_eps = (
						np.inf if optics_max_eps_input == 0.0 else optics_max_eps_input)
					
					optics_min_cluster_size = (
						None if optics_min_cluster_size_input == 0 else
						optics_min_cluster_size_input)
					
					optics_effective_eps = (
						optics_eps if optics_cluster_method == 'dbscan' else None)
					
					model_parameters = { 'Model': 'OPTICS', 'min_samples': optics_min_samples,
						'max_eps': optics_max_eps, 'metric': optics_metric, 'p': optics_p,
						'cluster_method': optics_cluster_method, 'eps': optics_effective_eps,
						'xi': optics_xi, 'predecessor_correction': optics_predecessor_correction,
						'min_cluster_size': optics_min_cluster_size, 'algorithm': optics_algorithm,
						'leaf_size': optics_leaf_size, 'n_jobs': optics_n_jobs }
					
					cluster_optics_signature = ((tuple( active_features ), cluster_data_signature),
						'OPTICS', tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = OPTICS( samples=optics_min_samples, max_eps=optics_max_eps,
						metric=optics_metric, algorithm=optics_algorithm,
						leaf_size=optics_leaf_size, eps=optics_effective_eps,
						predecessor_correction=optics_predecessor_correction,
						min_cluster_size=optics_min_cluster_size, p=optics_p, metric_params=None,
						cluster_method=optics_cluster_method, xi=optics_xi, memory=None,
						n_jobs=optics_n_jobs )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					df_metrics = model.score( X )
					if df_metrics is None:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					unique_labels = np.unique( labels )
					cluster_labels = unique_labels[ unique_labels != -1 ]
					noise_count = int( np.sum( labels == -1 ) )
					detail_rows = [ { 'Property': 'features', 'Value': int( model.features ) },
						{ 'Property': 'clusters', 'Value': int( len( cluster_labels ) ) },
						{ 'Property': 'noise_samples', 'Value': noise_count },
						{ 'Property': 'min_samples', 'Value': model.min_samples },
						{ 'Property': 'max_eps', 'Value': model.max_eps },
						{ 'Property': 'metric', 'Value': model.metric },
						{ 'Property': 'p', 'Value': model.p },
						{ 'Property': 'cluster_method', 'Value': model.cluster_method },
						{ 'Property': 'eps', 'Value': model.eps },
						{ 'Property': 'xi', 'Value': model.xi },
						{ 'Property': 'predecessor_correction',
							'Value': model.predecessor_correction },
						{ 'Property': 'min_cluster_size', 'Value': model.min_cluster_size },
						{ 'Property': 'algorithm', 'Value': model.algorithm },
						{ 'Property': 'leaf_size', 'Value': model.leaf_size },
						{ 'Property': 'n_jobs', 'Value': model.n_jobs } ]
					
					df_details = pd.DataFrame( detail_rows )
					df_ordering = pd.DataFrame(
						{ 'Sample': np.arange( len( model.ordering ) ), 'Ordering': model.ordering,
							'Reachability': model.reachability,
							'Core Distance': model.core_distances,
							'Predecessor': model.predecessor } )
					
					df_centroids = pd.DataFrame( )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_optics_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_optics_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_optics_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_optics_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_optics_details' ] = df_details.copy( )
					st.session_state[ 'df_cluster_optics_ordering' ] = df_ordering.copy( )
					st.session_state[ 'cluster_optics_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_optics_signature' ] = cluster_optics_signature
					st.success( 'OPTICS clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_optics_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_details' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_optics_ordering' ] = pd.DataFrame( )
					st.session_state[ 'cluster_optics_plot_features' ] = [ ]
					st.session_state[ 'cluster_optics_signature' ]=None
					st.error( f'OPTICS clustering failed: {ex}' )
			
			optics_signature = st.session_state.get( 'cluster_optics_signature', None )
			if (isinstance( optics_signature, tuple ) and len( optics_signature ) > 1 and
					optics_signature[ 1 ] == 'OPTICS'):
				df_results = st.session_state.get( 'df_cluster_optics_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_optics_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_optics_metrics', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_optics_details', pd.DataFrame( ) )
				df_ordering = st.session_state.get( 'df_cluster_optics_ordering', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_details = pd.DataFrame( )
				df_ordering = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				render_data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					render_data_editor( df_metrics, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True )
				
				if not df_ordering.empty:
					st.caption( 'Reachability and Ordering' )
					render_data_editor( df_ordering, use_container_width=True )
			else:
				st.info( 'Run OPTICS Clustering to view cluster counts, noise counts, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					figure = create_cluster_figure( df_results, feature_columns,
						'OPTICS Cluster Assignments' )
					render_mathy_plotly_chart( figure, 'cluster_optics_assignment_chart',
						'mathy_optics_clusters', 'OPTICS Cluster Assignments', 560 )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run OPTICS Clustering to view the scatter plot.' )
		
		if 'cluster_mean_shift_bandwidth' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_bandwidth' ] = 0.0
		
		if 'cluster_mean_shift_bin_seeding' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_bin_seeding' ] = False
		
		if 'cluster_mean_shift_min_bin_freq' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_min_bin_freq' ] = 1
		
		if 'cluster_mean_shift_cluster_all' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_cluster_all' ] = True
		
		if 'cluster_mean_shift_n_jobs' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_n_jobs' ] = 1
		
		if 'cluster_mean_shift_max_iter' not in st.session_state:
			st.session_state[ 'cluster_mean_shift_max_iter' ] = 300
		
		with st.expander( 'Mean Shift Clustering', expanded=False ):
			st.caption(
				'Centroid-based clustering that discovers cluster centers from the density of the '
				'data.' )
			
			ms_c1, ms_c2, ms_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with ms_c1:
				mean_shift_bandwidth_input = float(
					st.number_input( 'Bandwidth (0 = Automatic)', min_value=0.0, step=0.1,
						format='%.4f', key='cluster_mean_shift_bandwidth' ) )
				
				mean_shift_bin_seeding = st.checkbox( 'Use Bin Seeding',
					key='cluster_mean_shift_bin_seeding' )
			
			with ms_c2:
				mean_shift_min_bin_freq = int( st.number_input( 'Minimum Bin Frequency',
					min_value=1, step=1, key='cluster_mean_shift_min_bin_freq' ) )
				
				mean_shift_cluster_all = st.checkbox( 'Assign All Samples',
					key='cluster_mean_shift_cluster_all' )
			
			with ms_c3:
				mean_shift_n_jobs = int( st.number_input( 'Parallel Jobs', min_value=-1, step=1,
					key='cluster_mean_shift_n_jobs' ) )
				
				mean_shift_max_iter = int( st.number_input( 'Maximum Iterations', min_value=1,
					step=1, key='cluster_mean_shift_max_iter' ) )
			
			ms_b1, ms_b2 = st.columns( 2 )
			
			with ms_b1:
				run_mean_shift = st.button( 'Run Mean Shift Clustering', icon='🏃',
					key='cluster_mean_shift_run', use_container_width=True )
			
			mean_shift_reset_keys = [ 'cluster_mean_shift_bandwidth',
				'cluster_mean_shift_bin_seeding', 'cluster_mean_shift_min_bin_freq',
				'cluster_mean_shift_cluster_all', 'cluster_mean_shift_n_jobs',
				'cluster_mean_shift_max_iter', 'df_cluster_mean_shift_results',
				'df_cluster_mean_shift_counts', 'df_cluster_mean_shift_metrics',
				'df_cluster_mean_shift_centroids', 'df_cluster_mean_shift_details',
				'cluster_mean_shift_plot_features', 'cluster_mean_shift_signature' ]
			
			with ms_b2:
				st.button( 'Reset Mean Shift Clustering', icon='🔁',
					key='cluster_mean_shift_reset',
					use_container_width=True, on_click=clear_keys, args=(mean_shift_reset_keys,) )
			
			if run_mean_shift:
				try:
					mean_shift_bandwidth = (
						None if mean_shift_bandwidth_input == 0.0 else mean_shift_bandwidth_input)
					
					model_parameters = { 'Model': 'MeanShift', 'bandwidth': mean_shift_bandwidth,
						'bin_seeding': mean_shift_bin_seeding,
						'min_bin_freq': mean_shift_min_bin_freq,
						'cluster_all': mean_shift_cluster_all, 'n_jobs': mean_shift_n_jobs,
						'max_iter': mean_shift_max_iter }
					
					cluster_mean_shift_signature = (
						(tuple( active_features ), cluster_data_signature), 'MeanShift',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = MeanShift( min_bin=mean_shift_min_bin_freq,
						group_all=mean_shift_cluster_all, bandwidth=mean_shift_bandwidth,
						seeds=None, bin_seeding=mean_shift_bin_seeding, n_jobs=mean_shift_n_jobs,
						max_iter=mean_shift_max_iter, min_bin_freq=mean_shift_min_bin_freq,
						cluster_all=mean_shift_cluster_all )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					df_metrics = model.score( X )
					if df_metrics is None:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					centroids = np.asarray( model.centroids_ )
					df_centroids = pd.DataFrame( centroids, columns=feature_columns )
					df_centroids.insert( 0, 'Cluster', np.arange( len( df_centroids ) ) )
					detail_rows = [ { 'Property': 'features', 'Value': int( model.features ) },
						{ 'Property': 'clusters', 'Value': int( len( centroids ) ) },
						{ 'Property': 'bandwidth', 'Value': model.bandwidth },
						{ 'Property': 'bin_seeding', 'Value': model.bin_seeding },
						{ 'Property': 'min_bin_freq', 'Value': model.min_bin_freq },
						{ 'Property': 'cluster_all', 'Value': model.cluster_all },
						{ 'Property': 'n_jobs', 'Value': model.n_jobs },
						{ 'Property': 'max_iter', 'Value': model.max_iter },
						{ 'Property': 'iterations', 'Value': model.iterations } ]
					
					df_details = pd.DataFrame( detail_rows )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_mean_shift_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_mean_shift_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_mean_shift_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_mean_shift_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_mean_shift_details' ] = df_details.copy( )
					st.session_state[ 'cluster_mean_shift_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_mean_shift_signature' ] = cluster_mean_shift_signature
					st.success( 'Mean Shift clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_mean_shift_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_mean_shift_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_mean_shift_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_mean_shift_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_mean_shift_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_mean_shift_plot_features' ] = [ ]
					st.session_state[ 'cluster_mean_shift_signature' ]=None
					st.error( f'Mean Shift clustering failed: {ex}' )
			
			mean_shift_signature = st.session_state.get( 'cluster_mean_shift_signature', None )
			if (isinstance( mean_shift_signature, tuple ) and len( mean_shift_signature ) > 1 and
					mean_shift_signature[ 1 ] == 'MeanShift'):
				df_results = st.session_state.get( 'df_cluster_mean_shift_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_mean_shift_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_mean_shift_metrics', pd.DataFrame( ) )
				df_centroids = st.session_state.get( 'df_cluster_mean_shift_centroids', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_mean_shift_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_centroids = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				render_data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					render_data_editor( df_metrics, use_container_width=True )
				
				if not df_centroids.empty:
					st.caption( 'Cluster Centroids' )
					render_data_editor( df_centroids, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True )
			else:
				st.info(
					'Run Mean Shift Clustering to view cluster counts, centroids, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					figure = create_cluster_figure( df_results, feature_columns,
						'Mean Shift Cluster Assignments', df_centroids )
					render_mathy_plotly_chart( figure, 'cluster_mean_shift_assignment_chart',
						'mathy_mean_shift_clusters', 'Mean Shift Cluster Assignments', 560 )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Mean Shift Clustering to view the scatter plot.' )
		
		with st.expander( 'Affinity Propagation Clustering', expanded=False ):
			affinity_defaults = { 'cluster_affinity_damping': 0.5, 'cluster_affinity_max_iter': 200,
				'cluster_affinity_convergence_iter': 15, 'cluster_affinity_use_preference': False,
				'cluster_affinity_preference': 0.0, 'cluster_affinity_copy': True,
				'cluster_affinity_verbose': False, 'cluster_affinity_random_state': 42 }
			
			for key, value in affinity_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			st.caption(
				'Exemplar-based clustering that determines cluster representatives from pairwise '
				'similarities.' )
			
			ap_c1, ap_c2, ap_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with ap_c1:
				affinity_damping = float( st.number_input( 'Damping', min_value=0.5,
					max_value=0.9999, step=0.01, format='%.4f', key='cluster_affinity_damping' ) )
				
				affinity_max_iter = int( st.number_input( 'Maximum Iterations', min_value=1,
					step=1, key='cluster_affinity_max_iter' ) )
			
			with ap_c2:
				affinity_convergence_iter = int( st.number_input( 'Convergence Iterations',
					min_value=1, step=1, key='cluster_affinity_convergence_iter' ) )
				
				affinity_copy = st.checkbox( 'Copy Input Data', key='cluster_affinity_copy' )
			
			with ap_c3:
				affinity_use_preference = st.checkbox( 'Use Custom Preference',
					key='cluster_affinity_use_preference' )
				
				affinity_preference_input = float( st.number_input( 'Preference', step=0.1,
					format='%.4f', disabled=not affinity_use_preference,
					key='cluster_affinity_preference' ) )
				
				affinity_verbose = st.checkbox( 'Verbose', key='cluster_affinity_verbose' )
				affinity_random_state = int( st.number_input( 'Random State', min_value=0, step=1,
					key='cluster_affinity_random_state' ) )
			
			affinity_reset_keys = list( affinity_defaults.keys( ) ) + [
				'df_cluster_affinity_results', 'df_cluster_affinity_counts',
				'df_cluster_affinity_metrics', 'df_cluster_affinity_centroids',
				'df_cluster_affinity_details', 'cluster_affinity_plot_features',
				'cluster_affinity_signature' ]
			
			ap_b1, ap_b2 = st.columns( 2 )
			with ap_b1:
				run_affinity = st.button( 'Run Affinity Propagation', icon='🏃',
					key='cluster_affinity_run', use_container_width=True )
			
			with ap_b2:
				st.button( 'Reset Affinity Propagation', icon='🔁', key='cluster_affinity_reset',
					use_container_width=True, on_click=clear_keys, args=(affinity_reset_keys,) )
			
			if run_affinity:
				try:
					if affinity_convergence_iter > affinity_max_iter:
						st.warning( '⚠️ Convergence Iterations cannot exceed Maximum Iterations.' )
						st.stop( )
					
					affinity_preference = ( affinity_preference_input if affinity_use_preference else None)
					model_parameters = { 'Model': 'AffinityPropagation',
						'damping': affinity_damping, 'max_iter': affinity_max_iter,
						'convergence_iter': affinity_convergence_iter,
						'preference': affinity_preference, 'affinity': 'euclidean',
						'copy': affinity_copy, 'verbose': affinity_verbose,
						'random_state': affinity_random_state }
					
					cluster_affinity_signature = (
						(tuple( active_features ), cluster_data_signature), 'AffinityPropagation',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = AffinityPropagation( damping=affinity_damping,
						max_iter=affinity_max_iter, convergence_iter=affinity_convergence_iter,
						preference=affinity_preference, affinity='euclidean', copy=affinity_copy,
						verbose=affinity_verbose, random_state=affinity_random_state )
					
					start_time = time.time( )
					labels = model.project( X )
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts(
						dropna=False ).rename_axis( 'Cluster' ).reset_index(
						name='Count' ).sort_values( by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					centroids = np.asarray( model.centroids_ )
					df_centroids = pd.DataFrame( centroids, columns=feature_columns )
					df_centroids.insert( 0, 'Cluster', range( len( df_centroids ) ) )
					
					detail_rows = [ { 'Property': 'features', 'Value': model.features },
						{ 'Property': 'clusters', 'Value': len( df_centroids ) },
						{ 'Property': 'iterations', 'Value': model.iterations },
						{ 'Property': 'damping', 'Value': model.damping },
						{ 'Property': 'max_iter', 'Value': model.max_iter },
						{ 'Property': 'convergence_iter', 'Value': model.convergence_iter },
						{ 'Property': 'preference', 'Value': model.preference },
						{ 'Property': 'affinity', 'Value': model.affinity },
						{ 'Property': 'copy', 'Value': model.copy },
						{ 'Property': 'verbose', 'Value': model.verbose },
						{ 'Property': 'random_state', 'Value': model.random_state } ]
					
					df_details = pd.DataFrame( detail_rows )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_affinity_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_affinity_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_affinity_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_affinity_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_affinity_details' ] = df_details.copy( )
					st.session_state[ 'cluster_affinity_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_affinity_signature' ] = cluster_affinity_signature
					st.success( 'Affinity Propagation clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_affinity_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_affinity_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_affinity_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_affinity_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_affinity_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_affinity_plot_features' ] = [ ]
					st.session_state[ 'cluster_affinity_signature' ]=None
					st.error( f'Affinity Propagation clustering failed: {ex}' )
			
			affinity_signature = st.session_state.get( 'cluster_affinity_signature', None )
			if (isinstance( affinity_signature, tuple ) and len( affinity_signature ) > 1 and
					affinity_signature[ 1 ] == 'AffinityPropagation'):
				df_results = st.session_state.get( 'df_cluster_affinity_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_affinity_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_affinity_metrics', pd.DataFrame( ) )
				df_centroids = st.session_state.get( 'df_cluster_affinity_centroids', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_affinity_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_centroids = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				render_data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					render_data_editor( df_metrics, use_container_width=True )
				
				if not df_centroids.empty:
					st.caption( 'Cluster Exemplars' )
					render_data_editor( df_centroids, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True )
			else:
				st.info(
					'Run Affinity Propagation to view cluster counts, exemplars, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					figure = create_cluster_figure( df_results, feature_columns,
						'Affinity Propagation Cluster Assignments', df_centroids, 'Exemplars' )
					render_mathy_plotly_chart( figure, 'cluster_affinity_assignment_chart',
						'mathy_affinity_clusters', 'Affinity Propagation Cluster Assignments', 560 )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Affinity Propagation to view the scatter plot.' )
		
		cluster_max_clusters = max( 2, len( df_cluster_input ) - 1 )
		if 'cluster_birch_n_clusters' in st.session_state:
			st.session_state[ 'cluster_birch_n_clusters' ] = min(
				max( 2, int( st.session_state[ 'cluster_birch_n_clusters' ] ) ),
				cluster_max_clusters )
		
		with st.expander( 'Birch Clustering', expanded=False ):
			birch_defaults = { 'cluster_birch_threshold': 0.5, 'cluster_birch_branching_factor':
				50,
				'cluster_birch_use_global_clusters': True, 'cluster_birch_n_clusters': 3,
				'cluster_birch_compute_labels': True }
			
			for key, value in birch_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			st.caption( 'Scalable hierarchical clustering using a clustering-feature tree.' )
			
			br_c1, br_c2, br_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with br_c1:
				birch_threshold = float(
					st.number_input( 'Threshold', min_value=0.0001, step=0.1, format='%.4f',
						key='cluster_birch_threshold' ) )
			
			with br_c2:
				birch_branching_factor = int(
					st.number_input( 'Branching Factor', min_value=2, step=1,
						key='cluster_birch_branching_factor' ) )
				
				birch_compute_labels = st.checkbox( 'Compute Labels',
					key='cluster_birch_compute_labels' )
			
			with br_c3:
				birch_use_global_clusters = st.checkbox( 'Apply Global Clustering',
					key='cluster_birch_use_global_clusters' )
				
				birch_n_clusters = int( st.number_input( 'Number of Clusters', min_value=2,
					max_value=max( 2, len( df_cluster_input ) - 1 ), step=1,
					disabled=not birch_use_global_clusters, key='cluster_birch_n_clusters' ) )
			
			br_b1, br_b2 = st.columns( 2 )
			with br_b1:
				run_birch = st.button( 'Run Birch Clustering', icon='🏃', key='cluster_birch_run',
					use_container_width=True )
			
			birch_reset_keys = [ 'cluster_birch_threshold', 'cluster_birch_branching_factor',
				'cluster_birch_use_global_clusters', 'cluster_birch_n_clusters',
				'cluster_birch_compute_labels', 'df_cluster_birch_results',
				'df_cluster_birch_counts', 'df_cluster_birch_metrics',
				'df_cluster_birch_centroids',
				'df_cluster_birch_details', 'cluster_birch_plot_features',
				'cluster_birch_signature' ]
			
			with br_b2:
				st.button( 'Reset Birch Clustering', icon='🔁', key='cluster_birch_reset',
					use_container_width=True, on_click=clear_keys, args=(birch_reset_keys,) )
			
			if run_birch:
				try:
					if (birch_use_global_clusters and birch_n_clusters >= len( df_cluster_input )):
						st.warning(
							'⚠️ Number of Clusters must be less than the number of samples.' )
						st.stop( )
					
					effective_clusters = (birch_n_clusters if birch_use_global_clusters else None)
					model_parameters = { 'Model': 'Birch', 'threshold': birch_threshold,
						'branching_factor': birch_branching_factor,
						'n_clusters': effective_clusters, 'compute_labels': birch_compute_labels }
					
					cluster_birch_signature = ((tuple( active_features ), cluster_data_signature),
						'Birch',
						tuple( (key, str( value )) for key, value in model_parameters.items( ) ))
					
					model = Birch( threshold=birch_threshold, branching_factor=birch_branching_factor,
						n_clusters=effective_clusters, compute_labels=birch_compute_labels )
					
					start_time = time.time( )
					if birch_compute_labels:
						labels = model.project( X )
					else:
						model.train( X )
						labels = model.predict( X )
					
					elapsed_seconds = time.time( ) - start_time
					df_results = df_cluster_input.copy( )
					df_results[ 'Cluster' ] = labels
					df_counts = (df_results[ 'Cluster' ].value_counts( dropna=False ).rename_axis(
						'Cluster' ).reset_index( name='Count' ).sort_values(
						by='Cluster' ).reset_index( drop=True ))
					
					try:
						df_metrics = model.score( X )
						if df_metrics is None:
							df_metrics = pd.DataFrame( )
					except Exception:
						df_metrics = pd.DataFrame( )
					
					if df_metrics.empty:
						df_metrics = pd.DataFrame(
							[ { 'Processing Time (sec)': round( elapsed_seconds, 4 ) } ] )
					else:
						df_metrics = df_metrics.copy( )
						df_metrics[ 'Processing Time (sec)' ] = round( elapsed_seconds, 4 )
					
					subcluster_centers = np.asarray( model.subcluster_centers )
					df_centroids = pd.DataFrame( subcluster_centers, columns=feature_columns )
					df_centroids.insert( 0, 'Subcluster', range( len( df_centroids ) ) )
					try:
						subcluster_labels = np.asarray( model.subcluster_labels ).reshape( -1 )
						if len( subcluster_labels ) == len( df_centroids ):
							df_centroids.insert( 1, 'Cluster', subcluster_labels )
					except Exception:
						pass
					
					detail_rows = [ { 'Property': 'features', 'Value': model.features },
						{ 'Property': 'clusters', 'Value': len( np.unique( labels ) ) },
						{ 'Property': 'subclusters', 'Value': len( df_centroids ) },
						{ 'Property': 'threshold', 'Value': model.threshold },
						{ 'Property': 'branching_factor', 'Value': model.branching_factor },
						{ 'Property': 'n_clusters', 'Value': model.n_clusters },
						{ 'Property': 'compute_labels', 'Value': model.compute_labels } ]
					
					df_details = pd.DataFrame( detail_rows )
					st.session_state[ 'model' ] = model
					st.session_state[ 'df_cluster_birch_results' ] = df_results.copy( )
					st.session_state[ 'df_cluster_birch_counts' ] = df_counts.copy( )
					st.session_state[ 'df_cluster_birch_metrics' ] = df_metrics.copy( )
					st.session_state[ 'df_cluster_birch_centroids' ] = df_centroids.copy( )
					st.session_state[ 'df_cluster_birch_details' ] = df_details.copy( )
					st.session_state[ 'cluster_birch_plot_features' ] = feature_columns.copy( )
					st.session_state[ 'cluster_birch_signature' ] = cluster_birch_signature
					st.success( 'Birch clustering complete.' )
				except Exception as ex:
					st.session_state[ 'df_cluster_birch_results' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_birch_counts' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_birch_metrics' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_birch_centroids' ] = pd.DataFrame( )
					st.session_state[ 'df_cluster_birch_details' ] = pd.DataFrame( )
					st.session_state[ 'cluster_birch_plot_features' ] = [ ]
					st.session_state[ 'cluster_birch_signature' ]=None
					st.error( f'Birch clustering failed: {ex}' )
			
			birch_signature = st.session_state.get( 'cluster_birch_signature', None )
			if (isinstance( birch_signature, tuple ) and len( birch_signature ) > 1 and
					birch_signature[ 1 ] == 'Birch'):
				df_results = st.session_state.get( 'df_cluster_birch_results', pd.DataFrame( ) )
				df_counts = st.session_state.get( 'df_cluster_birch_counts', pd.DataFrame( ) )
				df_metrics = st.session_state.get( 'df_cluster_birch_metrics', pd.DataFrame( ) )
				df_centroids = st.session_state.get( 'df_cluster_birch_centroids', pd.DataFrame( ) )
				df_details = st.session_state.get( 'df_cluster_birch_details', pd.DataFrame( ) )
			else:
				df_results = pd.DataFrame( )
				df_counts = pd.DataFrame( )
				df_metrics = pd.DataFrame( )
				df_centroids = pd.DataFrame( )
				df_details = pd.DataFrame( )
			
			# ------------------------------------------------------------------
			# CLUSTER SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Cluster Summary' )
			
			if not df_counts.empty:
				render_data_editor( df_counts, use_container_width=True )
				
				if not df_metrics.empty:
					st.caption( 'Metrics' )
					render_data_editor( df_metrics, use_container_width=True )
				
				if not df_centroids.empty:
					st.caption( 'Subcluster Centers' )
					render_data_editor( df_centroids, use_container_width=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True )
			else:
				st.info( 'Run Birch Clustering to view cluster counts, subclusters, and metrics.' )
			
			# ------------------------------------------------------------------
			# VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.subheader( 'Cluster Visualization' )
			
			if not df_results.empty:
				if len( feature_columns ) == 2:
					figure = create_cluster_figure( df_results, feature_columns,
						'Birch Cluster Assignments', df_centroids, 'Subcluster Centers' )
					render_mathy_plotly_chart( figure, 'cluster_birch_assignment_chart',
						'mathy_birch_clusters', 'Birch Cluster Assignments', 560 )
				else:
					st.info( 'Visualization limited to two features.' )
			else:
				st.info( 'Run Birch Clustering to view the scatter plot.' )

# ============================================
# TIME SERIES MODE
# ============================================
elif mode == 'Time-Series Models':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Time-Series Models' ] )
		st.divider( )
		
		# ------------------------------------------------------------------
		# TIME-SERIES INPUT
		# ------------------------------------------------------------------
		df_dataset = get_analysis_dataset( )
		if df_dataset is None or df_dataset.empty:
			st.warning( '⚠️ No dataset loaded.' )
			st.stop( )
		
		numeric_columns = st.session_state.get( 'numeric_columns', [ ] ).copy( )
		
		if not numeric_columns:
			st.warning( '⚠️ No numeric columns available for time-series analysis.' )
			st.stop( )
		
		timeseries_source_signature = (tuple( df_dataset.columns.tolist( ) ),
			tuple( str( dtype ) for dtype in df_dataset.dtypes.tolist( ) ),
			int( len( df_dataset ) ),
			int( pd.util.hash_pandas_object( df_dataset, index=True ).sum( ) ))
		
		prior_timeseries_source_signature = st.session_state.get( 'timeseries_source_signature',
			None )
		
		if ( prior_timeseries_source_signature is not None
				and prior_timeseries_source_signature != timeseries_source_signature):
			timeseries_keys = \
			[ key for key in list( st.session_state.keys( ) ) if key.startswith( 'timeseries_' ) ]
			clear_keys( timeseries_keys )
		
		st.session_state[ 'timeseries_source_signature' ] = timeseries_source_signature
		timeseries_defaults = { 'timeseries_col_box': numeric_columns[ 0 ] }
		for key, value in timeseries_defaults.items( ):
			if key not in st.session_state:
				st.session_state[ key ] = value
		
		if st.session_state[ 'timeseries_col_box' ] not in numeric_columns:
			st.session_state[ 'timeseries_col_box' ] = numeric_columns[ 0 ]
		
		st.markdown( '##### Time-Series Selection' )
		series_col = st.selectbox( 'Select Numeric Time-Series Column', numeric_columns,
			key='timeseries_col_box' )
		
		series_values = pd.to_numeric( df_dataset[ series_col ], errors='coerce' )
		series_values = series_values.replace( [ np.inf, -np.inf ], np.nan )
		invalid_observations = int( series_values.isna( ).sum( ) )
		if invalid_observations > 0:
			st.warning( f'⚠️ The selected series contains {invalid_observations:,} missing, '
			            'infinite, or nonnumeric observation(s). Resolve them before forecasting '
			            'to preserve temporal spacing.' )
			st.stop( )
		
		series = series_values.to_numpy( dtype=float )
		if series.ndim != 1:
			st.warning( '⚠️ The selected column could not be converted to a one-dimensional series.' )
			st.stop( )
		
		if len( series ) < 10:
			st.warning( '⚠️ Selected series must contain at least 10 valid observations.' )
			st.stop( )
		
		series_signature = int( pd.util.hash_pandas_object( series_values, index=True ).sum( ) )
		st.session_state[ 'timeseries_column' ] = series_col
		st.session_state[ 'timeseries_series' ] = series.copy( )
		df_timeseries_summary = pd.DataFrame(
			[ { 'Series': series_col, 'Observations': len( series ),
				'Minimum': float( np.min( series ) ), 'Maximum': float( np.max( series ) ),
				'Mean': float( np.mean( series ) ),
				'Standard Deviation': float( np.std( series ) ) } ] )
		
		render_data_editor( df_timeseries_summary, use_container_width=True, hide_index=True,
			disabled=True )
		
		# ------------------------------------------------------------------
		# LAGGED LINEAR REGRESSION
		# ------------------------------------------------------------------
		with st.expander( 'Lagged Linear Regression', expanded=False ):
			lag_linear_defaults = { 'timeseries_lag_linear_lag': 5,
				'timeseries_lag_linear_horizon': 5 }
			
			for key, value in lag_linear_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			maximum_lag = max( 1, len( series ) - 1 )
			st.session_state[ 'timeseries_lag_linear_lag' ] = min(
				int( st.session_state[ 'timeseries_lag_linear_lag' ] ), maximum_lag )
			
			st.caption( 'Linear autoregressive forecasting using lagged observations as predictors.' )
			ll_c1, ll_c2 = st.columns( 2, border=True )
			with ll_c1:
				lag_linear_order = int( st.number_input( 'Lag Order', min_value=1,
					max_value=max( 1, len( series ) - 1 ), step=1,
					key='timeseries_lag_linear_lag' ) )
			
			with ll_c2:
				lag_linear_horizon = int( st.number_input( 'Forecast Horizon', min_value=1, step=1,
					key='timeseries_lag_linear_horizon' ) )
			
			lag_linear_reset_keys = list( lag_linear_defaults.keys( ) ) + [
				'timeseries_lag_linear_model', 'timeseries_lag_linear_forecast',
				'timeseries_lag_linear_metrics', 'timeseries_lag_linear_results',
				'timeseries_lag_linear_signature' ]
			
			ll_b1, ll_b2 = st.columns( 2 )
			with ll_b1:
				run_lag_linear = st.button( 'Run Lagged Linear Regression', icon='🏃',
					key='timeseries_lag_linear_run', use_container_width=True )
			
			with ll_b2:
				st.button( 'Reset Lagged Linear Regression', icon='🔁',
					key='timeseries_lag_linear_reset', use_container_width=True,
					on_click=clear_keys, args=(lag_linear_reset_keys,) )
			
			if run_lag_linear:
				try:
					if lag_linear_order >= len( series ):
						st.warning( '⚠️ Lag Order must be less than the number of observations.' )
						st.stop( )
					
					model = LaggingSeries( lag=lag_linear_order )
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=lag_linear_horizon )
					elapsed_seconds = time.time( ) - start_time
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index,
						'Forecast': np.asarray( forecast, dtype=float ) } )
					
					lag_linear_signature = (series_col, len( series ), series_signature,
						lag_linear_order, lag_linear_horizon)
					
					st.session_state[ 'timeseries_lag_linear_model' ] = model
					st.session_state[ 'timeseries_lag_linear_forecast' ] = np.asarray( forecast,
						dtype=float )
					st.session_state[ 'timeseries_lag_linear_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_lag_linear_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_lag_linear_signature' ] = lag_linear_signature
					st.success( 'Lagged Linear Regression forecasting complete.' )
				except Exception as ex:
					st.session_state.pop( 'timeseries_lag_linear_model', None )
					st.session_state.pop( 'timeseries_lag_linear_forecast', None )
					st.session_state.pop( 'timeseries_lag_linear_metrics', None )
					st.session_state.pop( 'timeseries_lag_linear_results', None )
					st.session_state.pop( 'timeseries_lag_linear_signature', None )
					st.error( f'Lagged Linear Regression failed: {ex}' )
			
			lag_linear_signature = st.session_state.get( 'timeseries_lag_linear_signature', None )
			current_lag_linear_signature = (series_col, len( series ), series_signature,
				lag_linear_order, lag_linear_horizon)
			
			if lag_linear_signature == current_lag_linear_signature:
				df_metrics = st.session_state.get( 'timeseries_lag_linear_metrics',
					pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_lag_linear_results',
					pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_lag_linear_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				render_data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run Lagged Linear Regression to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				render_data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run Lagged Linear Regression to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			if len( forecast ) > 0:
				figure = create_forecast_figure( series, forecast, series_col,
					'Lagged Linear Regression Forecast' )
				render_mathy_plotly_chart( figure, 'timeseries_lag_linear_forecast_chart',
					'mathy_lag_linear_forecast', 'Lagged Linear Regression Forecast', 560 )
			else:
				st.info( 'Run Lagged Linear Regression to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# LAGGED BOOSTING REGRESSION
		# ------------------------------------------------------------------
		with st.expander( 'Lagged Boosting Regression', expanded=False ):
			lag_boost_defaults = { 'timeseries_lag_boost_lag': 12,
				'timeseries_lag_boost_loss': 'squared_error', 'timeseries_lag_boost_quantile': 0.5,
				'timeseries_lag_boost_rate': 0.1, 'timeseries_lag_boost_iters': 100,
				'timeseries_lag_boost_leaf_nodes': 31, 'timeseries_lag_boost_use_depth': False,
				'timeseries_lag_boost_depth': 3, 'timeseries_lag_boost_leaf': 20,
				'timeseries_lag_boost_regularization': 0.0, 'timeseries_lag_boost_features': 1.0,
				'timeseries_lag_boost_bins': 255, 'timeseries_lag_boost_warm': False,
				'timeseries_lag_boost_stopping': 'auto', 'timeseries_lag_boost_scoring': 'loss',
				'timeseries_lag_boost_validation': 0.1, 'timeseries_lag_boost_no_change': 10,
				'timeseries_lag_boost_tolerance': 0.0000001, 'timeseries_lag_boost_verbose': 0,
				'timeseries_lag_boost_random_state': 42, 'timeseries_lag_boost_horizon': 5 }
			
			for key, value in lag_boost_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			maximum_lag = max( 1, len( series ) - 1 )
			st.session_state[ 'timeseries_lag_boost_lag' ] = min(
				int( st.session_state[ 'timeseries_lag_boost_lag' ] ), maximum_lag )
			
			st.caption( 'Nonlinear autoregressive forecasting using histogram gradient boosting over '
				'lagged observations.' )
			
			lb_c1, lb_c2, lb_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with lb_c1:
				lag_boost_order = int( st.number_input( 'Lag Order', min_value=1,
					max_value=max( 1, len( series ) - 1 ), step=1,
					key='timeseries_lag_boost_lag' ) )
				
				lag_boost_loss = st.selectbox( 'Loss',
					options=[ 'squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile' ],
					key='timeseries_lag_boost_loss' )
				
				lag_boost_quantile = float(
					st.number_input( 'Quantile', min_value=0.0001, max_value=0.9999, step=0.05,
						format='%.4f', disabled=lag_boost_loss != 'quantile',
						key='timeseries_lag_boost_quantile' ) )
				
				lag_boost_rate = float(
					st.number_input( 'Learning Rate', min_value=0.0001, step=0.01, format='%.4f',
						key='timeseries_lag_boost_rate' ) )
				
				lag_boost_iters = int( st.number_input( 'Maximum Iterations', min_value=1, step=1,
					key='timeseries_lag_boost_iters' ) )
				
				lag_boost_leaf_nodes = int(
					st.number_input( 'Maximum Leaf Nodes', min_value=2, step=1,
						key='timeseries_lag_boost_leaf_nodes' ) )
				
				lag_boost_horizon = int( st.number_input( 'Forecast Horizon', min_value=1, step=1,
					key='timeseries_lag_boost_horizon' ) )
			
			with lb_c2:
				lag_boost_use_depth = st.checkbox( 'Limit Tree Depth',
					key='timeseries_lag_boost_use_depth' )
				
				lag_boost_depth_input = int( st.number_input( 'Maximum Depth', min_value=1, step=1,
					disabled=not lag_boost_use_depth, key='timeseries_lag_boost_depth' ) )
				
				lag_boost_leaf = int(
					st.number_input( 'Minimum Samples per Leaf', min_value=1, step=1,
						key='timeseries_lag_boost_leaf' ) )
				
				lag_boost_regularization = float(
					st.number_input( 'L2 Regularization', min_value=0.0, step=0.1, format='%.4f',
						key='timeseries_lag_boost_regularization' ) )
				
				lag_boost_features = float(
					st.number_input( 'Maximum Features', min_value=0.0001, max_value=1.0,
						step=0.05,
						format='%.4f', key='timeseries_lag_boost_features' ) )
				
				lag_boost_bins = int(
					st.number_input( 'Maximum Bins', min_value=2, max_value=255, step=1,
						key='timeseries_lag_boost_bins' ) )
				
				lag_boost_warm = st.checkbox( 'Warm Start', key='timeseries_lag_boost_warm' )
			
			with lb_c3:
				lag_boost_stopping = st.selectbox( 'Early Stopping',
					options=[ 'auto', True, False ], key='timeseries_lag_boost_stopping' )
				
				lag_boost_scoring = st.selectbox( 'Early-Stopping Scoring',
					options=[ 'loss', 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error' ],
					key='timeseries_lag_boost_scoring' )
				
				lag_boost_validation = float( st.number_input( 'Validation Fraction', min_value=0.01,
					max_value=0.99, step=0.01, format='%.4f', key='timeseries_lag_boost_validation' ) )
				
				lag_boost_no_change = int( st.number_input( 'Iterations Without Improvement',
					min_value=1, step=1, key='timeseries_lag_boost_no_change' ) )
				
				lag_boost_tolerance = float( st.number_input( 'Tolerance', min_value=0.0,
					step=0.0000001, format='%.8f', key='timeseries_lag_boost_tolerance' ) )
				
				lag_boost_verbose = int( st.number_input( 'Verbosity', min_value=0, step=1,
					key='timeseries_lag_boost_verbose' ) )
				
				lag_boost_random_state = int( st.number_input( 'Random State', min_value=0, step=1,
					key='timeseries_lag_boost_random_state' ) )
			
			lag_boost_reset_keys = list( lag_boost_defaults.keys( ) ) + [
				'timeseries_lag_boost_model', 'timeseries_lag_boost_forecast',
				'timeseries_lag_boost_metrics', 'timeseries_lag_boost_results',
				'timeseries_lag_boost_details', 'timeseries_lag_boost_signature' ]
			
			lb_b1, lb_b2 = st.columns( 2 )
			with lb_b1:
				run_lag_boost = st.button( 'Run Lagged Boosting Regression', icon='🏃',
					key='timeseries_lag_boost_run', use_container_width=True )
			
			with lb_b2:
				st.button( 'Reset Lagged Boosting Regression', icon='🔁',
					key='timeseries_lag_boost_reset', use_container_width=True,
					on_click=clear_keys,
					args=(lag_boost_reset_keys,) )
			
			if run_lag_boost:
				try:
					if lag_boost_order >= len( series ):
						st.warning( '⚠️ Lag Order must be less than the number of observations.' )
						st.stop( )
					
					if lag_boost_loss == 'quantile':
						quantile = lag_boost_quantile
					else:
						quantile = None
					
					if lag_boost_loss in [ 'gamma', 'poisson' ] and np.any( series <= 0 ):
						st.warning(
							'⚠️ Gamma and Poisson loss require strictly positive observations.' )
						st.stop( )
					
					if (lag_boost_stopping is not False and lag_boost_validation * (
							len( series ) - lag_boost_order) < 1):
						st.warning( '⚠️ Validation Fraction is too small for the available lagged '
						            'observations.' )
						st.stop( )
					
					max_depth = (lag_boost_depth_input if lag_boost_use_depth else None)
					model = LagBoostingSeries( lag=lag_boost_order, loss=lag_boost_loss,
						quantile=quantile, rate=lag_boost_rate, iters=lag_boost_iters,
						leaf_nodes=lag_boost_leaf_nodes, depth=max_depth, leaf=lag_boost_leaf,
						regularization=lag_boost_regularization, features=lag_boost_features,
						bins=lag_boost_bins, monotonic=None, interaction=None, warm=lag_boost_warm,
						stopping=lag_boost_stopping, scoring=lag_boost_scoring,
						validation=lag_boost_validation, no_change=lag_boost_no_change,
						tol=lag_boost_tolerance, verbose=lag_boost_verbose,
						rando=lag_boost_random_state )
					
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=lag_boost_horizon )
					elapsed_seconds = time.time( ) - start_time
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast = np.asarray( forecast, dtype=float ).reshape( -1 )
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index, 'Forecast': forecast } )
					detail_rows = [ { 'Property': 'Lag Order', 'Value': model.lag },
						{ 'Property': 'Loss', 'Value': model.loss },
						{ 'Property': 'Quantile', 'Value': model.quantile },
						{ 'Property': 'Learning Rate', 'Value': model.learning_rate },
						{ 'Property': 'Maximum Iterations', 'Value': model.max_iter },
						{ 'Property': 'Maximum Leaf Nodes', 'Value': model.max_leaf_nodes },
						{ 'Property': 'Maximum Depth', 'Value': model.max_depth },
						{ 'Property': 'Minimum Samples per Leaf', 'Value':
							model.min_samples_leaf },
						{ 'Property': 'L2 Regularization', 'Value': model.l2_regularization },
						{ 'Property': 'Maximum Features', 'Value': model.max_features },
						{ 'Property': 'Maximum Bins', 'Value': model.max_bins },
						{ 'Property': 'Warm Start', 'Value': model.warm_start },
						{ 'Property': 'Early Stopping', 'Value': model.early_stopping },
						{ 'Property': 'Scoring', 'Value': model.scoring },
						{ 'Property': 'Validation Fraction', 'Value': model.validation_fraction },
						{ 'Property': 'Iterations Without Improvement', 'Value': model.n_iter_no_change },
						{ 'Property': 'Tolerance', 'Value': model.tol },
						{ 'Property': 'Random State', 'Value': model.random_state } ]
					
					df_details = pd.DataFrame( detail_rows )
					lag_boost_signature = (series_col, len( series ), series_signature,
						lag_boost_order, lag_boost_loss, quantile, lag_boost_rate, lag_boost_iters,
						lag_boost_leaf_nodes, max_depth, lag_boost_leaf, lag_boost_regularization,
						lag_boost_features, lag_boost_bins, lag_boost_warm, lag_boost_stopping,
						lag_boost_scoring, lag_boost_validation, lag_boost_no_change,
						lag_boost_tolerance, lag_boost_verbose, lag_boost_random_state,
						lag_boost_horizon)
					
					st.session_state[ 'timeseries_lag_boost_model' ] = model
					st.session_state[ 'timeseries_lag_boost_forecast' ] = forecast.copy( )
					st.session_state[ 'timeseries_lag_boost_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_lag_boost_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_lag_boost_details' ] = df_details.copy( )
					st.session_state[ 'timeseries_lag_boost_signature' ] = lag_boost_signature
					st.success( 'Lagged Boosting Regression forecasting complete.' )
				except Exception as ex:
					for key in [ 'timeseries_lag_boost_model', 'timeseries_lag_boost_forecast',
						'timeseries_lag_boost_metrics', 'timeseries_lag_boost_results',
						'timeseries_lag_boost_details', 'timeseries_lag_boost_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'Lagged Boosting Regression failed: {ex}' )
			
			current_quantile = (lag_boost_quantile if lag_boost_loss == 'quantile' else None)
			current_max_depth = (lag_boost_depth_input if lag_boost_use_depth else None)
			current_lag_boost_signature = (series_col, len( series ), series_signature,
				lag_boost_order, lag_boost_loss, current_quantile, lag_boost_rate, lag_boost_iters,
				lag_boost_leaf_nodes, current_max_depth, lag_boost_leaf, lag_boost_regularization,
				lag_boost_features, lag_boost_bins, lag_boost_warm, lag_boost_stopping,
				lag_boost_scoring, lag_boost_validation, lag_boost_no_change, lag_boost_tolerance,
				lag_boost_verbose, lag_boost_random_state, lag_boost_horizon)
			
			lag_boost_signature = st.session_state.get( 'timeseries_lag_boost_signature', None )
			if lag_boost_signature == current_lag_boost_signature:
				df_metrics = st.session_state.get( 'timeseries_lag_boost_metrics', pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_lag_boost_results', pd.DataFrame( ) )
				df_details = st.session_state.get( 'timeseries_lag_boost_details', pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_lag_boost_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				df_details = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				render_data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Run Lagged Boosting Regression to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				render_data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run Lagged Boosting Regression to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			
			if len( forecast ) > 0:
				figure = create_forecast_figure( series, forecast, series_col,
					'Lagged Boosting Regression Forecast' )
				render_mathy_plotly_chart( figure, 'timeseries_lag_boost_forecast_chart',
					'mathy_lag_boost_forecast', 'Lagged Boosting Regression Forecast', 560 )
			else:
				st.info( 'Run Lagged Boosting Regression to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# LAGGED QUANTILE REGRESSION
		# ------------------------------------------------------------------
		with st.expander( 'Lagged Quantile Regression', expanded=False ):
			lag_quantile_defaults = { 'timeseries_lag_quantile_lag': 12,
				'timeseries_lag_quantile_quantile': 0.5, 'timeseries_lag_quantile_alpha': 1.0,
				'timeseries_lag_quantile_fit': True, 'timeseries_lag_quantile_solver': 'highs',
				'timeseries_lag_quantile_horizon': 5 }
			
			for key, value in lag_quantile_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			maximum_lag = max( 1, len( series ) - 1 )
			st.session_state[ 'timeseries_lag_quantile_lag' ] = min( int(
				st.session_state[ 'timeseries_lag_quantile_lag' ] ), maximum_lag )
			
			st.caption( 'Conditional quantile forecasting using linear regression over lagged '
			            'observations.' )
			
			lq_c1, lq_c2, lq_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with lq_c1:
				lag_quantile_order = int( st.number_input( 'Lag Order', min_value=1,
					max_value=max( 1, len( series ) - 1 ), step=1,
					key='timeseries_lag_quantile_lag' ) )
				
				lag_quantile_value = float(
					st.number_input( 'Conditional Quantile', min_value=0.0001, max_value=0.9999,
						step=0.05, format='%.4f', key='timeseries_lag_quantile_quantile' ) )
			
			with lq_c2:
				lag_quantile_alpha = float( st.number_input( 'L1 Regularization', min_value=0.0,
					step=0.1, format='%.4f', key='timeseries_lag_quantile_alpha' ) )
				
				lag_quantile_fit = st.checkbox( 'Fit Intercept',
					key='timeseries_lag_quantile_fit' )
			
			with lq_c3:
				lag_quantile_solver = st.selectbox( 'Solver',
					options=[ 'highs', 'highs-ds', 'highs-ipm', 'interior-point',
						'revised simplex' ], key='timeseries_lag_quantile_solver' )
				
				lag_quantile_horizon = int(
					st.number_input( 'Forecast Horizon', min_value=1, step=1,
						key='timeseries_lag_quantile_horizon' ) )
			
			lag_quantile_reset_keys = list( lag_quantile_defaults.keys( ) ) + [
				'timeseries_lag_quantile_model', 'timeseries_lag_quantile_forecast',
				'timeseries_lag_quantile_metrics', 'timeseries_lag_quantile_results',
				'timeseries_lag_quantile_details', 'timeseries_lag_quantile_signature' ]
			
			lq_b1, lq_b2 = st.columns( 2 )
			with lq_b1:
				run_lag_quantile = st.button( 'Run Lagged Quantile Regression', icon='🏃',
					key='timeseries_lag_quantile_run', use_container_width=True )
			
			with lq_b2:
				st.button( 'Reset Lagged Quantile Regression', icon='🔁',
					key='timeseries_lag_quantile_reset', use_container_width=True,
					on_click=clear_keys, args=(lag_quantile_reset_keys,) )
			
			if run_lag_quantile:
				try:
					if lag_quantile_order >= len( series ):
						st.warning( '⚠️ Lag Order must be less than the number of observations.' )
						st.stop( )
					
					if not 0.0 < lag_quantile_value < 1.0:
						st.warning( '⚠️ Conditional Quantile must be strictly between 0 and 1.' )
						st.stop( )
					
					if lag_quantile_alpha < 0.0:
						st.warning( '⚠️ L1 Regularization cannot be negative.' )
						st.stop( )
					
					model = LagQuantileSeries( lag=lag_quantile_order, quantile=lag_quantile_value,
						alpha=lag_quantile_alpha, fit=lag_quantile_fit, solver=lag_quantile_solver,
						solver_options=None )
					
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=lag_quantile_horizon )
					elapsed_seconds = time.time( ) - start_time
					if metrics is None:
						metrics = { }
					
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast = np.asarray( forecast, dtype=float ).reshape( -1 )
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index, 'Quantile': lag_quantile_value,
							'Forecast': forecast } )
					
					detail_rows = [ { 'Property': 'Lag Order', 'Value': model.lag },
						{ 'Property': 'Conditional Quantile', 'Value': model.quantile },
						{ 'Property': 'L1 Regularization', 'Value': model.alpha },
						{ 'Property': 'Fit Intercept', 'Value': model.fit_intercept },
						{ 'Property': 'Solver', 'Value': model.solver },
						{ 'Property': 'Training Samples', 'Value': len( model.training_values ) },
						{ 'Property': 'Forecast Horizon', 'Value': lag_quantile_horizon } ]
					
					df_details = pd.DataFrame( detail_rows )
					lag_quantile_signature = (series_col, len( series ), series_signature,
						lag_quantile_order, lag_quantile_value, lag_quantile_alpha,
						lag_quantile_fit, lag_quantile_solver, lag_quantile_horizon)
					
					st.session_state[ 'timeseries_lag_quantile_model' ] = model
					st.session_state[ 'timeseries_lag_quantile_forecast' ] = forecast.copy( )
					st.session_state[ 'timeseries_lag_quantile_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_lag_quantile_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_lag_quantile_details' ] = df_details.copy( )
					st.session_state[ 'timeseries_lag_quantile_signature' ] = lag_quantile_signature
					st.success( 'Lagged Quantile Regression forecasting complete.' )
				except Exception as ex:
					for key in [ 'timeseries_lag_quantile_model',
						'timeseries_lag_quantile_forecast', 'timeseries_lag_quantile_metrics',
						'timeseries_lag_quantile_results', 'timeseries_lag_quantile_details',
						'timeseries_lag_quantile_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'Lagged Quantile Regression failed: {ex}' )
			
			current_lag_quantile_signature = (series_col, len( series ), series_signature,
				lag_quantile_order, lag_quantile_value, lag_quantile_alpha, lag_quantile_fit,
				lag_quantile_solver, lag_quantile_horizon)
			
			lag_quantile_signature = st.session_state.get( 'timeseries_lag_quantile_signature',
				None )
			
			if lag_quantile_signature == current_lag_quantile_signature:
				df_metrics = st.session_state.get( 'timeseries_lag_quantile_metrics',
					pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_lag_quantile_results',
					pd.DataFrame( ) )
				df_details = st.session_state.get( 'timeseries_lag_quantile_details',
					pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_lag_quantile_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				df_details = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				render_data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Run Lagged Quantile Regression to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				render_data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run Lagged Quantile Regression to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			
			if len( forecast ) > 0:
				figure = create_forecast_figure( series, forecast, series_col,
					'Lagged Quantile Regression Forecast',
					f'Quantile {lag_quantile_value:.2f}' )
				render_mathy_plotly_chart( figure, 'timeseries_lag_quantile_forecast_chart',
					'mathy_lag_quantile_forecast', 'Lagged Quantile Regression Forecast', 560 )
			else:
				st.info( 'Run Lagged Quantile Regression to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# ARIMA
		# ------------------------------------------------------------------
		with st.expander( 'ARIMA', expanded=False ):
			arima_defaults = { 'timeseries_arima_p': 1, 'timeseries_arima_d': 0,
				'timeseries_arima_q': 0, 'timeseries_arima_horizon': 5 }
			
			for key, value in arima_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			st.caption( 'Nonseasonal autoregressive integrated moving-average forecasting.' )
			ar_c1, ar_c2, ar_c3, ar_c4 = st.columns( 4, border=True )
			with ar_c1:
				arima_p = int( st.number_input( 'Autoregressive Order (p)', min_value=0, step=1,
					key='timeseries_arima_p' ) )
			
			with ar_c2:
				arima_d = int( st.number_input( 'Differencing Order (d)', min_value=0, step=1,
					key='timeseries_arima_d' ) )
			
			with ar_c3:
				arima_q = int( st.number_input( 'Moving-Average Order (q)', min_value=0, step=1,
					key='timeseries_arima_q' ) )
			
			with ar_c4:
				arima_horizon = int( st.number_input( 'Forecast Horizon', min_value=1, step=1,
					key='timeseries_arima_horizon' ) )
			
			arima_reset_keys = list( arima_defaults.keys( ) ) + [ 'timeseries_arima_model',
				'timeseries_arima_forecast', 'timeseries_arima_metrics',
				'timeseries_arima_results', 'timeseries_arima_details', 'timeseries_arima_signature' ]
			
			ar_b1, ar_b2 = st.columns( 2 )
			with ar_b1:
				run_arima = st.button( 'Run ARIMA', icon='🏃', key='timeseries_arima_run',
					use_container_width=True )
			
			with ar_b2:
				st.button( 'Reset ARIMA', icon='🔁', key='timeseries_arima_reset',
					use_container_width=True, on_click=clear_keys, args=(arima_reset_keys,) )
			
			if run_arima:
				try:
					minimum_observations = max( arima_p, arima_q, 1 )
					if len( series ) <= minimum_observations:
						st.warning(
							'⚠️ The selected series does not contain enough observations for the '
							'ARIMA order.' )
						st.stop( )
					
					if arima_p == 0 and arima_d == 0 and arima_q == 0:
						st.warning( '⚠️ At least one ARIMA order must be greater than zero.' )
						st.stop( )
					
					arima_order = (arima_p, arima_d, arima_q)
					model = ARIMA( order=arima_order )
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=arima_horizon )
					elapsed_seconds = time.time( ) - start_time
					if metrics is None:
						metrics = { }
					
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast = np.asarray( forecast, dtype=float ).reshape( -1 )
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index, 'Forecast': forecast } )
					detail_rows = [ { 'Property': 'Order', 'Value': str( model.order ) },
						{ 'Property': 'Autoregressive Order', 'Value': arima_p },
						{ 'Property': 'Differencing Order', 'Value': arima_d },
						{ 'Property': 'Moving-Average Order', 'Value': arima_q },
						{ 'Property': 'Training Observations', 'Value': len( model.train_data ) },
						{ 'Property': 'Forecast Horizon', 'Value': arima_horizon },
						{ 'Property': 'AIC', 'Value': float( model.results.aic ) },
						{ 'Property': 'BIC', 'Value': float( model.results.bic ) } ]
					
					df_details = pd.DataFrame( detail_rows )
					arima_signature = (series_col, len( series ), series_signature, arima_p,
						arima_d, arima_q, arima_horizon)
					
					st.session_state[ 'timeseries_arima_model' ] = model
					st.session_state[ 'timeseries_arima_forecast' ] = forecast.copy( )
					st.session_state[ 'timeseries_arima_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_arima_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_arima_details' ] = df_details.copy( )
					st.session_state[ 'timeseries_arima_signature' ] = arima_signature
					st.success( 'ARIMA forecasting complete.' )
				except Exception as ex:
					for key in [ 'timeseries_arima_model', 'timeseries_arima_forecast',
						'timeseries_arima_metrics', 'timeseries_arima_results',
						'timeseries_arima_details', 'timeseries_arima_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'ARIMA forecasting failed: {ex}' )
			
			current_arima_signature = (series_col, len( series ), series_signature, arima_p,
				arima_d, arima_q, arima_horizon)
			
			arima_signature = st.session_state.get( 'timeseries_arima_signature', None )
			if arima_signature == current_arima_signature:
				df_metrics = st.session_state.get( 'timeseries_arima_metrics', pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_arima_results', pd.DataFrame( ) )
				df_details = st.session_state.get( 'timeseries_arima_details', pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_arima_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				df_details = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				render_data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Run ARIMA to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				render_data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run ARIMA to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			
			if len( forecast ) > 0:
				title = f'ARIMA{(arima_p, arima_d, arima_q)} Forecast'
				figure = create_forecast_figure( series, forecast, series_col, title )
				render_mathy_plotly_chart( figure, 'timeseries_arima_forecast_chart',
					'mathy_arima_forecast', title, 560 )
			else:
				st.info( 'Run ARIMA to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# SARIMA
		# ------------------------------------------------------------------
		with st.expander( 'SARIMA', expanded=False ):
			sarima_defaults = { 'timeseries_sarima_p': 1, 'timeseries_sarima_d': 0,
				'timeseries_sarima_q': 0, 'timeseries_sarima_seasonal_p': 0,
				'timeseries_sarima_seasonal_d': 0, 'timeseries_sarima_seasonal_q': 0,
				'timeseries_sarima_seasonal_period': 0, 'timeseries_sarima_horizon': 5 }
			
			for key, value in sarima_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			st.caption( 'Seasonal autoregressive integrated moving-average forecasting.' )
			sa_c1, sa_c2, sa_c3, sa_c4 = st.columns( 4, border=True )
			with sa_c1:
				sarima_p = int( st.number_input( 'Autoregressive Order (p)', min_value=0, step=1,
					key='timeseries_sarima_p' ) )
				
				sarima_seasonal_p = int(
					st.number_input( 'Seasonal Autoregressive Order (P)', min_value=0, step=1,
						key='timeseries_sarima_seasonal_p' ) )
			
			with sa_c2:
				sarima_d = int( st.number_input( 'Differencing Order (d)', min_value=0, step=1,
					key='timeseries_sarima_d' ) )
				
				sarima_seasonal_d = int(
					st.number_input( 'Seasonal Differencing Order (D)', min_value=0, step=1,
						key='timeseries_sarima_seasonal_d' ) )
			
			with sa_c3:
				sarima_q = int( st.number_input( 'Moving-Average Order (q)', min_value=0, step=1,
					key='timeseries_sarima_q' ) )
				
				sarima_seasonal_q = int(
					st.number_input( 'Seasonal Moving-Average Order (Q)', min_value=0, step=1,
						key='timeseries_sarima_seasonal_q' ) )
			
			with sa_c4:
				sarima_seasonal_period = int(
					st.number_input( 'Seasonal Period (s)', min_value=0, step=1,
						key='timeseries_sarima_seasonal_period' ) )
				
				sarima_horizon = int( st.number_input( 'Forecast Horizon', min_value=1, step=1,
					key='timeseries_sarima_horizon' ) )
			
			sarima_reset_keys = list( sarima_defaults.keys( ) ) + [ 'timeseries_sarima_model',
				'timeseries_sarima_forecast', 'timeseries_sarima_metrics',
				'timeseries_sarima_results', 'timeseries_sarima_details',
				'timeseries_sarima_signature' ]
			
			sa_b1, sa_b2 = st.columns( 2 )
			with sa_b1:
				run_sarima = st.button( 'Run SARIMA', icon='🏃', key='timeseries_sarima_run',
					use_container_width=True )
			
			with sa_b2:
				st.button( 'Reset SARIMA', icon='🔁', key='timeseries_sarima_reset',
					use_container_width=True, on_click=clear_keys, args=(sarima_reset_keys,) )
			
			if run_sarima:
				try:
					nonseasonal_order = (sarima_p, sarima_d, sarima_q)
					seasonal_terms = (sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q)
					uses_seasonality = any( value > 0 for value in seasonal_terms )
					if (all( value == 0 for value in nonseasonal_order ) and not uses_seasonality):
						st.warning(
							'⚠️ At least one nonseasonal or seasonal order must be greater than '
							'zero.' )
						st.stop( )
					
					if uses_seasonality and sarima_seasonal_period < 2:
						st.warning(
							'⚠️ Seasonal Period must be at least 2 when seasonal orders are '
							'used.' )
						st.stop( )
					
					if (not uses_seasonality and sarima_seasonal_period not in [ 0, 1 ]):
						st.warning(
							'⚠️ Set Seasonal Period to 0 when no seasonal orders are used.' )
						st.stop( )
					
					effective_seasonal_period = (sarima_seasonal_period if uses_seasonality else 0)
					seasonal_order = (sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q,
						effective_seasonal_period)
					
					minimum_observations = max( sarima_p + sarima_d + sarima_q,
						sarima_seasonal_p + sarima_seasonal_d + sarima_seasonal_q, 1 )
					
					if len( series ) <= minimum_observations:
						st.warning(
							'⚠️ The selected series does not contain enough observations for the '
							'SARIMA orders.' )
						st.stop( )
					
					if (uses_seasonality and len( series ) <= effective_seasonal_period):
						st.warning(
							'⚠️ The selected series must contain more observations than the '
							'Seasonal Period.' )
						st.stop( )
					
					model = SARIMA( order=nonseasonal_order, seasonal=seasonal_order )
					start_time = time.time( )
					model.train( series )
					metrics = model.analyze( )
					forecast = model.project( n_steps=sarima_horizon )
					elapsed_seconds = time.time( ) - start_time
					if metrics is None:
						metrics = { }
					
					df_metrics = pd.DataFrame(
						[ { 'Metric': metric, 'Value': float( value ) } for metric, value in
							metrics.items( ) ] )
					
					df_metrics = pd.concat( [ df_metrics, pd.DataFrame(
						[ { 'Metric': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] ) ], ignore_index=True )
					
					forecast = np.asarray( forecast, dtype=float ).reshape( -1 )
					forecast_index = np.arange( len( series ), len( series ) + len( forecast ) )
					df_results = pd.DataFrame( { 'Period': forecast_index, 'Forecast': forecast } )
					detail_rows = [ { 'Property': 'Order', 'Value': str( model.order ) },
						{ 'Property': 'Seasonal Order', 'Value': str( model.seasonal_order ) },
						{ 'Property': 'Training Observations',
							'Value': len( model.training_data ) },
						{ 'Property': 'Forecast Horizon', 'Value': sarima_horizon },
						{ 'Property': 'AIC', 'Value': float( model.results.aic ) },
						{ 'Property': 'BIC', 'Value': float( model.results.bic ) } ]
					
					df_details = pd.DataFrame( detail_rows )
					sarima_signature = (series_col, len( series ), series_signature, sarima_p,
						sarima_d, sarima_q, sarima_seasonal_p, sarima_seasonal_d,
						sarima_seasonal_q,
						effective_seasonal_period, sarima_horizon)
					
					st.session_state[ 'timeseries_sarima_model' ] = model
					st.session_state[ 'timeseries_sarima_forecast' ] = forecast.copy( )
					st.session_state[ 'timeseries_sarima_metrics' ] = df_metrics.copy( )
					st.session_state[ 'timeseries_sarima_results' ] = df_results.copy( )
					st.session_state[ 'timeseries_sarima_details' ] = df_details.copy( )
					st.session_state[ 'timeseries_sarima_signature' ] = sarima_signature
					st.success( 'SARIMA forecasting complete.' )
				except Exception as ex:
					for key in [ 'timeseries_sarima_model', 'timeseries_sarima_forecast',
						'timeseries_sarima_metrics', 'timeseries_sarima_results',
						'timeseries_sarima_details', 'timeseries_sarima_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'SARIMA forecasting failed: {ex}' )
			
			current_uses_seasonality = any(
				value > 0 for value in (sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q) )
			
			current_seasonal_period = (sarima_seasonal_period if current_uses_seasonality else 0)
			current_sarima_signature = (series_col, len( series ), series_signature, sarima_p,
				sarima_d, sarima_q, sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q,
				current_seasonal_period, sarima_horizon)
			
			sarima_signature = st.session_state.get( 'timeseries_sarima_signature', None )
			if sarima_signature == current_sarima_signature:
				df_metrics = st.session_state.get( 'timeseries_sarima_metrics', pd.DataFrame( ) )
				df_results = st.session_state.get( 'timeseries_sarima_results', pd.DataFrame( ) )
				df_details = st.session_state.get( 'timeseries_sarima_details', pd.DataFrame( ) )
				forecast = st.session_state.get( 'timeseries_sarima_forecast',
					np.array( [ ], dtype=float ) )
			else:
				df_metrics = pd.DataFrame( )
				df_results = pd.DataFrame( )
				df_details = pd.DataFrame( )
				forecast = np.array( [ ], dtype=float )
			
			# ------------------------------------------------------------------
			# MODEL EVALUATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Model Evaluation' )
			
			if not df_metrics.empty:
				render_data_editor( df_metrics, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_details.empty:
					st.caption( 'Model Details' )
					render_data_editor( df_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Run SARIMA to view model metrics.' )
			
			# ------------------------------------------------------------------
			# FORECAST RESULTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Forecast Results' )
			
			if not df_results.empty:
				render_data_editor( df_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Run SARIMA to view forecast values.' )
			
			# ------------------------------------------------------------------
			# FORECAST VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Observed vs Forecast' )
			
			if len( forecast ) > 0:
				title = ( f'SARIMA{(sarima_p, sarima_d, sarima_q)}'
					f'{(sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q, current_seasonal_period)} Forecast' )
				figure = create_forecast_figure( series, forecast, series_col, title )
				render_mathy_plotly_chart( figure, 'timeseries_sarima_forecast_chart',
					'mathy_sarima_forecast', title, 560 )
			else:
				st.info( 'Run SARIMA to view the forecast plot.' )
		
		# ------------------------------------------------------------------
		# TIME-SERIES CROSS-VALIDATION
		# ------------------------------------------------------------------
		with st.expander( 'Time-Series Validation Windows', expanded=False ):
			splitter_defaults = { 'timeseries_splitter_splits': 5,
				'timeseries_splitter_use_train_size': False, 'timeseries_splitter_train_size': 20,
				'timeseries_splitter_use_test_size': False, 'timeseries_splitter_test_size': 5,
				'timeseries_splitter_gap': 0 }
			
			for key, value in splitter_defaults.items( ):
				if key not in st.session_state:
					st.session_state[ key ] = value
			
			maximum_split_count = max( 2, len( series ) - 1 )
			maximum_window_size = max( 1, len( series ) - 1 )
			st.session_state[ 'timeseries_splitter_splits' ] = min(
				max( 2, int( st.session_state[ 'timeseries_splitter_splits' ] ) ),
				maximum_split_count )
			st.session_state[ 'timeseries_splitter_train_size' ] = min(
				max( 1, int( st.session_state[ 'timeseries_splitter_train_size' ] ) ),
				maximum_window_size )
			st.session_state[ 'timeseries_splitter_test_size' ] = min(
				max( 1, int( st.session_state[ 'timeseries_splitter_test_size' ] ) ),
				maximum_window_size )
			st.session_state[ 'timeseries_splitter_gap' ] = min(
				max( 0, int( st.session_state[ 'timeseries_splitter_gap' ] ) ),
				max( 0, len( series ) - 2 ) )
			
			st.caption( 'Chronological windows for subsequent model evaluation without random shuffling.' )
			cv_c1, cv_c2, cv_c3 = st.columns( [ 0.33, 0.33, 0.34 ], border=True )
			with cv_c1:
				splitter_splits = int( st.number_input( 'Number of Splits', min_value=2,
					max_value=max( 2, len( series ) - 1 ), step=1,
					key='timeseries_splitter_splits' ) )
				
				splitter_gap = int( st.number_input( 'Temporal Gap', min_value=0,
					max_value=max( 0, len( series ) - 2 ), step=1, key='timeseries_splitter_gap' ) )
			
			with cv_c2:
				splitter_use_train_size = st.checkbox( 'Limit Training Window',
					key='timeseries_splitter_use_train_size' )
				
				splitter_train_size_input = int( st.number_input( 'Maximum Training Size', min_value=1,
						max_value=max( 1, len( series ) - 1 ), step=1,
						disabled=not splitter_use_train_size, key='timeseries_splitter_train_size' ) )
			
			with cv_c3:
				splitter_use_test_size = st.checkbox( 'Use Fixed Test Size',
					key='timeseries_splitter_use_test_size' )
				
				splitter_test_size_input = int( st.number_input( 'Test Size', min_value=1,
					max_value=max( 1, len( series ) - 1 ), step=1,
					disabled=not splitter_use_test_size, key='timeseries_splitter_test_size' ) )
			
			splitter_reset_keys = list( splitter_defaults.keys( ) ) + [ 'timeseries_splitter_model',
				'timeseries_splitter_results', 'timeseries_splitter_summary',
				'timeseries_splitter_details', 'timeseries_splitter_figure',
				'timeseries_splitter_signature' ]
			
			cv_b1, cv_b2 = st.columns( 2 )
			with cv_b1:
				run_splitter = st.button( 'Generate Time-Series Splits', icon='🏃',
					key='timeseries_splitter_run', use_container_width=True )
			
			with cv_b2:
				st.button( 'Reset Time-Series Splits', icon='🔁', key='timeseries_splitter_reset',
					use_container_width=True, on_click=clear_keys, args=(splitter_reset_keys,) )
			
			if run_splitter:
				try:
					max_train_size = ( splitter_train_size_input if splitter_use_train_size else None)
					test_size = (splitter_test_size_input if splitter_use_test_size else None)
					computed_test_size = ( len( series ) // (splitter_splits + 1) if test_size is None else test_size)
					required_observations = (splitter_splits * computed_test_size + splitter_gap)
					if computed_test_size < 1:
						st.warning( '⚠️ The selected configuration produces an invalid test-window size.' )
						st.stop( )
					
					if required_observations >= len( series ):
						st.warning( '⚠️ The selected series does not contain enough observations for the '
							'requested splits, test size, and gap.' )
						st.stop( )
					
					splitter = TimeSeriesSpliter( splits=splitter_splits, max_train_size=max_train_size,
						test_size=test_size, gap=splitter_gap )
					
					start_time = time.time( )
					split_pairs = splitter.get_splits( series )
					elapsed_seconds = time.time( ) - start_time
					
					if not split_pairs:
						st.warning( '⚠️ No time-series splits were generated.' )
						st.stop( )
					
					result_rows = [ ]
					summary_rows = [ ]
					for split_number, split_pair in enumerate( split_pairs, start=1 ):
						train_index, test_index = split_pair
						train_start = int( train_index[ 0 ] )
						train_end = int( train_index[ -1 ] )
						test_start = int( test_index[ 0 ] )
						test_end = int( test_index[ -1 ] )
						summary_rows.append( { 'Split': split_number, 'Train Start': train_start,
							'Train End': train_end, 'Train Size': len( train_index ),
							'Test Start': test_start, 'Test End': test_end,
							'Test Size': len( test_index ), 'Gap': splitter_gap } )
						
						for index in train_index:
							result_rows.append( { 'Split': split_number, 'Period': int( index ),
								'Value': float( series[ index ] ), 'Partition': 'Train' } )
						
						for index in test_index:
							result_rows.append( { 'Split': split_number, 'Period': int( index ),
								'Value': float( series[ index ] ), 'Partition': 'Test' } )
					
					df_split_results = pd.DataFrame( result_rows )
					figure = create_split_figure( df_split_results )
					df_split_summary = pd.DataFrame( summary_rows )
					df_processing = pd.DataFrame( [ { 'Property': 'Configured Splits',
						'Value': splitter.get_n_splits( series ) },
						{ 'Property': 'Generated Splits', 'Value': len( split_pairs ) },
						{ 'Property': 'Maximum Training Size', 'Value': max_train_size },
						{ 'Property': 'Test Size', 'Value': computed_test_size },
						{ 'Property': 'Temporal Gap', 'Value': splitter_gap },
						{ 'Property': 'Processing Time (sec)',
							'Value': round( elapsed_seconds, 4 ) } ] )
					
					splitter_signature = (series_col, len( series ), series_signature,
						splitter_splits, max_train_size, test_size, splitter_gap)
					
					st.session_state[ 'timeseries_splitter_model' ] = splitter
					st.session_state[ 'timeseries_splitter_results' ] = df_split_results.copy( )
					st.session_state[ 'timeseries_splitter_summary' ] = df_split_summary.copy( )
					st.session_state[ 'timeseries_splitter_details' ] = df_processing.copy( )
					st.session_state[ 'timeseries_splitter_figure' ] = figure
					st.session_state[ 'timeseries_splitter_signature' ] = splitter_signature
					st.success( 'Time-series cross-validation splits generated.' )
				except Exception as ex:
					for key in [ 'timeseries_splitter_model', 'timeseries_splitter_results',
						'timeseries_splitter_summary', 'timeseries_splitter_details',
						'timeseries_splitter_figure', 'timeseries_splitter_signature' ]:
						st.session_state.pop( key, None )
					
					st.error( f'Time-Series Cross-Validation failed: {ex}' )
			
			current_max_train_size = ( splitter_train_size_input if splitter_use_train_size else None)
			current_test_size = (splitter_test_size_input if splitter_use_test_size else None)
			current_splitter_signature = (series_col, len( series ), series_signature,
				splitter_splits, current_max_train_size, current_test_size, splitter_gap)
			
			splitter_signature = st.session_state.get( 'timeseries_splitter_signature', None )
			if splitter_signature == current_splitter_signature:
				df_split_results = st.session_state.get( 'timeseries_splitter_results',
					pd.DataFrame( ) )
				df_split_summary = st.session_state.get( 'timeseries_splitter_summary',
					pd.DataFrame( ) )
				df_split_details = st.session_state.get( 'timeseries_splitter_details',
					pd.DataFrame( ) )
				split_figure = st.session_state.get( 'timeseries_splitter_figure', None )
			else:
				df_split_results = pd.DataFrame( )
				df_split_summary = pd.DataFrame( )
				df_split_details = pd.DataFrame( )
				split_figure = None
			
			# ------------------------------------------------------------------
			# SPLIT SUMMARY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Split Summary' )
			
			if not df_split_summary.empty:
				render_data_editor( df_split_summary, use_container_width=True, hide_index=True,
					disabled=True )
				
				if not df_split_details.empty:
					st.caption( 'Splitter Details' )
					render_data_editor( df_split_details, use_container_width=True, hide_index=True,
						disabled=True )
			else:
				st.info( 'Generate time-series splits to view the train-and-test windows.' )
			
			# ------------------------------------------------------------------
			# SPLIT ASSIGNMENTS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Split Assignments' )
			
			if not df_split_results.empty:
				render_data_editor( df_split_results, use_container_width=True, hide_index=True,
					disabled=True )
			else:
				st.info( 'Generate time-series splits to view individual period assignments.' )
			
			# ------------------------------------------------------------------
			# SPLIT VISUALIZATION
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Split Visualization' )
			
			if split_figure is not None:
				render_mathy_plotly_chart( split_figure, 'timeseries_splitter_chart',
					'mathy_timeseries_splits', 'Time-Series Validation Windows', 600 )
			else:
				st.info( 'Generate time-series splits to view the validation windows.' )

# ============================================
# DATA OVERVIEW MODE
# ============================================
elif mode == 'Data Overview':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Data Overview', divider='gray' )
		render_visualization_metric_styles( )
		df_source = get_loaded_dataset( )
		if df_source is None:
			st.info( 'No data loaded.' )
			st.stop( )
		
		synchronize_dataset_columns( df_source )
		profiles = st.session_state[ 'column_profiles' ].copy( )
		schema = st.session_state[ 'column_schema' ].copy( )
		df_dataset = st.session_state[ 'df_analysis' ].copy( )
		type_counts = pd.Series( schema ).value_counts( )
		
		preview_c1, preview_c2 = st.columns( [ .25, .75 ], border=True )
		with preview_c1:
			max_preview_rows = min( 100, len( df_dataset ) )
			if max_preview_rows <= 5:
				preview_rows = st.selectbox( 'Preview Rows', [ max_preview_rows ],
					key='visualization_overview_rows_fixed' )
			else:
				preview_rows = st.slider( 'Preview Rows', min_value=5,
					max_value=max_preview_rows, value=min( 20, max_preview_rows ), step=1,
					key='visualization_overview_rows' )
		
		with preview_c2:
			preview_columns = st.multiselect( 'Preview Columns', df_dataset.columns.tolist( ),
				default=df_dataset.columns.tolist( ), key='visualization_overview_columns' )

		st.divider( )
		
		if preview_columns:
			render_table( df_dataset.loc[ :, preview_columns ].head( preview_rows ) )
		else:
			st.info( 'Select one or more columns to preview.' )
		
		missing_cells = int( df_source.isna( ).sum( ).sum( ) )
		duplicate_rows = int( df_source.duplicated( ).sum( ) )
		
		m1, m2, m3, m4, m5 = st.columns( 5, border=True )
		m1.metric( 'Rows', f'{len( df_source ):,}' )
		m2.metric( 'Numeric', f'{int( type_counts.get( "numeric", 0 ) ):,}' )
		ordinal_id_count = (int( type_counts.get( "ordinal", 0 ) )
			+ int( type_counts.get( "identifier", 0 ) ))
		m3.metric( 'Ordinal / ID', f'{ordinal_id_count:,}' )
		m4.metric( 'Categorical', f'{int( type_counts.get( "categorical", 0 ) ):,}' )
		m5.metric( 'Datetime', f'{int( type_counts.get( "datetime", 0 ) ):,}' )
		st.caption( f'Columns: {len( df_source.columns ):,} · Missing cells: {missing_cells:,} · '
			f'Duplicate rows: {duplicate_rows:,}.' )
		
		blue_divider( )
		
		st.markdown( '##### Schema' )
		schema_rows: List[ Dict[ str, Any ] ] = [ ]
		for column in df_source.columns:
			profile = profiles[ column ]
			non_null = int( df_source[ column ].notna( ).sum( ) )
			missing = int( df_source[ column ].isna( ).sum( ) )
			schema_rows.append( { 'Column': column,
				'Storage Type': str( profile[ 'storage_dtype' ] ),
				'Inferred Type': str( profile[ 'inferred_dtype' ] ),
				'Analytical Role': str( profile[ 'analytical_role' ] ),
				'Confidence': float( profile[ 'confidence' ] ),
				'Non-Null': non_null, 'Populated': int( profile[ 'non_null_count' ] ),
				'Missing': missing,
				'Missing %': missing / len( df_source ) * 100.0,
				'Unique': int( profile[ 'distinct_count' ] ),
				'Evidence': str( profile[ 'reason' ] ) } )
		
		df_schema = pd.DataFrame( schema_rows, columns=[ 'Column', 'Storage Type',
			'Inferred Type', 'Analytical Role', 'Confidence', 'Non-Null', 'Populated',
			'Missing', 'Missing %', 'Unique', 'Evidence' ] )
		schema_sort = st.selectbox( 'Sort Schema By',
			[ 'Original Order', 'Analytical Role', 'Storage Type', 'Inferred Type',
				'Missing Percentage', 'Cardinality' ], key='visualization_overview_schema_sort' )
		if schema_sort == 'Analytical Role':
			df_schema = df_schema.sort_values( [ 'Analytical Role', 'Column' ],
				ignore_index=True )
		elif schema_sort == 'Storage Type':
			df_schema = df_schema.sort_values( [ 'Storage Type', 'Column' ], ignore_index=True )
		elif schema_sort == 'Inferred Type':
			df_schema = df_schema.sort_values( [ 'Inferred Type', 'Column' ], ignore_index=True )
		elif schema_sort == 'Missing Percentage':
			df_schema = df_schema.sort_values( 'Missing %', ascending=False, ignore_index=True )
		elif schema_sort == 'Cardinality':
			df_schema = df_schema.sort_values( 'Unique', ascending=False, ignore_index=True )
		render_data_editor( df_schema, use_container_width=True, hide_index=True, disabled=True,
			column_config={
				'Confidence': st.column_config.NumberColumn( 'Confidence', format='%.2f' ),
				'Missing %': st.column_config.NumberColumn( 'Missing %', format='%.2f' ),
				'Evidence': st.column_config.TextColumn( 'Evidence', width='large' ) },
			key='visualization_overview_schema_table' )
		
		blue_divider( )
		chart_c1, chart_c2 = st.columns( 2, border=True )
		with chart_c1:
			df_type_counts = type_counts.rename_axis( 'Analytical Role' ).reset_index(
				name='Columns' )
			figure = px.bar( df_type_counts, x='Analytical Role', y='Columns',
				color='Analytical Role', text_auto=True )
			render_mathy_plotly_chart( figure, 'visualization_overview_types_chart',
				'mathy_column_types', 'Analytical Role Distribution', 450 )
		
		with chart_c2:
			df_cardinality = df_schema.nlargest( min( 15, len( df_schema ) ), 'Unique' ).sort_values(
				'Unique' )
			figure = px.bar( df_cardinality, x='Unique', y='Column', orientation='h',
				color='Unique', color_continuous_scale='Blues' )
			render_mathy_plotly_chart( figure, 'visualization_overview_cardinality_chart',
				'mathy_cardinality', 'Top Columns by Cardinality', 450 )

		if missing_cells:
			df_missing = df_schema[ df_schema[ 'Missing' ] > 0 ].sort_values( 'Missing %' )
			figure = px.bar( df_missing, x='Missing %', y='Column', orientation='h',
				color='Missing %', color_continuous_scale='Reds' )
			render_mathy_plotly_chart( figure, 'visualization_overview_missing_chart',
				'mathy_missing_overview', 'Missing Values by Column', 480 )

# ============================================
# NUMERIC DISTRIBUTIONS MODE
# ============================================
elif mode == 'Numeric Distributions':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Numeric Distributions', divider='gray' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		if df_dataset.empty:
			st.info( 'No data loaded.' )
			st.stop( )

		numeric_columns = get_visualization_columns( df_dataset )[ 'numeric' ]
		if not numeric_columns:
			st.info( 'No numeric columns are available.' )
			st.stop( )

		c1, c2, c3, c4 = st.columns( 4, border=True )
		with c1:
			selected_columns = st.multiselect( 'Numeric Variables', numeric_columns,
				default=numeric_columns[ :min( 4, len( numeric_columns ) ) ],
				key='visualization_numeric_columns' )
		
		with c2:
			bins = st.slider( 'Bins', 10, 100, 30, 5, key='visualization_numeric_bins' )
		
		with c3:
			display_mode = st.radio( 'Display', [ 'Frequency', 'Density' ], horizontal=True,
				key='visualization_numeric_display' )
		
		with c4:
			shape_mode = st.radio( 'Shape', [ 'Box', 'Violin' ], horizontal=True,
				key='visualization_numeric_shape' )

		show_kde = st.checkbox( 'Show KDE Overlay', value=True,
			key='visualization_numeric_kde' )
		if not selected_columns:
			st.info( 'Select one or more numeric variables.' )

		for index, column in enumerate( selected_columns ):
			series = pd.to_numeric( df_dataset[ column ], errors='coerce' ).replace(
				[ np.inf, -np.inf ], np.nan ).dropna( )
			if series.empty:
				st.warning( f'{column}: no plottable numeric values.' )
				continue

			blue_divider( )
			st.markdown( f'##### {column}' )
			plot_c1, plot_c2 = st.columns( 2, border=True )
			with plot_c1:
				histnorm = 'probability density' if display_mode == 'Density' else None
				figure = go.Figure( )
				figure.add_trace( go.Histogram( x=series.tolist( ), nbinsx=bins,
					histnorm=histnorm, name=column, marker_color='#38BDF8', opacity=0.82 ) )
				if show_kde and len( series ) > 1 and float( series.std( ddof=1 ) ) > 0:
					x_values = np.linspace( float( series.min( ) ), float( series.max( ) ), 250 )
					kde_values = stats.gaussian_kde( series.to_numpy( ) )( x_values )
					if display_mode == 'Frequency':
						bin_width = (float( series.max( ) ) - float( series.min( ) )) / bins
						kde_values = kde_values * len( series ) * bin_width
					figure.add_trace( go.Scatter( x=x_values, y=kde_values, mode='lines',
						name='KDE', line={ 'color': '#F472B6', 'width': 3 } ) )
				add_statistical_reference_lines( figure, float( series.mean( ) ),
					float( series.median( ) ) )
				figure.update_xaxes( title_text=column )
				figure.update_yaxes( title_text=display_mode )
				render_mathy_plotly_chart( figure, f'visualization_numeric_hist_{index}',
					f'mathy_distribution_{index}', f'Distribution — {column}', 470 )

			with plot_c2:
				if shape_mode == 'Box':
					figure = px.box( x=series, points='outliers', labels={ 'x': column } )
				else:
					figure = px.violin( x=series, box=True, points='outliers',
						labels={ 'x': column } )
				render_mathy_plotly_chart( figure, f'visualization_numeric_shape_{index}',
					f'mathy_shape_{index}', f'{shape_mode} — {column}', 470 )

			m1, m2, m3, m4, m5, m6 = st.columns( 6, border=True )
			m1.metric( 'Count', f'{len( series ):,}' )
			m2.metric( 'Mean', f'{float( series.mean( ) ):,.2f}' )
			m3.metric( 'Median', f'{float( series.median( ) ):,.2f}' )
			m4.metric( 'Std', f'{float( series.std( ddof=1 ) ):,.2f}' if len( series ) > 1 else '0.00' )
			m5.metric( 'Skew', f'{float( series.skew( ) ):,.3f}' if len( series ) > 2 else 'n/a' )
			m6.metric( 'Kurtosis', f'{float( series.kurt( ) ):,.3f}' if len( series ) > 3 else 'n/a' )

# ============================================
# CORRELATION ANALYSIS MODE
# ============================================
elif mode == 'Correlation Analysis':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Correlation Analysis', divider='gray' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		if df_dataset.empty:
			st.info( 'No data loaded.' )
			st.stop( )

		numeric_columns = get_visualization_columns( df_dataset )[ 'numeric' ]
		if len( numeric_columns ) < 2:
			st.info( 'At least two numeric columns are required.' )
			st.stop( )

		c1, c2, c3 = st.columns( 3, border=True )
		with c1:
			selected_columns = st.multiselect( 'Numeric Variables', numeric_columns,
				default=numeric_columns[ :min( 8, len( numeric_columns ) ) ],
				key='visualization_correlation_columns' )
		
		with c2:
			method = st.radio( 'Method', [ 'Pearson', 'Spearman' ], horizontal=True,
				key='visualization_correlation_method' )
		
		with c3:
			threshold = st.slider( 'Pair Threshold', 0.0, 1.0, 0.0, 0.05,
				key='visualization_correlation_threshold' )

		show_annotations = st.checkbox( 'Show Heatmap Values', value=True,
			key='visualization_correlation_annotations' )
		if len( selected_columns ) < 2:
			st.info( 'Select at least two numeric variables.' )
			st.stop( )

		df_numeric = df_dataset[ selected_columns ].apply( pd.to_numeric, errors='coerce' )
		df_correlation = df_numeric.corr( method=method.lower( ) )
		df_pairs = create_correlation_pairs( df_correlation )
		df_filtered_pairs = df_pairs[ df_pairs[ 'Absolute Correlation' ] >= threshold ].copy( )

		chart_c1, chart_c2 = st.columns( 2, border=True )
		with chart_c1:
			render_data_editor( df_correlation, use_container_width=True, disabled=True,
				key='visualization_correlation_matrix_table' )
		
		with chart_c2:
			text_values = np.round( df_correlation.values, 2 ) if show_annotations else None
			figure = go.Figure( data=go.Heatmap( z=df_correlation.values,
				x=df_correlation.columns, y=df_correlation.index, zmin=-1, zmax=1,
				colorscale='RdBu', reversescale=True, text=text_values,
				texttemplate='%{text}' if show_annotations else None,
				hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.4f}<extra></extra>' ) )
			render_mathy_plotly_chart( figure, 'visualization_correlation_heatmap_chart',
				'mathy_correlation_heatmap', f'{method} Correlation Heatmap', 580 )

		blue_divider( )
		st.markdown( '##### Ranked Relationships' )
		render_data_editor( df_filtered_pairs, use_container_width=True, hide_index=True,
			disabled=True, key='visualization_correlation_pairs_table' )
		
		st.divider( )
		
		if not df_filtered_pairs.empty:
			df_plot = df_filtered_pairs.head( 20 ).copy( )
			df_plot[ 'Pair' ] = df_plot[ 'Variable 1' ] + ' ↔ ' + df_plot[ 'Variable 2' ]
			figure = px.bar( df_plot.sort_values( 'Absolute Correlation' ),
				x='Absolute Correlation', y='Pair', orientation='h', color='Correlation',
				color_continuous_scale='RdBu', range_color=[ -1, 1 ] )
			render_mathy_plotly_chart( figure, 'visualization_correlation_pairs_chart',
				'mathy_ranked_correlations', 'Strongest Correlation Pairs', 550 )

# ============================================
# SCATTER ANALYSIS MODE
# ============================================
elif mode == 'Scatter Analysis':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Scatter Analysis', divider='gray' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		if df_dataset.empty:
			st.info( 'No data loaded.' )
			st.stop( )

		groups = get_visualization_columns( df_dataset )
		numeric_columns = groups[ 'numeric' ]
		if len( numeric_columns ) < 2:
			st.info( 'At least two numeric columns are required.' )
			st.stop( )

		c1, c2, c3, c4 = st.columns( 4, border=True )
		with c1:
			x_column = st.selectbox( 'X Variable', numeric_columns,
				key='visualization_scatter_x' )
		
		with c2:
			y_options = [ column for column in numeric_columns if column != x_column ]
			y_column = st.selectbox( 'Y Variable', y_options,
				key='visualization_scatter_y' )
		
		with c3:
			color_column = st.selectbox( 'Color Group', [ '<None>' ] + groups[ 'categorical' ],
				key='visualization_scatter_color' )
		
		with c4:
			size_column = st.selectbox( 'Size Variable', [ '<None>' ] + numeric_columns,
				key='visualization_scatter_size' )

		c5, c6, c7 = st.columns( 3, border=True )
		with c5:
			show_trend = st.checkbox( 'Show Linear Trend', value=True,
				key='visualization_scatter_trend' )
		
		with c6:
			opacity = st.slider( 'Point Opacity', 0.20, 1.0, 0.75, 0.05,
				key='visualization_scatter_opacity' )
		
		with c7:
			marker_size = st.slider( 'Marker Size', 4, 24, 9, 1,
				key='visualization_scatter_marker_size' )

		blue_divider( )
		
		plot_columns = [ x_column, y_column ]
		if color_column != '<None>':
			plot_columns.append( color_column )
		if size_column != '<None>' and size_column not in plot_columns:
			plot_columns.append( size_column )
		df_plot = df_dataset[ plot_columns ].copy( )
		df_plot[ x_column ] = pd.to_numeric( df_plot[ x_column ], errors='coerce' )
		df_plot[ y_column ] = pd.to_numeric( df_plot[ y_column ], errors='coerce' )
		if size_column != '<None>':
			df_plot[ size_column ] = pd.to_numeric( df_plot[ size_column ], errors='coerce' ).abs( )
		df_plot = df_plot.dropna( subset=[ x_column, y_column ] )
		if df_plot.empty:
			st.info( 'No paired observations are available.' )
			st.stop( )

		figure = px.scatter( df_plot, x=x_column, y=y_column,
			color=None if color_column == '<None>' else color_column,
			size=None if size_column == '<None>' else size_column,
			hover_data=df_plot.columns.tolist( ), opacity=opacity )
		figure.update_traces( marker={ 'size': marker_size } if size_column == '<None>' else { } )
		if show_trend and len( df_plot ) >= 2 and df_plot[ x_column ].nunique( ) > 1:
			coefficient, intercept = np.polyfit( df_plot[ x_column ], df_plot[ y_column ], 1 )
			x_line = np.linspace( float( df_plot[ x_column ].min( ) ),
				float( df_plot[ x_column ].max( ) ), 200 )
			figure.add_trace( go.Scatter( x=x_line, y=(coefficient * x_line) + intercept,
				mode='lines', name='Linear Trend', line={ 'color': '#F59E0B', 'width': 3,
					'dash': 'dash' } ) )

		render_mathy_plotly_chart( figure, 'visualization_scatter_chart',
			'mathy_scatter_analysis', f'{y_column} vs {x_column}', 620 )
		m1, m2, m3, m4, m5 = st.columns( 5, border=True )
		m1.metric( 'Pairs', f'{len( df_plot ):,}' )
		if len( df_plot ) < 2:
			m2.metric( 'Pearson R', 'n/a' )
			m3.metric( 'Pearson P', 'n/a' )
			m4.metric( 'Spearman R', 'n/a' )
			m5.metric( 'Covariance', 'n/a' )
			st.caption( 'At least two paired observations are required for association statistics.' )
		else:
			pearson_r, pearson_p = stats.pearsonr( df_plot[ x_column ], df_plot[ y_column ] )
			spearman_r, spearman_p = stats.spearmanr( df_plot[ x_column ], df_plot[ y_column ] )
			covariance = float( df_plot[ [ x_column, y_column ] ].cov( ).iloc[ 0, 1 ] )
			m2.metric( 'Pearson R', f'{pearson_r:,.3f}' )
			m3.metric( 'Pearson P', f'{pearson_p:,.3g}' )
			m4.metric( 'Spearman R', f'{spearman_r:,.3f}' )
			m5.metric( 'Covariance', f'{covariance:,.3f}' )
			st.caption( f'Spearman p-value: {spearman_p:.3g}.' )

# ============================================
# CATEGORICAL DISTRIBUTIONS MODE
# ============================================
elif mode == 'Categorical Distributions':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Categorical Distributions', divider='gray' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		if df_dataset.empty:
			st.info( 'No data loaded.' )
			st.stop( )

		categorical_columns = get_usable_categorical_columns( df_dataset,
			get_visualization_columns( df_dataset )[ 'categorical' ] )
		if not categorical_columns:
			st.info( 'No categorical columns are available.' )
			st.stop( )

		c1, c2, c3, c4 = st.columns( 4, border=True )
		with c1:
			column = st.selectbox( 'Categorical Variable', categorical_columns,
				key='visualization_categorical_column' )
		
		with c2:
			category_limit = st.slider( 'Maximum Categories', 2, 50, 15, 1,
				key='visualization_categorical_limit' )
		
		with c3:
			display_mode = st.radio( 'Display', [ 'Count', 'Percentage' ], horizontal=True,
				key='visualization_categorical_display' )
		
		with c4:
			include_missing = st.checkbox( 'Include Missing', value=True,
				key='visualization_categorical_missing' )

		series = df_dataset[ column ].copy( )
		if include_missing:
			series = series.astype( object ).where( series.notna( ), 'Missing' )
		else:
			series = series.dropna( )
		series = series.astype( str )
		counts = series.value_counts( dropna=False )
		if len( counts ) > category_limit:
			top_counts = counts.iloc[ :category_limit - 1 ].copy( )
			top_counts.loc[ 'Other' ] = int( counts.iloc[ category_limit - 1: ].sum( ) )
			counts = top_counts
		df_frequency = counts.rename_axis( 'Category' ).reset_index( name='Count' )
		df_frequency[ 'Percentage' ] = df_frequency[ 'Count' ] / max( 1, int( counts.sum( ) ) ) * 100.0
		value_column = 'Count' if display_mode == 'Count' else 'Percentage'
		df_plot = df_frequency.sort_values( value_column )
		figure = px.bar( df_plot, x=value_column, y='Category', orientation='h',
			color=value_column, color_continuous_scale='Blues', text=value_column )
		figure.update_traces( texttemplate='%{text:.2f}' if display_mode == 'Percentage' else '%{text}' )
		render_mathy_plotly_chart( figure, 'visualization_categorical_chart',
			'mathy_categorical_distribution', f'Distribution — {column}', 580 )
		render_data_editor( df_frequency, use_container_width=True, hide_index=True, disabled=True,
			key='visualization_categorical_table' )
		most_common = df_frequency.iloc[ 0 ] if not df_frequency.empty else None
		m1, m2, m3, m4 = st.columns( 4, border=True )
		m1.metric( 'Observations', f'{len( series ):,}' )
		m2.metric( 'Unique', f'{df_dataset[ column ].nunique( dropna=True ):,}' )
		m3.metric( 'Most Common', str( most_common[ 'Category' ] ) if most_common is not None else 'n/a' )
		m4.metric( 'Leading Share', f'{float( most_common[ "Percentage" ] ):,.2f}%'
			if most_common is not None else 'n/a' )

# ============================================
# CATEGORY COMPARISONS MODE
# ============================================
elif mode == 'Category Comparisons':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Category Comparisons', divider='gray',
			help='Compare numeric measures across categorical groups using aggregated bars, '
				'distribution plots, or hierarchical treemaps.' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		if df_dataset.empty:
			st.info( 'No data loaded.' )
			st.stop( )

		groups = get_visualization_columns( df_dataset )
		usable_categories = get_usable_categorical_columns( df_dataset, groups[ 'categorical' ] )
		if not groups[ 'numeric' ] or not usable_categories:
			st.info( 'At least one numeric field and one categorical field containing multiple '
				'categories are required.' )
			st.stop( )

		c1, c2, c3, c4 = st.columns( 4, border=True )
		with c1:
			measure = st.selectbox( 'Numeric Measure', groups[ 'numeric' ],
				key='visualization_comparison_measure',
				help='Numeric variable summarized or distributed across the selected groups.' )
		with c2:
			category = st.selectbox( 'Grouping Variable', usable_categories,
				key='visualization_comparison_category',
				help='Categorical or ordinal field used to define comparison groups. Single-valued '
					'fields are excluded automatically.' )
		with c3:
			chart_type = st.selectbox( 'Chart', [ 'Bar', 'Column', 'Box', 'Violin', 'Treemap' ],
				key='visualization_comparison_chart_type',
				help='Bar and Column compare aggregated values; Box and Violin show distributions; '
					'Treemap shows relative magnitude within a categorical hierarchy.' )
		with c4:
			category_limit = st.slider( 'Maximum Categories', 2, 30, 12, 1,
				key='visualization_comparison_limit',
				help='Limits the number of groups displayed to keep labels and comparisons readable.' )

		aggregation = 'Mean'
		show_points = False
		hierarchy_columns: List[ str ] = [ category ]
		if chart_type in [ 'Bar', 'Column', 'Treemap' ]:
			option_c1, option_c2 = st.columns( 2, border=True )
			with option_c1:
				aggregation = st.selectbox( 'Aggregation',
					[ 'Sum', 'Mean', 'Median', 'Minimum', 'Maximum', 'Count' ],
					key='visualization_comparison_aggregation',
					help='Determines how observations within each category are summarized.' )
			with option_c2:
				if chart_type == 'Treemap':
					hierarchy_candidates = [ column for column in usable_categories if column != category ]
					secondary = st.selectbox( 'Secondary Hierarchy', [ '<None>' ] + hierarchy_candidates,
						key='visualization_comparison_hierarchy',
						help='Optional second categorical level nested beneath the grouping variable.' )
					if secondary != '<None>':
						hierarchy_columns.append( secondary )
				else:
					st.caption( 'Aggregated charts summarize the measure before plotting.' )
		else:
			show_points = st.checkbox( 'Show Individual Observations', value=True,
				key='visualization_comparison_points',
				help='Overlays individual records on Box or Violin distributions.' )

		df_plot = df_dataset[ list( dict.fromkeys( hierarchy_columns + [ measure ] ) ) ].copy( )
		df_plot[ measure ] = pd.to_numeric( df_plot[ measure ], errors='coerce' )
		df_plot = df_plot.dropna( subset=[ category, measure ] )
		top_categories = df_plot[ category ].value_counts( ).head( category_limit ).index
		df_plot = df_plot[ df_plot[ category ].isin( top_categories ) ].copy( )
		if df_plot.empty:
			st.info( 'No valid grouped observations are available.' )
			st.stop( )

		if chart_type in [ 'Bar', 'Column', 'Treemap' ]:
			df_chart = aggregate_visualization_dataframe( df_plot, hierarchy_columns, measure,
				aggregation )
			if df_chart.empty:
				st.info( 'No aggregated observations are available.' )
				st.stop( )
			if chart_type == 'Bar':
				df_chart = df_chart.sort_values( measure, ascending=True )
				figure = px.bar( df_chart, x=measure, y=category, orientation='h',
					color=measure, color_continuous_scale='Blues', text=measure )
			elif chart_type == 'Column':
				df_chart = df_chart.sort_values( measure, ascending=False )
				figure = px.bar( df_chart, x=category, y=measure, color=measure,
					color_continuous_scale='Blues', text=measure )
			else:
				df_chart[ 'Plot Magnitude' ] = df_chart[ measure ].abs( )
				figure = px.treemap( df_chart, path=hierarchy_columns, values='Plot Magnitude',
					color=measure, color_continuous_scale='RdBu' )
		else:
			points = 'all' if show_points else False
			if chart_type == 'Box':
				figure = px.box( df_plot, x=category, y=measure, color=category, points=points )
			else:
				figure = px.violin( df_plot, x=category, y=measure, color=category,
					box=True, points=points )
			figure.update_xaxes( categoryorder='total descending' )

		render_mathy_plotly_chart( figure, 'visualization_comparison_chart',
			'mathy_category_comparison', f'{chart_type} — {measure} by {category}', 620 )

		df_summary = df_plot.groupby( category, observed=True )[ measure ].agg(
			[ 'count', 'mean', 'median', 'std', 'min', 'max' ] ).reset_index( )
		df_summary.columns = [ category, 'Count', 'Mean', 'Median', 'Std', 'Min', 'Max' ]
		df_summary = df_summary.sort_values( 'Mean', ascending=False, ignore_index=True )
		render_data_editor( df_summary, use_container_width=True, hide_index=True, disabled=True,
			key='visualization_comparison_table' )
		m1, m2, m3, m4 = st.columns( 4, border=True )
		m1.metric( 'Groups', f'{len( df_summary ):,}' )
		m2.metric( 'Observations', f'{len( df_plot ):,}' )
		m3.metric( 'Highest Mean', str( df_summary.iloc[ 0 ][ category ] ) )
		m4.metric( 'Lowest Mean', str( df_summary.iloc[ -1 ][ category ] ) )

# ============================================
# TIME-SERIES VISUALIZATION MODE
# ============================================
elif mode == 'Time-Series Visualization':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Time-Series Visualization', divider='gray',
			help='Visualize numeric measures across time with line, area, or category-ranking '
				'trajectories.' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		if df_dataset.empty:
			st.info( 'No data loaded.' )
			st.stop( )

		groups = get_visualization_columns( df_dataset )
		usable_categories = get_usable_categorical_columns( df_dataset, groups[ 'categorical' ] )
		if not groups[ 'datetime' ] or not groups[ 'numeric' ]:
			st.info( 'At least one datetime and one numeric column are required.' )
			st.stop( )

		chart_options = [ 'Line', 'Area' ]
		if usable_categories:
			chart_options.append( 'Ribbon / Ranking' )

		c1, c2, c3, c4 = st.columns( 4, border=True )
		with c1:
			date_column = st.selectbox( 'Datetime Variable', groups[ 'datetime' ],
				key='visualization_timeseries_date',
				help='Date or timestamp used to order observations and define resampling periods.' )
		with c2:
			value_column = st.selectbox( 'Numeric Variable', groups[ 'numeric' ],
				key='visualization_timeseries_value',
				help='Numeric measure summarized and plotted across time.' )
		with c3:
			chart_type = st.selectbox( 'Chart', chart_options,
				key='visualization_timeseries_chart_type',
				help='Line shows trend, Area emphasizes magnitude, and Ribbon / Ranking shows how '
					'category rank changes across periods.' )
		with c4:
			frequency = st.selectbox( 'Frequency',
				[ 'Original', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annual' ],
				key='visualization_timeseries_frequency',
				help='Optional resampling interval used to summarize observations over time.' )

		option_c1, option_c2, option_c3 = st.columns( 3, border=True )
		with option_c1:
			aggregation = st.selectbox( 'Aggregation',
				[ 'Mean', 'Sum', 'Median', 'Minimum', 'Maximum' ],
				key='visualization_timeseries_aggregation',
				help='Summary statistic applied when multiple observations occur in the same period.' )
		with option_c2:
			if chart_type == 'Ribbon / Ranking':
				rank_category = st.selectbox( 'Ranking Category', usable_categories,
					key='visualization_timeseries_rank_category',
					help='Category whose relative rank is calculated independently within each period.' )
			else:
				show_rolling = st.checkbox( 'Show Rolling Average', value=True,
					key='visualization_timeseries_rolling',
					help='Adds a moving average to smooth short-term variation in the selected measure.' )
		with option_c3:
			if chart_type == 'Ribbon / Ranking':
				rank_limit = st.slider( 'Maximum Ranked Categories', 2, 20, 8, 1,
					key='visualization_timeseries_rank_limit',
					help='Limits the ranking chart to the leading categories by total measure.' )
			else:
				rolling_window = st.slider( 'Rolling Window', 2, 30, 5, 1,
					key='visualization_timeseries_window', disabled=not show_rolling,
					help='Number of plotted periods included in each rolling-average calculation.' )

		frequency_map = { 'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME',
			'Quarterly': 'QE', 'Annual': 'YE' }
		aggregation_map = { 'Mean': 'mean', 'Sum': 'sum', 'Median': 'median',
			'Minimum': 'min', 'Maximum': 'max' }

		if chart_type == 'Ribbon / Ranking':
			df_series = df_dataset[ [ date_column, rank_category, value_column ] ].copy( )
			df_series[ date_column ] = pd.to_datetime( df_series[ date_column ], errors='coerce' )
			df_series[ value_column ] = pd.to_numeric( df_series[ value_column ], errors='coerce' )
			df_series = df_series.dropna( subset=[ date_column, rank_category, value_column ] )
			if df_series.empty:
				st.info( 'No valid ranked time-series observations are available.' )
				st.stop( )
			if frequency == 'Original':
				df_series[ 'Period' ] = df_series[ date_column ].dt.normalize( )
			else:
				df_series = df_series.set_index( date_column ).groupby( rank_category,
					observed=True )[ value_column ].resample( frequency_map[ frequency ] ).agg(
						aggregation_map[ aggregation ] ).dropna( ).reset_index( )
				df_series = df_series.rename( columns={ date_column: 'Period' } )
			if 'Period' not in df_series.columns:
				df_series = df_series.groupby( [ 'Period', rank_category ], observed=True )[
					value_column ].agg( aggregation_map[ aggregation ] ).reset_index( )
			else:
				df_series = df_series.groupby( [ 'Period', rank_category ], observed=True )[
					value_column ].agg( aggregation_map[ aggregation ] ).reset_index( )
			top_categories = df_series.groupby( rank_category, observed=True )[ value_column ].sum( ).nlargest(
				rank_limit ).index
			df_series = df_series[ df_series[ rank_category ].isin( top_categories ) ].copy( )
			df_series[ 'Rank' ] = df_series.groupby( 'Period' )[ value_column ].rank(
				method='dense', ascending=False )
			figure = px.line( df_series, x='Period', y='Rank', color=rank_category, markers=True,
				hover_data=[ value_column ] )
			figure.update_yaxes( autorange='reversed', dtick=1,
				title_text='Rank (1 = highest)' )
			figure.update_xaxes( title_text='Period' )
			render_mathy_plotly_chart( figure, 'visualization_timeseries_chart',
				'mathy_time_series', f'Ranking of {rank_category} by {value_column}', 650 )
			render_data_editor( df_series, use_container_width=True, hide_index=True, disabled=True,
				key='visualization_timeseries_table' )
			m1, m2, m3 = st.columns( 3, border=True )
			m1.metric( 'Periods', f'{df_series[ "Period" ].nunique( ):,}' )
			m2.metric( 'Categories', f'{df_series[ rank_category ].nunique( ):,}' )
			m3.metric( 'Observations', f'{len( df_series ):,}' )
		else:
			df_series = df_dataset[ [ date_column, value_column ] ].copy( )
			df_series[ date_column ] = pd.to_datetime( df_series[ date_column ], errors='coerce' )
			df_series[ value_column ] = pd.to_numeric( df_series[ value_column ], errors='coerce' )
			df_series = df_series.dropna( ).sort_values( date_column )
			if df_series.empty:
				st.info( 'No valid time-series observations are available.' )
				st.stop( )

			if frequency != 'Original':
				df_series = df_series.set_index( date_column ).resample(
					frequency_map[ frequency ] )[ value_column ].agg(
						aggregation_map[ aggregation ] ).dropna( ).reset_index( )

			figure = go.Figure( )
			if chart_type == 'Area':
				figure.add_trace( go.Scatter( x=df_series[ date_column ], y=df_series[ value_column ],
					mode='lines', name=value_column, fill='tozeroy',
					line={ 'color': '#38BDF8', 'width': 2.5 } ) )
			else:
				figure.add_trace( go.Scatter( x=df_series[ date_column ], y=df_series[ value_column ],
					mode='lines+markers', name=value_column,
					line={ 'color': '#38BDF8', 'width': 2.5 }, marker={ 'size': 5 } ) )
			if show_rolling:
				df_series[ 'Rolling Average' ] = df_series[ value_column ].rolling(
					rolling_window, min_periods=1 ).mean( )
				figure.add_trace( go.Scatter( x=df_series[ date_column ],
					y=df_series[ 'Rolling Average' ], mode='lines', name='Rolling Average',
					line={ 'color': '#F472B6', 'width': 3 } ) )
			figure.update_xaxes( rangeslider={ 'visible': True }, rangeselector={ 'buttons': [
				{ 'count': 1, 'label': '1m', 'step': 'month', 'stepmode': 'backward' },
				{ 'count': 6, 'label': '6m', 'step': 'month', 'stepmode': 'backward' },
				{ 'count': 1, 'label': '1y', 'step': 'year', 'stepmode': 'backward' },
				{ 'step': 'all', 'label': 'All' } ] } )
			render_mathy_plotly_chart( figure, 'visualization_timeseries_chart',
				'mathy_time_series', f'{chart_type} — {value_column} over Time', 650 )

			first_value = float( df_series[ value_column ].iloc[ 0 ] )
			last_value = float( df_series[ value_column ].iloc[ -1 ] )
			net_change = last_value - first_value
			percent_change = net_change / first_value * 100.0 if first_value != 0 else np.nan
			m1, m2, m3, m4, m5 = st.columns( 5, border=True )
			m1.metric( 'Observations', f'{len( df_series ):,}' )
			m2.metric( 'Minimum', f'{float( df_series[ value_column ].min( ) ):,.2f}' )
			m3.metric( 'Maximum', f'{float( df_series[ value_column ].max( ) ):,.2f}' )
			m4.metric( 'Net Change', f'{net_change:,.2f}' )
			m5.metric( 'Percent Change', f'{percent_change:,.2f}%'
				if np.isfinite( percent_change ) else 'n/a' )
			render_data_editor( df_series, use_container_width=True, hide_index=True, disabled=True,
				key='visualization_timeseries_table' )

# ============================================
# VARIANCE & DECOMPOSITION MODE
# ============================================
elif mode == 'Variance & Decomposition':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Variance & Decomposition', divider='gray',
			help='Explain numeric change using waterfall contributions, pairwise variance, or '
				'hierarchical decomposition of a measure.' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		if df_dataset.empty:
			st.info( 'No data loaded.' )
			st.stop( )

		groups = get_visualization_columns( df_dataset )
		usable_categories = get_usable_categorical_columns( df_dataset, groups[ 'categorical' ] )
		chart_options: List[ str ] = [ ]
		if usable_categories and groups[ 'numeric' ]:
			chart_options.extend( [ 'Waterfall', 'Decomposition Hierarchy' ] )
		if len( groups[ 'numeric' ] ) >= 2:
			chart_options.append( 'Variance Bars' )
		if not chart_options:
			st.info( 'The loaded data does not satisfy the requirements for variance analysis.' )
			st.stop( )

		chart_type = st.selectbox( 'Visualization', chart_options,
			key='visualization_variance_type',
			help='Waterfall shows sequential contributions; Variance Bars compare two measures; '
				'Decomposition Hierarchy breaks a measure into successive categorical drivers.' )

		if chart_type == 'Waterfall':
			c1, c2, c3, c4 = st.columns( 4, border=True )
			with c1:
				category = st.selectbox( 'Contribution Category', usable_categories,
					key='visualization_waterfall_category',
					help='Categorical field whose aggregated groups form the waterfall steps.' )
			
			with c2:
				measure = st.selectbox( 'Numeric Measure', groups[ 'numeric' ],
					key='visualization_waterfall_measure',
					help='Numeric value aggregated for each contribution step.' )
			
			with c3:
				aggregation = st.selectbox( 'Aggregation',
					[ 'Sum', 'Mean', 'Median', 'Minimum', 'Maximum', 'Count' ],
					key='visualization_waterfall_aggregation',
					help='Summary statistic used to calculate each category contribution.' )
			
			with c4:
				category_limit = st.slider( 'Maximum Categories', 2, 30, 12, 1,
					key='visualization_waterfall_limit',
					help='Maximum number of contribution steps shown before the total.' )

			df_plot = aggregate_visualization_dataframe( df_dataset, [ category ], measure,
				aggregation )
			df_plot = df_plot.reindex( df_plot[ measure ].abs( ).sort_values(
				ascending=False ).index ).head( category_limit )
			
			if df_plot.empty:
				st.info( 'No valid waterfall contributions are available.' )
				st.stop( )
			figure = go.Figure( go.Waterfall( x=df_plot[ category ].astype( str ).tolist( ) + [ 'Total' ],
				y=df_plot[ measure ].astype( float ).tolist( ) + [ 0.0 ],
				measure=[ 'relative' ] * len( df_plot ) + [ 'total' ],
				text=[ f'{value:,.2f}' for value in df_plot[ measure ] ] + [ 'Total' ],
				textposition='outside', connector={ 'line': { 'width': 1 } } ) )
			render_mathy_plotly_chart( figure, 'visualization_waterfall_chart',
				'mathy_waterfall', f'Waterfall — {measure} by {category}', 620 )
			render_data_editor( df_plot, use_container_width=True, hide_index=True, disabled=True,
				key='visualization_waterfall_table' )

		elif chart_type == 'Variance Bars':
			c1, c2, c3, c4 = st.columns( 4, border=True )
			with c1:
				base_measure = st.selectbox( 'Baseline Measure', groups[ 'numeric' ],
					key='visualization_variance_base',
					help='Reference measure subtracted from the comparison measure.' )
			
			with c2:
				comparison_options = [ column for column in groups[ 'numeric' ] if column != base_measure ]
				comparison_measure = st.selectbox( 'Comparison Measure', comparison_options,
					key='visualization_variance_compare',
					help='Measure compared against the baseline.' )
			
			with c3:
				category = st.selectbox( 'Grouping Variable', [ '<None>' ] + usable_categories,
					key='visualization_variance_category',
					help='Optional categorical dimension used to calculate separate variances.' )
			
			with c4:
				display_mode = st.radio( 'Variance', [ 'Absolute', 'Percentage' ], horizontal=True,
					key='visualization_variance_display',
					help='Absolute uses Comparison − Baseline; Percentage divides that difference by '
						'the baseline amount.' )

			if category == '<None>':
				df_plot = df_dataset[ [ base_measure, comparison_measure ] ].copy( )
				for column in [ base_measure, comparison_measure ]:
					df_plot[ column ] = pd.to_numeric( df_plot[ column ], errors='coerce' )
				df_plot = df_plot.dropna( )
				df_plot = pd.DataFrame( { 'Category': [ 'Overall' ],
					'Baseline': [ df_plot[ base_measure ].sum( ) ],
					'Comparison': [ df_plot[ comparison_measure ].sum( ) ] } )
				category_column = 'Category'
			else:
				df_plot = df_dataset[ [ category, base_measure, comparison_measure ] ].copy( )
				for column in [ base_measure, comparison_measure ]:
					df_plot[ column ] = pd.to_numeric( df_plot[ column ], errors='coerce' )
				df_plot = df_plot.dropna( subset=[ category, base_measure, comparison_measure ] )
				df_plot = df_plot.groupby( category, observed=True )[ [ base_measure,
					comparison_measure ] ].sum( ).reset_index( ).rename( columns={
					base_measure: 'Baseline', comparison_measure: 'Comparison' } )
				category_column = category
			df_plot[ 'Variance' ] = df_plot[ 'Comparison' ] - df_plot[ 'Baseline' ]
			if display_mode == 'Percentage':
				df_plot[ 'Variance %' ] = np.where( df_plot[ 'Baseline' ].ne( 0 ),
					df_plot[ 'Variance' ] / df_plot[ 'Baseline' ] * 100.0, np.nan )
				value_column = 'Variance %'
			else:
				value_column = 'Variance'
			df_chart = df_plot.dropna( subset=[ value_column ] ).sort_values( value_column )
			figure = px.bar( df_chart, x=value_column, y=category_column, orientation='h',
				color=value_column, color_continuous_scale='RdBu', text=value_column )
			render_mathy_plotly_chart( figure, 'visualization_variance_bar_chart',
				'mathy_variance_bars', f'{display_mode} Variance — {comparison_measure} vs {base_measure}',
				620 )
			render_data_editor( df_plot, use_container_width=True, hide_index=True, disabled=True,
				key='visualization_variance_table' )

		else:
			c1, c2, c3 = st.columns( 3, border=True )
			with c1:
				measure = st.selectbox( 'Numeric Measure', groups[ 'numeric' ],
					key='visualization_decomposition_measure',
					help='Measure decomposed across the selected hierarchy levels.' )
			
			with c2:
				aggregation = st.selectbox( 'Aggregation',
					[ 'Sum', 'Mean', 'Median', 'Minimum', 'Maximum', 'Count' ],
					key='visualization_decomposition_aggregation',
					help='Summary statistic used at every hierarchy branch.' )
			
			with c3:
				hierarchy_limit = min( 4, len( usable_categories ) )
				if hierarchy_limit == 1:
					if st.session_state.get( 'visualization_decomposition_levels', 1 ) != 1:
						clear_keys( [ 'visualization_decomposition_levels' ] )
					hierarchy_count = st.selectbox( 'Hierarchy Levels', [ 1 ],
						key='visualization_decomposition_levels',
						help='One categorical hierarchy level is available for this dataset.' )
				else:
					hierarchy_count = st.slider( 'Hierarchy Levels', 1, hierarchy_limit,
						min( 2, hierarchy_limit ), 1, key='visualization_decomposition_levels',
						help='Number of categorical levels used to break the measure into nested drivers.' )

			hierarchy_columns: List[ str ] = [ ]
			hierarchy_controls = st.columns( hierarchy_count, border=True )
			available = usable_categories.copy( )
			for index in range( hierarchy_count ):
				with hierarchy_controls[ index ]:
					selected = st.selectbox( f'Hierarchy {index + 1}', available,
						key=f'visualization_decomposition_hierarchy_{index}',
						help='Categorical dimension used at this decomposition level.' )
					hierarchy_columns.append( selected )
				available = [ column for column in available if column != selected ] or available
			
			blue_divider( )
			
			df_plot = aggregate_visualization_dataframe( df_dataset, hierarchy_columns, measure,
				aggregation )
			df_plot[ 'Plot Magnitude' ] = df_plot[ measure ].abs( )
			figure = px.treemap( df_plot, path=hierarchy_columns, values='Plot Magnitude',
				color=measure, color_continuous_scale='RdBu' )
			render_mathy_plotly_chart( figure, 'visualization_decomposition_chart',
				'mathy_decomposition', f'Decomposition Hierarchy — {measure}', 650 )
			
			st.divider( )
			
			render_data_editor( df_plot, use_container_width=True, hide_index=True, disabled=True,
				key='visualization_decomposition_table' )

# ============================================
# COMPOSITION & ALLOCATION MODE
# ============================================
elif mode == 'Composition & Allocation':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Composition & Allocation', divider='gray',
			help='Show how numeric totals are distributed across categorical components using '
				'stacked bars, hierarchical sunbursts, or category-to-category flows.' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		groups = get_visualization_columns( df_dataset )
		usable_categories = get_usable_categorical_columns( df_dataset, groups[ 'categorical' ] )
		if df_dataset.empty or len( usable_categories ) < 2 or not groups[ 'numeric' ]:
			st.info( 'At least two varying categorical fields and one numeric measure are required.' )
			st.stop( )

		chart_type = st.selectbox( 'Visualization', [ 'Stacked Bar', 'Sunburst', 'Sankey' ],
			key='visualization_composition_type',
			help='Stacked Bar compares component contributions; Sunburst shows nested composition; '
				'Sankey shows weighted flow between categories.' )
		if chart_type == 'Stacked Bar':
			c1, c2, c3, c4 = st.columns( 4, border=True )
			with c1:
				x_category = st.selectbox( 'Primary Category', usable_categories,
					key='visualization_stacked_primary', help='Category defining each complete bar.' )
			with c2:
				stack_options = [ column for column in usable_categories if column != x_category ]
				stack_category = st.selectbox( 'Stack Category', stack_options,
					key='visualization_stacked_stack',
					help='Category defining colored components within each bar.' )
			with c3:
				measure = st.selectbox( 'Numeric Measure', groups[ 'numeric' ],
					key='visualization_stacked_measure', help='Numeric value summarized in each stack.' )
			with c4:
				aggregation = st.selectbox( 'Aggregation', [ 'Sum', 'Mean', 'Median', 'Count' ],
					key='visualization_stacked_aggregation',
					help='Summary statistic calculated for each category combination.' )
			df_plot = aggregate_visualization_dataframe( df_dataset,
				[ x_category, stack_category ], measure, aggregation )
			figure = px.bar( df_plot, x=x_category, y=measure, color=stack_category,
				barmode='stack' )
			render_mathy_plotly_chart( figure, 'visualization_stacked_chart',
				'mathy_stacked_bar', f'Stacked {measure} by {x_category}', 650 )
			render_data_editor( df_plot, use_container_width=True, hide_index=True, disabled=True,
				key='visualization_stacked_table' )

		elif chart_type == 'Sunburst':
			c1, c2, c3 = st.columns( 3, border=True )
			with c1:
				measure = st.selectbox( 'Numeric Measure', groups[ 'numeric' ],
					key='visualization_sunburst_measure',
					help='Numeric value controlling sector size within the hierarchy.' )
			with c2:
				aggregation = st.selectbox( 'Aggregation', [ 'Sum', 'Mean', 'Median', 'Count' ],
					key='visualization_sunburst_aggregation',
					help='Summary statistic used for each hierarchy path.' )
			with c3:
				max_levels = min( 4, len( usable_categories ) )
				if max_levels == 2:
					hierarchy_count = st.selectbox( 'Hierarchy Levels', [ 2 ],
						key='visualization_sunburst_levels_fixed',
						help='Two categorical hierarchy levels are available for this dataset.' )
				else:
					hierarchy_count = st.slider( 'Hierarchy Levels', 2, max_levels,
						min( 3, max_levels ), 1, key='visualization_sunburst_levels',
						help='Number of nested categorical levels shown from center outward.' )
			hierarchy_columns: List[ str ] = [ ]
			hierarchy_controls = st.columns( hierarchy_count, border=True )
			available = usable_categories.copy( )
			for index in range( hierarchy_count ):
				with hierarchy_controls[ index ]:
					selected = st.selectbox( f'Hierarchy {index + 1}', available,
						key=f'visualization_sunburst_hierarchy_{index}',
						help='Categorical field used at this sunburst level.' )
					hierarchy_columns.append( selected )
				available = [ column for column in available if column != selected ] or available
			df_plot = aggregate_visualization_dataframe( df_dataset, hierarchy_columns, measure,
				aggregation )
			df_plot[ 'Plot Magnitude' ] = df_plot[ measure ].abs( )
			figure = px.sunburst( df_plot, path=hierarchy_columns, values='Plot Magnitude',
				color=measure, color_continuous_scale='RdBu' )
			render_mathy_plotly_chart( figure, 'visualization_sunburst_chart',
				'mathy_sunburst', f'Sunburst — {measure}', 680 )
			render_data_editor( df_plot, use_container_width=True, hide_index=True, disabled=True,
				key='visualization_sunburst_table' )

		else:
			c1, c2, c3, c4 = st.columns( 4, border=True )
			with c1:
				source_column = st.selectbox( 'Source Category', usable_categories,
					key='visualization_sankey_source',
					help='Categorical field defining the starting nodes in the flow.' )
			with c2:
				target_options = [ column for column in usable_categories if column != source_column ]
				target_column = st.selectbox( 'Target Category', target_options,
					key='visualization_sankey_target',
					help='Categorical field defining destination nodes in the flow.' )
			with c3:
				measure = st.selectbox( 'Weight Measure', groups[ 'numeric' ],
					key='visualization_sankey_measure',
					help='Numeric measure controlling the width of each flow link.' )
			with c4:
				aggregation = st.selectbox( 'Aggregation', [ 'Sum', 'Mean', 'Median', 'Count' ],
					key='visualization_sankey_aggregation',
					help='Summary statistic used to combine repeated source-target links.' )
			figure = create_sankey_from_stages( df_dataset, [ source_column, target_column ],
				measure, aggregation )
			render_mathy_plotly_chart( figure, 'visualization_sankey_chart',
				'mathy_sankey', f'Sankey — {source_column} to {target_column}', 680 )

# ============================================
# PATTERN HEATMAPS MODE
# ============================================
elif mode == 'Pattern Heatmaps':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Pattern Heatmaps', divider='gray',
			help='Use color intensity to reveal concentration or magnitude across two categorical, '
				'ordinal, or time-based dimensions.' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		groups = get_visualization_columns( df_dataset )
		usable_categories = get_usable_categorical_columns( df_dataset, groups[ 'categorical' ] )
		axis_columns = usable_categories + groups[ 'datetime' ]
		if df_dataset.empty or len( axis_columns ) < 2:
			st.info( 'At least two varying categorical, ordinal, or datetime dimensions are required.' )
			st.stop( )

		c1, c2, c3, c4 = st.columns( 4, border=True )
		with c1:
			row_column = st.selectbox( 'Row Dimension', axis_columns,
				key='visualization_heatmap_row',
				help='Field displayed on the vertical heatmap axis.' )
		with c2:
			column_options = [ column for column in axis_columns if column != row_column ]
			column_column = st.selectbox( 'Column Dimension', column_options,
				key='visualization_heatmap_column',
				help='Field displayed on the horizontal heatmap axis.' )
		with c3:
			value_options = [ '<Count>' ] + groups[ 'numeric' ]
			value_column = st.selectbox( 'Value', value_options,
				key='visualization_heatmap_value',
				help='Use Count for frequency intensity or choose a numeric measure to aggregate.' )
		with c4:
			aggregation = st.selectbox( 'Aggregation', [ 'Sum', 'Mean', 'Median', 'Minimum', 'Maximum' ],
				key='visualization_heatmap_aggregation', disabled=value_column == '<Count>',
				help='Summary statistic used when a numeric heatmap value is selected.' )

		df_plot = df_dataset[ [ row_column, column_column ] +
			([] if value_column == '<Count>' else [ value_column ]) ].copy( )
		for dimension in [ row_column, column_column ]:
			if dimension in groups[ 'datetime' ]:
				df_plot[ dimension ] = pd.to_datetime( df_plot[ dimension ], errors='coerce' ).dt.date
		df_plot = df_plot.dropna( subset=[ row_column, column_column ] )
		if value_column == '<Count>':
			df_matrix = pd.crosstab( df_plot[ row_column ], df_plot[ column_column ] )
			value_label = 'Count'
		else:
			df_plot[ value_column ] = pd.to_numeric( df_plot[ value_column ], errors='coerce' )
			df_plot = df_plot.dropna( subset=[ value_column ] )
			aggregation_map = { 'Sum': 'sum', 'Mean': 'mean', 'Median': 'median',
				'Minimum': 'min', 'Maximum': 'max' }
			df_matrix = pd.pivot_table( df_plot, index=row_column, columns=column_column,
				values=value_column, aggfunc=aggregation_map[ aggregation ] )
			value_label = value_column
		figure = go.Figure( data=[ go.Heatmap( z=df_matrix.to_numpy( ),
			x=[ str( value ) for value in df_matrix.columns.tolist( ) ],
			y=[ str( value ) for value in df_matrix.index.tolist( ) ], colorscale='Blues',
			hovertemplate=f'{row_column}: %{{y}}<br>{column_column}: %{{x}}<br>'
				f'{value_label}: %{{z:.4g}}<extra></extra>' ) ] )
		render_mathy_plotly_chart( figure, 'visualization_pattern_heatmap_chart',
			'mathy_pattern_heatmap', f'Heatmap — {row_column} × {column_column}', 650 )

# ============================================
# LIFECYCLE & FLOW MODE
# ============================================
elif mode == 'Lifecycle & Flow':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Lifecycle & Flow', divider='gray',
			help='Follow categorical transitions through ordered stages or display activities across '
				'start and end dates.' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		groups = get_visualization_columns( df_dataset )
		usable_categories = get_usable_categorical_columns( df_dataset, groups[ 'categorical' ] )
		chart_options: List[ str ] = [ ]
		if len( usable_categories ) >= 2:
			chart_options.append( 'Lifecycle Flow' )
		if len( groups[ 'datetime' ] ) >= 2 and usable_categories:
			chart_options.append( 'Gantt Timeline' )
		if not chart_options:
			st.info( 'Lifecycle Flow requires two varying categories; Gantt Timeline requires a '
				'label field and two datetime fields.' )
			st.stop( )

		chart_type = st.selectbox( 'Visualization', chart_options,
			key='visualization_lifecycle_type',
			help='Lifecycle Flow maps stage-to-stage transitions; Gantt Timeline shows activity '
				'duration and overlap.' )
		if chart_type == 'Lifecycle Flow':
			c1, c2, c3 = st.columns( 3, border=True )
			with c1:
				max_stage_count = min( 4, len( usable_categories ) )
				if max_stage_count == 2:
					stage_count = st.selectbox( 'Lifecycle Stages', [ 2 ],
						key='visualization_lifecycle_stage_count_fixed',
						help='Two categorical lifecycle stages are available for this dataset.' )
				else:
					stage_count = st.slider( 'Lifecycle Stages', 2, max_stage_count, 2, 1,
						key='visualization_lifecycle_stage_count',
						help='Number of ordered categorical stages included in the flow.' )
			with c2:
				weight_column = st.selectbox( 'Flow Weight', [ '<Count>' ] + groups[ 'numeric' ],
					key='visualization_lifecycle_weight',
					help='Use Count for record frequency or choose a numeric measure for link width.' )
			with c3:
				aggregation = st.selectbox( 'Aggregation', [ 'Sum', 'Mean', 'Median', 'Count' ],
					key='visualization_lifecycle_aggregation', disabled=weight_column == '<Count>',
					help='Summary statistic applied to the selected numeric flow weight.' )
			stage_columns: List[ str ] = [ ]
			stage_controls = st.columns( stage_count, border=True )
			available = usable_categories.copy( )
			for index in range( stage_count ):
				with stage_controls[ index ]:
					selected = st.selectbox( f'Stage {index + 1}', available,
						key=f'visualization_lifecycle_stage_{index}',
						help='Categorical state used at this point in the lifecycle.' )
				stage_columns.append( selected )
				available = [ column for column in available if column != selected ] or available
			figure = create_sankey_from_stages( df_dataset, stage_columns,
				'' if weight_column == '<Count>' else weight_column, aggregation )
			render_mathy_plotly_chart( figure, 'visualization_lifecycle_flow_chart',
				'mathy_lifecycle_flow', 'Lifecycle Flow', 700 )
		else:
			c1, c2, c3, c4 = st.columns( 4, border=True )
			with c1:
				label_column = st.selectbox( 'Activity Label', usable_categories,
					key='visualization_gantt_label',
					help='Categorical field naming each task, activity, phase, or reporting item.' )
			
			with c2:
				start_column = st.selectbox( 'Start Datetime', groups[ 'datetime' ],
					key='visualization_gantt_start', help='Date or timestamp at which each activity begins.' )
			
			with c3:
				end_options = [ column for column in groups[ 'datetime' ] if column != start_column ]
				end_column = st.selectbox( 'End Datetime', end_options,
					key='visualization_gantt_end', help='Date or timestamp at which each activity ends.' )
			
			with c4:
				color_column = st.selectbox( 'Color Group', [ '<None>' ] + usable_categories,
					key='visualization_gantt_color',
					help='Optional categorical field used to color related activities.' )
			df_plot = df_dataset[ [ label_column, start_column, end_column ] +
				([] if color_column == '<None>' else [ color_column ]) ].copy( )
			df_plot[ start_column ] = pd.to_datetime( df_plot[ start_column ], errors='coerce' )
			df_plot[ end_column ] = pd.to_datetime( df_plot[ end_column ], errors='coerce' )
			df_plot = df_plot.dropna( subset=[ label_column, start_column, end_column ] )
			df_plot = df_plot[ df_plot[ end_column ] >= df_plot[ start_column ] ]
			figure = px.timeline( df_plot, x_start=start_column, x_end=end_column, y=label_column,
				color=None if color_column == '<None>' else color_column,
				hover_data=df_plot.columns.tolist( ) )
			figure.update_yaxes( autorange='reversed' )
			render_mathy_plotly_chart( figure, 'visualization_gantt_chart',
				'mathy_gantt_timeline', f'Gantt Timeline — {label_column}', 680 )
			render_data_editor( df_plot, use_container_width=True, hide_index=True, disabled=True,
				key='visualization_gantt_table' )

# ============================================
# KPI SUMMARY MODE
# ============================================
elif mode == 'KPI Summary':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'KPI Summary', divider='gray',
			help='Summarize selected numeric measures into high-level indicators for rapid '
				'comparison and executive review.' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		groups = get_visualization_columns( df_dataset )
		if df_dataset.empty or not groups[ 'numeric' ]:
			st.info( 'At least one numeric measure is required.' )
			st.stop( )

		c1, c2 = st.columns( [ 0.75, 0.25 ], border=True )
		with c1:
			selected_measures = st.multiselect( 'KPI Measures', groups[ 'numeric' ],
				default=groups[ 'numeric' ][ :min( 4, len( groups[ 'numeric' ] ) ) ],
				key='visualization_kpi_measures',
				help='Select numeric fields to summarize as KPI cards.' )
		
		with c2:
			aggregation = st.selectbox( 'Aggregation',
				[ 'Sum', 'Mean', 'Median', 'Minimum', 'Maximum', 'Count' ],
				key='visualization_kpi_aggregation',
				help='Summary statistic displayed for each selected KPI measure.' )
		if not selected_measures:
			st.info( 'Select at least one KPI measure.' )
			st.stop( )
			
		aggregation_map = { 'Sum': 'sum', 'Mean': 'mean', 'Median': 'median',
			'Minimum': 'min', 'Maximum': 'max', 'Count': 'count' }
		rows: List[ Dict[ str, object ] ] = [ ]
		for row_start in range( 0, len( selected_measures ), 5 ):
			metric_columns = st.columns( 5, border=True )
			row_measures = selected_measures[ row_start: row_start + 5 ]
			for column_index, measure in enumerate( row_measures ):
				series = pd.to_numeric( df_dataset[ measure ], errors='coerce' ).dropna( )
				value = float( getattr( series, aggregation_map[ aggregation ] )( ) ) if not series.empty else np.nan
				with metric_columns[ column_index ]:
					st.metric( measure, f'{value:,.2f}' if np.isfinite( value ) else 'n/a',
						help=f'{aggregation} of populated numeric observations in {measure}.' )
				rows.append( { 'Measure': measure, 'Aggregation': aggregation, 'Value': value,
					'Observations': int( len( series ) ) } )
		
		blue_divider( )
		
		df_kpis = pd.DataFrame( rows )
		render_data_editor( df_kpis, use_container_width=True, hide_index=True, disabled=True,
			key='visualization_kpi_table' )

# ============================================
# MISSING DATA VISUALIZATION MODE
# ============================================
elif mode == 'Missing Data Visualization':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( 'Missing Data Visualization', divider='gray' )
		render_visualization_metric_styles( )
		df_dataset = get_visualization_dataframe( )
		if df_dataset.empty:
			st.info( 'No data loaded.' )
			st.stop( )

		missing_columns = get_visualization_columns( df_dataset )[ 'missing' ]
		if not missing_columns:
			st.info( 'The dataset does not contain missing values.' )
			st.stop( )

		c1, c2, c3 = st.columns( 3, border=True )
		with c1:
			selected_columns = st.multiselect( 'Columns', df_dataset.columns.tolist( ),
				default=missing_columns, key='visualization_missing_columns' )
		with c2:
			display_mode = st.radio( 'Display', [ 'Count', 'Percentage' ], horizontal=True,
				key='visualization_missing_display' )
		with c3:
			max_matrix_rows = min( 500, len( df_dataset ) )
			if max_matrix_rows <= 10:
				matrix_rows = st.selectbox( 'Matrix Rows', [ max_matrix_rows ],
					key='visualization_missing_rows_fixed' )
			else:
				matrix_rows = st.slider( 'Matrix Rows', 10, max_matrix_rows,
					min( 100, max_matrix_rows ), 1, key='visualization_missing_rows' )

		if not selected_columns:
			st.info( 'Select one or more columns.' )
			st.stop( )

		df_selected = df_dataset[ selected_columns ].copy( )
		missing_counts = df_selected.isna( ).sum( )
		df_missing = pd.DataFrame( { 'Column': selected_columns,
			'Missing': [ int( missing_counts[ column ] ) for column in selected_columns ] } )
		df_missing[ 'Missing %' ] = df_missing[ 'Missing' ] / len( df_selected ) * 100.0
		df_missing = df_missing.sort_values( 'Missing %', ascending=False, ignore_index=True )
		value_column = 'Missing' if display_mode == 'Count' else 'Missing %'
		figure = px.bar( df_missing.sort_values( value_column ), x=value_column, y='Column',
			orientation='h', color=value_column, color_continuous_scale='Reds', text=value_column )
		figure.update_traces( texttemplate='%{text:.2f}' if display_mode == 'Percentage' else '%{text}' )
		render_mathy_plotly_chart( figure, 'visualization_missing_bar_chart',
			'mathy_missing_values', 'Missing Values by Column', 520 )
		render_data_editor( df_missing, use_container_width=True, hide_index=True, disabled=True,
			key='visualization_missing_table' )

		blue_divider( )
		st.markdown( '##### Missingness Matrix' )
		if len( df_selected ) > matrix_rows:
			positions = np.linspace( 0, len( df_selected ) - 1, matrix_rows, dtype=int )
			df_matrix = df_selected.iloc[ positions ]
		else:
			df_matrix = df_selected
		matrix_values = df_matrix.notna( ).astype( int ).to_numpy( )
		figure = go.Figure( data=go.Heatmap( z=matrix_values,
			x=df_matrix.columns.tolist( ), y=df_matrix.index.astype( str ).tolist( ),
			zmin=0, zmax=1, colorscale=[ [ 0.0, '#EF4444' ], [ 0.499, '#EF4444' ],
				[ 0.5, '#0F172A' ], [ 1.0, '#0F172A' ] ], showscale=False,
			hovertemplate='Row: %{y}<br>Column: %{x}<br>Present: %{z}<extra></extra>' ) )
		render_mathy_plotly_chart( figure, 'visualization_missing_matrix_chart',
			'mathy_missingness_matrix', 'Present and Missing Cells', 600 )

		total_cells = int( df_dataset.shape[ 0 ] * df_dataset.shape[ 1 ] )
		missing_cells = int( df_dataset.isna( ).sum( ).sum( ) )
		complete_rows = int( df_dataset.notna( ).all( axis=1 ).sum( ) )
		incomplete_rows = len( df_dataset ) - complete_rows
		completeness = (total_cells - missing_cells) / max( 1, total_cells ) * 100.0
		m1, m2, m3, m4, m5 = st.columns( 5, border=True )
		m1.metric( 'Total Cells', f'{total_cells:,}' )
		m2.metric( 'Missing Cells', f'{missing_cells:,}' )
		m3.metric( 'Complete Rows', f'{complete_rows:,}' )
		m4.metric( 'Incomplete Rows', f'{incomplete_rows:,}' )
		m5.metric( 'Completeness', f'{completeness:,.2f}%' )

# ============================================
# DATA MANAGEMENT MODE
# ============================================
elif mode == 'Data Management':
	st.subheader( cfg.MODE[ 'Data Management' ], divider='gray' )
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		tabs = st.tabs( [ 'Import', 'Browse', 'CRUD', 'Explore', 'Filter', 'Aggregate',
			'Visualize', 'Admin', 'SQL' ] )
		
		tables = list_tables( )
		if not tables:
			st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			st.header( '' )
			upl_c1, upl_c2 = st.columns( [ 0.5, 0.5 ], border=True )
			with upl_c1:
				uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
			
			with upl_c2:
				overwrite = st.checkbox( 'Overwrite existing tables', value=True )
				if uploaded_file:
					try:
						sheets = pd.read_excel( uploaded_file, sheet_name=None )
						with create_connection( ) as conn:
							conn.execute( 'BEGIN' )
							for sheet_name, df in sheets.items( ):
								table_name = create_identifier( sheet_name )
								if overwrite:
									conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
								
								# --- Create Table ---
								columns = [ ]
								df.columns = [ create_identifier( c ) for c in df.columns ]
								for col in df.columns:
									sql_type = get_sqlite_type( df[ col ].dtype )
									columns.append( f'"{col}" {sql_type}' )
								
								create_stmt = (f'CREATE TABLE "{table_name}" '
								               f'({", ".join( columns )});')
								
								conn.execute( create_stmt )
								
								# --- Insert Data ---
								placeholders = ", ".join( [ "?" ] * len( df.columns ) )
								insert_stmt = (f'INSERT INTO "{table_name}" '
								               f'VALUES ({placeholders});')
								
								conn.executemany( insert_stmt, df.where(
									pd.notnull( df ), None ).values.tolist( ) )
							
							conn.commit( )
						
						st.success( 'Import completed successfully (transaction committed).' )
						st.rerun( )
					
					except Exception as e:
						try:
							conn.rollback( )
						except:
							pass
						st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				st.header( '' )
				browse_left, browse_center, browse_right = st.columns( 3 )
				with browse_left:
					table = st.selectbox( 'Select Table:', tables, key='table_name' )
				
				blue_divider( )
				df = read_table( table )
				render_data_editor( df, use_container_width=True, height=400 )
			else:
				st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# CRUD (Schema-Aware)
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				st.header( '' )
				st.markdown( '##### Data Table' )
				crud_left, crud_mid, crud_right = st.columns( 3 )
				with crud_left:
					table = st.selectbox( 'Select', tables, key='crud_table' )
				df = read_table( table )
				schema = create_schema( table )
				
				# ------------------------------------------------------------------
				# Build Type Map
				# ------------------------------------------------------------------
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				blue_divider( )
				st.markdown( '##### Insert Row' )
				insert_data = { }
				insert_columns = st.columns( 4 )
				
				for index, (column, col_type) in enumerate( type_map.items( ) ):
					target_column = insert_columns[ index % 4 ]
					
					with target_column:
						if 'INT' in col_type:
							insert_data[ column ] = st.number_input( column, step=1,
								key=f'ins_{column}' )
						
						elif 'REAL' in col_type:
							insert_data[ column ] = st.number_input( column, format='%.6f',
								key=f'ins_{column}' )
						
						elif 'BOOL' in col_type:
							insert_data[ column ] = 1 if st.checkbox( column,
								key=f'ins_{column}' ) else 0
						
						else:
							insert_data[ column ] = st.text_input( column,
								key=f'ins_{column}' )
				
				if st.button( 'Insert Row' ):
					cols = list( insert_data.keys( ) )
					placeholders = ', '.join( [ '?' ] * len( cols ) )
					stmt = f'INSERT INTO "{table}" ({", ".join( cols )}) VALUES ({placeholders});'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( insert_data.values( ) ) )
						conn.commit( )
					
					st.success( 'Row inserted.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				blue_divider( )
				st.markdown( '##### Update Row' )
				rowid = st.number_input( 'Row ID', min_value=1, step=1 )
				update_data = { }
				update_columns = st.columns( 4 )
				
				for index, (column, col_type) in enumerate( type_map.items( ) ):
					target_column = update_columns[ index % 4 ]
					
					with target_column:
						if 'INT' in col_type:
							val = st.number_input( column, step=1,
								key=f'upd_{column}' )
							update_data[ column ] = val
						
						elif 'REAL' in col_type:
							val = st.number_input( column, format='%.6f',
								key=f'upd_{column}' )
							update_data[ column ] = val
						
						elif 'BOOL' in col_type:
							val = 1 if st.checkbox( column,
								key=f'upd_{column}' ) else 0
							update_data[ column ] = val
						
						else:
							val = st.text_input( column, key=f'upd_{column}' )
							update_data[ column ] = val
				
				if st.button( 'Update Row' ):
					set_clause = ', '.join( [ f'{c}=?' for c in update_data ] )
					stmt = f'UPDATE {table} SET {set_clause} WHERE rowid=?;'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
						conn.commit( )
					
					st.success( 'Row updated.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				blue_divider( )
				st.markdown( '##### Delete Row' )
				delete_left, delete_mid, delete_right = st.columns( 3 )
				with delete_left:
					delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1 )
					
				if st.button( 'Delete Row' ):
					with create_connection( ) as conn:
						conn.execute( f'DELETE FROM {table} WHERE rowid=?;', (delete_id,) )
						conn.commit( )
					
					st.success( 'Row deleted.' )
					st.rerun( )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			st.header( '' )
			tables = list_tables( )
			if tables:
				explore_c1, explore_c2, explore_c3 = st.columns( 3, border=True )
				with explore_c1:
					table = st.selectbox( 'Table', tables, key='explore_table' )
				
				with explore_c2:
					page = st.number_input( 'Page', min_value=1, step=1 )
				
				with explore_c3:
					page_size = st.slider( 'Rows per page', 10, 500, 50 )
				
				blue_divider( )
				offset = (page - 1) * page_size
				df_page = read_table( table, page_size, offset )
				render_data_editor( df_page, use_container_width=True, height=400 )
		
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			st.header( '' )
			tables = list_tables( )
			if tables:
				filter_c1, filter_c2, filter_c3 = st.columns( 3, border=True )
				with filter_c1:
					table = st.selectbox( 'Table', tables, key='filter_table' )
					df = read_table( table )
				with filter_c2:
					column = st.selectbox( 'Filter Column', df.columns, key='filter_column_box' )
				with filter_c3:
					value = st.text_input( 'Column Criteria (Contains)' )
					if value:
						df = df[ df[ column ].astype( str ).str.contains( value ) ]
						
				blue_divider( )
				render_data_editor( df, use_container_width=True, key='filter_frame', height=400 )
		
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			st.header( '' )
			tables = list_tables( )
			st.session_state.get( 'aggregation', None )
			if tables:
				agg_c1, agg_c2, agg_c3, agg_c4 = st.columns( 4, border=True )
				with agg_c1:
					table = st.selectbox( 'Table', tables, key='agg_table' )
					df = read_table( table )
				with agg_c2:
					numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
					if numeric_columns:
						col = st.selectbox( 'Column', numeric_columns, key='col_box' )
				with agg_c3:
					aggregation = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ], key='agg_box' )
				with agg_c4:
					if aggregation == 'SUM':
						st.metric( 'Result', df[ col ].sum( ) )
					elif aggregation == 'AVG':
						st.metric( 'Result', df[ col ].mean( ) )
					elif aggregation == 'COUNT':
						st.metric( 'Result', df[ col ].count( ) )
		
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				numeric_columns = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_columns:
					col = st.selectbox( 'Column', numeric_columns, key='numeric_col_box' )
					fig = px.histogram( df, x=col )
					render_mathy_plotly_chart( fig, 'data_management_visualization_chart',
						'mathy_data_management_histogram', f'Histogram — {col}', 500 )
		
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			st.header( '' )
			df_profile = st.session_state.get( 'df_profile' )
			st.markdown( '##### Data Profiling' )
			tables = list_tables( )
			if tables:
				adm_c1, adm_c2, adm_c3 = st.columns( 3 )
				with adm_c1:
					table = st.selectbox( 'Select Table', tables, key='profile_table' )
					
				if st.button( label='Generate Profile', icon='⚡' ):
					df_profile = create_profile_table( table )
			
				render_data_editor( df_profile, use_container_width=True, height=400 )
			
			blue_divider( )
			st.markdown( '##### Drop Table' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( label='Drop Table', key='admin_drop_button', icon='❌' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
					            'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns, key='index_box' )
				
				if st.button( label='Create Index', icon='➕' ):
					create_index( table, col )
					st.success( 'Index created.' )
			
			blue_divider( )
			
			st.markdown( '##### Create Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20,
				value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'##### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( { 'name': col_name, 'type': col_type, 'not_null': not_null,
					'primary_key': primary_key, 'auto_increment': auto_inc } )
			
			if st.button( label='Create Table', icon='➕' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			blue_divider( )
			st.markdown( '##### Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				schema = create_schema( table )
				schema_df = pd.DataFrame( schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "##### Columns" )
				render_data_editor( schema_df, use_container_width=True,
					key='schema_editor', height=400 )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute( f'SELECT COUNT(*) FROM "{table}"' ).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame( indexes,
						columns=[ 'seq', 'name', 'unique', 'origin', 'partial' ] )
					
					st.markdown( "##### Indexes" )
					render_data_editor( idx_df, use_container_width=True,
						key='schema_editor', height=400 )
				else:
					st.info( "No indexes defined." )
			
			blue_divider( )
			st.markdown( "##### ALTER TABLE" )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ],
					key='operation_box' )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
						key='type_box' )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names, key='column_selectbox' )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names, key='drop_box' )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
		
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.markdown( '##### SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv, 'query_results.csv',
								'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )

# ============================================
# DATA UPLOAD MODE
# ============================================
elif mode == 'Data Upload':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Upload' ] )
		st.divider( )
		upl_c1, upl_c2 = st.columns( [ 0.5, 0.5 ], border=True )
		with upl_c1:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ],
				key='data_upload_file' )
		
		with upl_c2:
			overwrite = st.checkbox( 'Overwrite existing tables', value=True,
				key='data_upload_overwrite' )
			
			if uploaded_file is not None:
				file_bytes = uploaded_file.getvalue( )
				upload_signature = (uploaded_file.name, len( file_bytes ), hash( file_bytes ),
					overwrite)
				
				if st.session_state[ 'data_upload_signature' ] != upload_signature:
					try:
						uploaded_file.seek( 0 )
						sheets = pd.read_excel( uploaded_file, sheet_name=None )
						if not sheets:
							raise ValueError( 'The uploaded workbook does not contain any worksheets.' )

						source_sheet_names = list( sheets.keys( ) )
						table_names = create_unique_upload_identifiers( source_sheet_names )
						sheet_table_map = dict( zip( source_sheet_names, table_names ) )
						df_tables = { table_name: sheets[ sheet_name ] for sheet_name, table_name
							in sheet_table_map.items( ) }
						imported_tables = write_dataframe_tables_to_database( df_tables,
							overwrite=overwrite )
						imported_sheet_names = { table_name: sheet_name for sheet_name, table_name
							in sheet_table_map.items( ) }

						st.session_state[ 'data_upload_tables' ] = imported_tables
						st.session_state[ 'data_upload_sheet_names' ] = imported_sheet_names
						st.session_state[ 'data_upload_filename' ] = uploaded_file.name
						st.session_state[ 'data_upload_signature' ] = upload_signature
						first_table_name = next( iter( imported_tables ) )
						activate_database_table( first_table_name )
						queue_database_success(
							f'Imported {len( imported_tables ):,} worksheet table(s) from '
							f'"{uploaded_file.name}". "{first_table_name}" is now the active table.' )
						st.rerun( )
					except Exception as e:
						st.session_state[ 'data_upload_signature' ]=None
						st.error( f'Import failed — transaction rolled back.\n\n{e}' )

		# ------------------------------------------------------------------
		# IMPORTED TABLE SELECTION
		# ------------------------------------------------------------------
		imported_tables = st.session_state.get( 'data_upload_tables', { } )
		imported_sheet_names = st.session_state.get( 'data_upload_sheet_names', { } )
		imported_filename = st.session_state.get( 'data_upload_filename', None )
		if uploaded_file is not None and imported_tables and (
				imported_filename == uploaded_file.name):
			table_options = list( imported_tables.keys( ) )
			if len( table_options ) > 1:
				selected_table = st.selectbox( 'Imported Table', options=table_options,
					format_func=lambda value: (
						f'{imported_sheet_names.get( value, value )} ({value})'),
					key='data_upload_selected_table' )
			else:
				selected_table = table_options[ 0 ]
			
			if st.session_state.get( 'active_database_table', '' ) != selected_table:
				activate_database_table( selected_table )
			df_uploaded = read_table( selected_table )
			st.session_state[ 'data_upload_tables' ][ selected_table ] = df_uploaded.copy( )
			declared_schema = create_declared_schema_map( create_schema( selected_table ) )
			profiles = profile_dataframe_schema( df_uploaded, declared_schema )
			schema = { column: str( profile[ 'analytical_role' ] ) for column, profile in profiles.items( ) }
			type_counts = pd.Series( schema ).value_counts( )
			
			# ------------------------------------------------------------------
			# SCHEMA METRICS
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Types' )
			
			m1, m2, m3, m4, m5 = st.columns( 5, border=True )
			m1.metric( 'Rows', f'{len( df_uploaded ):,}' )
			m2.metric( 'Numeric', int( type_counts.get( 'numeric', 0 ) ) )
			m3.metric( 'Ordinal / ID',
				int( type_counts.get( 'ordinal', 0 ) ) + int( type_counts.get( 'identifier',
					0 ) ) )
			m4.metric( 'Categorical', int( type_counts.get( 'categorical', 0 ) ) )
			m5.metric( 'Datetime', int( type_counts.get( 'datetime', 0 ) ) )
			
			# ------------------------------------------------------------------
			# DATABASE TABLE DISPLAY
			# ------------------------------------------------------------------
			blue_divider( )
			st.markdown( f'##### Data — {selected_table}' )
			
			render_data_editor( df_uploaded, use_container_width=True, height=500, hide_index=True,
				disabled=True, key=f'data_upload_editor_{selected_table}' )
		
# ============================================
# CRUD OPS MODE
# ============================================
elif mode == 'CRUD Ops':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'CRUD Ops' ] )
		blue_divider( )

		st.markdown( '##### Select Data' )
		tables = [ table_name for table_name in list_tables( ) if not table_name.startswith( 'sqlite_' ) ]
		if not tables:
			st.info( 'No database tables are available.' )
		else:
			pending_table = st.session_state.get( 'crud_next_table', '' )
			if pending_table and pending_table in tables:
				st.session_state[ 'crud_table' ] = pending_table
				st.session_state[ 'crud_next_table' ] = ''
			elif st.session_state.get( 'crud_table', None ) not in tables:
				clear_keys( [ 'crud_table' ] )

			crud_c1, crud_c2, crud_c3, crud_c4, crud_c5 = st.columns( 5, border=True )
			with crud_c1:
				table = st.selectbox( 'Select Table', tables, key='crud_table' )
			
			st.divider( )
			
			df_database = read_table( table )
			database_schema = create_schema( table )
			supports_rowid = table_supports_rowid( table )
			df_database_rows = read_table_with_rowid( table ) if supports_rowid else pd.DataFrame( )
			declared_schema = create_declared_schema_map( database_schema )
			crud_profiles = profile_dataframe_schema( df_database, declared_schema )
			crud_type_counts = pd.Series( { column: str( profile[ 'analytical_role' ] )
				for column, profile in crud_profiles.items( ) } ).value_counts( )
			with crud_c2:
				st.metric( 'Total Records', f'{len( df_database ):,}' )
			with crud_c3:
				st.metric( 'Numeric Fields', int( crud_type_counts.get( 'numeric', 0 ) ) )
			with crud_c4:
				st.metric( 'Ordinal Fields', int( crud_type_counts.get( 'ordinal', 0 ) ) )
			with crud_c5:
				st.metric( 'Categorical Fields', int( crud_type_counts.get( 'categorical', 0 ) ) )
			type_map = { str( item[ 1 ] ): str( item[ 2 ] or 'TEXT' ).upper( )
				for item in database_schema }
			primary_key_column_names = { str( item[ 1 ] ) for item in database_schema
				if int( item[ 5 ] or 0 ) > 0 }
			schema_signature = (table, tuple( (str( item[ 1 ] ), str( item[ 2 ] or '' ))
				for item in database_schema ))
			previous_signature = st.session_state.get( 'crud_database_schema_signature', None )
			if previous_signature != schema_signature:
				owned_prefixes = ( 'crud_insert_', 'crud_update_', 'crud_record_',
					'crud_schema_drop_', 'crud_schema_rename_', 'crud_schema_type_',
					'crud_table_manage_' )
				stale_keys = [ key for key in st.session_state.keys( ) if str( key ).startswith(
					owned_prefixes ) ]
				clear_keys( stale_keys )
				st.session_state[ 'crud_database_schema_signature' ] = schema_signature
			
			st.data_editor( df_database )
			
			if not supports_rowid:
				st.warning( 'This table was created WITHOUT ROWID. Record-level CRUD operations are '
					'unavailable, but schema and table-management operations remain available.' )
			else:
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				blue_divider( )
				st.markdown( '##### ➕ Add Record' )
				with st.expander( label='Insert', expanded=True ):
					insert_data_values: Dict[ str, object ] = { }
					primary_key_columns = [ item for item in database_schema if int( item[ 5 ] or 0 ) > 0 ]
					auto_generated_columns = { str( primary_key_columns[ 0 ][ 1 ] ) } if (
						len( primary_key_columns ) == 1 and
						str( primary_key_columns[ 0 ][ 2 ] or '' ).strip( ).upper( ) == 'INTEGER') else set( )
					insert_schema = [ item for item in database_schema if str( item[ 1 ] ) not in
						auto_generated_columns ]
					insert_columns = st.columns( 4 )
					for index, schema_item in enumerate( insert_schema ):
						column = str( schema_item[ 1 ] )
						declared_type = str( schema_item[ 2 ] or 'TEXT' ).upper( )
						target_column = insert_columns[ index % 4 ]
						with target_column:
							unique_values = df_database[ column ].dropna( ).drop_duplicates( ).tolist( )
							choice_key = f'crud_insert_choice_{table}_{index}'
							new_key = f'crud_insert_new_{table}_{index}'
							if unique_values:
								display_map = { str( value ): value for value in unique_values }
								options = [ '<Enter New Value>' ] + list( display_map.keys( ) )
								selection = st.selectbox( column, options, key=choice_key,
									help='Select an existing unique field value or choose Enter New Value.' )
								if selection == '<Enter New Value>':
									value = st.text_input( f'New {column}', key=new_key )
								else:
									value = display_map[ selection ]
							else:
								value = st.text_input( column, key=new_key )
							insert_data_values[ column ] = value
				
					st.divider( )
				
					ins_c1, ins_c2 = st.columns( 2 )
					with ins_c1:
						if st.button( label='Insert Row', icon='➕', width='stretch',
								key='crud_insert_submit_button' ):
							try:
								persisted_values = { column: coerce_sqlite_value( value,
									type_map[ column ] ) for column, value in insert_data_values.items( ) }
								if persisted_values and database_record_exists( table, persisted_values ):
									st.warning( 'This record already exists. No row was inserted.' )
								else:
									with create_connection( ) as conn:
										if persisted_values:
											columns = list( persisted_values.keys( ) )
											column_list = ', '.join( f'"{column}"' for column in columns )
											placeholders = ', '.join( [ '?' ] * len( columns ) )
											statement = f'INSERT INTO "{table}" ({column_list}) VALUES ({placeholders});'
											cursor = conn.execute( statement,
												[ persisted_values[ column ] for column in columns ] )
										else:
											cursor = conn.execute( f'INSERT INTO "{table}" DEFAULT VALUES;' )
										inserted_rowid = int( cursor.lastrowid )
										conn.commit( )
									if persisted_values and not database_record_matches( table, inserted_rowid, persisted_values ):
										raise RuntimeError( 'Inserted row could not be verified.' )
									if not persisted_values and read_database_record( table, inserted_rowid ) is None:
										raise RuntimeError( 'Inserted row could not be verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Row inserted successfully into "{table}".' )
									st.rerun( )
							except Exception as ex:
								st.error( f'Unable to insert row: {ex}' )
				
					with ins_c2:
						st.button( label='Reset', icon='🔄', width='stretch',
							key='crud_insert_reset_button', on_click=reset_session_prefixes,
							args=((f'crud_insert_choice_{table}_', f'crud_insert_new_{table}_'),) )

				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				blue_divider( )
				st.markdown( '##### 📤 Update Data' )
				with st.expander( label='Update', expanded=True ):
					upd_c1, upd_c2, upd_c3, upd_c4 = st.columns( 4 )
					with upd_c1:
						rowid = st.number_input( 'Row ID', min_value=1, step=1,
							key='crud_update_database_rowid' )
					current_record = read_database_record( table, int( rowid ) )
					editable_update_columns = [ column for column in current_record
						if column not in primary_key_column_names ] if current_record else [ ]
					update_values: Dict[ str, object ] = { }
					if current_record is None:
						st.info( 'Row ID not found.' )
					else:
						update_columns = st.columns( 4 )
						for index, column in enumerate( current_record.keys( ) ):
							target_column = update_columns[ index % 4 ]
							with target_column:
								current_value = current_record[ column ]
								update_values[ column ] = st.text_input( column,
									value='' if current_value is None else str( current_value ),
									key=f'crud_update_value_{table}_{int( rowid )}_{index}',
									disabled=column in primary_key_column_names,
									help='Primary-key fields are read-only.' if column in
										primary_key_column_names else None )

					st.divider( )
				
					btn_c1, btn_c2 = st.columns( 2 )
					with btn_c1:
						if st.button( label='Update Row', icon='⬆️', width='stretch',
								key='crud_update_submit_button',
								disabled=current_record is None or not editable_update_columns ):
							try:
								persisted_values = { column: coerce_sqlite_value( value,
									type_map[ column ] ) for column, value in update_values.items( )
									if column not in primary_key_column_names }
								if not persisted_values:
									raise ValueError( 'No editable non-primary-key fields are available.' )
								set_clause = ', '.join( f'"{column}"=?' for column in persisted_values )
								parameters = list( persisted_values.values( ) ) + [ int( rowid ) ]
								with create_connection( ) as conn:
									cursor = conn.execute(
										f'UPDATE "{table}" SET {set_clause} WHERE rowid=?;', parameters )
									conn.commit( )
								if cursor.rowcount == 0:
									st.warning( 'Row ID not found. No row was updated.' )
								else:
									if not database_record_matches( table, int( rowid ), persisted_values ):
										raise RuntimeError( 'Updated row could not be verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Row {int( rowid )} updated successfully in "{table}".' )
									st.rerun( )
							except Exception as ex:
								st.error( f'Unable to update row: {ex}' )
				
					with btn_c2:
						st.button( label='Reset', icon='🔄', width='stretch',
							key='crud_update_reset_button', on_click=reset_session_prefixes,
							args=((f'crud_update_value_{table}_{int( rowid )}_',),) )

				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				blue_divider( )
				st.markdown( '##### ❌ Remove Record' )
				with st.expander( label='Delete', expanded=True ):
					delete_c1, delete_c2, delete_c3, delete_c4 = st.columns( 4 )
					with delete_c1:
						delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1,
							key='crud_delete_database_rowid' )
				
					st.divider( )
				
					delete_btn_c1, delete_btn_c2  = st.columns( 2 )
					with delete_btn_c1:
						if st.button( label='Delete Row', icon='❌', width='stretch',
								key='crud_delete_submit_button' ):
							with create_connection( ) as conn:
								cursor = conn.execute( f'DELETE FROM "{table}" WHERE rowid=?;',
									(int( delete_id ),) )
								conn.commit( )
							if cursor.rowcount == 0:
								st.warning( 'Row ID not found. No row was deleted.' )
							else:
								if read_database_record( table, int( delete_id ) ) is not None:
									st.error( 'Deleted row could not be verified.' )
								else:
									refresh_crud_database_state( table )
									queue_database_success( f'Row {int( delete_id )} deleted successfully from "{table}".' )
									st.rerun( )
					with delete_btn_c2:
						st.button( label='Reset', icon='🔄', width='stretch',
							key='crud_delete_reset_button', on_click=reset_session_keys,
							args=([ 'crud_delete_database_rowid' ],) )

				blue_divider( )

				# -------------------------------------------------------------------------------------
				# EDIT
				# -------------------------------------------------------------------------------------
				st.markdown( '##### 🛠️ Edit Data' )
				with st.expander( label='Update', expanded=True ):
					if df_database_rows.empty:
						st.info( 'The selected table does not contain any records.' )
					else:
						index_c1, index_c2 = st.columns( [ 0.20, 0.80 ] )
						with index_c1:
							max_row_position = len( df_database_rows ) - 1
							default_row_position = int( st.session_state.get(
								'crud_record_row_position', 0 ) )
							default_row_position = max( 0, min( default_row_position,
								max_row_position ) )
							row_position = st.number_input( 'Select Row Position', min_value=0,
								max_value=max_row_position, value=default_row_position, step=1,
								key='crud_record_row_position' )
						selected_row = df_database_rows.iloc[ int( row_position ) ]
						selected_rowid = int( selected_row[ '__mathy_rowid__' ] )
						with index_c2:
							st.caption( f'SQLite Row ID: {selected_rowid}' )

						current_record = { column: selected_row[ column ] for column in df_database.columns }
						editable_record_columns = [ column for column in current_record
							if column not in primary_key_column_names ]
						updated_values: Dict[ str, object ] = { }
						with st.form( f'crud_record_edit_form_{table}_{selected_rowid}' ):
							record_columns = st.columns( 4 )
							for index, column in enumerate( df_database.columns ):
								target_column = record_columns[ index % 4 ]
								with target_column:
									current_value = current_record[ column ]
									updated_values[ column ] = st.text_input( column,
										value='' if pd.isna( current_value ) else str( current_value ),
										key=f'crud_record_value_{table}_{selected_rowid}_{index}',
										disabled=column in primary_key_column_names,
										help='Primary-key fields are read-only.' if column in
											primary_key_column_names else None )
						
							st.divider( )
						
							record_btn_c1, record_btn_c2 = st.columns( 2 )
							with record_btn_c1:
								apply_record = st.form_submit_button( label='Apply Row Update', icon='✔️',
									width='stretch', disabled=not editable_record_columns )
						
							with record_btn_c2:
								st.form_submit_button( label='Reset', icon='🔄', width='stretch',
									on_click=reset_session_prefixes,
									args=((f'crud_record_value_{table}_{selected_rowid}_',),) )

						if apply_record:
							try:
								persisted_values = { column: coerce_sqlite_value( value,
									type_map[ column ] ) for column, value in updated_values.items( )
									if column not in primary_key_column_names }
								if not persisted_values:
									raise ValueError( 'No editable non-primary-key fields are available.' )
								set_clause = ', '.join( f'"{column}"=?' for column in persisted_values )
								with create_connection( ) as conn:
									cursor = conn.execute( f'UPDATE "{table}" SET {set_clause} WHERE rowid=?;',
										list( persisted_values.values( ) ) + [ selected_rowid ] )
									conn.commit( )
								if cursor.rowcount == 0:
									st.warning( 'The selected record no longer exists.' )
								else:
									if not database_record_matches( table, selected_rowid, persisted_values ):
										raise RuntimeError( 'Updated record could not be verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Row position {int( row_position )} updated successfully in "{table}".' )
									st.rerun( )
							except Exception as ex:
								st.error( f'Unable to update record: {ex}' )

				blue_divider( )

			# -------------------------------------------------------------------------------------
			# COLUMN CRUD
			# -------------------------------------------------------------------------------------
			st.markdown( '##### 🏗️ Edit Schema' )
			with st.expander( label='Rename Table', expanded=True ):
				protected_tables = { 'chat_history', 'embeddings', 'Prompts' }
				is_protected_table = table in protected_tables
				table_mgmt_c1, table_mgmt_c2 = st.columns( 2, border=True )
				with table_mgmt_c1:
					new_table_name = st.text_input( 'New Table Name',
						key='crud_table_manage_new_name', disabled=is_protected_table )
					if is_protected_table:
						st.info( 'This application infrastructure table is protected from schema-level table changes.' )
					rename_table_c1, rename_table_c2 = st.columns( 2 )
					with rename_table_c1:
						if st.button( label='Rename Table', icon='✔️', width='stretch',
								key='crud_table_manage_rename_submit', disabled=is_protected_table ):
							try:
								throw_if( 'new_table_name', new_table_name )
								safe_table_name = create_identifier( new_table_name )
								if safe_table_name == table:
									st.info( 'The selected table already uses this name.' )
								elif safe_table_name.lower( ) in { name.lower( ) for name in list_tables( ) }:
									st.error( f'Table "{safe_table_name}" already exists.' )
								else:
									was_active = st.session_state.get( 'active_database_table', '' ) == table
									rename_table( table, safe_table_name )
									current_tables = list_tables( )
									if table in current_tables or safe_table_name not in current_tables:
										raise RuntimeError( 'Renamed table could not be verified.' )
									reconcile_import_table_metadata( table, safe_table_name )
									st.session_state[ 'crud_next_table' ] = safe_table_name
									if was_active:
										activate_database_table( safe_table_name )
									queue_database_success(
										f'Table "{table}" was renamed to "{safe_table_name}".' )
									st.rerun( )
							except Exception as ex:
								st.error( f'Unable to rename table: {ex}' )
					with rename_table_c2:
						st.button( label='Reset', icon='🔄', width='stretch',
							key='crud_table_manage_rename_reset', on_click=reset_session_keys,
							args=([ 'crud_table_manage_new_name' ],) )

				with table_mgmt_c2:
					confirm_delete = st.checkbox( 'Confirm permanent table deletion', value=False,
						key='crud_table_manage_delete_confirm', disabled=is_protected_table )
					if is_protected_table:
						st.info( 'This application infrastructure table is protected from deletion.' )
					delete_table_c1, delete_table_c2 = st.columns( 2 )
					with delete_table_c1:
						if st.button( label='Delete Table', icon='❌', width='stretch',
								key='crud_table_manage_delete_submit',
								disabled=is_protected_table or not confirm_delete ):
							try:
								was_active = st.session_state.get( 'active_database_table', '' ) == table
								drop_table( table )
								if table in list_tables( ):
									raise RuntimeError( 'Deleted table could not be verified.' )
								reconcile_import_table_metadata( table )
								remaining_tables = [ name for name in list_tables( )
									if not name.startswith( 'sqlite_' ) and name not in protected_tables ]
								replacement_table = remaining_tables[ 0 ] if remaining_tables else ''
								if was_active:
									if replacement_table:
										activate_database_table( replacement_table )
									else:
										clear_active_database_table( )
								if replacement_table:
									st.session_state[ 'crud_next_table' ] = replacement_table
								message = f'Table "{table}" was deleted.'
								if was_active and replacement_table:
									message += f' "{replacement_table}" is now the active table.'
								queue_database_success( message )
								st.rerun( )
							except Exception as ex:
								st.error( f'Unable to delete table: {ex}' )
					with delete_table_c2:
						st.button( label='Reset', icon='🔄', width='stretch',
							key='crud_table_manage_delete_reset', on_click=reset_session_keys,
							args=([ 'crud_table_manage_delete_confirm' ],) )

			with st.expander( label='Drop Columns', expanded=True ):
				label_c1, label_c2 = st.columns( 2, border=True )
				with label_c1:
					drop_columns = st.multiselect( 'Columns to Drop', df_database.columns.tolist( ),
						key='crud_schema_drop_columns' )
					drop_btn_c1, drop_btn_c2 = st.columns( 2 )
					with drop_btn_c1:
						if st.button( label='Drop', icon='❌', width='stretch',
								key='crud_schema_drop_submit_button' ):
							if not drop_columns:
								st.info( 'Select one or more columns to drop.' )
							elif len( drop_columns ) == len( df_database.columns ):
								st.error( 'Cannot Drop All Columns.' )
							else:
								try:
									for column in drop_columns:
										drop_column( table, column )
									remaining_columns = [ str( item[ 1 ] ) for item in create_schema( table ) ]
									if any( column in remaining_columns for column in drop_columns ):
										raise RuntimeError( 'Dropped column could not be verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Selected column(s) dropped successfully from "{table}".' )
									st.rerun( )
								except Exception as ex:
									st.error( f'Unable to drop column: {ex}' )
					
					with drop_btn_c2:
						st.button( label='Reset', icon='🔄', width='stretch',
							key='crud_schema_drop_reset_button', on_click=reset_session_keys,
							args=([ 'crud_schema_drop_columns' ],) )

				with label_c2:
					rename_target = st.selectbox( 'Rename Column',
						[ '<None>' ] + df_database.columns.tolist( ), key='crud_schema_rename_column' )
					
					new_column_name = st.text_input( 'New Column Name',
						key='crud_schema_rename_new_column_name' )
					
					rename_btn_c1, rename_btn_c2 = st.columns( 2 )
					with rename_btn_c1:
						if st.button( label='Rename', icon='✔️', width='stretch',
								key='crud_schema_rename_submit_button' ):
							if rename_target == '<None>' or not new_column_name:
								st.info( 'Select a column and enter a new name.' )
							elif new_column_name in df_database.columns:
								st.error( 'Column Name Already Exists.' )
							else:
								try:
									rename_column( table, rename_target, new_column_name )
									updated_names = [ str( item[ 1 ] ) for item in create_schema( table ) ]
									if new_column_name not in updated_names or rename_target in updated_names:
										raise RuntimeError( 'Renamed column could not be verified.' )
									refresh_crud_database_state( table )
									queue_database_success( f'Column "{rename_target}" was renamed to "{new_column_name}" in "{table}".' )
									st.rerun( )
								except Exception as ex:
									st.error( f'Unable to rename column: {ex}' )
					
					with rename_btn_c2:
						st.button( label='Reset', icon='🔄', width='stretch',
							key='crud_schema_rename_reset_button', on_click=reset_session_keys,
							args=([ 'crud_schema_rename_column',
								'crud_schema_rename_new_column_name' ],) )

			st.markdown( '##### 📝 Define Columns' )
			with st.expander( label='Data Types', expanded=True ):
				type_c1, type_c2, type_c3 = st.columns( 3 )
				with type_c1:
					type_column = st.selectbox( 'Column', df_database.columns.tolist( ),
						key='crud_schema_type_column' )
				current_type = type_map.get( type_column, 'TEXT' )
				with type_c2:
					st.text_input( 'Current Data Type', value=current_type, disabled=True,
						key=f'crud_schema_type_current_{table}_{type_column}' )
				supported_types = [ 'INTEGER', 'REAL', 'NUMERIC', 'TEXT', 'BLOB' ]
				default_type = current_type if current_type in supported_types else 'TEXT'
				with type_c3:
					new_type = st.selectbox( 'New Data Type', supported_types,
						index=supported_types.index( default_type ), key='crud_schema_type_new' )
				
				st.divider( )
				
				type_btn_c1, type_btn_c2 = st.columns( 2 )
				with type_btn_c1:
					if st.button( label='Define Type', icon='✔️', width='stretch',
							key='crud_schema_type_submit_button' ):
						if new_type == current_type:
							st.info( 'The selected column already uses this data type.' )
						else:
							try:
								change_column_type( table, type_column, new_type )
								updated_type_map = { str( item[ 1 ] ): str( item[ 2 ] or 'TEXT' ).upper( )
									for item in create_schema( table ) }
								if updated_type_map.get( type_column, '' ) != new_type:
									raise RuntimeError( 'Updated column type could not be verified.' )
								refresh_crud_database_state( table )
								queue_database_success( f'Column "{type_column}" changed to {new_type} in "{table}".' )
								st.rerun( )
							except Exception as ex:
								st.error( f'Unable to change data type: {ex}' )
				
				with type_btn_c2:
					st.button( label='Reset', icon='🔄', width='stretch',
						key='crud_schema_type_reset_button', on_click=reset_session_keys,
						args=([ 'crud_schema_type_column', 'crud_schema_type_new' ],) )

			# -------------------------------------------------------------------------------------
			# DATABASE TABLE CRUD
			# -------------------------------------------------------------------------------------
			st.markdown( '##### 🆕 Create Table' )
			with st.expander( label='Create Data', expanded=True ):
				create_c1, create_c2 = st.columns( 2, border=True )
				with create_c1:
					crud_create_table_name = st.text_input( 'Table Name',
						key='crud_create_table_name', help='Enter the name of the new SQLite table.' )
				with create_c2:
					crud_create_column_count = st.slider( 'Number of Columns', min_value=1,
						max_value=20, value=5, step=1, key='crud_create_column_count',
						help='Number of user-defined columns. Mathy adds an Id primary key automatically.' )
	
				st.caption( 'Mathy automatically adds: Id INTEGER PRIMARY KEY AUTOINCREMENT.' )
				crud_create_columns: List[ Dict[ str, str ] ] = [ ]
				crud_create_supported_types = [ 'INTEGER', 'REAL', 'NUMERIC', 'TEXT', 'BLOB' ]
				for row_start in range( 0, int( crud_create_column_count ), 5 ):
					schema_columns = st.columns( 5, border=True )
					row_end = min( row_start + 5, int( crud_create_column_count ) )
					for column_index in range( row_start, row_end ):
						with schema_columns[ column_index - row_start ]:
							st.markdown( f'**Column {column_index + 1}**' )
							column_name = st.text_input( 'Column Name',
								key=f'crud_create_column_name_{column_index}' )
							column_type = st.selectbox( 'Data Type', crud_create_supported_types,
								key=f'crud_create_column_type_{column_index}' )
							initial_value = st.text_input( 'Initial Value (optional)',
								key=f'crud_create_initial_value_{column_index}' )
							crud_create_columns.append( { 'name': column_name, 'type': column_type,
								'initial_value': initial_value } )
	
				st.divider( )
				create_btn_c1, create_btn_c2 = st.columns( 2 )
				with create_btn_c1:
					if st.button( label='Create Table', icon='➕', width='stretch',
							key='crud_create_table_submit' ):
						try:
							created_table, inserted_initial_row = create_crud_database_table(
								crud_create_table_name, crud_create_columns )
							st.session_state[ 'crud_next_table' ] = created_table
							activate_database_table( created_table )
							message = f'Table "{created_table}" was created successfully.'
							if inserted_initial_row:
								message += ' One initial row was inserted.'
							queue_database_success( message )
							st.rerun( )
						except Exception as ex:
							st.error( f'Unable to create table: {ex}' )
				with create_btn_c2:
					st.button( label='Reset', icon='🔄', width='stretch',
						key='crud_create_table_reset', on_click=reset_session_prefixes,
						args=(( 'crud_create_table_name', 'crud_create_column_count',
							'crud_create_column_', 'crud_create_initial_value_' ),) )
			
			st.divider( )
			
			if st.session_state[ 'pipeline_log' ]:
				st.caption( 'Active Dataset Operations' )
				for step in st.session_state[ 'pipeline_log' ]:
					st.write( f'• {step}' )
			
			blue_divider( )
			
# ============================================
# DATA FILTER MODE
# ============================================
elif mode == 'Data Filter':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Filter' ] )
		st.divider( )
		tables = list_tables( )
		if tables:
			explore_c1, explore_c2, explore_c3 = st.columns( 3, border=True )
			with explore_c1:
				table = st.selectbox( 'Table', tables, key='explore_table' )
			
			with explore_c2:
				page = st.number_input( 'Page', min_value=1, step=1 )
			
			with explore_c3:
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
			
			blue_divider( )
			offset = (page - 1) * page_size
			df_page = read_table( table, page_size, offset )
			render_data_editor( df_page, use_container_width=True, height=400 )

# ============================================
# DATA AGGREGATION MODE
# ============================================
elif mode == 'Data Aggregation':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'Data Aggregation' ] )
		st.divider( )
		tables = list_tables( )
		st.session_state.get( 'aggregation', None )
		if tables:
			agg_c1, agg_c2, agg_c3, agg_c4 = st.columns( 4, border=True )
			with agg_c1:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
				declared_schema = create_declared_schema_map( create_schema( table ) )
				aggregation_profiles = profile_dataframe_schema( df, declared_schema )
			with agg_c2:
				numeric_columns = [ column for column, profile in aggregation_profiles.items( )
					if profile[ 'analytical_role' ] == 'numeric' ]
				col = st.selectbox( 'Column', numeric_columns, key='col_box' ) if numeric_columns else None
				if not numeric_columns:
					st.info( 'No numeric columns are available.' )
			with agg_c3:
				aggregation = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ], key='agg_box',
					disabled=col is None )
			with agg_c4:
				if col is not None:
					numeric_series = pd.to_numeric( df[ col ], errors='coerce' )
					if aggregation == 'SUM':
						st.metric( 'Result', numeric_series.sum( ) )
					elif aggregation == 'AVG':
						st.metric( 'Result', numeric_series.mean( ) )
					elif aggregation == 'COUNT':
						st.metric( 'Result', numeric_series.count( ) )

# ============================================
# SQL CONSOLE MODE
# ============================================
elif mode == 'SQL Console':
	left, center, right = st.columns( [ 0.025, 0.95, 0.025 ] )
	with center:
		st.subheader( cfg.MODE[ 'SQL Console' ] )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Session State
		# ------------------------------------------------------------------
		if 'sql_console_query' not in st.session_state:
			st.session_state[ 'sql_console_query' ] = ''
		
		if 'sql_console_result' not in st.session_state:
			st.session_state[ 'sql_console_result' ] = pd.DataFrame( )
		
		if 'sql_console_elapsed' not in st.session_state:
			st.session_state[ 'sql_console_elapsed' ] = 0.0
		
		if 'sql_console_executed' not in st.session_state:
			st.session_state[ 'sql_console_executed' ] = False
		
		if 'sql_console_table_name' not in st.session_state:
			st.session_state[ 'sql_console_table_name' ] = ''
			
		if 'sql_console_last_saved_table' not in st.session_state:
			st.session_state[ 'sql_console_last_saved_table' ]=None
		
		def reset_sql_console( ) -> None:
			"""Reset SQL Console query and execution state.

			Purpose:
			    Clears the active SQL statement, stored query result, elapsed execution time,
			    execution-status flag, and destination table name so the SQL Console returns to
			    its initial empty state.

			Returns:
			    None: This function updates SQL Console values in Streamlit session state.
			"""
			st.session_state[ 'sql_console_query' ] = ''
			st.session_state[ 'sql_console_result' ] = pd.DataFrame( )
			st.session_state[ 'sql_console_elapsed' ] = 0.0
			st.session_state[ 'sql_console_executed' ] = False
			st.session_state[ 'sql_console_table_name' ] = ''
		
		# ------------------------------------------------------------------
		# Query Editor
		# ------------------------------------------------------------------
		query = st.text_area( 'Enter SQL Query', key='sql_console_query' )
		
		run_c1, run_c2 = st.columns( 2 )
		with run_c1:
			run_query = st.button( 'Run Query', icon='▶️', use_container_width=True,
				key='sql_console_run_query' )
		
		with run_c2:
			st.button( 'Clear Query', icon='🔄', use_container_width=True,
				key='sql_console_clear_query', on_click=reset_sql_console )
		
		# ------------------------------------------------------------------
		# Query Execution
		# ------------------------------------------------------------------
		if run_query:
			if not is_safe_query( query ):
				st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
			else:
				try:
					start_time = time.perf_counter( )
					
					with create_connection( ) as conn:
						df_result = pd.read_sql_query( query, conn )
					
					end_time = time.perf_counter( )
					elapsed = end_time - start_time
					
					st.session_state[ 'sql_console_result' ] = df_result.copy( )
					st.session_state[ 'sql_console_elapsed' ] = elapsed
					st.session_state[ 'sql_console_executed' ] = True
					st.session_state[ 'sql_console_table_name' ] = ''
				
				except Exception as e:
					st.session_state[ 'sql_console_result' ] = pd.DataFrame( )
					st.session_state[ 'sql_console_elapsed' ] = 0.0
					st.session_state[ 'sql_console_executed' ] = False
					st.session_state[ 'sql_console_table_name' ] = ''
					st.error( f'Execution failed: {e}' )
		
		# ------------------------------------------------------------------
		# Query Results
		# ------------------------------------------------------------------
		if st.session_state[ 'sql_console_executed' ]:
			df_result_source = st.session_state[ 'sql_console_result' ].copy( )
			result_profiles = profile_dataframe_schema( df_result_source )
			df_result = create_typed_analysis_dataframe( df_result_source, result_profiles )
			elapsed = float( st.session_state[ 'sql_console_elapsed' ] )
			
			blue_divider( )
			st.markdown( '##### Results' )
			render_data_editor( df_result, use_container_width=True, disabled=True,
				key='sql_console_result_editor' )
			
			row_count = len( df_result )
			column_count = len( df_result.columns )
			
			numeric_columns = [ column for column, profile in result_profiles.items( ) if
				profile[ 'analytical_role' ] == 'numeric' ]
			datetime_columns = [ column for column, profile in result_profiles.items( ) if
				profile[ 'analytical_role' ] == 'datetime' ]
			text_columns = [ column for column in df_result.columns if
				column not in numeric_columns and column not in datetime_columns ]
			
			# --------------------------------------------------------------
			# Execution and Schema Metrics
			# --------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Query Metrics' )
			
			col1, col2, col3, col4, col5, col6 = st.columns( 6, border=True )
			col1.metric( 'Rows Returned', f'{row_count:,}' )
			col2.metric( 'Columns Returned', f'{column_count:,}' )
			col3.metric( 'Numeric Columns', f'{len( numeric_columns ):,}' )
			col4.metric( 'Datetime Columns', f'{len( datetime_columns ):,}' )
			col5.metric( 'Text / Other', f'{len( text_columns ):,}' )
			col6.metric( 'Execution Time', f'{elapsed:.6f}' )
			
			# --------------------------------------------------------------
			# Result Schema
			# --------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Result Schema' )
			
			schema_rows: list[ dict[ str, object ] ] = [ ]
			for column in df_result.columns:
				series = df_result[ column ]
				null_count = int( series.isna( ).sum( ) )
				non_null_count = int( series.notna( ).sum( ) )
				distinct_count = int( series.nunique( dropna=True ) )
				
				schema_rows.append( {
					'Column': column,
					'Data Type': str( series.dtype ),
					'Non-Null': non_null_count,
					'Null': null_count,
					'Distinct': distinct_count } )
			
			df_schema = pd.DataFrame( schema_rows,
				columns=[ 'Column', 'Data Type', 'Non-Null', 'Null', 'Distinct' ] )
			
			render_data_editor( df_schema, use_container_width=True, hide_index=True, disabled=True,
				column_config={
					'Column': st.column_config.TextColumn( 'Column', width='large' ),
					'Data Type': st.column_config.TextColumn( 'Data Type', width='medium' ),
					'Non-Null': st.column_config.NumberColumn( 'Non-Null', format='%d' ),
					'Null': st.column_config.NumberColumn( 'Null', format='%d' ),
					'Distinct': st.column_config.NumberColumn( 'Distinct', format='%d' ) },
				key='sql_console_schema_editor' )
			
			# --------------------------------------------------------------
			# Slow Query Warning
			# --------------------------------------------------------------
			if elapsed > 2.0:
				st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
			
			# --------------------------------------------------------------
			# Save, Export, and Undo Results
			# --------------------------------------------------------------
			blue_divider( )
			st.markdown( '##### Save / Export Results' )
			
			last_saved_table = st.session_state.get(
				'sql_console_last_saved_table', None )
			
			left_container, right_container = st.columns( [ 0.50, 0.50 ] )
			
			with left_container:
				with st.container( border=True, height=87 ):
					table_name = st.text_input( 'New Table Name',
						key='sql_console_table_name',
						help='Creates a new SQLite table from the current query result.' )
			
			with right_container:
				with st.container( border=True, height=87 ):
					st.write( '' )
					
					action_c1, action_c2, action_c3 = st.columns(
						[ 0.34, 0.33, 0.33 ], border=False )
					
					with action_c1:
						save_table = st.button( 'Save as New Table', icon='💾',
							use_container_width=True, key='sql_console_save_table' )
					
					with action_c2:
						csv = df_result.to_csv( index=False ).encode( 'utf-8' )
						
						st.download_button( 'Download CSV', csv, 'query_results.csv',
							'text/csv', icon='📥', use_container_width=True,
							key='sql_console_download', disabled=df_result.empty )
					
					with action_c3:
						undo_save = st.button( 'Undo Last Save', icon='↩️',
							use_container_width=True, key='sql_console_undo_save',
							disabled=last_saved_table is None,
							help=(
								f'Deletes the last table saved by SQL Console: '
								f'{last_saved_table}'
								if last_saved_table else
								'No table has been saved during this session.' ) )
			
			# --------------------------------------------------------------
			# Save Result to Database
			# --------------------------------------------------------------
			if save_table:
				if not table_name or not table_name.strip( ):
					st.error( 'Enter a table name.' )
				
				elif len( df_result.columns ) == 0:
					st.error( 'The query result does not contain any columns to save.' )
				
				elif df_result.columns.duplicated( ).any( ):
					duplicate_columns = df_result.columns[
						df_result.columns.duplicated( ) ].tolist( )
					
					st.error(
						f'The query result contains duplicate column names: {duplicate_columns}. '
						f'Use SQL aliases to make each result column unique.' )
				
				else:
					try:
						safe_table_name = create_identifier( table_name )
						existing_tables = {
							existing_table.lower( ): existing_table
							for existing_table in list_tables( ) }
						
						if safe_table_name.lower( ) in existing_tables:
							st.error(
								f'Table "{existing_tables[ safe_table_name.lower( ) ]}" already '
								f'exists. Enter a different table name.' )
						else:
							with create_connection( ) as conn:
								df_result.to_sql( safe_table_name, conn, if_exists='fail',
									index=False )
							
							st.session_state[
								'sql_console_last_saved_table' ] = safe_table_name
							
							queue_database_success(
								f'Query results saved as table "{safe_table_name}" with '
								f'{row_count:,} row(s) and {column_count:,} column(s).' )
							
							if safe_table_name != table_name.strip( ):
								st.info(
									f'The table name was normalized to "{safe_table_name}".' )
							
							st.rerun( )
					
					except Exception as e:
						st.error( f'Unable to save query results: {e}' )
			
			# --------------------------------------------------------------
			# Undo Last Table Save
			# --------------------------------------------------------------
			if undo_save:
				try:
					existing_tables = {
						existing_table.lower( ): existing_table
						for existing_table in list_tables( ) }
					
					tracked_table = existing_tables.get(
						last_saved_table.lower( ), None )
					
					if tracked_table is None:
						st.session_state[ 'sql_console_last_saved_table' ]=None
						
						st.warning(
							f'Table "{last_saved_table}" no longer exists in the database.' )
					
					else:
						drop_table( tracked_table )
						
						st.session_state[
							'sql_console_last_saved_table' ]=None
						
						queue_database_success(
							f'Table "{tracked_table}" was deleted. '
							f'The query and displayed results were preserved.' )
					
					st.rerun( )
				
				except Exception as e:
					st.error( f'Unable to undo the last table save: {e}' )
