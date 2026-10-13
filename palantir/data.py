"""******************************************************************************************
  Assembly:                mathy
  Filename:                data.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="data.py" company="Terry D. Eppler">

     data.py
     Copyright ©  2023  Terry Eppler

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
    Provides statistical helper functions and the DataSource preparation wrapper used by
    Mathy modeling workflows. The module centralizes entropy, information-gain, impurity,
    clustering-distance, scaling, encoding, profiling, plotting, and train/test split
    utilities for pandas, NumPy, SciPy, seaborn, matplotlib, and scikit-learn pipelines.
</summary>
******************************************************************************************"""
from argparse import ArgumentError
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from .scalers import Scaler, NormalScaler, StandardScaler, MinMaxScaler
import scipy.stats as stats
from typing import Dict, Optional, List, Any
from .boogr import Error, Logger
from .encoders import Encoder, LabelEncoder, TargetEncoder, OrdinalEncoder, OneHotEncoder

def throw_if( name: str, value: object ) -> None:
	"""Validate a required clustering argument.
	
	Purpose:
	    Enforces the presence of required clustering inputs before estimator execution. The
	    validation accepts populated NumPy arrays and standard Python containers while
	    rejecting null values and empty collections that would otherwise cause downstream
	    sklearn operations to fail or produce undefined clustering results.
	
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

def entropy( y: np.ndarray ) -> float | None:
	"""Calculate label entropy.
	
	Purpose:
	    Computes Shannon entropy for a one-dimensional label distribution. The calculation
	    converts observed class counts into probabilities and returns a non-negative
	    impurity measure suitable for split evaluation, decision-tree criteria, and
	    information-gain calculations.
	
	Args:
	    y (np.ndarray): Class-label array used to estimate empirical class probabilities.
	
	Returns:
	    float | None: Entropy value for the observed label distribution.
	
	Raises:
	    Error: Raised when argument validation or entropy calculation fails."""
	try:
		throw_if( 'y', y )
		unique, counts = np.unique( y, return_counts=True )
		probs = counts / len( y )
		_entropy = -np.sum( probs * np.log2( probs + 1e-9 ) )
		return _entropy
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'entropy( y: np.ndarray ) -> float | None'
		Logger( ).write( exception )
		raise exception

def information_gain( X_column: np.ndarray, y: np.ndarray, threshold: float ) -> float | None:
	"""Compute information gain.
	
	Purpose:
	    Calculates the reduction in entropy achieved by splitting a target label vector on a
	    single feature threshold. The result supports decision-tree split selection by
	    comparing parent entropy against the weighted entropy of the left and right
	    partitions.
	
	Args:
	    X_column (np.ndarray): One-dimensional feature array used to form the threshold
	                           split.
	    y (np.ndarray): Target-label array aligned to `X_column`.
	    threshold (float): Split value used to separate left and right partitions.
	
	Returns:
	    float | None: Information-gain value for the requested split, or zero when the split
	                  produces an empty partition.
	
	Raises:
	    Error: Raised when entropy calculation or threshold partitioning fails."""
	try:
		parent_entropy = entropy( y )
		left_idx = X_column <= threshold
		right_idx = X_column > threshold
		if sum( left_idx ) == 0 or sum( right_idx ) == 0:
			return 0  # Avoid splits with empty subset
		left_entropy = entropy( y[ left_idx ] )
		right_entropy = entropy( y[ right_idx ] )
		num_left, num_right = sum( left_idx ), sum( right_idx )
		weighted_entropy = (num_left / len( y )) * left_entropy + (
				num_right / len( y )) * right_entropy
		_information = parent_entropy - weighted_entropy
		return _information
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'information_gain( *args ) -> float | None'
		Logger( ).write( exception )
		raise exception

def best_split( X: np.ndarray, y: np.ndarray, number: int = 10 ) -> Tuple[ int, float ] | None:
	"""Find the best entropy split.
	
	Purpose:
	    Searches each feature column across evenly spaced candidate thresholds and selects
	    the feature-threshold pair with the highest information gain. The returned pair is
	    suitable for constructing a simple decision rule or one-level decision-tree stump.
	
	Args:
	    X (np.ndarray): Feature matrix with rows as samples and columns as candidate split
	                    features.
	    y (np.ndarray): Target-label array aligned to the rows of `X`.
	    number (int): Number of candidate thresholds evaluated per feature.
	
	Returns:
	    Tuple[int, float] | None: Tuple containing the selected feature index and threshold
	                              value.
	
	Raises:
	    Error: Raised when validation, threshold generation, or information-gain scoring
	           fails."""
	try:
		throw_if( 'X', X )
		throw_if( 'y', y )
		best_gain = 0.0
		best_feature = None
		best_threshold = None
		for feature in range( X.shape[ 1 ] ):
			thresholds = np.linspace( X[ :, feature ].min( ), X[ :, feature ].max( ), number )
			for t in thresholds:
				gain = information_gain( X[ :, feature ], y, t )
				if gain > best_gain:
					best_gain = gain
					best_feature = feature
					best_threshold = t
		
		return (best_feature, best_threshold)
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'best_split( *args ) -> Tuple[ int, float ] | None'
		Logger( ).write( exception )
		raise exception

def gini_impurity( p: float ) -> float | None:
	"""Calculate Bernoulli Gini impurity.
	
	Purpose:
	    Computes the Gini impurity for a binary class distribution represented by a success
	    probability. The value measures expected classification impurity and is valid only
	    for probabilities in the inclusive range from zero to one.
	
	Args:
	    p (float): Success probability for the positive class.
	
	Returns:
	    float | None: Gini impurity value for the supplied Bernoulli probability.
	
	Raises:
	    Error: Raised when the probability is missing or outside the valid range."""
	try:
		throw_if( 'p', p )
		if p < 0 or p > 1:
			raise Exception( 'Argument "p" must be in [0, 1]' )
		_impurity = 1.0 - (p ** 2) - ((1.0 - p) ** 2)
		return _impurity
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'gini_impurity( p: float ) -> float'
		Logger( ).write( exception )
		raise exception

def decision_tree_stump( X: np.ndarray, y: np.ndarray, num_thresholds: int = 10 ) -> Dict[ str, Any ]:
	"""Build a one-level decision stump.
	
	Purpose:
	    Creates a single-feature classification rule by selecting the feature and threshold
	    with maximum information gain. The resulting dictionary stores the selected split
	    and majority labels for the left and right partitions.
	
	Args:
	    X (np.ndarray): Feature matrix containing normalized or comparable numeric feature
	                    values.
	    y (np.ndarray): Integer class-label array aligned to the rows of `X`.
	    num_thresholds (int): Number of candidate thresholds evaluated per feature.
	
	Returns:
	    Dict[str, Any]: Dictionary describing the decision stump, or `None` when no valid
	                    split is identified.
	
	Raises:
	    Error: Raised when split selection, partitioning, or majority-label calculation
	           fails."""
	try:
		field, depth = best_split( X, y, num_thresholds )
		if field is None or depth is None:
			return None
		
		left_idx = X[ :, field ] <= depth
		right_idx = X[ :, field ] > depth
		left_label = np.bincount( y[ left_idx ] ).argmax( )
		right_label = np.bincount( y[ right_idx ] ).argmax( )
		return \
			{
					'feature': field,
					'threshold': depth,
					'left_label': left_label,
					'right_label': right_label
			}
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'decision_tree_stump( *args ) -> dict | None'
		Logger( ).write( exception )
		raise exception

def euclidian_distance( X: np.ndarray, centroids: np.ndarray ) -> np.ndarray:
	"""Compute distances to centroids.
	
	Purpose:
	    Calculates Euclidean distances from each sample row to each centroid row. The
	    resulting distance matrix supports assignment steps in K-Means-style clustering
	    workflows by identifying the nearest centroid for every sample.
	
	Args:
	    X (np.ndarray): Feature matrix with rows as samples and columns as features.
	    centroids (np.ndarray): Centroid matrix with rows as cluster centers and columns as
	                            features.
	
	Returns:
	    np.ndarray: Distance matrix where each row corresponds to a sample and each column
	                corresponds to a centroid.
	
	Raises:
	    Error: Raised when validation or distance-matrix construction fails."""
	try:
		throw_if( 'X', X )
		throw_if( 'centroids', centroids )
		_distances = np.zeros( (X.shape[ 0 ], centroids.shape[ 0 ]) )
		for i in range( centroids.shape[ 0 ] ):
			_distances[ :, i ] = np.linalg.norm( X - centroids[ i ], axis=1 )
		return _distances
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'euclidian_distance( *args ) -> np.ndarray'
		Logger( ).write( exception )
		raise exception

def k_means( X: np.ndarray, k: int, iters=10 ) -> Tuple[ np.ndarray, np.ndarray ] | None:
	"""Cluster samples with K-Means.
	
	Purpose:
	    Performs a compact manual K-Means clustering loop using random centroid
	    initialization, Euclidean-distance assignment, centroid recomputation, and early
	    stopping when centroids stabilize. The routine provides a lightweight clustering
	    implementation independent of sklearn estimator classes.
	
	Args:
	    X (np.ndarray): Feature matrix with rows as samples and columns as numeric features.
	    k (int): Number of clusters to form.
	    iters (object): Maximum number of assignment and centroid-update iterations.
	
	Returns:
	    Tuple[np.ndarray, np.ndarray] | None: Tuple containing the assigned cluster labels
	                                          and final centroid matrix.
	
	Raises:
	    Error: Raised when validation, centroid initialization, distance calculation, or
	           updates fail."""
	try:
		throw_if( 'X', X )
		centroids = X[ np.random.choice( X.shape[ 0 ], k, replace=False ) ]
		labels = np.zeros( X.shape[ 0 ], dtype=int )
		for _ in range( iters ):
			distances = euclidian_distance( X, centroids )
			labels = np.argmin( distances, axis=1 )
			new_centroids = np.array(
				[
						X[ labels == i ].mean( axis=0 ) if np.any( labels == i ) else centroids[ i ]
						for i in range( k )
				]
			)
			if np.allclose( centroids, new_centroids ):
				centroids = new_centroids
				break
			
			centroids = new_centroids
		
		return labels, centroids
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'k_means( *args ) -> Tuple[ np.ndarray, np.ndarray ] | None'
		Logger( ).write( exception )
		raise exception

def misclassification_error( p: float ) -> float | None:
	"""Calculate Bernoulli misclassification error.
	
	Purpose:
	    Computes binary misclassification error from a class probability by subtracting the
	    larger class probability from one. The value supports impurity comparisons for
	    simple binary split criteria.
	
	Args:
	    p (float): Success probability for the positive class.
	
	Returns:
	    float | None: Misclassification error rate for the supplied probability.
	
	Raises:
	    Error: Raised when validation or error-rate calculation fails."""
	try:
		throw_if( 'p', p )
		_errors = 1 - np.max( [ p, 1 - p ] )
		return _errors
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'misclassification_error( p: float ) -> float'
		Logger( ).write( exception )
		raise exception

def sigmoid( z: float ) -> float | None:
	"""Calculate the logistic sigmoid.
	
	Purpose:
	    Maps a real-valued input onto the interval from zero to one using the logistic
	    sigmoid transformation. The input is clipped to a stable exponent range before
	    evaluating the exponential expression.
	
	Args:
	    z (float): Real-valued input to transform.
	
	Returns:
	    float | None: Logistic sigmoid value for the supplied input.
	
	Raises:
	    Error: Raised when validation, numeric conversion, clipping, or exponentiation
	           fails."""
	try:
		throw_if( 'z', z )
		z = float( np.clip( z, -709, 709 ) )
		_input = 1.0 / (1.0 + np.exp( -z ))
		return _input
	except Exception as e:
		exception = Error( e )
		exception.module = 'mathy'
		exception.cause = 'data'
		exception.method = 'sigmoid( z: float ) -> float'
		Logger( ).write( exception )
		raise exception

class DataSource( ):
	"""Prepare tabular modeling data.
	
	Purpose:
	    Wraps a pandas DataFrame with derived metadata, descriptive statistics, train/test
	    splits, categorical and numeric partitions, encoding hooks, scaling hooks, pivot
	    creation, export, and profiling utilities. The wrapper preserves the selected target
	    column and maintains synchronized feature, target, numeric, categorical, and split
	    attributes after supported transformations.
	
	Attributes:
	    data (pd.DataFrame): Working dataframe used by transformation and analysis methods.
	    size (float): Test-set proportion used for train/test splitting.
	    seed (int): Random seed used by the splitter.
	    scaler (Optional[Scaler]): Most recent scaler wrapper used by scaling methods.
	    label_encoder (Optional[OrdinalEncoder]): Most recent label or ordinal encoder
	                                              wrapper.
	    target_encoder (Optional[TargetEncoder]): Target encoder reserved for target-aware
	                                              encoding.
	    target (Optional[str]): Name of the selected target column.
	    targets (Optional[np.ndarray]): Current target values from the working dataframe.
	    n_samples (Optional[int]): Number of rows in the working dataframe at
	                               initialization.
	    n_features (Optional[int]): Number of non-target columns at initialization.
	    percentiles (Optional[List[float]]): Percentile boundaries used to summarize numeric
	                                         distributions.
	    scaling_factor (Optional[int]): Multiplicative factor applied during data scaling.
	    feature_names (Optional[List[str]]): Current non-target feature column names.
	    target_names (Optional[np.ndarray]): Sorted unique target values.
	    categorical_columns (Optional[List[str]]): Current categorical column names.
	    numeric_columns (Optional[List[str]]): Current numeric column names.
	    numeric_data (Optional[pd.DataFrame]): Current numeric-column dataframe slice.
	    categorical_data (Optional[pd.DataFrame]): Current categorical-column dataframe
	                                               slice.
	    datatuple (Optional[List[Tuple[str, Encoder, List[str]]]]): ColumnTransformer
	                                                                definitions.
	    numeric_metrics (Optional[pd.DataFrame]): Descriptive statistics for numeric
	                                              columns.
	    pivot_table (Optional[pd.DataFrame]): Most recent pivot table generated by
	                                          `create_pivot`.
	    mean_standard_error (Optional[pd.DataFrame]): Standard error calculated from the
	                                                  feature means.
	    average (Optional[pd.Series]): Averaging strategy or calculated arithmetic-mean
	                                   values.
	    kurtosis (Optional[pd.Series]): Calculated kurtosis values for numeric features.
	    skew (Optional[pd.Series]): Calculated skewness values for numeric features.
	    variance (Optional[pd.Series]): Calculated variance values for numeric features.
	    covariance (Optional[pd.DataFrame]): Calculated covariance matrix for numeric
	                                         features.
	    standard_deviation (Optional[pd.Series]): Calculated standard-deviation values for
	                                              numeric features.
	    column_transformer (Optional[ColumnTransformer]): Most recent fitted column
	                                                      transformer.
	    X_training (Optional[pd.DataFrame]): Feature training split.
	    X_testing (Optional[pd.DataFrame]): Feature testing split.
	    y_training (Optional[pd.Series]): Target training split.
	    y_testing (Optional[pd.Series]): Target testing split."""
	data: pd.DataFrame
	size: float
	seed: int
	scaler: Optional[ Scaler ]
	label_encoder: Optional[ OrdinalEncoder ]
	target_encoder: Optional[ TargetEncoder ]
	target: Optional[ str ]
	targets: Optional[ np.ndarray ]
	n_samples: Optional[ int ]
	n_features: Optional[ int ]
	percentiles: Optional[ List[ float ] ]
	scaling_factor: Optional[ int ]
	feature_names: Optional[ List[ str ] ]
	target_names: Optional[ np.ndarray ]
	categorical_columns: Optional[ List[ str ] ]
	numeric_columns: Optional[ List[ str ] ]
	numeric_data: Optional[ pd.DataFrame ]
	categorical_data: Optional[ pd.DataFrame ]
	datatuple: Optional[ List[ Tuple[ str, Encoder, List[ str ] ] ] ]
	numeric_metrics: Optional[ pd.DataFrame ]
	pivot_table: Optional[ pd.DataFrame ]
	mean_standard_error: Optional[ pd.DataFrame ]
	average: Optional[ pd.Series ]
	kurtosis: Optional[ pd.Series ]
	skew: Optional[ pd.Series ]
	variance: Optional[ pd.Series ]
	covariance: Optional[ pd.DataFrame ]
	standard_deviation: Optional[ pd.Series ]
	column_transformer: Optional[ ColumnTransformer ]
	X_training: Optional[ pd.DataFrame ]
	X_testing: Optional[ pd.DataFrame ]
	y_training: Optional[ pd.Series ]
	y_testing: Optional[ pd.Series ]
	
	def __init__( self, df: pd.DataFrame, target: str, size: float=0.25, rando: int=42 ):
		"""Initialize the data source.
		
		Purpose:
		    Copies the source dataframe, validates the requested target column, derives feature
		    and target metadata, partitions numeric and categorical columns, computes
		    descriptive statistics, initializes transformation state, and creates the initial
		    train/test split used by downstream modeling wrappers.
		
		Args:
		    df (pd.DataFrame): Source dataframe containing feature columns and the target
		                       column.
		    target (str): Name of the target column in `df`.
		    size (float): Proportion of rows reserved for the testing split.
		    rando (int): Random seed used by the train/test splitter.
		
		Returns:
		    None: This method initializes the object and does not return a value.
		
		Raises:
		    ArgumentError: Raised when the requested target column is not present in `df`.
		    ValueError: Raised when required constructor arguments are missing."""
		throw_if( 'df', df )
		throw_if( 'target', target )
		self.data = df.copy( )
		self.size = size
		self.seed = rando
		if target not in df.columns:
			raise ArgumentError( None, f'target "{target}" not in dataframe' )
		
		self.target = target
		self.feature_names = [ c for c in self.data.columns if c != target ]
		self.numeric_columns = self.data.select_dtypes( include=[ 'number' ] ).columns.tolist( )
		self.categorical_columns = self.data.select_dtypes(
			include=[ 'object', 'category', ] ).columns.tolist( )
		self.n_samples = self.data.shape[ 0 ]
		self.n_features = len( self.feature_names )
		self.targets = self.data[ target ].values
		self.target_names = np.array( sorted( np.unique( self.data[ target ].to_numpy( ) ) ) )
		self.numeric_data = self.data[ self.numeric_columns ].copy( )
		self.categorical_data = self.data[ self.categorical_columns ].copy( )
		self.skew = self.data.skew( axis=0, numeric_only=True )
		self.variance = self.data.var( axis=0, ddof=1, numeric_only=True )
		self.kurtosis = self.data.kurt( axis=0, numeric_only=True )
		self.average = self.data.mean( axis=0, numeric_only=True )
		self.mean_standard_error = self.data.sem( axis=0, ddof=1, numeric_only=True )
		self.standard_deviation = self.data.std( axis=0, ddof=1, numeric_only=True )
		self.covariance = self.data.cov( ddof=1, numeric_only=True )
		self.numeric_metrics = self.data[ self.numeric_columns ].describe(
			percentiles=[ .05, .1, .25, .3, .5, .70, .8, .95 ] )
		self.datatuple = [ ]
		self.scaler = None
		self.label_encoder = None
		self.target_encoder = None
		self.pivot_table = None
		self.column_transformer = None
		self.X_training, self.X_testing, self.y_training, self.y_testing = split(
			self.data[ self.feature_names ], self.data[ self.target ], test_size=self.size,
			random_state=self.seed )
	
	def __dir__( self ):
		"""List public members.
		
		Purpose:
		    Returns the stable set of attribute and method names exposed by the data-source
		    wrapper for interactive discovery, IDE inspection, and notebook-oriented
		    exploration.
		
		Returns:
		    None: Public member names exposed by the wrapper."""
		return [ 'data', 'target', 'size', 'seed', 'scaler', 'n_samples', 'n_features', 'targets',
			'target_names', 'feature_names', 'label_encoder', 'target_encoder',
			'categorical_columns', 'numeric_columns', 'pivot_table', 'mean_standard_error',
			'average', 'kurtosis', 'skew', 'variance', 'covariance', 'numeric_data',
			'numeric_metrics', 'standard_deviation', 'categorical_data', 'column_transformer',
			'datatuple', 'X_training', 'X_testing', 'y_training', 'y_testing', # Methods
			'export_excel', 'create_heatmap', 'transform_columns', 'maxminize', 'normalize',
			'standardize', 'encode_targets', 'encode_labels', 'encode_features', 'create_pivot',
			'create_histogram', ]
	
	def transform_columns( self, name: str, encoder: Encoder,
			columns: List[ str ] ) -> pd.DataFrame:
		"""Transform selected columns.
		
		Purpose:
		    Adds a named encoder definition to the internal ColumnTransformer, fits the
		    transformer against current feature columns, rebuilds the working dataframe with
		    transformed and passthrough columns, preserves the target values, refreshes
		    metadata, and recreates the train/test split from the transformed dataset.
		
		Args:
		    name (str): Transformer name used inside the ColumnTransformer definition.
		    encoder (Encoder): Encoder or transformer wrapper applied to the selected columns.
		    columns (List[str]): Column names transformed by the supplied encoder.
		
		Returns:
		    pd.DataFrame: Updated working dataframe after column transformation.
		
		Raises:
		    Error: Raised when validation, transformation, dataframe reconstruction, or split
		           refresh fails."""
		try:
			throw_if( 'name', name )
			throw_if( 'encoder', encoder )
			throw_if( 'columns', columns )
			self.datatuple.append( (name, encoder, columns) )
			self.column_transformer = ColumnTransformer( transformers=self.datatuple,
				remainder='passthrough' )
			values = self.data[ self.feature_names ]
			transformed = self.column_transformer.fit_transform( values )
			passthrough = [ c for c in self.feature_names if c not in columns ]
			feature_names = columns + passthrough
			df_transformed = pd.DataFrame( transformed, columns=feature_names,
				index=self.data.index )
			df_transformed[ self.target ] = self.data[ self.target ].values
			self.data = df_transformed
			self.feature_names = [ c for c in self.data.columns if c != self.target ]
			self.numeric_columns = self.data.select_dtypes( include=[ 'number' ] ).columns.tolist( )
			self.categorical_columns = self.data.select_dtypes(
				include=[ 'object', 'category', ] ).columns.tolist( )
			self.numeric_data = self.data[ self.numeric_columns ].copy( )
			self.categorical_data = self.data[ self.categorical_columns ].copy( )
			self.targets = self.data[ self.target ].values
			self.X_training, self.X_testing, self.y_training, self.y_testing = split(
				self.data[ self.feature_names ], self.data[ self.target ], test_size=self.size,
				random_state=self.seed )
			return self.data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'transform_columns( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def standardize( self ) -> pd.DataFrame:
		"""Standardize numeric data.
		
		Purpose:
		    Applies the sklearn StandardScaler wrapper to the current numeric dataframe slice
		    and returns the standardized numeric matrix. The fitted scaler is stored on the
		    instance for later inspection or inverse transformation when supported by the scaler
		    wrapper.
		
		Returns:
		    pd.DataFrame: Standardized numeric data produced by the scaler wrapper.
		
		Raises:
		    Error: Raised when scaler construction or numeric transformation fails."""
		try:
			self.scaler = StandardScaler( )
			standard_data = self.scaler.train_transform( self.numeric_data )
			return standard_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'standardize( self ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def maxminize( self ) -> pd.DataFrame:
		"""Scale numeric data to a bounded range.
		
		Purpose:
		    Applies the MinMaxScaler wrapper to the current numeric dataframe slice and returns
		    the scaled numeric matrix. The fitted scaler is stored on the instance so the
		    scaling operation remains discoverable after execution.
		
		Returns:
		    pd.DataFrame: Min-max scaled numeric data produced by the scaler wrapper.
		
		Raises:
		    Error: Raised when scaler construction or numeric transformation fails."""
		try:
			self.scaler = MinMaxScaler( )
			standardized_data = self.scaler.train_transform( self.numeric_data )
			return standardized_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'maxminize( self ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def normalize( self ) -> pd.DataFrame:
		"""Normalize numeric data.
		
		Purpose:
		    Applies the NormalScaler to the current numeric dataframe slice and returns the
		    normalized numeric matrix. The fitted scaler is stored on the instance for
		    consistent access after the normalization operation completes.
		
		Returns:
		    pd.DataFrame: Normalized numeric data produced by the scaler wrapper.
		
		Raises:
		    Error: Raised when scaler construction or numeric transformation fails."""
		try:
			self.scaler = NormalScaler( )
			normalized_data = self.scaler.train_transform( self.numeric_data )
			return normalized_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'normalize( self ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def encode_labels( self, col: str ) -> np.ndarray:
		"""Encode a label column.
		
		Purpose:
		    Fits a LabelEncoder wrapper to the specified dataframe column, writes encoded labels
		    back into the working dataframe, and refreshes target, numeric, or categorical
		    cached slices when the encoded column participates in those instance attributes.
		
		Args:
		    col (str): Name of the dataframe column to label-encode.
		
		Returns:
		    np.ndarray: Encoded label array produced by the label encoder.
		
		Raises:
		    Error: Raised when validation, encoder fitting, transformation, or cache refresh
		           fails."""
		try:
			throw_if( 'col', col )
			self.label_encoder = LabelEncoder( )
			y = self.data[ col ].values
			labels = self.label_encoder.train_transform( y )
			self.data[ col ] = labels
			if col == self.target:
				self.targets = self.data[ self.target ].values
			if col in self.numeric_columns:
				self.numeric_data = self.data[ self.numeric_columns ].copy( )
			if col in self.categorical_columns:
				self.categorical_data = self.data[ self.categorical_columns ].copy( )
			return labels
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'encode_labels( self, col: str ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def encode_features( self ) -> pd.DataFrame:
		"""Encode feature columns.
		
		Purpose:
		    Applies an OrdinalEncoder wrapper to every non-target column, rebuilds the working
		    dataframe with encoded features and the unchanged target column, refreshes metadata
		    and cached slices, and recreates the train/test split from the encoded feature set.
		
		Returns:
		    pd.DataFrame: Updated working dataframe with encoded feature columns.
		
		Raises:
		    Error: Raised when feature collection, encoding, dataframe reconstruction, or split
		           refresh fails."""
		try:
			features = [ ]
			self.label_encoder = OrdinalEncoder( )
			for col in self.data.columns:
				if col != self.target:
					features.append( col )
			
			values = self.data[ features ].values
			encoded_features = self.label_encoder.train_transform( values )
			df_encoded = pd.DataFrame( encoded_features, columns=features, index=self.data.index )
			df_encoded[ self.target ] = self.data[ self.target ].values
			self.data = df_encoded
			self.feature_names = [ c for c in self.data.columns if c != self.target ]
			self.numeric_columns = self.data.select_dtypes( include=[ 'number' ] ).columns.tolist( )
			self.categorical_columns = self.data.select_dtypes(
				include=[ 'object', 'category', ] ).columns.tolist( )
			self.numeric_data = self.data[ self.numeric_columns ].copy( )
			self.categorical_data = self.data[ self.categorical_columns ].copy( )
			self.targets = self.data[ self.target ].values
			self.X_training, self.X_testing, self.y_training, self.y_testing = split(
				self.data[ self.feature_names ], self.data[ self.target ], test_size=self.size,
				random_state=self.seed )
			return self.data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'encode_features( self ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def encode_targets( self ) -> np.ndarray:
		"""Encode target values.
		
		Purpose:
		    Fits a LabelEncoder wrapper to the current target values, writes encoded targets
		    back into the working dataframe, refreshes target arrays and target names, and
		    recreates the train/test split with the encoded target column.
		
		Returns:
		    np.ndarray: Encoded target array produced by the label encoder.
		
		Raises:
		    Error: Raised when target encoding, metadata refresh, or split recreation fails."""
		try:
			self.label_encoder = LabelEncoder( )
			encoded_targets = self.label_encoder.train_transform( self.targets )
			self.data[ self.target ] = encoded_targets
			self.targets = self.data[ self.target ].values
			self.target_names = np.array( sorted( np.unique( self.targets ) ) )
			self.X_training, self.X_testing, self.y_training, self.y_testing = split(
				self.data[ self.feature_names ], self.data[ self.target ], test_size=self.size,
				random_state=self.seed )
			return encoded_targets
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'encode_targets( self ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def create_pivot( self, cols: List, vals: List, idx: List ) -> pd.DataFrame:
		"""Create a pivot table.
		
		Purpose:
		    Builds a spreadsheet-style pivot table from the working dataframe using explicit
		    row-index, column-axis, and value-column selections. The generated pivot is cached
		    on the instance for later access by reporting or inspection code.
		
		Args:
		    cols (List): Columns used to define the pivot-table column axis.
		    vals (List): Value columns aggregated inside the pivot table.
		    idx (List): Columns used to define the pivot-table row index.
		
		Returns:
		    pd.DataFrame: Pivot table generated from the working dataframe.
		
		Raises:
		    Error: Raised when validation or pivot-table creation fails."""
		try:
			throw_if( 'cols', cols )
			throw_if( 'vals', vals )
			throw_if( 'idx', idx )
			self.pivot_table = self.data.pivot_table( index=idx, columns=cols, values=vals,
				dropna=True, margins=True )
			return self.pivot_table
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_pivot( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def export_excel( self, filepath: str = None ) -> None:
		"""Export data to Excel.
		
		Purpose:
		    Writes the current working dataframe to an Excel workbook at the supplied file path.
		    The method delegates file creation to pandas and preserves the dataframe exactly as
		    stored on the instance at export time.
		
		Args:
		    filepath (str): Output workbook path passed to pandas `to_excel`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or workbook export fails."""
		try:
			throw_if( 'filepath', filepath )
			self.data.to_excel( filepath )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'export_excel( self, filepath: str=None ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def create_histogram( self ) -> None:
		"""Render a numeric histogram.
		
		Purpose:
		    Aggregates numeric dataframe columns, plots their distribution with seaborn, and
		    renders the histogram through matplotlib for exploratory review of numeric feature
		    totals.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when numeric aggregation or plot rendering fails."""
		try:
			plt.figure( figsize=(8, 6) )
			data = self.data.sum( axis=0, numeric_only=True )
			sns.histplot( data, kde=True, legend=True, line_kws={ 'color': 'red' } )
			plt.title( 'Distributions' )
			plt.xlabel( 'Mean' )
			plt.ylabel( 'Frequency' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_histogram( self ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def create_heatmap( self, numeric: bool = True ) -> None:
		"""Render a correlation heatmap.
		
		Purpose:
		    Computes the Pearson correlation matrix for the working dataframe and renders a
		    seaborn heatmap for exploratory analysis of numeric relationships. The `numeric`
		    flag controls pandas correlation handling for numeric-only selection.
		
		Args:
		    numeric (bool): Flag passed to pandas correlation logic for numeric-only behavior.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when correlation calculation or heatmap rendering fails."""
		try:
			correlations = self.data.corr( 'pearson', numeric_only=numeric )
			plt.figure( figsize=(8, 6) )
			sns.heatmap( correlations, cmap='coolwarm', annot=True )
			plt.title( 'Correlations' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DataSource'
			exception.method = 'create_heatmap( self, numeric: bool=True ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def safe_numeric_series( self, df: pd.DataFrame, col: str ) -> np.ndarray:
		"""Return a clean numeric series.
		
		Purpose:
		    Converts the selected dataframe column to numeric values, coerces invalid entries to
		    missing values, drops missing observations, and returns a one-dimensional float
		    array suitable for descriptive statistics and distribution profiling.
		
		Args:
		    df (pd.DataFrame): Source dataframe containing the requested column.
		    col (str): Column name converted to numeric values.
		
		Returns:
		    np.ndarray: One-dimensional float array with invalid and missing values removed."""
		v = pd.to_numeric( df[ col ], errors="coerce" ).dropna( ).values.astype( float )
		return v
	
	def create_profile( self, df: pd.DataFrame, cols: List[ str ] ) -> pd.DataFrame:
		"""Create a numeric feature profile.
		
		Purpose:
		    Computes an extended descriptive statistics profile for selected numeric columns,
		    including missing-rate, count, mean, standard deviation, variance, percentile,
		    interquartile range, median absolute deviation, skewness, kurtosis, zero-rate,
		    outlier-rate, and normality fields. The returned profile is sorted to prioritize
		    complete columns with higher IQR-based outlier rates.
		
		Args:
		    df (pd.DataFrame): Source dataframe containing the columns to profile.
		    cols (List[str]): Numeric column names included in the profile.
		
		Returns:
		    pd.DataFrame: Dataframe containing one profile row per numeric feature with
		                  descriptive and quality metrics."""
		rows: List[ Dict[ str, Any ] ] = [ ]
		n = df.shape[ 0 ]
		percentiles = [ 0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100 ]
		
		for c in cols:
			v = self.safe_numeric_series( df, c )
			non_missing = int( np.isfinite( v ).sum( ) )
			missing = int( n - non_missing )
			if non_missing == 0:
				continue
			
			q_vals = np.nanpercentile( v, percentiles )
			q = dict( zip( percentiles, q_vals ) )
			
			mean = float( np.nanmean( v ) )
			std = float( np.nanstd( v, ddof=0 ) )
			var = float( np.nanvar( v, ddof=0 ) )
			med = float( np.nanmedian( v ) )
			mad = float( np.nanmedian( np.abs( v - med ) ) )
			iqr = float( q[ 75 ] - q[ 25 ] )
			rng = float( q[ 100 ] - q[ 0 ] )
			
			skew = float( stats.skew( v ) ) if v.size >= 3 else 0.0
			kurt = float( stats.kurtosis( v ) ) if v.size >= 4 else 0.0
			zero_pct = float( (v == 0).mean( ) * 100.0 )
			
			lo = q[ 25 ] - 1.5 * iqr
			hi = q[ 75 ] + 1.5 * iqr
			out_iqr = float( ((v < lo) | (v > hi)).mean( ) * 100.0 )
			
			z = (v - mean) / (std + 1e-12)
			out_z3 = float( (np.abs( z ) > 3.0).mean( ) * 100.0 )
			
			normal_p: Optional[ float ] = None
			try:
				if 8 <= v.size <= 5000:
					_, p = stats.shapiro( v )
					normal_p = float( p )
				elif v.size > 5000:
					_, p = stats.normaltest( v[ :5000 ] )
					normal_p = float( p )
				elif v.size >= 8:
					_, p = stats.normaltest( v )
					normal_p = float( p )
			except Exception:
				normal_p = None
			
			rows.append( { "feature": c, "count": int( v.size ),
				"missing_pct": float( (missing / n) * 100.0 ) if n else 0.0, "mean": mean,
				"std": std, "var": var, "min": float( q[ 0 ] ), "p01": float( q[ 1 ] ),
				"p05": float( q[ 5 ] ), "p10": float( q[ 10 ] ), "q1": float( q[ 25 ] ),
				"median": float( q[ 50 ] ), "q3": float( q[ 75 ] ), "p90": float( q[ 90 ] ),
				"p95": float( q[ 95 ] ), "p99": float( q[ 99 ] ), "max": float( q[ 100 ] ),
				"iqr": iqr, "range": rng, "mad": mad, "skew": skew, "kurtosis": kurt,
				"zero_pct": zero_pct, "outlier_iqr_pct": out_iqr, "outlier_z3_pct": out_z3,
				"normality_p": normal_p, } )
		
		out = pd.DataFrame( rows )
		if out.empty:
			return out
		return out.sort_values( [ "missing_pct", "outlier_iqr_pct" ], ascending=[ True, False ] )
