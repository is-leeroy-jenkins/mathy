"""******************************************************************************************
  Assembly:                Mathy
  Filename:                features.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="features.py" company="Terry D. Eppler">

     features.py
     Copyright ©  2022  Terry Eppler

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
    Provides feature-selection and dimensionality-reduction class/objects for Mathy modeling
    workflows. The module centralizes variance thresholding, canonical correlation analysis,
    principal component analysis, univariate feature selection, sequential backward selection,
    recursive feature elimination, train/test splitting, projection, transformation, and
    scoring utilities behind a consistent selector interface.
</summary>
******************************************************************************************"""
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import sklearn.cross_decomposition as sd
import sklearn.decomposition as sd
from sklearn.decomposition import PCA as PrincipalComponentAnalysis
from sklearn.cross_decomposition import CCA as CanonicalCorrelationAnalysis
import sklearn.feature_selection as sf
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from itertools import combinations
from sklearn.model_selection import train_test_split as split
from .classifications import Classifier, NearestNeighbor
from .boogr import Error, Logger

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

class Selector( ):
	"""Selector feature-selection class/object.
	
	Purpose:
	    Defines the shared feature-selection interface used by dimensionality-reduction and
	    selector class/objects. The base class stores prediction, transformed-data, accuracy, and
	    plotting-marker state and specifies the train, project, transform, fit-transform,
	    scoring, and splitting contracts implemented by concrete selectors.
	
	Attributes:
	    markers (Optional[List[str]]): Matplotlib marker symbols available for selector
	                                   plots.
	    prediction (Optional[np.ndarray]): Most recent classifier predictions generated
	                                       during scoring.
	    transformed_data (Optional[np.ndarray]): Most recent feature matrix produced by
	                                             projection or transformation.
	    accuracy (Optional[float]): Most recent accuracy score produced by selector
	                                evaluation."""
	markers: Optional[ List[ str ] ]
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	
	def __init__( self ) -> None:
		"""Initialize Selector.
		
		Purpose:
		    Initializes shared selector runtime state, plotting markers, prediction storage,
		    transformed-data storage, and accuracy storage used by concrete feature-selection
		    class/objects.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		self.markers = [ '.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*',
			'h', 'H', '+', 'x', 'X', 'd', 'D' ]
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
	
	def split_data( self, X: np.ndarray, y: np.ndarray ) -> tuple:
		"""Split training and testing data.
		
		Purpose:
		    Splits feature and target arrays into aligned training and testing partitions using
		    the configured random seed and test-size proportion.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    tuple: Tuple containing training features, testing features, training targets, and
		           testing targets.
		
		Raises:
		    NotImplementedError: Raised when required fitted estimator metadata is unavailable
		                         or when the base interface is called directly."""
		raise NotImplementedError
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""Fit Selector.
		
		Purpose:
		    Defines the training contract for concrete feature selectors and
		    dimensionality-reduction class/objects.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Fitted selector class/object or fitted estimator result.
		
		Raises:
		    NotImplementedError: Raised when required fitted estimator metadata is unavailable
		                         or when the base interface is called directly."""
		raise NotImplementedError
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Project features with Selector.
		
		Purpose:
		    Defines the projection contract for concrete selectors that convert feature matrices
		    into selected or reduced representations.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Projected or selected feature matrix produced by the fitted selector.
		
		Raises:
		    NotImplementedError: Raised when required fitted estimator metadata is unavailable
		                         or when the base interface is called directly."""
		raise NotImplementedError
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""Transform features with Selector.
		
		Purpose:
		    Defines the transformation contract for fitted selectors and
		    dimensionality-reduction class/objects.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Transformed feature matrix produced by the fitted selector.
		
		Raises:
		    NotImplementedError: Raised when required fitted estimator metadata is unavailable
		                         or when the base interface is called directly."""
		raise NotImplementedError
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""Fit and transform with Selector.
		
		Purpose:
		    Defines the fit-transform contract for selectors that learn and apply a selected or
		    reduced feature representation in one operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Transformed feature matrix produced after fitting the selector.
		
		Raises:
		    NotImplementedError: Raised when required fitted estimator metadata is unavailable
		                         or when the base interface is called directly."""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""Score Selector.
		
		Purpose:
		    Defines the scoring contract for concrete selectors that evaluate model performance
		    on selected or reduced features.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    Dict[str, float] | None: Dataframe or dictionary containing training, testing, and
		                             accuracy metrics.
		
		Raises:
		    NotImplementedError: Raised when required fitted estimator metadata is unavailable
		                         or when the base interface is called directly."""
		raise NotImplementedError

class VarianceThreshold( Selector ):
	"""VarianceThreshold feature-selection class/object.
	
	Purpose:
	    Removes low-variance features with  VarianceThreshold. The
	    class/object stores the threshold configuration, transformed feature matrix, prediction
	    state, and train/test scoring metrics used to evaluate the reduced feature space.
	
	Attributes:
	    model (sf.VarianceThreshold): Underlying sklearn selector, decomposition, or
	                                  cross-decomposition estimator.
	    prediction (Optional[np.ndarray]): Most recent classifier predictions generated
	                                       during scoring.
	    transformed_data (Optional[np.ndarray]): Most recent feature matrix produced by
	                                             projection or transformation.
	    threshold (Optional[float]): Variance threshold used to retain features.
	    accuracy (Optional[float]): Most recent accuracy score produced by selector
	                                evaluation.
	    training_score (Optional[float]): Estimator score calculated on the training split.
	    testing_score (Optional[float]): Estimator score calculated on the testing split."""
	model: sf.VarianceThreshold
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	threshold: Optional[ float ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, thresh: float = 0.0 ) -> None:
		"""Initialize VarianceThreshold.
		
		Purpose:
		    Initializes the variance-threshold selector with the requested variance cutoff,
		    wrapped sklearn estimator, prediction state, transformed-data storage, and score
		    fields.
		
		Args:
		    thresh (float): Variance threshold used to retain features.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.threshold = thresh
		self.model = sf.VarianceThreshold( threshold=self.threshold )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable list of public members exposed by the VarianceThreshold class/object
		    for interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'threshold', 'model', 'prediction', 'transformed_data', 'split_data', 'train',
			'project', 'score', 'transform', 'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split training and testing data.
		
		Purpose:
		    Splits feature and target arrays into aligned training and testing partitions using
		    the configured random seed and test-size proportion.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray,
			y: Optional[ np.ndarray ] = None ) -> sf.VarianceThreshold | None:
		"""Fit VarianceThreshold.
		
		Purpose:
		    Fits the variance-threshold selector to the supplied feature matrix so low-variance
		    columns can be removed.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    sf.VarianceThreshold | None: Fitted selector class/object or fitted estimator result.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'train( self, *args ) -> sf.VarianceThreshold | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Project features with VarianceThreshold.
		
		Purpose:
		    Projects the supplied feature matrix with the fitted VarianceThreshold selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Projected or selected feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame:
		"""Score VarianceThreshold.
		
		Purpose:
		    Evaluates classifier performance after applying the fitted VarianceThreshold
		    selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame: Dataframe or dictionary containing training, testing, and accuracy
		                  metrics.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			if self.transformed_data is None:
				self.transformed_data = self.transform( X )
			_support = self.model.get_support( )
			_original_count = int( X.shape[ 1 ] )
			_retained_count = int( self.transformed_data.shape[ 1 ] )
			_removed_count = int( _original_count - _retained_count )
			_metrics = \
			{
					'Threshold': [ self.threshold ],
					'Original Features': [ _original_count ],
					'Retained Features': [ _retained_count ],
					'Removed Features': [ _removed_count ],
					'Selection Ratio': [ _retained_count / _original_count if _original_count else 0.0 ],
					'Removed Ratio': [
							_removed_count / _original_count if _original_count else 0.0 ],
					'Minimum Retained Variance': [
							float( np.min( self.model.variances_[ _support ] ) )
							if np.any( _support ) else np.nan ],
					'Maximum Retained Variance': [
							float( np.max( self.model.variances_[ _support ] ) )
							if np.any( _support ) else np.nan ],
			}
			return pd.DataFrame( _metrics )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform features with VarianceThreshold.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted VarianceThreshold selector
		    and stores the transformed output.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with VarianceThreshold.
		
		Purpose:
		    Fits the VarianceThreshold selector and returns the transformed feature matrix in
		    one operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VarianceThreshold'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class CCA( Selector ):
	"""CCA feature-selection class/object.
	
	Purpose:
	    Projects paired feature and target matrices with canonical correlation analysis. The
	    class/object fits sklearn cross-decomposition CCA, stores canonical components, supports
	    supervised projections, and evaluates downstream classifier performance on the
	    transformed representation.
	
	Attributes:
	    model (Optional[object]): Underlying sklearn selector, decomposition, or
	                              cross-decomposition estimator.
	    prediction (Optional[np.ndarray]): Most recent classifier predictions generated
	                                       during scoring.
	    n_components (Optional[int]): Number of components retained or projected by the
	                                  selector.
	    scale (Optional[bool]): Flag controlling feature scaling inside canonical
	                            correlation analysis.
	    max_iter (Optional[int]): Maximum solver iterations used by the wrapped estimator.
	    transformed_data (Optional[Tuple[np.ndarray, np.ndarray]]): Most recent feature
	                                                                matrix produced by
	                                                                projection or
	                                                                transformation.
	    accuracy (Optional[float]): Most recent accuracy score produced by selector
	                                evaluation.
	    training_score (Optional[float]): Estimator score calculated on the training split.
	    testing_score (Optional[float]): Estimator score calculated on the testing split."""
	model: Optional[ CanonicalCorrelationAnalysis ]
	prediction: Optional[ np.ndarray ]
	n_components: Optional[ int ]
	scale: Optional[ bool ]
	max_iter: Optional[ int ]
	transformed_data: Optional[ Tuple[ np.ndarray, np.ndarray ] ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, num: int=2, scale: bool=True, size: int=500 ) -> None:
		"""Initialize CCA.
		
		Purpose:
		    Initializes the canonical-correlation selector with component count, scaling
		    behavior, maximum iteration count, wrapped sklearn estimator, transformed-data
		    storage, and score fields.
		
		Args:
		    num (int): Number of components or features retained by the selector.
		    scale (bool): Flag controlling whether CCA scales input variables.
		    size (int): Testing-set proportion used by train/test splitting.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.scale = scale
		self.n_components = num
		self.max_iter = size
		self.model = CanonicalCorrelationAnalysis( n_components=self.n_components, scale=self.scale,
			max_iter=self.max_iter )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable list of public members exposed by the CCA class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'n_components', 'scale', 'max_iter', 'prediction', 'transformed_data',
			'split_data', 'train', 'project', 'score', 'transform', 'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split training and testing data.
		
		Purpose:
		    Splits feature and target arrays into aligned training and testing partitions using
		    the configured random seed and test-size proportion.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Project features with CCA.
		
		Purpose:
		    Projects feature data into the canonical correlation space learned by the fitted CCA
		    estimator.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Projected or selected feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame:
		"""Score CCA.
		
		Purpose:
		    Evaluates classifier performance on canonical-correlation transformed training and
		    testing partitions.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame: Dataframe or dictionary containing training, testing, and accuracy
		                  metrics.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_training, X_testing, y_training, y_testing = self.split_data( X, y )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			y_pred = self.project( X )
			_target_count = int( y.shape[ 1 ] ) if len( y.shape ) > 1 else 1
			_prediction_count = int( y_pred.shape[ 1 ] ) if len( y_pred.shape ) > 1 else 1
			_metrics = \
			{
					'Components': [ self.n_components ],
					'Training Score': [ float( self.training_score ) ],
					'Testing Score': [ float( self.testing_score ) ],
					'Predictor Count': [ int( X.shape[ 1 ] ) ],
					'Target Count': [ _target_count ],
					'Prediction Count': [ _prediction_count ],
			}
			df_metrics = pd.DataFrame( _metrics )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""Fit CCA.
		
		Purpose:
		    Fits canonical correlation analysis to paired feature and target arrays so shared
		    covariance directions can be learned.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Fitted selector class/object or fitted estimator result.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'train( self, *args ) -> object | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> Tuple[ np.ndarray, np.ndarray ] | np.ndarray:
		"""Transform features with CCA.
		
		Purpose:
		    Transforms feature data, and target data when supplied, with the fitted canonical
		    correlation estimator.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray] | np.ndarray: Transformed feature matrix produced by
		                                                the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			if y is None:
				self.transformed_data = self.model.transform( X )
			else:
				self.transformed_data = self.model.transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'transform( self, *args ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray,
		y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ] | np.ndarray:
		"""Fit and transform with CCA.
		
		Purpose:
		    Fits the CCA selector and returns the transformed feature matrix in one operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray] | np.ndarray: Transformed feature matrix produced
		                                                after fitting the selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CCA'
			exception.method = 'train_transform( self, *args ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray'
			Logger( ).write( exception )
			raise exception

class PCA( Selector ):
	"""PCA feature-selection class/object.
	
	Purpose:
	    Projects numeric features into a lower-dimensional principal-component space with
	    sklearn decomposition PCA. The class/object stores component configuration,
	    explained-variance metadata, transformed features, and train/test evaluation metrics
	    for reduced feature sets.
	
	Attributes:
	    model (Optional[object]): Underlying sklearn selector, decomposition, or
	                              cross-decomposition estimator.
	    prediction (Optional[np.ndarray]): Most recent classifier predictions generated
	                                       during scoring.
	    svd_solver (Optional[str]): SVD solver strategy used by PCA.
	    n_components (Optional[int]): Number of components retained or projected by the
	                                  selector.
	    transformed_data (Optional[np.ndarray]): Most recent feature matrix produced by
	                                             projection or transformation.
	    accuracy (Optional[float]): Most recent accuracy score produced by selector
	                                evaluation.
	    training_score (Optional[float]): Estimator score calculated on the training split.
	    testing_score (Optional[float]): Estimator score calculated on the testing split."""
	model: Optional[ PrincipalComponentAnalysis ]
	prediction: Optional[ np.ndarray ]
	svd_solver: Optional[ str ]
	n_components: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, num: int = 2, solver: str = 'auto' ) -> None:
		"""Initialize PCA.
		
		Purpose:
		    Initializes the principal-component selector with component count, SVD solver
		    configuration, wrapped sklearn estimator, transformed-data storage, and score
		    fields.
		
		Args:
		    num (int): Number of components or features retained by the selector.
		    solver (str): SVD solver strategy passed to the PCA estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_components = num
		self.svd_solver = solver
		self.model = PrincipalComponentAnalysis( n_components=self.n_components,
			svd_solver=self.svd_solver )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable list of public members exposed by the PCA class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'svd_solver', 'n_components', 'transformed_data',
			'split_data', 'train', 'project', 'score', 'transform', 'train_transform' ]
	
	@property
	def explained_variance_ratio( self ) -> float:
		"""Performs the `explained_variance_ratio` operation defined by `PCA`.
		
		Purpose:
		    Performs the `explained_variance_ratio` operation defined by `PCA`.
		
		Returns:
		    float: Value produced by the operation."""
		if self.model is not None:
			return self.model.explained_variance_ratio_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split training and testing data.
		
		Purpose:
		    Splits feature and target arrays into aligned training and testing partitions using
		    the configured random seed and test-size proportion.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""Fit PCA.
		
		Purpose:
		    Fits principal component analysis to the supplied feature matrix so principal axes
		    and explained variance can be learned.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Fitted selector class/object or fitted estimator result.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'train( self, *args ) -> object | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Project features with PCA.
		
		Purpose:
		    Projects feature data into the principal-component space learned by the fitted PCA
		    estimator.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Projected or selected feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame:
		"""Score PCA.
		
		Purpose:
		    Evaluates classifier performance on PCA-transformed training and testing partitions.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame: Dataframe or dictionary containing training, testing, and accuracy
		                  metrics.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			if self.transformed_data is None:
				self.transformed_data = self.transform( X )
			_original_count = int( X.shape[ 1 ] )
			_component_count = int( self.transformed_data.shape[ 1 ] )
			_total_explained = float( np.sum( self.model.explained_variance_ratio_ ) ) \
				if hasattr( self.model, 'explained_variance_ratio_' ) else np.nan
			_metrics = { 'Original Features': [ _original_count ],
				'Components': [ _component_count ],
				'Explained Variance Total': [ _total_explained ], 'Largest Component Variance': [
					float( np.max( self.model.explained_variance_ratio_ ) ) if hasattr( self.model,
						'explained_variance_ratio_' ) else np.nan ],
				'Smallest Component Variance': [
					float( np.min( self.model.explained_variance_ratio_ ) ) if hasattr( self.model,
						'explained_variance_ratio_' ) else np.nan ],
				'Solver': [ self.svd_solver ], }
			df_metrics = pd.DataFrame( _metrics )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform features with PCA.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted PCA selector and stores the
		    transformed output.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with PCA.
		
		Purpose:
		    Fits the PCA selector and returns the transformed feature matrix in one operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'PCA'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class SelectBest( Selector ):
	"""SelectBest feature-selection class/object.
	
	Purpose:
	    Selects the top scoring features with sklearn.feature_selection.SelectKBest. The
	    class/object applies univariate statistical tests, stores scores and transformed
	    matrices, and evaluates classifier performance after retaining the configured number
	    of features.
	
	Attributes:
	    model (Optional[object]): Underlying sklearn selector, decomposition, or
	                              cross-decomposition estimator.
	    prediction (Optional[np.ndarray]): Most recent classifier predictions generated
	                                       during scoring.
	    score_function (Optional[object]): Univariate scoring function used to rank
	                                       candidate features.
	    n_features (Optional[int]): Number of top-ranked features retained by SelectBest.
	    transformed_data (Optional[np.ndarray]): Most recent feature matrix produced by
	                                             projection or transformation.
	    accuracy (Optional[float]): Most recent accuracy score produced by selector
	                                evaluation.
	    training_score (Optional[float]): Estimator score calculated on the training split.
	    testing_score (Optional[float]): Estimator score calculated on the testing split."""
	model: Optional[ sf.SelectKBest ]
	prediction: Optional[ np.ndarray ]
	score_function: Optional[ object ]
	n_features: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, score_func: object = sf.chi2, num: int = 10 ) -> None:
		"""Initialize SelectBest.
		
		Purpose:
		    Initializes the top-k univariate selector with its scoring function, feature count,
		    wrapped sklearn estimator, transformed-data storage, and score fields.
		
		Args:
		    score_func (object): Univariate score function used to rank candidate features.
		    num (int): Number of components or features retained by the selector.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.score_function = score_func
		self.n_features = num
		self.model = sf.SelectKBest( score_func=self.score_function, k=self.n_features )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable list of public members exposed by the SelectBest class/object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'score_function', 'n_features', 'transformed_data',
			'split_data', 'chi_square', 'train', 'project', 'score', 'transform',
			'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split training and testing data.
		
		Purpose:
		    Splits feature and target arrays into aligned training and testing partitions using
		    the configured random seed and test-size proportion.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def chi_square( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ] | None:
		"""Calculate chi-square scores.
		
		Purpose:
		    Computes chi-square feature scores and p-values for the supplied feature matrix and
		    target vector using the SelectBest scoring configuration.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray] | None: Tuple containing chi-square scores and
		                                          p-values.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return sf.chi2( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'chi_square( self, *args ) -> Tuple[np.ndarray, np.ndarray] | None'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""Fit SelectBest.
		
		Purpose:
		    Fits the top-k univariate selector to feature and target arrays so the highest
		    scoring features can be retained.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Fitted selector class/object or fitted estimator result.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'train( self, *args ) -> object | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Project features with SelectBest.
		
		Purpose:
		    Projects the supplied feature matrix with the fitted SelectBest selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Projected or selected feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame:
		"""Score SelectBest.
		
		Purpose:
		    Evaluates classifier performance after applying the fitted SelectBest selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame: Dataframe or dictionary containing training, testing, and accuracy
		                  metrics.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			if self.transformed_data is None:
				self.transformed_data = self.transform( X )
			_support = self.model.get_support( )
			_scores = self.model.scores_
			_pvalues = self.model.pvalues_ if hasattr( self.model, 'pvalues_' ) else None
			df_scores = pd.DataFrame( { 'Feature': np.arange( 0, X.shape[ 1 ] ), 'Score': _scores,
				'PValue': _pvalues if _pvalues is not None else np.full( X.shape[ 1 ], np.nan ),
				'Selected': _support, } )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform features with SelectBest.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted SelectBest selector and
		    stores the transformed output.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with SelectBest.
		
		Purpose:
		    Fits the SelectBest selector and returns the transformed feature matrix in one
		    operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectBest'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class SelectPercent( Selector ):
	"""SelectPercent feature-selection class/object.
	
	Purpose:
	    Selects the highest scoring feature percentile with
	    sklearn.feature_selection.SelectPercentile. The object applies univariate
	    statistical tests, stores transformed matrices, and evaluates classifier performance
	    after retaining the configured percentage of features.
	
	Attributes:
	    model (Optional[object]): Underlying sklearn selector, decomposition, or
	                              cross-decomposition estimator.
	    prediction (Optional[np.ndarray]): Most recent classifier predictions generated
	                                       during scoring.
	    score_function (Optional[object]): Univariate scoring function used to rank
	                                       candidate features.
	    percentile (Optional[int]): Percentage of top-ranked features retained by
	                                SelectPercent.
	    transformed_data (Optional[np.ndarray]): Most recent feature matrix produced by
	                                             projection or transformation.
	    accuracy (Optional[float]): Most recent accuracy score produced by selector
	                                evaluation.
	    training_score (Optional[float]): Estimator score calculated on the training split.
	    testing_score (Optional[float]): Estimator score calculated on the testing split."""
	model: Optional[ sf.SelectPercentile ]
	prediction: Optional[ np.ndarray ]
	score_function: Optional[ object ]
	percentile: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, score_func: object = sf.chi2, pct: int = 10 ) -> None:
		"""Initialize SelectPercent.
		
		Purpose:
		    Initializes the percentile-based univariate selector with its scoring function,
		    retained percentile, wrapped sklearn estimator, transformed-data storage, and score
		    fields.
		
		Args:
		    score_func (object): Univariate score function used to rank candidate features.
		    pct (int): Percentage of top-ranked features retained by SelectPercent.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.score_function = score_func
		self.percentile = pct
		self.model = sf.SelectPercentile( score_func=self.score_function,
			percentile=self.percentile )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable list of public members exposed by the SelectPercent class/object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'score_function', 'percentile', 'transformed_data',
			'split_data', 'chi_square', 'train', 'project', 'score', 'transform',
			'train_transform' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split training and testing data.
		
		Purpose:
		    Splits feature and target arrays into aligned training and testing partitions using
		    the configured random seed and test-size proportion.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def chi_square( self, X: np.ndarray, y: np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ] | None:
		"""Calculate chi-square scores.
		
		Purpose:
		    Computes chi-square feature scores and p-values for the supplied feature matrix and
		    target vector using the SelectPercent scoring configuration.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray] | None: Tuple containing chi-square scores and
		                                          p-values.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			return sf.chi2( X, y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'chi_square( self, *args ) -> Tuple[np.ndarray, np.ndarray] | None'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""Fit SelectPercent.
		
		Purpose:
		    Fits the percentile-based univariate selector to feature and target arrays so the
		    highest scoring percentage of features can be retained.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Fitted selector class/object or fitted estimator result.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'train( self, *args ) -> object | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Project features with SelectPercent.
		
		Purpose:
		    Projects the supplied feature matrix with the fitted SelectPercent selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Projected or selected feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame:
		"""Score SelectPercent.
		
		Purpose:
		    Evaluates classifier performance after applying the fitted SelectPercent selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame: Dataframe or dictionary containing training, testing, and accuracy
		                  metrics.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			if self.transformed_data is None:
				self.transformed_data = self.transform( X )
			_support = self.model.get_support( )
			_scores = self.model.scores_
			_pvalues = self.model.pvalues_ if hasattr( self.model, 'pvalues_' ) else None
			df_scores = pd.DataFrame( { 'Feature': np.arange( 0, X.shape[ 1 ] ), 'Score': _scores,
				'PValue': _pvalues if _pvalues is not None else np.full( X.shape[ 1 ], np.nan ),
				'Selected': _support, } )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform features with SelectPercent.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted SelectPercent selector and
		    stores the transformed output.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with SelectPercent.
		
		Purpose:
		    Fits the SelectPercent selector and returns the transformed feature matrix in one
		    operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SelectPercent'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class SBS( Selector ):
	"""SBS feature-selection class.
	
	Purpose:
	    Implements sequential backward selection with a cloned classifier and scoring
	    function. The class iteratively removes features, tracks candidate subsets,
	    records subset scores, and preserves the selected feature indices for projection and
	    evaluation.
	
	Attributes:
	    scoring (Optional[callable]): Callable used to score classifier predictions during
	                                  sequential selection.
	    prediction (Optional[np.ndarray]): Most recent classifier predictions generated
	                                       during scoring.
	    classifier (Optional[Classifier]): Classifier instance cloned or fitted by the
	                                       feature selector.
	    random_state (Optional[int]): Random seed used for reproducible train/test
	                                  splitting.
	    test_size (Optional[float]): Testing-set proportion used during feature-selection
	                                 evaluation.
	    k_features (Optional[int]): Target number of features retained by sequential
	                                backward selection.
	    accuracy (Optional[float]): Most recent accuracy score produced by selector
	                                evaluation.
	    indices_ (Optional[Tuple[int, ...]]): Selected feature-index tuple produced by the
	                                          selector.
	    subsets_ (Optional[List[Tuple[int, ...]]]): Feature-index subsets evaluated during
	                                                sequential selection.
	    scores_ (Optional[List[float]]): Scores recorded for evaluated feature subsets.
	    k_score_ (Optional[float]): Best score associated with the selected feature subset."""
	scoring: Optional[ callable ]
	prediction: Optional[ np.ndarray ]
	classifier: Optional[ Classifier ]
	random_state: Optional[ int ]
	test_size: Optional[ float ]
	k_features: Optional[ int ]
	accuracy: Optional[ float ]
	indices_: Optional[ Tuple[ int, ... ] ]
	subsets_: Optional[ List[ Tuple[ int, ... ] ] ]
	scores_: Optional[ List[ float ] ]
	k_score_: Optional[ float ]
	
	def __init__( self, classifier: Classifier, k_features: int, scoring: callable=accuracy_score,
			test_size: float=0.25, random_state: int=1 ) -> None:
		"""Initialize SBS.
		
		Purpose:
		    Initializes sequential backward selection with a classifier, target feature count,
		    scoring callable, split configuration, and tracking fields for selected subsets and
		    scores.
		
		Args:
		    classifier (Classifier): Classifier used for feature-selection evaluation.
		    k_features (int): Target number of features retained by sequential backward
		                      selection.
		    scoring (callable): Callable used to evaluate candidate feature subsets.
		    test_size (float): Testing-set proportion used during selector evaluation.
		    random_state (int): Random seed used during selector evaluation.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.scoring = scoring
		self.classifier = clone( classifier )
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.indices_ = None
		self.subsets_ = None
		self.scores_ = None
		self.k_score_ = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable list of public members exposed by the SBS class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'scoring', 'classifier', 'k_features', 'test_size', 'random_state', 'prediction',
			'transformed_data', 'indices_', 'subsets_', 'scores_', 'k_score_', 'split_data',
			'train', 'project', 'score', 'transform', 'train_transform', 'calc_score' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
			random: int=42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split training and testing data.
		
		Purpose:
		    Splits feature and target arrays into aligned training and testing partitions using
		    the configured random seed and test-size proportion.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""Fit SBS.
		
		Purpose:
		    Runs sequential backward selection by repeatedly evaluating feature subsets and
		    removing features until the target subset size is reached.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Fitted selector class/object or fitted estimator result.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y,
				test_size=self.test_size,
				random_state=self.random_state )
			dim = X_train.shape[ 1 ]
			self.indices_ = tuple( range( dim ) )
			self.subsets_ = [ self.indices_ ]
			score = self.calc_score( X_train, y_train, X_test, y_test, self.indices_ )
			self.scores_ = [ score ]
			while dim > self.k_features:
				scores = [ ]
				subsets = [ ]
				
				for p in combinations( self.indices_, r=dim - 1 ):
					score = self.calc_score( X_train, y_train, X_test, y_test, p )
					scores.append( score )
					subsets.append( p )
				
				best = int( np.argmax( scores ) )
				self.indices_ = subsets[ best ]
				self.subsets_.append( self.indices_ )
				dim -= 1
				self.scores_.append( scores[ best ] )
			
			self.k_score_ = self.scores_[ -1 ]
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'train( self, *args ) -> object | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Project features with SBS.
		
		Purpose:
		    Projects the supplied feature matrix onto the feature indices selected by sequential
		    backward selection.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Projected or selected feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame:
		"""Score SBS.
		
		Purpose:
		    Evaluates classifier performance after applying the fitted SBS selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame: Dataframe or dictionary containing training, testing, and accuracy
		                  metrics.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			_original_count = int( X.shape[ 1 ] )
			_retained_count = len( self.indices_ ) if self.indices_ is not None else 0
			_metrics = { 'Original Features': [ _original_count ],
				'Retained Features': [ _retained_count ],
				'Removed Features': [ _original_count - _retained_count ],
				'Best Score': [ float( self.k_score_ ) if self.k_score_ is not None else np.nan ],
				'Iterations': [ len( self.subsets_ ) if self.subsets_ is not None else 0 ], }
			return pd.DataFrame( _metrics )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform features with SBS.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted SBS selector and stores the
		    transformed output.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation.
		    ValueError: Raised when the `transform` operation cannot complete."""
		try:
			throw_if( 'X', X )
			if self.indices_ is None:
				raise ValueError( 'The SBS class/object must be trained before calling transform.' )
			self.transformed_data = X[ :, self.indices_ ]
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Fit and transform with SBS.
		
		Purpose:
		    Fits sequential backward selection and returns the feature matrix projected onto the
		    selected subset.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.train( X, y )
			self.transformed_data = self.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def calc_score( self, X_train: np.ndarray, y_train: np.ndarray,
			X_test: np.ndarray, y_test: np.ndarray, indices: Tuple[ int, ... ] ) -> float:
		"""Calculate subset score.
		
		Purpose:
		    Fits a cloned classifier on the selected training columns and returns prediction
		    accuracy on the matching testing columns.
		
		Args:
		    X_train (np.ndarray): Training feature matrix used to fit the classifier.
		    y_train (np.ndarray): Training target vector used to fit the classifier.
		    X_test (np.ndarray): Testing feature matrix used to generate predictions.
		    y_test (np.ndarray): Testing target vector used to evaluate predictions.
		    indices (Tuple[int, ...]): Feature-index subset evaluated by the scoring routine.
		
		Returns:
		    float: Accuracy score calculated for the supplied feature-index subset.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X_train', X_train )
			throw_if( 'y_train', y_train )
			throw_if( 'X_test', X_test )
			throw_if( 'y_test', y_test )
			throw_if( 'indices', indices )
			self.classifier.fit( X_train[ :, indices ], y_train )
			y_pred = self.classifier.predict( X_test[ :, indices ] )
			score = self.scoring( y_test, y_pred )
			return score
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SBS'
			exception.method = 'calc_score( self, *args ) -> float'
			Logger( ).write( exception )
			raise exception

class RFE( Selector ):
	"""RFE feature-selection class/object.
	
	Purpose:
	    Performs recursive feature elimination with an sklearn-compatible estimator. The
	    class recursively prunes low-importance features, stores feature rankings,
	    supports projection into the selected feature set, and evaluates downstream
	    classifier performance.
	
	Attributes:
	    model (Optional[sf.RFE]): Underlying sklearn selector, decomposition, or
	                              cross-decomposition estimator.
	    prediction (Optional[np.ndarray]): Most recent classifier predictions generated
	                                       during scoring.
	    classifier (Optional[NearestNeighbor]): Classifier instance cloned or fitted by the
	                                            feature selector.
	    transformed_data (Optional[np.ndarray]): Most recent feature matrix produced by
	                                             projection or transformation.
	    n_features_to_select (Optional[int]): Number of features retained by recursive
	                                          feature elimination.
	    verbose (Optional[int]): Verbosity level passed to the recursive feature elimination
	                             estimator.
	    accuracy (Optional[float]): Most recent accuracy score produced by selector
	                                evaluation.
	    training_score (Optional[float]): Estimator score calculated on the training split.
	    testing_score (Optional[float]): Estimator score calculated on the testing split."""
	model: Optional[ sf.RFE ]
	prediction: Optional[ np.ndarray ]
	classifier: Optional[ NearestNeighbor ]
	transformed_data: Optional[ np.ndarray ]
	n_features_to_select: Optional[ int ]
	verbose: Optional[ int ]
	accuracy: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self, k_features: int = None, verbose: int = 0 ) -> None:
		"""Initialize RFE.
		
		Purpose:
		    Initializes recursive feature elimination with a nearest-neighbor classifier,
		    retained feature count, verbosity configuration, wrapped sklearn RFE estimator,
		    transformed-data storage, and score fields.
		
		Args:
		    k_features (int): Target number of features retained by sequential backward
		                      selection.
		    verbose (int): Verbosity level passed to recursive feature elimination.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_features_to_select = k_features
		self.classifier = NearestNeighbor( )
		self.verbose = verbose
		self.model = sf.RFE( estimator=self.classifier,
			n_features_to_select=self.n_features_to_select,
			verbose=self.verbose )
		self.prediction = None
		self.transformed_data = None
		self.accuracy = None
		self.training_score = None
		self.testing_score = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable list of public members exposed by the RFE class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'classifier', 'n_features_to_select', 'verbose', 'prediction', 'transformed_data',
			'features_in', 'ranking', 'split_data', 'train', 'project', 'score', 'transform',
			'train_transform' ]
	
	@property
	def features_in( self ) -> int:
		"""Return fitted feature count.
		
		Purpose:
		    Returns the number of input features observed by the fitted recursive feature
		    elimination estimator.
		
		Returns:
		    int: Number of input features observed by the fitted RFE estimator.
		
		Raises:
		    AttributeError: Raised when required fitted estimator metadata is unavailable or
		                    when the base interface is called directly."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def ranking( self ) -> np.ndarray:
		"""Return feature rankings.
		
		Purpose:
		    Returns the feature-ranking array produced by the fitted recursive feature
		    elimination estimator.
		
		Returns:
		    np.ndarray: Feature ranking array produced by recursive feature elimination.
		
		Raises:
		    AttributeError: Raised when required fitted estimator metadata is unavailable or
		                    when the base interface is called directly."""
		if not hasattr( self.model, 'ranking_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.ranking_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float = 0.2,
			random: int = 42 ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split training and testing data.
		
		Purpose:
		    Splits feature and target arrays into aligned training and testing partitions using
		    the configured random seed and test-size proportion.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size,
				random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'split_data( self, *args ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""Fit RFE.
		
		Purpose:
		    Fits recursive feature elimination to feature and target arrays so selected features
		    and rankings can be learned.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    object | None: Fitted selector class/object or fitted estimator result.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'train( self, *args ) -> object | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Project features with RFE.
		
		Purpose:
		    Projects the supplied feature matrix with the fitted RFE selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Projected or selected feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.prediction = self.transform( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'project( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame:
		"""Score RFE.
		
		Purpose:
		    Evaluates classifier performance after applying the fitted RFE selector.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame: Dataframe or dictionary containing training, testing, and accuracy
		                  metrics.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			_support = self.model.get_support( )
			df_scores = pd.DataFrame(
				{ 'Feature': np.arange( 0, X.shape[ 1 ] ), 'Ranking': self.model.ranking_,
					'Selected': _support, } )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'score( self, *args ) -> pd.DataFrame'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform features with RFE.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted RFE selector and stores the
		    transformed output.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (Optional[np.ndarray]): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: np.ndarray ) -> np.ndarray:
		"""Fit and transform with RFE.
		
		Purpose:
		    Fits the RFE selector and returns the transformed feature matrix in one operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the selector, projector, estimator, or
		                    scoring routine.
		    y (np.ndarray): Target vector or target matrix aligned to the rows of `X`.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the selector.
		
		Raises:
		    Error: Raised when validation, fitting, transformation, projection, or scoring fails
		           inside the wrapped selector operation."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RFE'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
			
