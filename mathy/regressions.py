"""******************************************************************************************
  Assembly:                mathy
  Filename:                regressions.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="regressions.py" company="Terry D. Eppler">

         regressions.py

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
    Provides regression model wrappers for Mathy modeling workflows. The module centralizes
    linear, regularized linear, Bayesian, stochastic-gradient, nearest-neighbor, tree,
    ensemble, support-vector, Gaussian-process, and neural-network regressors behind a
    consistent split, train, predict, score, analyze, and diagnostic plotting interface.
</summary>
******************************************************************************************
"""
from __future__ import annotations
from boogr import Error, Logger
from typing import Dict
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble as ske
import sklearn.linear_model as skl
import sklearn.neighbors as skn
import sklearn.svm as skv
import sklearn.tree as skd
from sklearn.base import ClassifierMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split as split
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import Binarizer
import sklearn.neural_network as skm
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             root_mean_squared_error, explained_variance_score,
                             median_absolute_error, max_error, accuracy_score, )

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

class Regression( ):
	"""Regression base class/object.
	
	Purpose:
	    Defines the shared regression class/object contract and common evaluation-state fields
	    used by concrete Mathy regressor implementations. Regression in machine learning is a type
	    of supervised learning used to predict continuous numerical values. Unlike classification
	    models that sort data into distinct categories (like "spam" or "not spam"), regression
	    algorithms find mathematical relationships between input variables (features) and a target
	    outcome to forecast specific numbers, such as house prices, stock values, or temperatures
	
	Attributes:
	    max_iter (Optional[int]): Maximum number of estimator iterations.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    learning_rate (Optional[float]): Estimator learning-rate configuration.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    testing_score (Optional[float]): Most recent estimator score on the testing split."""
	
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	learning_rate: Optional[ float ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	max_error: Optional[ float ]
	training_score: Optional[ float ]
	testing_score: Optional[ float ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the Regression class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		self.max_iter = None
		self.random_state = None
		self.learning_rate = None
		self.prediction = None
		self.mean_absolute_error = None
		self.mean_squared_error = None
		self.root_mean_squared_error = None
		self.r2_score = None
		self.explained_variance_score = None
		self.max_error = None
		self.training_score = None
		self.testing_score = None
	
	def split_data( self, X: np.ndarray,
		y: np.ndarray ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray ] | None:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None: Tuple containing
		                                                                  training features,
		                                                                  testing features,
		                                                                  training targets, and
		                                                                  testing targets.
		
		Raises:
		    NotImplementedError: Raised when the abstract interface method is called directly."""
		raise NotImplementedError
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""Train.
		
		Purpose:
		    Fits the underlying Regression regression estimator to feature and target arrays and
		    returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    object | None: Fitted class/object instance.
		
		Raises:
		    NotImplementedError: Raised when the abstract interface method is called directly."""
		raise NotImplementedError
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted Regression regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    NotImplementedError: Raised when the abstract interface method is called directly."""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> float | pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted Regression regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    float | pd.DataFrame | None: Dataframe containing the primary model score and
		                                 evaluation fields.
		
		Raises:
		    NotImplementedError: Raised when the abstract interface method is called directly."""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted Regression model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    Dict[str, float] | pd.DataFrame | None: Dataframe containing regression metrics
		                                            computed from observed and predicted target
		                                            values.
		
		Raises:
		    NotImplementedError: Raised when the abstract interface method is called directly."""
		raise NotImplementedError

class LeastSquares( Regression ):
	"""LeastSquares regression class/object.
	
	Purpose:
	    Least squares regression is a statistical method used to find the best-fit line or curve
	    for a set of data points by minimizing the sum of the squared differences (residuals)
	    between actual and predicted values
	
	Attributes:
	    model (skl.LinearRegression): Underlying sklearn regression estimator managed by the
	                                  class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    fit_intercept (bool): Flag indicating whether the estimator fits an intercept term.
	    copy_X (bool): Flag indicating whether estimator input data is copied during
	                   fitting.
	    tol (float): Optimization tolerance passed to the estimator.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    positive (bool): Flag constraining coefficients to positive values when supported.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	
	model: skl.LinearRegression
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	fit_intercept: bool
	copy_X: bool
	tol: float
	n_jobs: Optional[ int ]
	positive: bool
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, fit: bool=True, copy: bool=True, tol: float=1e-6,
		jobs: Optional[ int ] = None, positive: bool=False ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the LeastSquares class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    fit (bool): Flag indicating whether the estimator fits an intercept term.
		    copy (bool): Flag indicating whether input feature data is copied during fitting.
		    tol (float): Numerical tolerance used by estimator optimization or convergence
		                 checks.
		    jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
		    positive (bool): Flag constraining fitted coefficients to positive values when
		                     supported.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.fit_intercept = fit
		self.copy_X = copy
		self.tol = tol
		self.n_jobs = jobs
		self.positive = positive
		self.model = skl.LinearRegression( fit_intercept=self.fit_intercept, copy_X=self.copy_X,
			tol=self.tol, n_jobs=self.n_jobs, positive=self.positive )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the LeastSquares class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'fit_intercept', 'copy_X', 'tol', 'n_jobs', 'positive',
			'weights', 'intercept', 'features', 'mean_absolute_error', 'mean_squared_error',
			'root_mean_squared_error', 'r2_score', 'explained_variance_score', 'max_error',
			'training_score', 'testing_score', 'split_data', 'train', 'project', 'score',
			'analyze', 'scatter_plot' ]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.
		
		Purpose:
		    Returns fitted `intercept` metadata from the underlying LeastSquares estimator after
		    training.
		
		Returns:
		    np.ndarray | float | None: Fitted `intercept` metadata from the underlying
		                               estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.
		
		Purpose:
		    Returns fitted `weights` metadata from the underlying LeastSquares estimator after
		    training.
		
		Returns:
		    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying LeastSquares estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The data has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastSquares | None:
		"""Train.
		
		Purpose:
		    Fits the underlying LeastSquares regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    LeastSquares | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'train( self, *args ) -> LeastSquares | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted LeastSquares regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted LeastSquares regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted LeastSquares model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted LeastSquares regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = (f'Training Score = {_training:.1%}\n'
			         f'Testing Score = {_testing:.1%}\n')
			
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastSquares'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class Ridge( Regression ):
	"""Ridge regression class/object.
	
	Purpose:
	    Ridge regression is a modified form of linear regression that uses L2 regularization to
	    shrink large coefficient values. By adding a penalty proportional to the sum of squared
	    coefficients, it reduces model variance and prevents overfitting—especially when features
	    are highly correlated—at the cost of introducing a small amount of bias
	
	Attributes:
	    model (skl.Ridge): Underlying regression estimator managed by the class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    alpha (float): Regularization strength or loss parameter passed to the estimator.
	    fit_intercept (bool): Flag indicating whether the estimator fits an intercept term.
	    copy_X (bool): Flag indicating whether estimator input data is copied during
	                   fitting.
	    max_iter (Optional[int]): Maximum number of estimator iterations.
	    tol (float): Optimization tolerance passed to the estimator.
	    solver (str): Solver selected for estimator optimization.
	    positive (bool): Flag constraining coefficients to positive values when supported.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: skl.Ridge
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	alpha: float
	fit_intercept: bool
	copy_X: bool
	max_iter: Optional[ int ]
	tol: float
	solver: str
	positive: bool
	random_state: Optional[ int ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float=1.0, fit: bool=True, copy: bool=True,
		iters: Optional[ int ] = None, tol: float=1e-4, solver: str = 'auto',
		positive: bool=False, rando: Optional[ int ] = None ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the Ridge class/object with estimator configuration, runtime state, cached
		    prediction fields, and regression metric fields required by training and evaluation
		    workflows.
		
		Args:
		    alpha (float): Regularization strength, loss parameter, or model-specific alpha
		                   value.
		    fit (bool): Flag indicating whether the estimator fits an intercept term.
		    copy (bool): Flag indicating whether input feature data is copied during fitting.
		    iters (Optional[int]): Maximum number of optimization iterations.
		    tol (float): Numerical tolerance used by estimator optimization or convergence
		                 checks.
		    solver (str): Optimization solver used by the estimator.
		    positive (bool): Flag constraining fitted coefficients to positive values when
		                     supported.
		    rando (Optional[int]): Random-state seed passed to the underlying estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.alpha = alpha
		self.fit_intercept = fit
		self.copy_X = copy
		self.max_iter = iters
		self.tol = tol
		self.solver = solver
		self.positive = positive
		self.random_state = rando
		self.model = skl.Ridge( alpha=self.alpha, fit_intercept=self.fit_intercept,
			copy_X=self.copy_X, max_iter=self.max_iter, tol=self.tol, solver=self.solver,
			positive=self.positive, random_state=self.random_state )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the Ridge class/object for interactive
		    inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'alpha', 'fit_intercept', 'copy_X', 'max_iter', 'tol',
			'solver', 'positive', 'random_state', 'weights', 'intercept', 'features',
			'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'train', 'project',
			'score', 'analyze', 'scatter_plot', 'training_score', 'testing_score' ]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.
		
		Purpose:
		    Returns fitted `intercept` metadata from the underlying Ridge estimator after
		    training.
		
		Returns:
		    np.ndarray | float | None: Fitted `intercept` metadata from the underlying
		                               estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.
		
		Purpose:
		    Returns fitted `weights` metadata from the underlying Ridge estimator after
		    training.
		
		Returns:
		    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying Ridge estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2, random: int=42 ) -> \
	tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Ridge | None:
		"""Train.
		
		Purpose:
		    Fits the underlying Ridge regression estimator to feature and target arrays and
		    returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    Ridge | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'train( self, *args ) -> Ridge | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted Ridge regression estimator
		    and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted Ridge regression estimator with its primary scoring behavior
		    and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted Ridge model, including error,
		    explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted Ridge regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			y_pred = self.project( X )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = (f'Training Score = {_training:.1%}\n'
			         f'Testing Score = {_testing:.1%}\n')
			
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Ridge'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class Lasso( Regression ):
	"""Lasso regression class/object.
	
	Purpose:
	    Lasso regression (Least Absolute Shrinkage and Selection Operator) is a linear regression
	    method that uses L₁ regularization. By adding a penalty equal to the absolute value of the
	    magnitude of the coefficients, it shrinks less important feature coefficients down to
	    absolute zero, performing automatic feature selection and preventing model overfitting
	
	Attributes:
	    model (skl.Lasso): Underlying regression estimator managed by the class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    alpha (float): Regularization strength or loss parameter passed to the estimator.
	    fit_intercept (bool): Flag indicating whether the estimator fits an intercept term.
	    precompute (bool): Policy controlling precomputation of the estimator Gram matrix.
	    copy_X (bool): Flag indicating whether estimator input data is copied during
	                   fitting.
	    max_iter (int): Maximum number of estimator iterations.
	    tol (float): Optimization tolerance passed to the estimator.
	    warm_start (bool): Whether repeated fits reuse the preceding fitted estimator state.
	    positive (bool): Flag constraining coefficients to positive values when supported.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    selection (str): Coordinate-selection strategy used by compatible estimators.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: skl.Lasso
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	alpha: float
	fit_intercept: bool
	precompute: bool
	copy_X: bool
	max_iter: int
	tol: float
	warm_start: bool
	positive: bool
	random_state: Optional[ int ]
	selection: str
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float=0.01, fit: bool=True, precompute: bool=False,
		copy: bool=True, iters: int=1000, tol: float=1e-4, warm: bool=False,
		positive: bool=False, rando: Optional[ int ] = None, select: str = 'cyclic',
		selection: Optional[ str ] = None ) -> None:
		"""Initialize the Lasso regression class/object.
		
		Purpose:
		    Configures an L1-regularized linear regression estimator for sparse coefficient
		    estimation, continuous-target prediction, model scoring, residual analysis, and
		    coefficient diagnostics. Supports the existing `select` parameter and the
		    application-facing `selection` parameter for coordinate-descent feature selection.
		
		Args:
		    alpha (float): Regularization strength applied to the L1 penalty.
		    fit (bool): Flag indicating whether the estimator fits an intercept.
		    precompute (bool): Flag controlling Gram-matrix precomputation.
		    copy (bool): Flag indicating whether the input feature matrix is copied.
		    iters (int): Maximum number of coordinate-descent iterations.
		    tol (float): Optimization tolerance used to determine convergence.
		    warm (bool): Flag indicating whether the previous fitted solution is reused.
		    positive (bool): Flag constraining fitted coefficients to nonnegative values.
		    rando (Optional[int]): Random seed used when random coordinate selection is enabled.
		    select (str): Coordinate-selection strategy used when `selection` is not supplied.
		    selection (Optional[str]): Coordinate-selection strategy supplied by the
		                               application.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.alpha = alpha
		self.fit_intercept = fit
		self.precompute = precompute
		self.copy_X = copy
		self.max_iter = iters
		self.tol = tol
		self.warm_start = warm
		self.positive = positive
		self.random_state = rando
		self.selection = selection if selection is not None else select
		self.model = skl.Lasso( alpha=self.alpha, fit_intercept=self.fit_intercept,
			precompute=self.precompute, copy_X=self.copy_X, max_iter=self.max_iter, tol=self.tol,
			warm_start=self.warm_start, positive=self.positive, random_state=self.random_state,
			selection=self.selection )
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the Lasso class/object for interactive
		    inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'alpha', 'fit_intercept', 'precompute', 'copy_X',
			'max_iter', 'tol', 'warm_start', 'positive', 'random_state', 'selection', 'weights',
			'intercept', 'features', 'mean_absolute_error', 'mean_squared_error',
			'root_mean_squared_error', 'r2_score', 'explained_variance_score',
			'median_absolute_error', 'max_error', 'train', 'project', 'score', 'analyze',
			'scatter_plot', 'training_score', 'testing_score' ]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.
		
		Purpose:
		    Returns fitted `intercept` metadata from the underlying Lasso estimator after
		    training.
		
		Returns:
		    np.ndarray | float | None: Fitted `intercept` metadata from the underlying
		                               estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.
		
		Purpose:
		    Returns fitted `weights` metadata from the underlying Lasso estimator after
		    training.
		
		Returns:
		    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying Lasso estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> Lasso | None:
		"""Train.
		
		Purpose:
		    Fits the underlying Lasso regression estimator to feature and target arrays and
		    returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    Lasso | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'train( self, *args ) -> Lasso | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted Lasso regression estimator
		    and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted Lasso regression estimator with its primary scoring behavior
		    and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted Lasso model, including error,
		    explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted Lasso regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Lasso'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class ElasticNet( Regression ):
	"""ElasticNet regression class/object.
	
	Purpose:
	    Elastic net regression is a popular regularized linear regression method that linearly
	    combines both L1 (Lasso) and L2 (Ridge) penalty terms. By merging these two approaches,
	    it overcomes individual limitations, performs automated feature selection, and effectively
	    handles groups of correlated variables
	
	Attributes:
	    model (skl.ElasticNet): Underlying regression estimator managed by the
	                            class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    alpha (float): Regularization strength or loss parameter passed to the estimator.
	    l1_ratio (float): Elastic-net mixing parameter.
	    fit_intercept (bool): Flag indicating whether the estimator fits an intercept term.
	    precompute (bool): Policy controlling precomputation of the estimator Gram matrix.
	    max_iter (int): Maximum number of estimator iterations.
	    copy_X (bool): Flag indicating whether estimator input data is copied during
	                   fitting.
	    tol (float): Optimization tolerance passed to the estimator.
	    warm_start (bool): Whether repeated fits reuse the preceding fitted estimator state.
	    positive (bool): Flag constraining coefficients to positive values when supported.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    selection (str): Coordinate-selection strategy used by compatible estimators.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: skl.ElasticNet
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	alpha: float
	l1_ratio: float
	fit_intercept: bool
	precompute: bool
	max_iter: int
	copy_X: bool
	tol: float
	warm_start: bool
	positive: bool
	random_state: Optional[ int ]
	selection: str
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, alpha: float=1.0, ratio: float=0.5, fit: bool=True,
		precompute: bool=False, iters: int=1000, copy: bool=True, tol: float=1e-4,
		warm: bool=False, positive: bool=False, rando: Optional[ int ] = None,
		select: str = 'cyclic' ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the ElasticNet class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    alpha (float): Regularization strength, loss parameter, or model-specific alpha
		                   value.
		    ratio (float): Elastic-net L1/L2 mixing ratio.
		    fit (bool): Flag indicating whether the estimator fits an intercept term.
		    precompute (bool): Flag or setting controlling Gram-matrix precomputation.
		    iters (int): Maximum number of optimization iterations.
		    copy (bool): Flag indicating whether input feature data is copied during fitting.
		    tol (float): Numerical tolerance used by estimator optimization or convergence
		                 checks.
		    warm (bool): Flag indicating whether previous estimator state is reused across fits.
		    positive (bool): Flag constraining fitted coefficients to positive values when
		                     supported.
		    rando (Optional[int]): Random-state seed passed to the underlying estimator.
		    select (str): Coordinate-selection strategy used during optimization.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.alpha = alpha
		self.l1_ratio = ratio
		self.fit_intercept = fit
		self.precompute = precompute
		self.max_iter = iters
		self.copy_X = copy
		self.tol = tol
		self.warm_start = warm
		self.positive = positive
		self.random_state = rando
		self.selection = select
		self.model = skl.ElasticNet( alpha=self.alpha, l1_ratio=self.l1_ratio,
			fit_intercept=self.fit_intercept, precompute=self.precompute, max_iter=self.max_iter,
			copy_X=self.copy_X, tol=self.tol, warm_start=self.warm_start, positive=self.positive,
			random_state=self.random_state, selection=self.selection )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the ElasticNet class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'alpha', 'l1_ratio', 'fit_intercept', 'precompute',
			'max_iter', 'copy_X', 'tol', 'warm_start', 'positive', 'random_state', 'selection',
			'weights', 'intercept', 'features', 'mean_absolute_error', 'mean_squared_error',
			'root_mean_squared_error', 'r2_score', 'explained_variance_score',
			'median_absolute_error', 'max_error', 'train', 'project', 'score', 'analyze',
			'scatter_plot', 'training_score', 'testing_score' ]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.
		
		Purpose:
		    Returns fitted `intercept` metadata from the underlying ElasticNet estimator after
		    training.
		
		Returns:
		    np.ndarray | float | None: Fitted `intercept` metadata from the underlying
		                               estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.
		
		Purpose:
		    Returns fitted `weights` metadata from the underlying ElasticNet estimator after
		    training.
		
		Returns:
		    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying ElasticNet estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> ElasticNet | None:
		"""Train.
		
		Purpose:
		    Fits the underlying ElasticNet regression estimator to feature and target arrays and
		    returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    ElasticNet | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'train( self, *args ) -> ElasticNet | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted ElasticNet regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted ElasticNet regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted ElasticNet model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted ElasticNet regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ElasticNet'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class LeastAngle( Regression ):
	"""LeastAngle regression class/object.
	
	Purpose:
	    Least Angle Regression (LARS) is a fast and efficient variable-selection algorithm for
	    high-dimensional linear regression. It works like forward selection, but instead of fully
	    fitting variables one at a time, LARS increases coefficients equiangularly in directions
	    equiangular to their correlations with the residual, closely connecting to Lasso.
	
	Attributes:
	    model (skl.Lars): Underlying regression estimator managed by the class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    n_nonzero_coefs (int): Target number of nonzero coefficients in the fitted model.
	    fit_intercept (bool): Flag indicating whether the estimator fits an intercept term.
	    precompute (bool): Policy controlling precomputation of the estimator Gram matrix.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split."""
	model: skl.Lars
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	n_nonzero_coefs: int
	fit_intercept: bool
	precompute: bool
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, coeffs: int=500, fit: bool=True, precompute: bool=True ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the LeastAngle class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    coeffs (int): Maximum number of coefficients or active variables used by the
		                  estimator.
		    fit (bool): Flag indicating whether the estimator fits an intercept term.
		    precompute (bool): Flag or setting controlling Gram-matrix precomputation.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.fit_intercept = fit
		self.n_nonzero_coefs = coeffs
		self.precompute = precompute
		self.model = skl.Lars( fit_intercept=self.fit_intercept, precompute=self.precompute,
			n_nonzero_coefs=self.n_nonzero_coefs )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the LeastAngle class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'fit_intercept', 'precompute', 'n_nonzero_coefs',
			'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'train', 'project',
			'score', 'analyze', 'scatter_plot', 'training_score', 'testing_score' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> LeastAngle | None:
		"""Train.
		
		Purpose:
		    Fits the underlying LeastAngle regression estimator to feature and target arrays and
		    returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    LeastAngle | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'train( self, *args ) -> LeastAngle | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted LeastAngle regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted LeastAngle regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			_metrics = { 'Training Score': self.training_score, 'Testing Score':
				self.testing_score,
				'R-Squared Score': self.r2_score, }
			
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted LeastAngle model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			_metrics = { 'MAE': self.mean_absolute_error, 'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error, 'EVS': self.explained_variance_score,
				'MAX': self.max_error, }
			
			_data = pd.Series( _metrics )
			df_metrics = pd.DataFrame( _data )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted LeastAngle regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LeastAngle'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class BayesianRidge( Regression ):
	"""BayesianRidge regression class/object.
	
	Purpose:
	    Bayesian ridge regression is a probabilistic linear model that treats weights as random
	    variables and uses L2 regularization to estimate parameter distributions and noise. Key
	    aspects include Gaussian priors, automatic hyperparameter tuning, and uncertainty estimation.
	
	Attributes:
	    model (skl.BayesianRidge): Underlying regression estimator managed by the
	                               class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    scale_alpha (float): Whether the initial weight precision is scaled during fitting.
	    shape_lambda (float): Gamma-prior shape parameter for target-noise precision.
	    shape_alpha (float): Gamma-prior shape parameter for model-weight precision.
	    max_iter (int): Maximum number of estimator iterations.
	    scale_lambda (float): Whether the initial noise precision is scaled during fitting.
	    tol (float): Optimization tolerance passed to the estimator.
	    alpha_init (Optional[float]): Initial precision estimate for model weights.
	    lambda_init (Optional[float]): Initial precision estimate for the target noise.
	    compute_score (bool): Whether the estimator computes out-of-bag or training scores.
	    fit_intercept (bool): Flag indicating whether the estimator fits an intercept term.
	    copy_X (bool): Flag indicating whether estimator input data is copied during
	                   fitting.
	    verbose (bool): Estimator verbosity level.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: skl.BayesianRidge
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	scale_alpha: float
	shape_lambda: float
	shape_alpha: float
	max_iter: int
	scale_lambda: float
	tol: float
	alpha_init: Optional[ float ]
	lambda_init: Optional[ float ]
	compute_score: bool
	fit_intercept: bool
	copy_X: bool
	verbose: bool
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, max: int=300, shape_alpha: float=1e-06, scale_alpha: float=1e-06,
		shape_lambda: float=1e-06, scale_lambda: float=1e-06, tol: float=1e-3,
		alpha_init: Optional[ float ]=None, lambda_init: Optional[ float ]=None,
		compute_score: bool=False, fit: bool=True, copy: bool=True,
		verbose: bool=False ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the BayesianRidge class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    max (int): Configuration value passed to the underlying regression estimator.
		    shape_alpha (float): Shape parameter for the Bayesian alpha prior.
		    scale_alpha (float): Scale parameter for the Bayesian alpha prior.
		    shape_lambda (float): Shape parameter for the Bayesian lambda prior.
		    scale_lambda (float): Scale parameter for the Bayesian lambda prior.
		    tol (float): Numerical tolerance used by estimator optimization or convergence
		                 checks.
		    alpha_init (Optional[float]): Initial alpha value for Bayesian regression.
		    lambda_init (Optional[float]): Initial lambda value for Bayesian regression.
		    compute_score (bool): Flag indicating whether Bayesian scores are computed during
		                          fitting.
		    fit (bool): Flag indicating whether the estimator fits an intercept term.
		    copy (bool): Flag indicating whether input feature data is copied during fitting.
		    verbose (bool): Verbosity flag or level passed to the estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.max_iter = max
		self.shape_alpha = shape_alpha
		self.scale_alpha = scale_alpha
		self.shape_lambda = shape_lambda
		self.scale_lambda = scale_lambda
		self.tol = tol
		self.alpha_init = alpha_init
		self.lambda_init = lambda_init
		self.compute_score = compute_score
		self.fit_intercept = fit
		self.copy_X = copy
		self.verbose = verbose
		self.model = skl.BayesianRidge( max_iter=self.max_iter, tol=self.tol,
			alpha_1=self.shape_alpha, alpha_2=self.scale_alpha, lambda_1=self.shape_lambda,
			lambda_2=self.scale_lambda, alpha_init=self.alpha_init, lambda_init=self.lambda_init,
			compute_score=self.compute_score, fit_intercept=self.fit_intercept, copy_X=self.copy_X,
			verbose=self.verbose )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the BayesianRidge class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'shape_alpha', 'scale_alpha', 'shape_lambda',
			'scale_lambda', 'max_iter', 'tol', 'alpha_init', 'lambda_init', 'compute_score',
			'fit_intercept', 'copy_X', 'verbose', 'weights', 'intercept', 'features',
			'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'train', 'project',
			'score', 'analyze', 'scatter_plot', 'training_score', 'testing_score' ]
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.
		
		Purpose:
		    Returns fitted `intercept` metadata from the underlying BayesianRidge estimator
		    after training.
		
		Returns:
		    np.ndarray | float | None: Fitted `intercept` metadata from the underlying
		                               estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.
		
		Purpose:
		    Returns fitted `weights` metadata from the underlying BayesianRidge estimator after
		    training.
		
		Returns:
		    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying BayesianRidge estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BayesianRidge | None:
		"""Train.
		
		Purpose:
		    Fits the underlying BayesianRidge regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    BayesianRidge | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'train( self, *args ) -> BayesianRidge | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted BayesianRidge regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted BayesianRidge regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted BayesianRidge model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted BayesianRidge regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BayesianRidge'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class GradientDescent( Regression ):
	"""GradientDescent regression class/object.
	
	Purpose:
	    Gradient descent regression is an optimization technique that iteratively updates a model's
	    parameters (weights and bias) to minimize prediction error, typically measured by Mean Squared Error (MSE).
	    Instead of calculating parameters in a single step like Ordinary Least Squares, it uses the
	    gradient (slope) of the loss function to step downhill toward the global minimum
	
	Attributes:
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    loss (Optional[str]): Loss function minimized by the estimator.
	    max_iter (Optional[int]): Maximum number of estimator iterations.
	    penalty (Optional[str]): Regularization penalty applied during estimator fitting.
	    learning_rate (Optional[str]): Estimator learning-rate configuration.
	    l1_ratio (Optional[float]): Elastic-net mixing parameter.
	    alpha (Optional[float]): Regularization strength or loss parameter passed to the
	                             estimator.
	    fit_intercept (bool): Flag indicating whether the estimator fits an intercept term.
	    tol (float): Optimization tolerance passed to the estimator.
	    shuffle (bool): Whether training samples are shuffled between fitting iterations.
	    verbose (int): Estimator verbosity level.
	    epsilon (float): Estimator-specific tolerance or insensitive-loss margin.
	    eta0 (float): Initial learning rate used by stochastic gradient descent.
	    power_t (float): Exponent used by the inverse-scaling learning-rate schedule.
	    early_stopping (bool): Policy controlling validation-based early stopping.
	    validation_fraction (float): Fraction of training data reserved for early stopping.
	    n_iter_no_change (int): Stable-iteration count required to trigger early stopping.
	    warm_start (bool): Whether repeated fits reuse the preceding fitted estimator state.
	    average (bool): Averaging strategy or calculated arithmetic-mean values.
	    model (skl.SGDRegressor): Underlying regression estimator managed by the
	                              class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	random_state: Optional[ int ]
	loss: Optional[ str ]
	max_iter: Optional[ int ]
	penalty: Optional[ str ]
	learning_rate: Optional[ str ]
	l1_ratio: Optional[ float ]
	alpha: Optional[ float ]
	fit_intercept: bool
	tol: float
	shuffle: bool
	verbose: int
	epsilon: float
	eta0: float
	power_t: float
	early_stopping: bool
	validation_fraction: float
	n_iter_no_change: int
	warm_start: bool
	average: bool
	model: skl.SGDRegressor
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, loss: str='squared_error', iters: int=1000, penalty: str='l2',
		alpha: float=0.0001, rando: Optional[ int ]=42, learning_rate: str='invscaling',
		l1_ratio: float=0.15, fit: bool=True, tol: float=1e-3, shuffle: bool=True,
		verbose: int=0, epsilon: float=0.1, eta0: float=0.01, power_t: float=0.25,
		early_stopping: bool=False, validation_fraction: float=0.1, n_iter_no_change: int=5,
		warm: bool=False, average: bool=False ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the GradientDescent class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    loss (str): Loss function used by the estimator.
		    iters (int): Maximum number of optimization iterations.
		    penalty (str): Regularization parameter used by the estimator.
		    alpha (float): Regularization strength, loss parameter, or model-specific alpha
		                   value.
		    rando (Optional[int]): Random-state seed passed to the underlying estimator.
		    learning_rate (str): Learning-rate schedule used by the estimator.
		    l1_ratio (float): Elastic-net mixing ratio for L1 and L2 penalties.
		    fit (bool): Flag indicating whether the estimator fits an intercept term.
		    tol (float): Numerical tolerance used by estimator optimization or convergence
		                 checks.
		    shuffle (bool): Flag indicating whether training samples are shuffled between
		                    epochs.
		    verbose (int): Verbosity flag or level passed to the estimator.
		    epsilon (float): Epsilon-insensitive loss parameter or robustness threshold.
		    eta0 (float): Initial learning-rate value.
		    power_t (float): Exponent used by inverse-scaling learning-rate schedules.
		    early_stopping (bool): Flag enabling validation-based early stopping.
		    validation_fraction (float): Training-data fraction reserved for early-stopping
		                                 validation.
		    n_iter_no_change (int): Number of iterations with no improvement before early
		                            stopping.
		    warm (bool): Flag indicating whether previous estimator state is reused across fits.
		    average (bool): Flag enabling averaged stochastic-gradient coefficients.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.random_state = rando
		self.loss = loss
		self.max_iter = iters
		self.penalty = penalty
		self.learning_rate = learning_rate
		self.l1_ratio = l1_ratio
		self.alpha = alpha
		self.fit_intercept = fit
		self.tol = tol
		self.shuffle = shuffle
		self.verbose = verbose
		self.epsilon = epsilon
		self.eta0 = eta0
		self.power_t = power_t
		self.early_stopping = early_stopping
		self.validation_fraction = validation_fraction
		self.n_iter_no_change = n_iter_no_change
		self.warm_start = warm
		self.average = average
		self.model = skl.SGDRegressor( loss=self.loss, penalty=self.penalty, alpha=self.alpha,
			l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept, max_iter=self.max_iter,
			tol=self.tol, shuffle=self.shuffle, verbose=self.verbose, epsilon=self.epsilon,
			random_state=self.random_state, learning_rate=self.learning_rate, eta0=self.eta0,
			power_t=self.power_t, early_stopping=self.early_stopping,
			validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
			warm_start=self.warm_start, average=self.average )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the GradientDescent class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'penalty', 'max_iter', 'random_state', 'loss',
			'learning_rate', 'l1_ratio', 'alpha', 'fit_intercept', 'tol', 'shuffle', 'verbose',
			'epsilon', 'eta0', 'power_t', 'early_stopping', 'validation_fraction',
			'n_iter_no_change', 'warm_start', 'average', 'mean_absolute_error',
			'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score',
			'median_absolute_error', 'max_error', 'train', 'project', 'score', 'analyze',
			'scatter_plot', 'weights', 'intercept', 'features', 'iterations', 'training_score',
			'testing_score' ]
	
	@property
	def weights( self ) -> np.ndarray | None:
		"""Weights.
		
		Purpose:
		    Returns fitted `weights` metadata from the underlying GradientDescent estimator
		    after training.
		
		Returns:
		    np.ndarray | None: Fitted `weights` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'coef_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.coef_
	
	@property
	def intercept( self ) -> np.ndarray | float | None:
		"""Intercept.
		
		Purpose:
		    Returns fitted `intercept` metadata from the underlying GradientDescent estimator
		    after training.
		
		Returns:
		    np.ndarray | float | None: Fitted `intercept` metadata from the underlying
		                               estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'intercept_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.intercept_
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying GradientDescent estimator
		    after training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def iterations( self ) -> int | None:
		"""Iterations.
		
		Purpose:
		    Returns fitted `iterations` metadata from the underlying GradientDescent estimator
		    after training.
		
		Returns:
		    int | None: Fitted `iterations` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 't_' ):
			raise AttributeError( 'The model has not been initialized!' )
		return self.model.t_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientDescent | None:
		"""Train.
		
		Purpose:
		    Fits the underlying GradientDescent regression estimator to feature and target
		    arrays and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    GradientDescent | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'train( self, *args ) -> GradientDescent | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted GradientDescent regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted GradientDescent regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted GradientDescent model,
		    including error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted GradientDescent regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientDescent'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class NearestNeighbor( Regression ):
	"""NearestNeighbor regression class/object.
	
	Purpose:
	    Nearest neighbor regression (KNN regression) is a simple machine learning method that
	    predicts a number for a new data point. It finds the closest known points in the data and
	    calculates their average value
	
	Attributes:
	    model (skn.KNeighborsRegressor): Underlying regression estimator managed by
	                                     the class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    n_neighbors (int): Number of neighboring samples used by the estimator.
	    weights (str): Optional estimator or neighbor weights.
	    algorithm (str): Algorithm selected for estimator fitting or neighbor search.
	    leaf_size (int): Leaf size used by tree-based neighbor-search structures.
	    power (float): Power parameter used by the Minkowski distance metric.
	    metric (str): Distance or scoring metric used by the estimator.
	    metric_params (Optional[Dict[str, object]]): Additional keyword values supplied to
	                                                 the selected metric.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: skn.KNeighborsRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	n_neighbors: int
	weights: str
	algorithm: str
	leaf_size: int
	power: float
	metric: str
	metric_params: Optional[ Dict[ str, object ] ]
	n_jobs: Optional[ int ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, num: int=5, weight: str='uniform', algo: str='auto', leaf: int=30,
		power: float=2.0, metric: str='minkowski',
		metric_params: Optional[ Dict[ str, object ] ]=None,
		jobs: Optional[ int ]=None ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the NearestNeighbor class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    num (int): Number of neighbors, estimators, or model-specific components.
		    weight (str): Neighbor weighting strategy.
		    algo (str): Neighbor-search algorithm.
		    leaf (int): Leaf-size or minimum leaf configuration.
		    power (float): Power parameter for Minkowski-style distance metrics.
		    metric (str): Distance metric used by neighbor estimators.
		    metric_params (Optional[Dict[str, object]]): Additional metric-specific keyword
		                                                 parameters.
		    jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_neighbors = num
		self.weights = weight
		self.algorithm = algo
		self.leaf_size = leaf
		self.power = power
		self.metric = metric
		self.metric_params = metric_params
		self.n_jobs = jobs
		self.model = skn.KNeighborsRegressor( n_neighbors=self.n_neighbors, weights=self.weights,
			algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.power, metric=self.metric,
			metric_params=self.metric_params, n_jobs=self.n_jobs )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the NearestNeighbor class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'weights', 'algorithm', 'n_neighbors', 'leaf_size',
			'power', 'metric', 'metric_params', 'n_jobs', 'mean_absolute_error', 'mean_squared_error',
			'root_mean_squared_error', 'r2_score', 'explained_variance_score',
			'median_absolute_error', 'max_error', 'train', 'project', 'score', 'analyze',
			'scatter_plot', 'training_score', 'testing_score', 'features' ]
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying NearestNeighbor estimator
		    after training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> NearestNeighbor | None:
		"""Train.
		
		Purpose:
		    Fits the underlying NearestNeighbor regression estimator to feature and target
		    arrays and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    NearestNeighbor | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'train( self, *args ) -> NearestNeighbor | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted NearestNeighbor regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted NearestNeighbor regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted NearestNeighbor model,
		    including error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted NearestNeighbor regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestNeighbor'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class DecisionTree( Regression ):
	"""DecisionTree regression class/object.
	
	Purpose:
	    Decision tree regression is a supervised machine learning algorithm that predicts continuous
	    numerical outcomes by recursively partitioning data into smaller, more homogeneous subsets.
	    Unlike classification trees that output labels, a regression tree divides the feature space
	    into distinct regions and outputs a single numeric value—typically the average (mean) of
	    the target values belonging to that specific region.
	
	Attributes:
	    model (skd.DecisionTreeRegressor): Underlying regression estimator managed
	                                       by the class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    criterion (str): Split or impurity criterion used by tree-based estimators.
	    splitter (str): Tree splitter strategy.
	    max_depth (int): Maximum tree depth.
	    random_state (int): Random seed or random-state configuration used by the estimator.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split."""
	model: skd.DecisionTreeRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	criterion: str
	splitter: str
	max_depth: int
	random_state: int
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, criterion: str = 'squared_error', splitter: str = 'best', depth: int=3,
		rando: int=42 ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the DecisionTree class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    criterion (str): Tree split or ensemble loss criterion.
		    splitter (str): Tree split strategy.
		    depth (int): Maximum tree depth.
		    rando (int): Random-state seed passed to the underlying estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.criterion = criterion
		self.splitter = splitter
		self.max_depth = depth
		self.random_state = rando
		self.model = skd.DecisionTreeRegressor( criterion=self.criterion, splitter=self.splitter,
			max_depth=self.max_depth, random_state=self.random_state )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the DecisionTree class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'criterion', 'splitter', 'random_state', 'max_depth',
			'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'train', 'project',
			'score', 'analyze', 'scatter_plot', 'training_score', 'testing_score' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> DecisionTree | None:
		"""Train.
		
		Purpose:
		    Fits the underlying DecisionTree regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    DecisionTree | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'train( self, *args ) -> DecisionTree | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted DecisionTree regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted DecisionTree regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			_metrics = { 'Training Score': self.training_score, 'Testing Score':
				self.testing_score,
				'R-Squared Score': self.r2_score, }
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted DecisionTree model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			_metrics = { 'MAE': self.mean_absolute_error, 'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error, 'EVS': self.explained_variance_score,
				'MAX': self.max_error, }
			_data = pd.Series( _metrics )
			df_metrics = pd.DataFrame( _data )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted DecisionTree regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DecisionTree'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class ExtraTreesModel( Regression ):
	"""ExtraTreesModel regression class/object.
	
	Purpose:
	    Extra Tree Regression (Extremely Randomized Trees Regressor) is an ensemble machine learning
	    algorithm that builds multiple unpruned decision trees. It differs from Random Forest by
	    choosing split points completely at random for each feature rather than searching for the
	    optimal split, which speeds up training and reduces variance
	
	Attributes:
	    model (ske.ExtraTreesRegressor): Underlying regression estimator managed by
	                                     the class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    n_estimators (int): Number of ensemble estimators.
	    criterion (str): Split or impurity criterion used by tree-based estimators.
	    max_depth (Optional[int]): Maximum tree depth.
	    min_samples_split (int | float): Minimum number of samples required to split a tree
	                                     node.
	    min_samples_leaf (int | float): Minimum number of samples permitted in a tree leaf.
	    min_weight_fraction_leaf (float): Minimum weighted sample fraction permitted in a
	                                      tree leaf.
	    max_features (int | float | str | None): Feature subset size considered during model
	                                             fitting.
	    max_leaf_nodes (Optional[int]): Maximum number of leaf nodes permitted in each tree.
	    min_impurity_decrease (float): Minimum impurity reduction required to split a tree
	                                   node.
	    bootstrap (bool): Whether ensemble estimators are fitted from bootstrap samples.
	    oob_score (bool): Whether out-of-bag samples are used to estimate generalization
	                      performance.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    verbose (int): Estimator verbosity level.
	    warm_start (bool): Whether repeated fits reuse the preceding fitted estimator state.
	    ccp_alpha (float): Complexity parameter used for minimal cost-complexity pruning.
	    max_samples (Optional[int | float]): Maximum sample count or fraction drawn for each
	                                         ensemble estimator.
	    monotonic_cst (Optional[object]): Monotonic feature constraints applied during model
	                                      fitting.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: ske.ExtraTreesRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	n_estimators: int
	criterion: str
	max_depth: Optional[ int ]
	min_samples_split: int | float
	min_samples_leaf: int | float
	min_weight_fraction_leaf: float
	max_features: int | float | str | None
	max_leaf_nodes: Optional[ int ]
	min_impurity_decrease: float
	bootstrap: bool
	oob_score: bool
	n_jobs: Optional[ int ]
	random_state: Optional[ int ]
	verbose: int
	warm_start: bool
	ccp_alpha: float
	max_samples: Optional[ int | float ]
	monotonic_cst: Optional[ object ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, estimators: int=100, criterion: str='squared_error',
		depth: Optional[ int ]=None, split: int | float=2, leaf: int | float=1,
		weight_fraction: float=0.0, features: int | float | str | None=1.0,
		leaf_nodes: Optional[ int ]=None, impurity: float=0.0, bootstrap: bool=False,
		oob_score: bool=False, jobs: Optional[ int ]=None, rando: Optional[ int ]=42,
		verbose: int=0, warm: bool=False, ccp_alpha: float=0.0,
		samples: Optional[ int | float ]=None, monotonic: Optional[ object ]=None ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the ExtraTreesModel class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    estimators (int): Number of estimators in an ensemble.
		    criterion (str): Tree split or ensemble loss criterion.
		    depth (Optional[int]): Maximum tree depth.
		    split (int | float): Minimum samples required to split an internal tree node.
		    leaf (int | float): Leaf-size or minimum leaf configuration.
		    weight_fraction (float): Minimum weighted fraction required at a leaf node.
		    features (int | float | str | None): Maximum feature-selection setting used by the
		                                         estimator.
		    leaf_nodes (Optional[int]): Maximum number of leaf nodes.
		    impurity (float): Minimum impurity decrease required for a split.
		    bootstrap (bool): Flag enabling bootstrap sampling.
		    oob_score (bool): Flag enabling out-of-bag scoring.
		    jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
		    rando (Optional[int]): Random-state seed passed to the underlying estimator.
		    verbose (int): Verbosity flag or level passed to the estimator.
		    warm (bool): Flag indicating whether previous estimator state is reused across fits.
		    ccp_alpha (float): Complexity parameter used for minimal cost-complexity pruning.
		    samples (Optional[int | float]): Maximum sample setting used by ensemble estimators.
		    monotonic (Optional[object]): Monotonic constraint configuration for compatible
		                                  estimators.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_estimators = estimators
		self.criterion = criterion
		self.max_depth = depth
		self.min_samples_split = split
		self.min_samples_leaf = leaf
		self.min_weight_fraction_leaf = weight_fraction
		self.max_features = features
		self.max_leaf_nodes = leaf_nodes
		self.min_impurity_decrease = impurity
		self.bootstrap = bootstrap
		self.oob_score = oob_score
		self.n_jobs = jobs
		self.random_state = rando
		self.verbose = verbose
		self.warm_start = warm
		self.ccp_alpha = ccp_alpha
		self.max_samples = samples
		self.monotonic_cst = monotonic
		self.model = ske.ExtraTreesRegressor( n_estimators=self.n_estimators,
			criterion=self.criterion, max_depth=self.max_depth,
			min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
			max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
			bootstrap=self.bootstrap, oob_score=self.oob_score, n_jobs=self.n_jobs,
			random_state=self.random_state, verbose=self.verbose, warm_start=self.warm_start,
			ccp_alpha=self.ccp_alpha, max_samples=self.max_samples,
			monotonic_cst=self.monotonic_cst )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the ExtraTreesModel class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'n_estimators', 'criterion', 'max_depth',
			'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
			'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs',
			'random_state', 'verbose', 'warm_start', 'ccp_alpha', 'max_samples', 'monotonic_cst',
			'features', 'training_score', 'testing_score', 'mean_absolute_error',
			'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'split_data',
			'train', 'project', 'score', 'analyze', 'scatter_plot' ]
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying ExtraTreesModel estimator
		    after training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> ExtraTreesModel | None:
		"""Train.
		
		Purpose:
		    Fits the underlying ExtraTreesModel regression estimator to feature and target
		    arrays and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    ExtraTreesModel | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'train( self, *args ) -> ExtraTreesModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted ExtraTreesModel regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted ExtraTreesModel regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted ExtraTreesModel model,
		    including error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted ExtraTreesModel regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ExtraTreesModel'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class RandomForest( Regression ):
	"""RandomForest regression class/object.
	
	Purpose:
	    Random Forest Regression is a powerful ensemble learning technique that builds a multitude
	    of independent decision trees during training and averages their outputs to predict a
	    continuous numerical value. By combining predictions from many individual trees, it
	    drastically reduces the risk of overfitting, making it much more robust and accurate
	    than a single decision tree.
	
	Attributes:
	    model (ske.RandomForestRegressor): Underlying regression estimator managed
	                                       by the class/object.
	    n_estimators (int): Number of ensemble estimators.
	    criterion (str): Split or impurity criterion used by tree-based estimators.
	    max_depth (Optional[int]): Maximum tree depth.
	    min_samples_split (int | float): Minimum number of samples required to split a tree
	                                     node.
	    min_samples_leaf (int | float): Minimum number of samples permitted in a tree leaf.
	    min_weight_fraction_leaf (float): Minimum weighted sample fraction permitted in a
	                                      tree leaf.
	    max_features (int | float | str | None): Feature subset size considered during model
	                                             fitting.
	    max_leaf_nodes (Optional[int]): Maximum number of leaf nodes permitted in each tree.
	    min_impurity_decrease (float): Minimum impurity reduction required to split a tree
	                                   node.
	    bootstrap (bool): Whether ensemble estimators are fitted from bootstrap samples.
	    oob_score (bool): Whether out-of-bag samples are used to estimate generalization
	                      performance.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    verbose (int): Estimator verbosity level.
	    warm_start (bool): Whether repeated fits reuse the preceding fitted estimator state.
	    ccp_alpha (float): Complexity parameter used for minimal cost-complexity pruning.
	    max_samples (Optional[int | float]): Maximum sample count or fraction drawn for each
	                                         ensemble estimator.
	    monotonic_cst (Optional[object]): Monotonic feature constraints applied during model
	                                      fitting.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: ske.RandomForestRegressor
	n_estimators: int
	criterion: str
	max_depth: Optional[ int ]
	min_samples_split: int | float
	min_samples_leaf: int | float
	min_weight_fraction_leaf: float
	max_features: int | float | str | None
	max_leaf_nodes: Optional[ int ]
	min_impurity_decrease: float
	bootstrap: bool
	oob_score: bool
	n_jobs: Optional[ int ]
	random_state: Optional[ int ]
	verbose: int
	warm_start: bool
	ccp_alpha: float
	max_samples: Optional[ int | float ]
	monotonic_cst: Optional[ object ]
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, estimators: int=100, criterion: str = 'squared_error',
		depth: Optional[ int ] = None, split: int | float = 2, leaf: int | float = 1,
		weight_fraction: float=0.0, features: int | float | str | None = 1.0,
		leaf_nodes: Optional[ int ] = None, impurity: float=0.0, bootstrap: bool=True,
		oob_score: bool=False, jobs: Optional[ int ] = None, rando: Optional[ int ] = 42,
		verbose: int=0, warm: bool=False, ccp_alpha: float=0.0,
		samples: Optional[ int | float ]=None, monotonic: Optional[ object ]=None ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the RandomForest class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    estimators (int): Number of estimators in an ensemble.
		    criterion (str): Tree split or ensemble loss criterion.
		    depth (Optional[int]): Maximum tree depth.
		    split (int | float): Minimum samples required to split an internal tree node.
		    leaf (int | float): Leaf-size or minimum leaf configuration.
		    weight_fraction (float): Minimum weighted fraction required at a leaf node.
		    features (int | float | str | None): Maximum feature-selection setting used by the
		                                         estimator.
		    leaf_nodes (Optional[int]): Maximum number of leaf nodes.
		    impurity (float): Minimum impurity decrease required for a split.
		    bootstrap (bool): Flag enabling bootstrap sampling.
		    oob_score (bool): Flag enabling out-of-bag scoring.
		    jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
		    rando (Optional[int]): Random-state seed passed to the underlying estimator.
		    verbose (int): Verbosity flag or level passed to the estimator.
		    warm (bool): Flag indicating whether previous estimator state is reused across fits.
		    ccp_alpha (float): Complexity parameter used for minimal cost-complexity pruning.
		    samples (Optional[int | float]): Maximum sample setting used by ensemble estimators.
		    monotonic (Optional[object]): Monotonic constraint configuration for compatible
		                                  estimators.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_estimators = estimators
		self.criterion = criterion
		self.max_depth = depth
		self.min_samples_split = split
		self.min_samples_leaf = leaf
		self.min_weight_fraction_leaf = weight_fraction
		self.max_features = features
		self.max_leaf_nodes = leaf_nodes
		self.min_impurity_decrease = impurity
		self.bootstrap = bootstrap
		self.oob_score = oob_score
		self.n_jobs = jobs
		self.random_state = rando
		self.verbose = verbose
		self.warm_start = warm
		self.ccp_alpha = ccp_alpha
		self.max_samples = samples
		self.monotonic_cst = monotonic
		self.model = ske.RandomForestRegressor( n_estimators=self.n_estimators,
			criterion=self.criterion, max_depth=self.max_depth,
			min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features,
			max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
			bootstrap=self.bootstrap, oob_score=self.oob_score, n_jobs=self.n_jobs,
			random_state=self.random_state, verbose=self.verbose, warm_start=self.warm_start,
			ccp_alpha=self.ccp_alpha, max_samples=self.max_samples,
			monotonic_cst=self.monotonic_cst )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the RandomForest class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'n_estimators', 'criterion', 'max_depth',
			'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
			'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs',
			'random_state', 'verbose', 'warm_start', 'ccp_alpha', 'max_samples', 'monotonic_cst',
			'features', 'training_score', 'testing_score', 'mean_absolute_error',
			'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'split_data',
			'train', 'project', 'score', 'analyze', 'scatter_plot' ]
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying RandomForest estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> RandomForest | None:
		"""Train.
		
		Purpose:
		    Fits the underlying RandomForest regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    RandomForest | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'train( self, *args ) -> RandomForest | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted RandomForest regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted RandomForest regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted RandomForest model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted RandomForest regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RandomForest'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class GradientBoost( Regression ):
	"""GradientBoost regression class/object.
	
	Purpose:
	    Gradient Boosting Regression is a powerful machine learning technique that builds an ensemble
	    of weak prediction models, typically small decision trees, in a sequential manner. Each new
	    tree targets and predicts the residual errors of the combined sequence of prior trees,
	    gradually minimizing a chosen loss function to yield high predictive accuracy.
	
	Attributes:
	    model (ske.GradientBoostingRegressor): Underlying regression estimator
	                                           managed by the class/object.
	    loss (str): Loss function minimized by the estimator.
	    learning_rate (float): Estimator learning-rate configuration.
	    n_estimators (int): Number of ensemble estimators.
	    subsample (float): Fraction of training samples used for each boosting stage.
	    criterion (str): Split or impurity criterion used by tree-based estimators.
	    min_samples_split (int | float): Minimum number of samples required to split a tree
	                                     node.
	    min_samples_leaf (int | float): Minimum number of samples permitted in a tree leaf.
	    min_weight_fraction_leaf (float): Minimum weighted sample fraction permitted in a
	                                      tree leaf.
	    max_depth (Optional[int]): Maximum tree depth.
	    min_impurity_decrease (float): Minimum impurity reduction required to split a tree
	                                   node.
	    init (Optional[object]): Centroid, parameter, or weight initialization strategy.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    max_features (int | float | str | None): Feature subset size considered during model
	                                             fitting.
	    alpha (float): Regularization strength or loss parameter passed to the estimator.
	    verbose (int): Estimator verbosity level.
	    max_leaf_nodes (Optional[int]): Maximum number of leaf nodes permitted in each tree.
	    warm_start (bool): Whether repeated fits reuse the preceding fitted estimator state.
	    validation_fraction (float): Fraction of training data reserved for early stopping.
	    n_iter_no_change (Optional[int]): Stable-iteration count required to trigger early
	                                      stopping.
	    tol (float): Optimization tolerance passed to the estimator.
	    ccp_alpha (float): Complexity parameter used for minimal cost-complexity pruning.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: ske.GradientBoostingRegressor
	loss: str
	learning_rate: float
	n_estimators: int
	subsample: float
	criterion: str
	min_samples_split: int | float
	min_samples_leaf: int | float
	min_weight_fraction_leaf: float
	max_depth: Optional[ int ]
	min_impurity_decrease: float
	init: Optional[ object ]
	random_state: Optional[ int ]
	max_features: int | float | str | None
	alpha: float
	verbose: int
	max_leaf_nodes: Optional[ int ]
	warm_start: bool
	validation_fraction: float
	n_iter_no_change: Optional[ int ]
	tol: float
	ccp_alpha: float
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, loss: str = 'squared_error', rate: float=0.1, estimators: int=100,
		subsample: float=1.0, criterion: str = 'friedman_mse', split: int | float = 2,
		leaf: int | float = 1, weight_fraction: float=0.0, depth: Optional[ int ] = 3,
		impurity: float=0.0, init: Optional[ object ]=None, rando: Optional[ int ]=42,
		features: int | float | str | None=None, alpha: float=0.9, verbose: int=0,
		leaf_nodes: Optional[ int ]=None, warm: bool=False, validation_fraction: float=0.1,
		no_change: Optional[ int ]=None, tol: float=1e-4, ccp_alpha: float=0.0 ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the GradientBoost class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    loss (str): Loss function used by the estimator.
		    rate (float): Learning rate used by boosting estimators.
		    estimators (int): Number of estimators in an ensemble.
		    subsample (float): Subsample fraction used by boosting estimators.
		    criterion (str): Tree split or ensemble loss criterion.
		    split (int | float): Minimum samples required to split an internal tree node.
		    leaf (int | float): Leaf-size or minimum leaf configuration.
		    weight_fraction (float): Minimum weighted fraction required at a leaf node.
		    depth (Optional[int]): Maximum tree depth.
		    impurity (float): Minimum impurity decrease required for a split.
		    init (Optional[object]): Initial estimator used by boosting.
		    rando (Optional[int]): Random-state seed passed to the underlying estimator.
		    features (int | float | str | None): Maximum feature-selection setting used by the
		                                         estimator.
		    alpha (float): Regularization strength, loss parameter, or model-specific alpha
		                   value.
		    verbose (int): Verbosity flag or level passed to the estimator.
		    leaf_nodes (Optional[int]): Maximum number of leaf nodes.
		    warm (bool): Flag indicating whether previous estimator state is reused across fits.
		    validation_fraction (float): Training-data fraction reserved for early-stopping
		                                 validation.
		    no_change (Optional[int]): Number of iterations with no improvement before stopping.
		    tol (float): Numerical tolerance used by estimator optimization or convergence
		                 checks.
		    ccp_alpha (float): Complexity parameter used for minimal cost-complexity pruning.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.loss = loss
		self.learning_rate = rate
		self.n_estimators = estimators
		self.subsample = subsample
		self.criterion = criterion
		self.min_samples_split = split
		self.min_samples_leaf = leaf
		self.min_weight_fraction_leaf = weight_fraction
		self.max_depth = depth
		self.min_impurity_decrease = impurity
		self.init = init
		self.random_state = rando
		self.max_features = features
		self.alpha = alpha
		self.verbose = verbose
		self.max_leaf_nodes = leaf_nodes
		self.warm_start = warm
		self.validation_fraction = validation_fraction
		self.n_iter_no_change = no_change
		self.tol = tol
		self.ccp_alpha = ccp_alpha
		self.model = ske.GradientBoostingRegressor( loss=self.loss,
			learning_rate=self.learning_rate, n_estimators=self.n_estimators,
			subsample=self.subsample, criterion=self.criterion,
			min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_depth=self.max_depth,
			min_impurity_decrease=self.min_impurity_decrease, init=self.init,
			random_state=self.random_state, max_features=self.max_features, alpha=self.alpha,
			verbose=self.verbose, max_leaf_nodes=self.max_leaf_nodes, warm_start=self.warm_start,
			validation_fraction=self.validation_fraction, n_iter_no_change=self.n_iter_no_change,
			tol=self.tol, ccp_alpha=self.ccp_alpha )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the GradientBoost class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'loss', 'learning_rate', 'n_estimators', 'subsample',
			'criterion', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf',
			'max_depth', 'min_impurity_decrease', 'init', 'random_state', 'max_features', 'alpha',
			'verbose', 'max_leaf_nodes', 'warm_start', 'validation_fraction', 'n_iter_no_change',
			'tol', 'ccp_alpha', 'features', 'training_score', 'testing_score',
			'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'split_data',
			'train',
			'project', 'score', 'analyze', 'scatter_plot' ]
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying GradientBoost estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GradientBoost | None:
		"""Train.
		
		Purpose:
		    Fits the underlying GradientBoost regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    GradientBoost | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'train( self, *args ) -> GradientBoost | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted GradientBoost regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted GradientBoost regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted GradientBoost model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted GradientBoost regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GradientBoost'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class AdaptiveBoost( Regression ):
	"""AdaptiveBoost regression class/object.
	
	Purpose:
	    Adaptive boosting regression (AdaBoost regression) is a machine learning technique that
	    builds a strong numerical prediction model by sequentially combining multiple weak models,
	    typically small decision trees. It adjusts data point weights after each step, forcing later
	    models to focus heavily on previously hard-to-predict data points
	
	Attributes:
	    model (ske.AdaBoostRegressor): Underlying regression estimator managed by
	                                   the class/object.
	    estimator (Optional[object]): Estimator used by the selector, ensemble, or
	                                  forecasting workflow.
	    n_estimators (int): Number of ensemble estimators.
	    learning_rate (float): Estimator learning-rate configuration.
	    loss (str): Loss function minimized by the estimator.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: ske.AdaBoostRegressor
	estimator: Optional[ object ]
	n_estimators: int
	learning_rate: float
	loss: str
	random_state: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, estimator: Optional[ object ]=None, estimators: int=50,
		rate: float=1.0, loss: str='linear', rando: Optional[ int ]=42 ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the AdaptiveBoost class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    estimator (Optional[object]): Base estimator used by an ensemble class/object.
		    estimators (int): Number of estimators in an ensemble.
		    rate (float): Learning rate used by boosting estimators.
		    loss (str): Loss function used by the estimator.
		    rando (Optional[int]): Random-state seed passed to the underlying estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.estimator = estimator
		self.n_estimators = estimators
		self.learning_rate = rate
		self.loss = loss
		self.random_state = rando
		self.model = ske.AdaBoostRegressor( estimator=self.estimator,
			n_estimators=self.n_estimators, learning_rate=self.learning_rate, loss=self.loss,
			random_state=self.random_state )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the AdaptiveBoost class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'estimator', 'n_estimators', 'learning_rate', 'loss',
			'random_state', 'base_estimator', 'features', 'mean_absolute_error',
			'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score',
			'median_absolute_error', 'max_error', 'train', 'project', 'score', 'analyze',
			'scatter_plot', 'training_score', 'testing_score' ]
	
	@property
	def base_estimator( self ) -> object | None:
		"""Base estimator.
		
		Purpose:
		    Returns fitted `base_estimator` metadata from the underlying AdaptiveBoost estimator
		    after training.
		
		Returns:
		    object | None: Fitted `base_estimator` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if hasattr( self.model, 'estimator_' ):
			return self.model.estimator_
		raise AttributeError( 'The model has not been trained!' )
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying AdaptiveBoost estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> AdaptiveBoost | None:
		"""Train.
		
		Purpose:
		    Fits the underlying AdaptiveBoost regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    AdaptiveBoost | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'train( self, *args ) -> AdaptiveBoost | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted AdaptiveBoost regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted AdaptiveBoost regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted AdaptiveBoost model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted AdaptiveBoost regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'AdaptiveBoost'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class BaggingModel( Regression ):
	"""BaggingModel regression class/object.
	
	Purpose:
	    Bagging regression (Bootstrap Aggregating) is a machine learning ensemble method that
	    reduces model variance and prevents overfitting. It creates random subsets of data with
	    replacement, trains a base regression model (like a decision tree) on each subset
	    independently, and averages all the individual numerical predictions
	
	Attributes:
	    model (ske.BaggingRegressor): Underlying regression estimator managed by the
	                                  class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    estimator (Optional[object]): Estimator used by the selector, ensemble, or
	                                  forecasting workflow.
	    n_estimators (int): Number of ensemble estimators.
	    max_samples (Optional[int | float]): Maximum sample count or fraction drawn for each
	                                         ensemble estimator.
	    max_features (int | float): Feature subset size considered during model fitting.
	    bootstrap (bool): Whether ensemble estimators are fitted from bootstrap samples.
	    bootstrap_features (bool): Whether ensemble features are sampled with replacement.
	    oob_score (bool): Whether out-of-bag samples are used to estimate generalization
	                      performance.
	    warm_start (bool): Whether repeated fits reuse the preceding fitted estimator state.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    verbose (int): Estimator verbosity level.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: ske.BaggingRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	estimator: Optional[ object ]
	n_estimators: int
	max_samples: Optional[ int | float ]
	max_features: int | float
	bootstrap: bool
	bootstrap_features: bool
	oob_score: bool
	warm_start: bool
	n_jobs: Optional[ int ]
	random_state: Optional[ int ]
	verbose: int
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, estimator: Optional[ object ]=None, num: int=10,
		samples: Optional[ int | float ]=None, features: int | float=1.0,
		bootstrap: bool=True, bootstrap_features: bool=False, oob_score: bool=False,
		warm: bool=False, jobs: Optional[ int ]=None, rando: Optional[ int ]=None,
		verbose: int=0 ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the BaggingModel class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    estimator (Optional[object]): Base estimator used by an ensemble class/object.
		    num (int): Number of neighbors, estimators, or model-specific components.
		    samples (Optional[int | float]): Maximum sample setting used by ensemble estimators.
		    features (int | float): Maximum feature-selection setting used by the estimator.
		    bootstrap (bool): Flag enabling bootstrap sampling.
		    bootstrap_features (bool): Configuration value passed to the underlying regression
		                               estimator.
		    oob_score (bool): Flag enabling out-of-bag scoring.
		    warm (bool): Flag indicating whether previous estimator state is reused across fits.
		    jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
		    rando (Optional[int]): Random-state seed passed to the underlying estimator.
		    verbose (int): Verbosity flag or level passed to the estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.estimator = estimator
		self.n_estimators = num
		self.max_samples = samples
		self.max_features = features
		self.bootstrap = bootstrap
		self.bootstrap_features = bootstrap_features
		self.oob_score = oob_score
		self.warm_start = warm
		self.n_jobs = jobs
		self.random_state = rando
		self.verbose = verbose
		self.model = ske.BaggingRegressor( estimator=self.estimator,  n_estimators=self.n_estimators,
			max_samples=self.max_samples, max_features=self.max_features, bootstrap=self.bootstrap,
			bootstrap_features=self.bootstrap_features, oob_score=self.oob_score,
			warm_start=self.warm_start, n_jobs=self.n_jobs, random_state=self.random_state,
			verbose=self.verbose )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the BaggingModel class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'estimator', 'n_estimators', 'max_samples', 'max_features',
			'bootstrap', 'bootstrap_features', 'oob_score', 'warm_start', 'n_jobs', 'random_state',
			'verbose', 'base_estimator', 'features', 'training_score', 'testing_score',
			'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'split_data',
			'train', 'project', 'score', 'analyze', 'scatter_plot' ]
	
	@property
	def base_estimator( self ) -> object | None:
		"""Base estimator.
		
		Purpose:
		    Returns fitted `base_estimator` metadata from the underlying BaggingModel estimator
		    after training.
		
		Returns:
		    object | None: Fitted `base_estimator` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if hasattr( self.model, 'estimator_' ):
			return self.model.estimator_
		raise AttributeError( 'The model has not been trained!' )
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying BaggingModel estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> BaggingModel | None:
		"""Train.
		
		Purpose:
		    Fits the underlying BaggingModel regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    BaggingModel | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'train( self, *args ) -> BaggingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted BaggingModel regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted BaggingModel regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted BaggingModel model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted BaggingModel regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'BaggingModel'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class VotingModel( Regression ):
	"""VotingModel regression class/object.
	
	Purpose:
	    A voting regressor is a machine learning ensemble meta-estimator that combines distinct
	    base regression models to predict continuous numerical values. It fits multiple models on
	    the entire dataset and computes a final prediction by taking the average or the weighted
	    average of all individual model predictions
	
	Attributes:
	    model (ske.VotingRegressor): Underlying regression estimator managed by the
	                                 class/object.
	    estimators (List[tuple[str, object]]): Named estimator collection used by ensemble
	                                           wrappers.
	    weights (Optional[List[float]]): Optional estimator or neighbor weights.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    verbose (bool): Estimator verbosity level.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: ske.VotingRegressor
	estimators: List[ tuple[ str, object ] ]
	weights: Optional[ List[ float ] ]
	n_jobs: Optional[ int ]
	verbose: bool
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, est: Optional[ List[ tuple[ str, object ] ] ] = None,
		weights: Optional[ List[ float ] ] = None, jobs: Optional[ int ] = None,
		verbose: bool=False ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the VotingModel class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    est (Optional[List[tuple[str, object]]]): Named estimator list used by voting or
		                                              stacking wrappers.
		    weights (Optional[List[float]]): Estimator weights used during voting or averaging.
		    jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
		    verbose (bool): Verbosity flag or level passed to the estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.estimators = est if est is not None else [ ('least_squares', skl.LinearRegression( )),
			('ridge', skl.Ridge( )), ('nearest_neighbor', skn.KNeighborsRegressor( )) ]
		self.weights = weights
		self.n_jobs = jobs
		self.verbose = verbose
		self.model = ske.VotingRegressor( estimators=self.estimators, weights=self.weights,
			n_jobs=self.n_jobs, verbose=self.verbose )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the VotingModel class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'estimators', 'weights', 'n_jobs', 'verbose', 'features',
			'named_estimators', 'training_score', 'testing_score', 'mean_absolute_error',
			'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score',
			'median_absolute_error', 'max_error', 'split_data', 'train', 'project', 'score',
			'analyze', 'scatter_plot' ]
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying VotingModel estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	@property
	def named_estimators( self ) -> Dict[ str, object ] | None:
		"""Named estimators.
		
		Purpose:
		    Returns fitted `named_estimators` metadata from the underlying VotingModel estimator
		    after training.
		
		Returns:
		    Dict[str, object] | None: Fitted `named_estimators` metadata from the underlying
		                              estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'named_estimators_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.named_estimators_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> VotingModel | None:
		"""Train.
		
		Purpose:
		    Fits the underlying VotingModel regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    VotingModel | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'train( self, *args ) -> VotingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted VotingModel regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted VotingModel regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted VotingModel model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted VotingModel regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'VotingModel'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class StackingModel( Regression ):
	"""StackingModel regression class/object.
	
	Purpose:
	    Stacking regression is an advanced machine learning ensemble technique that combines
	    multiple base regression models (like linear regression, support vector regression, or
	    random forests) using a final meta-model. Instead of taking a simple average, the meta-model
	    learns how to best weigh and combine the predictions of the base models to output a final,
	    more accurate prediction.
	
	Attributes:
	    model (ske.StackingRegressor): Underlying regression estimator managed by
	                                   the class/object.
	    estimators (List[tuple[str, object]]): Named estimator collection used by ensemble
	                                           wrappers.
	    final_estimator (object | None): Final estimator used by stacked regression.
	    cv (int | None): Cross-validation strategy or number of folds.
	    n_jobs (int | None): Number of parallel worker jobs used by the estimator.
	    passthrough (bool): Whether unselected columns bypass the configured
	                        transformations.
	    verbose (int): Estimator verbosity level.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: ske.StackingRegressor
	estimators: List[ tuple[ str, object ] ]
	final_estimator: object | None
	cv: int | None
	n_jobs: int | None
	passthrough: bool
	verbose: int
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, est: Optional[ List[ tuple[ str, object ] ] ] = None, final: object = None,
		cv: int=None, jobs: int=None, passthrough: bool=False, verbose: int=0 ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the StackingModel class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    est (Optional[List[tuple[str, object]]]): Named estimator list used by voting or
		                                              stacking wrappers.
		    final (object): Final estimator used by stacked regression.
		    cv (int): Cross-validation configuration used by stacked regression.
		    jobs (int): Number of parallel worker jobs used by the estimator.
		    passthrough (bool): Flag indicating whether original features are passed to the
		                        final estimator.
		    verbose (int): Verbosity flag or level passed to the estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.estimators = est if est is not None else [ ('least_squares', skl.LinearRegression( )),
			('ridge', skl.Ridge( )), ('nearest_neighbor', skn.KNeighborsRegressor( )) ]
		self.final_estimator = final
		self.cv = cv
		self.n_jobs = jobs
		self.passthrough = passthrough
		self.verbose = verbose
		self.model = ske.StackingRegressor( estimators=self.estimators,
			final_estimator=self.final_estimator, cv=self.cv, n_jobs=self.n_jobs,
			passthrough=self.passthrough )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the StackingModel class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'estimators', 'final_estimator', 'cv', 'n_jobs',
			'passthrough', 'verbose', 'features', 'estimator_list', 'final', 'training_score',
			'testing_score', 'mean_absolute_error', 'mean_squared_error',
			'root_mean_squared_error', 'r2_score', 'explained_variance_score',
			'median_absolute_error', 'max_error', 'split_data', 'train', 'project', 'score',
			'analyze', 'scatter_plot' ]
	
	@property
	def estimator_list( self ) -> List[ object ] | None:
		"""Estimator list.
		
		Purpose:
		    Returns fitted `estimator_list` metadata from the underlying StackingModel estimator
		    after training.
		
		Returns:
		    List[object] | None: Fitted `estimator_list` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'estimators_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.estimators_
	
	@property
	def final( self ) -> object | None:
		"""Final.
		
		Purpose:
		    Returns fitted `final` metadata from the underlying StackingModel estimator after
		    training.
		
		Returns:
		    object | None: Fitted `final` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'final_estimator_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.final_estimator_
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying StackingModel estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return (self.X_train, self.X_test, self.y_train, self.y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> StackingModel | None:
		"""Train.
		
		Purpose:
		    Fits the underlying StackingModel regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    StackingModel | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'train( self, *args ) -> StackingModel | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted StackingModel regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted StackingModel regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame( { 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted StackingModel model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted StackingModel regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StackingModel'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class SupportVector( Regression ):
	"""SupportVector regression class/object.
	
	Purpose:
	    Support Vector Regression (SVR) is a machine learning algorithm that predicts continuous
	    numbers. Instead of minimizing all errors like normal regression, SVR builds a flexible
	    error boundary (tube) around the data. It ignores errors smaller than a set limit (ε)
	    and focuses only on hard points outside the boundary
	
	Attributes:
	    model (skv.SVR): Underlying regression estimator managed by the class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    kernel (str): Kernel function used to transform feature-space relationships.
	    degree (int): Degree used by the polynomial kernel or feature expansion.
	    gamma (str | float): Kernel coefficient or minimum loss reduction used by the
	                         estimator.
	    coef0 (float): Independent term used by polynomial and sigmoid kernels.
	    tol (float): Optimization tolerance passed to the estimator.
	    penalty (float): Regularization penalty applied during estimator fitting.
	    epsilon (float): Estimator-specific tolerance or insensitive-loss margin.
	    shrinking (bool): Whether the support-vector shrinking heuristic is enabled.
	    cache_size (float): Kernel cache size, in megabytes.
	    verbose (bool): Estimator verbosity level.
	    max_iter (int): Maximum number of estimator iterations.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split.
	    X_train (Optional[np.ndarray]): Feature matrix used to fit the model.
	    X_test (Optional[np.ndarray]): Feature matrix reserved for model evaluation.
	    y_train (Optional[np.ndarray]): Target values used to fit the model.
	    y_test (Optional[np.ndarray]): Target values reserved for model evaluation."""
	model: skv.SVR
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	kernel: str
	degree: int
	gamma: str | float
	coef0: float
	tol: float
	penalty: float
	epsilon: float
	shrinking: bool
	cache_size: float
	verbose: bool
	max_iter: int
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	X_train: Optional[ np.ndarray ]
	X_test: Optional[ np.ndarray ]
	y_train: Optional[ np.ndarray ]
	y_test: Optional[ np.ndarray ]
	
	def __init__( self, kernel: str = 'rbf', degree: int=3, gamma: str | float = 'scale',
		coef0: float=0.0, tol: float=1e-3, penalty: float=1.0, epsilon: float=0.1,
		shrinking: bool=True, cache: float=200.0, verbose: bool=False, iters: int=-1 ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the SupportVector class/object with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    kernel (str): Kernel configuration used by kernel-based estimators.
		    degree (int): Polynomial degree used by kernel-based estimators.
		    gamma (str | float): Kernel coefficient used by support-vector regression.
		    coef0 (float): Independent kernel coefficient used by polynomial or sigmoid kernels.
		    tol (float): Numerical tolerance used by estimator optimization or convergence
		                 checks.
		    penalty (float): Regularization parameter used by the estimator.
		    epsilon (float): Epsilon-insensitive loss parameter or robustness threshold.
		    shrinking (bool): Flag enabling the shrinking heuristic for support-vector
		                      optimization.
		    cache (float): Kernel cache size in megabytes.
		    verbose (bool): Verbosity flag or level passed to the estimator.
		    iters (int): Maximum number of optimization iterations.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.kernel = kernel
		self.degree = degree
		self.gamma = gamma
		self.coef0 = coef0
		self.tol = tol
		self.penalty = penalty
		self.epsilon = epsilon
		self.shrinking = shrinking
		self.cache_size = cache
		self.verbose = verbose
		self.max_iter = iters
		self.model = skv.SVR( kernel=self.kernel, degree=self.degree, gamma=self.gamma,
			coef0=self.coef0, tol=self.tol, C=self.penalty, epsilon=self.epsilon,
			shrinking=self.shrinking, cache_size=self.cache_size, verbose=self.verbose,
			max_iter=self.max_iter )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.median_absolute_error = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the SupportVector class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'kernel', 'degree', 'gamma', 'coef0', 'tol', 'penalty',
			'epsilon', 'shrinking', 'cache_size', 'verbose', 'max_iter', 'features',
			'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score', 'median_absolute_error', 'max_error', 'train', 'project',
			'score', 'analyze', 'scatter_plot', 'training_score', 'testing_score' ]
	
	@property
	def features( self ) -> int:
		"""Features.
		
		Purpose:
		    Returns fitted `features` metadata from the underlying SupportVector estimator after
		    training.
		
		Returns:
		    int: Fitted `features` metadata from the underlying estimator.
		
		Raises:
		    AttributeError: Raised when fitted estimator metadata is unavailable."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model has not been trained!' )
		return self.model.n_features_in_
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.X_train, self.X_test, self.y_train, self.y_test = split( X, y, test_size=size,
				random_state=random )
			return self.X_train, self.X_test, self.y_train, self.y_test
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> SupportVector | None:
		"""Train.
		
		Purpose:
		    Fits the underlying SupportVector regression estimator to feature and target arrays
		    and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    SupportVector | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			self.prediction = None
			
			if self.X_train is not None and self.y_train is not None:
				self.training_score = self.model.score( self.X_train, self.y_train )
			
			if self.X_test is not None and self.y_test is not None:
				self.testing_score = self.model.score( self.X_test, self.y_test )
			
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'train( self, *args ) -> SupportVector | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted SupportVector regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted SupportVector regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.r2_score = r2_score( y, self.prediction )
			df_metrics = pd.DataFrame(
				{ 'Metric': [ 'Training Score', 'Testing Score', 'R-Squared Score' ],
					'Value': [ self.training_score if self.training_score is not None else np.nan,
						self.testing_score if self.testing_score is not None else np.nan,
						self.r2_score ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted SupportVector model, including
		    error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.r2_score = r2_score( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			self.median_absolute_error = median_absolute_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			df_metrics = pd.DataFrame( {
				'Metric': [ 'Training Score', 'Testing Score', 'R-Squared', 'MAE', 'MSE', 'RMSE',
					'EVS', 'Median AE', 'MAX' ],
				'Value': [ self.training_score if self.training_score is not None else np.nan,
					self.testing_score if self.testing_score is not None else np.nan,
					self.r2_score,
					self.mean_absolute_error, self.mean_squared_error,
					self.root_mean_squared_error,
					self.explained_variance_score, self.median_absolute_error, self.max_error ] } )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted SupportVector regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.project( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SupportVector'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class GaussianProcess( Regression ):
	"""GaussianProcess regression class/object.
	
	Purpose:
	    Gaussian Process Regression (GPR) is a non-parametric, probabilistic machine learning method.
	    Instead of outputting a single estimated value, GPR predicts a complete probability distribution
	    over all possible functions. It provides both a predicted mean value and an explicit measure
	    of uncertainty for every data point
	
	Attributes:
	    model (GaussianProcessRegressor): Underlying regression estimator managed by
	                                      the class/object.
	    kernel (object | None): Kernel function used to transform feature-space
	                            relationships.
	    alpha (float | np.ndarray): Regularization strength or loss parameter passed to the
	                                estimator.
	    optimizer (str | object | None): Optimizer used to select Gaussian-process kernel
	                                     parameters.
	    n_restarts_optimizer (int): Number of optimizer restarts used for kernel fitting.
	    normalize_y (bool): Whether target values are normalized before Gaussian-process
	                        fitting.
	    copy_X_train (bool): Whether the fitted neighbor model stores a copy of training
	                         features.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    max_error (Optional[float]): Most recent maximum residual error metric.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split."""
	model: GaussianProcessRegressor
	kernel: object | None
	alpha: float | np.ndarray
	optimizer: str | object | None
	n_restarts_optimizer: int
	normalize_y: bool
	copy_X_train: bool
	random_state: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	max_error: Optional[ float ]
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, kernel: object = None, alpha: float=1e-10,
		optimizer: str = 'fmin_l_bfgs_b', restarts: int=0, normalize: bool=False,
		copy: bool=True, rando: int=None ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the GaussianProcess class with estimator configuration, runtime state,
		    cached prediction fields, and regression metric fields required by training and
		    evaluation workflows.
		
		Args:
		    kernel (object): Kernel configuration used by kernel-based estimators.
		    alpha (float): Regularization strength, loss parameter, or model-specific alpha
		                   value.
		    optimizer (str): Optimizer used by Gaussian-process regression.
		    restarts (int): Number of optimizer restarts.
		    normalize (bool): Flag indicating whether targets are normalized before fitting.
		    copy (bool): Flag indicating whether input feature data is copied during fitting.
		    rando (int): Random-state seed passed to the underlying estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.kernel = kernel
		self.alpha = alpha
		self.optimizer = optimizer
		self.n_restarts_optimizer = restarts
		self.normalize_y = normalize
		self.copy_X_train = copy
		self.random_state = rando
		self.model = GaussianProcessRegressor( kernel=self.kernel, alpha=self.alpha,
			optimizer=self.optimizer, n_restarts_optimizer=self.n_restarts_optimizer,
			normalize_y=self.normalize_y, copy_X_train=self.copy_X_train,
			random_state=self.random_state )
		self.prediction = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the GaussianProcess class/object for
		    interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'kernel', 'alpha', 'optimizer', 'n_restarts_optimizer',
			'normalize_y', 'copy_X_train', 'random_state', 'mean_absolute_error',
			'mean_squared_error', 'root_mean_squared_error', 'r2_score',
			'explained_variance_score',
			'median_absolute_error', 'max_error', 'train', 'project', 'score', 'analyze',
			'scatter_plot', 'training_score', 'testing_score' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> GaussianProcess | None:
		"""Train.
		
		Purpose:
		    Fits the underlying GaussianProcess regression estimator to feature and target
		    arrays and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    GaussianProcess | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'train( self, *args ) -> GaussianProcess | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted GaussianProcess regression
		    estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted GaussianProcess regression estimator with its primary scoring
		    behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			_metrics = { 'Training Score': self.training_score, 'Testing Score':
				self.testing_score,
				'R-Squared Score': self.r2_score, }
			
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted GaussianProcess model,
		    including error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			_metrics = { 'MAE': self.mean_absolute_error, 'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error, 'EVS': self.explained_variance_score,
				'MAX': self.max_error, }
			
			_data = pd.Series( _metrics )
			df_metrics = pd.DataFrame( _data )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted GaussianProcess regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'GaussianProcess'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception

class MultiLayerPerceptron( Regression ):
	"""MultiLayerPerceptron regression class/object.
	
	Purpose:
	    A Multilayer Perceptron (MLP) for regression is a feedforward artificial neural network
	    designed to predict continuous numerical values. Unlike classification models that output
	    class probabilities via a softmax layer, a regression MLP uses a linear output node with
	    an identity activation function and is optimized using loss functions like mean squared error (MSE)
	
	Attributes:
	    model (skm.MLPRegressor): Underlying regression estimator managed by the
	                              class/object.
	    prediction (Optional[np.ndarray]): Most recent prediction array returned by the
	                                       fitted estimator.
	    transformed_data (Optional[np.ndarray]): Most recent output produced by the
	                                             preprocessing operation.
	    mean_absolute_error (Optional[float]): Most recent mean absolute error metric.
	    mean_squared_error (Optional[float]): Most recent mean squared error metric.
	    root_mean_squared_error (Optional[float]): Most recent root mean squared error
	                                               metric.
	    r2_score (Optional[float]): Most recent coefficient-of-determination metric.
	    explained_variance_score (Optional[float]): Most recent explained-variance metric.
	    median_absolute_error (Optional[float]): Most recently calculated median absolute
	                                             error.
	    random_state (Optional[int]): Random seed or random-state configuration used by the
	                                  estimator.
	    alpha (Optional[float]): Regularization strength or loss parameter passed to the
	                             estimator.
	    learning (str): Learning-rate schedule used by stochastic gradient descent.
	    activation_function (str): Activation function used by the neural network.
	    solver (str): Solver selected for estimator optimization.
	    hidden_layers (tuple): Sizes of the neural network hidden layers.
	    testing_score (Optional[float]): Most recent estimator score on the testing split.
	    training_score (Optional[float]): Most recent estimator score on the training split."""
	model: skm.MLPRegressor
	prediction: Optional[ np.ndarray ]
	transformed_data: Optional[ np.ndarray ]
	mean_absolute_error: Optional[ float ]
	mean_squared_error: Optional[ float ]
	root_mean_squared_error: Optional[ float ]
	r2_score: Optional[ float ]
	explained_variance_score: Optional[ float ]
	median_absolute_error: Optional[ float ]
	random_state: Optional[ int ]
	alpha: Optional[ float ]
	learning: str
	activation_function: str
	solver: str
	hidden_layers: tuple
	testing_score: Optional[ float ]
	training_score: Optional[ float ]
	
	def __init__( self, hidden: tuple = (100,), activ: str='relu', solver: str='adam',
		alpha: float=0.0001, learning: str='constant', rando: int=42 ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes the MultiLayerPerceptron class/object with estimator configuration, runtime
		    state, cached prediction fields, and regression metric fields required by training
		    and evaluation workflows.
		
		Args:
		    hidden (tuple): Hidden-layer architecture for neural-network regression.
		    activ (str): Activation function used by neural-network regression.
		    solver (str): Optimization solver used by the estimator.
		    alpha (float): Regularization strength, loss parameter, or model-specific alpha
		                   value.
		    learning (str): Learning-rate schedule used by neural-network regression.
		    rando (int): Random-state seed passed to the underlying estimator.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.hidden_layers = hidden
		self.activation_function = activ
		self.solver = solver
		self.alpha = alpha
		self.learning = learning
		self.random_state = rando
		self.model = skm.MLPRegressor( hidden_layer_sizes=self.hidden_layers,
			activation=self.activation_function, solver=self.solver, alpha=self.alpha,
			learning_rate=self.learning, random_state=self.random_state )
		self.prediction = None
		self.transformed_data = None
		self.mean_absolute_error = 0.0
		self.mean_squared_error = 0.0
		self.root_mean_squared_error = 0.0
		self.r2_score = 0.0
		self.explained_variance_score = 0.0
		self.max_error = 0.0
		self.training_score = 0.0
		self.testing_score = 0.0
	
	def __dir__( self ) -> List[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public API surface exposed by the MultiLayerPerceptron class/object
		    for interactive inspection and documentation generation.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'prediction', 'hidden_layers', 'activation_function', 'solver', 'alpha',
			'learning', 'random_state', 'mean_absolute_error', 'mean_squared_error',
			'root_mean_squared_error', 'r2_score', 'explained_variance_score',
			'median_absolute_error', 'max_error', 'train', 'project', 'score', 'analyze',
			'scatter_plot', 'training_score', 'testing_score' ]
	
	def split_data( self, X: np.ndarray, y: np.ndarray, size: float=0.2,
		random: int=42 ) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
		"""Split data.
		
		Purpose:
		    Splits aligned feature and target arrays into training and testing partitions using
		    sklearn.model_selection.train_test_split with class/object defaults.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		    size (float): Testing-set proportion used by train/test splitting.
		    random (int): Random seed used by train/test splitting.
		
		Returns:
		    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing training
		                                                           features, testing features,
		                                                           training targets, and testing
		                                                           targets.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			X_train, X_test, y_train, y_test = split( X, y, test_size=size, random_state=random )
			return (X_train, X_test, y_train, y_test)
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = ('split_data( self, *args ) -> tuple[np.ndarray, np.ndarray, '
			                    'np.ndarray, np.ndarray]')
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> MultiLayerPerceptron | None:
		"""Train.
		
		Purpose:
		    Fits the underlying MultiLayerPerceptron regression estimator to feature and target
		    arrays and returns the class/object for chained modeling workflows.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    MultiLayerPerceptron | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'train( self, *args ) -> MultiLayerPerceptron | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates continuous target predictions from the fitted MultiLayerPerceptron
		    regression estimator and caches the prediction array on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		
		Returns:
		    np.ndarray | None: Predicted continuous target values generated by the fitted
		                       estimator.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Evaluates the fitted MultiLayerPerceptron regression estimator with its primary
		    scoring behavior and returns the score in the class/object reporting format.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing the primary model score and evaluation
		                         fields.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			X_training, X_testing, y_training, y_testing = split( X, y, test_size=0.2 )
			self.training_score = self.model.score( X_training, y_training )
			self.testing_score = self.model.score( X_testing, y_testing )
			self.r2_score = r2_score( y, self.prediction )
			_metrics = { 'Training Score': self.training_score, 'Testing Score':
				self.testing_score,
				'R-Squared Score': self.r2_score, }
			
			idx = range( len( _metrics.items( ) ) )
			df_metrics = pd.DataFrame( _metrics, index=idx )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> pd.DataFrame | None:
		"""Analyze.
		
		Purpose:
		    Computes regression evaluation metrics for the fitted MultiLayerPerceptron model,
		    including error, explained-variance, and coefficient-of-determination measures.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing regression metrics computed from observed
		                         and predicted target values.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			self.prediction = self.project( X )
			self.mean_absolute_error = mean_absolute_error( y, self.prediction )
			self.mean_squared_error = mean_squared_error( y, self.prediction )
			self.root_mean_squared_error = root_mean_squared_error( y, self.prediction )
			self.max_error = max_error( y, self.prediction )
			self.explained_variance_score = explained_variance_score( y, self.prediction )
			_metrics = { 'MAE': self.mean_absolute_error, 'MSE': self.mean_squared_error,
				'RMSE': self.root_mean_squared_error, 'EVS': self.explained_variance_score,
				'MAX': self.max_error, }
			
			_data = pd.Series( _metrics )
			df_metrics = pd.DataFrame( _data )
			return df_metrics
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def scatter_plot( self, X: np.ndarray, y: np.ndarray ) -> None:
		"""Scatter plot.
		
		Purpose:
		    Renders a diagnostic scatter plot comparing observed target values against
		    predictions from the fitted MultiLayerPerceptron regression estimator.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting, prediction, splitting, scoring, or
		                    metric evaluation.
		    y (np.ndarray): Target vector aligned to the rows of `X`.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    Error: Raised when validation or wrapped regression estimator execution fails."""
		try:
			throw_if( 'X', X )
			throw_if( 'y', y )
			_training = self.training_score if self.training_score is not None else 0.0
			_testing = self.testing_score if self.testing_score is not None else 0.0
			_text = f'Training Score = {_training:.1%}\nTesting Score = {_testing:.1%}\n'
			y_pred = self.model.predict( X )
			plt.figure( figsize=(8, 6) )
			sns.regplot( x=y, y=y_pred, scatter_kws={ 'alpha': 0.6 }, line_kws={ 'color': 'red' } )
			plt.plot( [ y.min( ), y.max( ) ], [ y.min( ), y.max( ) ], 'k--',
				label='Perfect Prediction' )
			plt.text( x=y.min( ), y=y.max( ) * 0.95, s=_text, fontsize=8,
				bbox=dict( facecolor='white', alpha=0.7 ) )
			plt.xlabel( 'Observations' )
			plt.ylabel( 'Estimates' )
			plt.title( 'Observations vs Estimates' )
			plt.grid( visible=True )
			plt.tight_layout( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLayerPerceptron'
			exception.method = 'scatter_plot( self, *args ) -> None'
			Logger( ).write( exception )
			raise exception
