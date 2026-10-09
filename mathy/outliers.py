"""******************************************************************************************
  Assembly:                mathy
  Filename:                outliers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="outliers.py" company="Terry D. Eppler">

     outliers.py
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
    Provides anomaly-detection and outlier-analysis wrappers for Mathy workflows. The module
    centralizes Isolation Forest, One-Class SVM, Local Outlier Factor, and Elliptic Envelope
    estimators behind a common training, prediction, scoring, and summary-analysis interface.
</summary>

Purpose:
    Supports unsupervised anomaly detection workflows by standardizing input validation, model
    fitting, inlier/outlier prediction, anomaly-score extraction, aggregate-count calculation,
    and compact visualization of detection results. The wrappers expose consistent attributes for
    predictions, anomaly scores, inlier counts, and outlier counts so downstream notebooks,
    dashboards, and documentation can consume estimator outputs without estimator-specific code.

Notes:
    Prediction labels follow the sklearn anomaly-detection convention where `1` represents an
    inlier and `-1` represents an outlier.
******************************************************************************************"""
from __future__ import annotations
from typing import Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble as en
import sklearn.svm as sv
import sklearn.neighbors as nn
import sklearn.covariance as cv
from sklearn.metrics import classification_report
from boogr import Error, Logger

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

class Outlier( ):
	"""Outlier-analysis interface.
	
	Purpose:
	    Establishes the common contract implemented by anomaly-detection wrappers in this
	    module. The interface defines shared runtime attributes for predictions, anomaly
	    scores, inlier counts, and outlier counts while requiring concrete estimators to
	    provide training, projection, scoring, and analysis behavior.
	
	Attributes:
	    prediction (Optional[np.ndarray]): Most recent inlier/outlier prediction vector.
	    probability (Optional[np.ndarray]): Optional probability output retained for
	                                        interface compatibility.
	    decision (Optional[np.ndarray]): Optional decision-function output retained for
	                                     interface compatibility.
	    max_depth (Optional[int]): Optional model-depth metadata retained for interface
	                               compatibility.
	    random_state (Optional[int]): Optional random-state metadata retained for interface
	                                  compatibility.
	    anomaly_scores (Optional[float]): Most recent anomaly-score output or summary value.
	    learning_rate (Optional[float]): Optional learning-rate metadata retained for
	                                     interface compatibility.
	    outliers (Optional[float]): Count of samples predicted as outliers.
	    inliers (Optional[float]): Count of samples predicted as inliers."""
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	decision: Optional[ np.ndarray ]
	max_depth: Optional[ int ]
	random_state: Optional[ int ]
	anomaly_scores: Optional[ float ]
	learning_rate: Optional[ float ]
	outliers: Optional[ float ]
	inliers: Optional[ float ]
	
	def __init__( self ):
		"""Initializes a `Outlier` instance and its runtime state.
		
		Purpose:
		    Initializes a `Outlier` instance and its runtime state.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		self.outliers = 0.0
		self.inliers = 0.0
	
	def train( self, X: np.ndarray, y: np.ndarray ) -> object | None:
		"""Fit an outlier detector.
		
		Purpose:
		    Defines the required training contract for concrete anomaly-detection wrappers.
		    Implementations must fit the wrapped estimator to a feature matrix and return the
		    fitted class/object when training completes.
		
		Args:
		    X (np.ndarray): Feature matrix used to fit the detector.
		    y (np.ndarray): Optional target vector accepted for interface consistency.
		
		Returns:
		    object | None: Fitted concrete detector class/object.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> np.ndarray:
		"""Predict inliers and outliers.
		
		Purpose:
		    Defines the required projection contract for concrete anomaly-detection wrappers.
		    Implementations must return a vector of inlier and outlier labels for the supplied
		    samples.
		
		Args:
		    X (np.ndarray): Feature matrix used to generate predictions.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    np.ndarray: Prediction vector where `1` denotes inlier samples and `-1` denotes
		                outlier samples.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] ) -> float:
		"""Score anomaly predictions.
		
		Purpose:
		    Defines the required scoring contract for concrete anomaly-detection wrappers.
		    Implementations must return estimator-specific score output suitable for per-sample
		    review or model diagnostics.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute anomaly scores.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    float: Estimator-specific anomaly score output.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: np.ndarray ) -> Dict[ str, float ] | None:
		"""Analyze anomaly results.
		
		Purpose:
		    Defines the required analysis contract for concrete anomaly-detection wrappers.
		    Implementations must summarize inlier and outlier counts, calculate quality or
		    contamination metrics, and return a compact analysis object for downstream
		    reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to generate the analysis.
		    y (np.ndarray): Optional target vector accepted for interface consistency.
		
		Returns:
		    Dict[str, float] | None: Dictionary or dataframe containing aggregate
		                             anomaly-detection metrics.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError

class IsolationForest( Outlier ):
	"""Provides IsolationForest model functionality.
	
	Purpose:
	    Detects anomalies by fitting an ensemble of randomized isolation trees and
	    identifying samples that require fewer splits to isolate. The class/object stores the
	    fitted estimator, contamination setting, prediction vector, anomaly scores, and
	    aggregate inlier/outlier counts for consistent downstream analysis.
	
	Attributes:
	    model (en.IsolationForest): Underlying isolation-forest estimator.
	    contamination (float): Expected proportion of outliers in the input data.
	    prediction (Optional[np.ndarray]): Most recent inlier/outlier prediction vector.
	    anomaly_scores (Optional[np.ndarray]): Most recent isolation-forest decision scores."""
	model: en.IsolationForest
	contamination: float
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	
	def __init__( self, contamination: float = 0.1 ) -> None:
		"""Initialize IsolationForest.
		
		Purpose:
		    Configures the IsolationForest estimator with the expected contamination
		    rate and initializes prediction and anomaly-score caches for later training and
		    scoring operations.
		
		Args:
		    contamination (float): Expected proportion of outliers in the input data.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.contamination = contamination
		self.model = en.IsolationForest( contamination=self.contamination )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> IsolationForest | None:
		"""Fit the IsolationForest detector.
		
		Purpose:
		    Validates the supplied feature matrix, fits the underlying IsolationForest
		    estimator, refreshes prediction and anomaly-score state when the estimator exposes
		    training-set predictions, and updates aggregate inlier and outlier counts for
		    downstream scoring and reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to fit the anomaly-detection estimator.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    IsolationForest | None: Fitted IsolationForest class/object instance.
		
		Raises:
		    Error: Raised when validation or estimator fitting fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( self.prediction == -1 ) )
			self.inliers = float( np.sum( self.prediction == 1 ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'train( self, *args ) -> IsolationForest | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""Predict with the IsolationForest detector.
		
		Purpose:
		    Validates the supplied feature matrix and generates inlier/outlier labels with the
		    fitted IsolationForest estimator. The prediction vector is cached on the class/object for
		    later scoring, analysis, and inspection.
		
		Args:
		    X (np.ndarray): Feature matrix used to generate inlier and outlier predictions.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    np.ndarray | None: Prediction vector where `1` denotes inlier samples and `-1`
		                       denotes outlier samples.
		
		Raises:
		    Error: Raised when validation or estimator prediction fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'project( self, *args ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""Create per-sample IsolationForest scores.
		
		Purpose:
		    Validates the supplied feature matrix, generates or refreshes inlier/outlier
		    predictions, computes estimator-specific anomaly scores, updates aggregate inlier
		    and outlier counts, and returns a dataframe containing prediction, anomaly, inlier,
		    and outlier columns for sample-level review.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute predictions and anomaly scores.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing per-sample predictions, anomaly scores,
		                         inlier flags, and outlier flags.
		
		Raises:
		    Error: Raised when validation, prediction, score calculation, or dataframe
		           construction fails."""
		try:
			throw_if( 'X', X )
			y_pred = self.project( X, y )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( y_pred == -1 ) )
			self.inliers = float( np.sum( y_pred == 1 ) )
			df_scores = pd.DataFrame( )
			df_scores[ 'Prediction' ] = y_pred
			df_scores[ 'Anomaly' ] = self.anomaly_scores
			df_scores[ 'Inlier' ] = (y_pred == 1).astype( int )
			df_scores[ 'Outlier' ] = (y_pred == -1).astype( int )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""Summarize IsolationForest anomaly results.
		
		Purpose:
		    Validates the supplied feature matrix, builds per-sample score output, creates
		    aggregate inlier, outlier, contamination, and quality metrics where supported,
		    renders an inlier-versus-outlier bar chart, and returns the summary dataframe for
		    reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute anomaly-analysis output.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    pd.DataFrame | None: Summary dataframe containing aggregate anomaly-detection
		                         metrics.
		
		Raises:
		    Error: Raised when validation, scoring, summary construction, or chart rendering
		           fails."""
		try:
			throw_if( 'X', X )
			df_scores = self.score( X, y )
			df_summary = pd.DataFrame( {
						'Metric': [ 'Inliers', 'Outliers', 'Contamination', 'Quality' ],
						'Value': [ float( self.inliers ), float( self.outliers ),
						           float( self.model.contamination ),
						           float( round( self.inliers / len( df_scores ), 4 ) ) ]
				} )
			
			df_plot = pd.DataFrame( {
						'Label': [ 'Inliers', 'Outliers' ],
						'Count': [ float( self.inliers ), float( self.outliers ) ]
				} )
			
			plt.figure( figsize=(8, 6) )
			sns.barplot( data=df_plot, x='Label', y='Count' )
			plt.title( 'IsolationForest Detection Summary' )
			plt.xlabel( 'Classification' )
			plt.ylabel( 'Count' )
			plt.tight_layout( )
			plt.show( )
			
			return df_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IsolationForest'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception

class OneClass( Outlier ):
	"""One-Class Support Vector Machine.
	
	Purpose:
	    Detects novelty and outliers by learning the support of the normal data distribution
	    with a one-class support-vector machine. The class/object stores kernel configuration,
	    the fitted estimator, prediction labels, anomaly scores, and aggregate
	    inlier/outlier counts for consistent reporting.
	
	Attributes:
	    model (Optional[sv.OneClassSVM]): Underlying One-Class SVM estimator.
	    data (Optional[np.ndarray]): Optional fitted-data cache retained for interface
	                                 compatibility.
	    prediction (Optional[np.ndarray]): Most recent inlier/outlier prediction vector.
	    anomaly_scores (Optional[np.ndarray]): Most recent One-Class SVM decision scores.
	    kernel (Optional[str]): Kernel used by the underlying One-Class SVM estimator."""
	model: Optional[ sv.OneClassSVM ]
	data: Optional[ np.ndarray ]
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	kernel: Optional[ str ]
	
	def __init__( self, kernel: str = 'rbf', nu: float = 0.05, gamma: str = 'scale' ) -> None:
		"""Initialize OneClassSVM.
		
		Purpose:
		    Configures the One-Class SVM estimator with kernel, nu, and gamma settings
		    and initializes prediction and anomaly-score caches for novelty and
		    outlier-detection workflows.
		
		Args:
		    kernel (str): Kernel type used by the underlying One-Class SVM estimator.
		    nu (float): Upper bound on the fraction of training errors and lower bound on
		                support vectors.
		    gamma (str): Kernel coefficient used by supported kernel functions.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.kernel = kernel
		self.model = sv.OneClassSVM( kernel=kernel, nu=nu, gamma=gamma )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> OneClass | None:
		"""Fit the One-Class SVM detector.
		
		Purpose:
		    Validates the supplied feature matrix, fits the underlying One-Class SVM estimator,
		    refreshes prediction and anomaly-score state when the estimator exposes training-set
		    predictions, and updates aggregate inlier and outlier counts for downstream scoring
		    and reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to fit the anomaly-detection estimator.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    OneClass | None: Fitted OneClass class/object instance.
		
		Raises:
		    Error: Raised when validation or estimator fitting fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( self.prediction == -1 ) )
			self.inliers = float( np.sum( self.prediction == 1 ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneClass'
			exception.method = 'train( self, *args ) -> OneClass | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""Predict with the One-Class SVM detector.
		
		Purpose:
		    Validates the supplied feature matrix and generates inlier/outlier labels with the
		    fitted One-Class SVM estimator. The prediction vector is cached on the class/object for
		    later scoring, analysis, and inspection.
		
		Args:
		    X (np.ndarray): Feature matrix used to generate inlier and outlier predictions.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    np.ndarray | None: Prediction vector where `1` denotes inlier samples and `-1`
		                       denotes outlier samples.
		
		Raises:
		    Error: Raised when validation or estimator prediction fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneClass'
			exception.method = 'project( self, *args ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""Create per-sample One-Class SVM scores.
		
		Purpose:
		    Validates the supplied feature matrix, generates or refreshes inlier/outlier
		    predictions, computes estimator-specific anomaly scores, updates aggregate inlier
		    and outlier counts, and returns a dataframe containing prediction, anomaly, inlier,
		    and outlier columns for sample-level review.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute predictions and anomaly scores.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing per-sample predictions, anomaly scores,
		                         inlier flags, and outlier flags.
		
		Raises:
		    Error: Raised when validation, prediction, score calculation, or dataframe
		           construction fails."""
		try:
			throw_if( 'X', X )
			y_pred = self.project( X, y )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( y_pred == -1 ) )
			self.inliers = float( np.sum( y_pred == 1 ) )
			df_scores = pd.DataFrame( )
			df_scores[ 'Prediction' ] = y_pred
			df_scores[ 'Anomaly' ] = self.anomaly_scores
			df_scores[ 'Inlier' ] = (y_pred == 1).astype( int )
			df_scores[ 'Outlier' ] = (y_pred == -1).astype( int )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneClass'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""Summarize One-Class SVM anomaly results.
		
		Purpose:
		    Validates the supplied feature matrix, builds per-sample score output, creates
		    aggregate inlier, outlier, contamination, and quality metrics where supported,
		    renders an inlier-versus-outlier bar chart, and returns the summary dataframe for
		    reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute anomaly-analysis output.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    pd.DataFrame | None: Summary dataframe containing aggregate anomaly-detection
		                         metrics.
		
		Raises:
		    Error: Raised when validation, scoring, summary construction, or chart rendering
		           fails."""
		try:
			throw_if( 'X', X )
			df_scores = self.score( X, y )
			df_summary = pd.DataFrame( {
						'Metric': [ 'Inliers', 'Outliers', 'Quality' ],
						'Value': [ float( self.inliers ), float( self.outliers ),
						           float( round( self.inliers / len( df_scores ), 4 ) ) ]
				} )			
			df_plot = pd.DataFrame( {
						'Label': [ 'Inliers', 'Outliers' ],
						'Count': [ float( self.inliers ), float( self.outliers ) ]
				} )
			
			plt.figure( figsize=(8, 6) )
			sns.barplot( data=df_plot, x='Label', y='Count' )
			plt.title( 'OneClassSVM Detection Summary' )
			plt.xlabel( 'Classification' )
			plt.ylabel( 'Count' )
			plt.tight_layout( )
			plt.show( )
			
			return df_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OneClass'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception

class OutlierFactor( Outlier ):
	"""Wrap LocalOutlierFactor.
	
	Purpose:
	    Detects density-based anomalies by comparing each sample's local density with the
	    local density of neighboring samples. The class/object supports both novelty and
	    fit-predict behavior while storing predictions, anomaly scores, neighborhood
	    settings, contamination settings, and aggregate counts.
	
	Attributes:
	    model (nn.LocalOutlierFactor): Underlying Local Outlier Factor estimator.
	    prediction (Optional[np.ndarray]): Most recent inlier/outlier prediction vector.
	    anomaly_scores (Optional[np.ndarray]): Most recent local outlier scores or decision
	                                           scores.
	    neighbors (Optional[int]): Number of neighbors used for local density estimation.
	    contamination (Optional[float]): Expected proportion of outliers in the data.
	    novelty (Optional[bool]): Flag indicating whether novelty detection is enabled."""
	model: nn.LocalOutlierFactor
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	neighbors: Optional[ int ]
	contamination: Optional[ float ]
	novelty: Optional[ bool ]
	
	def __init__( self, n_neighbors: int=20, contamination: float=0.1, novelty: bool=True ) -> None:
		"""Initialize LocalOutlierFactor.
		
		Purpose:
		    Configures the Local Outlier Factor estimator with neighborhood,
		    contamination, and novelty-detection settings and initializes prediction and
		    anomaly-score caches.
		
		Args:
		    n_neighbors (int): Number of neighbors used for local density estimation.
		    contamination (float): Expected proportion of outliers in the input data.
		    novelty (bool): Flag indicating whether novelty detection is enabled for unseen
		                    samples.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.neighbors = n_neighbors
		self.contamination = contamination
		self.novelty = novelty
		self.model = nn.LocalOutlierFactor( n_neighbors=self.neighbors,
			contamination=self.contamination, novelty=self.novelty )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> OutlierFactor | None:
		"""Fit the Local Outlier Factor detector.
		
		Purpose:
		    Validates the supplied feature matrix, fits the underlying Local Outlier Factor
		    estimator, refreshes prediction and anomaly-score state when the estimator exposes
		    training-set predictions, and updates aggregate inlier and outlier counts for
		    downstream scoring and reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to fit the anomaly-detection estimator.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    OutlierFactor | None: Fitted OutlierFactor class/object instance.
		
		Raises:
		    Error: Raised when validation or estimator fitting fails."""
		try:
			throw_if( 'X', X )
			if self.novelty:
				self.model.fit( X )
				self.prediction = None
				self.anomaly_scores = None
			else:
				self.prediction = self.model.fit_predict( X )
				self.anomaly_scores = self.model.negative_outlier_factor_
				self.outliers = float( np.sum( self.prediction == -1 ) )
				self.inliers = float( np.sum( self.prediction == 1 ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'train( self, *args ) -> OutlierFactor | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""Predict with the Local Outlier Factor detector.
		
		Purpose:
		    Validates the supplied feature matrix and generates inlier/outlier labels with the
		    fitted Local Outlier Factor estimator. The prediction vector is cached on the
		    class/object for later scoring, analysis, and inspection.
		
		Args:
		    X (np.ndarray): Feature matrix used to generate inlier and outlier predictions.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    np.ndarray | None: Prediction vector where `1` denotes inlier samples and `-1`
		                       denotes outlier samples.
		
		Raises:
		    Error: Raised when validation or estimator prediction fails."""
		try:
			throw_if( 'X', X )
			if self.novelty:
				self.prediction = self.model.predict( X )
			else:
				self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'project( self, *args ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""Create per-sample Local Outlier Factor scores.
		
		Purpose:
		    Validates the supplied feature matrix, generates or refreshes inlier/outlier
		    predictions, computes estimator-specific anomaly scores, updates aggregate inlier
		    and outlier counts, and returns a dataframe containing prediction, anomaly, inlier,
		    and outlier columns for sample-level review.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute predictions and anomaly scores.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing per-sample predictions, anomaly scores,
		                         inlier flags, and outlier flags.
		
		Raises:
		    Error: Raised when validation, prediction, score calculation, or dataframe
		           construction fails."""
		try:
			throw_if( 'X', X )
			if self.novelty:
				y_pred = self.project( X, y )
				self.anomaly_scores = self.model.decision_function( X )
			else:
				y_pred = self.model.fit_predict( X )
				self.prediction = y_pred
				self.anomaly_scores = self.model.negative_outlier_factor_
			
			self.outliers = float( np.sum( y_pred == -1 ) )
			self.inliers = float( np.sum( y_pred == 1 ) )
			df_scores = pd.DataFrame( )
			df_scores[ 'Prediction' ] = y_pred
			df_scores[ 'Anomaly' ] = self.anomaly_scores
			df_scores[ 'Inlier' ] = (y_pred == 1).astype( int )
			df_scores[ 'Outlier' ] = (y_pred == -1).astype( int )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""Summarize Local Outlier Factor anomaly results.
		
		Purpose:
		    Validates the supplied feature matrix, builds per-sample score output, creates
		    aggregate inlier, outlier, contamination, and quality metrics where supported,
		    renders an inlier-versus-outlier bar chart, and returns the summary dataframe for
		    reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute anomaly-analysis output.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    pd.DataFrame | None: Summary dataframe containing aggregate anomaly-detection
		                         metrics.
		
		Raises:
		    Error: Raised when validation, scoring, summary construction, or chart rendering
		           fails."""
		try:
			throw_if( 'X', X )
			df_scores = self.score( X, y )
			df_summary = pd.DataFrame( 	{ 'Metric': [ 'Inliers', 'Outliers', 'Contamination', 'Quality' ],
					'Value': [ float( self.inliers ), float( self.outliers ),
						float( self.contamination ),
						float( round( self.inliers / len( df_scores ), 4 ) ) ] } )
			
			df_plot = pd.DataFrame( { 'Label': [ 'Inliers', 'Outliers' ],
				'Count': [ float( self.inliers ), float( self.outliers ) ] } )
			
			plt.figure( figsize=(8, 6) )
			sns.barplot( data=df_plot, x='Label', y='Count' )
			plt.title( 'LocalOutlierFactor Detection Summary' )
			plt.xlabel( 'Classification' )
			plt.ylabel( 'Count' )
			plt.tight_layout( )
			plt.show( )
			
			return df_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'OutlierFactor'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception

class EllipticSquare( Outlier ):
	"""Provides the EllipticEnvelope functionality.
	
	Purpose:
	    Detects multivariate outliers by fitting a robust covariance estimate and
	    identifying samples outside the learned elliptical data envelope. The class/object stores
	    the fitted estimator, contamination setting, prediction labels, anomaly scores, and
	    aggregate inlier/outlier counts.
	
	Attributes:
	    model (cv.EllipticEnvelope): Underlying robust covariance estimator.
	    prediction (Optional[np.ndarray]): Most recent inlier/outlier prediction vector.
	    anomaly_scores (Optional[np.ndarray]): Most recent EllipticEnvelope decision scores.
	    contamination (Optional[float]): Expected proportion of outliers in the input data."""
	model: cv.EllipticEnvelope
	prediction: Optional[ np.ndarray ]
	anomaly_scores: Optional[ np.ndarray ]
	contamination: Optional[ float ]
	
	def __init__( self, contamination: float=0.1 ) -> None:
		"""Initialize EllipticEnvelope.
		
		Purpose:
		    Configures the EllipticEnvelope estimator with the expected contamination
		    rate and initializes prediction and anomaly-score caches for robust covariance
		    outlier detection.
		
		Args:
		    contamination (float): Expected proportion of outliers in the input data.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.contamination = contamination
		self.model = cv.EllipticEnvelope( contamination=contamination )
		self.prediction = None
		self.anomaly_scores = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> EllipticSquare | None:
		"""Fit the EllipticEnvelope detector.
		
		Purpose:
		    Validates the supplied feature matrix, fits the underlying EllipticEnvelope
		    estimator, refreshes prediction and anomaly-score state when the estimator exposes
		    training-set predictions, and updates aggregate inlier and outlier counts for
		    downstream scoring and reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to fit the anomaly-detection estimator.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    EllipticSquare | None: Fitted EllipticSquare class/object instance.
		
		Raises:
		    Error: Raised when validation or estimator fitting fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.predict( X )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( self.prediction == -1 ) )
			self.inliers = float( np.sum( self.prediction == 1 ) )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'train( self, *args ) -> EllipticSquare | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray | None:
		"""Predict with the EllipticEnvelope detector.
		
		Purpose:
		    Validates the supplied feature matrix and generates inlier/outlier labels with the
		    fitted EllipticEnvelope estimator. The prediction vector is cached on the class/object
		    for later scoring, analysis, and inspection.
		
		Args:
		    X (np.ndarray): Feature matrix used to generate inlier and outlier predictions.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    np.ndarray | None: Prediction vector where `1` denotes inlier samples and `-1`
		                       denotes outlier samples.
		
		Raises:
		    Error: Raised when validation or estimator prediction fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'project( self, *args ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""Create per-sample EllipticEnvelope scores.
		
		Purpose:
		    Validates the supplied feature matrix, generates or refreshes inlier/outlier
		    predictions, computes estimator-specific anomaly scores, updates aggregate inlier
		    and outlier counts, and returns a dataframe containing prediction, anomaly, inlier,
		    and outlier columns for sample-level review.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute predictions and anomaly scores.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing per-sample predictions, anomaly scores,
		                         inlier flags, and outlier flags.
		
		Raises:
		    Error: Raised when validation, prediction, score calculation, or dataframe
		           construction fails."""
		try:
			throw_if( 'X', X )
			y_pred = self.project( X, y )
			self.anomaly_scores = self.model.decision_function( X )
			self.outliers = float( np.sum( y_pred == -1 ) )
			self.inliers = float( np.sum( y_pred == 1 ) )
			df_scores = pd.DataFrame( )
			df_scores[ 'Prediction' ] = y_pred
			df_scores[ 'Anomaly' ] = self.anomaly_scores
			df_scores[ 'Inlier' ] = (y_pred == 1).astype( int )
			df_scores[ 'Outlier' ] = (y_pred == -1).astype( int )
			return df_scores
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> pd.DataFrame | None:
		"""Summarize EllipticEnvelope anomaly results.
		
		Purpose:
		    Validates the supplied feature matrix, builds per-sample score output, creates
		    aggregate inlier, outlier, contamination, and quality metrics where supported,
		    renders an inlier-versus-outlier bar chart, and returns the summary dataframe for
		    reporting.
		
		Args:
		    X (np.ndarray): Feature matrix used to compute anomaly-analysis output.
		    y (Optional[np.ndarray]): Optional target vector accepted for interface consistency.
		
		Returns:
		    pd.DataFrame | None: Summary dataframe containing aggregate anomaly-detection
		                         metrics.
		
		Raises:
		    Error: Raised when validation, scoring, summary construction, or chart rendering
		           fails."""
		try:
			throw_if( 'X', X )
			df_scores = self.score( X, y )
			df_summary = pd.DataFrame( { 'Metric': [ 'Inliers', 'Outliers', 'Contamination', 'Quality' ],
					'Value': [ float( self.inliers ), float( self.outliers ),
						float( self.contamination ),
						float( round( self.inliers / len( df_scores ), 4 ) ) ] } )
			
			df_plot = pd.DataFrame( { 'Label': [ 'Inliers', 'Outliers' ],
				'Count': [ float( self.inliers ), float( self.outliers ) ] } )
			plt.figure( figsize=(8, 6) )
			sns.barplot( data=df_plot, x='Label', y='Count' )
			plt.title( 'EllipticEnvelope Detection Summary' )
			plt.xlabel( 'Classification' )
			plt.ylabel( 'Count' )
			plt.tight_layout( )
			plt.show( )
			
			return df_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'EllipticSquare'
			exception.method = 'analyze( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
