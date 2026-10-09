"""******************************************************************************************
  Assembly:                mathy
  Filename:                clusters.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="clusters.py" company="Terry D. Eppler">

     clusters.py
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
    Provides sklearn clustering wrappers for Mathy modeling workflows. The module centralizes
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift,
    AffinityPropagation, Birch, and OPTICS behind a consistent training, projection, scoring,
    and analysis interface with shared clustering metrics and fitted-model metadata access.
</summary>
******************************************************************************************
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from boogr import Error, Logger
import sklearn.cluster as skc
from sklearn.metrics import (silhouette_score, completeness_score, homogeneity_score,
                             mutual_info_score, v_measure_score)

def throw_if( name: str, value: object ) -> None:
	"""Validate a required argument.
	
	Purpose:
	    Validate that a required argument is not null or empty without evaluating NumPy
	    arrays as Boolean values.
	
	Args:
	    name (str): Argument name used in the validation message.
	    value (object): Argument value to validate.
	
	Returns:
	    None: This function performs its work through side effects and does not return a
	          value.
	
	Raises:
	    ValueError: Raised when the `throw_if` operation cannot complete."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, np.ndarray ) and value.size == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (str, list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Cluster( ):
	"""Defines the shared clustering wrapper contract for Mathy estimators.
	
	Purpose:
	    Defines the shared clustering object contract for Mathy estimators. The interface
	    standardizes model fitting, label projection, metric scoring, and analysis payload
	    generation across clustering algorithms.
	
	Attributes:
	    n_clusters (Optional[int]): Number of clusters requested or learned by the
	                                clustering object.
	    random_state (Optional[int]): Random seed or estimator random-state configuration.
	    max_iter (Optional[int]): Maximum iteration count used by iterative clustering
	                              estimators.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Optional probability-like prediction output
	                                        retained for interface compatibility.
	    completeness (Optional[float]): Most recent completeness score when reference labels
	                                    are supplied.
	    homogeneity (Optional[float]): Most recent homogeneity score when reference labels
	                                   are supplied.
	    mutual_info (Optional[float]): Most recent mutual-information score when reference
	                                   labels are supplied.
	    silouette (Optional[float]): Most recent silhouette score computed from features and
	                                 predicted labels.
	    v_measure (Optional[float]): Most recent V-measure score when reference labels are
	                                 supplied."""
	n_clusters: Optional[ int ]
	random_state: Optional[ int ]
	max_iter: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self ) -> None:
		"""Initializes a `Cluster` instance and its runtime state.
		
		Purpose:
		    Initializes a `Cluster` instance and its runtime state.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		pass
	
	def train( self, X: np.ndarray ) -> object | None:
		"""Train.
		
		Purpose:
		    Fits the underlying Cluster estimator to the supplied feature matrix, refreshes
		    fitted prediction metadata when available, and returns the current object for
		    chained clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    object | None: Fitted object instance.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the Cluster
		    estimator. The method preserves compatibility with wrappers that fit and predict in
		    a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the Cluster
		    output, including silhouette, completeness, homogeneity, mutual information, and
		    V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the Cluster clustering result by combining
		    fitted labels, estimator metadata, and metric output generated from the supplied
		    feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError

class KMeans( Cluster ):
	"""Represent the `KMeans` workflow.
	
	Purpose:
	    Wraps KMeans for centroid-based partitioning of numeric feature matrices.
	    The object stores cluster assignments, centroids, inertia, iteration metadata, and
	    feature counts while exposing a consistent training, projection, scoring, and
	    analysis interface.
	
	Attributes:
	    model (skc.KMeans): Underlying clustering estimator.
	    n_clusters (Optional[int]): Number of clusters requested by the object.
	    init (object): Centroid, parameter, or weight initialization strategy.
	    n_init (object): Number of independent initializations evaluated by the estimator.
	    tolerance (Optional[float]): Convergence tolerance used by the estimator.
	    random_state (Optional[int]): Random seed or estimator random-state configuration.
	    max_iter (Optional[int]): Maximum iteration count used by the clustering estimator.
	    verbose (Optional[int]): Estimator verbosity level.
	    copy_x (Optional[bool]): Whether K-Means copies input data before centering it.
	    algorithm (Optional[str]): Algorithm selected for estimator fitting or neighbor
	                               search.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Most recent class-probability estimates
	                                        generated by the model.
	    completeness (Optional[float]): Most recently calculated clustering completeness
	                                    score.
	    homogeneity (Optional[float]): Most recently calculated clustering homogeneity
	                                   score.
	    mutual_info (Optional[float]): Most recently calculated mutual-information score.
	    silouette (Optional[float]): Most recently calculated silhouette score.
	    v_measure (Optional[float]): Most recently calculated V-measure score."""
	model: skc.KMeans
	n_clusters: Optional[ int ]
	init: object
	n_init: object
	tolerance: Optional[ float ]
	random_state: Optional[ int ]
	max_iter: Optional[ int ]
	verbose: Optional[ int ]
	copy_x: Optional[ bool ]
	algorithm: Optional[ str ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters: int = 8, init: object = 'k-means++',
			n_init: object = 'auto', tol: float = 0.0001, rando: int | None = 42,
			max_iter: int = 300, verbose: int = 0, copy_x: bool = True,
			algorithm: str = 'lloyd', n_clusters: int | None = None,
			random_state: int | None = None ) -> None:
		"""Initialize clustering object.
		
		Purpose:
		    Initializes the KMeans clustering object with estimator configuration, runtime
		    metadata fields, prediction caches, and the underlying clustering model used
		    by training and projection methods.
		
		Args:
		    clusters (int): Requested number of clusters or initial cluster-count configuration.
		    init (object): Centroid initialization strategy passed to KMeans.
		    n_init (object): Number of centroid initializations evaluated by KMeans.
		    tol (float): Convergence tolerance passed to the clustering estimator.
		    rando (int | None): Random seed used when constructing the estimator.
		    max_iter (int): Maximum number of estimator iterations.
		    verbose (int): Estimator verbosity flag.
		    copy_x (bool): Flag controlling whether KMeans copies the input matrix before
		                   centering.
		    algorithm (str): Estimator algorithm selection passed to sklearn.
		    n_clusters (int | None): Optional override for the requested number of clusters.
		    random_state (int | None): Random seed or random-state value.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_clusters = n_clusters if n_clusters is not None else clusters
		self.init = init
		self.n_init = n_init
		self.random_state = random_state if random_state is not None else rando
		self.max_iter = max_iter
		self.tolerance = tol
		self.verbose = verbose
		self.copy_x = copy_x
		self.algorithm = algorithm
		self.model = skc.KMeans( n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
			max_iter=self.max_iter, tol=self.tolerance, verbose=self.verbose,
			random_state=self.random_state, copy_x=self.copy_x, algorithm=self.algorithm )
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public member names exposed by the KMeans object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the object."""
		return [ 'model', 'train', 'score', 'project', 'transform', 'analyze', 'n_clusters',
			'init', 'n_init', 'random_state', 'tolerance', 'max_iter', 'verbose', 'copy_x',
			'algorithm', 'clusters', 'centroids_', 'labels', 'inertia', 'silouette', 'homogeneity',
			'mutual_info', 'v_measure', 'iterations', 'features', 'prediction' ]
	
	@property
	def clusters( self ) -> np.ndarray:
		"""Return clusters metadata.
		
		Purpose:
		    Returns fitted cluster identifiers or exemplar indices exposed by the clustering
		    estimator.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the KMeans estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `clusters` operation cannot complete."""
		if not hasattr( self.model, 'cluster_centers_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.cluster_centers_
	
	@property
	def centroids_( self ) -> np.ndarray:
		"""Return centroids metadata.
		
		Purpose:
		    Returns fitted cluster centers or exemplar center coordinates exposed by the
		    estimator.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the KMeans estimator."""
		return self.clusters
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.
		
		Purpose:
		    Returns fitted cluster-label assignments.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the KMeans estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `labels` operation cannot complete."""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def inertia( self ) -> float:
		"""Return inertia metadata.
		
		Purpose:
		    Returns KMeans inertia from the fitted estimator.
		
		Returns:
		    float: Fitted metadata exposed by the KMeans estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `inertia` operation cannot complete."""
		if not hasattr( self.model, 'inertia_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.inertia_
	
	@property
	def iterations( self ) -> int:
		"""Return iterations metadata.
		
		Purpose:
		    Returns the number of iterations completed by the fitted estimator.
		
		Returns:
		    int: Fitted metadata exposed by the KMeans estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `iterations` operation cannot complete."""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.
		
		Purpose:
		    Returns the number of input features observed during fitting.
		
		Returns:
		    int: Fitted metadata exposed by the KMeans estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `features` operation cannot complete."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> KMeans | None:
		"""Train.
		
		Purpose:
		    Fits the underlying KMeans estimator to the supplied feature matrix, refreshes
		    fitted prediction metadata when available, and returns the current object for
		    chained clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    KMeans | None: Fitted object instance.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'train( self, X: np.ndarray ) -> KMeans | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the KMeans
		    estimator. The method preserves compatibility with wrappers that fit and predict in
		    a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			if hasattr( self.model, 'cluster_centers_' ):
				self.prediction = self.model.predict( X )
			else:
				self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""Transform.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted KMeans estimator and returns
		    estimator-specific distances, embeddings, or transformed cluster-space values.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Transformed cluster-space representation produced by the
		                       estimator.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			if not hasattr( self.model, 'cluster_centers_' ):
				self.model.fit( X )
				self.prediction = self.model.labels_
			
			return self.model.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the KMeans
		    output, including silhouette, completeness, homogeneity, mutual information, and
		    V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			if hasattr( self.model, 'cluster_centers_' ):
				labels = self.model.predict( X )
			else:
				labels = self.model.fit_predict( X )
			
			self.prediction = labels
			scores = { }
			unique_labels = np.unique( labels )
			if len( unique_labels ) >= 2 and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			scores[ 'Inertia' ] = self.inertia
			scores[ 'Iterations' ] = self.iterations
			scores[ 'Clusters' ] = len( unique_labels )
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the KMeans clustering result by combining
		    fitted labels, estimator metadata, and metric output generated from the supplied
		    feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			df_score = self.score( X, y )
			return { 'labels': self.prediction, 'centroids': self.centroids_,
				'inertia': self.inertia, 'iterations': self.iterations, 'features': self.features,
				'score': df_score }
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'KMeans'
			exception.method = 'analyze( self, *args ) -> Dict[str, Any] | None'
			Logger( ).write( exception )
			raise exception

class DBSCAN( Cluster ):
	"""Wraps DBSCAN for density-based clustering with noise detection.
	
	Purpose:
	    Wraps DBSCAN for density-based clustering with noise detection. The object
	    stores fitted labels, core-sample indices, core components, and feature metadata
	    while exposing the shared clustering workflow interface.
	
	Attributes:
	    model (skc.DBSCAN): Underlying clustering estimator.
	    epsilon (Optional[float]): Estimator-specific tolerance or insensitive-loss margin.
	    min_samples (Optional[int]): Minimum neighborhood sample count required to form a
	                                 dense region.
	    metric (object): Distance or scoring metric used by the estimator.
	    metric_params (Optional[Dict[str, Any]]): Additional keyword values supplied to the
	                                              selected metric.
	    algorithm (Optional[str]): Algorithm selected for estimator fitting or neighbor
	                               search.
	    leaf_size (Optional[int]): Leaf size used by tree-based neighbor-search structures.
	    p (Optional[float]): Power parameter used by the Minkowski distance metric.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Most recent class-probability estimates
	                                        generated by the model.
	    completeness (Optional[float]): Most recently calculated clustering completeness
	                                    score.
	    homogeneity (Optional[float]): Most recently calculated clustering homogeneity
	                                   score.
	    mutual_info (Optional[float]): Most recently calculated mutual-information score.
	    silouette (Optional[float]): Most recently calculated silhouette score.
	    v_measure (Optional[float]): Most recently calculated V-measure score."""
	model: skc.DBSCAN
	epsilon: Optional[ float ]
	min_samples: Optional[ int ]
	metric: object
	metric_params: Optional[ Dict[ str, Any ] ]
	algorithm: Optional[ str ]
	leaf_size: Optional[ int ]
	p: Optional[ float ]
	n_jobs: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, eps: float = 0.5, samples: int = 5,
			metric: object = 'euclidean',
			metric_params: Dict[ str, Any ] | None = None,
			algorithm: str = 'auto', leaf_size: int = 30,
			p: float | None = None, n_jobs: int | None = None,
			min_samples: int | None = None ) -> None:
		"""Initialize clustering object.
		
		Purpose:
		    Initializes the DBSCAN clustering object with estimator configuration, runtime
		    metadata fields, prediction caches, and the underlying clustering model used
		    by training and projection methods.
		
		Args:
		    eps (float): Maximum neighborhood radius for density-based clustering.
		    samples (int): Minimum sample count used for density-neighborhood definitions.
		    metric (object): Distance metric used by the clustering estimator.
		    metric_params (Dict[str, Any] | None): Additional metric keyword arguments passed to
		                                           sklearn.
		    algorithm (str): Estimator algorithm selection passed to sklearn.
		    leaf_size (int): Leaf size used by tree-based neighbor-search algorithms.
		    p (float | None): Power parameter used by Minkowski distance metrics.
		    n_jobs (int | None): Number of parallel jobs used by when supported.
		    min_samples (int | None): Optional override for the minimum sample count.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.epsilon = eps
		self.min_samples = min_samples if min_samples is not None else samples
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.n_jobs = n_jobs
		self.model = skc.DBSCAN( eps=self.epsilon, min_samples=self.min_samples,
			metric=self.metric, metric_params=self.metric_params, algorithm=self.algorithm,
			leaf_size=self.leaf_size, p=self.p, n_jobs=self.n_jobs )
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public member names exposed by the DBSCAN object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the object."""
		return [ 'model', 'train', 'score', 'project', 'analyze', 'epsilon', 'eps', 'min_samples',
			'metric', 'metric_params', 'algorithm', 'leaf_size', 'p', 'n_jobs', 'labels',
			'core_samples', 'components', 'features', 'prediction', 'silouette', 'homogeneity',
			'mutual_info', 'v_measure', 'completeness' ]
	
	@property
	def eps( self ) -> float:
		"""Return eps metadata.
		
		Purpose:
		    Returns the fitted DBSCAN neighborhood radius.
		
		Returns:
		    float: Fitted metadata exposed by the DBSCAN estimator."""
		return self.epsilon
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.
		
		Purpose:
		    Returns fitted cluster-label assignments.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the DBSCAN estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `labels` operation cannot complete."""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def core_samples( self ) -> np.ndarray:
		"""Return core samples metadata.
		
		Purpose:
		    Returns DBSCAN core-sample indices.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the DBSCAN estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `core_samples` operation cannot complete."""
		if not hasattr( self.model, 'core_sample_indices_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.core_sample_indices_
	
	@property
	def components( self ) -> np.ndarray:
		"""Return components metadata.
		
		Purpose:
		    Returns DBSCAN core-sample feature vectors.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the DBSCAN estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `components` operation cannot complete."""
		if not hasattr( self.model, 'components_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.components_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.
		
		Purpose:
		    Returns the number of input features observed during fitting.
		
		Returns:
		    int: Fitted metadata exposed by the DBSCAN estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `features` operation cannot complete."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> DBSCAN | None:
		"""Train.
		
		Purpose:
		    Fits the underlying DBSCAN estimator to the supplied feature matrix, refreshes
		    fitted prediction metadata when available, and returns the current object for
		    chained clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    DBSCAN | None: Fitted object instance.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'train( self, X: np.ndarray ) -> DBSCAN | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the DBSCAN
		    estimator. The method preserves compatibility with wrappers that fit and predict in
		    a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the DBSCAN
		    output, including silhouette, completeness, homogeneity, mutual information, and
		    V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			unique_labels = np.unique( labels )
			cluster_labels = unique_labels[ unique_labels != -1 ]
			scores[ 'Clusters' ] = len( cluster_labels )
			scores[ 'Noise' ] = int( np.sum( labels == -1 ) )
			if len( cluster_labels ) >= 2:
				mask = labels != -1
				if np.sum( mask ) >= 2 and len( np.unique( labels[ mask ] ) ) >= 2:
					self.silouette = silhouette_score( X[ mask ], labels[ mask ] )
					scores[ 'Silouette' ] = self.silouette
				else:
					self.silouette = np.nan
					scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the DBSCAN clustering result by combining
		    fitted labels, estimator metadata, and metric output generated from the supplied
		    feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			df_score = self.score( X, y )
			core_samples = self.core_samples if hasattr(
				self.model, 'core_sample_indices_' ) else np.array( [ ] )
			components = self.components if hasattr( self.model,
				'components_' ) else np.empty( (0, X.shape[ 1 ]) )
			
			return { 'labels': self.prediction, 'core_samples': core_samples,
				'components': components, 'features': self.features, 'score': df_score }
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'DBSCAN'
			exception.method = 'analyze( self, *args ) -> Dict[str, Any] | None'
			Logger( ).write( exception )
			raise exception

class Agglomerative( Cluster ):
	"""Wraps AgglomerativeClustering for hierarchical bottom-up clustering.
	
	Purpose:
	    Wraps AgglomerativeClustering for hierarchical bottom-up clustering. The
	    object preserves linkage configuration, tree construction options, fitted labels,
	    linkage children, distances, leaf counts, component counts, and feature metadata.
	
	Attributes:
	    model (skc.AgglomerativeClustering): Underlying clustering estimator.
	    n_clusters (Optional[int]): Number of clusters requested by the object.
	    metric (object): Distance or scoring metric used by the estimator.
	    affinity (object): Affinity or similarity measure used to construct the clustering
	                       graph.
	    memory (object): Cache configuration used for reusable estimator computations.
	    connectivity (object): Connectivity constraints used during hierarchical clustering.
	    compute_full_tree (object): Policy controlling construction of the full hierarchical
	                                tree.
	    linkage (Optional[str]): Linkage rule used to measure distances between clusters.
	    distance_threshold (Optional[float]): Linkage-distance cutoff used to merge
	                                          hierarchical clusters.
	    compute_distances (Optional[bool]): Whether hierarchical-clustering distances are
	                                        retained.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Most recent class-probability estimates
	                                        generated by the model.
	    completeness (Optional[float]): Most recently calculated clustering completeness
	                                    score.
	    homogeneity (Optional[float]): Most recently calculated clustering homogeneity
	                                   score.
	    mutual_info (Optional[float]): Most recently calculated mutual-information score.
	    silouette (Optional[float]): Most recently calculated silhouette score.
	    v_measure (Optional[float]): Most recently calculated V-measure score."""
	model: skc.AgglomerativeClustering
	n_clusters: Optional[ int ]
	metric: object
	affinity: object
	memory: object
	connectivity: object
	compute_full_tree: object
	linkage: Optional[ str ]
	distance_threshold: Optional[ float ]
	compute_distances: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters: int = 2, affinity: object = 'euclidean',
			linkage: str = 'ward', metric: object | None = None,
			memory: object = None, connectivity: object = None,
			compute_full_tree: object = 'auto',
			distance_threshold: float | None = None,
			compute_distances: bool = False,
			n_clusters: int | None = None ) -> None:
		"""Initialize clustering object.
		
		Purpose:
		    Initializes the Agglomerative clustering object with estimator configuration,
		    runtime metadata fields, prediction caches, and the underlying clustering
		    model used by training and projection methods.
		
		Args:
		    clusters (int): Requested number of clusters or initial cluster-count configuration.
		    affinity (object): Affinity configuration or similarity measure used by the
		                       estimator.
		    linkage (str): Linkage criterion used by agglomerative clustering.
		    metric (object | None): Distance metric used by the clustering estimator.
		    memory (object): Optional joblib memory configuration used by sklearn.
		    connectivity (object): Connectivity matrix or callable used by agglomerative
		                           clustering.
		    compute_full_tree (object): Tree-construction policy used by agglomerative
		                                clustering.
		    distance_threshold (float | None): Distance threshold used to stop hierarchical
		                                       clustering.
		    compute_distances (bool): Flag indicating whether agglomerative distances are
		                              computed and stored.
		    n_clusters (int | None): Optional override for the requested number of clusters.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_clusters = n_clusters if n_clusters is not None else clusters
		self.metric = metric if metric is not None else affinity
		self.affinity = self.metric
		self.memory = memory
		self.connectivity = connectivity
		self.compute_full_tree = compute_full_tree
		self.linkage = linkage
		self.distance_threshold = distance_threshold
		self.compute_distances = compute_distances
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
		self.model = self.create_model( )
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public member names exposed by the Agglomerative object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the object."""
		return [ 'model', 'train', 'score', 'project', 'analyze', 'create_model', 'n_clusters',
			'metric', 'affinity', 'memory', 'connectivity', 'compute_full_tree', 'linkage',
			'distance_threshold', 'compute_distances', 'labels', 'children', 'distances', 'leaves',
			'connected_components', 'features', 'prediction', 'silouette', 'homogeneity',
			'mutual_info', 'v_measure', 'completeness' ]
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.
		
		Purpose:
		    Returns fitted cluster-label assignments.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the Agglomerative estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `labels` operation cannot complete."""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def children( self ) -> np.ndarray:
		"""Return children metadata.
		
		Purpose:
		    Returns agglomerative merge children.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the Agglomerative estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `children` operation cannot complete."""
		if not hasattr( self.model, 'children_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.children_
	
	@property
	def distances( self ) -> np.ndarray:
		"""Return distances metadata.
		
		Purpose:
		    Returns agglomerative merge distances.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the Agglomerative estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `distances` operation cannot complete."""
		if not hasattr( self.model, 'distances_' ):
			raise AttributeError( 'The model distances are not available!' )
		return self.model.distances_
	
	@property
	def leaves( self ) -> int:
		"""Return leaves metadata.
		
		Purpose:
		    Returns the number of leaves in the hierarchical clustering tree.
		
		Returns:
		    int: Fitted metadata exposed by the Agglomerative estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `leaves` operation cannot complete."""
		if not hasattr( self.model, 'n_leaves_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_leaves_
	
	@property
	def connected_components( self ) -> int:
		"""Return connected components metadata.
		
		Purpose:
		    Returns the number of connected components in the fitted graph.
		
		Returns:
		    int: Fitted metadata exposed by the Agglomerative estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `connected_components` operation cannot complete."""
		if not hasattr( self.model, 'n_connected_components_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_connected_components_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.
		
		Purpose:
		    Returns the number of input features observed during fitting.
		
		Returns:
		    int: Fitted metadata exposed by the Agglomerative estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `features` operation cannot complete."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def create_model( self ) -> skc.AgglomerativeClustering:
		"""Create Model.
		
		Purpose:
		    Creates the configured AgglomerativeClustering estimator from the object's
		    normalized constructor options.
		
		Returns:
		    skc.AgglomerativeClustering: Configured agglomerative clustering estimator.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails.
		    ValueError: Raised when the `create_model` operation cannot complete."""
		try:
			n_clusters = self.n_clusters
			compute_full_tree = self.compute_full_tree
			
			if self.linkage == 'ward' and self.metric != 'euclidean':
				raise ValueError( 'When linkage is "ward", metric must be "euclidean".' )
			
			if self.distance_threshold is not None:
				n_clusters = None
				compute_full_tree = True
			
			return skc.AgglomerativeClustering( n_clusters=n_clusters, metric=self.metric,
				memory=self.memory, connectivity=self.connectivity,
				compute_full_tree=compute_full_tree, linkage=self.linkage,
				distance_threshold=self.distance_threshold,
				compute_distances=self.compute_distances )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'create_model( self ) -> skc.AgglomerativeClustering'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: np.ndarray ) -> Agglomerative | None:
		"""Train.
		
		Purpose:
		    Fits the underlying Agglomerative estimator to the supplied feature matrix,
		    refreshes fitted prediction metadata when available, and returns the current object
		    for chained clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    Agglomerative | None: Fitted object instance.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model = self.create_model( )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'train( self, X: np.ndarray ) -> Agglomerative | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the
		    Agglomerative estimator. The method preserves compatibility with wrappers that fit
		    and predict in a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model = self.create_model( )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the
		    Agglomerative output, including silhouette, completeness, homogeneity, mutual
		    information, and V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model = self.create_model( )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			scores[ 'Leaves' ] = self.leaves
			scores[ 'Connected-Components' ] = self.connected_components
			if self.metric != 'precomputed' and len( unique_labels ) >= 2 \
					and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			if hasattr( self.model, 'distances_' ):
				scores[ 'Distances-Computed' ] = True
			else:
				scores[ 'Distances-Computed' ] = False
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the Agglomerative clustering result by
		    combining fitted labels, estimator metadata, and metric output generated from the
		    supplied feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			df_score = self.score( X, y )
			distances = self.distances if hasattr(
				self.model, 'distances_' ) else np.array( [ ] )
			
			return { 'labels': self.prediction, 'children': self.children, 'distances': distances,
				'leaves': self.leaves, 'connected_components': self.connected_components,
				'features': self.features, 'score': df_score }
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Agglomerative'
			exception.method = 'analyze( self, *args ) -> Dict[str, Any] | None'
			Logger( ).write( exception )
			raise exception

class Spectral( Cluster ):
	"""Represent the `Spectral` workflow.
	
	Purpose:
	    Wraps SpectralClustering for graph-based clustering of nonlinear structures.
	    The object stores affinity configuration, eigen-solver options, label assignments,
	    and feature metadata while exposing the shared clustering workflow interface.
	
	Attributes:
	    model (skc.SpectralClustering): Underlying clustering estimator.
	    n_clusters (Optional[int]): Number of clusters requested by the object.
	    eigen_solver (Optional[str]): Eigensolver used to calculate the spectral embedding.
	    n_components (Optional[int]): Number of components retained by the transformation or
	                                  model.
	    random_state (Optional[int]): Random seed or estimator random-state configuration.
	    n_init (Optional[int]): Number of independent initializations evaluated by the
	                            estimator.
	    gamma (Optional[float]): Kernel coefficient or minimum loss reduction used by the
	                             estimator.
	    affinity (object): Affinity or similarity measure used to construct the clustering
	                       graph.
	    n_neighbors (Optional[int]): Number of neighboring samples used by the estimator.
	    eigen_tolerance (object): Convergence tolerance supplied to the eigensolver.
	    assign_labels (Optional[str]): Strategy used to assign labels from the spectral
	                                   embedding.
	    degree (Optional[float]): Degree used by the polynomial kernel or feature expansion.
	    coef0 (Optional[float]): Independent term used by polynomial and sigmoid kernels.
	    kernel_params (Optional[Dict[str, Any]]): Additional parameters supplied to the
	                                              selected kernel.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    verbose (Optional[bool]): Estimator verbosity level.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Most recent class-probability estimates
	                                        generated by the model.
	    completeness (Optional[float]): Most recently calculated clustering completeness
	                                    score.
	    homogeneity (Optional[float]): Most recently calculated clustering homogeneity
	                                   score.
	    mutual_info (Optional[float]): Most recently calculated mutual-information score.
	    silouette (Optional[float]): Most recently calculated silhouette score.
	    v_measure (Optional[float]): Most recently calculated V-measure score."""
	model: skc.SpectralClustering
	n_clusters: Optional[ int ]
	eigen_solver: Optional[ str ]
	n_components: Optional[ int ]
	random_state: Optional[ int ]
	n_init: Optional[ int ]
	gamma: Optional[ float ]
	affinity: object
	n_neighbors: Optional[ int ]
	eigen_tolerance: object
	assign_labels: Optional[ str ]
	degree: Optional[ float ]
	coef0: Optional[ float ]
	kernel_params: Optional[ Dict[ str, Any ] ]
	n_jobs: Optional[ int ]
	verbose: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, clusters: int = 8, random_state: int | None = 42,
			n_init: int = 10, gama: float = 1.0, distance: object = 'rbf',
			neighbors: int = 10, tolerance: object = 'auto',
			assign: str = 'kmeans', eigen_solver: str | None = None,
			n_components: int | None = None, degree: float = 3,
			coef0: float = 1, kernel_params: Dict[ str, Any ] | None = None,
			n_jobs: int | None = None, verbose: bool = False,
			n_clusters: int | None = None, gamma: float | None = None,
			affinity: object | None = None, n_neighbors: int | None = None,
			eigen_tol: object | None = None,
			assign_labels: str | None = None ) -> None:
		"""Initialize clustering object.
		
		Purpose:
		    Initializes the Spectral clustering object with estimator configuration, runtime
		    metadata fields, prediction caches, and the underlying clustering model used
		    by training and projection methods.
		
		Args:
		    clusters (int): Requested number of clusters or initial cluster-count configuration.
		    random_state (int | None): Random seed or random-state value.
		    n_init (int): Number of centroid initializations evaluated by KMeans.
		    gama (float): Legacy gamma parameter value used when constructing spectral affinity.
		    distance (object): Legacy affinity or distance configuration used by spectral
		                       clustering.
		    neighbors (int): Legacy nearest-neighbor count for spectral affinity construction.
		    tolerance (object): Legacy eigenvalue-convergence tolerance configuration.
		    assign (str): Legacy label-assignment strategy used by spectral clustering.
		    eigen_solver (str | None): Eigenvalue solver used by spectral clustering.
		    n_components (int | None): Number of eigenvectors used for spectral embedding.
		    degree (float): Polynomial-kernel degree used by spectral clustering.
		    coef0 (float): Polynomial or sigmoid kernel coefficient used by spectral clustering.
		    kernel_params (Dict[str, Any] | None): Additional kernel parameters passed to
		                                           spectral clustering.
		    n_jobs (int | None): Number of parallel jobs used by when supported.
		    verbose (bool): Estimator verbosity flag.
		    n_clusters (int | None): Optional override for the requested number of clusters.
		    gamma (float | None): Optional override for the spectral-clustering gamma parameter.
		    affinity (object | None): Affinity configuration or similarity measure used by the
		                              estimator.
		    n_neighbors (int | None): Optional override for nearest-neighbor affinity
		                              construction.
		    eigen_tol (object | None): Optional override for eigenvalue-convergence tolerance.
		    assign_labels (str | None): Optional override for spectral label-assignment
		                                strategy.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_clusters = n_clusters if n_clusters is not None else clusters
		self.eigen_solver = eigen_solver
		self.n_components = n_components
		self.random_state = random_state
		self.n_init = n_init
		self.gamma = gamma if gamma is not None else gama
		self.affinity = affinity if affinity is not None else distance
		self.n_neighbors = n_neighbors if n_neighbors is not None else neighbors
		self.eigen_tolerance = eigen_tol if eigen_tol is not None else tolerance
		self.assign_labels = assign_labels if assign_labels is not None else assign
		self.degree = degree
		self.coef0 = coef0
		self.kernel_params = kernel_params
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.model = skc.SpectralClustering( n_clusters=self.n_clusters,
			eigen_solver=self.eigen_solver, n_components=self.n_components,
			random_state=self.random_state, n_init=self.n_init, gamma=self.gamma,
			affinity=self.affinity, n_neighbors=self.n_neighbors, eigen_tol=self.eigen_tolerance,
			assign_labels=self.assign_labels, degree=self.degree, coef0=self.coef0,
			kernel_params=self.kernel_params, n_jobs=self.n_jobs, verbose=self.verbose )
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public member names exposed by the Spectral object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the object."""
		return [ 'model', 'n_clusters', 'eigen_solver', 'n_components', 'random_state', 'n_init',
			'gamma', 'affinity', 'n_neighbors', 'eigen_tolerance', 'assign_labels', 'degree',
			'coef0', 'kernel_params', 'n_jobs', 'verbose', 'labels', 'features', 'train', 'score',
			'project', 'analyze', 'silouette', 'homogeneity', 'mutual_info', 'v_measure',
			'completeness', 'prediction' ]
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.
		
		Purpose:
		    Returns fitted cluster-label assignments.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the Spectral estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `labels` operation cannot complete."""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.
		
		Purpose:
		    Returns the number of input features observed during fitting.
		
		Returns:
		    int: Fitted metadata exposed by the Spectral estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `features` operation cannot complete."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> Spectral | None:
		"""Train.
		
		Purpose:
		    Fits the underlying Spectral estimator to the supplied feature matrix, refreshes
		    fitted prediction metadata when available, and returns the current object for
		    chained clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    Spectral | None: Fitted object instance.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'train( self, X: np.ndarray ) -> Spectral | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the Spectral
		    estimator. The method preserves compatibility with wrappers that fit and predict in
		    a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the Spectral
		    output, including silhouette, completeness, homogeneity, mutual information, and
		    V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			if len( unique_labels ) >= 2 and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the Spectral clustering result by combining
		    fitted labels, estimator metadata, and metric output generated from the supplied
		    feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			return { 'Labels': labels, 'Clusters': self.n_clusters,
				'Eigen-Solver': self.eigen_solver, 'N-Components': self.n_components,
				'Random-State': self.random_state, 'N-Init': self.n_init, 'Gamma': self.gamma,
				'Affinity': self.affinity, 'N-Neighbors': self.n_neighbors,
				'Eigen-Tolerance': self.eigen_tolerance, 'Assign-Labels': self.assign_labels,
				'Degree': self.degree, 'Coef0': self.coef0, 'N-Jobs': self.n_jobs,
				'Verbose': self.verbose, 'Features': self.features, 'Score': df_score }
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Spectral'
			exception.method = 'analyze( self, *args ) -> Dict[str, Any] | None'
			Logger( ).write( exception )
			raise exception

class MeanShift( Cluster ):
	"""Represent the `MeanShift` workflow.
	
	Purpose:
	    Wraps MeanShift for centroid discovery through mode seeking in feature
	    space. The object stores fitted labels, cluster centers, iteration counts,
	    prediction output, and feature metadata.
	
	Attributes:
	    model (skc.MeanShift): Underlying clustering estimator.
	    bandwidth (Optional[float]): Kernel bandwidth used by mean-shift clustering.
	    seeds (Optional[np.ndarray]): Initial candidate centroids supplied to mean-shift
	                                  clustering.
	    bin_seeding (Optional[bool]): Whether mean-shift seeds are initialized on a
	                                  discretized grid.
	    min_bin_freq (Optional[int]): Minimum seed-bin frequency used by mean-shift
	                                  clustering.
	    cluster_all (Optional[bool]): Whether every sample is assigned to a mean-shift
	                                  cluster.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    max_iter (Optional[int]): Maximum iteration count used by the clustering estimator.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Most recent class-probability estimates
	                                        generated by the model.
	    completeness (Optional[float]): Most recently calculated clustering completeness
	                                    score.
	    homogeneity (Optional[float]): Most recently calculated clustering homogeneity
	                                   score.
	    mutual_info (Optional[float]): Most recently calculated mutual-information score.
	    silouette (Optional[float]): Most recently calculated silhouette score.
	    v_measure (Optional[float]): Most recently calculated V-measure score."""
	model: skc.MeanShift
	bandwidth: Optional[ float ]
	seeds: Optional[ np.ndarray ]
	bin_seeding: Optional[ bool ]
	min_bin_freq: Optional[ int ]
	cluster_all: Optional[ bool ]
	n_jobs: Optional[ int ]
	max_iter: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, min_bin: int = 1, group_all: bool = True,
			bandwidth: float | None = None, seeds: np.ndarray | None = None,
			bin_seeding: bool = False, n_jobs: int | None = None,
			max_iter: int = 300, min_bin_freq: int | None = None,
			cluster_all: bool | None = None ) -> None:
		"""Initialize clustering object.
		
		Purpose:
		    Initializes the MeanShift clustering object with estimator configuration, runtime
		    metadata fields, prediction caches, and the underlying clustering model used
		    by training and projection methods.
		
		Args:
		    min_bin (int): Legacy minimum bin frequency used by mean-shift bin seeding.
		    group_all (bool): Legacy flag controlling whether all samples are assigned to
		                      clusters.
		    bandwidth (float | None): Kernel bandwidth used by MeanShift.
		    seeds (np.ndarray | None): Initial seed locations used by MeanShift.
		    bin_seeding (bool): Flag indicating whether bin seeding is used by MeanShift.
		    n_jobs (int | None): Number of parallel jobs used by when supported.
		    max_iter (int): Maximum number of estimator iterations.
		    min_bin_freq (int | None): Optional override for minimum bin frequency.
		    cluster_all (bool | None): Optional override for assigning all samples in MeanShift.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.bandwidth = bandwidth
		self.seeds = seeds
		self.bin_seeding = bin_seeding
		self.min_bin_freq = min_bin_freq if min_bin_freq is not None else min_bin
		self.cluster_all = cluster_all if cluster_all is not None else group_all
		self.n_jobs = n_jobs
		self.max_iter = max_iter
		self.model = skc.MeanShift( bandwidth=self.bandwidth, seeds=self.seeds,
			bin_seeding=self.bin_seeding, min_bin_freq=self.min_bin_freq,
			cluster_all=self.cluster_all, n_jobs=self.n_jobs, max_iter=self.max_iter )
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public member names exposed by the MeanShift object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the object."""
		return [ 'model', 'train', 'score', 'project', 'predict', 'analyze', 'bandwidth', 'seeds',
			'bin_seeding', 'min_bin_freq', 'cluster_all', 'n_jobs', 'max_iter', 'labels',
			'clusters', 'centroids_', 'iterations', 'features', 'silouette', 'homogeneity',
			'mutual_info', 'v_measure', 'completeness', 'prediction' ]
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.
		
		Purpose:
		    Returns fitted cluster-label assignments.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the MeanShift estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `labels` operation cannot complete."""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def clusters( self ) -> np.ndarray:
		"""Return clusters metadata.
		
		Purpose:
		    Returns fitted cluster identifiers or exemplar indices exposed by the clustering
		    estimator.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the MeanShift estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `clusters` operation cannot complete."""
		if not hasattr( self.model, 'cluster_centers_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.cluster_centers_
	
	@property
	def centroids_( self ) -> np.ndarray:
		"""Return centroids metadata.
		
		Purpose:
		    Returns fitted cluster centers or exemplar center coordinates exposed by the
		    estimator.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the MeanShift estimator."""
		return self.clusters
	
	@property
	def iterations( self ) -> int:
		"""Return iterations metadata.
		
		Purpose:
		    Returns the number of iterations completed by the fitted estimator.
		
		Returns:
		    int: Fitted metadata exposed by the MeanShift estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `iterations` operation cannot complete."""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.
		
		Purpose:
		    Returns the number of input features observed during fitting.
		
		Returns:
		    int: Fitted metadata exposed by the MeanShift estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `features` operation cannot complete."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> MeanShift | None:
		"""Train.
		
		Purpose:
		    Fits the underlying MeanShift estimator to the supplied feature matrix, refreshes
		    fitted prediction metadata when available, and returns the current object for
		    chained clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    MeanShift | None: Fitted object instance.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'train( self, X: np.ndarray ) -> MeanShift | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the MeanShift
		    estimator. The method preserves compatibility with wrappers that fit and predict in
		    a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""Predict.
		
		Purpose:
		    Predicts cluster labels for the supplied feature matrix with the fitted MeanShift
		    estimator and caches the prediction output on the object.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Predicted cluster labels for the supplied samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails.
		    AttributeError: Raised when the `predict` operation cannot complete."""
		try:
			throw_if( 'X', X )
			if not hasattr( self.model, 'cluster_centers_' ):
				raise AttributeError( 'The model data has not been trained!' )
			
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the MeanShift
		    output, including silhouette, completeness, homogeneity, mutual information, and
		    V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			scores[ 'Iterations' ] = self.iterations
			if len( unique_labels ) >= 2 and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the MeanShift clustering result by
		    combining fitted labels, estimator metadata, and metric output generated from the
		    supplied feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			return { 'Labels': labels, 'Bandwidth': self.bandwidth, 'Bin-Seeding':
				self.bin_seeding,
				'Min-Bin-Freq': self.min_bin_freq, 'Cluster-All': self.cluster_all,
				'N-Jobs': self.n_jobs, 'Max-Iter': self.max_iter, 'Iterations': self.iterations,
				'Centroids': self.centroids_, 'Features': self.features, 'Score': df_score }
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'MeanShift'
			exception.method = 'analyze( self, *args ) -> Dict[str, Any] | None'
			Logger( ).write( exception )
			raise exception

class AffinityPropagation( Cluster ):
	"""Represent the `AffinityPropagation` workflow.
	
	Purpose:
	    Wraps AffinityPropagation for exemplar-based clustering from pairwise
	    similarities. The object stores cluster labels, exemplar indices, cluster centers,
	    affinity matrix metadata, iteration counts, and feature metadata.
	
	Attributes:
	    model (skc.AffinityPropagation): Underlying clustering estimator.
	    damping (Optional[float]): Affinity-propagation damping factor applied to message
	                               updates.
	    max_iter (Optional[int]): Maximum iteration count used by the clustering estimator.
	    convergence_iter (Optional[int]): Consecutive stable iterations required for
	                                      convergence.
	    copy (Optional[bool]): Whether the estimator copies input data before processing.
	    preference (object): Sample preferences used by affinity propagation to select
	                         exemplars.
	    affinity (Optional[str]): Affinity or similarity measure used to construct the
	                              clustering graph.
	    verbose (Optional[bool]): Estimator verbosity level.
	    random_state (Optional[int]): Random seed or estimator random-state configuration.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Most recent class-probability estimates
	                                        generated by the model.
	    completeness (Optional[float]): Most recently calculated clustering completeness
	                                    score.
	    homogeneity (Optional[float]): Most recently calculated clustering homogeneity
	                                   score.
	    mutual_info (Optional[float]): Most recently calculated mutual-information score.
	    silouette (Optional[float]): Most recently calculated silhouette score.
	    v_measure (Optional[float]): Most recently calculated V-measure score."""
	model: skc.AffinityPropagation
	damping: Optional[ float ]
	max_iter: Optional[ int ]
	convergence_iter: Optional[ int ]
	copy: Optional[ bool ]
	preference: object
	affinity: Optional[ str ]
	verbose: Optional[ bool ]
	random_state: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, damping: float = 0.5, max_iter: int = 200,
			convergence_iter: int = 15, preference: object = None,
			affinity: str = 'euclidean', copy: bool = True,
			verbose: bool = False, random_state: int | None = None ) -> None:
		"""Initialize clustering object.
		
		Purpose:
		    Initializes the AffinityPropagation clustering object with estimator configuration,
		    runtime metadata fields, prediction caches, and the underlying clustering
		    model used by training and projection methods.
		
		Args:
		    damping (float): Damping factor used by AffinityPropagation.
		    max_iter (int): Maximum number of estimator iterations.
		    convergence_iter (int): Iteration count used to determine AffinityPropagation
		                            convergence.
		    preference (object): Preference value controlling the number of exemplars.
		    affinity (str): Affinity configuration or similarity measure used by the estimator.
		    copy (bool): Flag controlling whether the estimator copies input data.
		    verbose (bool): Estimator verbosity flag.
		    random_state (int | None): Random seed or random-state value.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.damping = damping
		self.max_iter = max_iter
		self.convergence_iter = convergence_iter
		self.copy = copy
		self.preference = preference
		self.affinity = affinity
		self.verbose = verbose
		self.random_state = random_state
		self.model = skc.AffinityPropagation( damping=self.damping, max_iter=self.max_iter,
			convergence_iter=self.convergence_iter, copy=self.copy, preference=self.preference,
			affinity=self.affinity, verbose=self.verbose, random_state=self.random_state )
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public member names exposed by the AffinityPropagation object
		    for interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the object."""
		return [ 'model', 'train', 'score', 'project', 'predict', 'analyze', 'damping', 'max_iter',
			'convergence_iter', 'copy', 'preference', 'affinity', 'verbose', 'random_state',
			'labels', 'clusters', 'centroids_', 'affinity_matrix', 'iterations', 'features',
			'prediction', 'silouette', 'homogeneity', 'mutual_info', 'v_measure', 'completeness' ]
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.
		
		Purpose:
		    Returns fitted cluster-label assignments.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the AffinityPropagation estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `labels` operation cannot complete."""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def clusters( self ) -> np.ndarray:
		"""Return clusters metadata.
		
		Purpose:
		    Returns fitted cluster identifiers or exemplar indices exposed by the clustering
		    estimator.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the AffinityPropagation estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `clusters` operation cannot complete."""
		if not hasattr( self.model, 'cluster_centers_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.cluster_centers_
	
	@property
	def centroids_( self ) -> np.ndarray:
		"""Return centroids metadata.
		
		Purpose:
		    Returns fitted cluster centers or exemplar center coordinates exposed by the
		    estimator.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the AffinityPropagation estimator."""
		return self.clusters
	
	@property
	def affinity_matrix( self ) -> np.ndarray:
		"""Return affinity matrix metadata.
		
		Purpose:
		    Returns the fitted AffinityPropagation affinity matrix.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the AffinityPropagation estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `affinity_matrix` operation cannot complete."""
		if not hasattr( self.model, 'affinity_matrix_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.affinity_matrix_
	
	@property
	def iterations( self ) -> int:
		"""Return iterations metadata.
		
		Purpose:
		    Returns the number of iterations completed by the fitted estimator.
		
		Returns:
		    int: Fitted metadata exposed by the AffinityPropagation estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `iterations` operation cannot complete."""
		if not hasattr( self.model, 'n_iter_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_iter_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.
		
		Purpose:
		    Returns the number of input features observed during fitting.
		
		Returns:
		    int: Fitted metadata exposed by the AffinityPropagation estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `features` operation cannot complete."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> AffinityPropagation | None:
		"""Train.
		
		Purpose:
		    Fits the underlying AffinityPropagation estimator to the supplied feature matrix,
		    refreshes fitted prediction metadata when available, and returns the current object
		    for chained clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    AffinityPropagation | None: Fitted object instance.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'train( self, X: np.ndarray ) -> AffinityPropagation | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the
		    AffinityPropagation estimator. The method preserves compatibility with wrappers that
		    fit and predict in a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""Predict.
		
		Purpose:
		    Predicts cluster labels for the supplied feature matrix with the fitted
		    AffinityPropagation estimator and caches the prediction output on the object.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Predicted cluster labels for the supplied samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails.
		    AttributeError: Raised when the `predict` operation cannot complete."""
		try:
			throw_if( 'X', X )
			if not hasattr( self.model, 'cluster_centers_' ):
				raise AttributeError( 'The model data has not been trained!' )
			
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the
		    AffinityPropagation output, including silhouette, completeness, homogeneity, mutual
		    information, and V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			scores[ 'Iterations' ] = self.iterations
			if self.affinity != 'precomputed' and len( unique_labels ) >= 2 \
					and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the AffinityPropagation clustering result
		    by combining fitted labels, estimator metadata, and metric output generated from the
		    supplied feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			return { 'Labels': labels, 'Damping': self.damping, 'Max-Iter': self.max_iter,
				'Convergence-Iter': self.convergence_iter, 'Copy': self.copy,
				'Preference': self.preference, 'Affinity': self.affinity, 'Verbose': self.verbose,
				'Random-State': self.random_state, 'Iterations': self.iterations,
				'Centroids': self.centroids_, 'Features': self.features, 'Score': df_score }
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'AffinityPropagation'
			exception.method = 'analyze( self, *args ) -> Dict[str, Any] | None'
			Logger( ).write( exception )
			raise exception

class Birch( Cluster ):
	"""Represent the `Birch` workflow.
	
	Purpose:
	    Wraps Birch for scalable clustering through clustering-feature subtrees. The
	    object stores labels, subcluster centers, global centroids, subcluster labels, and
	    feature metadata while supporting projection, prediction, transformation, scoring,
	    and analysis.
	
	Attributes:
	    model (skc.Birch): Underlying clustering estimator.
	    threshold (Optional[float]): Decision, variance, or distance threshold applied by
	                                 the workflow.
	    branching_factor (Optional[int]): Maximum number of child subclusters in each BIRCH
	                                      node.
	    n_clusters (object): Number of clusters requested by the object.
	    compute_labels (Optional[bool]): Whether BIRCH computes final cluster labels.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Most recent class-probability estimates
	                                        generated by the model.
	    completeness (Optional[float]): Most recently calculated clustering completeness
	                                    score.
	    homogeneity (Optional[float]): Most recently calculated clustering homogeneity
	                                   score.
	    mutual_info (Optional[float]): Most recently calculated mutual-information score.
	    silouette (Optional[float]): Most recently calculated silhouette score.
	    v_measure (Optional[float]): Most recently calculated V-measure score."""
	model: skc.Birch
	threshold: Optional[ float ]
	branching_factor: Optional[ int ]
	n_clusters: object
	compute_labels: Optional[ bool ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, threshold: float = 0.5, branching_factor: int = 50,
			n_clusters: object = 3, compute_labels: bool = True ) -> None:
		"""Initialize clustering object.
		
		Purpose:
		    Initializes the Birch clustering object with estimator configuration, runtime
		    metadata fields, prediction caches, and the underlying clustering model used
		    by training and projection methods.
		
		Args:
		    threshold (float): Birch radius threshold for merging subclusters.
		    branching_factor (int): Maximum number of child subclusters in each Birch node.
		    n_clusters (object): Optional override for the requested number of clusters.
		    compute_labels (bool): Flag controlling whether Birch computes labels after fitting.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.threshold = threshold
		self.branching_factor = branching_factor
		self.n_clusters = n_clusters
		self.compute_labels = compute_labels
		self.model = skc.Birch( threshold=self.threshold, branching_factor=self.branching_factor,
			n_clusters=self.n_clusters, compute_labels=self.compute_labels )
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public member names exposed by the Birch object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the object."""
		return [ 'model', 'score', 'project', 'predict', 'train', 'transform', 'analyze',
			'threshold', 'branching_factor', 'n_clusters', 'compute_labels', 'labels',
			'subcluster_centers', 'centroids_', 'subcluster_labels', 'features', 'silouette',
			'homogeneity', 'mutual_info', 'v_measure', 'completeness', 'prediction' ]
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.
		
		Purpose:
		    Returns fitted cluster-label assignments.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the Birch estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `labels` operation cannot complete."""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def subcluster_centers( self ) -> np.ndarray:
		"""Return subcluster centers metadata.
		
		Purpose:
		    Returns Birch subcluster centers.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the Birch estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `subcluster_centers` operation cannot complete."""
		if not hasattr( self.model, 'subcluster_centers_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.subcluster_centers_
	
	@property
	def centroids_( self ) -> np.ndarray:
		"""Return centroids metadata.
		
		Purpose:
		    Returns fitted cluster centers or exemplar center coordinates exposed by the
		    estimator.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the Birch estimator."""
		return self.subcluster_centers
	
	@property
	def subcluster_labels( self ) -> np.ndarray:
		"""Return subcluster labels metadata.
		
		Purpose:
		    Returns Birch subcluster labels.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the Birch estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `subcluster_labels` operation cannot complete."""
		if not hasattr( self.model, 'subcluster_labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.subcluster_labels_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.
		
		Purpose:
		    Returns the number of input features observed during fitting.
		
		Returns:
		    int: Fitted metadata exposed by the Birch estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `features` operation cannot complete."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> Birch | None:
		"""Train.
		
		Purpose:
		    Fits the underlying Birch estimator to the supplied feature matrix, refreshes fitted
		    prediction metadata when available, and returns the current object for chained
		    clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    Birch | None: Fitted object instance.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			if hasattr( self.model, 'labels_' ):
				self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'train( self, X: np.ndarray ) -> Birch | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the Birch
		    estimator. The method preserves compatibility with wrappers that fit and predict in
		    a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""Predict.
		
		Purpose:
		    Predicts cluster labels for the supplied feature matrix with the fitted Birch
		    estimator and caches the prediction output on the object.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Predicted cluster labels for the supplied samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails.
		    AttributeError: Raised when the `predict` operation cannot complete."""
		try:
			throw_if( 'X', X )
			if not hasattr( self.model, 'subcluster_centers_' ):
				raise AttributeError( 'The model data has not been trained!' )
			
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray | None:
		"""Transform.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted Birch estimator and returns
		    estimator-specific distances, embeddings, or transformed cluster-space values.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Transformed cluster-space representation produced by the
		                       estimator.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			if not hasattr( self.model, 'subcluster_centers_' ):
				self.model.fit( X )
				if hasattr( self.model, 'labels_' ):
					self.prediction = self.model.labels_
			
			return self.model.transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the Birch
		    output, including silhouette, completeness, homogeneity, mutual information, and
		    V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			unique_labels = np.unique( labels )
			scores[ 'Clusters' ] = len( unique_labels )
			scores[ 'Subclusters' ] = len( self.subcluster_centers )
			if len( unique_labels ) >= 2 and len( unique_labels ) < len( labels ):
				self.silouette = silhouette_score( X, labels )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the Birch clustering result by combining
		    fitted labels, estimator metadata, and metric output generated from the supplied
		    feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			return { 'Labels': labels, 'Threshold': self.threshold,
				'Branching-Factor': self.branching_factor, 'N-Clusters': self.n_clusters,
				'Compute-Labels': self.compute_labels,
				'Subcluster-Centers': self.subcluster_centers,
				'Subcluster-Labels': self.subcluster_labels, 'Features': self.features,
				'Score': df_score }
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'Birch'
			exception.method = 'analyze( self, *args ) -> Dict[str, Any] | None'
			Logger( ).write( exception )
			raise exception

class OPTICS( Cluster ):
	"""Represent the `OPTICS` workflow.
	
	Purpose:
	    Wraps OPTICS for density-based clustering across varying neighborhood radii.
	    The object stores labels, ordering, reachability distances, core distances,
	    predecessors, and feature metadata for inspection and scoring.
	
	Attributes:
	    model (skc.OPTICS): Underlying clustering estimator.
	    min_samples (Optional[int]): Minimum neighborhood sample count required to form a
	                                 dense region.
	    max_eps (Optional[float]): Maximum neighborhood radius considered by OPTICS.
	    metric (object): Distance or scoring metric used by the estimator.
	    p (Optional[float]): Power parameter used by the Minkowski distance metric.
	    metric_params (Optional[Dict[str, Any]]): Additional keyword values supplied to the
	                                              selected metric.
	    cluster_method (Optional[str]): Extraction method used to form clusters from the
	                                    reachability graph.
	    eps (Optional[float]): Neighborhood radius used to identify nearby samples.
	    xi (Optional[float]): Minimum steepness used to identify OPTICS cluster boundaries.
	    predecessor_correction (Optional[bool]): Whether OPTICS corrects clusters using
	                                             predecessor reachability.
	    min_cluster_size (object): Minimum sample count or fraction required for an OPTICS
	                               cluster.
	    algorithm (Optional[str]): Algorithm selected for estimator fitting or neighbor
	                               search.
	    leaf_size (Optional[int]): Leaf size used by tree-based neighbor-search structures.
	    memory (object): Cache configuration used for reusable estimator computations.
	    n_jobs (Optional[int]): Number of parallel worker jobs used by the estimator.
	    prediction (Optional[np.ndarray]): Most recent cluster-label assignments generated
	                                       by the object.
	    probability (Optional[np.ndarray]): Most recent class-probability estimates
	                                        generated by the model.
	    completeness (Optional[float]): Most recently calculated clustering completeness
	                                    score.
	    homogeneity (Optional[float]): Most recently calculated clustering homogeneity
	                                   score.
	    mutual_info (Optional[float]): Most recently calculated mutual-information score.
	    silouette (Optional[float]): Most recently calculated silhouette score.
	    v_measure (Optional[float]): Most recently calculated V-measure score."""
	model: skc.OPTICS
	min_samples: Optional[ int ]
	max_eps: Optional[ float ]
	metric: object
	p: Optional[ float ]
	metric_params: Optional[ Dict[ str, Any ] ]
	cluster_method: Optional[ str ]
	eps: Optional[ float ]
	xi: Optional[ float ]
	predecessor_correction: Optional[ bool ]
	min_cluster_size: object
	algorithm: Optional[ str ]
	leaf_size: Optional[ int ]
	memory: object
	n_jobs: Optional[ int ]
	prediction: Optional[ np.ndarray ]
	probability: Optional[ np.ndarray ]
	completeness: Optional[ float ]
	homogeneity: Optional[ float ]
	mutual_info: Optional[ float ]
	silouette: Optional[ float ]
	v_measure: Optional[ float ]
	
	def __init__( self, samples: int = 5, max_eps: float = np.inf,
			metric: object = 'minkowski', algorithm: str = 'auto',
			leaf_size: int = 30, eps: float | None = None,
			predecessor_correction: bool = True,
			min_cluster_size: int | float | None = None,
			min_samples: int | None = None, p: float = 2,
			metric_params: Dict[ str, Any ] | None = None,
			cluster_method: str = 'xi', xi: float = 0.05,
			memory: object = None, n_jobs: int | None = None ) -> None:
		"""Initialize clustering object.
		
		Purpose:
		    Initializes the OPTICS clustering object with estimator configuration, runtime
		    metadata fields, prediction caches, and the underlying clustering model used
		    by training and projection methods.
		
		Args:
		    samples (int): Minimum sample count used for density-neighborhood definitions.
		    max_eps (float): Maximum neighborhood distance considered by OPTICS.
		    metric (object): Distance metric used by the clustering estimator.
		    algorithm (str): Estimator algorithm selection passed to sklearn.
		    leaf_size (int): Leaf size used by tree-based neighbor-search algorithms.
		    eps (float | None): Maximum neighborhood radius for density-based clustering.
		    predecessor_correction (bool): Flag controlling OPTICS predecessor correction
		                                   behavior.
		    min_cluster_size (int | float | None): Minimum cluster size used by OPTICS
		                                           extraction.
		    min_samples (int | None): Optional override for the minimum sample count.
		    p (float): Power parameter used by Minkowski distance metrics.
		    metric_params (Dict[str, Any] | None): Additional metric keyword arguments passed to
		                                           sklearn.
		    cluster_method (str): Cluster extraction method used by OPTICS.
		    xi (float): Steepness threshold used by OPTICS xi extraction.
		    memory (object): Optional joblib memory configuration used by sklearn.
		    n_jobs (int | None): Number of parallel jobs used by when supported.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.min_samples = min_samples if min_samples is not None else samples
		self.max_eps = max_eps
		self.metric = metric
		self.p = p
		self.metric_params = metric_params
		self.cluster_method = cluster_method
		self.eps = eps
		self.xi = xi
		self.predecessor_correction = predecessor_correction
		self.min_cluster_size = min_cluster_size
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.memory = memory
		self.n_jobs = n_jobs
		self.model = skc.OPTICS( min_samples=self.min_samples, max_eps=self.max_eps,
			metric=self.metric, p=self.p, metric_params=self.metric_params,
			cluster_method=self.cluster_method, eps=self.eps, xi=self.xi,
			predecessor_correction=self.predecessor_correction,
			min_cluster_size=self.min_cluster_size, algorithm=self.algorithm,
			leaf_size=self.leaf_size, memory=self.memory, n_jobs=self.n_jobs )
		self.prediction = None
		self.probability = None
		self.silouette = 0.0
		self.homogeneity = 0.0
		self.mutual_info = 0.0
		self.v_measure = 0.0
		self.completeness = 0.0
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable public member names exposed by the OPTICS object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the object."""
		return [ 'model', 'train', 'score', 'project', 'analyze', 'min_samples', 'max_eps',
			'metric', 'p', 'metric_params', 'cluster_method', 'eps', 'xi',
			'predecessor_correction', 'min_cluster_size', 'algorithm', 'leaf_size', 'memory',
			'n_jobs', 'labels', 'ordering', 'reachability', 'core_distances', 'predecessor',
			'features', 'prediction', 'silouette', 'homogeneity', 'mutual_info',
			'v_measure', 'completeness' ]
	
	@property
	def labels( self ) -> np.ndarray:
		"""Return labels metadata.
		
		Purpose:
		    Returns fitted cluster-label assignments.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the OPTICS estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `labels` operation cannot complete."""
		if not hasattr( self.model, 'labels_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.labels_
	
	@property
	def ordering( self ) -> np.ndarray:
		"""Return ordering metadata.
		
		Purpose:
		    Returns the OPTICS cluster ordering.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the OPTICS estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `ordering` operation cannot complete."""
		if not hasattr( self.model, 'ordering_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.ordering_
	
	@property
	def reachability( self ) -> np.ndarray:
		"""Return reachability metadata.
		
		Purpose:
		    Returns OPTICS reachability distances.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the OPTICS estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `reachability` operation cannot complete."""
		if not hasattr( self.model, 'reachability_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.reachability_
	
	@property
	def core_distances( self ) -> np.ndarray:
		"""Return core distances metadata.
		
		Purpose:
		    Returns OPTICS core-distance values.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the OPTICS estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `core_distances` operation cannot complete."""
		if not hasattr( self.model, 'core_distances_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.core_distances_
	
	@property
	def predecessor( self ) -> np.ndarray:
		"""Return predecessor metadata.
		
		Purpose:
		    Returns OPTICS predecessor indices.
		
		Returns:
		    np.ndarray: Fitted metadata exposed by the OPTICS estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `predecessor` operation cannot complete."""
		if not hasattr( self.model, 'predecessor_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.predecessor_
	
	@property
	def features( self ) -> int:
		"""Return features metadata.
		
		Purpose:
		    Returns the number of input features observed during fitting.
		
		Returns:
		    int: Fitted metadata exposed by the OPTICS estimator.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly.
		    AttributeError: Raised when the `features` operation cannot complete."""
		if not hasattr( self.model, 'n_features_in_' ):
			raise AttributeError( 'The model data has not been trained!' )
		return self.model.n_features_in_
	
	def train( self, X: np.ndarray ) -> OPTICS | None:
		"""Train.
		
		Purpose:
		    Fits the underlying OPTICS estimator to the supplied feature matrix, refreshes
		    fitted prediction metadata when available, and returns the current object for
		    chained clustering workflows.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    OPTICS | None: Fitted object instance.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			self.prediction = self.model.labels_
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'train( self, X: np.ndarray ) -> OPTICS | None'
			Logger( ).write( exception )
			raise exception
	
	def project( self, X: np.ndarray ) -> np.ndarray | None:
		"""Project.
		
		Purpose:
		    Generates cluster assignments for the supplied feature matrix using the OPTICS
		    estimator. The method preserves compatibility with wrappers that fit and predict in
		    a single operation when fitted-state metadata is unavailable.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		
		Returns:
		    np.ndarray | None: Cluster labels or projection output generated for the supplied
		                       samples.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			self.prediction = self.model.fit_predict( X )
			return self.prediction
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'project( self, X: np.ndarray ) -> np.ndarray | None'
			Logger( ).write( exception )
			raise exception
	
	def score( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> pd.DataFrame | None:
		"""Score.
		
		Purpose:
		    Computes intrinsic and optional reference-label clustering metrics for the OPTICS
		    output, including silhouette, completeness, homogeneity, mutual information, and
		    V-measure when supported by the available labels.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    pd.DataFrame | None: Dataframe containing clustering metrics.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			labels = self.model.fit_predict( X )
			self.prediction = labels
			scores = { }
			unique_labels = np.unique( labels )
			cluster_labels = unique_labels[ unique_labels != -1 ]
			scores[ 'Clusters' ] = len( cluster_labels )
			scores[ 'Noise' ] = int( np.sum( labels == -1 ) )
			mask = labels != -1
			if np.sum( mask ) >= 2 and len( np.unique( labels[ mask ] ) ) >= 2:
				self.silouette = silhouette_score( X[ mask ], labels[ mask ] )
				scores[ 'Silouette' ] = self.silouette
			else:
				self.silouette = np.nan
				scores[ 'Silouette' ] = self.silouette
			
			if y is not None:
				self.homogeneity = homogeneity_score( y, labels )
				self.mutual_info = mutual_info_score( y, labels )
				self.completeness = completeness_score( y, labels )
				self.v_measure = v_measure_score( y, labels )
				scores[ 'Homogeneity' ] = self.homogeneity
				scores[ 'Mutual-Info' ] = self.mutual_info
				scores[ 'Completeness' ] = self.completeness
				scores[ 'V-Measure' ] = self.v_measure
			
			return pd.DataFrame( [ scores ] )
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'score( self, *args ) -> pd.DataFrame | None'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> Dict[ str, Any ] | None:
		"""Analyze.
		
		Purpose:
		    Builds a structured analysis payload for the OPTICS clustering result by combining
		    fitted labels, estimator metadata, and metric output generated from the supplied
		    feature matrix.
		
		Args:
		    X (np.ndarray): Feature matrix with rows as samples and columns as numeric
		                    clustering features.
		    y (Optional[np.ndarray]): Optional reference labels aligned to `X` for external
		                              clustering metrics.
		
		Returns:
		    Dict[str, Any] | None: Dictionary containing labels, estimator metadata, and metric
		                           output.
		
		Raises:
		    Error: Raised when validation, estimator execution, metric calculation, or wrapped
		           clustering logic fails."""
		try:
			throw_if( 'X', X )
			if self.prediction is None or len( self.prediction ) != len( X ):
				labels = self.model.fit_predict( X )
				self.prediction = labels
			else:
				labels = self.prediction
			
			df_score = self.score( X, y )
			return { 'Labels': labels, 'Min-Samples': self.min_samples, 'Max-Eps': self.max_eps,
				'Metric': self.metric, 'P': self.p, 'Metric-Params': self.metric_params,
				'Cluster-Method': self.cluster_method, 'Eps': self.eps, 'Xi': self.xi,
				'Predecessor-Correction': self.predecessor_correction,
				'Min-Cluster-Size': self.min_cluster_size, 'Algorithm': self.algorithm,
				'Leaf-Size': self.leaf_size, 'N-Jobs': self.n_jobs, 'Ordering': self.ordering,
				'Reachability': self.reachability, 'Core-Distances': self.core_distances,
				'Predecessor': self.predecessor, 'Features': self.features, 'Score': df_score }
		except Exception as e:
			exception = Error( e )
			exception.module = 'clusters'
			exception.cause = 'OPTICS'
			exception.method = 'analyze( self, *args ) -> Dict[str, Any] | None'
			Logger( ).write( exception )
			raise exception
			
