"""******************************************************************************************
  Assembly:                mathy
  Filename:                imputers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="imputers.py" company="Terry D. Eppler">

     imputers.py
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
    Provides imputation class/object built on scikit estimators for Mathy data
    preparation workflows. The module defines a shared imputer interface and concrete
    class/object for mean, nearest-neighbor, iterative, and strategy-based missing-value
    replacement.
</summary>
******************************************************************************************"""
from __future__ import annotations

from typing import Optional
import numpy as np
import sklearn.impute as im
from boogr import Error, Logger
from sklearn.experimental import enable_iterative_imputer

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

class Imputer( ):
	"""Define the imputer interface.
	
	Purpose:
	    Defines the shared imputation contract used by Mathy preprocessing class/object. The
	    base class establishes a common transformed-data attribute and requires concrete
	    subclasses to implement fit, transform, fit-transform, and inverse-transform
	    behavior where the underlying imputer supports it.
	
	Attributes:
	    transformed_data (Optional[np.ndarray]): Most recent imputed matrix produced by a
	                                             concrete imputer class/object."""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize imputer state.
		
		Purpose:
		    Initializes the common transformed-data cache used by concrete imputer class/object to
		    store the most recent matrix returned by imputation operations.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		self.transformed_data = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> object | None:
		"""Fit an imputer.
		
		Purpose:
		    Defines the required training contract for concrete imputer class/object. Subclasses
		    must fit their underlying  imputer to the supplied feature matrix and return
		    a fitted class/object or compatible result.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the concrete imputer implementation.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    object | None: Fitted concrete imputer class/object or implementation-specific training
		                   result.
		
		Raises:
		    NotImplementedError: Raised when the base method is called directly."""
		raise NotImplementedError( )
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Transform missing values.
		
		Purpose:
		    Defines the required transformation contract for concrete imputer class/object.
		    Subclasses must replace missing values in the supplied feature matrix with values
		    learned during fitting and return the imputed matrix.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the concrete imputer implementation.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced by the concrete imputer class/object.
		
		Raises:
		    NotImplementedError: Raised when the base method is called directly."""
		raise NotImplementedError( )
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Fit and transform missing values.
		
		Purpose:
		    Defines the required combined fit-transform contract for concrete imputer class/object.
		    Subclasses must fit their underlying imputer and return the imputed matrix in one
		    operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the concrete imputer implementation.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced after fitting the concrete imputer
		                class/object.
		
		Raises:
		    NotImplementedError: Raised when the base method is called directly."""
		raise NotImplementedError( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert an imputed matrix.
		
		Purpose:
		    Defines the inverse-transformation contract for concrete imputer class/object.
		    Subclasses must reconstruct a pre-imputation representation when the underlying
		     imputer supports inverse transformation.
		
		Args:
		    X (np.ndarray): Transformed feature matrix supplied to the concrete imputer
		                    implementation.
		
		Returns:
		    np.ndarray: Reconstructed feature matrix produced by the concrete imputer class/object.
		
		Raises:
		    NotImplementedError: Raised when the base method is called directly."""
		raise NotImplementedError( )

class MeanImputer( Imputer ):
	"""Wrap mean-based SimpleImputer.
	
	Purpose:
	    Replaces missing values using the arithmetic mean of each fitted feature column. The
	    class/object stores the  SimpleImputer instance, selected strategy, missing-value
	    indicator setting, and most recent transformed output.
	
	Attributes:
	    strategy (Optional[str]): Imputation strategy passed to  SimpleImputer.
	    add_indicator (bool): Flag indicating whether missing-value indicator columns are
	                          appended.
	    imputer (im.SimpleImputer): Underlying  imputer.
	    transformed_data (Optional[np.ndarray]): Most recent imputed matrix produced by the
	                                             class/object."""
	strategy: Optional[ str ]
	add_indicator: bool
	imputer: im.SimpleImputer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, strategy: str = 'mean', add_indicator: bool = False ) -> None:
		"""Initialize MeanImputer.
		
		Purpose:
		    Initializes the mean-imputation class/object by configuring the underlying
		    SimpleImputer and preparing the transformed-data cache used by later transformation
		    methods.
		
		Args:
		    strategy (str): Imputation strategy passed to  SimpleImputer.
		    add_indicator (bool): Flag indicating whether missing-value indicator columns are
		                          appended.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.strategy = strategy
		self.add_indicator = add_indicator
		self.imputer = im.SimpleImputer( strategy=self.strategy, add_indicator=self.add_indicator )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the mean-imputer class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'imputer', 'transformed_data', 'strategy', 'add_indicator', 'train', 'transform',
			'train_transform', 'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> MeanImputer | None:
		"""Fit MeanImputer.
		
		Purpose:
		    Fits the underlying  SimpleImputer to the supplied feature matrix and returns
		    the current class/object for consistent preprocessing chains.
		
		Args:
		    X (np.ndarray): Feature matrix used to learn imputation statistics.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    MeanImputer | None: Fitted mean-imputer class/object.
		
		Raises:
		    Error: Raised when validation or  fitting fails."""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'train( self, *args ) -> MeanImputer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Transform with MeanImputer.
		
		Purpose:
		    Replaces missing values in the supplied feature matrix using mean statistics learned
		    during fitting and stores the imputed matrix on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix transformed by the fitted imputer.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced by the fitted mean imputer.
		
		Raises:
		    Error: Raised when validation or  transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Fit and transform with MeanImputer.
		
		Purpose:
		    Fits the underlying  SimpleImputer to the supplied feature matrix, imputes
		    the same matrix immediately, and stores the imputed result on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting and imputation.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced after fitting the mean imputer.
		
		Raises:
		    Error: Raised when validation or  fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert MeanImputer output.
		
		Purpose:
		    Reconstructs a feature matrix toward its pre-imputed form using  inverse
		    transformation when missing-value indicator columns were included during fitting.
		
		Args:
		    X (np.ndarray): Transformed feature matrix to invert.
		
		Returns:
		    np.ndarray: Reconstructed feature matrix produced by the  imputer.
		
		Raises:
		    Error: Raised when validation fails, indicator columns are unavailable, or inverse
		           transformation fails.
		    ValueError: Raised when the `inverse_transform` operation cannot complete."""
		try:
			throw_if( 'X', X )
			if not self.add_indicator:
				raise ValueError(
					'inverse_transform requires add_indicator=True in the '
					'underlying .impute.SimpleImputer.' )
			return self.imputer.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MeanImputer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class NearestImputer( Imputer ):
	"""Wrap KNNImputer.
	
	Purpose:
	    Replaces missing values using values from nearest neighboring samples. The class/object
	    stores the  KNNImputer instance, neighbor count, and most recent transformed
	    output while exposing a consistent Mathy imputation interface.
	
	Attributes:
	    n_neighbors (Optional[int]): Number of neighboring samples used for imputation.
	    imputer (im.KNNImputer): Underlying  nearest-neighbor imputer.
	    transformed_data (Optional[np.ndarray]): Most recent imputed matrix produced by the
	                                             class/object."""
	n_neighbors: Optional[ int ]
	imputer: im.KNNImputer
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, neighbors: int = 5 ) -> None:
		"""Initialize NearestImputer.
		
		Purpose:
		    Initializes the nearest-neighbor imputation class/object by configuring the underlying
		     KNNImputer with the requested neighbor count and preparing the
		    transformed-data cache.
		
		Args:
		    neighbors (int): Number of neighboring samples used to impute missing values.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_neighbors = neighbors
		self.imputer = im.KNNImputer( n_neighbors=self.n_neighbors )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the nearest-neighbor imputer class/object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'imputer', 'transformed_data', 'n_neighbors', 'train', 'transform',
			'train_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> NearestImputer | None:
		"""Fit NearestImputer.
		
		Purpose:
		    Fits the underlying  KNNImputer to the supplied feature matrix and returns
		    the current class/object for consistent preprocessing chains.
		
		Args:
		    X (np.ndarray): Feature matrix used to learn nearest-neighbor imputation structure.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    NearestImputer | None: Fitted nearest-neighbor imputer class/object.
		
		Raises:
		    Error: Raised when validation or fitting fails."""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'train( self, *args ) -> NearestImputer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Transform with NearestImputer.
		
		Purpose:
		    Replaces missing values in the supplied feature matrix using nearest-neighbor
		    statistics learned during fitting and stores the imputed matrix on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix transformed by the fitted imputer.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced by the fitted nearest-neighbor imputer.
		
		Raises:
		    Error: Raised when validation or transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Fit and transform with NearestImputer.
		
		Purpose:
		    Fits the underlying KNNImputer to the supplied feature matrix, imputes the
		    same matrix immediately, and stores the imputed result on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting and imputation.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced after fitting the nearest-neighbor
		                imputer.
		
		Raises:
		    Error: Raised when validation or fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'NearestImputer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class IterativeImputer( Imputer ):
	"""Wrap IterativeImputer.
	
	Purpose:
	    Replaces missing values by modeling each feature with missing observations as a
	    function of the other features and iterating through round-robin imputation rounds.
	    The class/object stores iteration configuration, random state, the sklearn
	    IterativeImputer instance, and the most recent transformed output.
	
	Attributes:
	    imputer (im.IterativeImputer): Underlying iterative imputer.
	    max_iter (Optional[int]): Maximum number of imputation rounds.
	    random_state (Optional[int]): Random seed used by the imputer.
	    transformed_data (Optional[np.ndarray]): Most recent imputed matrix produced by the
	                                             class/object."""
	imputer: im.IterativeImputer
	max_iter: Optional[ int ]
	random_state: Optional[ int ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, max_iter: int = 10, random_state: int = 0 ) -> None:
		"""Initialize IterativeImputer.
		
		Purpose:
		    Initializes the iterative imputation class/object by configuring maximum imputation
		    rounds and random state on the underlying IterativeImputer. The constructor
		    prepares the transformed-data cache used by later transformation methods.
		
		Args:
		    max_iter (int): Maximum number of iterative imputation rounds.
		    random_state (int): Random seed used by the underlying imputer.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.max_iter = max_iter
		self.random_state = random_state
		self.imputer = im.IterativeImputer( max_iter=self.max_iter,
			random_state=self.random_state )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the iterative imputer class/object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'imputer', 'transformed_data', 'max_iter', 'random_state', 'train', 'transform',
			'train_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> IterativeImputer | None:
		"""Fit IterativeImputer.
		
		Purpose:
		    Fits the underlying IterativeImputer to the supplied feature matrix and
		    returns the current class/object for consistent preprocessing chains.
		
		Args:
		    X (np.ndarray): Feature matrix used to learn iterative imputation models.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    IterativeImputer | None: Fitted iterative-imputer class/object.
		
		Raises:
		    Error: Raised when validation or fitting fails."""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'train( self, *args ) -> IterativeImputer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Transform with IterativeImputer.
		
		Purpose:
		    Replaces missing values in the supplied feature matrix using the fitted iterative
		    imputation model and stores the imputed matrix on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix transformed by the fitted imputer.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced by the fitted iterative imputer.
		
		Raises:
		    Error: Raised when validation or transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Fit and transform with IterativeImputer.
		
		Purpose:
		    Fits the underlying IterativeImputer to the supplied feature matrix, imputes
		    the same matrix immediately, and stores the imputed result on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting and imputation.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced after fitting the iterative imputer.
		
		Raises:
		    Error: Raised when validation or fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'IterativeImputer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class SimpleImputer( Imputer ):
	"""Wrap strategy-based SimpleImputer.
	
	Purpose:
	    Replaces missing values using SimpleImputer strategies such as mean, median,
	    most-frequent, or constant replacement. The class/object stores imputation configuration,
	    missing-value indicator behavior, empty-feature behavior, and most recent
	    transformed output.
	
	Attributes:
	    imputer (im.SimpleImputer): Underlying simple imputer.
	    transformed_data (Optional[np.ndarray]): Most recent imputed matrix produced by the
	                                             class/object.
	    strategy (Optional[str]): Imputation strategy used by the imputer.
	    fill_value (Optional[object]): Replacement value used for constant imputation.
	    add_indicator (bool): Flag indicating whether missing-value indicator columns are
	                          appended.
	    keep_empty_features (bool): Flag indicating whether entirely missing fitted features
	                                are preserved."""
	imputer: im.SimpleImputer
	transformed_data: Optional[ np.ndarray ]
	strategy: Optional[ str ]
	fill_value: Optional[ object ]
	add_indicator: bool
	keep_empty_features: bool
	
	def __init__( self, strategy: str = 'mean', fill_value: object = 0.0,
			add_indicator: bool = False, keep_empty_features: bool = False ) -> None:
		"""Initialize SimpleImputer.
		
		Purpose:
		    Initializes the strategy-based imputer class/object by configuring SimpleImputer
		    with the selected strategy, constant fill value, missing-value indicator setting,
		    and empty-feature behavior.
		
		Args:
		    strategy (str): Imputation strategy used by SimpleImputer.
		    fill_value (object): Replacement value used when `strategy` is `constant`.
		    add_indicator (bool): Flag indicating whether missing-value indicator columns are
		                          appended.
		    keep_empty_features (bool): Flag indicating whether entirely missing fitted features
		                                are preserved.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.strategy = strategy
		self.fill_value = fill_value
		self.add_indicator = add_indicator
		self.keep_empty_features = keep_empty_features
		self.imputer = im.SimpleImputer( strategy=self.strategy, fill_value=self.fill_value,
			add_indicator=self.add_indicator, keep_empty_features=self.keep_empty_features )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the simple-imputer class/object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'imputer', 'transformed_data', 'strategy', 'fill_value', 'add_indicator',
			'keep_empty_features', 'train', 'transform', 'train_transform', 'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> SimpleImputer | None:
		"""Fit SimpleImputer.
		
		Purpose:
		    Fits the underlying SimpleImputer to the supplied feature matrix and returns
		    the current class/object for consistent preprocessing chains.
		
		Args:
		    X (np.ndarray): Feature matrix used to learn simple imputation statistics.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    SimpleImputer | None: Fitted simple-imputer class/object.
		
		Raises:
		    Error: Raised when validation or fitting fails."""
		try:
			throw_if( 'X', X )
			self.imputer.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SimpleImputer'
			exception.method = 'train( self, *args ) -> SimpleImputer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Transform with SimpleImputer.
		
		Purpose:
		    Replaces missing values in the supplied feature matrix using fitted strategy-based
		    statistics or constants and stores the imputed matrix on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix transformed by the fitted imputer.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced by the fitted simple imputer.
		
		Raises:
		    Error: Raised when validation or transformation fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SimpleImputer'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ] = None ) -> np.ndarray:
		"""Fit and transform with SimpleImputer.
		
		Purpose:
		    Fits the underlying SimpleImputer to the supplied feature matrix, imputes
		    the same matrix immediately, and stores the imputed result on the class/object.
		
		Args:
		    X (np.ndarray): Feature matrix used for fitting and imputation.
		    y (Optional[np.ndarray]): Optional target array accepted for API compatibility.
		
		Returns:
		    np.ndarray: Imputed feature matrix produced after fitting the simple imputer.
		
		Raises:
		    Error: Raised when validation or fit-transform execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.imputer.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SimpleImputer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert SimpleImputer output.
		
		Purpose:
		    Reconstructs a feature matrix toward its pre-imputed form using inverse
		    transformation when missing-value indicator columns were included during fitting.
		
		Args:
		    X (np.ndarray): Transformed feature matrix to invert.
		
		Returns:
		    np.ndarray: Reconstructed feature matrix produced by the imputer.
		
		Raises:
		    Error: Raised when validation fails, indicator columns are unavailable, or inverse
		           transformation fails.
		    ValueError: Raised when the `inverse_transform` operation cannot complete."""
		try:
			throw_if( 'X', X )
			if not self.add_indicator:
				raise ValueError( 'inverse_transform requires add_indicator=True in the '
					'underlying sklearn.impute.SimpleImputer.' )
			return self.imputer.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'SimpleImputer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
