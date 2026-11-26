"""******************************************************************************************
  Assembly:                mathy
  Filename:                scalers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="guro.py" company="Terry D. Eppler">

     scalers.py
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
    Provides scaler wrappers built on sklearn.preprocessing estimators for Mathy data
    preparation workflows. The module defines a shared scaler interface and concrete
    wrappers for standardization, min-max scaling, robust scaling, sample normalization,
    and maximum-absolute-value scaling.
</summary>
******************************************************************************************"""
from __future__ import annotations
from typing import Optional
import numpy as np
import sklearn.preprocessing as pp
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

class Scaler( ):
	"""Define the scaler interface.
	
	Purpose:
	    Defines the shared scaler interface used by all Mathy preprocessing wrappers. The
	    base class establishes a common transformed-data attribute and requires concrete
	    subclasses to implement fit, transform, fit-transform, and inverse-transform
	    operations compatible with preprocessing estimators.
	
	Attributes:
	    transformed_data (Optional[np.ndarray]): Most recent transformed matrix produced by
	                                             a concrete scaler class/object."""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize scaler state.
		
		Purpose:
		    Initializes shared runtime state for scaler wrappers by creating the
		    transformed-data attribute used to cache the most recent transformation output.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		self.transformed_data = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""Fit a scaler.
		
		Purpose:
		    Defines the required training contract for concrete scaler wrappers. Subclasses must
		    fit their underlying preprocessing estimator to the supplied feature matrix and
		    return the fitted class/object or compatible result.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    object | None: Fitted concrete scaler class/object or implementation-specific training
		                   result.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails.
		    NotImplementedError: Raised when the `train` operation cannot complete."""
		try:
			msg = 'Concrete scaler wrappers must implement train( self, X, y=None ).'
			raise NotImplementedError( msg )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Scaler'
			exception.method = 'train( self, *args ) -> object | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform features.
		
		Purpose:
		    Defines the required transformation contract for concrete scaler wrappers.
		    Subclasses must transform the supplied feature matrix with a previously fitted
		    preprocessing estimator and return the transformed matrix.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the concrete scaler class/object.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails.
		    NotImplementedError: Raised when the `transform` operation cannot complete."""
		try:
			raise NotImplementedError(
				'Concrete scaler wrappers must implement transform( self, X, y=None ).' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Scaler'
			exception.method = ('transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform features.
		
		Purpose:
		    Defines the required combined training and transformation contract for concrete
		    scaler wrappers. Subclasses must fit the preprocessing estimator and return the
		    transformed feature matrix in one operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the concrete scaler
		                class/object.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails.
		    NotImplementedError: Raised when the `train_transform` operation cannot complete."""
		try:
			raise NotImplementedError(
				'Concrete scaler wrappers must implement train_transform( self, X, y=None ).' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Scaler'
			exception.method = ('train_transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert scaled features.
		
		Purpose:
		    Defines the required inverse-transformation contract for concrete scaler wrappers.
		    Subclasses must map transformed feature values back toward their original scale when
		    the underlying estimator supports that operation.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		
		Returns:
		    np.ndarray: Feature matrix reconstructed in the original input scale.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails.
		    NotImplementedError: Raised when the `inverse_transform` operation cannot complete."""
		try:
			raise NotImplementedError(
				'Concrete scaler wrappers must implement inverse_transform( self, X ).' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Scaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class StandardScaler( Scaler ):
	"""Provides the StandardScaler feature scaling.
	
	Purpose:
	    Standardizes numeric features by removing the fitted mean and scaling to unit
	    variance. The class/object stores the estimator and cached transformed matrix
	    while preserving a uniform Mathy preprocessing API.
	
	Attributes:
	    model (pp.StandardScaler): Underlying preprocessing estimator used by the
	                               class/object.
	    transformed_data (Optional[np.ndarray]): Most recent transformed feature matrix
	                                             produced by the class/object."""
	model: pp.StandardScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize StandardScaler.
		
		Purpose:
		    Initializes the StandardScaler class/object by configuring the underlying sklearn
		    preprocessing estimator and preparing the transformed-data cache used by later
		    transformation methods.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.model = pp.StandardScaler( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the StandardScaler class/object for
		    interactive inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'scaler', 'transformed_data', 'train', 'transform', 'train_transform',
			'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> StandardScaler | None:
		"""Fit StandardScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix
		    and returns the current class/object so callers can chain preprocessing operations
		    through the Mathy interface.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    StandardScaler | None: Fitted scaler class/object instance.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('train( self, *args ) -> StandardScaler | None')
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform data with StandardScaler feature scaling.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted preprocessing
		    estimator and stores the transformed output on the class/object for later inspection or
		    reuse.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with StandardScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix,
		    transforms the same matrix immediately, and stores the transformed output on the
		    class/object.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = ('train_transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Reverse StandardScaler transformation.
		
		Purpose:
		    Maps transformed feature values back toward their original scale using the inverse
		    transformation provided by the underlying preprocessing estimator.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		
		Returns:
		    np.ndarray: Feature matrix reconstructed in the original input scale.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'StandardScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class MinMaxScaler( Scaler ):
	"""Provides MinMaxScaler feature scaling.
	
	Purpose:
	    Scales numeric features into a bounded range using minimum and maximum values
	    learned from the training data. The class/object stores the estimator and cached
	    transformed matrix while preserving a uniform Mathy preprocessing API.
	
	Attributes:
	    model (pp.MinMaxScaler): Underlying preprocessing estimator used by the
	                             class/object.
	    transformed_data (Optional[np.ndarray]): Most recent transformed feature matrix
	                                             produced by the class/object."""
	model: pp.MinMaxScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize MinMaxScaler.
		
		Purpose:
		    Initializes the MinMaxScaler class/object by configuring the underlying sklearn
		    preprocessing estimator and preparing the transformed-data cache used by later
		    transformation methods.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.model = pp.MinMaxScaler( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the MinMaxScaler class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'scaler', 'transformed_data', 'train', 'transform', 'train_transform',
			'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> MinMaxScaler | None:
		"""Fit MinMaxScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix
		    and returns the current class/object so callers can chain preprocessing operations
		    through the Mathy interface.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    MinMaxScaler | None: Fitted scaler class/object instance.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = ('train( self, *args ) -> MinMaxScaler | None')
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform with MinMaxScaler.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted preprocessing
		    estimator and stores the transformed output on the class/object for later inspection or
		    reuse.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = ('transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with MinMaxScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix,
		    transforms the same matrix immediately, and stores the transformed output on the
		    class/object.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Error: Raised when validation, preprocessing, or wrapped scaler
		                execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = ('train_transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert MinMaxScaler transformation.
		
		Purpose:
		    Maps transformed feature values back toward their original scale using the inverse
		    transformation provided by the underlying preprocessing estimator.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		
		Returns:
		    np.ndarray: Feature matrix reconstructed in the original input scale.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MinMaxScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class RobustScaler( Scaler ):
	"""Provides RobustScaler feature scaling.
	
	Purpose:
	    Scales numeric features with statistics that are robust to outliers by using medians
	    and interquartile ranges learned from the training data. The class/object stores the
	    estimator and cached transformed matrix while preserving a uniform Mathy
	    preprocessing API.
	
	Attributes:
	    model (pp.RobustScaler): Underlying preprocessing estimator used by the
	                             class/object.
	    transformed_data (Optional[np.ndarray]): Most recent transformed feature matrix
	                                             produced by the class/object."""
	model: pp.RobustScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize RobustScaler.
		
		Purpose:
		    Initializes the RobustScaler class/object by configuring the underlying sklearn
		    preprocessing estimator and preparing the transformed-data cache used by later
		    transformation methods.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.model = pp.RobustScaler( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the RobustScaler class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'scaler', 'transformed_data', 'train', 'transform', 'train_transform',
			'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> RobustScaler | None:
		"""Fit RobustScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix
		    and returns the current class/object so callers can chain preprocessing operations
		    through the Mathy interface.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    RobustScaler | None: Fitted scaler class/object instance.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = ('train( self, *args ) -> RobustScaler | None')
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform with RobustScaler.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted preprocessing
		    estimator and stores the transformed output on the class/object for later inspection or
		    reuse.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = ('transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with RobustScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix,
		    transforms the same matrix immediately, and stores the transformed output on the
		    class/object.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = ('train_transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert RobustScaler transformation.
		
		Purpose:
		    Maps transformed feature values back toward their original scale using the inverse
		    transformation provided by the underlying preprocessing estimator.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		
		Returns:
		    np.ndarray: Feature matrix reconstructed in the original input scale.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.inverse_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'RobustScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class NormalScaler( Scaler ):
	"""Provides normalizing feature scaling.
	
	Purpose:
	    Normalizes individual sample vectors to the requested norm so each row is rescaled
	    independently. The class/object stores the estimator, selected norm, and cached
	    transformed matrix while preserving a uniform Mathy preprocessing API.
	
	Attributes:
	    model (pp.Normalizer): Underlying preprocessing estimator used by the
	                           class/object.
	    transformed_data (Optional[np.ndarray]): Most recent transformed feature matrix
	                                             produced by the class/object.
	    norm (str): Vector norm used by the normalizer."""
	model: pp.Normalizer
	transformed_data: Optional[ np.ndarray ]
	norm: str
	
	def __init__( self, norm: str = 'l2' ) -> None:
		"""Initialize NormalScaler.
		
		Purpose:
		    Initializes the NormalScaler class/object by configuring the underlying sklearn
		    preprocessing estimator and preparing the transformed-data cache used by later
		    transformation methods.
		
		Args:
		    norm (str): Norm used to normalize each sample vector.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.norm = norm
		self.model = pp.Normalizer( norm=self.norm )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the NormalScaler class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'scaler', 'transformed_data', 'norm', 'train', 'transform', 'train_transform',
			'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> NormalScaler | None:
		"""Fit NormalScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix
		    and returns the current class/object so callers can chain preprocessing operations
		    through the Mathy interface.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    NormalScaler | None: Fitted scaler class/object instance.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = ('train( self, *args ) -> NormalScaler | None')
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform with NormalScaler.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted preprocessing
		    estimator and stores the transformed output on the class/object for later inspection or
		    reuse.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = ('transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with NormalScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix,
		    transforms the same matrix immediately, and stores the transformed output on the
		    class/object.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = ('train_transform( self, *args ) -> np.ndarray')
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert NormalScaler transformation.
		
		Purpose:
		    Maps transformed feature values back toward their original scale using the inverse
		    transformation provided by the underlying preprocessing estimator.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		
		Returns:
		    np.ndarray: Feature matrix reconstructed in the original input scale.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails.
		    NotImplementedError: Raised when the `inverse_transform` operation cannot complete."""
		try:
			throw_if( 'X', X )
			raise NotImplementedError( 'Normalizer does not support inverse_transform( X ).' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Normalizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class MaxAbsScaler( Scaler ):
	"""Provide Max Absolute value feature scaling.
	
	Purpose:
	    Scales each feature by its maximum absolute value while preserving sparsity and
	    sign. The class/object stores the estimator and cached transformed matrix while
	    preserving a uniform Mathy preprocessing API.
	
	Attributes:
	    model (pp.MaxAbsScaler): Underlying preprocessing estimator used by the
	                             class/object.
	    transformed_data (Optional[np.ndarray]): Most recent transformed feature matrix
	                                             produced by the class/object."""
	model: pp.MaxAbsScaler
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize MaxAbsScaler.
		
		Purpose:
		    Initializes the MaxAbsScaler class/object by configuring the underlying sklearn
		    preprocessing estimator and preparing the transformed-data cache used by later
		    transformation methods.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.model = pp.MaxAbsScaler( )
		self.transformed_data = None
	
	def __dir__( self ) -> list[ str ]:
		"""List public members.
		
		Purpose:
		    Returns the stable member names exposed by the MaxAbsScaler class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    list[str]: Public member names exposed by the class/object."""
		return [ 'scaler', 'transformed_data', 'train', 'transform', 'train_transform',
			'inverse_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> MaxAbsScaler | None:
		"""Fit MaxAbsScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix
		    and returns the current class/object so callers can chain preprocessing operations
		    through the Mathy interface.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    MaxAbsScaler | None: Fitted scaler class/object instance.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MaxAbsScaler'
			exception.method = 'train( self, *args ) -> MaxAbsScaler | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Transform with MaxAbsScaler.
		
		Purpose:
		    Transforms the supplied feature matrix with the fitted preprocessing
		    estimator and stores the transformed output on the class/object for later inspection or
		    reuse.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced by the fitted scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MaxAbsScaler'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with MaxAbsScaler.
		
		Purpose:
		    Fits the underlying preprocessing estimator to the supplied feature matrix,
		    transforms the same matrix immediately, and stores the transformed output on the
		    class/object.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		    y (Optional[np.ndarray]): Optional target array accepted for estimator API
		                              compatibility and ignored by scaler implementations.
		
		Returns:
		    np.ndarray: Transformed feature matrix produced after fitting the scaler.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MaxAbsScaler'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert MaxAbsScaler transformation.
		
		Purpose:
		    Maps transformed feature values back toward their original scale using the inverse
		    transformation provided by the underlying preprocessing estimator.
		
		Args:
		    X (np.ndarray): Feature matrix supplied to the underlying preprocessing
		                    estimator.
		
		Returns:
		    np.ndarray: Feature matrix reconstructed in the original input scale.
		
		Raises:
		    Error: Raised when validation, preprocessing, or wrapped scaler execution
		           fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.inverse_transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MaxAbsScaler'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
