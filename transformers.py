"""******************************************************************************************
  Assembly:                mathy
  Filename:                transformers.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="transformers.py" company="Terry D. Eppler">

     transformers.py
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
    Provides preprocessing and feature-extraction wrappers for Mathy data workflows.
    The module standardizes binary transformation, label binarization, multilabel
    binarization, TF-IDF transformation, column transformation, text vectorization,
    dictionary vectorization, hashing vectorization, and feature hashing behind a
    consistent train, transform, train-transform, and inverse-transform interface.
</summary>
******************************************************************************************"""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import sklearn.feature_extraction as fe
import sklearn.feature_extraction.text as sk
import sklearn.preprocessing as pp
import sklearn.compose as sc
from boogr import Error, Logger

def throw_if( name: str, value: object ) -> None:
	"""Validate a required argument.
	
	Purpose:
	    Validates that a required argument contains a usable value before the surrounding
	    workflow continues.
	
	Args:
	    name (str): Argument name used in the validation error message.
	    value (object): Argument value checked by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a
	          value.
	
	Raises:
	    ValueError: Raised when the `throw_if` operation cannot complete."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, np.ndarray ) and value.size == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set, str) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Transformer( ):
	"""Transformer.
	
	Purpose:
	    Defines the base preprocessing contract for Mathy transformer wrappers. The
	    interface standardizes fit, transform, fit-transform, and inverse-transform
	    operations while retaining the most recent transformed output for downstream feature
	    engineering workflows.
	
	Attributes:
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self ) -> None:
		"""Initialize Transformer.
		
		Purpose:
		    Initializes the transformer with configured objects, runtime options, and
		    transformed- output cache required by later preprocessing operations.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		self.transformed_data = None
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> object | None:
		"""Fit Transformer.
		
		Purpose:
		    Fits the underlying transformer to supplied input data and returns the class/object for
		    chained preprocessing workflows.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    object | None: Fitted class/object instance.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError( )
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform with Transformer.
		
		Purpose:
		    Transforms supplied input data with the fitted transformer and caches the
		    transformed output for downstream workflow steps.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError( )
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with Transformer.
		
		Purpose:
		    Fits the underlying transformer and immediately transforms the supplied input data
		    in one operation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError( )
	
	def inverse_transform( self, X: np.ndarray ) -> np.ndarray:
		"""Invert Transformer output.
		
		Purpose:
		    Maps transformed output back toward the source representation when the wrapped
		    transformer supports inverse transformation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    np.ndarray: Decoded or reconstructed output produced by the fitted class/object.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError( )

class Binarizer( Transformer ):
	"""Binarizer.
	
	Purpose:
	    Converts numeric feature values into binary indicators using a configured threshold.
	    The class/object delegates to sklearn.preprocessing.Binarizer and caches the transformed
	    matrix for downstream preprocessing and modeling steps.
	
	Attributes:
	    model (pp.Binarizer): Underlying estimator or transformer used by the
	                          class/object.
	    threshold (Optional[float]): Numeric cutoff used to convert values into binary
	                                 indicators.
	    copy (Optional[bool]): Flag controlling whether binarization copies input data
	                           before transformation.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: pp.Binarizer
	threshold: Optional[ float ]
	copy: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, threshold: float = 0.0, copy: bool = True ) -> None:
		"""Initialize Binarizer.
		
		Purpose:
		    Initializes the binary-threshold transformer with configured objects,
		    runtime options, and transformed-output cache required by later preprocessing
		    operations.
		
		Args:
		    threshold (float): Threshold used to convert numeric values into binary indicators.
		    copy (bool): Flag controlling whether the binarizer copies input data before
		                 transformation.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.threshold = threshold
		self.copy = copy
		self.model = pp.Binarizer( threshold=self.threshold, copy=self.copy )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'threshold', 'copy', 'model', 'transformed_data', 'train', 'transform',
			'train_transform' ]
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> Binarizer | None:
		"""Fit Binarizer.
		
		Purpose:
		    Fits the underlying binary-threshold transformer to supplied input data and returns
		    the class/object for chained preprocessing workflows.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    Binarizer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Binarizer'
			exception.method = 'train( self, *args ) -> Binarizer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform with Binarizer.
		
		Purpose:
		    Transforms supplied input data with the fitted binary-threshold transformer and
		    caches the transformed output for downstream workflow steps.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Binarizer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with Binarizer.
		
		Purpose:
		    Fits the underlying binary-threshold transformer and immediately transforms the
		    supplied input data in one operation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X, y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'Binarizer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class LabelBinarizer( Transformer ):
	"""LabelBinarizer.
	
	Purpose:
	    Converts single-label target vectors into one-vs-rest binary indicator matrices. The
	    class/object exposes fitted class metadata, target-type metadata, inverse decoding, and
	    cached transformed output.
	
	Attributes:
	    model (pp.LabelBinarizer): Underlying estimator or transformer used by the
	                               class/object.
	    pos_label (Optional[int]): Positive indicator value assigned during label
	                               binarization.
	    neg_label (Optional[int]): Negative indicator value assigned during label
	                               binarization.
	    sparse_output (Optional[bool]): Flag controlling sparse output for label or
	                                    multilabel binarization.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: pp.LabelBinarizer
	pos_label: Optional[ int ]
	neg_label: Optional[ int ]
	sparse_output: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, pos_label: int = 1, neg_label: int = 0,
			sparse_output: bool = False ) -> None:
		"""Initialize LabelBinarizer.
		
		Purpose:
		    Initializes the label-binarizer transformer with configured objects, runtime
		    options, and transformed-output cache required by later preprocessing operations.
		
		Args:
		    pos_label (int): Positive indicator value assigned to present labels.
		    neg_label (int): Negative indicator value assigned to absent labels.
		    sparse_output (bool): Flag controlling sparse output from the binarizer.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.pos_label = pos_label
		self.neg_label = neg_label
		self.sparse_output = sparse_output
		self.model = pp.LabelBinarizer( neg_label=self.neg_label,
			pos_label=self.pos_label, sparse_output=self.sparse_output )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'pos_label', 'neg_label', 'sparse_output', 'model', 'transformed_data', 'classes',
			'types', 'train', 'transform', 'train_transform', 'inverse_transform' ]
	
	@property
	def classes( self ) -> List[ str ]:
		"""Return classes.
		
		Purpose:
		    Returns fitted class metadata learned by the wrapped transformer for
		    inspection and compatibility checks.
		
		Returns:
		    List[str]: Fitted class labels or category metadata.
		
		Raises:
		    AttributeError: Raised when fitted metadata is unavailable."""
		if getattr( self.model, 'classes_', None ) is None:
			raise AttributeError( 'LabelBinarizer has not been initialized.' )
		else:
			return self.model.classes_
	
	@property
	def types( self ) -> str:
		"""Return types.
		
		Purpose:
		    Returns fitted target-type metadata learned by the wrapped label binarizer for
		    inspection and compatibility checks.
		
		Returns:
		    str: Fitted label target type.
		
		Raises:
		    AttributeError: Raised when fitted metadata is unavailable."""
		if getattr( self.model, 'y_type_', None ) is None:
			raise AttributeError( 'LabelBinarizer has not been initialized.' )
		else:
			return self.model.y_type_
	
	def train( self, y: np.ndarray ) -> LabelBinarizer | None:
		"""Fit LabelBinarizer.
		
		Purpose:
		    Fits the underlying label-binarizer transformer to supplied input data and returns
		    the class/object for chained preprocessing workflows.
		
		Args:
		    y (np.ndarray): Target vector or label collection aligned to the input data.
		
		Returns:
		    LabelBinarizer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'y', y )
			self.model.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'train( self, y: np.ndarray ) -> LabelBinarizer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, y: np.ndarray ) -> np.ndarray:
		"""Transform with LabelBinarizer.
		
		Purpose:
		    Transforms supplied input data with the fitted label-binarizer transformer and
		    caches the transformed output for downstream workflow steps.
		
		Args:
		    y (np.ndarray): Target vector or label collection aligned to the input data.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'transform( self, y: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, y: np.ndarray ) -> np.ndarray:
		"""Fit and transform with LabelBinarizer.
		
		Purpose:
		    Fits the underlying label-binarizer transformer and immediately transforms the
		    supplied input data in one operation.
		
		Args:
		    y (np.ndarray): Target vector or label collection aligned to the input data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'train_transform( self, y: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, Y: np.ndarray ) -> np.ndarray:
		"""Invert LabelBinarizer output.
		
		Purpose:
		    Maps transformed output back toward the source representation when the wrapped
		    label-binarizer transformer supports inverse transformation.
		
		Args:
		    Y (np.ndarray): Encoded indicator matrix or transformed feature matrix passed to
		                    inverse transformation.
		
		Returns:
		    np.ndarray: Decoded or reconstructed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'Y', Y )
			return self.model.inverse_transform( Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'LabelBinarizer'
			exception.method = 'inverse_transform( self, Y: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class MultiLabelBinarizer( Transformer ):
	"""MultiLabelBinarizer.
	
	Purpose:
	    Converts iterable multilabel targets into binary indicator matrices with stable
	    class ordering. The class/object supports dense or sparse output, inverse decoding, and
	    cached transformed output for multilabel modeling workflows.
	
	Attributes:
	    model (pp.MultiLabelBinarizer): Underlying estimator or transformer used by
	                                    the class/object.
	    sparse_output (Optional[bool]): Flag controlling sparse output for label or
	                                    multilabel binarization.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: pp.MultiLabelBinarizer
	sparse_output: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, classes: Optional[ np.ndarray ]=None,
			sparse_output: bool = False ) -> None:
		"""Initialize MultiLabelBinarizer.
		
		Purpose:
		    Initializes the multilabel-binarizer transformer with configured objects,
		    runtime options, and transformed-output cache required by later preprocessing
		    operations.
		
		Args:
		    classes (Optional[np.ndarray]): Optional fixed class ordering used by the multilabel
		                                    binarizer.
		    sparse_output (bool): Flag controlling sparse output from the binarizer.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.sparse_output = sparse_output
		self.model = pp.MultiLabelBinarizer( classes=classes,
			sparse_output=self.sparse_output )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'sparse_output', 'model', 'transformed_data', 'classes', 'train', 'transform',
			'train_transform', 'inverse_transform' ]
	
	@property
	def classes( self ) -> List[ str ]:
		"""Return classes.
		
		Purpose:
		    Returns fitted class metadata learned by the wrapped transformer for
		    inspection and compatibility checks.
		
		Returns:
		    List[str]: Fitted class labels or category metadata.
		
		Raises:
		    AttributeError: Raised when fitted metadata is unavailable."""
		if getattr( self.model, 'classes_', None ) is None:
			raise AttributeError( 'MultiLabelBinarizer has not been initialized.' )
		else:
			return self.model.classes_
	
	def train( self, y: np.ndarray ) -> MultiLabelBinarizer | None:
		"""Fit MultiLabelBinarizer.
		
		Purpose:
		    Fits the underlying multilabel-binarizer transformer to supplied input data and
		    returns the class/object for chained preprocessing workflows.
		
		Args:
		    y (np.ndarray): Target vector or label collection aligned to the input data.
		
		Returns:
		    MultiLabelBinarizer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'y', y )
			self.model.fit( y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'train( self, y: np.ndarray ) -> MultiLabelBinarizer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, y: np.ndarray ) -> np.ndarray:
		"""Transform with MultiLabelBinarizer.
		
		Purpose:
		    Transforms supplied input data with the fitted multilabel-binarizer transformer and
		    caches the transformed output for downstream workflow steps.
		
		Args:
		    y (np.ndarray): Target vector or label collection aligned to the input data.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'transform( self, y: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, y: np.ndarray ) -> np.ndarray:
		"""Fit and transform with MultiLabelBinarizer.
		
		Purpose:
		    Fits the underlying multilabel-binarizer transformer and immediately transforms the
		    supplied input data in one operation.
		
		Args:
		    y (np.ndarray): Target vector or label collection aligned to the input data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'y', y )
			self.transformed_data = self.model.fit_transform( y )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'train_transform( self, y: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, Y: np.ndarray ) -> np.ndarray:
		"""Invert MultiLabelBinarizer output.
		
		Purpose:
		    Maps transformed output back toward the source representation when the wrapped
		    multilabel- binarizer transformer supports inverse transformation.
		
		Args:
		    Y (np.ndarray): Encoded indicator matrix or transformed feature matrix passed to
		                    inverse transformation.
		
		Returns:
		    np.ndarray: Decoded or reconstructed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'Y', Y )
			return self.model.inverse_transform( Y )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'MultiLabelBinarizer'
			exception.method = 'inverse_transform( self, Y: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class TfidfTransformer( Transformer ):
	"""TfidfTransformer.
	
	Purpose:
	    Transforms token-count matrices into normalized term-frequency
	    inverse-document-frequency representations. The class/object exposes fitted IDF and
	    feature-count metadata while preserving a consistent transformer API.
	
	Attributes:
	    model (sk.TfidfTransformer): Underlying estimator or transformer used by the
	                                 class/object.
	    norm (Optional[str]): Normalization strategy applied to transformed vectors.
	    use_idf (Optional[bool]): Flag controlling inverse-document-frequency weighting.
	    smooth_idf (Optional[bool]): Flag controlling IDF smoothing.
	    sublinear_tf (Optional[bool]): Flag controlling logarithmic term-frequency scaling.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: sk.TfidfTransformer
	norm: Optional[ str ]
	use_idf: Optional[ bool ]
	smooth_idf: Optional[ bool ]
	sublinear_tf: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, norm: str='l2', use_idf: bool=True,
			smooth_idf: bool=True, sublinear_tf: bool=False ) -> None:
		"""Initialize TfidfTransformer.
		
		Purpose:
		    Initializes the TF-IDF transformer with configured objects, runtime options,
		    and transformed-output cache required by later preprocessing operations.
		
		Args:
		    norm (str): Normalization strategy applied to output vectors.
		    use_idf (bool): Flag controlling inverse-document-frequency weighting.
		    smooth_idf (bool): Flag controlling IDF smoothing.
		    sublinear_tf (bool): Flag controlling logarithmic term-frequency scaling.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.norm = norm
		self.use_idf = use_idf
		self.smooth_idf = smooth_idf
		self.sublinear_tf = sublinear_tf
		self.model = sk.TfidfTransformer( norm=self.norm, use_idf=self.use_idf,
			smooth_idf=self.smooth_idf, sublinear_tf=self.sublinear_tf )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'norm', 'use_idf', 'smooth_idf', 'sublinear_tf', 'model', 'transformed_data',
			'idf_vector', 'features', 'train', 'transform', 'train_transform' ]
	
	@property
	def idf_vector( self ) -> np.ndarray:
		"""Return idf vector.
		
		Purpose:
		    Returns fitted inverse-document-frequency weights learned by the wrapped TF-IDF
		    transformer or vectorizer.
		
		Returns:
		    np.ndarray: Fitted inverse-document-frequency vector.
		
		Raises:
		    AttributeError: Raised when fitted metadata is unavailable."""
		if getattr( self.model, 'idf_', None ) is None:
			raise AttributeError( 'TfidfTransformer must be initialized.' )
		else:
			return self.model.idf_
	
	@property
	def features( self ) -> int:
		"""Return features.
		
		Purpose:
		    Returns the fitted input feature count recorded by the wrapped transformer.
		
		Returns:
		    int: Number of input features observed during fitting.
		
		Raises:
		    AttributeError: Raised when fitted metadata is unavailable."""
		if getattr( self.model, 'n_features_in_', None ) is None:
			raise AttributeError( 'TfidfTransformer must be initialized.' )
		else:
			return self.model.n_features_in_
	
	def train( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> TfidfTransformer | None:
		"""Fit TfidfTransformer.
		
		Purpose:
		    Fits the underlying TF-IDF transformer to supplied input data and returns the
		    class/object for chained preprocessing workflows.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    TfidfTransformer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'train( self, *args ) -> TfidfTransformer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform with TfidfTransformer.
		
		Purpose:
		    Transforms supplied input data with the fitted TF-IDF transformer and caches the
		    transformed output for downstream workflow steps.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.transform( X, copy=True ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray, y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with TfidfTransformer.
		
		Purpose:
		    Fits the underlying TF-IDF transformer and immediately transforms the supplied input
		    data in one operation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.transformed_data = self.model.fit_transform( X, y ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfTransformer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class ColumnTransformer( Transformer ):
	"""ColumnTransformer.
	
	Purpose:
	    Applies named transformers to selected input columns and coordinates passthrough,
	    dropping, sparse-threshold behavior, parallel execution, and optional transformer
	    weights through sklearn.compose.ColumnTransformer.
	
	Attributes:
	    model (sc.ColumnTransformer): Underlying estimator or transformer used by
	                                  the class/object.
	    transformers (Optional[List[Tuple[str, object, List[str]]]]): Named transformer
	                                                                  definitions passed to
	                                                                  sklearn
	                                                                  ColumnTransformer.
	    remainder (Optional[str]): Policy for columns not explicitly assigned to a
	                               transformer.
	    transformer_weights (Optional[Dict[str, float]]): Optional per-transformer weighting
	                                                      applied to transformed outputs.
	    sparse_threshold (Optional[float]): Density threshold controlling sparse combined
	                                        output.
	    n_jobs (Optional[int]): Parallel worker count used by during fitting or
	                            transformation.
	    verbose (Optional[bool]): Flag controlling verbose execution output.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: sc.ColumnTransformer
	transformers: Optional[ List[ Tuple[ str, object, List[ str ] ] ] ]
	remainder: Optional[ str ]
	transformer_weights: Optional[ Dict[ str, float ] ]
	sparse_threshold: Optional[ float ]
	n_jobs: Optional[ int ]
	verbose: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, transformers: List[ Tuple[ str, object, List[ str ] ] ],
			remainder: str='drop', sparse_threshold: float=0.3,
			n_jobs: Optional[ int ]=None,
			transformer_weights: Optional[ Dict[ str, float ] ]=None,
			verbose: bool=False ) -> None:
		"""Initialize ColumnTransformer.
		
		Purpose:
		    Initializes the column transformer with configured objects, runtime options,
		    and transformed-output cache required by later preprocessing operations.
		
		Args:
		    transformers (List[Tuple[str, object, List[str]]]): Named transformer definitions
		                                                        containing name, transformer,
		                                                        and column selections.
		    remainder (str): Policy applied to columns not listed in `transformers`.
		    sparse_threshold (float): Density threshold controlling sparse combined output.
		    n_jobs (Optional[int]): Parallel worker count used during operations.
		    transformer_weights (Optional[Dict[str, float]]): Optional per-transformer weights
		                                                      applied to transformed outputs.
		    verbose (bool): Flag controlling verbose execution output.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.transformers = transformers
		self.remainder = remainder
		self.sparse_threshold = sparse_threshold
		self.n_jobs = n_jobs
		self.transformer_weights = transformer_weights
		self.verbose = verbose
		self.model = sc.ColumnTransformer( transformers=self.transformers,
			remainder=self.remainder,
			sparse_threshold=self.sparse_threshold, n_jobs=self.n_jobs,
			transformer_weights=self.transformer_weights, verbose=self.verbose )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'transformers', 'remainder', 'transformer_weights', 'sparse_threshold',
			'n_jobs', 'verbose', 'transformed_data', 'train', 'transform', 'train_transform' ]
	
	def train( self, X: np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> ColumnTransformer | None:
		"""Fit ColumnTransformer.
		
		Purpose:
		    Fits the underlying column transformer to supplied input data and returns the
		    class/object for chained preprocessing workflows.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    ColumnTransformer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'train( self, *args ) -> ColumnTransformer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: np.ndarray ) -> np.ndarray:
		"""Transform with ColumnTransformer.
		
		Purpose:
		    Transforms supplied input data with the fitted column transformer and caches the
		    transformed output for downstream workflow steps.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			result = self.model.transform( X )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'transform( self, X: np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with ColumnTransformer.
		
		Purpose:
		    Fits the underlying column transformer and immediately transforms the supplied input
		    data in one operation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			result = self.model.fit_transform( X, y )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'ColumnTransformer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception

class TfidfVectorizer( Transformer ):
	"""TfidfVectorizer.
	
	Purpose:
	    Converts raw text documents into TF-IDF weighted token-feature matrices. The class/object
	    centralizes text decoding, tokenization, analyzer, vocabulary, n-gram,
	    document-frequency, normalization, and IDF configuration.
	
	Attributes:
	    model (sk.TfidfVectorizer): Underlying estimator or transformer used by the
	                                class/object.
	    input (Optional[str]): Input source mode used by text vectorizers.
	    encoding (Optional[str]): Character encoding used when decoding text input.
	    decode_error (Optional[str]): Error-handling policy used during text decoding.
	    strip_accents (Optional[Any]): Accent stripping configuration used during text
	                                   preprocessing.
	    lowercase (Optional[bool]): Flag controlling lowercase normalization before
	                                tokenization.
	    preprocessor (Optional[Any]): Optional callable applied before tokenization.
	    tokenizer (Optional[Any]): Optional callable used to tokenize preprocessed text.
	    analyzer (Optional[str | Any]): Analyzer mode or callable used to extract features.
	    stop_words (Optional[Any]): Stop-word configuration used during token extraction.
	    token_pattern (Optional[str]): Regular expression defining token boundaries for word
	                                   analyzers.
	    ngram_range (Optional[Tuple[int, int]]): Inclusive lower and upper n-gram lengths
	                                             extracted from text.
	    max_df (Optional[float | int]): Upper document-frequency threshold used to filter
	                                    terms.
	    min_df (Optional[float | int]): Lower document-frequency threshold used to filter
	                                    terms.
	    max_features (Optional[int | None]): Maximum number of retained vocabulary features.
	    vocabulary (Optional[Dict[str, int] | List[str] | None]): Fixed vocabulary or
	                                                              learned vocabulary mapping
	                                                              used by vectorizers.
	    binary (Optional[bool]): Flag controlling binary occurrence counts instead of
	                             integer counts.
	    dtype (Optional[Any]): Numeric dtype used for transformed feature matrices.
	    norm (Optional[str | None]): Normalization strategy applied to transformed vectors.
	    use_idf (Optional[bool]): Flag controlling inverse-document-frequency weighting.
	    smooth_idf (Optional[bool]): Flag controlling IDF smoothing.
	    sublinear_tf (Optional[bool]): Flag controlling logarithmic term-frequency scaling.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: sk.TfidfVectorizer
	input: Optional[ str ]
	encoding: Optional[ str ]
	decode_error: Optional[ str ]
	strip_accents: Optional[ Any ]
	lowercase: Optional[ bool ]
	preprocessor: Optional[ Any ]
	tokenizer: Optional[ Any ]
	analyzer: Optional[ str | Any ]
	stop_words: Optional[ Any ]
	token_pattern: Optional[ str ]
	ngram_range: Optional[ Tuple[ int, int ] ]
	max_df: Optional[ float | int ]
	min_df: Optional[ float | int ]
	max_features: Optional[ int | None ]
	vocabulary: Optional[ Dict[ str, int ] | List[ str ] | None ]
	binary: Optional[ bool ]
	dtype: Optional[ Any ]
	norm: Optional[ str | None ]
	use_idf: Optional[ bool ]
	smooth_idf: Optional[ bool ]
	sublinear_tf: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, input: str='content', encoding: str='utf-8',
			decode_error: str='strict', strip_accents: Any=None,
			lowercase: bool=True, preprocessor: Any=None, tokenizer: Any=None,
			analyzer: str | Any='word', stop_words: Any=None,
			token_pattern: str=r'(?u)\b\w\w+\b',
			ngram_range: Tuple[ int, int ]=(1, 1),
			max_df: float | int=1.0, min_df: float | int=1,
			max_features: int | None=None,
			vocabulary: Dict[ str, int ] | List[ str ] | None=None,
			binary: bool=False, dtype: Any=np.float64, norm: str | None='l2',
			use_idf: bool=True, smooth_idf: bool=True,
			sublinear_tf: bool=False ) -> None:
		"""Initialize TfidfVectorizer.
		
		Purpose:
		    Initializes the TF-IDF vectorizer with configured objects, runtime options,
		    and transformed-output cache required by later preprocessing operations.
		
		Args:
		    input (str): Input source mode used by text vectorizers.
		    encoding (str): Character encoding used when decoding text input.
		    decode_error (str): Error-handling policy used during text decoding.
		    strip_accents (Any): Accent stripping configuration used during preprocessing.
		    lowercase (bool): Flag controlling lowercase normalization.
		    preprocessor (Any): Optional callable applied before tokenization.
		    tokenizer (Any): Optional callable used to tokenize text.
		    analyzer (str | Any): Analyzer mode or callable used to extract features.
		    stop_words (Any): Stop-word configuration used during token extraction.
		    token_pattern (str): Regular expression used to identify tokens.
		    ngram_range (Tuple[int, int]): Inclusive lower and upper n-gram lengths.
		    max_df (float | int): Upper document-frequency threshold used to filter vocabulary
		                          terms.
		    min_df (float | int): Lower document-frequency threshold used to filter vocabulary
		                          terms.
		    max_features (int | None): Maximum number of retained vocabulary features.
		    vocabulary (Dict[str, int] | List[str] | None): Fixed vocabulary mapping or iterable
		                                                    vocabulary supplied to the
		                                                    vectorizer.
		    binary (bool): Flag controlling binary occurrence counts instead of counts.
		    dtype (Any): Numeric dtype used for transformed outputs.
		    norm (str | None): Normalization strategy applied to output vectors.
		    use_idf (bool): Flag controlling inverse-document-frequency weighting.
		    smooth_idf (bool): Flag controlling IDF smoothing.
		    sublinear_tf (bool): Flag controlling logarithmic term-frequency scaling.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.input = input
		self.encoding = encoding
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.lowercase = lowercase
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.stop_words = stop_words
		self.token_pattern = token_pattern
		self.ngram_range = ngram_range
		self.max_df = max_df
		self.min_df = min_df
		self.max_features = max_features
		self.vocabulary = vocabulary
		self.binary = binary
		self.dtype = dtype
		self.norm = norm
		self.use_idf = use_idf
		self.smooth_idf = smooth_idf
		self.sublinear_tf = sublinear_tf
		self.model = sk.TfidfVectorizer( input=self.input, encoding=self.encoding,
			decode_error=self.decode_error, strip_accents=self.strip_accents,
			lowercase=self.lowercase, preprocessor=self.preprocessor, tokenizer=self.tokenizer,
			analyzer=self.analyzer, stop_words=self.stop_words, token_pattern=self.token_pattern,
			ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df,
			max_features=self.max_features, vocabulary=self.vocabulary, binary=self.binary,
			dtype=self.dtype, norm=self.norm, use_idf=self.use_idf, smooth_idf=self.smooth_idf,
			sublinear_tf=self.sublinear_tf )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'input', 'encoding', 'decode_error', 'strip_accents', 'lowercase',
			'preprocessor', 'tokenizer', 'analyzer', 'stop_words', 'token_pattern', 'ngram_range',
			'max_df', 'min_df', 'max_features', 'vocabulary', 'binary', 'dtype', 'norm', 'use_idf',
			'smooth_idf', 'sublinear_tf', 'transformed_data', 'idf_vector', 'train', 'transform',
			'train_transform', 'inverse_transform' ]
	
	@property
	def idf_vector( self ) -> np.ndarray:
		"""Return idf vector.
		
		Purpose:
		    Returns fitted inverse-document-frequency weights learned by the wrapped TF-IDF
		    transformer or vectorizer.
		
		Returns:
		    np.ndarray: Fitted inverse-document-frequency vector.
		
		Raises:
		    AttributeError: Raised when fitted metadata is unavailable."""
		if getattr( self.model, 'idf_', None ) is None:
			raise AttributeError( 'TfidfVectorizer must be initialized.' )
		else:
			return self.model.idf_
	
	def train( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> TfidfVectorizer | None:
		"""Fit TfidfVectorizer.
		
		Purpose:
		    Fits the underlying TF-IDF vectorizer to supplied input data and returns the class/object
		    for chained preprocessing workflows.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    TfidfVectorizer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			self.model.fit( text, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'train( self, *args ) -> TfidfVectorizer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray:
		"""Transform with TfidfVectorizer.
		
		Purpose:
		    Transforms supplied input data with the fitted TF-IDF vectorizer and caches the
		    transformed output for downstream workflow steps.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'transform( self, text: List[str] | np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with TfidfVectorizer.
		
		Purpose:
		    Fits the underlying TF-IDF vectorizer and immediately transforms the supplied input
		    data in one operation.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.fit_transform( text, y ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> List[ np.ndarray ] | None:
		"""Invert TfidfVectorizer output.
		
		Purpose:
		    Maps transformed output back toward the source representation when the wrapped
		    TF-IDF vectorizer supports inverse transformation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    List[np.ndarray] | None: Decoded or reconstructed output produced by the fitted
		                             class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'TfidfVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[np.ndarray] | None'
			Logger( ).write( exception )
			raise exception

class CountVectorizer( Transformer ):
	"""CountVectorizer.
	
	Purpose:
	    Converts raw text documents into token-count feature matrices. The class/object
	    centralizes text decoding, tokenization, analyzer, vocabulary, n-gram,
	    document-frequency, binary-count, and dtype configuration.
	
	Attributes:
	    model (sk.CountVectorizer): Underlying estimator or transformer used by the
	                                class/object.
	    input (Optional[str]): Input source mode used by text vectorizers.
	    encoding (Optional[str]): Character encoding used when decoding text input.
	    decode_error (Optional[str]): Error-handling policy used during text decoding.
	    strip_accents (Optional[Any]): Accent stripping configuration used during text
	                                   preprocessing.
	    lowercase (Optional[bool]): Flag controlling lowercase normalization before
	                                tokenization.
	    preprocessor (Optional[Any]): Optional callable applied before tokenization.
	    tokenizer (Optional[Any]): Optional callable used to tokenize preprocessed text.
	    analyzer (Optional[str | Any]): Analyzer mode or callable used to extract features.
	    stop_words (Optional[Any]): Stop-word configuration used during token extraction.
	    token_pattern (Optional[str]): Regular expression defining token boundaries for word
	                                   analyzers.
	    ngram_range (Optional[Tuple[int, int]]): Inclusive lower and upper n-gram lengths
	                                             extracted from text.
	    max_df (Optional[float | int]): Upper document-frequency threshold used to filter
	                                    terms.
	    min_df (Optional[float | int]): Lower document-frequency threshold used to filter
	                                    terms.
	    max_features (Optional[int | None]): Maximum number of retained vocabulary features.
	    vocabulary (Optional[Dict[str, int] | List[str] | None]): Fixed vocabulary or
	                                                              learned vocabulary mapping
	                                                              used by vectorizers.
	    binary (Optional[bool]): Flag controlling binary occurrence counts instead of
	                             integer counts.
	    dtype (Optional[Any]): Numeric dtype used for transformed feature matrices.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: sk.CountVectorizer
	input: Optional[ str ]
	encoding: Optional[ str ]
	decode_error: Optional[ str ]
	strip_accents: Optional[ Any ]
	lowercase: Optional[ bool ]
	preprocessor: Optional[ Any ]
	tokenizer: Optional[ Any ]
	analyzer: Optional[ str | Any ]
	stop_words: Optional[ Any ]
	token_pattern: Optional[ str ]
	ngram_range: Optional[ Tuple[ int, int ] ]
	max_df: Optional[ float | int ]
	min_df: Optional[ float | int ]
	max_features: Optional[ int | None ]
	vocabulary: Optional[ Dict[ str, int ] | List[ str ] | None ]
	binary: Optional[ bool ]
	dtype: Optional[ Any ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, input: str='content', encoding: str='utf-8', decode_error: str='strict',
		strip_accents: Any=None, lowercase: bool=True, preprocessor: Any=None, tokenizer: Any=None,
		analyzer: str | Any='word', stop_words: Any=None, token_pattern: str=r'(?u)\b\w\w+\b',
		ngram_range: Tuple[ int, int ]=(1, 1), max_df: float | int=1.0, min_df: float | int=1,
		max_features: int | None=None, vocabulary: Dict[ str, int ] | List[ str ] | None=None,
		binary: bool=False, dtype: Any=np.int64 ) -> None:
		"""Initialize CountVectorizer.
		
		Purpose:
		    Initializes the count vectorizer with configured objects, runtime options,
		    and transformed-output cache required by later preprocessing operations.
		
		Args:
		    input (str): Input source mode used by text vectorizers.
		    encoding (str): Character encoding used when decoding text input.
		    decode_error (str): Error-handling policy used during text decoding.
		    strip_accents (Any): Accent stripping configuration used during preprocessing.
		    lowercase (bool): Flag controlling lowercase normalization.
		    preprocessor (Any): Optional callable applied before tokenization.
		    tokenizer (Any): Optional callable used to tokenize text.
		    analyzer (str | Any): Analyzer mode or callable used to extract features.
		    stop_words (Any): Stop-word configuration used during token extraction.
		    token_pattern (str): Regular expression used to identify tokens.
		    ngram_range (Tuple[int, int]): Inclusive lower and upper n-gram lengths.
		    max_df (float | int): Upper document-frequency threshold used to filter vocabulary
		                          terms.
		    min_df (float | int): Lower document-frequency threshold used to filter vocabulary
		                          terms.
		    max_features (int | None): Maximum number of retained vocabulary features.
		    vocabulary (Dict[str, int] | List[str] | None): Fixed vocabulary mapping or iterable
		                                                    vocabulary supplied to the
		                                                    vectorizer.
		    binary (bool): Flag controlling binary occurrence counts instead of counts.
		    dtype (Any): Numeric dtype used for transformed outputs.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.input = input
		self.encoding = encoding
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.lowercase = lowercase
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.stop_words = stop_words
		self.token_pattern = token_pattern
		self.ngram_range = ngram_range
		self.max_df = max_df
		self.min_df = min_df
		self.max_features = max_features
		self.vocabulary = vocabulary
		self.binary = binary
		self.dtype = dtype
		self.model = sk.CountVectorizer( input=self.input, encoding=self.encoding,
			decode_error=self.decode_error, strip_accents=self.strip_accents,
			lowercase=self.lowercase, preprocessor=self.preprocessor, tokenizer=self.tokenizer,
			analyzer=self.analyzer, stop_words=self.stop_words, token_pattern=self.token_pattern,
			ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df,
			max_features=self.max_features, vocabulary=self.vocabulary, binary=self.binary,
			dtype=self.dtype )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'input', 'encoding', 'decode_error', 'strip_accents', 'lowercase',
			'preprocessor', 'tokenizer', 'analyzer', 'stop_words', 'token_pattern', 'ngram_range',
			'max_df', 'min_df', 'max_features', 'vocabulary', 'binary', 'dtype',
			'transformed_data',
			'train', 'transform', 'train_transform', 'inverse_transform' ]
	
	def train( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> CountVectorizer | None:
		"""Fit CountVectorizer.
		
		Purpose:
		    Fits the underlying count vectorizer to supplied input data and returns the class/object
		    for chained preprocessing workflows.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    CountVectorizer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			self.model.fit( text, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'train( self, *args ) -> CountVectorizer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray:
		"""Transform with CountVectorizer.
		
		Purpose:
		    Transforms supplied input data with the fitted count vectorizer and caches the
		    transformed output for downstream workflow steps.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'transform( self, text: List[str] | np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with CountVectorizer.
		
		Purpose:
		    Fits the underlying count vectorizer and immediately transforms the supplied input
		    data in one operation.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.model.fit_transform( text, y ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> List[ np.ndarray ] | None:
		"""Invert CountVectorizer output.
		
		Purpose:
		    Maps transformed output back toward the source representation when the wrapped count
		    vectorizer supports inverse transformation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    List[np.ndarray] | None: Decoded or reconstructed output produced by the fitted
		                             class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'CountVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[np.ndarray] | None'
			Logger( ).write( exception )
			raise exception

class HashVectorizer( Transformer ):
	"""HashVectorizer.
	
	Purpose:
	    Convert a collection of text documents to a matrix of token occurrences. It turns a collection
	    of text documents into a scipy.sparse matrix holding token occurrence counts (or binary
	    occurrence information), possibly normalized as token frequencies if norm=’l1’ or projected
	    on the euclidean unit sphere if norm=’l2’.
	    
	    This text vectorizer implementation uses the hashing trick to find the token string name to
	    feature integer index mapping. This strategy has several advantages: 1) it is very low memory
	    scalable to large datasets as there is no need to store a vocabulary dictionary in memory.
	    2) it is fast to pickle and un-pickle as it holds no state besides the constructor parameters.


	
	Attributes:
	    vectorizer (sk.HashingVectorizer): Underlying hashing vectorizer used by the
	                                       class/object.
	    input (Optional[str]): Input source mode used by text vectorizers.
	    encoding (Optional[str]): Character encoding used when decoding text input.
	    decode_error (Optional[str]): Error-handling policy used during text decoding.
	    strip_accents (Optional[Any]): Accent stripping configuration used during text
	                                   preprocessing.
	    lowercase (Optional[bool]): Flag controlling lowercase normalization before
	                                tokenization.
	    preprocessor (Optional[Any]): Optional callable applied before tokenization.
	    tokenizer (Optional[Any]): Optional callable used to tokenize preprocessed text.
	    analyzer (Optional[str | Any]): Analyzer mode or callable used to extract features.
	    stop_words (Optional[Any]): Stop-word configuration used during token extraction.
	    token_pattern (Optional[str]): Regular expression defining token boundaries for word
	                                   analyzers.
	    ngram_range (Optional[Tuple[int, int]]): Inclusive lower and upper n-gram lengths
	                                             extracted from text.
	    binary (Optional[bool]): Flag controlling binary occurrence counts instead of
	                             integer counts.
	    norm (Optional[str | None]): Normalization strategy applied to transformed vectors.
	    alternate_sign (Optional[bool]): Flag controlling signed hashing behavior.
	    n_features (Optional[int]): Number of hashed output features.
	    dtype (Optional[Any]): Numeric dtype used for transformed feature matrices.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	vectorizer: sk.HashingVectorizer
	input: Optional[ str ]
	encoding: Optional[ str ]
	decode_error: Optional[ str ]
	strip_accents: Optional[ Any ]
	lowercase: Optional[ bool ]
	preprocessor: Optional[ Any ]
	tokenizer: Optional[ Any ]
	analyzer: Optional[ str | Any ]
	stop_words: Optional[ Any ]
	token_pattern: Optional[ str ]
	ngram_range: Optional[ Tuple[ int, int ] ]
	binary: Optional[ bool ]
	norm: Optional[ str | None ]
	alternate_sign: Optional[ bool ]
	n_features: Optional[ int ]
	dtype: Optional[ Any ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, num: int=1048576, input: str='content', encoding: str='utf-8',
		decode_error: str='strict', strip_accents: Any=None, lowercase: bool=True, preprocessor: Any=None,
		tokenizer: Any=None, analyzer: str | Any='word', stop_words: Any=None,
		token_pattern: str=r'(?u)\b\w\w+\b', ngram_range: Tuple[ int, int ]=(1, 1),
		binary: bool=False, norm: str | None='l2', alternate_sign: bool=True,
		dtype: Any=np.float64 ) -> None:
		"""Initialize HashVectorizer.
		
		Purpose:
		    Initializes the hashing vectorizer with configured objects, runtime options,
		    and transformed-output cache required by later preprocessing operations.
		
		Args:
		    num (int): Number of hashed output features used by HashingVectorizer.
		    input (str): Input source mode used by text vectorizers.
		    encoding (str): Character encoding used when decoding text input.
		    decode_error (str): Error-handling policy used during text decoding.
		    strip_accents (Any): Accent stripping configuration used during preprocessing.
		    lowercase (bool): Flag controlling lowercase normalization.
		    preprocessor (Any): Optional callable applied before tokenization.
		    tokenizer (Any): Optional callable used to tokenize text.
		    analyzer (str | Any): Analyzer mode or callable used to extract features.
		    stop_words (Any): Stop-word configuration used during token extraction.
		    token_pattern (str): Regular expression used to identify tokens.
		    ngram_range (Tuple[int, int]): Inclusive lower and upper n-gram lengths.
		    binary (bool): Flag controlling binary occurrence counts instead of counts.
		    norm (str | None): Normalization strategy applied to output vectors.
		    alternate_sign (bool): Flag controlling signed hashing behavior.
		    dtype (Any): Numeric dtype used for transformed outputs.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.input = input
		self.encoding = encoding
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.lowercase = lowercase
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.stop_words = stop_words
		self.token_pattern = token_pattern
		self.ngram_range = ngram_range
		self.binary = binary
		self.norm = norm
		self.alternate_sign = alternate_sign
		self.n_features = num
		self.dtype = dtype
		self.vectorizer = sk.HashingVectorizer( n_features=self.n_features, input=self.input,
			encoding=self.encoding, decode_error=self.decode_error,
			strip_accents=self.strip_accents, lowercase=self.lowercase,
			preprocessor=self.preprocessor, tokenizer=self.tokenizer, analyzer=self.analyzer,
			stop_words=self.stop_words, token_pattern=self.token_pattern,
			ngram_range=self.ngram_range, binary=self.binary, norm=self.norm,
			alternate_sign=self.alternate_sign, dtype=self.dtype )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'vectorizer', 'input', 'encoding', 'decode_error', 'strip_accents', 'lowercase',
			'preprocessor', 'tokenizer', 'analyzer', 'stop_words', 'token_pattern', 'ngram_range',
			'binary', 'norm', 'alternate_sign', 'n_features', 'dtype', 'transformed_data', 'train',
			'transform', 'train_transform', 'inverse_transform' ]
	
	def train( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> HashVectorizer | None:
		"""Fit HashVectorizer.
		
		Purpose:
		    Fits the underlying hashing vectorizer to supplied input data and returns the
		    class/object for chained preprocessing workflows.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    HashVectorizer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashVectorizer'
			exception.method = 'train( self, *args ) -> HashVectorizer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, text: List[ str ] | np.ndarray ) -> np.ndarray:
		"""Transform with HashVectorizer.
		
		Purpose:
		    Transforms supplied input data with the fitted hashing vectorizer and caches the
		    transformed output for downstream workflow steps.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.vectorizer.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashVectorizer'
			exception.method = 'transform( self, text: List[str] | np.ndarray ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, text: List[ str ] | np.ndarray,
			y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with HashVectorizer.
		
		Purpose:
		    Fits the underlying hashing vectorizer and immediately transforms the supplied input
		    data in one operation.
		
		Args:
		    text (List[str] | np.ndarray): Raw text documents supplied to the vectorizer.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'text', text )
			self.transformed_data = self.vectorizer.transform( text ).toarray( )
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'HashVectorizer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> None:
		"""Invert HashVectorizer output.
		
		Purpose:
		    Maps transformed output back toward the source representation when the wrapped
		    hashing vectorizer supports inverse transformation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError( 'HashingVectorizer does not support inverse_transform.' )

class DictVectorizer( Transformer ):
	"""DictVectorizer.
	
	Purpose:
	    Converts dictionaries of named feature values into numeric feature matrices. The
	    class/object exposes fitted feature names, vocabulary metadata, inverse decoding,
	    sparse-output behavior, and cached transformed output.
	
	Attributes:
	    model (fe.DictVectorizer): Underlying estimator or transformer used by the
	                               class/object.
	    dtype (Optional[Any]): Numeric dtype used for transformed feature matrices.
	    separator (Optional[str]): String separating dictionary keys and values in generated
	                               feature names.
	    sparse (Optional[bool]): Flag controlling sparse output from DictVectorizer.
	    sort (Optional[bool]): Flag controlling sorted feature ordering in DictVectorizer.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: fe.DictVectorizer
	dtype: Optional[ Any ]
	separator: Optional[ str ]
	sparse: Optional[ bool ]
	sort: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, dtype: Any=np.float64, separator: str='=', sparse: bool=True,
		sort: bool=True ) -> None:
		"""Initialize DictVectorizer.
		
		Purpose:
		    Initializes the dictionary vectorizer with configured objects, runtime
		    options, and transformed-output cache required by later preprocessing operations.
		
		Args:
		    dtype (Any): Numeric dtype used for transformed outputs.
		    separator (str): String separating dictionary keys and values in generated feature
		                     names.
		    sparse (bool): Flag controlling sparse matrix output.
		    sort (bool): Flag controlling sorted feature-name output.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.dtype = dtype
		self.separator = separator
		self.sparse = sparse
		self.sort = sort
		self.model = fe.DictVectorizer( dtype=self.dtype, separator=self.separator,
			sparse=self.sparse, sort=self.sort )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'dtype', 'separator', 'sparse', 'sort', 'transformed_data',
			'feature_names', 'vocabulary', 'train', 'transform', 'train_transform',
			'inverse_transform' ]
	
	@property
	def feature_names( self ) -> np.ndarray:
		"""Return feature names.
		
		Purpose:
		    Returns fitted output feature names generated by the wrapped dictionary vectorizer.
		
		Returns:
		    np.ndarray: Output feature names generated by the fitted vectorizer.
		
		Raises:
		    AttributeError: Raised when fitted metadata is unavailable."""
		try:
			return self.model.get_feature_names_out( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'feature_names( self ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	@property
	def vocabulary( self ) -> Dict[ str, int ]:
		"""Return vocabulary.
		
		Purpose:
		    Returns the fitted vocabulary mapping generated by the wrapped dictionary
		    vectorizer.
		
		Returns:
		    Dict[str, int]: Vocabulary mapping generated by the fitted vectorizer.
		
		Raises:
		    AttributeError: Raised when fitted metadata is unavailable."""
		try:
			return self.model.vocabulary_
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'vocabulary( self ) -> Dict[str, int]'
			Logger( ).write( exception )
			raise exception
	
	def train( self, X: List[ Dict[ str, Any ] ],
			y: Optional[ np.ndarray ]=None ) -> DictVectorizer | None:
		"""Fit DictVectorizer.
		
		Purpose:
		    Fits the underlying dictionary vectorizer to supplied input data and returns the
		    class/object for chained preprocessing workflows.
		
		Args:
		    X (List[Dict[str, Any]]): Input matrix, sequence, or feature collection transformed
		                              by the class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    DictVectorizer | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'train( self, *args ) -> DictVectorizer | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: List[ Dict[ str, Any ] ] ) -> np.ndarray:
		"""Transform with DictVectorizer.
		
		Purpose:
		    Transforms supplied input data with the fitted dictionary vectorizer and caches the
		    transformed output for downstream workflow steps.
		
		Args:
		    X (List[Dict[str, Any]]): Input matrix, sequence, or feature collection transformed
		                              by the class/object.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			result = self.model.transform( X )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'transform( self, X: List[Dict[str, Any]] ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: List[ Dict[ str, Any ] ],
		y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with DictVectorizer.
		
		Purpose:
		    Fits the underlying dictionary vectorizer and immediately transforms the supplied
		    input data in one operation.
		
		Args:
		    X (List[Dict[str, Any]]): Input matrix, sequence, or feature collection transformed
		                              by the class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			result = self.model.fit_transform( X, y )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> List[ Dict[ str, Any ] ] | None:
		"""Invert DictVectorizer output.
		
		Purpose:
		    Maps transformed output back toward the source representation when the wrapped
		    dictionary vectorizer supports inverse transformation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    List[Dict[str, Any]] | None: Decoded or reconstructed output produced by the fitted
		                                 class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			return self.model.inverse_transform( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'DictVectorizer'
			exception.method = 'inverse_transform( self, X: np.ndarray ) -> List[Dict[str, Any]] | None'
			Logger( ).write( exception )
			raise exception

class FeatureHasher( Transformer ):
	"""FeatureHasher.
	
	Purpose:
	    Converts dictionaries, feature-name pairs, or string feature collections into hashed
	    numeric feature matrices. The class/object stores hashing dimension, input type,
	    alternate-sign behavior, dtype, and cached transformed output.
	
	Attributes:
	    model (fe.FeatureHasher): Underlying estimator or transformer used by the
	                              class/object.
	    n_features (Optional[int]): Number of hashed output features.
	    input_type (Optional[str]): Input collection type accepted by FeatureHasher.
	    dtype (Optional[Any]): Numeric dtype used for transformed feature matrices.
	    alternate_sign (Optional[bool]): Flag controlling signed hashing behavior.
	    transformed_data (Optional[np.ndarray]): Most recent transformed output produced by
	                                             the class/object."""
	model: fe.FeatureHasher
	n_features: Optional[ int ]
	input_type: Optional[ str ]
	dtype: Optional[ Any ]
	alternate_sign: Optional[ bool ]
	transformed_data: Optional[ np.ndarray ]
	
	def __init__( self, n_features: int = 1048576, input_type: str = 'dict',
			dtype: Any = np.float64, alternate_sign: bool = True ) -> None:
		"""Initialize FeatureHasher.
		
		Purpose:
		    Initializes the feature hasher with configured objects, runtime options, and
		    transformed-output cache required by later preprocessing operations.
		
		Args:
		    n_features (int): Number of hashed output features.
		    input_type (str): Input collection type accepted by FeatureHasher.
		    dtype (Any): Numeric dtype used for transformed outputs.
		    alternate_sign (bool): Flag controlling signed hashing behavior.
		
		Returns:
		    None: This method initializes the object and does not return a value."""
		super( ).__init__( )
		self.n_features = n_features
		self.input_type = input_type
		self.dtype = dtype
		self.alternate_sign = alternate_sign
		self.model = fe.FeatureHasher( n_features=self.n_features,
			input_type=self.input_type, dtype=self.dtype,
			alternate_sign=self.alternate_sign )
		self.transformed_data = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return dir .
		
		Purpose:
		    Returns the stable public member list exposed by the class/object for interactive
		    inspection, notebook exploration, and IDE discovery.
		
		Returns:
		    List[str]: Public member names exposed by the class/object."""
		return [ 'model', 'n_features', 'input_type', 'dtype', 'alternate_sign',
			'transformed_data', 'train', 'transform', 'train_transform', 'inverse_transform' ]
	
	def train( self, X: List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ],
			y: Optional[ np.ndarray ]=None ) -> FeatureHasher | None:
		"""Fit FeatureHasher.
		
		Purpose:
		    Fits the underlying feature hasher to supplied input data and returns the class/object
		    for chained preprocessing workflows.
		
		Args:
		    X (List[Dict[str, Any]] | List[Tuple[str, Any]] | List[str]): Input matrix,
		                                                                  sequence, or feature
		                                                                  collection transformed
		                                                                  by the class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    FeatureHasher | None: Fitted class/object instance.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			self.model.fit( X, y )
			return self
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'FeatureHasher'
			exception.method = 'train( self, *args ) -> FeatureHasher | None'
			Logger( ).write( exception )
			raise exception
	
	def transform( self, X: List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ] ) -> np.ndarray:
		"""Transform with FeatureHasher.
		
		Purpose:
		    Transforms supplied input data with the fitted feature hasher and caches the
		    transformed output for downstream workflow steps.
		
		Args:
		    X (List[Dict[str, Any]] | List[Tuple[str, Any]] | List[str]): Input matrix,
		                                                                  sequence, or feature
		                                                                  collection transformed
		                                                                  by the class/object.
		
		Returns:
		    np.ndarray: Transformed output produced by the fitted class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			result = self.model.transform( X )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'FeatureHasher'
			exception.method = 'transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def train_transform( self, X: List[ Dict[ str, Any ] ] | List[ Tuple[ str, Any ] ] | List[ str ],
			y: Optional[ np.ndarray ]=None ) -> np.ndarray:
		"""Fit and transform with FeatureHasher.
		
		Purpose:
		    Fits the underlying feature hasher and immediately transforms the supplied input
		    data in one operation.
		
		Args:
		    X (List[Dict[str, Any]] | List[Tuple[str, Any]] | List[str]): Input matrix,
		                                                                  sequence, or feature
		                                                                  collection transformed
		                                                                  by the class/object.
		    y (Optional[np.ndarray]): Target vector or label collection aligned to the input
		                              data.
		
		Returns:
		    np.ndarray: Transformed output produced after fitting the class/object.
		
		Raises:
		    Error: Raised when validation or wrapped preprocessing execution fails."""
		try:
			throw_if( 'X', X )
			result = self.model.fit_transform( X, y )
			self.transformed_data = result.toarray( ) if hasattr( result, 'toarray' ) else result
			return self.transformed_data
		except Exception as e:
			exception = Error( e )
			exception.module = 'mathy'
			exception.cause = 'FeatureHasher'
			exception.method = 'train_transform( self, *args ) -> np.ndarray'
			Logger( ).write( exception )
			raise exception
	
	def inverse_transform( self, X: np.ndarray ) -> None:
		"""Invert FeatureHasher output.
		
		Purpose:
		    Maps transformed output back toward the source representation when the wrapped
		    feature hasher supports inverse transformation.
		
		Args:
		    X (np.ndarray): Input matrix, sequence, or feature collection transformed by the
		                    class/object.
		
		Returns:
		    None: This function performs its work through side effects and does not return a
		          value.
		
		Raises:
		    NotImplementedError: Raised when the base interface method is called directly."""
		raise NotImplementedError( 'FeatureHasher does not support inverse_transform.' )
	
