"""Regression tests for independent predictive evaluation."""
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
from mathy.predictive import create_predictive_pipeline, evaluate_predictive_model


class PredictiveContracts( unittest.TestCase ):
	"""Verify training-only transformations and holdout evaluation."""

	def test_training_only_statistics( self ) -> None:
		"""Verify held-out values and categories never alter fitted transformations."""
		X = pd.DataFrame( { 'numeric': [ 1., 3., np.nan ], 'category': [ 'a', 'b', 'a' ] } )
		model = create_predictive_pipeline( X, LinearRegression( ) )
		model.fit( X, [ 1., 2., 3. ] )
		preprocessing = model.named_steps[ 'preprocessing' ]
		numeric = preprocessing.named_transformers_[ 'numeric' ]
		encoder = preprocessing.named_transformers_[ 'categorical' ].named_steps[ 'encoder' ]
		model.predict( pd.DataFrame( { 'numeric': [ 10000. ], 'category': [ 'unseen' ] } ) )
		np.testing.assert_allclose( numeric.named_steps[ 'imputer' ].statistics_, [ 2. ] )
		np.testing.assert_allclose( numeric.named_steps[ 'scaler' ].mean_, [ 2. ] )
		self.assertEqual( encoder.categories_[ 0 ].tolist( ), [ 'a', 'b' ] )

	def test_cross_validation_statistics( self ) -> None:
		"""Verify each clone fits the numeric imputer on its own training fold."""
		X = pd.DataFrame( { 'numeric': [ 0., 1., 2., 3., 100., 200. ] } )
		y = np.arange( 6 )
		cv = KFold( n_splits=3 )
		result = cross_validate( create_predictive_pipeline( X, LinearRegression( ) ),
			X, y, cv=cv, return_estimator=True, error_score='raise' )
		for ( train, test ), model in zip( cv.split( X ), result[ 'estimator' ] ):
			imputer = model.named_steps[ 'preprocessing' ].named_transformers_[ 'numeric' ].named_steps[ 'imputer' ]
			np.testing.assert_allclose( imputer.statistics_, [ X.iloc[ train ][ 'numeric' ].median( ) ] )

	def test_predictive_modes( self ) -> None:
		"""Exercise independent evaluation and nested stacking in both modes."""
		df_source = pd.DataFrame( { 'feature': np.arange( 80, dtype=float ),
			'category': np.tile( [ 'a', 'b' ], 40 ), 'class': np.tile( [ 0, 1 ], 40 ),
			'target': np.arange( 80, dtype=float ) * 2 } )
		df_source.loc[ 0, 'feature' ] = np.nan
		for mode, target, name in [ ( 'classification', 'class', 'Logistic Regression' ),
			( 'classification', 'class', 'Stacking' ),
			( 'regression', 'target', 'Ordinary Least Squares' ),
			( 'regression', 'target', 'Stacking' ) ]:
			with self.subTest( mode=mode, name=name ):
				result = evaluate_predictive_model( df_source, [ 'feature', 'category' ],
					target, mode, name, folds=3 )
				self.assertTrue( np.isfinite( result[ 'Value' ] ).all( ) )
				self.assertEqual( result[ 'Value' ].iloc[ -2:].sum( ), len( df_source ) )
