"""Regression tests for UI-to-estimator integration contracts."""

import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from mathy.features import SBS, RFE
from mathy.classifications import VotingModel, StackingModel, LogisticRegression as LogisticModel
from mathy.outliers import OutlierFactor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.multiclass import OneVsRestClassifier
from mathy.clusters import Birch


class UIIntegrationContracts( unittest.TestCase ):
	"""Verify the estimator contracts exercised by the application controls."""

	def test_feature_selection( self ) -> None:
		"""Verify classification and regression selectors retain two features."""
		X = np.random.default_rng( 42 ).normal( size=( 60, 4 ) )
		y_class = ( X[ :, 0 ] > 0 ).astype( int )
		y_regression = X[ :, 0 ] * 2.0 + X[ :, 1 ]
		selectors = [
			( SBS( classifier=KNeighborsClassifier( ), k_features=2 ), y_class ),
			( SBS( classifier=KNeighborsRegressor( ), k_features=2, scoring=r2_score ), y_regression ),
			( RFE( k_features=2, estimator=DecisionTreeClassifier( random_state=42 ) ), y_class ),
			( RFE( k_features=2, estimator=DecisionTreeRegressor( random_state=42 ) ), y_regression ) ]
		for selector, y in selectors:
			selector.train( X, y )
			self.assertEqual( selector.transform( X ).shape, ( 60, 2 ) )

	def test_ensembles( self ) -> None:
		"""Verify native logistic estimators are accepted by ensemble wrappers."""
		X = np.random.default_rng( 42 ).normal( size=( 60, 3 ) )
		y = ( X[ :, 0 ] > 0 ).astype( int )
		estimators = [ ( 'logistic', LogisticRegression( max_iter=1000, random_state=42 ) ),
			( 'tree', DecisionTreeClassifier( random_state=42 ) ) ]
		models = [ VotingModel( estimators=estimators, vote='soft' ),
			StackingModel( est=estimators, final=LogisticRegression( max_iter=1000 ) ) ]
		for model in models:
			model.train( X, y )
			self.assertEqual( len( model.project( X ) ), len( y ) )

	def test_birch_evaluation( self ) -> None:
		"""Verify evaluation does not fit again in either label mode."""
		X = np.array( [ [ 0.0 ], [ 0.1 ], [ 10.0 ], [ 10.1 ] ] )
		for compute_labels in [ True, False ]:
			model = Birch( n_clusters=2, compute_labels=compute_labels )
			model.train( X )
			with patch.object( model.model, 'fit', side_effect=AssertionError( 'Unexpected refit' ) ):
				with patch.object( model.model, 'fit_predict', side_effect=AssertionError( 'Unexpected refit' ) ):
					self.assertEqual( int( model.score( X )[ 'Clusters' ].iloc[ 0 ] ), 2 )
					self.assertEqual( len( model.analyze( X )[ 'Labels' ] ), len( X ) )

	def test_ui_reset_keys( self ) -> None:
		"""Verify both application sources use the initialized least-squares keys."""
		root = Path( __file__ ).resolve( ).parents[ 1 ]
		for path in [ root / 'app.py', root / 'palantir' / 'app.py' ]:
			source = path.read_text( encoding='utf-8' )
			for suffix in [ 'alpha', 'eta', 'iters', 'shuffle', 'penalty', 'test_size', 'random_state' ]:
				self.assertIn( f"key='classification_least_squares_{suffix}'", source )
				self.assertNotIn( f"key='classification_leastsquares_{suffix}'", source )

	def test_logistic_configuration( self ) -> None:
		"""Verify elastic-net configuration and explicit one-vs-rest fitting."""
		X = np.random.default_rng( 42 ).normal( size=( 90, 4 ) )
		y = np.tile( np.arange( 3 ), 30 )
		model = LogisticModel( penalty='elasticnet', solver='saga', multiclass='ovr',
			l1_ratio=0.35, iters=2000 )
		self.assertIsInstance( model.model, OneVsRestClassifier )
		self.assertEqual( model.model.estimator.l1_ratio, 0.35 )
		model.train( X, y )
		self.assertEqual( model.weights.shape, ( 3, 4 ) )
		self.assertEqual( model.predict_probability( X ).shape, ( 90, 3 ) )

	def test_lof_native_parity( self ) -> None:
		"""Verify wrapped training-set detection preserves native automatic contamination."""
		X = np.random.default_rng( 42 ).normal( size=( 60, 3 ) )
		expected = LocalOutlierFactor( n_neighbors=10, contamination='auto' ).fit_predict( X )
		model = OutlierFactor( n_neighbors=10, contamination='auto', novelty=False )
		model.train( X )
		np.testing.assert_array_equal( model.prediction, expected )

	def test_cca_control_contract( self ) -> None:
		"""Verify separate CCA output names and all three application call sites."""
		root = Path( __file__ ).resolve( ).parents[ 1 ]
		for path in [ root / 'app.py', root / 'palantir' / 'app.py' ]:
			source = path.read_text( encoding='utf-8' )
			self.assertIn( "prefix='cca_X'", source )
			self.assertIn( "prefix='cca_y'", source )
			self.assertEqual( source.count( ' = commit_cca_columns( ' ), 3 )
