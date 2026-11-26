"""Regression coverage for scikit-learn estimator configuration contracts."""

import unittest
import numpy as np
from clusters import Agglomerative
from regressions import LeastSquares


class SklearnRuntimeContracts( unittest.TestCase ):
	"""Verify that wrapper configuration reaches the underlying estimators.

	Purpose:
	    Exercises previously invalid estimator configurations.
	"""

	def test_agglomerative_distance_threshold( self ) -> None:
		"""Verify threshold-based clustering can fit without a fixed cluster count."""
		model = Agglomerative( clusters=2, distance_threshold=1.0,
			compute_full_tree=False )
		self.assertIsNone( model.model.n_clusters )
		self.assertIs( model.model.compute_full_tree, True )
		X = np.array( [ [ 0.0 ], [ 0.1 ], [ 10.0 ], [ 10.1 ] ] )
		labels = model.model.fit_predict( X )
		self.assertEqual( len( np.unique( labels ) ), 2 )

	def test_agglomerative_fixed_count( self ) -> None:
		"""Verify fixed-count clustering retains its configured cluster count."""
		model = Agglomerative( clusters=2 )
		self.assertEqual( model.model.n_clusters, 2 )
		self.assertIsNone( model.model.distance_threshold )

	def test_least_squares_tolerance( self ) -> None:
		"""Verify the supported estimator accepts the wrapper tolerance."""
		model = LeastSquares( tol=1e-5 )
		self.assertEqual( model.model.tol, 1e-5 )
		X = np.array( [ [ 0.0 ], [ 1.0 ], [ 2.0 ] ] )
		y = np.array( [ 0.0, 2.0, 4.0 ] )
		model.model.fit( X, y )
		np.testing.assert_allclose( model.model.predict( X ), y )
