"""Streamlit application startup and predictive navigation regression coverage."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class ApplicationRuntime( unittest.TestCase ):
	"""Exercise the standard Streamlit application."""

	def test_startup_and_model_navigation( self ) -> None:
		"""Verify startup and supervised model modes render without uncaught exceptions."""
		path = Path( __file__ ).resolve( ).parents[ 1 ] / 'app.py'
		app = AppTest.from_file( str( path ), default_timeout=90 ).run( )
		self.assertFalse( app.exception, str( app.exception ) )
		for mode in [ 'Classification Models', 'Regression Models' ]:
			app.radio( key='ml_mode_radio' ).set_value( mode ).run( )
			self.assertFalse( app.exception, str( app.exception ) )
			self.assertTrue( any( item.label == 'Independent Predictive Evaluation' for item in app.expander ) )
