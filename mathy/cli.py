"""Launch the packaged Mathy Streamlit application.

Purpose:
    Provides the mathy-app command installed by mathy-py.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys


def main( ) -> None:
    """Launch the packaged Streamlit application.

    Returns:
        None: Exits with the Streamlit process status.
    """
    app_path = Path( __file__ ).resolve( ).parent / 'app.py'
    code = subprocess.call( [ sys.executable, '-m', 'streamlit', 'run', str( app_path ) ] )
    raise SystemExit( code )
