"""Launch the bundled Mathy Streamlit application."""
import os
import sys
from pathlib import Path
from streamlit.web import cli as stcli

def main( ) -> None:
    """Start the Mathy Streamlit application.

    Returns:
        None: Streamlit manages the application lifecycle.
    """
    root = Path( sys._MEIPASS ) if getattr( sys, 'frozen', False ) else Path( __file__ ).resolve( ).parents[ 1 ]
    app_path = root / 'mathy' / 'app.py'
    if not app_path.is_file( ):
        raise FileNotFoundError( f'Mathy app not found: {app_path}' )
    os.chdir( str( Path.home( ) ) )
    sys.argv = [ 'streamlit', 'run', str( app_path ), '--server.headless=false', '--browser.gatherUsageStats=false' ]
    raise SystemExit( stcli.main( ) )

if __name__ == '__main__':
    main( )
