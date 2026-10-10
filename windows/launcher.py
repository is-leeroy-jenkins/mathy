"""Mathy desktop application launcher.

Purpose:
    Run Streamlit on a local loopback server and display Mathy inside a
    dedicated Microsoft Edge WebView2 window without browser tabs.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def application_root( ) -> Path:
    """Return the directory containing the bundled Mathy application.

    Returns:
        Path: Read-only application payload directory.
    """
    base = Path( getattr( sys, '_MEIPASS', Path( __file__ ).resolve( ).parents[ 1 ] ) )
    return base / 'mathy'


def user_data_root( ) -> Path:
    """Return the writable user-specific application directory.

    Returns:
        Path: Local application data directory for Mathy.
    """
    root = Path( os.environ.get( 'LOCALAPPDATA', str( Path.home( ) / 'AppData' / 'Local' ) ) ) / 'Mathy'
    root.mkdir( parents=True, exist_ok=True )
    return root


def copy_if_missing( source: str, destination: str ) -> str:
    """Copy a bundled resource without overwriting existing user data.

    Args:
        source (str): Bundled source file.
        destination (str): Target file in writable application data.

    Returns:
        str: Destination filename.
    """
    if not Path( destination ).exists( ):
        shutil.copy2( source, destination )
    return destination


def prepare_user_data( ) -> Path:
    """Provision writable Mathy resources and stores.

    Returns:
        Path: Writable server working directory.
    """
    source = application_root( )
    destination = user_data_root( )
    for name in ( 'resources', '.streamlit', 'stores' ):
        folder = source / name
        if folder.is_dir( ):
            shutil.copytree( folder, destination / name, dirs_exist_ok=True,
                copy_function=copy_if_missing )
    (destination / 'stores' / 'sqlite').mkdir( parents=True, exist_ok=True )
    return destination


def find_available_port( ) -> int:
    """Find an available loopback TCP port.

    Returns:
        int: Available TCP port.
    """
    with socket.socket( socket.AF_INET, socket.SOCK_STREAM ) as listener:
        listener.bind( ('127.0.0.1', 0) )
        return int( listener.getsockname( )[ 1 ] )


def run_server( port: int, directory: Path ) -> None:
    """Run the bundled Streamlit application in the child process.

    Args:
        port (int): Local Streamlit port.
        directory (Path): Writable server working directory.

    Returns:
        None: Blocks until the server exits.
    """
    from streamlit.web import bootstrap

    os.chdir( directory )
    app_path = application_root( ) / 'app.py'
    if not app_path.is_file( ):
        raise FileNotFoundError( f'Mathy app not found: {app_path}' )
    bootstrap.run( str( app_path ), False, [ ], {
        'server.address': '127.0.0.1',
        'server.port': port,
        'server.headless': True,
        'browser.gatherUsageStats': False,
        'server.fileWatcherType': 'none',
        'server.enableCORS': True,
        'server.enableXsrfProtection': True,
    } )


def wait_for_server( process: subprocess.Popen, port: int ) -> None:
    """Wait until the Streamlit server is ready for the desktop window.

    Args:
        process (subprocess.Popen): Streamlit child process.
        port (int): Local Streamlit port.

    Returns:
        None: Server is ready.

    Raises:
        RuntimeError: Server exits or does not respond.
    """
    endpoint = f'http://127.0.0.1:{port}/_stcore/health'
    deadline = time.monotonic( ) + 120
    while time.monotonic( ) < deadline:
        if process.poll( ) is not None:
            raise RuntimeError( 'Mathy Streamlit process exited during startup.' )
        try:
            with urlopen( endpoint, timeout=1 ) as response:
                if response.status == 200:
                    return
        except (URLError, OSError, TimeoutError):
            time.sleep( 0.35 )
    raise RuntimeError( 'Mathy Streamlit server did not become ready within 120 seconds.' )


def main( ) -> None:
    """Open Mathy in a tab-free WebView2 window and stop its child server.

    Returns:
        None: Closes the server when the desktop window exits.
    """
    if len( sys.argv ) == 4 and sys.argv[ 1 ] == '--mathy-server':
        run_server( int( sys.argv[ 2 ] ), Path( sys.argv[ 3 ] ) )
        return

    import webview

    directory = prepare_user_data( )
    port = find_available_port( )
    command = [sys.executable, '--mathy-server', str( port ), str( directory )]
    if not getattr( sys, 'frozen', False ):
        command.insert( 1, str( Path( __file__ ).resolve( ) ) )
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    server = subprocess.Popen( command, cwd=str( directory ), creationflags=flags )
    try:
        wait_for_server( server, port )
        webview.create_window( 'Mathy', f'http://127.0.0.1:{port}',
            width=1440, height=900, min_size=(1024, 700) )
        webview.start( gui='edgechromium', debug=False )
    finally:
        if server.poll( ) is None:
            server.terminate( )
            try:
                server.wait( timeout=10 )
            except subprocess.TimeoutExpired:
                server.kill( )
                server.wait( )


if __name__ == '__main__':
    main( )
