# Mathy — Windows Installation

Mathy is a Streamlit-based machine-learning and statistical-analysis application. The Windows installer packages the application with its Python runtime so that a separate Python installation is not required.

## Requirements

- 64-bit Windows.
- Sufficient disk space for a bundled Python, scientific-computing libraries, and Mathy application resources.
- A modern web browser for the local Streamlit interface.

The Windows installer is distributed separately from the `mathy-py` Python package.

## Download

1. Open the [Mathy GitHub Releases](https://github.com/is-leeroy-jenkins/mathy/releases) page.
2. Select **v0.1.1** or a newer release containing a Windows installer asset.
3. Under **Assets**, download `Mathy-Setup-0.1.1-win64.exe` for version 0.1.1.

**Availability:** The v0.1.1 installer will appear only after the Windows build workflow completes and attaches the executable to the published release. If it is absent, the installer has not yet been made available for that release.

## Install

1. Open the downloaded `.exe` installer.
2. Follow the Mathy setup wizard. The default installation is within your Windows user profile, so an administrator installation is not normally required.
3. Optionally select **Create desktop shortcut**.
4. Complete the wizard, optionally selecting **Launch Mathy**.

The installer creates a **Mathy** shortcut in the Start Menu and can create a desktop shortcut. You do not need to install Python or run `pip` for this distribution.

## Run Mathy

1. Launch **Mathy** from the Windows Start Menu or optional desktop shortcut.
2. The application starts a local Streamlit server and opens its interface in your default web browser.
3. Keep the Mathy launcher process running while using the browser interface.
4. Close the launcher process when finished to stop the local Streamlit server.

Mathy runs locally on your computer; the Streamlit user interface is served through a local browser connection. Some analytical features may have additional resource or dependency requirements.

## Update

1. Download the installer asset for the newer GitHub Release.
2. Close Mathy before installing the update.
3. Run the newer installer and follow the setup wizard.

The installed version follows the version shown in the installer and GitHub Release.

## Uninstall

1. Open **Settings → Apps → Installed apps** (or **Apps & features**, depending on Windows).
2. Find **Mathy**.
3. Select **Uninstall** and follow the prompts.

Files that you separately created or saved outside the application installation directory are not automatically deleted by uninstalling Mathy.

## Troubleshooting

**The installer is not listed under a release.** The Windows build may not have completed, or its asset upload may have failed. Check the [Build Windows installer workflow](https://github.com/is-leeroy-jenkins/mathy/actions/workflows/windows-installer.yml).

**Mathy does not open in a browser.** Check whether the launcher is still running, and inspect its console output for Streamlit startup errors. Local security software or port conflicts may block startup.

**A Windows security warning appears.** Windows may warn about an installer that has not been code-signed. Verify the download is from this repository's GitHub Release and evaluate the publisher before choosing whether to proceed.

**An analysis mode fails.** The Windows package may be missing an optional dependency or data resource. Report the failure on [GitHub Issues](https://github.com/is-leeroy-jenkins/mathy/issues), including the application version, the selected analysis mode, and the relevant error output.

## Alternative: Install from PyPI

Developers who want the Python package rather than the Windows installer can run:

```powershell
python -m pip install mathy-py
mathy-app
```

The PyPI distribution is `mathy-py`; the import namespace remains `mathy`. This option requires a compatible Python installation.

## Building the Windows Installer

The automated Windows packaging workflow uses **GitHub Actions**, **PyInstaller**, and **Inno Setup**.

1. The Windows workflow installs Mathy's package requirements and packaging tools on a Windows runner.
2. PyInstaller bundles the existing Streamlit application, Python runtime, package modules, and configured resources into a Windows application directory.
3. Inno Setup builds the graphical `Mathy-Setup-<version>-win64.exe` installer, Start Menu shortcut, optional desktop shortcut, and uninstaller.
4. GitHub Actions publishes the installer as a workflow artifact and, for a published versioned release, attaches the installer to that release.

The `windows-installer.yml` workflow supports manual execution and published-release execution. Its source files are [Windows launcher](windows/launcher.py), [installer definition](windows/Mathy.iss), and [GitHub Actions workflow](.github/workflows/windows-installer.yml).

The separate [PyPI publishing workflow](.github/workflows/publish-to-pypi.yml) builds and publishes the Python package through Trusted Publishing. A successful Windows build does not itself publish `mathy-py` to PyPI.

**Release validation:** Verify installer compilation, clean Windows installation, application startup, representative analytical workflows, and uninstallation before distributing a release as production-ready.

[Back to README](README.md)
