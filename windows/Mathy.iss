#ifndef AppVersion
#define AppVersion "0.1.1"
#endif

[Setup]
AppId={{A8629A0B-E2B2-40D9-9D15-914CA8D6E017}
AppName=Mathy
AppVersion={#AppVersion}
AppPublisher=Terry D. Eppler
DefaultDirName={localappdata}\Programs\Mathy
DefaultGroupName=Mathy
OutputDir=..\installer-dist
OutputBaseFilename=Mathy-Setup-{#AppVersion}-win64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayIcon={app}\Mathy.exe

[Files]
Source: "..\dist\Mathy\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Mathy"; Filename: "{app}\Mathy.exe"
Name: "{autodesktop}\Mathy"; Filename: "{app}\Mathy.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\Mathy.exe"; Description: "Launch Mathy"; Flags: nowait postinstall skipifsilent
