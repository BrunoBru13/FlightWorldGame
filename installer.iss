; Script di Inno Setup per FlightWorld (NovaGame).
; La versione arriva dalla pipeline (iscc /DAppVersion=1.0.0); se compili a mano usa quella qui sotto.
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{8F3C2A71-5B64-4E19-9D0A-6C1E7B52A3F4}
AppName=FlightWorld
AppVersion={#AppVersion}
AppPublisher=NovaGame
; Installa nella cartella dell'utente (niente permessi di amministratore):
; il gioco salva hangar.json accanto all'eseguibile e in Program Files non potrebbe scrivere.
PrivilegesRequired=lowest
DefaultDirName={autopf}\FlightWorld
DefaultGroupName=FlightWorld
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\FlightWorld.exe
OutputDir=Output
OutputBaseFilename=FlightWorld_Setup_{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\FlightWorld\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\FlightWorld"; Filename: "{app}\FlightWorld.exe"
Name: "{autodesktop}\FlightWorld"; Filename: "{app}\FlightWorld.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FlightWorld.exe"; Description: "{cm:LaunchProgram,FlightWorld}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\hangar.json"
