!include "LogicLib.nsh"
!include "x64.nsh"

!macro customInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "tulpa requires Windows 11 x64 (Intel / AMD)."
    Quit
  ${EndIf}
  SetRegView 64
  ReadRegStr $1 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "PROCESSOR_ARCHITECTURE"
  ${If} $1 != "AMD64"
    MessageBox MB_OK|MB_ICONSTOP "This build supports Intel / AMD x64 PCs. Windows ARM is not supported yet."
    Quit
  ${EndIf}
  ReadRegStr $0 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" "CurrentBuildNumber"
  ${If} $0 < 22000
    MessageBox MB_OK|MB_ICONSTOP "tulpa requires Windows 11 (build 22000 or later)."
    Quit
  ${EndIf}
!macroend
