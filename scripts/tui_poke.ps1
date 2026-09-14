param(
  [Parameter(Mandatory=$true)][string]$TitlePattern,
  [Parameter(Mandatory=$true)][string]$Message,
  [switch]$FocusOnly
)
# Paste a watchdog wake message into the dedicated worker PowerShell window,
# as if typed: force-focus, Ctrl+V, Enter.
# Exit codes: 0 = poked/focused, 2 = no window, 3 = focus/paste failed.
# -FocusOnly just proves we can focus (no text sent) — safe to test live.

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class FgWin {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint t1, uint t2, bool a);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
}
"@

function Focus-Window($h) {
  if ($h -eq 0) { return $false }
  if ([FgWin]::IsIconic($h)) { [FgWin]::ShowWindow($h, 9) | Out-Null }  # SW_RESTORE
  for ($i = 0; $i -lt 3; $i++) {
    $fg = [FgWin]::GetForegroundWindow()
    if ($fg -eq $h) { return $true }
    $tFg = [FgWin]::GetWindowThreadProcessId($fg, [IntPtr]::Zero)
    $tMe = [FgWin]::GetCurrentThreadId()
    [FgWin]::AttachThreadInput($tFg, $tMe, $true) | Out-Null
    [FgWin]::SetForegroundWindow($h) | Out-Null
    [FgWin]::AttachThreadInput($tFg, $tMe, $false) | Out-Null
    Start-Sleep -Milliseconds 400
  }
  return ([FgWin]::GetForegroundWindow() -eq $h)
}

$proc = Get-Process | Where-Object { $_.MainWindowTitle -like "*$TitlePattern*" } | Select-Object -First 1
if (-not $proc) { Write-Error "no window matching '$TitlePattern'"; exit 2 }

$focused = Focus-Window $proc.MainWindowHandle
if (-not $focused) {
  # Fallback: COM AppActivate by PID (works when title lookup differs).
  $wshell = New-Object -ComObject WScript.Shell
  if ($wshell.AppActivate($proc.Id)) {
    Start-Sleep -Milliseconds 500
    $focused = ([FgWin]::GetForegroundWindow() -eq $proc.MainWindowHandle)
  }
}
if (-not $focused) { Write-Error "could not focus window pid $($proc.Id)"; exit 3 }
if ($FocusOnly) { exit 0 }

try {
  Set-Clipboard -Value $Message
} catch {
  Write-Error "clipboard failed: $_"
  exit 3
}
Start-Sleep -Milliseconds 200
$wshell = New-Object -ComObject WScript.Shell
$wshell.SendKeys('^v')
Start-Sleep -Milliseconds 300
$wshell.SendKeys('{ENTER}')
exit 0
