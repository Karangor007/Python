@echo off
setlocal enabledelayedexpansion

for %%f in (*image*) do (
  set "newname=%%f"
  ren "%%f" "!newname:image=invoice!"
)

endlocal
