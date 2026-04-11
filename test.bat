@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON=.\.venv\Scripts\python.exe"
set "APP=src\cc.py"
set "IMG_DIR=img"
set "OUT_DIR=output"

for %%F in ("%IMG_DIR%\*.NEF" "%IMG_DIR%\*.CR2" "%IMG_DIR%\*.CR3" "%IMG_DIR%\*.ARW" "%IMG_DIR%\*.RAF" "%IMG_DIR%\*.DNG") do (
    if exist "%%~fF" (
        set "CC_IMAGE=%%~fF"
        set "RAW_STEM=%%~nF"
        goto :run
    )
)

echo No RAW file found in "%IMG_DIR%".
exit /b 1

:run
echo Using RAW file: %CC_IMAGE%

"%PYTHON%" "%APP%" analyze --cc-image "%CC_IMAGE%" --output-dir "%OUT_DIR%" --no-show-detection-preview
if errorlevel 1 exit /b 1

set "RESULT_JSON=%OUT_DIR%\result_%RAW_STEM%.json"
if not exist "%RESULT_JSON%" (
    echo Result JSON not found: %RESULT_JSON%
    exit /b 1
)

"%PYTHON%" "%APP%" process "%RESULT_JSON%" "%IMG_DIR%" --output-dir "%OUT_DIR%\processed"
if errorlevel 1 exit /b 1

echo Done.
