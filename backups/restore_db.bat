@echo off
REM === CONFIGURATION ===
set MYSQL_PATH=C:\xampp\mysql\bin
set BACKUP_PATH=C:\path\to\your\vscode\project\backup
set DB_NAME=coursemap_db
set DB_USER=root
set DB_PASS=

echo.
echo === RESTORE DATABASE: %DB_NAME% ===
echo Available backups in %BACKUP_PATH%:
echo.

dir /b %BACKUP_PATH%\%DB_NAME%_*.sql

echo.
set /p FILE=👉 Enter the backup filename to restore (example: coursemap_db_2025-09-13_22.sql): 

if "%FILE%"=="" (
    echo ❌ No file entered. Exiting.
    pause
    exit /b
)

echo.
echo Restoring %FILE% to %DB_NAME% ...
echo.

REM === RESTORE COMMAND ===
"%MYSQL_PATH%\mysql.exe" -u %DB_USER% %DB_NAME% < "%BACKUP_PATH%\%FILE%"

echo.
echo ✅ Restore complete!
pause
