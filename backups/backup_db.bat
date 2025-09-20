@echo off
REM === CONFIGURATION ===
set MYSQL_PATH=C:\xampp\mysql\bin
set BACKUP_PATH=C:\path\to\your\vscode\project\backup
set DB_NAME=coursemap_db
set DB_USER=root
set DB_PASS=

REM === CREATE TIMESTAMP ===
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do (
    set day=%%a
    set month=%%b
    set year=%%c
)
for /f "tokens=1 delims=: " %%a in ('time /t') do (
    set hour=%%a
)
set timestamp=%year%-%month%-%day%_%hour%

REM === BACKUP COMMAND ===
"%MYSQL_PATH%\mysqldump.exe" -u %DB_USER% %DB_NAME% > "%BACKUP_PATH%\%DB_NAME%_%timestamp%.sql"

echo.
echo ✅ Backup complete! Saved to: %BACKUP_PATH%\%DB_NAME%_%timestamp%.sql
pause
