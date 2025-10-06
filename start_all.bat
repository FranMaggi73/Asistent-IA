@echo off
echo ========================================
echo   🚀 JARVIS AUTO-LAUNCHER (Windows)
echo ========================================
echo.

set PROJECT_DIR=%CD%

REM Verificar entorno virtual
if not exist "jarvis\Scripts\activate.bat" (
    echo ⚠️  No se encontró entorno virtual 'jarvis'
    pause
    exit /b 1
)

echo ✓ Iniciando servicios...
echo.

REM Terminal 1: Rasa Server
start "Rasa Server" cmd /k "cd /d %PROJECT_DIR% && jarvis\Scripts\activate && echo 🤖 Iniciando Rasa Server... && rasa run --enable-api --cors *"

timeout /t 3 /nobreak >nul

REM Terminal 2: Actions
start "Action Server" cmd /k "cd /d %PROJECT_DIR% && jarvis\Scripts\activate && echo ⚙️  Iniciando Action Server... && rasa run actions"

timeout /t 5 /nobreak >nul

REM Terminal 3: Jarvis
start "Jarvis" cmd /k "cd /d %PROJECT_DIR% && jarvis\Scripts\activate && echo 🎙️  Iniciando Jarvis... && timeout /t 3 /nobreak >nul && python main.py"

echo.
echo ✅ Ventanas lanzadas
echo 💡 Espera ~10s a que Rasa inicie
echo.
pause