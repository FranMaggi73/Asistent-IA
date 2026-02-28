# setup.ps1 - Instalacion completa de Jarvis
# Ejecutar desde la raiz del proyecto con el venv activado:
#   .venv\Scripts\Activate.ps1
#   .\setup.ps1

Write-Host "Actualizando pip..." -ForegroundColor Cyan
python.exe -m pip install --upgrade pip

Write-Host "`nInstalando dependencias..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`nForzando transformers compatible con coqui-tts..." -ForegroundColor Cyan
pip install transformers==4.45.2 --no-deps
pip install tokenizers huggingface-hub safetensors

Write-Host "`nBuscando libiomp5md.dll duplicado..." -ForegroundColor Cyan
$dlls = Get-ChildItem -Path .venv -Recurse -Filter "libiomp5md.dll" |
        Where-Object { $_.FullName -notlike "*torch*" }

if ($dlls) {
    foreach ($dll in $dlls) {
        Remove-Item $dll.FullName
        Write-Host "OK - Eliminado: $($dll.FullName)" -ForegroundColor Green
    }
} else {
    Write-Host "OK - Sin duplicados, nada que hacer" -ForegroundColor Green
}

Write-Host "`nSetup completo. Correr: python main.py" -ForegroundColor Green