# ⚙️ Guía de Configuración — Jarvis

## Requisitos previos

- Python 3.10.x (probado en 3.10.11)
- GPU NVIDIA con CUDA 12.1
- Git

---

## Paso 1 — Clonar el repositorio

```powershell
git clone https://github.com/tu-usuario/Asistent-IA.git
cd Asistent-IA
```

---

## Paso 2 — Crear entorno virtual

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Paso 3 — Instalar dependencias

```powershell
python.exe -m pip install --upgrade pip

pip install -r requirements.txt
```

Esto instala todo: PyTorch CUDA 12.1, Whisper, Coqui TTS, Ollama client, Spotify, Porcupine, etc.

---

## Paso 4 — Instalar Ollama y descargar modelo

1. Descargá e instalá Ollama desde https://ollama.com
2. Abrí una terminal y ejecutá:

```powershell
ollama pull llama3.2:3b
```

3. Asegurate de que esté corriendo antes de iniciar Jarvis:

```powershell
ollama serve
```

---

## Paso 5 — Configurar Picovoice (wake word)

1. Creá una cuenta gratuita en https://console.picovoice.ai/
2. Copiá tu Access Key
3. Descargá el modelo de wake word para Windows (.ppn) y el modelo de idioma español (.pv)
4. Colocá los archivos en la carpeta `keywords/`:

```
keywords/
├── jarvis_es_windows_v3_0_0.ppn
└── porcupine_params_es.pv
```

---

## Paso 6 — Configurar Spotify

1. Entrá a https://developer.spotify.com/dashboard
2. Creá una nueva app
3. En la app, agregá este Redirect URI: `http://localhost:8888/callback`
4. Copiá el Client ID y Client Secret

---

## Paso 7 — Configurar el archivo .env

Copiá el archivo de ejemplo y completá tus credenciales:

```powershell
copy .env.example .env
```

Editá `.env` con tus datos:

```env
PICOVOICE_API_KEY=tu_key_aqui

SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback

OLLAMA_MODEL=llama3.2:3b
```

---

## Paso 8 — Archivo speaker.wav

Jarvis necesita un archivo de audio de referencia para clonar la voz en el TTS.

- Grabá un audio tuyo o de la persona que quieres clonar de 10-30 segundos hablando en español con buena calidad
- Guardalo como `speaker.wav` en la raíz del proyecto
- Cuanto más limpio el audio (sin ruido de fondo), mejor será la voz generada

---

## Paso 9 — Ejecutar

Asegurate de tener Ollama corriendo y Spotify abierto, luego:

```powershell
python main.py
```

La primera vez va a descargar el modelo XTTS v2 (~1.9GB). Después de eso arranca en ~25 segundos.

---

## Verificar instalación

Podés probar cada componente por separado:

```powershell
# Verificar TTS
python -c "from TTS.api import TTS; print('TTS OK')"

# Verificar Whisper
python -c "import whisper; print('Whisper OK')"

# Verificar CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Verificar Ollama
curl http://localhost:11434/api/tags
```