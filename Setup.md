# ⚙️ Guía de Configuración — Jarvis Optimizado

Esta guía te ayuda a instalar Jarvis con todas las optimizaciones para lograr **~8-11s de latencia**.

---

## Requisitos previos

- Python 3.10.x (probado en 3.10.11)
- GPU NVIDIA con CUDA 12.1
- Git
- Conexión a internet (para Groq API, opcional)

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

---

## Paso 3 — Instalar dependencias

```powershell
python.exe -m pip install --upgrade pip

pip install -r requirements.txt
```

Esto instala:
- PyTorch CUDA 12.1
- Whisper (faster-whisper)
- Coqui TTS (XTTS v2)
- Ollama client
- Groq client (nuevo)
- Spotipy
- Porcupine

---

## Paso 4 — Instalar Ollama y descargar modelo

1. Descargá e instalá Ollama desde https://ollama.com
2. Abrí una terminal y ejecutá:

```powershell
ollama pull llama3.2:1b
```

**Nota:** Usamos `llama3.2:1b` (no 3b) para velocidad optimizada.

3. Asegurate de que esté corriendo:

```powershell
ollama serve
```

**Opcional:** Si solo vas a usar Groq (cloud), podés saltear este paso.

---

## Paso 5 — Configurar Groq API (recomendado para velocidad)

Groq es **10x más rápido** que Ollama local y es **gratis** para uso personal.

1. Creá una cuenta en https://console.groq.com
2. Creá una API key (botón "Create API Key")
3. Copiá la key (empieza con `gsk_`)

**Límites gratuitos:**
- 30 requests/minuto
- 14,400 requests/día
- Más que suficiente para uso personal

---

## Paso 6 — Configurar Picovoice (wake word)

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

## Paso 7 — Configurar Spotify

1. Entrá a https://developer.spotify.com/dashboard
2. Creá una nueva app
3. En la app, agregá este Redirect URI: `http://localhost:8888/callback`
4. Copiá el Client ID y Client Secret

**Nota:** Necesitás Spotify Premium para control de reproducción.

---

## Paso 8 — Configurar el archivo .env

Copiá el archivo de ejemplo:

```powershell
copy .env.example .env
```

Editá `.env` con tus datos:

```env
# Picovoice
PICOVOICE_API_KEY=tu_key_de_picovoice

# Spotify
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback

# Ollama (fallback local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b

# Groq (cloud LLM - RECOMENDADO)
GROQ_API_KEY=gsk_tu_key_de_groq
```

---

## Paso 9 — Archivo speaker.wav

Jarvis necesita un archivo de audio de referencia para clonar la voz en el TTS.

- Grabá un audio tuyo o de la persona que quieres clonar
- Duración: 10-15 segundos (más no es mejor)
- Hablando en español, en un ambiente silencioso
- Formato: WAV mono, 22050Hz o 44100Hz
- Guardalo como `speaker.wav` en la raíz del proyecto

**Tip:** Cuanto más limpio el audio (sin ruido de fondo), mejor será la voz generada.

---

## Paso 10 — Primera ejecución

```powershell
python main.py
```

**La primera vez:**
- Descargará Whisper small (~460MB)
- Descargará XTTS v2 (~1.9GB)
- Calculará embeddings de tu speaker.wav (~8s)
- Arranca en ~30-40 segundos

**Siguientes veces:**
- Arranca en ~5 segundos

---

## Verificar instalación

Podés probar cada componente:

```powershell
# Verificar CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Verificar Whisper
python -c "from faster_whisper import WhisperModel; print('Whisper OK')"

# Verificar TTS
python -c "from TTS.api import TTS; print('TTS OK')"

# Verificar Ollama
curl http://localhost:11434/api/tags

# Verificar Groq (con tu key)
curl -H "Authorization: Bearer tu_groq_key" https://api.groq.com/openai/v1/models
```

---

## Configuración avanzada

### **Usar solo Groq (sin Ollama):**

Si querés usar SOLO Groq y no tener Ollama instalado:

1. No ejecutes `ollama serve`
2. Asegurate de tener `GROQ_API_KEY` en `.env`
3. Jarvis usará Groq automáticamente

### **Usar solo Ollama (100% local):**

Si preferís 100% privacidad sin cloud:

1. No agregues `GROQ_API_KEY` a `.env` (o comentalo)
2. Ejecutá `ollama serve`
3. Jarvis usará Ollama local

### **Ajustar longitud de respuestas:**

Editá `modules/intent_router.py`, línea ~130:

```python
max_tokens: int = 30  # Cambiar a 20 para respuestas MÁS cortas
                      # o 40 para respuestas más largas
```

---

## Troubleshooting

### **Error: "faster-whisper not found"**
```powershell
pip install faster-whisper
```

### **Error: "groq not found"**
```powershell
pip install groq
```

### **Error: "Ollama model llama3.2:1b not found"**
```powershell
ollama pull llama3.2:1b
```

### **Warning TTS: "gpu will be deprecated"**
Es solo un aviso, no afecta el funcionamiento. Ignoralo.

### **TTS tarda mucho (15-20s)**
Verificá que el caché de embeddings esté funcionando. Después de la primera conversación, debería bajar a 4-5s.

### **Groq da error 429 (rate limit)**
Esperá 1 minuto o usá Ollama local como fallback.

### **No detecta "Jarvis"**
1. Verificá que el micrófono funcione
2. Verificá `PICOVOICE_API_KEY` en `.env`
3. Hablá más fuerte o cerca del micrófono

---

## Performance esperado

| Métrica | Primera vez | Siguientes |
|---------|-------------|------------|
| Startup | ~30-40s | ~5s |
| Acciones (música, apps) | ~3-4s | ~3-4s |
| Conversación (Groq) | ~18s | **~8s** ✅ |
| Conversación (Ollama) | ~18s | **~11s** ✅ |
| VRAM usado | 4-5GB | 4-5GB |

---

## Próximos pasos

1. **Probá comandos básicos:**
   - "Jarvis, ¿cómo estás?"
   - "Jarvis, abre Chrome"
   - "Jarvis, pon despacito"

2. **Leé la documentación adicional:**
   - `CAMBIOS_TECNICOS.md` — Explicación de optimizaciones
   - `INSTALACION_OPTIMIZADA.md` — Guía de migración desde versión anterior

3. **Si algo falla:**
   - Revisá los logs en la consola
   - Verificá que todos los servicios estén corriendo
   - Consultá la sección Troubleshooting arriba

---

## Soporte

Para reportar bugs o sugerir mejoras:
- Abrí un issue en GitHub
- Incluí los logs completos de la consola
- Mencioná tu configuración (GPU, Python version, etc.)

---

**¡Listo! Ahora tenés Jarvis funcionando con latencia optimizada de ~8-11s** 🚀