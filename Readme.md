# 🎙️ Jarvis — Asistente de Voz con IA (Optimizado)

Jarvis es un asistente de voz local que corre completamente en tu PC. Escucha por un wake word ("Jarvis"), transcribe lo que decís, entiende tu intención y ejecuta acciones — con latencia optimizada de **~8-11 segundos**.

---

## ¿Qué puede hacer?

| Comando | Ejemplo |
|---|---|
| Conversación natural | *"¿Cómo estás?"*, *"Contame un chiste"*, *"¿Qué es Python?"* |
| Abrir aplicaciones | *"Abre Chrome"*, *"Ejecuta Spotify"* |
| Listar accesos directos | *"¿Qué programas tengo?"* |
| Reproducir música | *"Pon Bohemian Rhapsody"*, *"Play despacito"*, *"Tocá algo de Queen"* |
| Controlar Spotify | *"Pausa"*, *"Siguiente"*, *"Anterior"*, *"Sube el volumen"* |

---

## Arquitectura Optimizada

```
Micrófono
   ↓
Porcupine (wake word "Jarvis")
   ↓
Whisper small (transcripción optimizada en CUDA)
   ↓
Clasificador por keywords (intent detection ~0ms)
   ↓
   ├─→ Acciones directas (Spotify, apps) → Respuesta inmediata
   └─→ Conversación → Groq llama-3.1-8b (cloud) o Ollama llama3.2:1b (local)
       ↓
Coqui XTTS v2 (texto a voz con caché de embeddings)
   ↓
Parlantes
```

---

## Optimizaciones vs versión original

| Componente | Antes | Ahora | Mejora |
|------------|-------|-------|--------|
| **Whisper** | medium | **small** | -50% latencia |
| **LLM** | llama3.2:3b local | **Groq 8b** (cloud) o **llama3.2:1b** (local) | -70% latencia |
| **TTS** | Sin caché | **Caché de embeddings** | -60% latencia (2da+ vez) |
| **Latencia total** | ~28s | **~8-11s** | **-71%** ✅ |

---

## Requisitos de hardware

- GPU NVIDIA con soporte CUDA (probado en RTX 2060 6GB)
- 8GB RAM mínimo recomendado
- Micrófono
- Cuenta Spotify Premium (para control de reproducción)
- Conexión a internet (opcional, solo para Groq)

---

## Cómo usar

1. Asegurate de que Ollama esté corriendo (si no usás Groq): `ollama serve`
2. Abrí Spotify en tu PC o celular
3. Ejecutá el asistente: `python main.py`
4. Decí **"Jarvis"** para activarlo
5. Hablá tu comando

---

## Estructura del proyecto

```
Asistent-IA/
├── main.py                  # Punto de entrada (optimizado)
├── speaker.wav              # Voz de referencia para TTS
├── keywords/
│   ├── jarvis_es_windows_v3_0_0.ppn   # Modelo wake word
│   └── porcupine_params_es.pv         # Modelo idioma español
├── modules/
│   ├── audio_listener.py    # Pipeline principal (optimizado)
│   ├── audioFunctions.py    # Whisper small + caché TTS
│   ├── intent_router.py     # Groq/Ollama + keywords
│   ├── action_handlers.py   # Ejecutores de acciones
│   ├── spotify_player.py    # Control de Spotify
│   └── wake_word_detector.py # Detección de wake word
├── .env                     # Credenciales (no se sube al repo)
├── .env.example             # Ejemplo de configuración
└── requirements.txt
```

---

## Variables de entorno (.env)

```env
# Picovoice (Wake Word)
PICOVOICE_API_KEY=tu_api_key

# Spotify API
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback

# Ollama (local LLM - fallback)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b

# Groq API (cloud LLM - recomendado para velocidad)
GROQ_API_KEY=tu_groq_api_key
```

---

## Stack tecnológico

| Componente | Tecnología | Latencia |
|------------|------------|----------|
| Wake word | Porcupine | 0.5s |
| STT | Whisper small (CUDA) | 1-1.5s |
| Intent | Keywords classifier | ~0ms |
| LLM | Groq llama-3.1-8b / Ollama llama3.2:1b | 0.8s / 3s |
| TTS | XTTS v2 (CUDA + caché) | 4-5s |
| **Total** | | **~8-11s** |

---

## Modos de operación

### **Modo híbrido (recomendado):**
- LLM: Groq (cloud, ultra-rápido)
- Privacidad: Solo texto va a la nube, audio NO sale de tu PC
- Latencia: **~8s** ✅

### **Modo 100% local:**
- LLM: Ollama llama3.2:1b
- Privacidad: Todo en tu PC
- Latencia: **~11s** ✅

---

## Documentación adicional

- **Setup completo:** Ver [Setup.md](Setup.md)
- **Cambios técnicos:** Ver `CAMBIOS_TECNICOS.md` (si lo incluiste)
- **Guía de instalación:** Ver `INSTALACION_OPTIMIZADA.md` (si lo incluiste)

---

## Performance

**Promedio de latencia por tipo de comando:**

| Comando | Latencia | TTS |
|---------|----------|-----|
| Acciones (música, apps) | ~3-4s | No |
| Conversación (1ra vez) | ~18s | Sí (calcula embeddings) |
| Conversación (siguientes) | **~8-11s** | Sí (caché) |

---

## Troubleshooting

### "Groq API no disponible"
→ Verificá conexión a internet o agregá `GROQ_API_KEY` en `.env`

### "Ollama no disponible"
→ Ejecutá `ollama serve` y asegurate de tener `llama3.2:1b` descargado

### "No hay dispositivos Spotify activos"
→ Abrí Spotify en tu PC o celular antes de usar comandos de música

### Latencia sigue siendo alta
→ Verificá que estés usando Whisper **small** y tengas el caché de embeddings activo

---

## Licencia

MIT

---

## Créditos

- **Whisper:** OpenAI
- **XTTS v2:** Coqui AI
- **Porcupine:** Picovoice
- **Ollama:** Ollama Team
- **Groq:** Groq Inc.