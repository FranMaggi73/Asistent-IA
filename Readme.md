# 🎙️ Jarvis — Asistente de Voz con IA

Jarvis es un asistente de voz local que corre completamente en tu PC. Escucha por un wake word ("Jarvis"), transcribe lo que decís, entiende tu intención y ejecuta acciones — todo sin mandar datos a la nube.

---

## ¿Qué puede hacer?

| Comando | Ejemplo |
|---|---|
| Abrir aplicaciones | *"Abre Chrome"*, *"Ejecuta Spotify"* |
| Listar accesos directos | *"¿Qué programas tengo?"* |
| Reproducir música | *"Pon Bohemian Rhapsody"*, *"Tocá algo de Queen"* |
| Controlar Spotify | *"Pausa"*, *"Siguiente"*, *"Anterior"*, *"Sube el volumen"* |
| Preguntas generales | *"¿Qué es la inteligencia artificial?"* |
| Saludar / despedirse | *"Hola"*, *"Adiós"* |

---

## Arquitectura

```
Micrófono
   ↓
Porcupine (wake word "Jarvis")
   ↓
Whisper large-v2 (transcripción de voz a texto, en CUDA)
   ↓
Ollama llama3.2:3b (clasificación de intención)
   ↓
Action Handler (abre app / controla Spotify / responde)
   ↓
Coqui XTTS v2 (texto a voz, en CUDA)
   ↓
Parlantes
```

---

## Requisitos de hardware

- GPU NVIDIA con soporte CUDA (probado en RTX 2060 6GB)
- 8GB RAM mínimo recomendado
- Micrófono
- Cuenta Spotify Premium (para control de reproducción)

---

## Cómo usar

1. Asegurate de que Ollama esté corriendo: `ollama serve`
2. Abrí Spotify en tu PC o celular
3. Ejecutá el asistente: `python main.py`
4. Decí **"Jarvis"** para activarlo
5. Hablá tu comando

---

## Estructura del proyecto

```
Asistent-IA/
├── main.py                  # Punto de entrada
├── speaker.wav              # Voz de referencia para TTS
├── keywords/
│   ├── jarvis_es_windows_v3_0_0.ppn   # Modelo wake word
│   └── porcupine_params_es.pv         # Modelo idioma español
├── modules/
│   ├── audio_listener.py    # Pipeline principal
│   ├── audioFunctions.py    # Whisper + grabación + TTS
│   ├── intent_router.py     # Clasificación de intención (Ollama)
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
PICOVOICE_API_KEY=...
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
OLLAMA_MODEL=llama3.2:3b
```