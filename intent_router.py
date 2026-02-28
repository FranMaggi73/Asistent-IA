# intent_router.py - con Ollama directo
import json
import requests
from typing import Optional
from dataclasses import dataclass


INTENTS = [
    "greet",           # hola, buenos días, hey
    "goodbye",         # adiós, hasta luego, chao
    "open_app",        # abre chrome, ejecuta word, lanza steam
    "list_apps",       # qué tengo, lista mis accesos, qué hay en el escritorio
    "play_music",      # pon música, reproduce, toca una canción
    "control_music",   # pausa, continúa, sube volumen, detén
    "general_question" # cualquier otra pregunta o conversación
]

INTENT_PROMPT = """Eres un clasificador de intenciones para un asistente de voz en español.
Dado el mensaje del usuario, responde SOLO con un objeto JSON con los campos:
- "intent": una de estas opciones exactas: greet, goodbye, open_app, list_apps, play_music, control_music, general_question
- "entity": el objeto o tema principal mencionado (nombre de app, canción, artista, etc.) o null si no hay

Ejemplos:
"hola" -> {"intent": "greet", "entity": null}
"abre chrome" -> {"intent": "open_app", "entity": "chrome"}
"pon Despacito de Luis Fonsi" -> {"intent": "play_music", "entity": "Despacito Luis Fonsi"}
"sube el volumen" -> {"intent": "control_music", "entity": "subir"}
"pausa la música" -> {"intent": "control_music", "entity": "pausa"}
"qué programas tengo" -> {"intent": "list_apps", "entity": null}
"qué es la inteligencia artificial" -> {"intent": "general_question", "entity": null}

Responde SOLO con el JSON, sin texto adicional, sin markdown.

Mensaje: """


@dataclass
class IntentResult:
    intent: str
    entity: Optional[str]
    raw_text: str


class IntentRouter:
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:3b"):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def classify(self, text: str) -> IntentResult:
        """Clasifica el intent del texto usando Ollama"""
        if not text or not text.strip():
            return IntentResult("general_question", None, text)

        # Fallback rápido con keywords si Ollama no está disponible
        if not self.is_available():
            print("⚠️  Ollama no disponible, usando clasificador por keywords")
            return self._keyword_fallback(text)

        try:
            payload = {
                "model": self.model,
                "prompt": INTENT_PROMPT + f'"{text}"',
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Baja temperatura = más determinista
                    "num_predict": 60
                }
            }

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=20
            )

            if response.status_code != 200:
                return self._keyword_fallback(text)

            raw = response.json().get("response", "").strip()

            # Limpiar posibles backticks de markdown
            raw = raw.replace("```json", "").replace("```", "").strip()

            data = json.loads(raw)
            intent = data.get("intent", "general_question")
            entity = data.get("entity")

            # Validar que el intent sea uno de los esperados
            if intent not in INTENTS:
                intent = "general_question"

            print(f"🎯 Intent: {intent} | Entity: {entity}")
            return IntentResult(intent, entity, text)

        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️  Intent classification error: {e}, usando fallback")
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> IntentResult:
        """Clasificador simple por keywords como backup"""
        t = text.lower()

        if any(w in t for w in ['hola', 'buenos', 'buenas', 'hey', 'qué tal']):
            return IntentResult("greet", None, text)

        if any(w in t for w in ['adiós', 'adios', 'hasta luego', 'chao', 'bye']):
            return IntentResult("goodbye", None, text)

        if any(w in t for w in ['abre', 'abrir', 'ejecuta', 'lanza', 'inicia', 'muestra']):
            words = t.split()
            idx = next((i for i, w in enumerate(words)
                       if w in ['abre', 'ejecuta', 'lanza', 'inicia']), -1)
            entity = ' '.join(words[idx+1:]) if idx >= 0 else None
            return IntentResult("open_app", entity, text)

        if any(w in t for w in ['lista', 'listar', 'qué tengo', 'qué hay', 'accesos']):
            return IntentResult("list_apps", None, text)

        if any(w in t for w in ['pon', 'reproduce', 'toca', 'escuchar', 'música']):
            # Extraer nombre de canción/artista
            for prefix in ['pon ', 'reproduce ', 'toca ', 'quiero escuchar ']:
                if prefix in t:
                    entity = t.split(prefix, 1)[1].strip()
                    return IntentResult("play_music", entity, text)
            return IntentResult("play_music", None, text)

        if any(w in t for w in ['pausa', 'para', 'detén', 'reanuda', 'continúa',
                                  'stop', 'sube', 'baja', 'volumen']):
            return IntentResult("control_music", t.split()[0], text)

        return IntentResult("general_question", None, text)

    def generate_response(self, text: str, max_tokens: int = 80) -> str:
        """Genera respuesta libre para preguntas generales"""
        if not self.is_available():
            return "El servicio de IA no está disponible en este momento."

        system = (
            "Eres Jarvis, un asistente de voz inteligente y conciso. "
            "Responde en español, de forma breve (máximo 2 oraciones). "
            "Tus respuestas deben ser fáciles de escuchar en voz alta."
        )

        try:
            payload = {
                "model": self.model,
                "prompt": f"{system}\n\nUsuario: {text}\nJarvis:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": max_tokens
                }
            }

            r = requests.post(self.generate_url, json=payload, timeout=15)

            if r.status_code == 200:
                response = r.json().get("response", "").strip()
                # Limitar longitud para TTS
                if len(response) > 300:
                    response = response[:297] + "..."
                return response

        except Exception as e:
            print(f"❌ Ollama generate error: {e}")

        return "No pude generar una respuesta en este momento."