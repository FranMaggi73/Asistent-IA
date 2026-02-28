# intent_router.py
# Clasificación por keywords como responsable principal.
# Ollama queda exclusivamente para generate_response (preguntas abiertas).

import unicodedata
import json
import requests
from typing import Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Intents soportados
# ─────────────────────────────────────────────

INTENTS = [
    "greet",
    "goodbye",
    "open_app",
    "list_apps",
    "play_music",
    "control_music",
    "general_question",
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Minúsculas, sin tildes, sin puntuación extra"""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Quitar signos de puntuación que no aportan
    for ch in '¿?¡!.,;:':
        text = text.replace(ch, ' ')
    # Colapsar espacios
    return ' '.join(text.split())


def _extract_after(text: str, triggers: list[str]) -> Optional[str]:
    """Extrae el fragmento que viene después del primer trigger encontrado"""
    for trigger in triggers:
        if trigger in text:
            rest = text.split(trigger, 1)[1].strip()
            if rest:
                return rest
    return None


# ─────────────────────────────────────────────
# Dataclass resultado
# ─────────────────────────────────────────────

@dataclass
class IntentResult:
    intent: str
    entity: Optional[str]
    raw_text: str


# ─────────────────────────────────────────────
# Clasificador por keywords (responsable principal)
# ─────────────────────────────────────────────

class KeywordClassifier:
    """
    Clasifica intents sin ninguna dependencia externa.
    Latencia ~0ms. Cubre todos los casos de uso del asistente.
    """

    # Control de música: mapeamos keyword → entity normalizada
    CONTROL_MAP = {
        # Siguiente
        'siguiente': 'siguiente', 'skip': 'siguiente',
        'proxima': 'siguiente', 'proxima cancion': 'siguiente',
        'saltar': 'siguiente',
        # Anterior
        'anterior': 'anterior', 'atras': 'anterior',
        'volver': 'anterior', 'cancion anterior': 'anterior',
        # Pausa
        'pausa': 'pausa', 'pausar': 'pausa',
        'para la musica': 'pausa', 'deten': 'pausa',
        'detener': 'pausa', 'stop': 'pausa',
        # Reanudar
        'reanuda': 'play', 'reanudar': 'play',
        'continua': 'play', 'continuar': 'play',
        'seguir': 'play', 'sigue': 'play',
        # Volumen arriba
        'sube el volumen': 'subir', 'subir volumen': 'subir',
        'mas volumen': 'subir', 'sube': 'subir',
        'subir': 'subir',
        # Volumen abajo
        'baja el volumen': 'bajar', 'bajar volumen': 'bajar',
        'menos volumen': 'bajar', 'baja': 'bajar',
        'bajar': 'bajar',
        # Info
        'que suena': 'info', 'que esta sonando': 'info',
        'que cancion es': 'info', 'que cancion suena': 'info',
        'que cancion': 'info',
    }

    OPEN_TRIGGERS = [
        'abre ', 'abrir ', 'ejecuta ', 'ejecutar ',
        'lanza ', 'inicia ', 'iniciar ', 'muestra ',
        'mostrar ', 'abrí ', 'ejecutá ', 'lanzá ',
    ]

    PLAY_TRIGGERS = [
        'pon ', 'reproduce ', 'toca ', 'quiero escuchar ',
        'pone ', 'ponme ', 'reproducir ', 'tocar ',
        'escuchar ', 'poné ',
    ]

    LIST_KEYWORDS = [
        'que tengo', 'que hay', 'lista de programas',
        'listar programas', 'accesos directos',
        'que programas', 'que aplicaciones', 'que apps',
        'que tengo instalado',
    ]

    GREET_KEYWORDS = [
        'hola', 'buenos dias', 'buenas tardes', 'buenas noches',
        'buenas', 'hey', 'que tal', 'como estas', 'como andas',
    ]

    GOODBYE_KEYWORDS = [
        'adios', 'hasta luego', 'chao', 'chau', 'bye',
        'nos vemos', 'hasta pronto', 'hasta manana',
        'me voy', 'apágate', 'apagate',
    ]

    MUSIC_GENERIC = [
        'musica', 'una cancion', 'algo de musica',
        'pon algo', 'pone algo', 'reproduce algo',
    ]

    def classify(self, text: str) -> IntentResult:
        t = _normalize(text)

        # ── Greet ──────────────────────────────────
        if any(t == kw or t.startswith(kw + ' ') or t.endswith(' ' + kw)
               for kw in self.GREET_KEYWORDS):
            return IntentResult("greet", None, text)

        # ── Goodbye ────────────────────────────────
        if any(kw in t for kw in self.GOODBYE_KEYWORDS):
            return IntentResult("goodbye", None, text)

        # ── Control música (antes que play para evitar falsos positivos) ──
        # Buscar frases más largas primero
        for keyword in sorted(self.CONTROL_MAP, key=len, reverse=True):
            if keyword in t:
                return IntentResult("control_music", self.CONTROL_MAP[keyword], text)

        # ── Open app ───────────────────────────────
        for trigger in self.OPEN_TRIGGERS:
            if trigger in t:
                entity = t.split(trigger, 1)[1].strip()
                # Limpiar palabras de relleno
                stopwords = {'el', 'la', 'los', 'las', 'mi', 'por', 'favor'}
                entity = ' '.join(w for w in entity.split() if w not in stopwords)
                return IntentResult("open_app", entity or None, text)

        # ── List apps ──────────────────────────────
        if any(kw in t for kw in self.LIST_KEYWORDS):
            return IntentResult("list_apps", None, text)

        # ── Play música ────────────────────────────
        for trigger in self.PLAY_TRIGGERS:
            if trigger in t:
                entity = t.split(trigger, 1)[1].strip()
                # Quitar "algo de" al inicio
                for prefix in ['algo de ', 'una cancion de ', 'musica de ']:
                    if entity.startswith(prefix):
                        entity = entity[len(prefix):]
                return IntentResult("play_music", entity or None, text)

        # Música genérica sin trigger explícito
        if any(kw in t for kw in self.MUSIC_GENERIC):
            return IntentResult("play_music", t, text)

        # ── Fallback: pregunta general ─────────────
        return IntentResult("general_question", None, text)


# ─────────────────────────────────────────────
# Intent Router (orquestador)
# ─────────────────────────────────────────────

class IntentRouter:
    """
    classify()         → KeywordClassifier (siempre, ~0ms)
    generate_response() → Ollama (solo para general_question)
    """

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:3b"):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self._available: Optional[bool] = None
        self._classifier = KeywordClassifier()

    # ── Clasificación ──────────────────────────

    def classify(self, text: str) -> IntentResult:
        """
        Clasificación instantánea por keywords.
        Ollama no participa aquí.
        """
        if not text or not text.strip():
            return IntentResult("general_question", None, text)

        result = self._classifier.classify(text)
        print(f"🎯 Intent: {result.intent} | Entity: {result.entity}")
        return result

    # ── Disponibilidad Ollama ───────────────────

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def reset_availability(self) -> None:
        """Forzar re-chequeo de Ollama en el próximo ciclo"""
        self._available = None

    # ── Generación de respuesta libre ──────────

    def generate_response(self, text: str, max_tokens: int = 80) -> str:
        """
        Genera respuesta libre para preguntas generales.
        Único punto donde se usa Ollama.
        """
        if not self.is_available():
            return "El servicio de IA no está disponible. Probá con 'ollama serve'."

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
                    "num_predict": max_tokens,
                }
            }

            r = requests.post(self.generate_url, json=payload, timeout=15)

            if r.status_code == 200:
                response = r.json().get("response", "").strip()
                if len(response) > 300:
                    response = response[:297] + "..."
                return response

        except Exception as e:
            print(f"❌ Ollama generate error: {e}")
            # Marcar como no disponible para evitar timeouts en cascada
            self._available = False

        return "No pude generar una respuesta en este momento."