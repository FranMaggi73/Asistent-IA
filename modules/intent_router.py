# intent_router.py - VERSIÓN CORREGIDA
# Prompt más estricto para respuestas cortas

import unicodedata
import json
import requests
import os
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
    "reset_conversation",  # NUEVO: para reiniciar memoria
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
        'pon ', 'pone ', 'poneme ', 'ponme ', 'poned ', 'poné ',
        'reproduce ', 'reproducir ', 'reproduci ',
        'toca ', 'tocar ', 'tocame ',
        'play ', 'plei ',  # Inglés y spanglish
        'quiero escuchar ', 'escuchar ',
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

    RESET_KEYWORDS = [
        'olvida', 'olvidate', 'olvida todo', 'nueva conversacion',
        'reinicia', 'reiniciar', 'borra', 'borrar', 'empeza de nuevo',
        'reset', 'limpiar memoria', 'no te acuerdes',
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

        # ── Reset conversación ─────────────────────
        if any(kw in t for kw in self.RESET_KEYWORDS):
            return IntentResult("reset_conversation", None, text)

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
    generate_response() → Groq (si disponible) o Ollama local + memoria conversacional
    """

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:1b",
                 use_groq: bool = True,
                 max_history: int = 6):  # NUEVO: últimos 6 mensajes (3 turnos)
        
        # Ollama config
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self._ollama_available: Optional[bool] = None
        
        # Groq config
        self.use_groq = use_groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_available: Optional[bool] = None
        
        # Classifier
        self._classifier = KeywordClassifier()
        
        # NUEVO: Memoria conversacional
        self.conversation_history: list = []
        self.max_history = max_history
        
        # System prompt ULTRA-ESTRICTO para Groq
        self.system_prompt = (
            "Eres Jarvis. Responde en MÁXIMO 10 PALABRAS. "
            "Prohíbido exceder 10 palabras bajo cualquier circunstancia. "
            "Sé directo y conciso. Sin preguntas de seguimiento.\n\n"
            "Ejemplos CORRECTOS:\n"
            "Usuario: '¿Qué es Python?' → 'Lenguaje de programación creado en 1991.'\n"
            "Usuario: '¿Cómo estás?' → 'Funcionando perfectamente.'\n"
            "Usuario: 'Contame un chiste' → 'Los átomos no se fían, lo hacen todo de materia.'\n"
            "Usuario: '¿Qué hiciste hoy?' → 'Actualicé sistemas y optimicé protocolos.'"
        )

    # ── Clasificación ──────────────────────────

    def classify(self, text: str) -> IntentResult:
        """
        Clasificación instantánea por keywords.
        LLM no participa aquí.
        """
        if not text or not text.strip():
            return IntentResult("general_question", None, text)

        result = self._classifier.classify(text)
        print(f"🎯 Intent: {result.intent} | Entity: {result.entity}")
        return result

    # ── Disponibilidad servicios ───────────────

    def _check_groq(self) -> bool:
        """Verifica si Groq está disponible"""
        if self._groq_available is not None:
            return self._groq_available
        
        if not self.use_groq or not self.groq_api_key:
            self._groq_available = False
            return False
        
        try:
            # Test rápido de conectividad
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.groq_api_key}"},
                timeout=2
            )
            self._groq_available = response.status_code == 200
        except Exception:
            self._groq_available = False
        
        return self._groq_available

    def _check_ollama(self) -> bool:
        """Verifica si Ollama está disponible"""
        if self._ollama_available is not None:
            return self._ollama_available
        
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            self._ollama_available = r.status_code == 200
        except Exception:
            self._ollama_available = False
        
        return self._ollama_available

    def is_available(self) -> bool:
        """Verifica si hay algún servicio LLM disponible"""
        return self._check_groq() or self._check_ollama()

    def reset_availability(self) -> None:
        """Forzar re-chequeo de servicios en el próximo ciclo"""
        self._groq_available = None
        self._ollama_available = None

    # ── Gestión de memoria conversacional ──────

    def add_to_history(self, user_msg: str, assistant_msg: str) -> None:
        """Agrega un turno de conversación al historial"""
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": assistant_msg})
        
        # Mantener solo los últimos N mensajes
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_history(self) -> None:
        """Limpia la memoria conversacional"""
        self.conversation_history = []
        print("💭 Memoria conversacional reiniciada")
    
    def get_history_context(self) -> str:
        """Retorna el historial formateado para Ollama (modo texto)"""
        if not self.conversation_history:
            return ""
        
        context = "\n\nConversación previa:\n"
        for msg in self.conversation_history:
            role = "Usuario" if msg["role"] == "user" else "Jarvis"
            context += f"{role}: {msg['content']}\n"
        return context

    # ── Generación de respuesta libre ──────────

    def _generate_with_groq(self, text: str, max_tokens: int = 30) -> Optional[str]:
        """Genera respuesta usando Groq API (más rápido) con memoria conversacional"""
        if not self.groq_api_key:
            return None
        
        try:
            # Construir mensajes con historial
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Agregar historial de conversación
            messages.extend(self.conversation_history)
            
            # Agregar mensaje actual
            messages.append({"role": "user", "content": text})
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "stop": ["\n", "?"],
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                
                # POST-PROCESAMIENTO: Cortar respuestas largas
                words = answer.split()
                if len(words) > 12:
                    answer = ' '.join(words[:12]) + '.'
                
                if "?" in answer:
                    answer = answer.split("?")[0] + "."
                
                if len(answer) > 120:
                    answer = answer[:117] + "..."
                
                return answer
            
        except Exception as e:
            print(f"⚠️  Groq error: {e}")
            self._groq_available = False
        
        return None

    def _generate_with_ollama(self, text: str, max_tokens: int = 40) -> Optional[str]:
        """Genera respuesta usando Ollama local con memoria conversacional"""
        try:
            # Construir prompt con historial
            prompt = self.system_prompt
            
            # Agregar historial si existe
            if self.conversation_history:
                prompt += self.get_history_context()
            
            # Agregar mensaje actual
            prompt += f"\n\nUsuario: {text}\nJarvis:"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": max_tokens,
                    "top_p": 0.8,
                    "top_k": 30,
                    "repeat_penalty": 1.3,
                    "num_ctx": 1024,  # ← Aumentado de 512 para soportar historial
                    "stop": ["\n", "?", "Usuario:"],
                }
            }

            r = requests.post(self.generate_url, json=payload, timeout=15)

            if r.status_code == 200:
                response = r.json().get("response", "").strip()
                
                # POST-PROCESAMIENTO igual que Groq
                words = response.split()
                if len(words) > 12:
                    response = ' '.join(words[:12]) + '.'
                
                if "?" in response:
                    response = response.split("?")[0] + "."
                
                if len(response) > 120:
                    response = response[:117] + "..."
                
                return response

        except Exception as e:
            print(f"❌ Ollama error: {e}")
            self._ollama_available = False
        
        return None

    def generate_response(self, text: str, max_tokens: int = 30) -> str:
        """
        Genera respuesta libre para preguntas generales con memoria conversacional.
        Intenta Groq primero (más rápido), fallback a Ollama.
        """
        # Intentar Groq primero (si está habilitado)
        response = None
        if self._check_groq():
            response = self._generate_with_groq(text, max_tokens)
            if not response:
                print("⚠️  Groq falló, intentando con Ollama local...")
        
        # Fallback a Ollama si Groq no funcionó
        if not response and self._check_ollama():
            response = self._generate_with_ollama(text, max_tokens)
        
        # Si ambos fallan
        if not response:
            return "No pude generar una respuesta. Verificá que Ollama esté corriendo."
        
        # NUEVO: Agregar al historial conversacional
        self.add_to_history(text, response)
        
        return response