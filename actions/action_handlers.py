# action_handlers.py - Handlers de acciones sin dependencia de Rasa
import os
import subprocess
import unicodedata
import string
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, List
import threading

from spotify_player import spotify_player
from intent_router import IntentResult


# ─────────────────────────────────────────────
# Desktop Cache (igual que antes, sin Rasa)
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_desktop_path() -> Path:
    for p in [
        Path.home() / "OneDrive" / "Escritorio",
        Path.home() / "Desktop",
        Path.home() / "Escritorio",
    ]:
        if p.exists():
            return p
    return Path.home() / "Desktop"


def _normalize(text: str) -> str:
    text = text.lower().strip()
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


class DesktopCache:
    def __init__(self, ttl: int = 600):
        self._items: List[Path] = []
        self._normalized: dict = {}
        self._last_update: float = 0
        self._lock = threading.Lock()
        self.ttl = ttl

    def get(self, force: bool = False) -> Tuple[List[Path], dict]:
        import time
        now = time.time()
        with self._lock:
            if force or not self._items or (now - self._last_update) > self.ttl:
                desktop = get_desktop_path()
                exts = {'.lnk', '.url', '.html', '.htm', '.exe'}
                self._items = [
                    f for f in desktop.iterdir()
                    if f.is_file() and f.suffix.lower() in exts
                ]
                self._normalized = {f: _normalize(f.stem) for f in self._items}
                self._last_update = now
                print(f"🔄 Desktop cache: {len(self._items)} items")
        return self._items, self._normalized


desktop_cache = DesktopCache()


# ─────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────

def handle_open_app(result: IntentResult) -> str:
    """Abre un acceso directo del escritorio"""
    query = result.entity or result.raw_text

    # Limpiar palabras de comando del query
    ignore = {
        'abre', 'abrir', 'ejecuta', 'ejecutar', 'lanza', 'inicia',
        'iniciar', 'pon', 'el', 'la', 'mi', 'por', 'favor',
        'quiero', 'necesito', 'programa', 'aplicacion', 'app'
    }
    words = [w for w in _normalize(query).split() if w not in ignore and len(w) > 1]
    search = ' '.join(words).strip()

    if not search:
        return "No entendí qué querés abrir."

    items, normalized = desktop_cache.get()
    if not items:
        return "No hay accesos directos en el escritorio."

    search_words = set(search.split())
    best, best_score = None, 0

    for item in items:
        name = normalized[item]
        name_words = set(name.split())

        if search == name:
            best, best_score = item, 101
            break
        elif search in name:
            score = 100
        elif name in search:
            score = 90
        elif search[:3] == name[:3]:
            score = 85
        else:
            common = search_words & name_words
            score = (len(common) / len(search_words)) * 80 if search_words else 0

        if score > best_score:
            best_score = score
            best = item

    if not best or best_score < 40:
        return f"No encontré '{search}' en el escritorio."

    try:
        if os.name == 'nt':
            os.startfile(str(best))
        else:
            subprocess.Popen(['xdg-open', str(best)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return f"✓ {best.stem}"
    except Exception as e:
        print(f"❌ Error abriendo {best}: {e}")
        return f"No pude abrir {best.stem}."


def handle_list_apps(_result: IntentResult) -> str:
    """Lista accesos directos del escritorio"""
    items, _ = desktop_cache.get()
    if not items:
        return "No hay accesos directos en el escritorio."

    names = sorted([f.stem for f in items])
    preview = names[:8]
    text = ", ".join(preview)

    if len(names) > 8:
        return f"Tenés {len(names)} accesos: {text}, y más."
    return f"Tenés: {text}."


def handle_play_music(result: IntentResult) -> str:
    """Reproduce música en Spotify"""
    query = result.entity or result.raw_text

    if not query:
        return "¿Qué querés escuchar?"

    # Limpiar palabras de comando
    ignore = {'pon', 'reproduce', 'toca', 'quiero', 'escuchar', 'música',
               'una', 'cancion', 'canción', 'algo', 'de'}
    words = [w for w in query.lower().split() if w not in ignore]
    clean_query = ' '.join(words).strip() or query

    return spotify_player.play(clean_query)


def handle_control_music(result: IntentResult) -> str:
    """Controla la reproducción de Spotify"""
    text = (result.entity or result.raw_text).lower()

    if any(w in text for w in ['pausa', 'para', 'detén', 'detener', 'stop']):
        return spotify_player.pause()
    elif any(w in text for w in ['reanuda', 'continúa', 'continua', 'play', 'seguí']):
        return spotify_player.resume()
    elif any(w in text for w in ['sube', 'aumenta', 'más volumen', 'subir']):
        return spotify_player.volume_up()
    elif any(w in text for w in ['baja', 'disminuye', 'menos volumen', 'bajar']):
        return spotify_player.volume_down()
    elif any(w in text for w in ['qué suena', 'qué está', 'que canción', 'que suena']):
        return spotify_player.current_track()
    else:
        return "No entendí el comando de música."


def handle_greet(_result: IntentResult) -> str:
    import random
    return random.choice([
        "Hola, ¿en qué puedo ayudarte?",
        "Hola, dime.",
        "¡Hola! ¿Qué necesitás?"
    ])


def handle_goodbye(_result: IntentResult) -> str:
    import random
    return random.choice([
        "Hasta luego.",
        "Adiós.",
        "Nos vemos."
    ])


# ─────────────────────────────────────────────
# Dispatcher principal
# ─────────────────────────────────────────────

def dispatch(result: IntentResult, ollama_router=None) -> str:
    """Rutea el intent al handler correspondiente"""
    handlers = {
        "greet":            handle_greet,
        "goodbye":          handle_goodbye,
        "open_app":         handle_open_app,
        "list_apps":        handle_list_apps,
        "play_music":       handle_play_music,
        "control_music":    handle_control_music,
    }

    handler = handlers.get(result.intent)
    if handler:
        return handler(result)

    # general_question → Ollama
    if ollama_router:
        return ollama_router.generate_response(result.raw_text)

    return "No entendí. ¿Podés repetir?"