from typing import Any, Dict, List, Text
from pathlib import Path
import os
import subprocess
import unicodedata
import string
import vlc
import yt_dlp
from functools import lru_cache

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# ------------------------------
# VLC Player singleton
# ------------------------------
class VLCManager:
    _instance = None
    _vlc_instance = None
    _player = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def instance(self):
        if self._vlc_instance is None:
            self._vlc_instance = vlc.Instance()
        return self._vlc_instance
    
    @property
    def player(self):
        return self._player
    
    @player.setter
    def player(self, value):
        self._player = value

vlc_manager = VLCManager()


# ------------------------------
# Cache de escritorio (evita escanear repetidamente)
# ------------------------------
@lru_cache(maxsize=1)
def get_desktop_path() -> Path:
    """Cache de la ruta del escritorio"""
    for desktop_path in [
        Path.home() / "OneDrive" / "Escritorio",
        Path.home() / "Desktop",
        Path.home() / "Escritorio"
    ]:
        if desktop_path.exists():
            return desktop_path
    return Path.home() / "Desktop"


class DesktopItemCache:
    """Cache inteligente de items del escritorio"""
    def __init__(self, ttl=300):  # 5 minutos
        self._cache = None
        self._last_update = 0
        self.ttl = ttl
    
    def get_items(self, force_refresh=False):
        import time
        current_time = time.time()
        
        if force_refresh or self._cache is None or (current_time - self._last_update) > self.ttl:
            desktop = get_desktop_path()
            valid_extensions = {'.lnk', '.url', '.html', '.htm'}
            
            self._cache = [
                f for f in desktop.iterdir() 
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
            self._last_update = current_time
            print(f"🔄 Desktop cache refreshed ({len(self._cache)} items)")
        
        return self._cache

desktop_cache = DesktopItemCache()


# ------------------------------
# Acción: abrir programas (OPTIMIZADO)
# ------------------------------
class ActionOpenDesktopItem(Action):
    def name(self) -> Text:
        return "action_open_desktop_item"
    
    def _normalize_text(self, text: str) -> str:
        """Normalización rápida de texto"""
        text = text.lower().strip()
        # Remover acentos
        return ''.join(
            c for c in unicodedata.normalize('NFD', text) 
            if unicodedata.category(c) != 'Mn'
        )
    
    def _extract_app_name(self, message: str) -> str:
        """Extrae nombre de app del mensaje"""
        ignore_words = {
            'abre','abrir','abrí','abri','ejecuta','ejecutar','lanza',
            'inicia','iniciar','pon','el','la','mi','por','favor',
            'quiero','necesito','programa','aplicación','app'
        }
        
        # Remover puntuación y filtrar palabras
        message = message.translate(str.maketrans('', '', string.punctuation))
        words = [
            w for w in message.lower().split() 
            if w not in ignore_words and len(w) > 1
        ]
        return ' '.join(words).strip()
    
    def _find_best_match(self, search_term: str, items: List[Path]) -> Path:
        """Búsqueda optimizada con scoring"""
        search_norm = self._normalize_text(search_term)
        search_words = set(search_norm.split())
        
        best_match = None
        best_score = 0
        
        for item in items:
            item_name = self._normalize_text(item.stem)
            item_words = set(item_name.split())
            
            # Scoring rápido
            if search_norm == item_name:
                return item  # Match exacto
            
            if search_norm in item_name:
                score = 100
            elif item_name in search_norm:
                score = 90
            else:
                # Intersección de palabras
                common = search_words & item_words
                score = (len(common) / len(search_words)) * 80 if search_words else 0
            
            if score > best_score:
                best_score = score
                best_match = item
        
        return best_match if best_score > 50 else None
    
    def _open_file(self, file_path: Path) -> bool:
        """Abre archivo de forma segura"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(str(file_path))
            else:  # Linux/Mac
                subprocess.Popen(['xdg-open', str(file_path)])
            return True
        except Exception as e:
            print(f"Error opening file: {e}")
            return False
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text', '')
        if not user_message:
            dispatcher.utter_message(text="No especificaste qué abrir.")
            return []
        
        app_name = self._extract_app_name(user_message)
        if not app_name:
            dispatcher.utter_message(text="No pude identificar el programa.")
            return []
        
        # Usar cache
        desktop_items = desktop_cache.get_items()
        if not desktop_items:
            dispatcher.utter_message(text="No hay accesos en el escritorio.")
            return []
        
        matched_item = self._find_best_match(app_name, desktop_items)
        if not matched_item:
            dispatcher.utter_message(text=f"No encontré '{app_name}'.")
            return []
        
        success = self._open_file(matched_item)
        if success:
            # Respuesta corta para acción exitosa
            dispatcher.utter_message(text=f"✓ {matched_item.stem}")
        else:
            dispatcher.utter_message(text=f"No pude abrir {matched_item.stem}")
        
        return []


# ------------------------------
# Acción: listar accesos (OPTIMIZADO)
# ------------------------------
class ActionListDesktopItems(Action):
    def name(self) -> Text:
        return "action_list_desktop_items"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Verificar si el usuario realmente pidió listar
        user_message = tracker.latest_message.get('text', '').lower()
        list_keywords = ['lista', 'listar', 'muestra', 'qué tengo', 'qué hay', 
                        'qué accesos', 'qué programas', 'cuáles son']
        
        # Si no hay palabras clave de listar, probablemente es un error de clasificación
        if not any(keyword in user_message for keyword in list_keywords):
            dispatcher.utter_message(text="No entendí. ¿Puedes repetir?")
            return []
        
        items = desktop_cache.get_items()
        
        if not items:
            dispatcher.utter_message(text="No hay accesos en el escritorio.")
            return []
        
        # Ordenar alfabéticamente
        items_names = sorted([f.stem for f in items])
        items_text = ", ".join(items_names[:8])
        
        if len(items_names) > 8:
            msg = f"Tienes {len(items_names)} accesos: {items_text}, y más"
        else:
            msg = f"Tienes: {items_text}"
        
        dispatcher.utter_message(text=msg)
        return []


# ------------------------------
# Acción: reproducir música (OPTIMIZADO + NO BLOQUEANTE)
# ------------------------------
class ActionPlayMusic(Action):
    def name(self) -> Text:
        return "action_play_music"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        query = tracker.latest_message.get("text", "")
        if not query:
            dispatcher.utter_message(text="No especificaste la canción.")
            return []
        
        # Opciones optimizadas de yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'noplaylist': True,
            'socket_timeout': 10,
            'retries': 2,  # Reducido de 3 a 2
            'extract_flat': False,
            'no_warnings': True
        }
        
        try:
            # Búsqueda rápida de YouTube
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)  # Solo 1 resultado
                
                if not info or 'entries' not in info or not info['entries']:
                    dispatcher.utter_message(text="No encontré esa canción.")
                    return []
                
                entry = info['entries'][0]
                url = entry['url']
                title = entry.get('title', 'Canción')
        
        except Exception as e:
            print(f"YouTube error: {e}")
            dispatcher.utter_message(text="Error al buscar la canción.")
            return []
        
        # Detener reproducción actual si existe
        if vlc_manager.player and vlc_manager.player.is_playing():
            vlc_manager.player.stop()
        
        # Reproducir nueva canción (NO bloqueante)
        try:
            media = vlc_manager.instance.media_new(url)
            vlc_manager.player = vlc_manager.instance.media_player_new()
            vlc_manager.player.set_media(media)
            vlc_manager.player.play()
            
            # Respuesta corta para acción exitosa
            dispatcher.utter_message(text=f"▶️ {title}")
        except Exception as e:
            print(f"VLC error: {e}")
            dispatcher.utter_message(text="Error al reproducir.")
        
        return []


# ------------------------------
# Acción: controlar música (OPTIMIZADO)
# ------------------------------
class ActionControlMusic(Action):
    def name(self) -> Text:
        return "action_control_music"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        player = vlc_manager.player
        
        if not player:
            dispatcher.utter_message(text="No hay música reproduciéndose.")
            return []
        
        msg = tracker.latest_message.get("text", "").lower()
        
        # Mapeo de comandos
        if any(word in msg for word in ['pausa', 'para', 'detén']):
            player.pause()
            dispatcher.utter_message(text="⏸ Pausado")
        
        elif any(word in msg for word in ['reanuda', 'continúa', 'play']):
            player.play()
            dispatcher.utter_message(text="▶️ Reanudado")
        
        elif any(word in msg for word in ['stop', 'termina']):
            player.stop()
            dispatcher.utter_message(text="⏹ Detenido")
        
        elif any(word in msg for word in ['sube', 'aumenta', 'más']):
            vol = min(player.audio_get_volume() + 25, 100)
            player.audio_set_volume(vol)
            dispatcher.utter_message(text=f"🔊 {vol}%")
        
        elif any(word in msg for word in ['baja', 'disminuye', 'menos']):
            vol = max(player.audio_get_volume() - 25, 0)
            player.audio_set_volume(vol)
            dispatcher.utter_message(text=f"🔉 {vol}%")
        
        else:
            dispatcher.utter_message(text="Comando no reconocido.")
        
        return []


# ------------------------------
# Acción: Fallback por defecto
# ------------------------------
class ActionDefaultFallback(Action):
    """Maneja casos donde Rasa no está seguro del intent"""
    
    def name(self) -> Text:
        return "action_default_fallback"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Obtener el mensaje original
        user_message = tracker.latest_message.get('text', '')
        
        # Mensajes de fallback amigables
        fallback_messages = [
            "No entendí. ¿Puedes repetir?",
            "No te entendí bien. Intenta de otra forma",
            "¿Podrías decirlo de otra manera?"
        ]
        
        import random
        message = random.choice(fallback_messages)
        
        dispatcher.utter_message(text=message)
        return []