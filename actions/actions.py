from typing import Any, Dict, List, Text
from pathlib import Path
import os
import subprocess
import unicodedata
import string
import vlc
import yt_dlp
from functools import lru_cache
import threading
import requests
import json

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


# --- Ollama Client ---
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        
    def is_available(self) -> bool:
        """Verifica si Ollama está disponible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=20)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Genera respuesta usando Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("⏱️  Ollama timeout")
            return None
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return None
    
    def chat(self, message: str, context: List[Dict] = None) -> str:
        """Chat con contexto usando Ollama"""
        try:
            messages = context or []
            messages.append({"role": "user", "content": message})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False
            }
            
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                return None
                
        except Exception as e:
            print(f"❌ Ollama chat error: {e}")
            return None


# Instancia global de Ollama
ollama_client = OllamaClient()


# --- VLC Player singleton thread-safe ---
class VLCManager:
    _instance = None
    _lock = threading.Lock()
    _vlc_instance = None
    _player = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def instance(self):
        if self._vlc_instance is None:
            with self._lock:
                if self._vlc_instance is None:
                    self._vlc_instance = vlc.Instance('--quiet')
        return self._vlc_instance
    
    @property
    def player(self):
        return self._player
    
    @player.setter
    def player(self, value):
        with self._lock:
            self._player = value


vlc_manager = VLCManager()


# --- OPTIMIZACIÓN: Cache persistente del escritorio ---
@lru_cache(maxsize=1)
def get_desktop_path() -> Path:
    """Cache permanente de la ruta del escritorio"""
    for desktop_path in [
        Path.home() / "OneDrive" / "Escritorio",
        Path.home() / "Desktop",
        Path.home() / "Escritorio"
    ]:
        if desktop_path.exists():
            return desktop_path
    return Path.home() / "Desktop"


class DesktopItemCache:
    """Cache inteligente con invalidación automática"""
    def __init__(self, ttl=600):
        self._cache = None
        self._last_update = 0
        self._lock = threading.Lock()
        self.ttl = ttl
        self._normalized_names = {}
    
    def get_items(self, force_refresh=False):
        import time
        current_time = time.time()
        
        with self._lock:
            if force_refresh or self._cache is None or (current_time - self._last_update) > self.ttl:
                desktop = get_desktop_path()
                valid_extensions = {'.lnk', '.url', '.html', '.htm', '.exe'}
                
                self._cache = [
                    f for f in desktop.iterdir() 
                    if f.is_file() and f.suffix.lower() in valid_extensions
                ]
                
                self._normalized_names = {
                    f: self._normalize_text(f.stem) for f in self._cache
                }
                
                self._last_update = current_time
                print(f"🔄 Desktop cache refreshed ({len(self._cache)} items)")
            
            return self._cache, self._normalized_names
    
    def _normalize_text(self, text: str) -> str:
        """Normalización rápida"""
        text = text.lower().strip()
        return ''.join(
            c for c in unicodedata.normalize('NFD', text) 
            if unicodedata.category(c) != 'Mn'
        )


desktop_cache = DesktopItemCache()


# --- ActionOpenDesktopItem ---
class ActionOpenDesktopItem(Action):
    def name(self) -> Text:
        return "action_open_desktop_item"
    
    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        return ''.join(
            c for c in unicodedata.normalize('NFD', text) 
            if unicodedata.category(c) != 'Mn'
        )
    
    def _extract_app_name(self, message: str) -> str:
        ignore_words = {
            'abre','abrir','abrí','abri','ejecuta','ejecutar','lanza',
            'inicia','iniciar','pon','el','la','mi','por','favor',
            'quiero','necesito','programa','aplicación','app','abriendo'
        }
        
        message = message.translate(str.maketrans('', '', string.punctuation))
        words = [
            w for w in message.lower().split() 
            if w not in ignore_words and len(w) > 1
        ]
        return ' '.join(words).strip()
    
    def _find_best_match(self, search_term: str, items: List[Path], 
                         normalized_cache: Dict[Path, str]) -> Path:
        search_norm = self._normalize_text(search_term)
        search_words = set(search_norm.split())
        
        best_match = None
        best_score = 0
        
        for item in items:
            item_name = normalized_cache[item]
            item_words = set(item_name.split())
            
            if search_norm == item_name:
                return item
            
            if search_norm in item_name:
                score = 100
            elif item_name in search_norm:
                score = 90
            elif search_norm.startswith(item_name[:3]):
                score = 85
            else:
                common = search_words & item_words
                if len(search_words) > 0:
                    score = (len(common) / len(search_words)) * 80
                else:
                    score = 0
            
            if score > best_score:
                best_score = score
                best_match = item
        
        return best_match if best_score > 40 else None
    
    def _open_file(self, file_path: Path) -> bool:
        try:
            if os.name == 'nt':
                os.startfile(str(file_path))
            else:
                subprocess.Popen(['xdg-open', str(file_path)], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
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
        
        desktop_items, normalized_names = desktop_cache.get_items()
        if not desktop_items:
            dispatcher.utter_message(text="No hay accesos en el escritorio.")
            return []
        
        matched_item = self._find_best_match(app_name, desktop_items, normalized_names)
        if not matched_item:
            dispatcher.utter_message(text=f"No encontré '{app_name}'.")
            return []
        
        success = self._open_file(matched_item)
        if success:
            dispatcher.utter_message(text=f"✓ {matched_item.stem}")
        else:
            dispatcher.utter_message(text=f"No pude abrir {matched_item.stem}")
        
        return []


# --- ActionListDesktopItems ---
class ActionListDesktopItems(Action):
    def name(self) -> Text:
        return "action_list_desktop_items"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text', '').lower()
        list_keywords = ['lista', 'listar', 'muestra', 'qué tengo', 'qué hay', 
                        'qué accesos', 'qué programas', 'cuáles son']
        
        if not any(keyword in user_message for keyword in list_keywords):
            dispatcher.utter_message(text="No entendí. ¿Puedes repetir?")
            return []
        
        items, _ = desktop_cache.get_items()
        
        if not items:
            dispatcher.utter_message(text="No hay accesos en el escritorio.")
            return []
        
        items_names = sorted([f.stem for f in items])
        items_text = ", ".join(items_names[:8])
        
        if len(items_names) > 8:
            msg = f"Tienes {len(items_names)} accesos: {items_text}, y más"
        else:
            msg = f"Tienes: {items_text}"
        
        dispatcher.utter_message(text=msg)
        return []


# --- ActionPlayMusic ---
class ActionPlayMusic(Action):
    def name(self) -> Text:
        return "action_play_music"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        query = tracker.latest_message.get("text", "")
        if not query:
            dispatcher.utter_message(text="No especificaste la canción.")
            return []
        
        ydl_opts = {
            'format': 'bestaudio/worst',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'socket_timeout': 20,
            'retries': 1,
            'extract_flat': False,
            'skip_download': False,
            'nocheckcertificate': True,
            'prefer_insecure': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                
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
        
        if vlc_manager.player and vlc_manager.player.is_playing():
            vlc_manager.player.stop()
        
        try:
            media = vlc_manager.instance.media_new(url)
            vlc_manager.player = vlc_manager.instance.media_player_new()
            vlc_manager.player.set_media(media)
            vlc_manager.player.play()
            
            dispatcher.utter_message(text=f"▶️ {title}")
        except Exception as e:
            print(f"VLC error: {e}")
            dispatcher.utter_message(text="Error al reproducir.")
        
        return []


# --- ActionControlMusic ---
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


# --- NUEVA: ActionLLMResponse con Ollama ---
class ActionLLMResponse(Action):
    def name(self) -> Text:
        return "action_llm_response"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text', '')
        
        if not user_message:
            dispatcher.utter_message(text="No entendí tu pregunta.")
            return []
        
        # Verificar disponibilidad de Ollama
        if not ollama_client.is_available():
            dispatcher.utter_message(
                text="El servicio de IA no está disponible en este momento."
            )
            return []
        
        # Crear prompt con contexto de Jarvis
        system_prompt = (
            "Eres Jarvis, un asistente de voz inteligente y conciso. "
            "Responde de manera breve, útil y en español. "
            "Mantén tus respuestas en 2-3 oraciones máximo para que sean fáciles de escuchar. "
            "Si no sabes algo, admítelo honestamente."
        )
        
        full_prompt = f"{system_prompt}\n\nUsuario: {user_message}\nJarvis:"
        
        # Generar respuesta
        print(f"🤖 Consultando a Ollama: {user_message}")
        response = ollama_client.generate(
            prompt=full_prompt,
            max_tokens=50,  # Respuestas cortas
            temperature=0.7
        )
        
        if response:
            # Limitar longitud para TTS
            if len(response) > 300:
                response = response[:297] + "..."
            
            dispatcher.utter_message(text=response)
            print(f"💬 Respuesta: {response}")
        else:
            dispatcher.utter_message(
                text="No pude generar una respuesta en este momento."
            )
        
        return []


# --- ActionDefaultFallback - Ahora redirige a LLM ---
class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text', '')
        
        # Intentar con Ollama si está disponible
        if ollama_client.is_available():
            print(f"🔄 Fallback -> LLM: {user_message}")
            
            # Usar ActionLLMResponse
            llm_action = ActionLLMResponse()
            return llm_action.run(dispatcher, tracker, domain)
        
        # Fallback tradicional si Ollama no disponible
        fallback_messages = [
            "No entendí. ¿Puedes repetir?",
            "No te entendí bien. Intenta de otra forma",
            "¿Podrías decirlo de otra manera?"
        ]
        
        import random
        message = random.choice(fallback_messages)
        
        dispatcher.utter_message(text=message)
        return []