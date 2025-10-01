from typing import Any, Dict, List, Text
from pathlib import Path
import os
import subprocess
import unicodedata
import string
import vlc
import yt_dlp

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# ------------------------------
# VLC Player global
# ------------------------------
vlc_instance = vlc.Instance()
player: vlc.MediaPlayer = None  # MediaPlayer global


# ------------------------------
# Acción: abrir programas / accesos del escritorio
# ------------------------------
class ActionOpenDesktopItem(Action):
    """Abre accesos directos o archivos web desde el escritorio"""

    def name(self) -> Text:
        return "action_open_desktop_item"

    def _get_desktop_path(self) -> Path:
        onedrive_desktop = Path.home() / "OneDrive" / "Escritorio"
        if onedrive_desktop.exists():
            return onedrive_desktop
        standard_desktop = Path.home() / "Desktop"
        if standard_desktop.exists():
            return standard_desktop
        spanish_desktop = Path.home() / "Escritorio"
        if spanish_desktop.exists():
            return spanish_desktop
        return onedrive_desktop

    def _get_all_desktop_items(self) -> List[Path]:
        desktop = self._get_desktop_path()
        valid_extensions = ['.lnk', '.url', '.html', '.htm']
        return [f for f in desktop.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

    def _extract_app_name_from_message(self, message: str) -> str:
        ignore_words = {
            'abre','abrir','abrí','abri','ejecuta','ejecutar','lanza','lanzar',
            'inicia','iniciar','pon','poner','dame','dar','muestra','mostrar',
            'carga','cargar','el','la','los','las','mi','mis','tu','tus',
            'por','favor','quiero','necesito','quisiera','podrías','podrias',
            'puedes','me','te','le','nos','abras','ejecutes','lances',
            'acceso','programa','aplicación','aplicacion','app','de','del',
            'a','al','en','con','que','para'
        }
        message = message.translate(str.maketrans('', '', string.punctuation))
        words = message.lower().split()
        app_words = [w for w in words if w not in ignore_words and len(w) > 1]
        return ' '.join(app_words).strip()

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    def _find_best_match(self, search_term: str, items: List[Path]) -> Path:
        search_normalized = self._normalize_text(search_term)
        search_words = search_normalized.split()
        matches = []

        for item in items:
            item_name = self._normalize_text(item.stem)
            item_words = item_name.split()

            if search_normalized == item_name:
                return item
            if search_normalized in item_name:
                matches.append((item, 100))
                continue
            if item_name in search_normalized:
                matches.append((item, 90))
                continue
            if all(any(sw in iw or iw in sw for iw in item_words) for sw in search_words):
                matches.append((item, 85))
                continue
            matching_words = sum(1 for sw in search_words if any(sw in iw or iw in sw for iw in item_words))
            if matching_words > 0 and matching_words / len(search_words) >= 0.5:
                matches.append((item, matching_words / len(search_words) * 70))
            for sw in search_words:
                for iw in item_words:
                    if len(sw) >= 3 and (iw.startswith(sw) or sw.startswith(iw)):
                        matches.append((item, 60))
                        break

        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0][0]
        return None

    def _open_file(self, file_path: Path) -> bool:
        try:
            os.startfile(str(file_path))
            return True
        except Exception:
            try:
                subprocess.Popen(['start', '', str(file_path)], shell=True)
                return True
            except:
                return False

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get('text', '')
        if not user_message:
            dispatcher.utter_message(text="No especificaste qué quieres abrir.")
            return []

        app_name = self._extract_app_name_from_message(user_message)
        if not app_name:
            dispatcher.utter_message(text="No pude identificar qué programa quieres abrir.")
            return []

        desktop_items = self._get_all_desktop_items()
        if not desktop_items:
            dispatcher.utter_message(text="No encontré accesos directos en tu escritorio.")
            return []

        matched_item = self._find_best_match(app_name, desktop_items)
        if not matched_item:
            dispatcher.utter_message(text=f"No encontré '{app_name}'.")
            return []

        success = self._open_file(matched_item)
        if success:
            dispatcher.utter_message(text=f"Abriendo {matched_item.stem}")
        else:
            dispatcher.utter_message(text=f"No pude abrir {matched_item.stem}")
        return []


# ------------------------------
# Acción: listar accesos del escritorio
# ------------------------------
class ActionListDesktopItems(Action):
    """Lista los accesos directos y archivos web disponibles en el escritorio"""

    def name(self) -> Text:
        return "action_list_desktop_items"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        desktop = Path.home() / "Escritorio"
        if not desktop.exists():
            desktop = Path.home() / "Desktop"
        valid_extensions = ['.lnk', '.url', '.html', '.htm']

        items = [f.stem for f in desktop.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

        if not items:
            dispatcher.utter_message(text="No encontré accesos directos en tu escritorio.")
            return []

        items_text = ", ".join(items[:8])
        if len(items) > 8:
            dispatcher.utter_message(text=f"Tienes {len(items)} accesos. Los principales son: {items_text}")
        else:
            dispatcher.utter_message(text=f"Tienes estos accesos: {items_text}")
        return []


# ------------------------------
# Acción: reproducir música
# ------------------------------
class ActionPlayMusic(Action):
    """Reproduce una canción de YouTube usando VLC"""

    def name(self) -> Text:
        return "action_play_music"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        global player

        query = tracker.latest_message.get("text", "")
        if not query:
            dispatcher.utter_message(text="No especificaste la canción a reproducir.")
            return []

        ydl_opts = {'format': 'bestaudio/best', 'quiet': True, 'noplaylist': True}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch:{query}", download=False)['entries'][0]
                url = info['url']
                title = info.get('title', 'Canción')
        except Exception as e:
            dispatcher.utter_message(text=f"No pude encontrar la canción: {e}")
            return []

        if player and player.is_playing():
            player.stop()

        media = vlc_instance.media_new(url)
        player = vlc_instance.media_player_new()
        player.set_media(media)
        player.play()

        dispatcher.utter_message(text=f"Reproduciendo: {title}")
        return []


# ------------------------------
# Acción: controlar música
# ------------------------------
class ActionControlMusic(Action):
    """Controla la reproducción: pausa, play, stop, volumen"""

    def name(self) -> Text:
        return "action_control_music"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        global player

        if not player:
            dispatcher.utter_message(text="No hay música reproduciéndose actualmente.")
            return []

        msg = tracker.latest_message.get("text", "").lower()

        if "pausa" in msg or "detén" in msg or "para" in msg:
            player.pause()
            dispatcher.utter_message(text="Música pausada.")
        elif "reanuda" in msg or "continúa" in msg:
            player.play()
            dispatcher.utter_message(text="Música reanudada.")
        elif "stop" in msg or "detén" in msg or "termina" in msg:
            player.stop()
            dispatcher.utter_message(text="Música detenida.")
        elif "sube" in msg or "aumenta" in msg or "más volumen" in msg:
            vol = min(player.audio_get_volume() + 25, 100)
            player.audio_set_volume(vol)
            dispatcher.utter_message(text=f"Volumen subido a {vol}%")
        elif "baja" in msg or "disminuye" in msg or "menos volumen" in msg:
            vol = max(player.audio_get_volume() - 25, 0)
            player.audio_set_volume(vol)
            dispatcher.utter_message(text=f"Volumen bajado a {vol}%")
        else:
            dispatcher.utter_message(text="No entendí el comando de música.")

        return []
