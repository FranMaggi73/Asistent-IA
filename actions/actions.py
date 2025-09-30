from typing import Any, Text, Dict, List
from pathlib import Path
import os
import subprocess
import unicodedata
import string

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# ------------------------------
# Función LLM
# ------------------------------
def call_llm(prompt: str) -> str:
    # Aquí puedes integrar tu modelo real
    # Por ahora un ejemplo simulado
    return f"[LLM] Respuesta para: {prompt}"

class ActionLLMResponse(Action):
    """Responde usando tu LLM y tus reglas de prompt"""

    def name(self) -> Text:
        return "action_llm_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text", "")
        
        # Construir prompt según tus reglas
        prompt = f"""
        You are Jarvis, a concise voice assistant.
        CRITICAL RULES:
        - Maximum 2 short sentences per response
        - Be direct and avoid elaboration
        - No greetings longer than "Hola"
        - Respond in Spanish
        User said: {user_message}
        """

        llm_response = call_llm(prompt)
        dispatcher.utter_message(text=llm_response)
        return []

# ------------------------------
# Acciones de escritorio
# ------------------------------
class ActionOpenDesktopItem(Action):
    """Abre accesos directos o archivos web desde el escritorio"""
    
    def name(self) -> Text:
        return "action_open_desktop_item"
    
    def _get_desktop_path(self) -> Path:
        return Path.home() / "Escritorio"
    
    def _get_all_desktop_items(self) -> List[Path]:
        valid_extensions = ['.lnk', '.url']
        desktop = self._get_desktop_path()
        return [f for f in desktop.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    def _extract_app_name_from_message(self, message: str) -> str:
        message = message.translate(str.maketrans('', '', string.punctuation))
        ignore_words = {
            'abre','abrir','abri','abrí','ejecuta','ejecutar','lanza','lanzar',
            'inicia','iniciar','pon','poner','dame','dar','muestra','mostrar',
            'carga','cargar','el','la','los','las','mi','mis','tu','tus',
            'por','favor','quiero','necesito','quisiera','podrías','podrias',
            'puedes','me','te','le','nos','abras','ejecutes','lances',
            'acceso','programa','aplicación','aplicacion','app','de','del',
            'a','al','en','con','que','para','por'
        }
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
            if matching_words > 0:
                match_ratio = matching_words / len(search_words)
                if match_ratio >= 0.5:
                    matches.append((item, match_ratio * 70))
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
        except:
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
        if matched_item is None:
            names_str = ", ".join([item.stem for item in desktop_items[:5]])
            dispatcher.utter_message(text=f"No encontré '{app_name}'. Tienes disponible: {names_str}")
            return []
        success = self._open_file(matched_item)
        if success:
            dispatcher.utter_message(text=f"Abriendo {matched_item.stem}")
        else:
            dispatcher.utter_message(text=f"No pude abrir {matched_item.stem}")
        return []

class ActionListDesktopItems(Action):
    """Lista los accesos directos y archivos web disponibles en el escritorio"""
    
    def name(self) -> Text:
        return "action_list_desktop_items"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        desktop = Path.home() / "Escritorio"
        valid_extensions = ['.lnk', '.url']
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
