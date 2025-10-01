from typing import Any, Dict, List, Text
import os
import subprocess
from pathlib import Path
import re

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


class ActionOpenDesktopItem(Action):
    """Abre accesos directos o archivos web desde el escritorio"""
    
    def name(self) -> Text:
        return "action_open_desktop_item"
    
    def _get_desktop_path(self) -> Path:
        """Obtiene la ruta del escritorio del usuario"""
        onedrive_desktop = Path.home() / "OneDrive" / "Escritorio"
        if onedrive_desktop.exists():
            return onedrive_desktop
        
        # Fallback a Desktop estándar en inglés
        standard_desktop = Path.home() / "Desktop"
        if standard_desktop.exists():
            return standard_desktop
        
        # Fallback a Escritorio en español
        spanish_desktop = Path.home() / "Escritorio"
        if spanish_desktop.exists():
            return spanish_desktop
        
        # Si nada funciona, retornar OneDrive por defecto
        return onedrive_desktop
    
    def _get_all_desktop_items(self) -> List[Path]:
        """Obtiene todos los accesos directos y archivos web del escritorio"""
        desktop = self._get_desktop_path()
        valid_extensions = ['.lnk', '.url', '.html', '.htm']
        
        items = [
            f for f in desktop.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        return items
    
    def _extract_app_name_from_message(self, message: str) -> str:
        """
        Extrae el nombre de la aplicación del mensaje del usuario.
        Elimina palabras comunes de comando y puntuación.
        """
        import string
        
        # Remover puntuación
        message = message.translate(str.maketrans('', '', string.punctuation))
        
        # Palabras a ignorar (más completa)
        ignore_words = {
            'abre', 'abrir', 'abri', 'abrí', 'ejecuta', 'ejecutar', 'lanza', 'lanzar',
            'inicia', 'iniciar', 'pon', 'poner', 'dame', 'dar', 'muestra', 'mostrar',
            'carga', 'cargar', 'el', 'la', 'los', 'las', 'mi', 'mis', 'tu', 'tus',
            'por', 'favor', 'quiero', 'necesito', 'quisiera', 'podrías', 'podrias',
            'puedes', 'me', 'te', 'le', 'nos', 'abras', 'ejecutes', 'lances',
            'acceso', 'programa', 'aplicación', 'aplicacion', 'app', 'de', 'del',
            'a', 'al', 'en', 'con', 'que', 'para', 'por'
        }
        
        # Convertir a minúsculas y dividir en palabras
        words = message.lower().split()
        
        # Filtrar palabras de comando, manteniendo el orden
        app_words = [w for w in words if w not in ignore_words and len(w) > 1]
        
        # Unir las palabras restantes
        app_name = ' '.join(app_words)
        
        return app_name.strip()
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparación (sin tildes, minúsculas, sin espacios extra)"""
        import unicodedata
        text = text.lower().strip()
        # Remover acentos
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        return text
    
    def _find_best_match(self, search_term: str, items: List[Path]) -> Path:
        """
        Encuentra el mejor match para el término de búsqueda.
        Usa fuzzy matching para encontrar similitudes.
        """
        search_normalized = self._normalize_text(search_term)
        search_words = search_normalized.split()
        
        # Almacenar scores de coincidencia
        matches = []
        
        for item in items:
            item_name = self._normalize_text(item.stem)
            item_words = item_name.split()
            
            # 1. Coincidencia exacta
            if search_normalized == item_name:
                return item
            
            # 2. El término de búsqueda está completamente contenido en el nombre
            if search_normalized in item_name:
                matches.append((item, 100))
                continue
            
            # 3. El nombre del archivo está contenido en la búsqueda
            if item_name in search_normalized:
                matches.append((item, 90))
                continue
            
            # 4. Todas las palabras de búsqueda están en el nombre
            if all(any(sw in iw or iw in sw for iw in item_words) for sw in search_words):
                score = 85
                matches.append((item, score))
                continue
            
            # 5. Coincidencia por palabras individuales (al menos 50%)
            matching_words = sum(1 for sw in search_words if any(sw in iw or iw in sw for iw in item_words))
            
            if matching_words > 0:
                match_ratio = matching_words / len(search_words)
                if match_ratio >= 0.5:  # Al menos 50% de coincidencia
                    score = match_ratio * 70
                    matches.append((item, score))
            
            # 6. Match por inicio de palabras (ej: "chro" → "chrome")
            for sw in search_words:
                for iw in item_words:
                    if len(sw) >= 3 and (iw.startswith(sw) or sw.startswith(iw)):
                        matches.append((item, 60))
                        break
        
        # Ordenar por score y retornar el mejor
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            best_match = matches[0]
            print(f"[DEBUG] Mejor match: '{best_match[0].stem}' (score: {best_match[1]})")
            return best_match[0]
        
        return None
    
    def _open_file(self, file_path: Path) -> bool:
        """Abre el archivo usando el manejador predeterminado de Windows"""
        try:
            os.startfile(str(file_path))
            return True
        except Exception as e:
            print(f"Error abriendo archivo: {e}")
            try:
                subprocess.Popen(['start', '', str(file_path)], shell=True)
                return True
            except:
                return False
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        # Obtener el mensaje completo del usuario
        user_message = tracker.latest_message.get('text', '')
        
        if not user_message:
            dispatcher.utter_message(text="No especificaste qué quieres abrir.")
            return []
        
        # Extraer el nombre de la aplicación del mensaje
        app_name = self._extract_app_name_from_message(user_message)
        
        if not app_name:
            dispatcher.utter_message(text="No pude identificar qué programa quieres abrir.")
            return []
        
        print(f"[DEBUG] Usuario pidió: '{user_message}'")
        print(f"[DEBUG] Nombre extraído: '{app_name}'")
        
        # Obtener todos los items del escritorio
        desktop_items = self._get_all_desktop_items()
        
        if not desktop_items:
            dispatcher.utter_message(text="No encontré accesos directos en tu escritorio.")
            return []
        
        print(f"[DEBUG] Items disponibles en escritorio: {[item.stem for item in desktop_items]}")
        
        # Buscar el mejor match
        matched_item = self._find_best_match(app_name, desktop_items)
        
        if matched_item is None:
            # Listar las opciones disponibles para ayudar
            available_names = [item.stem for item in desktop_items[:5]]
            names_str = ", ".join(available_names)
            dispatcher.utter_message(
                text=f"No encontré '{app_name}'. Tienes disponible: {names_str}"
            )
            return []
        
        print(f"[DEBUG] Archivo encontrado: '{matched_item.name}'")
        
        # Intentar abrir el archivo
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
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        desktop = Path.home() / "Escritorio"
        if not desktop.exists():
            desktop = Path.home() / "Desktop"
        if not desktop.exists():
            desktop = Path.home() / "Escritorio"
        
        valid_extensions = ['.lnk', '.url', '.html', '.htm']
        
        # Listar archivos válidos
        items = [
            f.stem for f in desktop.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        if not items:
            dispatcher.utter_message(text="No encontré accesos directos en tu escritorio.")
            return []
        
        # Limitar a los primeros 8 para no ser muy verboso
        items_text = ", ".join(items[:8])
        
        if len(items) > 8:
            dispatcher.utter_message(
                text=f"Tienes {len(items)} accesos. Los principales son: {items_text}"
            )
        else:
            dispatcher.utter_message(text=f"Tienes estos accesos: {items_text}")
        
        return []