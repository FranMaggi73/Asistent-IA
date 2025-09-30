# rasa_client.py
# Author: Claude Petit-Frere
# Date: 11/14/24
# Desc: Client for communicating with Rasa server

import requests
import os
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class RasaClient:
    def __init__(self, rasa_url: Optional[str] = None):
        """
        Inicializa el cliente de Rasa
        
        Args:
            rasa_url: URL del servidor Rasa (por defecto usa .env)
        """
        # Usar URL del .env o default
        self.rasa_url = rasa_url or os.getenv("RASA_URL", "http://localhost:5005")
        self.webhook_url = f"{self.rasa_url}/webhooks/rest/webhook"
        print(f"🔗 Rasa URL: {self.rasa_url}")
        
    def send_message(self, message: str, sender_id: str = "user") -> Optional[str]:
        """
        Envía un mensaje a Rasa y retorna la respuesta
        
        Args:
            message: Texto del usuario
            sender_id: ID del usuario (para mantener contexto)
            
        Returns:
            Respuesta de Rasa o None si hay error
        """
        try:
            payload = {
                "sender": sender_id,
                "message": message
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                rasa_responses = response.json()
                
                # Rasa puede devolver múltiples respuestas
                if rasa_responses and len(rasa_responses) > 0:
                    # Concatenar todas las respuestas de texto
                    texts = [r.get("text", "") for r in rasa_responses if "text" in r]
                    return " ".join(texts) if texts else None
                else:
                    return "No entendí tu pregunta. ¿Podrías reformularla?"
            else:
                print(f"❌ Rasa error: HTTP {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ No se pudo conectar con Rasa. ¿Está el contenedor corriendo?")
            print(f"   Verifica: docker ps | grep rasa")
            print(f"   URL configurada: {self.rasa_url}")
            return "No puedo conectarme al servidor. Por favor, verifica la conexión."
        except requests.exceptions.Timeout:
            print("❌ Timeout al conectar con Rasa")
            return "La respuesta está tardando demasiado."
        except Exception as e:
            print(f"❌ Error al comunicarse con Rasa: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Verifica si el servidor de Rasa está disponible
        
        Returns:
            True si Rasa está disponible, False en caso contrario
        """
        try:
            response = requests.get(f"{self.rasa_url}/status", timeout=3)
            return response.status_code == 200
        except:
            return False


# Test del cliente (opcional)
if __name__ == "__main__":
    print("Testing Rasa Client...")
    client = RasaClient()
    
    if client.is_available():
        print("✅ Rasa server is available")
        
        # Enviar mensaje de prueba
        response = client.send_message("hola", sender_id="test_user")
        print(f"Response: {response}")
    else:
        print("❌ Rasa server is not available")