# rasa_client.py - OPTIMIZED
import requests
import os
from typing import Optional
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()


class RasaClient:
    def __init__(self, rasa_url: Optional[str] = None):
        """
        Cliente Rasa optimizado con connection pooling y retries
        
        Args:
            rasa_url: URL del servidor Rasa (por defecto usa .env)
        """
        self.rasa_url = rasa_url or os.getenv("RASA_URL", "http://localhost:5005")
        self.webhook_url = f"{self.rasa_url}/webhooks/rest/webhook"
        
        # Session con connection pooling
        self.session = self._create_session()
        
        # Cache de disponibilidad
        self._is_available = None
        
        print(f"🔗 Rasa URL: {self.rasa_url}")
    
    def _create_session(self) -> requests.Session:
        """Crea sesión HTTP con retry logic y pooling"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=0.3,
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers por defecto
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        return session
    
    def send_message(self, message: str, sender_id: str = "user") -> Optional[str]:
        """
        Envía mensaje a Rasa con manejo optimizado
        
        Args:
            message: Texto del usuario
            sender_id: ID del usuario
            
        Returns:
            Respuesta de Rasa o None si hay error
        """
        if not message or not message.strip():
            return None
        
        try:
            payload = {
                "sender": sender_id,
                "message": message.strip()
            }
            
            response = self.session.post(
                self.webhook_url,
                json=payload,
                timeout=20  # Reducido de 10 a 8
            )
            
            if response.status_code == 200:
                rasa_responses = response.json()
                
                if not rasa_responses:
                    return "No entendí tu pregunta. ¿Podrías reformularla?"
                
                # Concatenar respuestas
                texts = [
                    r.get("text", "").strip() 
                    for r in rasa_responses 
                    if r.get("text")
                ]
                
                return " ".join(texts) if texts else None
            
            elif response.status_code == 404:
                print(f"❌ Rasa endpoint not found: {self.webhook_url}")
                return "El servidor no está configurado correctamente."
            
            else:
                print(f"❌ Rasa HTTP {response.status_code}: {response.text[:100]}")
                return None
                
        except requests.exceptions.ConnectionError as e:
            if self._is_available is not False:  # Solo mostrar una vez
                print(f"❌ Cannot connect to Rasa at {self.rasa_url}")
                print(f"   Check: docker ps | grep rasa")
                self._is_available = False
            return "No puedo conectarme al servidor."
        
        except requests.exceptions.Timeout:
            print("⏱️  Rasa timeout (>8s)")
            return "La respuesta está tardando demasiado."
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return None
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def is_available(self, use_cache: bool = True) -> bool:
        """
        Verifica disponibilidad de Rasa
        
        Args:
            use_cache: Si es True, usa resultado cacheado
            
        Returns:
            True si Rasa está disponible
        """
        if use_cache and self._is_available is not None:
            return self._is_available
        
        try:
            response = self.session.get(
                f"{self.rasa_url}/status", 
                timeout=20
            )
            self._is_available = (response.status_code == 200)
            return self._is_available
        
        except Exception:
            self._is_available = False
            return False
    
    def health_check(self) -> dict:
        """
        Health check completo del servidor
        
        Returns:
            Dict con información de salud
        """
        result = {
            'available': False,
            'status_code': None,
            'version': None,
            'model_loaded': False
        }
        
        try:
            # Check /status endpoint
            response = self.session.get(f"{self.rasa_url}/status", timeout=20)
            result['status_code'] = response.status_code
            
            if response.status_code == 200:
                result['available'] = True
                data = response.json()
                result['version'] = data.get('version')
                result['model_loaded'] = data.get('model_file') is not None
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def close(self):
        """Cierra la sesión HTTP"""
        if self.session:
            self.session.close()
    
    def __del__(self):
        """Cleanup al destruir el objeto"""
        self.close()


# Test del cliente
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Rasa Client (Optimized)")
    print("=" * 50)
    
    client = RasaClient()
    
    # Health check
    print("\n📊 Health Check:")
    health = client.health_check()
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    if health['available']:
        print("\n✅ Rasa server is available")
        
        # Test de latencia
        import time
        print("\n⏱️  Latency Test:")
        
        test_messages = ["hola", "qué tal", "adiós"]
        for msg in test_messages:
            start = time.time()
            response = client.send_message(msg, sender_id="test_user")
            latency = (time.time() - start) * 1000
            
            print(f"   '{msg}' -> {latency:.0f}ms")
            print(f"   Response: {response}")
    else:
        print("\n❌ Rasa server is not available")
        print("   Start with: docker-compose up -d")
    
    client.close()