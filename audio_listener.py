# audio_listener.py - OPTIMIZED
import sounddevice as sd
import torch
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dotenv import load_dotenv
import uuid

from audioFunctions import recordAudio, whisperTranscription, generateAudio
from rasa_client import RasaClient
from wake_word_detector import WakeWordDetector

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Thread pool para tareas bloqueantes (I/O, transcripción)
executor = ThreadPoolExecutor(max_workers=2)


class KeywordListener:
    def __init__(self, wake_word_file: str, speaker_file: str, model_file: str = None):
        self.wake_word_file = wake_word_file
        self.speaker_file = speaker_file
        self.session_id = str(uuid.uuid4())
        
        # Lazy loading de componentes
        self._tts = None
        self._rasa_client = None
        self._wake_word_detector = None
        
        # Cache de respuestas comunes
        self._response_cache = {}
        
        # Configuración inicial sin cargar modelos pesados
        self.wake_word_detector_config = {
            'keyword_path': wake_word_file,
            'model_path': model_file
        }
        
        print("✅ Listener initialized (lazy loading enabled)\n")
    
    @property
    def rasa_client(self):
        """Lazy loading de Rasa client"""
        if self._rasa_client is None:
            self._rasa_client = RasaClient()
            if not self._rasa_client.is_available():
                print("⚠️ Rasa server not available")
        return self._rasa_client
    
    @property
    def wake_word_detector(self):
        """Lazy loading de wake word detector"""
        if self._wake_word_detector is None:
            self._wake_word_detector = WakeWordDetector(**self.wake_word_detector_config)
        return self._wake_word_detector
    
    async def start_listening(self):
        """Loop principal optimizado con async/await"""
        print("=" * 50)
        print("🚀 Jarvis ready! Say 'Jarvis' to activate")
        print("=" * 50)
        print()
        
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Detectar wake word en thread separado (es bloqueante)
                detected = await loop.run_in_executor(
                    executor, 
                    self.wake_word_detector.listen_for_wake_word
                )
                
                if detected:
                    # Procesar comando sin bloquear el loop
                    await self.handle_command()
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Error in listening loop: {e}")
    
    async def handle_command(self):
        """Procesa comando con operaciones asíncronas"""
        loop = asyncio.get_event_loop()
        
        # 1. Grabación (bloqueante -> thread pool)
        print("\n🎙  Listening...")
        audio_data = await loop.run_in_executor(executor, recordAudio)
        
        if len(audio_data) == 0:
            print("⚠️  No audio detected")
            return
        
        # 2. Transcripción (bloqueante -> thread pool)
        text = await loop.run_in_executor(
            executor, 
            partial(whisperTranscription, audio_data)
        )
        
        if not text or not text.strip():
            print("⚠️  No speech detected")
            return
        
        print(f"📝 You: {text}")
        
        # 3. Consultar Rasa (I/O -> async)
        response_text = await self._get_rasa_response(text)
        
        if not response_text:
            response_text = "Lo siento, no pude procesar tu solicitud."
        
        print(f"🤖 Jarvis: {response_text}")
        
        # 4. TTS SOLO si es necesario (errores o conversación)
        if self._should_speak(text, response_text):
            await loop.run_in_executor(
                executor,
                partial(generateAudio, response_text, self.speaker_file)
            )
        else:
            print("✅ Action completed (silent)")
        
        print()
    
    def _should_speak(self, user_text: str, response_text: str) -> bool:
        """
        Determina si Jarvis debe responder con voz
        
        Criterios:
        - NO hablar SOLO si es una acción exitosa clara y explícita
        - SÍ hablar para todo lo demás (errores, conversación, fallbacks)
        """
        user_lower = user_text.lower()
        response_lower = response_text.lower()
        
        # Palabras clave del USUARIO que indican acción explícita
        user_action_keywords = [
            'pon', 'reproduce', 'toca', 'abre', 'abrir', 'ejecuta', 'lanza',
            'pausa', 'para', 'detén', 'reanuda', 'continúa', 'stop',
            'sube', 'baja', 'aumenta', 'disminuye', 'volumen'
        ]
        
        # Respuestas de ACCIÓN EXITOSA (símbolos inequívocos)
        action_success_symbols = ['▶️', '✓', '⏸', '⏹', '🔊', '🔉']
        
        # Verificar si el USUARIO pidió una acción específica
        user_requested_action = any(keyword in user_lower for keyword in user_action_keywords)
        
        # Verificar si la RESPUESTA confirma acción exitosa (emojis)
        response_shows_success = any(symbol in response_lower for symbol in action_success_symbols)
        
        # NO hablar SOLO si:
        # 1. Usuario pidió acción explícita Y
        # 2. Respuesta muestra símbolo de éxito
        if user_requested_action and response_shows_success:
            return False
        
        # Para TODO lo demás (conversación, errores, respuestas ambiguas): SÍ hablar
        return True
    
    async def _get_rasa_response(self, text: str) -> str:
        """Obtiene respuesta de Rasa con timeout y cache"""
        # Check cache simple
        cache_key = text.lower().strip()
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]
        
        loop = asyncio.get_event_loop()
        
        try:
            # Ejecutar request en thread pool con timeout
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(
                        self.rasa_client.send_message, 
                        text, 
                        sender_id=self.session_id
                    )
                ),
                timeout=10.0
            )
            
            # Cachear respuestas comunes (greet, goodbye)
            if response and any(word in cache_key for word in ['hola', 'adios', 'gracias']):
                self._response_cache[cache_key] = response
                # Limitar tamaño del cache
                if len(self._response_cache) > 20:
                    self._response_cache.pop(next(iter(self._response_cache)))
            
            return response
            
        except asyncio.TimeoutError:
            print("⏱️  Rasa timeout")
            return "La respuesta está tardando demasiado."
        except Exception as e:
            print(f"❌ Rasa error: {e}")
            return None
    
    def cleanup(self):
        """Limpia recursos al cerrar"""
        print("\n🧹 Cleaning up resources...")
        
        from audioFunctions import model_manager
        model_manager.unload_whisper()
        
        executor.shutdown(wait=True)
        print("✅ Cleanup complete")