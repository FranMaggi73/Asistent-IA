# audio_listener.py - ULTRA OPTIMIZED
import sounddevice as sd
import torch
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dotenv import load_dotenv
import uuid
import time

from audioFunctions import recordAudio, whisperTranscription, generateAudio
from rasa_client import RasaClient
from wake_word_detector import WakeWordDetector

# NUEVO: Sistema de prioridades (opcional, descomentar si usas priority_queue.py)
# from priority_queue import PriorityTaskQueue, TaskPriority

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Thread pool optimizado (reducido a 2 workers)
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio_worker")


class KeywordListener:
    def __init__(self, wake_word_file: str, speaker_file: str, model_file: str = None):
        self.wake_word_file = wake_word_file
        self.speaker_file = speaker_file
        self.session_id = str(uuid.uuid4())
        
        # Lazy loading de componentes
        self._tts = None
        self._rasa_client = None
        self._wake_word_detector = None
        
        # Cache de respuestas comunes (aumentado)
        self._response_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Configuración inicial
        self.wake_word_detector_config = {
            'keyword_path': wake_word_file,
            'model_path': model_file
        }
        
        # NUEVO: Estadísticas de rendimiento
        self._stats = {
            'commands_processed': 0,
            'avg_response_time': 0,
            'total_time': 0
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
        """Loop principal optimizado"""
        print("=" * 50)
        print("🚀 Jarvis ready! Say 'Jarvis' to activate")
        print("=" * 50)
        print()
        
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Detectar wake word
                detected = await loop.run_in_executor(
                    executor, 
                    self.wake_word_detector.listen_for_wake_word
                )
                
                if detected:
                    await self.handle_command()
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Error in listening loop: {e}")
    
    async def handle_command(self):
        """Pipeline de procesamiento optimizado con métricas"""
        start_time = time.time()
        loop = asyncio.get_event_loop()
        
        try:
            # 1. GRABACIÓN (async)
            print("\n🎙  Listening...")
            audio_data = await loop.run_in_executor(executor, recordAudio)
            
            if len(audio_data) == 0:
                print("⚠️  No audio detected")
                return
            
            # 2. TRANSCRIPCIÓN (async)
            text = await loop.run_in_executor(
                executor, 
                partial(whisperTranscription, audio_data)
            )
            
            if not text or not text.strip():
                print("⚠️  No speech detected")
                return
            
            print(f"📝 You: {text}")
            
            # 3. PROCESAMIENTO RASA (async con cache)
            response_text = await self._get_rasa_response_cached(text)
            
            if not response_text:
                response_text = "Lo siento, no pude procesar tu solicitud."
            
            print(f"🤖 Jarvis: {response_text}")
            
            # 4. TTS CONDICIONAL (async)
            if self._should_speak(text, response_text):
                await loop.run_in_executor(
                    executor,
                    partial(generateAudio, response_text, self.speaker_file)
                )
            else:
                print("✅ Action completed (silent)")
            
            # Actualizar estadísticas
            elapsed = time.time() - start_time
            self._update_stats(elapsed)
            
            print(f"⏱️  Response time: {elapsed:.2f}s")
            print()
            
        except Exception as e:
            print(f"❌ Command processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def _should_speak(self, user_text: str, response_text: str) -> bool:
        """Determina si debe hablar (lógica mejorada)"""
        user_lower = user_text.lower()
        response_lower = response_text.lower()
        
        # Palabras clave de acción del usuario
        user_action_keywords = [
            'pon', 'reproduce', 'toca', 'abre', 'abrir', 'ejecuta', 'lanza',
            'pausa', 'para', 'detén', 'reanuda', 'continúa', 'stop',
            'sube', 'baja', 'aumenta', 'disminuye', 'volumen'
        ]
        
        # Símbolos de éxito en respuesta
        action_success_symbols = ['▶️', '✓', '⏸', '⏹', '🔊', '🔉']
        
        # Verificaciones
        user_requested_action = any(keyword in user_lower for keyword in user_action_keywords)
        response_shows_success = any(symbol in response_lower for symbol in action_success_symbols)
        
        # NO hablar solo si acción exitosa confirmada
        if user_requested_action and response_shows_success:
            return False
        
        return True
    
    async def _get_rasa_response_cached(self, text: str) -> str:
        """Obtiene respuesta con cache inteligente"""
        # Normalizar para cache
        cache_key = text.lower().strip()
        
        # Check cache
        if cache_key in self._response_cache:
            self._cache_hits += 1
            print(f"💾 Cache hit ({self._cache_hits}/{self._cache_hits + self._cache_misses})")
            return self._response_cache[cache_key]
        
        self._cache_misses += 1
        
        # Query Rasa
        loop = asyncio.get_event_loop()
        
        try:
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
            
            # Cachear respuestas comunes y exitosas
            if response:
                # Cachear saludos, despedidas
                if any(word in cache_key for word in ['hola', 'adios', 'gracias', 'buenas']):
                    self._response_cache[cache_key] = response
                
                # Cachear acciones exitosas (con símbolos)
                elif any(symbol in response for symbol in ['✓', '▶️', '⏸', '⏹']):
                    # Cachear solo el patrón, no la respuesta específica
                    pattern = self._extract_action_pattern(cache_key)
                    if pattern:
                        self._response_cache[pattern] = response
                
                # Limitar tamaño del cache
                if len(self._response_cache) > 50:
                    # Eliminar entrada más antigua
                    self._response_cache.pop(next(iter(self._response_cache)))
            
            return response
            
        except asyncio.TimeoutError:
            print("⏱️  Rasa timeout")
            return "La respuesta está tardando demasiado."
        except Exception as e:
            print(f"❌ Rasa error: {e}")
            return None
    
    def _extract_action_pattern(self, text: str) -> str:
        """Extrae patrón de acción para cache"""
        action_words = ['abre', 'reproduce', 'pon', 'pausa', 'sube', 'baja']
        for word in action_words:
            if word in text:
                return word  # Retorna solo la acción base
        return None
    
    def _update_stats(self, elapsed: float):
        """Actualiza estadísticas de rendimiento"""
        self._stats['commands_processed'] += 1
        self._stats['total_time'] += elapsed
        self._stats['avg_response_time'] = (
            self._stats['total_time'] / self._stats['commands_processed']
        )
    
    def print_stats(self):
        """Imprime estadísticas de rendimiento"""
        print("\n" + "=" * 50)
        print("📊 Performance Statistics")
        print("=" * 50)
        print(f"Commands processed: {self._stats['commands_processed']}")
        print(f"Average response time: {self._stats['avg_response_time']:.2f}s")
        print(f"Cache hit rate: {self._cache_hits}/{self._cache_hits + self._cache_misses}")
        if self._cache_hits + self._cache_misses > 0:
            hit_rate = (self._cache_hits / (self._cache_hits + self._cache_misses)) * 100
            print(f"Cache efficiency: {hit_rate:.1f}%")
        print("=" * 50)
    
    def cleanup(self):
        """Limpia recursos al cerrar"""
        print("\n🧹 Cleaning up resources...")
        
        # Mostrar estadísticas finales
        if self._stats['commands_processed'] > 0:
            self.print_stats()
        
        from audioFunctions import model_manager
        model_manager.unload_whisper()
        
        executor.shutdown(wait=True)
        print("✅ Cleanup complete")