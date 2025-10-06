# main.py - ULTRA OPTIMIZED con pre-carga paralela
import asyncio
import os
import sys
import signal
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import time

# Cargar env antes de importar módulos
load_dotenv()

from audio_listener import KeywordListener


class JarvisAssistant:
    def __init__(self):
        self.listener = None
        self.shutdown_event = asyncio.Event()
        self.preload_executor = ThreadPoolExecutor(max_workers=2)
    
    def setup_signal_handlers(self):
        """Configura manejo de señales para shutdown limpio"""
        def signal_handler(sig, frame):
            print("\n\n⚠️  Shutdown signal received...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def validate_files(self, wake_word_file: str, speaker_file: str, 
                       model_file: str = None) -> bool:
        """Valida existencia de archivos necesarios"""
        errors = []
        
        if not os.path.exists(wake_word_file):
            errors.append(f"❌ Wake word file not found: {wake_word_file}")
        
        if not os.path.exists(speaker_file):
            errors.append(f"❌ Speaker file not found: {speaker_file}")
        
        if model_file and not os.path.exists(model_file):
            print(f"⚠️  Spanish model not found: {model_file}")
            print(f"   Using default English model\n")
        
        if errors:
            for error in errors:
                print(error)
            return False
        
        return True
    
    async def preload_models_parallel(self):
        """
        Pre-carga modelos pesados en paralelo para reducir tiempo de inicio
        """
        loop = asyncio.get_event_loop()
        
        def load_tts():
            try:
                print("⏳ [TTS] Loading...")
                start = time.time()
                from audioFunctions import get_tts_model
                get_tts_model()
                elapsed = time.time() - start
                print(f"✅ [TTS] Loaded in {elapsed:.1f}s")
                return True
            except Exception as e:
                print(f"⚠️  [TTS] Failed to preload: {e}")
                return False
        
        def load_whisper():
            try:
                print("⏳ [Whisper] Loading...")
                start = time.time()
                from audioFunctions import model_manager
                _ = model_manager.whisper  # Trigger lazy loading
                elapsed = time.time() - start
                print(f"✅ [Whisper] Loaded in {elapsed:.1f}s")
                return True
            except Exception as e:
                print(f"⚠️  [Whisper] Failed to preload: {e}")
                return False
        
        # Cargar ambos modelos en paralelo
        print("\n🚀 Preloading AI models (parallel)...")
        start_total = time.time()
        
        tasks = [
            loop.run_in_executor(self.preload_executor, load_tts),
            loop.run_in_executor(self.preload_executor, load_whisper)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed_total = time.time() - start_total
        success_count = sum(1 for r in results if r is True)
        
        print(f"\n📊 Preload Summary: {success_count}/2 models loaded in {elapsed_total:.1f}s")
        
        if success_count == 0:
            print("⚠️  No models were preloaded. System may be slower on first use.")
        
        return success_count > 0
    
    async def check_rasa_availability(self):
        """Verifica disponibilidad de Rasa de forma asíncrona"""
        from rasa_client import RasaClient
        
        print("\n🔍 Checking Rasa server...")
        client = RasaClient()
        
        health = client.health_check()
        
        if health['available']:
            print(f"✅ Rasa server: OK (v{health.get('version', 'unknown')})")
            if not health.get('model_loaded'):
                print("⚠️  Warning: No model loaded in Rasa")
        else:
            print("⚠️  Rasa server: NOT AVAILABLE")
            print("   Start with: docker-compose up -d")
            print("   Or: rasa run --enable-api --cors \"*\"")
        
        client.close()
        return health['available']
    
    async def run(self):
        """Entry point principal con optimizaciones"""
        base_dir = Path(__file__).parent
        wake_word_file = base_dir / "keywords" / "jarvis_es_windows_v3_0_0.ppn"
        model_file = base_dir / "keywords" / "porcupine_params_es.pv"
        speaker_file = base_dir / "speaker.wav"
        
        # Banner inicial
        print("=" * 60)
        print("🎙️  JARVIS VOICE ASSISTANT (ULTRA OPTIMIZED v2.0)")
        print("=" * 60)
        
        # Validar archivos
        if not self.validate_files(str(wake_word_file), str(speaker_file), 
                                   str(model_file)):
            sys.exit(1)
        
        # Verificar API key
        if not os.getenv("PICOVOICE_API_KEY"):
            print("⚠️  PICOVOICE_API_KEY not found in .env")
            print("   Wake word detection will use fallback mode\n")
        
        # Configuración
        print(f"📂 Wake word: {wake_word_file.name}")
        print(f"🔊 Speaker: {speaker_file.name}")
        if model_file.exists():
            print(f"🌐 Language model: {model_file.name}")
        print(f"🔗 Rasa URL: {os.getenv('RASA_URL', 'http://localhost:5005')}")
        print("=" * 60)
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # ==============================================
        # OPTIMIZACIÓN CRÍTICA: Pre-carga paralela
        # ==============================================
        try:
            # 1. Verificar Rasa (rápido)
            rasa_task = asyncio.create_task(self.check_rasa_availability())
            
            # 2. Pre-cargar modelos (lento, en paralelo)
            preload_task = asyncio.create_task(self.preload_models_parallel())
            
            # Esperar ambas tareas
            rasa_ok, models_ok = await asyncio.gather(rasa_task, preload_task)
            
            if not rasa_ok:
                print("\n⚠️  WARNING: Continuing without Rasa (limited functionality)")
            
        except Exception as e:
            print(f"\n⚠️  Preload error: {e}")
            print("   Continuing with lazy loading...")
        
        # ==============================================
        # Iniciar sistema principal
        # ==============================================
        print("\n" + "=" * 60)
        print("✅ System ready! Say 'Jarvis' to activate")
        print("=" * 60)
        print()
        
        try:
            # Inicializar listener (ahora más rápido porque modelos ya están cargados)
            self.listener = KeywordListener(
                wake_word_file=str(wake_word_file),
                speaker_file=str(speaker_file),
                model_file=str(model_file) if model_file.exists() else None
            )
            
            # Crear tarea de listening
            listen_task = asyncio.create_task(self.listener.start_listening())
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())
            
            # Esperar a que termine alguna tarea
            done, pending = await asyncio.wait(
                [listen_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancelar tareas pendientes
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except KeyboardInterrupt:
            print("\n\n👋 Keyboard interrupt detected")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Cleanup
            if self.listener:
                self.listener.cleanup()
            
            # Shutdown executor
            self.preload_executor.shutdown(wait=False)
            
            print("👋 Jarvis stopped. Goodbye!")
        
        return 0


def main():
    """Wrapper síncrono para asyncio"""
    assistant = JarvisAssistant()
    exit_code = asyncio.run(assistant.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()