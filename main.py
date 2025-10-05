# main.py - OPTIMIZED
import asyncio
import os
import sys
import signal
from pathlib import Path
from dotenv import load_dotenv

# Cargar env antes de importar módulos
load_dotenv()

from audio_listener import KeywordListener


class JarvisAssistant:
    def __init__(self):
        self.listener = None
        self.shutdown_event = asyncio.Event()
    
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
    
    async def run(self):
        """Entry point principal"""
        base_dir = Path(__file__).parent
        wake_word_file = base_dir / "keywords" / "jarvis_es_windows_v3_0_0.ppn"
        model_file = base_dir / "keywords" / "porcupine_params_es.pv"
        speaker_file = base_dir / "speaker.wav"
        
        # Validar archivos
        if not self.validate_files(str(wake_word_file), str(speaker_file), 
                                   str(model_file)):
            sys.exit(1)
        
        # Verificar API key
        if not os.getenv("PICOVOICE_API_KEY"):
            print("⚠️  PICOVOICE_API_KEY not found in .env")
            print("   Wake word detection will use fallback mode\n")
        
        # Banner
        print("=" * 50)
        print("🎙️  JARVIS VOICE ASSISTANT (OPTIMIZED)")
        print("=" * 50)
        print(f"📂 Wake word: {wake_word_file.name}")
        print(f"🔊 Speaker: {speaker_file.name}")
        if model_file.exists():
            print(f"🌐 Language model: {model_file.name}")
        print(f"🔗 Rasa URL: {os.getenv('RASA_URL', 'http://localhost:5005')}")
        print("=" * 50)
        print()
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        try:
            # Inicializar listener
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
            print("👋 Jarvis stopped. Goodbye!")
        
        return 0


def main():
    """Wrapper síncrono para asyncio"""
    assistant = JarvisAssistant()
    exit_code = asyncio.run(assistant.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()