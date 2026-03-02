# main.py - Jarvis (OPTIMIZADO)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

import asyncio
import os
import sys
import signal
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from audio_listener import KeywordListener


class JarvisAssistant:
    def __init__(self):
        self.listener: Optional[KeywordListener] = None
        self.shutdown_event = asyncio.Event()
        self.preload_executor = ThreadPoolExecutor(max_workers=2)

    def setup_signal_handlers(self) -> None:
        def handler(sig, frame):
            print("\n\n⚠️  Apagando Jarvis...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def validate_files(self, wake_word: str, speaker: str, model: Optional[str] = None) -> bool:
        ok = True
        if not os.path.exists(wake_word):
            print(f"❌ Wake word no encontrado: {wake_word}")
            ok = False
        if not os.path.exists(speaker):
            print(f"❌ Speaker file no encontrado: {speaker}")
            ok = False
        if model and not os.path.exists(model):
            print(f"⚠️  Modelo de idioma no encontrado: {model} (usando inglés)")
        return ok

    def check_services(self) -> None:
        """Verifica servicios disponibles"""
        import requests

        print("\n🔍 Verificando servicios...")

        # Groq API
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                r = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    timeout=2
                )
                if r.status_code == 200:
                    print("✅ Groq API configurada (LLM ultra-rápido en la nube)")
                else:
                    print("⚠️  Groq API key inválida")
            except Exception:
                print("⚠️  Groq API no disponible (sin conexión)")
        else:
            print("⚠️  Groq API no configurada — se usará Ollama local")
            print("   Obtené API key gratis en: https://console.groq.com")

        # Ollama
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if "llama3.2:1b" in models:
                    print(f"✅ Ollama OK — modelo llama3.2:1b disponible")
                elif models:
                    print(f"⚠️  Ollama OK pero llama3.2:1b no encontrado")
                    print(f"   Modelos: {', '.join(models)}")
                    print("   Ejecutá: ollama pull llama3.2:1b")
                else:
                    print("⚠️  Ollama sin modelos. Ejecutá: ollama pull llama3.2:1b")
            else:
                print("⚠️  Ollama no responde bien")
        except Exception:
            if not groq_key:
                print("⚠️  Ollama no disponible — respuestas generales no funcionarán")
                print("   Iniciá con: ollama serve")
            else:
                print("ℹ️  Ollama no disponible pero Groq está configurado")

        # Spotify
        spotify_id = os.getenv("SPOTIFY_CLIENT_ID")
        if spotify_id:
            print("✅ Spotify credentials encontradas")
        else:
            print("⚠️  Spotify no configurado — música no disponible")
            print("   Agregá SPOTIFY_CLIENT_ID y SPOTIFY_CLIENT_SECRET en .env")

    async def preload_models(self) -> bool:
        """Pre-carga Whisper y TTS en paralelo al inicio"""
        loop = asyncio.get_event_loop()

        def load_whisper() -> bool:
            try:
                t = time.time()
                from audioFunctions import model_manager
                _ = model_manager.whisper
                print(f"✅ Whisper listo ({time.time()-t:.1f}s)")
                return True
            except Exception as e:
                print(f"⚠️  Whisper falló: {e}")
                return False

        def load_tts() -> bool:
            try:
                t = time.time()
                from audioFunctions import get_tts_model, _get_speaker_embeddings
                tts = get_tts_model()
                
                # OPTIMIZACIÓN: Pre-computar embeddings del speaker
                speaker = Path(__file__).parent / "speaker.wav"
                if speaker.exists():
                    print("🔊 Pre-computando speaker embeddings...")
                    _get_speaker_embeddings(tts, str(speaker))
                
                print(f"✅ TTS listo ({time.time()-t:.1f}s)")
                return True
            except Exception as e:
                print(f"⚠️  TTS falló: {e}")
                return False

        print("\n🚀 Cargando modelos en paralelo...")
        t_total = time.time()

        results = await asyncio.gather(
            loop.run_in_executor(self.preload_executor, load_whisper),
            loop.run_in_executor(self.preload_executor, load_tts),
            return_exceptions=True
        )

        elapsed = time.time() - t_total
        ok = sum(1 for r in results if r is True)
        print(f"📊 {ok}/2 modelos cargados en {elapsed:.1f}s")

        # Mostrar uso de VRAM después de cargar
        from audioFunctions import print_cuda_stats
        print_cuda_stats()

        return ok > 0

    async def run(self) -> int:
        base = Path(__file__).parent
        wake_word = base / "keywords" / "jarvis_es_windows_v3_0_0.ppn"
        model_file = base / "keywords" / "porcupine_params_es.pv"
        speaker   = base / "speaker.wav"

        # Banner
        print("=" * 60)
        print("🎙️  JARVIS — OPTIMIZADO (Groq + llama3.2:1b + XTTS caché)")
        print("=" * 60)

        if not self.validate_files(str(wake_word), str(speaker), str(model_file)):
            sys.exit(1)

        if not os.getenv("PICOVOICE_API_KEY"):
            print("⚠️  PICOVOICE_API_KEY no encontrada — modo fallback")

        print(f"\n📂 Wake word: {wake_word.name}")
        print(f"🔊 Speaker:   {speaker.name}")

        self.setup_signal_handlers()
        self.check_services()

        # Pre-cargar modelos (incluye warmup de embeddings)
        await self.preload_models()

        print("\n" + "=" * 60)
        print("✅ Sistema listo — decí 'Jarvis' para activar")
        print("=" * 60 + "\n")

        try:
            # Manejar model_file opcional correctamente
            model_file_str: Optional[str] = str(model_file) if model_file.exists() else None
            
            self.listener = KeywordListener(
                wake_word_file=str(wake_word),
                speaker_file=str(speaker),
                model_file=model_file_str
            )

            listen_task   = asyncio.create_task(self.listener.start_listening())
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())

            done, pending = await asyncio.wait(
                [listen_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        except KeyboardInterrupt:
            print("\n\n👋 Interrupción de teclado")
        except Exception as e:
            print(f"\n❌ Error fatal: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            if self.listener:
                self.listener.cleanup()
            self.preload_executor.shutdown(wait=False)
            print("👋 Jarvis detenido.")

        return 0


def main() -> None:
    assistant = JarvisAssistant()
    sys.exit(asyncio.run(assistant.run()))


if __name__ == "__main__":
    main()