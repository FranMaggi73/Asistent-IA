# audio_listener.py - Pipeline principal sin Rasa
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from audioFunctions import recordAudio, whisperTranscription, generateAudio
from intent_router import IntentRouter
from action_handlers import dispatch
from wake_word_detector import WakeWordDetector


executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jarvis_worker")


class KeywordListener:
    def __init__(self, wake_word_file: str, speaker_file: str,
                 model_file: str = None):
        self.speaker_file = speaker_file
        self.session_id = str(uuid.uuid4())

        # Intent router (Ollama)
        self.router = IntentRouter()

        # Wake word
        self._wake_detector = None
        self._wake_config = {
            "keyword_path": wake_word_file,
            "model_path": model_file
        }

        # Estadísticas
        self._stats = {"total": 0, "total_time": 0.0}

        print("✅ Listener listo\n")

    @property
    def wake_detector(self):
        if self._wake_detector is None:
            self._wake_detector = WakeWordDetector(**self._wake_config)
        return self._wake_detector

    async def start_listening(self):
        print("=" * 50)
        print("🎙  Di 'Jarvis' para activar")
        print("=" * 50)

        loop = asyncio.get_event_loop()

        while True:
            try:
                detected = await loop.run_in_executor(
                    executor,
                    self.wake_detector.listen_for_wake_word
                )
                if detected:
                    await self.handle_command()

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Error en loop: {e}")
                await asyncio.sleep(0.5)

    async def handle_command(self):
        start = time.time()
        loop = asyncio.get_event_loop()

        try:
            # 1. Grabar
            print("\n🎙  Escuchando...")
            audio = await loop.run_in_executor(executor, recordAudio)

            if len(audio) == 0:
                print("⚠️  Sin audio")
                return

            # 2. Transcribir (Whisper en CUDA)
            text = await loop.run_in_executor(
                executor,
                partial(whisperTranscription, audio)
            )

            if not text or not text.strip():
                print("⚠️  Sin texto detectado")
                return

            # 3. Clasificar intent (Ollama)
            intent_result = await loop.run_in_executor(
                executor,
                partial(self.router.classify, text)
            )

            # 4. Ejecutar acción
            response = await loop.run_in_executor(
                executor,
                partial(dispatch, intent_result, self.router)
            )

            if not response:
                response = "No pude procesar eso."

            print(f"🤖 Jarvis: {response}")

            # 5. TTS (solo si corresponde hablar)
            if self._should_speak(intent_result.intent, response):
                await loop.run_in_executor(
                    executor,
                    partial(generateAudio, response, self.speaker_file)
                )

            # Stats
            elapsed = time.time() - start
            self._stats["total"] += 1
            self._stats["total_time"] += elapsed
            avg = self._stats["total_time"] / self._stats["total"]
            print(f"⏱️  {elapsed:.2f}s (promedio: {avg:.2f}s)\n")

        except Exception as e:
            print(f"❌ Error procesando comando: {e}")
            import traceback
            traceback.print_exc()

    def _should_speak(self, intent: str, response: str) -> bool:
        """No hablar si la acción ya se confirmó con un símbolo"""
        action_intents = {"open_app", "play_music", "control_music"}
        success_symbols = {"✓", "▶️", "⏸", "⏹", "🔊", "🔉"}

        if intent in action_intents and any(s in response for s in success_symbols):
            return False

        return True

    def cleanup(self):
        print("\n🧹 Limpiando recursos...")
        from audioFunctions import model_manager
        model_manager.unload_whisper()
        executor.shutdown(wait=False)
        print("✅ Listo")