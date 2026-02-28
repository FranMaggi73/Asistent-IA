# audio_listener.py - Pipeline principal con ejecución paralela optimizada
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

from audioFunctions import recordAudio, whisperTranscription, generateAudio
from intent_router import IntentRouter, IntentResult
from action_handlers import dispatch
from wake_word_detector import WakeWordDetector


executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="jarvis_worker")


class KeywordListener:
    def __init__(self, wake_word_file: str, speaker_file: str,
                 model_file: Optional[str] = None):
        self.speaker_file = speaker_file
        self.session_id = str(uuid.uuid4())

        self.router = IntentRouter()

        self._wake_detector: Optional[WakeWordDetector] = None
        self._wake_config = {
            "keyword_path": wake_word_file,
            "model_path": model_file,
        }

        self._stats = {"total": 0, "total_time": 0.0}

        print("✅ Listener listo\n")

    @property
    def wake_detector(self) -> WakeWordDetector:
        if self._wake_detector is None:
            kwargs = {k: v for k, v in self._wake_config.items() if v is not None}
            self._wake_detector = WakeWordDetector(**kwargs)
        return self._wake_detector

    # ─────────────────────────────────────────────
    # Loop principal
    # ─────────────────────────────────────────────

    async def start_listening(self) -> None:
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

    # ─────────────────────────────────────────────
    # Pipeline de comando optimizado
    # ─────────────────────────────────────────────

    async def handle_command(self) -> None:
        """
        Pipeline con paralelismo donde es posible:

        [grabar] ──────────────────────────────────────────┐
                                                           ▼
                                                    [transcribir]
                                                           │
                                                    [clasificar] ~0ms
                                                           │
                              ┌────────────────────────────┤
                              ▼                            ▼
                        [ejecutar acción]         (si general_question)
                              │                    [Ollama genera]
                              ▼                            │
                           [TTS] ◄─────────────────────────┘
        """
        start = time.time()
        loop = asyncio.get_event_loop()

        try:
            # ── 1. Grabar ──────────────────────────────
            print("\n🎙  Escuchando...")
            audio = await loop.run_in_executor(executor, recordAudio)

            if len(audio) == 0:
                print("⚠️  Sin audio detectado")
                return

            # ── 2. Transcribir ─────────────────────────
            text = await loop.run_in_executor(
                executor,
                partial(whisperTranscription, audio)
            )

            if not text or not text.strip():
                print("⚠️  Sin texto detectado")
                return

            # ── 3. Clasificar — ~0ms, no necesita executor ──
            intent_result: IntentResult = self.router.classify(text)

            # ── 4. Ejecutar + TTS según tipo de intent ──
            response = await self._execute(loop, intent_result)

            if not response:
                response = "No pude procesar eso."

            print(f"🤖 Jarvis: {response}")

            # ── 5. TTS ─────────────────────────────────
            if self._should_speak(intent_result.intent, response):
                await loop.run_in_executor(
                    executor,
                    partial(generateAudio, response, self.speaker_file)
                )

            # ── Stats ──────────────────────────────────
            elapsed = time.time() - start
            self._stats["total"] += 1
            self._stats["total_time"] += elapsed
            avg = self._stats["total_time"] / self._stats["total"]
            print(f"⏱️  {elapsed:.2f}s (promedio: {avg:.2f}s)\n")

        except Exception as e:
            print(f"❌ Error procesando comando: {e}")
            import traceback
            traceback.print_exc()

    async def _execute(self, loop: asyncio.AbstractEventLoop,
                       intent_result: IntentResult) -> str:
        """
        Estrategia de ejecución según intent:

        - Acciones rápidas (open_app, control_music, play_music, greet, goodbye,
          list_apps): dispatch directo, sin Ollama.

        - general_question: dispatch llama a router.generate_response() que
          es el único punto que toca Ollama.
        """

        # Acciones instantáneas: lanzar en executor pero sin overhead de Ollama
        fast_intents = {
            "open_app", "list_apps", "play_music",
            "control_music", "greet", "goodbye",
        }

        if intent_result.intent in fast_intents:
            # dispatch es síncrono pero puede hacer I/O (Spotify, filesystem)
            # lo corremos en executor para no bloquear el event loop
            response = await loop.run_in_executor(
                executor,
                partial(dispatch, intent_result)
                # No pasamos router: estas acciones nunca lo necesitan
            )
            return response

        # general_question: dispatch llama a router.generate_response
        response = await loop.run_in_executor(
            executor,
            partial(dispatch, intent_result, self.router)
        )
        return response

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _should_speak(self, intent: str, response: str) -> bool:
        """
        No hablar si la acción ya se confirmó con un símbolo
        (evita latencia de TTS para acciones triviales)
        """
        action_intents = {"open_app", "play_music", "control_music"}
        success_symbols = {"✓", "▶️", "⏸", "⏹", "🔊", "🔉", "⏭", "⏮"}

        if intent in action_intents and any(s in response for s in success_symbols):
            return False

        return True

    def cleanup(self) -> None:
        print("\n🧹 Limpiando recursos...")
        from audioFunctions import model_manager
        model_manager.unload_whisper()
        executor.shutdown(wait=False)
        print("✅ Listo")