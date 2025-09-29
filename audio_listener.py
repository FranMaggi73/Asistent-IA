# audio_listener.py
import os
import sounddevice as sd
import pvporcupine
from pvrecorder import PvRecorder
from dotenv import load_dotenv
from pathlib import Path
from whisper_helper import transcribe_audio  # Función para Whisper

load_dotenv()

PICOVOICE_API_KEY = os.getenv("PICOVOICE_API_KEY")
KEYWORD_FILE = str(Path(__file__).parent / "keywords" / "jarvis_es_windows_v3_0_0.ppn")
MODEL_FILE = str(Path(__file__).parent / "keywords" / "porcupine_params_es.pv")

class KeywordListener:
    def __init__(self, keyword_path=KEYWORD_FILE, model_path=MODEL_FILE, api_key=PICOVOICE_API_KEY, callback=None):
        self.callback = callback
        self.keyword_path = keyword_path
        self.model_path = model_path
        self.api_key = api_key
        self._stop = False

    async def start_listening(self):
        porcupine = pvporcupine.create(
            access_key=self.api_key,
            keyword_paths=[self.keyword_path],
            model_path=self.model_path
        )
        recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
        recorder.start()
        print("✅ Listening for 'Jarvis'...")

        try:
            while not self._stop:
                pcm = recorder.read()
                result = porcupine.process(pcm)
                if result >= 0:
                    print("🎯 Keyword detected!")
                    await self.handle_command()
        finally:
            recorder.stop()
            recorder.delete()
            porcupine.delete()

    async def handle_command(self):
        # Graba audio mientras hablas
        duration = 5  # segundos
        fs = 16000
        print("🎙 Recording command...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Transcribe audio directamente con Whisper
        command_text = transcribe_audio(recording, fs)
        print(f"📝 You said: {command_text}")

        # Llama al callback para enviar a Rasa
        if self.callback:
            await self.callback(command_text)

    def stop(self):
        self._stop = True
