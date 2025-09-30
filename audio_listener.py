
import sounddevice as sd
import torch
from TTS.api import TTS
from audioFunctions import recordAudio, whisperTranscription
from rasa_client import RasaClient
from wake_word_detector import WakeWordDetector
from dotenv import load_dotenv
import warnings
import uuid

# Cargar variables de entorno
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("⚠️ Running on CPU, may be slower.\n")

# Configuración de safe globals y monkey patching
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
    
    torch.serialization.add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        VoiceBpeTokenizer
    ])
except ImportError as e:
    print(f"⚠️ Could not import some XTTS classes: {e}")

original_load = torch.load

def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load
warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

# Cargar modelo TTS
print("Loading TTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device=="cuda"))
print("✅ TTS model loaded successfully!\n")

torch.load = original_load


class KeywordListener:
    def __init__(self, wake_word_file: str, speaker_file: str, model_file: str = None):
        self.wake_word_file = wake_word_file
        self.speaker_file = speaker_file
        self.tts = tts
        self.rasa_client = RasaClient()  # Usará RASA_URL del .env
        self.session_id = str(uuid.uuid4())
        
        # Inicializar detector de wake word con modelo en español
        self.wake_word_detector = WakeWordDetector(
            keyword_path=wake_word_file,
            model_path=model_file  # porcupine_params_es.pv
        )
        
        # Verificar si Rasa está disponible
        if self.rasa_client.is_available():
            print("✅ Rasa server is running\n")
        else:
            print("⚠️ Rasa server not detected")
            print("   Make sure Docker container is running:")
            print("   docker-compose up -d\n")

    async def start_listening(self):
        """Loop principal de escucha"""
        print("=" * 50)
        print("🚀 Assistant ready! Say 'Jarvis' to activate")
        print("=" * 50)
        print()
        
        while True:
            try:
                # Detectar wake word
                detected = self.wake_word_detector.listen_for_wake_word()
                
                if detected:
                    await self.handle_command()
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Error in listening loop: {e}")
                import traceback
                traceback.print_exc()

    async def handle_command(self):
        """Maneja el comando del usuario después de detectar wake word"""
        print("\n🎙  Listening to your command...")
        audio_data = recordAudio()
        
        # Transcripción
        text = whisperTranscription(audio_data)
        
        if not text or text.strip() == "":
            print("⚠️  No speech detected, try again")
            return
        
        print(f"📝 You: {text}")

        # Enviar a Rasa
        response_text = self.rasa_client.send_message(text, sender_id=self.session_id)
        
        # Fallback si Rasa no responde
        if response_text is None:
            response_text = "Lo siento, no pude procesar tu solicitud en este momento."
        
        print(f"🤖 Jarvis: {response_text}")
        
        # Generar respuesta TTS
        try:
            wav = self.tts.tts(
                text=response_text, 
                speaker_wav=self.speaker_file, 
                language="es"
            )
            
            if wav is None or len(wav) == 0:
                print("❌ TTS returned empty audio")
                return
                
            sd.play(wav, 24000)
            sd.wait()
            print("✅ Response completed\n")
            
        except Exception as e:
            print(f"❌ Error generating TTS: {e}")