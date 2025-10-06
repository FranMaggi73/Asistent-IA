import warnings
import importlib
from functools import lru_cache
import numpy as np
import time
import torch
import sounddevice as sd
from threading import Lock

# COM initialization (Windows)
try:
    import pythoncom
    try:
        pythoncom.CoInitialize()
        print("✅ COM initialized")
    except Exception as e:
        print(f"⚠️  COM CoInitialize failed: {e}")
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"


# --- OPTIMIZED ModelManager with threading safety ---
class ModelManager:
    _instance = None
    _whisper_model = None
    _lock = Lock()  # Thread-safe singleton

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check locking
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def whisper(self):
        if self._whisper_model is None:
            with self._lock:
                if self._whisper_model is None:  # Double-check
                    print("⏳ Loading Whisper model...")
                    import whisper
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        # OPTIMIZACIÓN: Usar modelo más rápido si CUDA no disponible
                        model_size = "large" if device == "cuda" else "medium"
                        self._whisper_model = whisper.load_model(model_size).to(device)
                    print(f"✅ Whisper loaded ({model_size})")
        return self._whisper_model

    def unload_whisper(self):
        if self._whisper_model is not None:
            with self._lock:
                if self._whisper_model is not None:
                    del self._whisper_model
                    self._whisper_model = None
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    print("🗑️  Whisper unloaded")


model_manager = ModelManager()


# --- OPTIMIZED Whisper transcription ---
def whisperTranscription(audio_data, language='es'):
    """Transcripción optimizada con resampling eficiente"""
    try:
        # Normalización rápida
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # OPTIMIZACIÓN: Resample solo si es necesario
        if len(audio_float) < 16000:  # Menos de 1 segundo
            print("⚠️  Audio muy corto, saltando transcripción")
            return ""
        
        # Resample eficiente usando scipy
        from scipy import signal
        orig_sr = 44100
        target_sr = 16000
        num_samples = int(len(audio_float) * target_sr / orig_sr)
        resampled = signal.resample(audio_float, num_samples)
        
        if resampled.ndim > 1:
            resampled = resampled.mean(axis=1)
        
        model = model_manager.whisper
        
        # OPTIMIZACIÓN: Parámetros ajustados para velocidad
        result = model.transcribe(
            resampled,
            language=language,
            fp16=(device == "cuda"),
            verbose=False,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            # NUEVO: Usar beam search más rápido
            beam_size=3,  # Reducido de 5 (default)
            best_of=3      # Reducido de 5
        )
        
        text = result['text'].strip()
        print(f"User: {text}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""


# --- OPTIMIZED recordAudio con detección de energía más agresiva ---
def recordAudio(silence_duration=1.5, volume_threshold=12, max_duration=20):
    """Grabación optimizada con detección adaptativa"""
    from collections import deque
    
    fs = 44100
    chunk_size = 1024
    started_recording = False
    silence_start = None
    buffer = deque(maxlen=int(0.8 * fs))  # Buffer reducido a 0.8s
    full_recording = []
    start_time = time.time()
    
    # OPTIMIZACIÓN: Detección adaptativa de energía
    energy_history = deque(maxlen=30)
    adaptive_threshold = volume_threshold
    
    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype=np.float32, blocksize=chunk_size) as stream:
            while True:
                if time.time() - start_time > max_duration:
                    print("⏱️  Max duration reached")
                    break
                
                audio_data, _ = stream.read(chunk_size)
                volume = np.sqrt(np.mean(np.square(audio_data))) * 1000
                
                # Actualizar umbral adaptativo
                energy_history.append(volume)
                if len(energy_history) == 30:
                    avg_noise = np.mean(list(energy_history)[:15])
                    adaptive_threshold = max(volume_threshold, avg_noise * 1.5)
                
                buffer.append(audio_data)
                
                if not started_recording:
                    print("*Listening...*", end='\r', flush=True)
                    if volume > adaptive_threshold:
                        print("*Recording...*", end='\r', flush=True)
                        full_recording = list(buffer)
                        started_recording = True
                        silence_start = None
                else:
                    full_recording.append(audio_data)
                    if volume < adaptive_threshold * 0.7:  # Umbral más bajo para silencio
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > silence_duration:
                            print("*Processing...*")
                            break
                    else:
                        silence_start = None
        
        if full_recording:
            full_recording = np.concatenate(full_recording)
            max_val = np.max(np.abs(full_recording))
            if max_val > 0:
                full_recording = np.int16(full_recording / max_val * 32767)
            else:
                full_recording = np.int16(full_recording * 32767)
        else:
            full_recording = np.array([], dtype=np.int16)
        
        return full_recording
    except Exception as e:
        print(f"\n❌ Recording error: {e}")
        return np.array([], dtype=np.int16)


# --- OPTIMIZED get_tts_model con mejor manejo de errores ---
@lru_cache(maxsize=1)
def get_tts_model():
    """TTS model con safe globals pre-configurados"""
    try:
        from TTS.api import TTS
    except Exception as e:
        raise RuntimeError(f"TTS library not available: {e}")

    # Safe globals expandidos
    candidates = [
        "TTS.config.shared_configs.BaseDatasetConfig",
        "TTS.tts.configs.xtts_config.XttsConfig",
        "TTS.tts.configs.xtts_config.XttsAudioConfig",
        "TTS.tts.models.xtts.XttsArgs",
        "TTS.tts.models.xtts.XttsAudioConfig",
        "TTS.tts.layers.xtts.tokenizer.VoiceBpeTokenizer",
        "TTS.tts.models.xtts.XttsEncoder",
        "TTS.tts.models.xtts.XttsDecoder",
        "TTS.utils.audio.AudioProcessor",
        "TTS.vocoder.models.base_vocoder.BaseVocoder",
    ]

    safe_globals = []
    for path in candidates:
        mod_path, _, cls_name = path.rpartition('.')
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            safe_globals.append(cls)
        except Exception:
            pass

    if safe_globals:
        try:
            torch.serialization.add_safe_globals(safe_globals)
            print(f"🔐 Registered {len(safe_globals)} safe globals")
        except Exception as e:
            print(f"⚠️  Could not add safe globals: {e}")

    print("⏳ Loading TTS model...")
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device=="cuda"))
    except Exception as e:
        print("❌ Error loading TTS model:", str(e))
        if "WeightsUnpickler" in str(e):
            print("🔎 Add missing classes to 'candidates' list")
        raise
    
    print("✅ TTS loaded")
    return tts


# --- OPTIMIZED generateAudio con streaming ---
def generateAudio(text, speaker_file, language="es", sample_rate=24000):
    """TTS optimizado con validación previa"""
    if not text or not text.strip():
        print("⚠️  Empty text, skipping TTS")
        return False
    
    # OPTIMIZACIÓN: Limitar longitud del texto
    if len(text) > 500:
        text = text[:500] + "..."
        print("⚠️  Text truncated to 500 chars")
    
    try:
        tts = get_tts_model()
        wav = tts.tts(text=text, speaker_wav=speaker_file, language=language)
        
        if wav is None or len(wav) == 0:
            print("❌ TTS returned empty audio")
            return False
        
        # OPTIMIZACIÓN: Reproducción sin bloqueo innecesario
        sd.play(wav, sample_rate)
        sd.wait()
        return True
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False