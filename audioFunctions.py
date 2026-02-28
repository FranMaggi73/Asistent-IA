# audioFunctions.py - Audio processing con CUDA optimizado
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
    except Exception:
        pass
except Exception:
    pass

# ─────────────────────────────────────────────
# CUDA Setup
# ─────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    # Optimizaciones CUDA para RTX 2060
    torch.backends.cudnn.benchmark = True       # Autotuning de kernels
    torch.backends.cuda.matmul.allow_tf32 = True # TF32 para matmul (más rápido, ~misma precisión)
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')   # Permite TF32
    print(f"🚀 CUDA activo: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    # Optimizaciones CPU
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_flush_denormal(True)
    print("⚠️  CUDA no disponible, usando CPU")


# ─────────────────────────────────────────────
# Model Manager (Whisper)
# ─────────────────────────────────────────────

class ModelManager:
    _instance = None
    _whisper_model = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def whisper(self):
        if self._whisper_model is None:
            with self._lock:
                if self._whisper_model is None:
                    print("⏳ Cargando Whisper...")
                    import whisper
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        # RTX 2060 (6GB): large-v2 en CUDA, small en CPU
                        if device == "cuda":
                            model_size = "large-v2"  # ~3GB VRAM
                        else:
                            model_size = "small"
                        self._whisper_model = whisper.load_model(
                            model_size,
                            device=device
                        )
                    print(f"✅ Whisper cargado ({model_size} en {device.upper()})")
        return self._whisper_model

    def unload_whisper(self):
        if self._whisper_model is not None:
            with self._lock:
                if self._whisper_model is not None:
                    del self._whisper_model
                    self._whisper_model = None
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    print("🗑️  Whisper descargado")


model_manager = ModelManager()


# ─────────────────────────────────────────────
# Whisper Transcription con CUDA
# ─────────────────────────────────────────────

def whisperTranscription(audio_data: np.ndarray, language: str = 'es') -> str:
    """Transcripción con Whisper, optimizada para CUDA"""
    try:
        if len(audio_data) == 0:
            return ""

        # Normalizar a float32
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Saltar audio demasiado corto (< 0.5s)
        if len(audio_float) < 22050:
            return ""

        # Resample de 44100 → 16000 Hz (requerido por Whisper)
        from scipy import signal as scipy_signal
        orig_sr, target_sr = 44100, 16000
        n_samples = int(len(audio_float) * target_sr / orig_sr)
        resampled = scipy_signal.resample(audio_float, n_samples)

        if resampled.ndim > 1:
            resampled = resampled.mean(axis=1)

        model = model_manager.whisper

        result = model.transcribe(
            resampled,
            language=language,
            fp16=(device == "cuda"),       # FP16 en GPU = 2x más rápido
            verbose=False,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            beam_size=3,                   # Reducido para más velocidad
            best_of=3,
            task="transcribe"
        )

        text = result['text'].strip()
        if text:
            print(f"📝 User: {text}")
        return text

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""


# ─────────────────────────────────────────────
# Audio Recording
# ─────────────────────────────────────────────

def recordAudio(silence_duration: float = 1.5,
                volume_threshold: float = 12,
                max_duration: float = 20) -> np.ndarray:
    """Graba audio con detección de silencio adaptativa"""
    from collections import deque

    fs = 44100
    chunk_size = 1024
    started_recording = False
    silence_start = None
    buffer = deque(maxlen=int(0.8 * fs))
    full_recording = []
    start_time = time.time()
    energy_history = deque(maxlen=30)
    adaptive_threshold = volume_threshold

    try:
        with sd.InputStream(samplerate=fs, channels=1,
                            dtype=np.float32, blocksize=chunk_size) as stream:
            while True:
                if time.time() - start_time > max_duration:
                    print("⏱️  Duración máxima alcanzada")
                    break

                audio_chunk, _ = stream.read(chunk_size)
                volume = np.sqrt(np.mean(np.square(audio_chunk))) * 1000

                # Umbral adaptativo
                energy_history.append(volume)
                if len(energy_history) == 30:
                    avg_noise = np.mean(list(energy_history)[:15])
                    adaptive_threshold = max(volume_threshold, avg_noise * 1.5)

                buffer.append(audio_chunk)

                if not started_recording:
                    print("*Escuchando...*", end='\r', flush=True)
                    if volume > adaptive_threshold:
                        print("*Grabando...*", end='\r', flush=True)
                        full_recording = list(buffer)
                        started_recording = True
                        silence_start = None
                else:
                    full_recording.append(audio_chunk)
                    if volume < adaptive_threshold * 0.7:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > silence_duration:
                            print("*Procesando...*")
                            break
                    else:
                        silence_start = None

        if full_recording:
            combined = np.concatenate(full_recording)
            max_val = np.max(np.abs(combined))
            if max_val > 0:
                return np.int16(combined / max_val * 32767)
            return np.int16(combined * 32767)

        return np.array([], dtype=np.int16)

    except Exception as e:
        print(f"\n❌ Recording error: {e}")
        return np.array([], dtype=np.int16)


# ─────────────────────────────────────────────
# TTS (Coqui-TTS) con CUDA
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_tts_model():
    """Carga el modelo TTS con CUDA si está disponible"""
    try:
        from TTS.api import TTS
    except ImportError:
        raise RuntimeError("coqui-tts no instalado: pip install coqui-tts")

    # Safe globals para torch.load
    candidates = [
        "TTS.config.shared_configs.BaseDatasetConfig",
        "TTS.tts.configs.xtts_config.XttsConfig",
        "TTS.tts.configs.xtts_config.XttsAudioConfig",
        "TTS.tts.models.xtts.XttsArgs",
        "TTS.tts.models.xtts.XttsAudioConfig",
        "TTS.tts.layers.xtts.tokenizer.VoiceBpeTokenizer",
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
        except Exception:
            pass

    print(f"⏳ Cargando TTS en {device.upper()}...")
    use_gpu = (device == "cuda")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    print(f"✅ TTS cargado en {device.upper()}")
    return tts


def generateAudio(text: str, speaker_file: str,
                  language: str = "es", sample_rate: int = 24000) -> bool:
    """Genera y reproduce audio TTS"""
    if not text or not text.strip():
        return False

    # Limitar longitud para evitar latencia excesiva
    if len(text) > 400:
        text = text[:397] + "..."

    try:
        tts = get_tts_model()

        # Generar audio (la inferencia corre en CUDA automáticamente)
        wav = tts.tts(text=text, speaker_wav=speaker_file, language=language)

        if wav is None or len(wav) == 0:
            print("❌ TTS devolvió audio vacío")
            return False

        sd.play(wav, sample_rate)
        sd.wait()
        return True

    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False


# ─────────────────────────────────────────────
# VRAM Monitor (útil para debug con RTX 2060)
# ─────────────────────────────────────────────

def print_cuda_stats():
    """Muestra uso de VRAM"""
    if device != "cuda":
        return
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved  = torch.cuda.memory_reserved(0) / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🖥️  VRAM: {allocated:.2f}GB usado / {reserved:.2f}GB reservado / {total:.1f}GB total")