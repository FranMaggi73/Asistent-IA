# audioFunctions.py - Audio processing con CUDA optimizado
import importlib
from functools import lru_cache
from typing import List, Any
import numpy as np
from numpy.typing import NDArray
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    print(f"🚀 CUDA activo: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_flush_denormal(True)
    print("⚠️  CUDA no disponible, usando CPU")


# ─────────────────────────────────────────────
# Model Manager (faster-whisper)
# ─────────────────────────────────────────────

class ModelManager:
    _instance = None
    _whisper_model: Any = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def whisper(self) -> Any:
        if self._whisper_model is None:
            with self._lock:
                if self._whisper_model is None:
                    self._load_whisper()
        return self._whisper_model

    def _load_whisper(self) -> None:
        from faster_whisper import WhisperModel

        print("⏳ Cargando Whisper...")
        t = time.time()

        if device == "cuda":
            self._whisper_model = WhisperModel(
                "medium",
                device="cuda",
                compute_type="int8_float16",
            )
        else:
            self._whisper_model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8",
            )

        elapsed = time.time() - t
        model_name = "medium (int8_float16)" if device == "cuda" else "small (int8)"
        print(f"✅ Whisper cargado ({model_name} en {device.upper()}) — {elapsed:.1f}s")

    def unload_whisper(self) -> None:
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
# Whisper Transcription con faster-whisper
# ─────────────────────────────────────────────

def whisperTranscription(audio_data: NDArray[np.int16], language: str = 'es') -> str:
    try:
        if len(audio_data) == 0:
            return ""

        # Normalizar a float32 en rango [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Aplanar a 1D — sounddevice puede devolver (N, 1)
        if audio_float.ndim > 1:
            audio_float = audio_float.mean(axis=1)

        # Saltar audio demasiado corto (< 0.3s a 44100Hz)
        if len(audio_float) < 13230:
            return ""

        # Resamplear de 44100 a 16000Hz con interpolacion lineal
        target_len = int(len(audio_float) * 16000 / 44100)
        indices = np.linspace(0, len(audio_float) - 1, target_len)
        audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float).astype(np.float32)

        model = model_manager.whisper

        segments, _ = model.transcribe(
            audio_float,
            language=language,
            beam_size=1,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 200,
            },
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )

        text = "".join(s.text for s in segments).strip()

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
                max_duration: float = 20) -> NDArray[np.int16]:
    """Graba audio con deteccion de silencio adaptativa"""
    from collections import deque

    fs = 44100
    chunk_size = 1024
    started_recording = False
    silence_start = None
    buffer: deque = deque(maxlen=int(0.3 * fs / chunk_size))
    full_recording: List[NDArray] = []
    start_time = time.time()
    energy_history: deque = deque(maxlen=30)
    adaptive_threshold = volume_threshold

    try:
        with sd.InputStream(samplerate=fs, channels=1,
                            dtype=np.float32, blocksize=chunk_size) as stream:
            while True:
                if time.time() - start_time > max_duration:
                    print("⏱️  Duracion maxima alcanzada")
                    break

                audio_chunk, _ = stream.read(chunk_size)
                volume = np.sqrt(np.mean(np.square(audio_chunk))) * 1000

                energy_history.append(volume)
                if len(energy_history) == 30:
                    avg_noise = np.mean(list(energy_history)[:15])
                    adaptive_threshold = max(volume_threshold, avg_noise * 1.5)

                buffer.append(audio_chunk)

                if not started_recording:
                    print("*Escuchando...*", end='\r', flush=True)
                    if volume > adaptive_threshold:
                        print("*Grabando...*  ", end='\r', flush=True)
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
            # Aplanar a 1D por si sounddevice devuelve (N, 1)
            if combined.ndim > 1:
                combined = combined.mean(axis=1)
            max_val = np.max(np.abs(combined))
            if max_val > 0:
                return (combined / max_val * 32767).astype(np.int16)
            return (combined * 32767).astype(np.int16)

        return np.array([], dtype=np.int16)

    except Exception as e:
        print(f"\n❌ Recording error: {e}")
        return np.array([], dtype=np.int16)


# ─────────────────────────────────────────────
# TTS (Coqui XTTS v2) con CUDA
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_tts_model() -> Any:
    """Carga el modelo TTS con CUDA si esta disponible"""
    try:
        from TTS.api import TTS
    except ImportError:
        raise RuntimeError("coqui-tts no instalado: pip install coqui-tts")

    candidates = [
        "TTS.config.shared_configs.BaseDatasetConfig",
        "TTS.tts.configs.xtts_config.XttsConfig",
        "TTS.tts.configs.xtts_config.XttsAudioConfig",
        "TTS.tts.models.xtts.XttsArgs",
        "TTS.tts.models.xtts.XttsAudioConfig",
        "TTS.tts.layers.xtts.tokenizer.VoiceBpeTokenizer",
    ]

    safe_globals: List[Any] = []
    for path in candidates:
        mod_path, _, cls_name = path.rpartition('.')
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            safe_globals.append(cls)
        except Exception:
            pass

    if safe_globals and hasattr(torch.serialization, 'add_safe_globals'):
        try:
            torch.serialization.add_safe_globals(safe_globals)  # type: ignore[attr-defined]
        except Exception:
            pass

    print(f"⏳ Cargando TTS en {device.upper()}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
    print(f"✅ TTS cargado en {device.upper()}")
    return tts


def generateAudio(text: str, speaker_file: str,
                  language: str = "es", sample_rate: int = 24000) -> bool:
    """Genera y reproduce audio TTS"""
    if not text or not text.strip():
        return False

    if len(text) > 400:
        text = text[:397] + "..."

    try:
        tts = get_tts_model()
        wav = tts.tts(text=text, speaker_wav=speaker_file, language=language)

        if wav is None or len(wav) == 0:
            print("❌ TTS devolvio audio vacio")
            return False

        sd.play(wav, sample_rate)
        sd.wait()
        return True

    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False


# ─────────────────────────────────────────────
# VRAM Monitor
# ─────────────────────────────────────────────

def print_cuda_stats() -> None:
    if device != "cuda":
        return
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved  = torch.cuda.memory_reserved(0) / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🖥️  VRAM: {allocated:.2f}GB usado / {reserved:.2f}GB reservado / {total:.1f}GB total")