import warnings
import importlib
from functools import lru_cache
import numpy as np
import time
import torch
import sounddevice as sd

# Attempt to initialize COM on Windows early (helps VLC/mmdevice errors)
try:
    import pythoncom
    try:
        pythoncom.CoInitialize()
        print("✅ COM initialized (pythoncom.CoInitialize)")
    except Exception as e:
        print(f"⚠️  COM CoInitialize failed: {e}")
except Exception:
    # pywin32 not installed or not running on Windows; ignore
    pass

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Simple singleton model manager for whisper (unchanged) ---
class ModelManager:
    _instance = None
    _whisper_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def whisper(self):
        if self._whisper_model is None:
            print("⏳ Loading Whisper model...")
            import whisper
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self._whisper_model = whisper.load_model("small").to(device)
            print("✅ Whisper loaded")
        return self._whisper_model

    def unload_whisper(self):
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            if device == "cuda":
                torch.cuda.empty_cache()
            print("🗑️  Whisper unloaded")

model_manager = ModelManager()

# --- Whisper transcription and recording (kept minimal / efficient) ---

def whisperTranscription(audio_data, language='es'):
    try:
        audio_float = audio_data.astype(np.float32) / 32768.0
        from scipy import signal
        orig_sr = 44100
        target_sr = 16000
        num_samples = int(len(audio_float) * target_sr / orig_sr)
        resampled = signal.resample(audio_float, num_samples)
        if resampled.ndim > 1:
            resampled = resampled.mean(axis=1)
        model = model_manager.whisper
        result = model.transcribe(
            resampled,
            language=language,
            fp16=(device == "cuda"),
            verbose=False,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6
        )
        text = result['text'].strip()
        print(f"User: {text}")
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""


def recordAudio(silence_duration=1.8, volume_threshold=12, max_duration=30):
    from collections import deque
    fs = 44100
    chunk_size = 1024
    started_recording = False
    silence_start = None
    buffer = deque(maxlen=int(1 * fs))
    full_recording = []
    start_time = time.time()
    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype=np.float32, blocksize=chunk_size) as stream:
            while True:
                if time.time() - start_time > max_duration:
                    print("⏱️  Max duration reached")
                    break
                audio_data, _ = stream.read(chunk_size)
                volume = np.sqrt(np.mean(np.square(audio_data))) * 1000
                buffer.append(audio_data)
                if not started_recording:
                    print("*Listening...*", end='\r', flush=True)
                    if volume > volume_threshold:
                        print("*Recording...*", end='\r', flush=True)
                        full_recording = list(buffer)
                        started_recording = True
                        silence_start = None
                else:
                    full_recording.append(audio_data)
                    if volume < volume_threshold:
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


# --- Robust get_tts_model that add_safe_globals for common classes ---
@lru_cache(maxsize=1)
def get_tts_model():
    """Return a cached Coqui TTS model. Registers many known classes to torch safe globals
    so loading xtts_v2 and other Coqui checkpoints works with PyTorch >=2.6.
    """
    try:
        from TTS.api import TTS
    except Exception as e:
        raise RuntimeError(f"TTS library not available: {e}")

    # Candidate class paths commonly required by Coqui checkpoints.
    candidates = [
        # core configs / classes
        "TTS.config.shared_configs.BaseDatasetConfig",
        "TTS.tts.configs.xtts_config.XttsConfig",
        "TTS.tts.configs.xtts_config.XttsAudioConfig",
        "TTS.tts.models.xtts.XttsArgs",
        "TTS.tts.models.xtts.XttsAudioConfig",
        "TTS.tts.layers.xtts.tokenizer.VoiceBpeTokenizer",
        "TTS.tts.models.xtts.XttsEncoder",
        "TTS.tts.models.xtts.XttsDecoder",
        # some common helpers — extend if error mentions others
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
        except Exception as e:
            # Not all classes exist in every TTS version; ignore missing ones
            # Print debug message so user can add classes that error shows
            print(f"⚠️  Candidate import failed: {path}: {e}")

    if safe_globals:
        try:
            torch.serialization.add_safe_globals(safe_globals)
            print(f"🔐 Registered {len(safe_globals)} safe globals for torch serialization.")
        except Exception as e:
            print(f"⚠️  Could not add safe globals: {e}")

    print("⏳ Loading TTS model...")
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device=="cuda"))
    except Exception as e:
        # If there is a WeightsUnpickler error, show guidance
        err = str(e)
        print("❌ Error loading TTS model:", err)
        if "WeightsUnpickler" in err or "Unsupported global" in err:
            print("🔎 The error indicates PyTorch weights-only loading blocked some classes.")
            print("→ Add the exact class name reported by the error to 'candidates' above and retry.")
            print("→ Or as a last resort (ONLY if you trust the checkpoint), load with weights_only=False -- see warnings in code.")
        raise
    print("✅ TTS loaded")
    return tts


def generateAudio(text, speaker_file, language="es", sample_rate=24000):
    if not text or not text.strip():
        print("⚠️  Empty text, skipping TTS")
        return False
    try:
        tts = get_tts_model()
        wav = tts.tts(text=text, speaker_wav=speaker_file, language=language)
        if wav is None or len(wav) == 0:
            print("❌ TTS returned empty audio")
            return False
        sd.play(wav, sample_rate)
        sd.wait()
        return True
    except Exception as e:
        print(f"❌ TTS error: {e}")
        # Helpful debug notes printed to user
        return False
