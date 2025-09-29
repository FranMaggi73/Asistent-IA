# whisper_helper.py
import whisper
import numpy as np

# Carga el modelo
model = whisper.load_model("base")

def transcribe_audio(audio: np.ndarray, samplerate: int) -> str:
    """
    Transcribe audio usando Whisper directamente desde el array.
    """
    # Asegurarse de que sea float32 y de aplanar si es mono
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0

    # Whisper espera 16kHz, si no es, habría que resamplear
    if samplerate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)
        samplerate = 16000

    result = model.transcribe(audio, fp16=False, language="es")
    return result.get("text", "")
