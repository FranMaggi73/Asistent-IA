# whisper_helper.py
import whisper
import numpy as np

# Carga el modelo de Whisper (puede ser 'tiny', 'base', 'small', 'medium', 'large')
model = whisper.load_model("base")

def transcribe_audio(audio: np.ndarray, samplerate: int) -> str:
    """
    Transcribe audio usando Whisper directamente desde un array numpy.
    
    Args:
        audio: np.ndarray con la señal de audio (int16 o float32)
        samplerate: frecuencia de muestreo del audio

    Returns:
        Texto transcrito.
    """
    # Asegurarse de que sea float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0  # Convierte int16 a float32

    # Whisper espera audio a 16000 Hz, si no está en 16kHz lo re-muestreamos
    if samplerate != 16000:
        import librosa
        audio = librosa.resample(audio.flatten(), orig_sr=samplerate, target_sr=16000)
        samplerate = 16000

    # Transcribe el audio directamente desde numpy
    result = model.transcribe(audio, fp16=False, language="es")  # fp16=False si no usas GPU

    # Asegurarse de que siempre devuelva string
    text = result.get("text", "")
    if isinstance(text, list):
        text = " ".join(text)

    return text
