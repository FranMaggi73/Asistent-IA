# audioFunctions.py
# Author: Claude Petit-Frere
# Date: 11/14/24
# Desc: Audio recording and transcription functions

import librosa
import torch
import warnings
import whisper
import numpy as np
import sounddevice as sd
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cpu":
    print("⚠️ Running on CPU, may be slower than using GPU.\n")

# Cargar Whisper
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
    model = whisper.load_model("small").to(device)
    print("✅ Whisper model loaded\n")


def whisperTranscription(audio_data):
    """
    Transcribe audio using Whisper
    
    Args:
        audio_data: Audio array in int16 format
        
    Returns:
        Transcribed text
    """
    try:
        # Convertir a float32 y normalizar
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Resamplear a 16kHz (requerido por Whisper)
        resampled_audio = librosa.resample(
            audio_data.astype(np.float32), 
            orig_sr=44100, 
            target_sr=16000
        )
        
        # Convertir a mono si es estéreo
        if len(resampled_audio.shape) > 1:
            resampled_audio = resampled_audio.mean(axis=1)
        
        # Transcribir
        result = model.transcribe(
            resampled_audio, 
            language='es', 
            fp16=torch.cuda.is_available()
        )
        
        transcribed_text = result['text'].strip()
        print(f"User: {transcribed_text}")
        
        return transcribed_text
        
    except Exception as e:
        print(f"❌ Error in transcription: {e}")
        return ""


def recordAudio(silence_duration=2.25, volume_threshold=10):
    """
    Record audio until silence is detected
    
    Args:
        silence_duration: Seconds of silence to stop recording (default: 2.25)
        volume_threshold: Volume threshold to detect speech (default: 10)
        
    Returns:
        Recorded audio as int16 numpy array
    """
    fs = 44100  # Sample rate
    chunk_size = 1024
    started_recording = False
    silence_start = None
    buffer = np.array([], dtype=np.float32)
    buffer_length = 1  # Buffer de 1 segundo
    full_recording = np.array([], dtype=np.float32)
    
    try:
        with sd.InputStream(
            samplerate=fs, 
            channels=1, 
            dtype=np.float32, 
            blocksize=chunk_size
        ) as stream:
            
            while True:
                audio_data, _ = stream.read(chunk_size)
                
                # Calcular volumen RMS
                volume = np.sqrt(np.mean(np.square(audio_data))) * 1000
                
                # Mantener buffer circular
                buffer = np.append(buffer, audio_data)
                if len(buffer) > int(buffer_length * fs):
                    buffer = buffer[-int(buffer_length * fs):]
                
                # Estado: esperando inicio de habla
                if not started_recording:
                    print("*Listening...*", end='\r', flush=True)
                    
                    if volume > volume_threshold:
                        print("*Recording...*", end='\r', flush=True)
                        # Incluir el buffer previo (para no perder inicio)
                        full_recording = np.append(full_recording, buffer)
                        started_recording = True
                        silence_start = None
                
                # Estado: grabando
                elif started_recording:
                    full_recording = np.append(full_recording, audio_data)
                    
                    # Detectar silencio
                    if volume < volume_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > silence_duration:
                            print("*Processing...*")
                            break
                    else:
                        # Resetear contador de silencio si hay audio
                        silence_start = None
        
        # Normalizar y convertir a int16
        if len(full_recording) > 0:
            max_val = np.max(np.abs(full_recording))
            if max_val > 0:
                full_recording = np.int16(full_recording / max_val * 32767)
            else:
                full_recording = np.int16(full_recording * 32767)
        else:
            full_recording = np.array([], dtype=np.int16)
        
        return full_recording
        
    except Exception as e:
        print(f"\n❌ Error recording audio: {e}")
        return np.array([], dtype=np.int16)


def generateAudio(text, speaker_file, tts, language="es", sample_rate=24000):
    """
    Generate and play audio from text using TTS
    
    Args:
        text: Text to convert to speech
        speaker_file: Path to speaker reference audio
        tts: TTS model instance
        language: Language code (default: "es")
        sample_rate: Playback sample rate (default: 24000)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generar audio
        wav = tts.tts(
            text=text, 
            speaker_wav=speaker_file, 
            language=language
        )
        
        if wav is None or len(wav) == 0:
            print("❌ TTS returned empty audio")
            return False
        
        # Reproducir
        sd.play(wav, sample_rate)
        sd.wait()
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        return False


# Función de prueba (opcional)
def test_recording():
    """Test function to record and transcribe audio"""
    print("🎙️  Testing audio recording...")
    print("Speak now (will stop after 2.25s of silence):\n")
    
    audio = recordAudio()
    
    if len(audio) > 0:
        print(f"✅ Recorded {len(audio)} samples")
        print("Transcribing...")
        text = whisperTranscription(audio)
        print(f"Result: '{text}'")
    else:
        print("❌ No audio recorded")


if __name__ == "__main__":
    # Ejecutar test si se corre directamente
    test_recording()