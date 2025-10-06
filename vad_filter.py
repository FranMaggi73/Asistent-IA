"""
Voice Activity Detection (VAD) ultra-rápido
Evita transcribir audio sin voz (ahorra ~2s por false positive)
"""

import numpy as np
from scipy import signal
from functools import lru_cache


class FastVAD:
    """
    VAD basado en energía y zero-crossing rate
    Detecta si hay voz ANTES de enviar a Whisper
    """
    def __init__(self, 
                 energy_threshold=0.02, 
                 zcr_threshold=0.1,
                 min_speech_duration=0.3):
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.min_speech_duration = min_speech_duration
    
    def contains_speech(self, audio_data: np.ndarray, 
                       sample_rate: int = 44100) -> bool:
        """
        Detecta si audio contiene voz (rápido, <10ms)
        
        Args:
            audio_data: Audio como numpy array
            sample_rate: Frecuencia de muestreo
            
        Returns:
            True si probablemente contiene voz
        """
        if len(audio_data) == 0:
            return False
        
        # Normalizar
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # 1. ENERGÍA (indicador primario)
        energy = np.sqrt(np.mean(np.square(audio_float)))
        
        if energy < self.energy_threshold:
            print(f"⚠️  VAD: Low energy ({energy:.4f}) - likely silence")
            return False
        
        # 2. ZERO-CROSSING RATE (diferencia ruido de voz)
        zcr = self._calculate_zcr(audio_float)
        
        if zcr < self.zcr_threshold:
            print(f"⚠️  VAD: Low ZCR ({zcr:.4f}) - likely noise")
            return False
        
        # 3. DURACIÓN (evitar clicks)
        duration = len(audio_data) / sample_rate
        if duration < self.min_speech_duration:
            print(f"⚠️  VAD: Too short ({duration:.2f}s)")
            return False
        
        print(f"✅ VAD: Speech detected (E={energy:.4f}, ZCR={zcr:.4f})")
        return True
    
    def _calculate_zcr(self, audio: np.ndarray) -> float:
        """Calcula Zero-Crossing Rate (rápido)"""
        # Contar cambios de signo
        signs = np.sign(audio)
        sign_changes = np.diff(signs) != 0
        zcr = np.sum(sign_changes) / len(audio)
        return zcr
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento opcional: remover DC offset, normalizar
        """
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Remover DC offset
        audio_float = audio_float - np.mean(audio_float)
        
        # Filtro pasa-altos (remover ruido bajo)
        sos = signal.butter(4, 80, 'hp', fs=44100, output='sos')
        audio_filtered = signal.sosfilt(sos, audio_float)
        
        return audio_filtered


# Singleton para evitar recrear filtros
@lru_cache(maxsize=1)
def get_vad_detector():
    return FastVAD()


# Integración con audioFunctions.py
def whisperTranscription_with_vad(audio_data, language='es'):
    """
    Drop-in replacement para whisperTranscription con VAD pre-filtrado
    """
    # Pre-filtro VAD (muy rápido, <10ms)
    vad = get_vad_detector()
    
    if not vad.contains_speech(audio_data):
        print("⚡ VAD: No speech detected, skipping transcription")
        return ""
    
    # Si pasa VAD, transcribir normalmente
    from audioFunctions import whisperTranscription
    return whisperTranscription(audio_data, language)


# Test standalone
if __name__ == "__main__":
    import sounddevice as sd
    from audioFunctions import recordAudio
    
    print("🎤 Testing VAD...")
    print("Say something after the beep...\n")
    
    # Grabar audio
    audio_data = recordAudio()
    
    # Test VAD
    vad = FastVAD()
    has_speech = vad.contains_speech(audio_data)
    
    print(f"\n{'✅' if has_speech else '❌'} Speech detected: {has_speech}")
    
    if has_speech:
        print("Transcribing...")
        from audioFunctions import whisperTranscription
        text = whisperTranscription(audio_data)
        print(f"Result: {text}")
    else:
        print("Skipping transcription (no speech)")