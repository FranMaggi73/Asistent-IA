# wake_word_detector.py
# Author: Claude Petit-Frere
# Date: 11/14/24
# Desc: Wake word detection using Porcupine keyword spotting

import os
import numpy as np
import sounddevice as sd
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Intentar importar porcupine
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    print("⚠️ pvporcupine not installed. Using fallback detection.")
    print("   Install with: pip install pvporcupine")


class WakeWordDetector:
    def __init__(self, keyword_path: str, model_path: Optional[str] = None, access_key: Optional[str] = None):
        """
        Inicializa el detector de wake word
        
        Args:
            keyword_path: Ruta al archivo .ppn
            model_path: Ruta al modelo de idioma (porcupine_params_es.pv)
            access_key: API key de Picovoice
        """
        self.keyword_path = keyword_path
        self.model_path = model_path
        
        # Usar API key del .env o la proporcionada
        self.access_key = access_key or os.getenv("PICOVOICE_API_KEY")
        
        self.porcupine = None
        
        if not PORCUPINE_AVAILABLE:
            print("⚠️ Porcupine not available. Using fallback detection.")
            return
        
        if not self.access_key:
            print("⚠️ PICOVOICE_API_KEY not found in .env file")
            print("   Using fallback detection (will accept all audio as wake word)")
            return
        
        # Inicializar Porcupine
        try:
            # Preparar argumentos
            porcupine_args = {
                "access_key": self.access_key,
                "keyword_paths": [keyword_path]
            }
            
            # Agregar modelo de idioma si existe
            if model_path and os.path.exists(model_path):
                porcupine_args["model_path"] = model_path
                print(f"✅ Using Spanish model: {os.path.basename(model_path)}")
            
            self.porcupine = pvporcupine.create(**porcupine_args)
            print(f"✅ Porcupine initialized with keyword: {os.path.basename(keyword_path)}")
            print(f"   Sample rate: {self.porcupine.sample_rate} Hz")
            print(f"   Frame length: {self.porcupine.frame_length}")
            
        except Exception as e:
            print(f"❌ Could not initialize Porcupine: {e}")
            print("   Using fallback detection method")
            self.porcupine = None
    
    def listen_for_wake_word(self) -> bool:
        """
        Escucha continuamente por el wake word usando streaming
        
        Returns:
            True cuando se detecta el wake word
        """
        if self.porcupine is None:
            # Fallback: usar el método de recordAudio original
            print("🎤 Press Enter or speak to continue (fallback mode)...", end='', flush=True)
            from audioFunctions import recordAudio
            recordAudio()  # Graba hasta silencio
            print()
            return True
        
        try:
            sample_rate = self.porcupine.sample_rate
            frame_length = self.porcupine.frame_length
            
            print("🎤 Listening for 'Jarvis'...", end='\r', flush=True)
            
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                blocksize=frame_length
            ) as stream:
                
                while True:
                    audio_frame, _ = stream.read(frame_length)
                    audio_frame = audio_frame.flatten()
                    
                    keyword_index = self.porcupine.process(audio_frame)
                    
                    if keyword_index >= 0:
                        print("🎯 'Jarvis' detected!        ")
                        return True
                        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n❌ Error in wake word detection: {e}")
            print("   Falling back to manual recording")
            from audioFunctions import recordAudio
            recordAudio()
            return True
    
    def __del__(self):
        """Cleanup"""
        if self.porcupine is not None:
            try:
                self.porcupine.delete()
            except:
                pass


# Test del detector (opcional)
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wake_word_detector.py <path_to_keyword.ppn>")
        sys.exit(1)
    
    keyword_path = sys.argv[1]
    detector = WakeWordDetector(keyword_path)
    
    print("Testing wake word detector...")
    print("Say 'Jarvis' to test (Ctrl+C to exit)")
    
    try:
        while True:
            detected = detector.listen_for_wake_word()
            if detected:
                print("✅ Wake word detected successfully!\n")
    except KeyboardInterrupt:
        print("\n👋 Test stopped")