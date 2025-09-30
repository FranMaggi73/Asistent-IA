# wake_word_detector.py
# Author: Claude Petit-Frere
# Date: 11/14/24
# Desc: Wake word detection using Porcupine keyword spotting

import pvporcupine
import numpy as np
import sounddevice as sd
import os
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

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
        """
        if self.porcupine is None:
            # Fallback: usar el método de recordAudio original
            print("⚠️ Using fallback detection (press any key to continue)")
            from audioFunctions import recordAudio
            recordAudio()  # Graba hasta silencio
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