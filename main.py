# main.py
# Author: Claude Petit-Frere
# Date: 11/14/24
# Desc: Entry point for the assistant

import asyncio
import os
import sys
from audio_listener import KeywordListener
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


def main():
    """Entry point for the voice assistant"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wake_word_file = os.path.join(base_dir, "keywords", "jarvis_es_windows_v3_0_0.ppn")
    model_file = os.path.join(base_dir, "keywords", "porcupine_params_es.pv")
    speaker_file = os.path.join(base_dir, "speaker.wav")
    
    # Verificar archivos necesarios
    if not os.path.exists(wake_word_file):
        print(f"❌ Error: Wake word file not found at {wake_word_file}")
        sys.exit(1)
    
    if not os.path.exists(speaker_file):
        print(f"❌ Error: Speaker file not found at {speaker_file}")
        sys.exit(1)
    
    # El modelo de idioma es opcional
    if not os.path.exists(model_file):
        print(f"⚠️  Spanish model not found at {model_file}")
        print(f"   Using default English model")
        model_file = None
    
    # Verificar API key
    if not os.getenv("PICOVOICE_API_KEY"):
        print("⚠️  PICOVOICE_API_KEY not found in .env")
        print("   Wake word detection will use fallback mode\n")
    
    print("=" * 50)
    print("🎙️  Jarvis Voice Assistant")
    print("=" * 50)
    print(f"📂 Wake word: {os.path.basename(wake_word_file)}")
    print(f"🔊 Speaker: {os.path.basename(speaker_file)}")
    if model_file:
        print(f"🌐 Language model: {os.path.basename(model_file)}")
    print(f"🔗 Rasa URL: {os.getenv('RASA_URL', 'http://localhost:5005')}")
    print("=" * 50)
    print()
    
    try:
        listener = KeywordListener(
            wake_word_file=wake_word_file,
            speaker_file=speaker_file,
            model_file=model_file
        )
        asyncio.run(listener.start_listening())
    except KeyboardInterrupt:
        print("\n\n👋 Jarvis stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()