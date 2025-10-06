#!/usr/bin/env python3
"""
Benchmark de rendimiento para Jarvis
Mide tiempos de respuesta de cada componente
"""

import time
import asyncio
import statistics
from typing import List, Dict
import sys
from pathlib import Path

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent))


class JarvisBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_imports(self) -> float:
        """Mide tiempo de importación de módulos"""
        print("\n📦 Benchmarking imports...")
        
        start = time.time()
        import torch
        import numpy as np
        import sounddevice as sd
        from rasa_sdk import Action
        elapsed = time.time() - start
        
        print(f"   ✓ Basic imports: {elapsed:.3f}s")
        return elapsed
    
    def benchmark_whisper_load(self) -> float:
        """Mide carga de Whisper"""
        print("\n🎤 Benchmarking Whisper load...")
        
        start = time.time()
        from audioFunctions import model_manager
        _ = model_manager.whisper
        elapsed = time.time() - start
        
        print(f"   ✓ Whisper load: {elapsed:.3f}s")
        return elapsed
    
    def benchmark_tts_load(self) -> float:
        """Mide carga de TTS"""
        print("\n🔊 Benchmarking TTS load...")
        
        try:
            start = time.time()
            from audioFunctions import get_tts_model
            _ = get_tts_model()
            elapsed = time.time() - start
            
            print(f"   ✓ TTS load: {elapsed:.3f}s")
            return elapsed
        except Exception as e:
            print(f"   ✗ TTS load failed: {e}")
            return -1
    
    def benchmark_rasa_client(self) -> Dict[str, float]:
        """Mide latencia de Rasa"""
        print("\n🤖 Benchmarking Rasa client...")
        
        from rasa_client import RasaClient
        client = RasaClient()
        
        if not client.is_available():
            print("   ✗ Rasa not available")
            return {'available': False}
        
        # Test de latencia con múltiples queries
        test_messages = [
            "hola",
            "abre chrome",
            "reproduce música",
            "pausa",
            "adiós"
        ]
        
        latencies = []
        for msg in test_messages:
            start = time.time()
            response = client.send_message(msg, sender_id="benchmark_user")
            elapsed = time.time() - start
            latencies.append(elapsed)
            print(f"   • '{msg}': {elapsed*1000:.0f}ms")
        
        client.close()
        
        return {
            'available': True,
            'min': min(latencies),
            'max': max(latencies),
            'avg': statistics.mean(latencies),
            'median': statistics.median(latencies)
        }
    
    def benchmark_desktop_cache(self) -> float:
        """Mide rendimiento de cache de escritorio"""
        print("\n📁 Benchmarking desktop cache...")
        
        from actions.actions import desktop_cache
        
        # Primer acceso (sin cache)
        start = time.time()
        items1, _ = desktop_cache.get_items(force_refresh=True)
        cold_time = time.time() - start
        
        # Segundo acceso (con cache)
        start = time.time()
        items2, _ = desktop_cache.get_items()
        hot_time = time.time() - start
        
        print(f"   ✓ Cold cache: {cold_time*1000:.1f}ms ({len(items1)} items)")
        print(f"   ✓ Hot cache: {hot_time*1000:.1f}ms")
        print(f"   🚀 Speedup: {cold_time/hot_time:.1f}x")
        
        return cold_time, hot_time
    
    def benchmark_transcription(self) -> float:
        """Mide velocidad de transcripción (mock)"""
        print("\n🎙️  Benchmarking transcription...")
        
        import numpy as np
        from audioFunctions import whisperTranscription
        
        # Audio mock (2 segundos de silencio)
        sample_rate = 44100
        duration = 2
        audio_data = np.zeros(sample_rate * duration, dtype=np.int16)
        
        start = time.time()
        result = whisperTranscription(audio_data)
        elapsed = time.time() - start
        
        print(f"   ✓ Transcription: {elapsed:.3f}s")
        return elapsed
    
    async def run_all_benchmarks(self):
        """Ejecuta todos los benchmarks"""
        print("=" * 70)
        print("🏁 JARVIS PERFORMANCE BENCHMARK")
        print("=" * 70)
        
        # 1. Imports
        self.results['imports'] = self.benchmark_imports()
        
        # 2. Whisper
        self.results['whisper_load'] = self.benchmark_whisper_load()
        
        # 3. TTS
        self.results['tts_load'] = self.benchmark_tts_load()
        
        # 4. Desktop cache
        cold, hot = self.benchmark_desktop_cache()
        self.results['desktop_cache_cold'] = cold
        self.results['desktop_cache_hot'] = hot
        
        # 5. Transcription
        self.results['transcription'] = self.benchmark_transcription()
        
        # 6. Rasa
        self.results['rasa'] = self.benchmark_rasa_client()
        
        # Resultados finales
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumen de resultados"""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 70)
        
        # Tiempos de carga
        print("\n🚀 Load Times:")
        print(f"   Imports:          {self.results['imports']:.3f}s")
        print(f"   Whisper:          {self.results['whisper_load']:.3f}s")
        if self.results['tts_load'] > 0:
            print(f"   TTS:              {self.results['tts_load']:.3f}s")
        print(f"   Total:            {sum([v for v in [self.results['imports'], self.results['whisper_load'], self.results['tts_load']] if v > 0]):.3f}s")
        
        # Cache
        print("\n💾 Cache Performance:")
        print(f"   Cold:             {self.results['desktop_cache_cold']*1000:.1f}ms")
        print(f"   Hot:              {self.results['desktop_cache_hot']*1000:.1f}ms")
        print(f"   Speedup:          {self.results['desktop_cache_cold']/self.results['desktop_cache_hot']:.1f}x")
        
        # Transcripción
        print("\n🎤 Audio Processing:")
        print(f"   Transcription:    {self.results['transcription']:.3f}s")
        
        # Rasa
        if self.results['rasa']['available']:
            print("\n🤖 Rasa Latency:")
            print(f"   Min:              {self.results['rasa']['min']*1000:.0f}ms")
            print(f"   Avg:              {self.results['rasa']['avg']*1000:.0f}ms")
            print(f"   Max:              {self.results['rasa']['max']*1000:.0f}ms")
            print(f"   Median:           {self.results['rasa']['median']*1000:.0f}ms")
        else:
            print("\n🤖 Rasa: NOT AVAILABLE")
        
        # Estimación de respuesta total
        total_response = (
            self.results['transcription'] + 
            (self.results['rasa']['avg'] if self.results['rasa']['available'] else 0.5) +
            0.5  # TTS estimado
        )
        
        print("\n⚡ Estimated Total Response Time:")
        print(f"   {total_response:.2f}s (transcription + rasa + tts)")
        
        # Grade
        if total_response < 2.0:
            grade = "🏆 EXCELLENT"
        elif total_response < 3.0:
            grade = "✅ GOOD"
        elif total_response < 4.0:
            grade = "⚠️  ACCEPTABLE"
        else:
            grade = "❌ NEEDS OPTIMIZATION"
        
        print(f"\n{grade}")
        print("=" * 70)


async def main():
    benchmark = JarvisBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())