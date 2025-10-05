#!/usr/bin/env python3
"""
utils.py - Herramientas de optimización y monitoreo para Jarvis
"""

import os
import sys
import psutil
import time
from pathlib import Path
from typing import Dict, List


class PerformanceMonitor:
    """Monitor de rendimiento del asistente"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.metrics = {
            'transcriptions': 0,
            'tts_generations': 0,
            'rasa_requests': 0,
            'errors': 0
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Retorna uso de memoria en MB"""
        mem = self.process.memory_info()
        return {
            'rss': mem.rss / 1024 / 1024,  # MB
            'vms': mem.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
    
    def get_cpu_usage(self) -> float:
        """Retorna uso de CPU en %"""
        return self.process.cpu_percent(interval=1)
    
    def get_uptime(self) -> float:
        """Retorna tiempo de ejecución en segundos"""
        return time.time() - self.start_time
    
    def increment_metric(self, metric_name: str):
        """Incrementa contador de métrica"""
        if metric_name in self.metrics:
            self.metrics[metric_name] += 1
    
    def get_report(self) -> str:
        """Genera reporte de rendimiento"""
        mem = self.get_memory_usage()
        cpu = self.get_cpu_usage()
        uptime = self.get_uptime()
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║              JARVIS PERFORMANCE REPORT                   ║
╠══════════════════════════════════════════════════════════╣
║ Uptime:          {uptime/60:.1f} minutes                 
║ Memory (RSS):    {mem['rss']:.1f} MB ({mem['percent']:.1f}%)
║ CPU Usage:       {cpu:.1f}%                              
╠══════════════════════════════════════════════════════════╣
║ Transcriptions:  {self.metrics['transcriptions']}
║ TTS Generated:   {self.metrics['tts_generations']}
║ Rasa Requests:   {self.metrics['rasa_requests']}
║ Errors:          {self.metrics['errors']}
╚══════════════════════════════════════════════════════════╝
"""
        return report


class CacheManager:
    """Gestiona cachés del sistema"""
    
    @staticmethod
    def clear_pycache(root_dir: Path = Path('.')):
        """Elimina archivos __pycache__"""
        count = 0
        for pycache in root_dir.rglob('__pycache__'):
            if pycache.is_dir():
                import shutil
                shutil.rmtree(pycache)
                count += 1
        print(f"🗑️  Removed {count} __pycache__ directories")
    
    @staticmethod
    def clear_model_cache():
        """Limpia cache de modelos Torch"""
        cache_dir = Path.home() / '.cache' / 'torch'
        if cache_dir.exists():
            size_before = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            size_mb = size_before / 1024 / 1024
            print(f"📦 Torch cache: {size_mb:.1f} MB")
            response = input("Clear cache? (y/n): ")
            if response.lower() == 'y':
                import shutil
                shutil.rmtree(cache_dir)
                print("✅ Cache cleared")
    
    @staticmethod
    def show_cache_sizes():
        """Muestra tamaños de cachés"""
        caches = {
            'Torch': Path.home() / '.cache' / 'torch',
            'Whisper': Path.home() / '.cache' / 'whisper',
            'TTS': Path.home() / '.local' / 'share' / 'tts'
        }
        
        print("\n📊 Cache Sizes:")
        for name, path in caches.items():
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                size_mb = size / 1024 / 1024
                print(f"   {name:10} {size_mb:8.1f} MB  ({path})")
            else:
                print(f"   {name:10} Not found")


class RasaValidator:
    """Valida configuración de Rasa"""
    
    @staticmethod
    def validate_nlu(nlu_file: Path) -> List[str]:
        """Valida archivo NLU"""
        issues = []
        
        if not nlu_file.exists():
            issues.append(f"❌ NLU file not found: {nlu_file}")
            return issues
        
        try:
            import yaml
            with open(nlu_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'nlu' not in data:
                issues.append("⚠️  'nlu' key not found in NLU file")
                return issues
            
            intents = data['nlu']
            intent_names = [intent.get('intent') for intent in intents]
            
            # Check for duplicates
            duplicates = [name for name in set(intent_names) if intent_names.count(name) > 1]
            if duplicates:
                issues.append(f"⚠️  Duplicate intents: {duplicates}")
            
            # Check example count
            for intent in intents:
                examples = intent.get('examples', '')
                example_count = len([l for l in examples.split('\n') if l.strip() and l.strip().startswith('-')])
                
                if example_count < 5:
                    issues.append(f"⚠️  Intent '{intent['intent']}' has only {example_count} examples (recommend 10+)")
            
            if not issues:
                issues.append(f"✅ NLU file valid ({len(intents)} intents)")
        
        except Exception as e:
            issues.append(f"❌ Error parsing NLU: {e}")
        
        return issues
    
    @staticmethod
    def validate_domain(domain_file: Path) -> List[str]:
        """Valida archivo domain"""
        issues = []
        
        if not domain_file.exists():
            issues.append(f"❌ Domain file not found: {domain_file}")
            return issues
        
        try:
            import yaml
            with open(domain_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Check required keys
            required_keys = ['intents', 'responses', 'actions']
            for key in required_keys:
                if key not in data:
                    issues.append(f"⚠️  Missing '{key}' in domain")
            
            # Check actions
            if 'actions' in data:
                custom_actions = [a for a in data['actions'] if a.startswith('action_')]
                if custom_actions:
                    issues.append(f"ℹ️  Found {len(custom_actions)} custom actions")
            
            if not issues:
                issues.append("✅ Domain file valid")
        
        except Exception as e:
            issues.append(f"❌ Error parsing domain: {e}")
        
        return issues


def benchmark_audio():
    """Benchmarks para funciones de audio"""
    print("\n🎯 Audio Benchmarking\n")
    
    from audioFunctions import recordAudio, whisperTranscription
    import numpy as np
    
    # Test transcription speed
    print("Testing transcription speed...")
    dummy_audio = np.random.randint(-32768, 32767, size=44100*3, dtype=np.int16)  # 3 seconds
    
    start = time.time()
    result = whisperTranscription(dummy_audio)
    elapsed = time.time() - start
    
    print(f"⏱️  Transcription: {elapsed:.2f}s (3s audio)")
    print(f"   Real-time factor: {3/elapsed:.2f}x\n")


def optimize_rasa_model():
    """Sugerencias para optimizar modelo Rasa"""
    print("""
🚀 RASA MODEL OPTIMIZATION TIPS:

1. Reduce epochs in config.yml:
   - DIETClassifier: epochs: 50 (instead of 100)
   - ResponseSelector: epochs: 50
   - TEDPolicy: epochs: 50

2. Use smaller pipeline:
   - Remove char_wb CountVectorsFeaturizer if not needed
   - Consider using only one CountVectorsFeaturizer

3. Limit training data:
   - Keep 10-15 examples per intent (remove extras)
   - Remove unused intents

4. Training command:
   rasa train --fixed-model-name jarvis --num-threads 4

5. Export optimized model:
   rasa export --format tar --out models/
""")


def main():
    """CLI principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jarvis Optimization Utils")
    parser.add_argument('command', choices=[
        'monitor', 'cache', 'validate', 'benchmark', 'optimize', 'clean'
    ])
    
    args = parser.parse_args()
    
    if args.command == 'monitor':
        monitor = PerformanceMonitor()
        print(monitor.get_report())
    
    elif args.command == 'cache':
        CacheManager.show_cache_sizes()
        print()
        response = input("Clean Python cache? (y/n): ")
        if response.lower() == 'y':
            CacheManager.clear_pycache()
    
    elif args.command == 'validate':
        print("\n🔍 Validating Rasa Configuration\n")
        
        issues = RasaValidator.validate_nlu(Path('data/nlu.yml'))
        for issue in issues:
            print(f"  {issue}")
        
        print()
        issues = RasaValidator.validate_domain(Path('domain.yml'))
        for issue in issues:
            print(f"  {issue}")
    
    elif args.command == 'benchmark':
        benchmark_audio()
    
    elif args.command == 'optimize':
        optimize_rasa_model()
    
    elif args.command == 'clean':
        CacheManager.clear_pycache()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("""
Usage: python utils.py <command>

Commands:
  monitor    - Show performance metrics
  cache      - Show and manage cache
  validate   - Validate Rasa configuration
  benchmark  - Run audio benchmarks
  optimize   - Show optimization tips
  clean      - Clean Python cache
""")
    else:
        main()