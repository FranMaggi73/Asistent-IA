"""
Performance Monitor para Jarvis
Uso: python performance_monitor.py
"""

import psutil
import time
import threading
from collections import deque
from datetime import datetime


class PerformanceMonitor:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        
        # Historiales
        self.cpu_history = deque(maxlen=30)
        self.memory_history = deque(maxlen=30)
        self.network_history = deque(maxlen=30)
        
        # Proceso actual
        self.process = psutil.Process()
    
    def start(self):
        """Inicia monitoreo en thread separado"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitor started")
    
    def stop(self):
        """Detiene monitoreo"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)
        print("📊 Performance monitor stopped")
    
    def _monitor_loop(self):
        """Loop de monitoreo"""
        last_net_io = psutil.net_io_counters()
        
        while self.running:
            try:
                # CPU
                cpu_percent = self.process.cpu_percent(interval=None)
                self.cpu_history.append(cpu_percent)
                
                # Memoria
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_history.append(memory_mb)
                
                # Red
                current_net_io = psutil.net_io_counters()
                net_sent = (current_net_io.bytes_sent - last_net_io.bytes_sent) / 1024
                net_recv = (current_net_io.bytes_recv - last_net_io.bytes_recv) / 1024
                self.network_history.append((net_sent, net_recv))
                last_net_io = current_net_io
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"⚠️  Monitor error: {e}")
    
    def get_stats(self):
        """Retorna estadísticas actuales"""
        if not self.cpu_history:
            return None
        
        return {
            'cpu_percent': self.cpu_history[-1],
            'cpu_avg': sum(self.cpu_history) / len(self.cpu_history),
            'memory_mb': self.memory_history[-1],
            'memory_avg': sum(self.memory_history) / len(self.memory_history),
            'threads': self.process.num_threads(),
            'connections': len(self.process.connections()),
        }
    
    def print_report(self):
        """Imprime reporte detallado"""
        stats = self.get_stats()
        if not stats:
            print("⚠️  No data available")
            return
        
        print("\n" + "=" * 60)
        print(f"📊 PERFORMANCE REPORT - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        print(f"🔥 CPU Usage:")
        print(f"   Current: {stats['cpu_percent']:.1f}%")
        print(f"   Average: {stats['cpu_avg']:.1f}%")
        print(f"\n💾 Memory:")
        print(f"   Current: {stats['memory_mb']:.1f} MB")
        print(f"   Average: {stats['memory_avg']:.1f} MB")
        print(f"\n🔗 Connections:")
        print(f"   Active threads: {stats['threads']}")
        print(f"   Network connections: {stats['connections']}")
        
        # Grafico ASCII simple de CPU
        print(f"\n📈 CPU History (last 30 samples):")
        max_cpu = max(self.cpu_history) if self.cpu_history else 100
        for i, cpu in enumerate(list(self.cpu_history)[-20:]):
            bar_length = int((cpu / max_cpu) * 30)
            bar = "█" * bar_length
            print(f"   {i+1:2d} [{bar:<30}] {cpu:.1f}%")
        
        print("=" * 60)


# Integración con Jarvis
if __name__ == "__main__":
    import sys
    
    print("Starting performance monitor...")
    print("Press Ctrl+C to stop and show report\n")
    
    monitor = PerformanceMonitor(interval=2.0)
    monitor.start()
    
    try:
        while True:
            time.sleep(5)
            # Imprimir stats cada 5 segundos
            stats = monitor.get_stats()
            if stats:
                print(f"⚡ CPU: {stats['cpu_percent']:.1f}% | "
                      f"💾 RAM: {stats['memory_mb']:.0f}MB | "
                      f"🧵 Threads: {stats['threads']}", end='\r')
    
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        monitor.stop()
        monitor.print_report()
        sys.exit(0)