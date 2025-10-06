#!/bin/bash
# optimize_system.sh - Optimización del sistema para Jarvis
# Ejecutar con: sudo bash optimize_system.sh

set -e

echo "=========================================="
echo "🚀 JARVIS SYSTEM OPTIMIZATION"
echo "=========================================="

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar permisos root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root${NC}" 
   exit 1
fi

echo -e "\n${GREEN}1. Optimizing CPU Governor...${NC}"
# Cambiar a performance mode
if command -v cpupower &> /dev/null; then
    cpupower frequency-set -g performance 2>/dev/null || echo "⚠️  cpupower not available"
    echo "✅ CPU governor set to performance"
else
    echo "⚠️  cpupower not installed, skipping"
fi

echo -e "\n${GREEN}2. Adjusting swappiness...${NC}"
# Reducir swappiness (menos swap, más RAM)
current_swappiness=$(cat /proc/sys/vm/swappiness)
echo "   Current swappiness: $current_swappiness"
sysctl -w vm.swappiness=10
echo "   ✅ Swappiness set to 10 (temporary)"
echo "   💡 To make permanent, add 'vm.swappiness=10' to /etc/sysctl.conf"

echo -e "\n${GREEN}3. Optimizing network stack...${NC}"
# Aumentar buffers TCP
sysctl -w net.core.rmem_max=16777216 2>/dev/null || true
sysctl -w net.core.wmem_max=16777216 2>/dev/null || true
sysctl -w net.ipv4.tcp_rmem='4096 87380 16777216' 2>/dev/null || true
sysctl -w net.ipv4.tcp_wmem='4096 65536 16777216' 2>/dev/null || true
echo "✅ Network buffers optimized"

echo -e "\n${GREEN}4. Setting process priorities...${NC}"
# Aumentar prioridad de procesos Python (Jarvis)
pgrep -f "python.*main.py" | while read pid; do
    renice -n -10 -p $pid 2>/dev/null && echo "   ✅ Process $pid renice to -10" || true
done

echo -e "\n${GREEN}5. Optimizing I/O scheduler...${NC}"
# Cambiar scheduler a deadline para SSDs
for disk in /sys/block/sd*/queue/scheduler; do
    if [ -f "$disk" ]; then
        echo deadline > "$disk" 2>/dev/null || true
        echo "   ✅ $(basename $(dirname $(dirname $disk))): deadline"
    fi
done

echo -e "\n${GREEN}6. Disabling unnecessary services...${NC}"
# Lista de servicios a desactivar (opcionales)
services_to_disable=(
    "bluetooth.service"
    "cups.service"
    "avahi-daemon.service"
)

for service in "${services_to_disable[@]}"; do
    if systemctl is-active --quiet "$service" 2>/dev/null; then
        read -p "   Disable $service? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            systemctl stop "$service"
            systemctl disable "$service"
            echo "   ✅ $service disabled"
        fi
    fi
done

echo -e "\n${GREEN}7. Setting ulimits...${NC}"
# Aumentar límites de archivos abiertos
ulimit -n 65536 2>/dev/null || true
echo "   ✅ File descriptors: $(ulimit -n)"

echo -e "\n${GREEN}8. Creating optimization script for user...${NC}"
cat > /usr/local/bin/jarvis-optimize << 'EOF'
#!/bin/bash
# Script de optimización para usuario normal

echo "🚀 Optimizing Jarvis environment..."

# Variables de entorno para Python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# PyTorch optimizations
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0

# Aumentar prioridad del proceso actual
renice -n -10 -p $$ 2>/dev/null || true

echo "✅ Environment optimized"
echo "💡 Run your Jarvis with: source jarvis-optimize && python main.py"
EOF

chmod +x /usr/local/bin/jarvis-optimize
echo "   ✅ Created /usr/local/bin/jarvis-optimize"

echo -e "\n${GREEN}9. Creating systemd service (optional)...${NC}"
read -p "   Create systemd service for auto-start? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "   Enter Jarvis installation path: " jarvis_path
    read -p "   Enter username: " jarvis_user
    
    cat > /etc/systemd/system/jarvis.service << EOF
[Unit]
Description=Jarvis Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=$jarvis_user
WorkingDirectory=$jarvis_path
Environment="PYTHONUNBUFFERED=1"
Environment="OMP_NUM_THREADS=4"
ExecStart=/usr/bin/python3 $jarvis_path/main.py
Restart=on-failure
RestartSec=10
Nice=-10

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    echo "   ✅ Service created"
    echo "   💡 Enable with: sudo systemctl enable jarvis"
    echo "   💡 Start with: sudo systemctl start jarvis"
fi

echo -e "\n=========================================="
echo -e "${GREEN}✅ OPTIMIZATION COMPLETE${NC}"
echo "=========================================="
echo -e "${YELLOW}💡 Recommended next steps:${NC}"
echo "   1. Run: source /usr/local/bin/jarvis-optimize"
echo "   2. Start Jarvis: python main.py"
echo "   3. Monitor with: python performance_monitor.py"
echo ""
echo -e "${YELLOW}⚠️  Note: Some optimizations are temporary.${NC}"
echo "   Run this script after each reboot, or create systemd service."
echo "=========================================="