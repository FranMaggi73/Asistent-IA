#!/bin/bash
# start_jarvis.sh - Script optimizado de inicio
# Uso: bash start_jarvis.sh

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Banner
echo -e "${BLUE}"
cat << "EOF"
     ___  ________  ________  ___      ___ ___  ________      
    |\  \|\   __  \|\   __  \|\  \    /  /|\  \|\   ____\     
    \ \  \ \  \|\  \ \  \|\  \ \  \  /  / | \  \ \  \___|_    
  __ \ \  \ \   __  \ \   _  _\ \  \/  / / \ \  \ \_____  \   
 |\  \\_\  \ \  \ \  \ \  \\  \\ \    / /   \ \  \|____|\  \  
 \ \________\ \__\ \__\ \__\\ _\\ \__/ /     \ \__\____\_\  \ 
  \|________|\|__|\|__|\|__|\|__|\|__|/       \|__|\_________\
                                                  \|_________|
EOF
echo -e "${NC}"
echo "=========================================="
echo "🚀 JARVIS OPTIMIZED LAUNCHER v2.0"
echo "=========================================="

# Directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1. Verificar entorno virtual
echo -e "\n${GREEN}1. Checking virtual environment...${NC}"
if [ ! -d "jarvis" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    read -p "   Create it? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        python3 -m venv jarvis
        source jarvis/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${RED}❌ Aborted${NC}"
        exit 1
    fi
else
    source jarvis/bin/activate
    echo -e "✅ Virtual environment activated"
fi

# 2. Verificar dependencias
echo -e "\n${GREEN}2. Checking dependencies...${NC}"
python -c "import torch; import whisper; import rasa_sdk" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Missing dependencies${NC}"
    read -p "   Install them? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        pip install -r requirements.txt
        echo -e "${GREEN}✅ Dependencies installed${NC}"
    fi
fi
echo "✅ Dependencies OK"

# 3. Verificar archivos críticos
echo -e "\n${GREEN}3. Checking critical files...${NC}"
critical_files=(
    "main.py"
    "audioFunctions.py"
    "audio_listener.py"
    "rasa_client.py"
    "speaker.wav"
    ".env"
)

missing_files=0
for file in "${critical_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ Missing: $file${NC}"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo -e "${RED}❌ $missing_files critical file(s) missing${NC}"
    exit 1
fi
echo "✅ All critical files present"

# 4. Verificar servicios
echo -e "\n${GREEN}4. Checking services...${NC}"

# Rasa
if curl -s http://localhost:5005/status > /dev/null 2>&1; then
    echo "✅ Rasa server: ONLINE"
else
    echo -e "${YELLOW}⚠️  Rasa server: OFFLINE${NC}"
    read -p "   Start Rasa with Docker? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d
            echo "⏳ Waiting for Rasa to start..."
            sleep 10
            echo "✅ Rasa started"
        else
            echo -e "${RED}❌ docker-compose not found${NC}"
            echo "   Install Docker: https://docs.docker.com/get-docker/"
            exit 1
        fi
    fi
fi

# Action server
if curl -s http://localhost:5055/health > /dev/null 2>&1; then
    echo "✅ Action server: ONLINE"
else
    echo -e "${YELLOW}⚠️  Action server: OFFLINE${NC}"
    echo "   Start manually: rasa run actions"
fi

# 5. Optimizaciones de entorno
echo -e "\n${GREEN}5. Applying optimizations...${NC}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0
echo "✅ Environment variables set"

# 6. Verificar GPU (opcional)
echo -e "\n${GREEN}6. Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        echo "✅ CUDA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
    fi
else
    echo "⚠️  No GPU detected (using CPU)"
fi

# 7. Opciones de inicio
echo -e "\n=========================================="
echo -e "${BLUE}Launch Options:${NC}"
echo "=========================================="
echo "1. Start Jarvis (normal)"
echo "2. Start with performance monitor"
echo "3. Run benchmark first"
echo "4. Start in debug mode"
echo "5. Exit"
echo ""
read -p "Select option (1-5): " -n 1 -r option
echo ""

case $option in
    1)
        echo -e "\n${GREEN}🚀 Starting Jarvis...${NC}"
        python main.py
        ;;
    2)
        echo -e "\n${GREEN}🚀 Starting Jarvis with monitor...${NC}"
        # Iniciar monitor en background
        python performance_monitor.py &
        MONITOR_PID=$!
        sleep 2
        # Iniciar Jarvis
        python main.py
        # Matar monitor al salir
        kill $MONITOR_PID 2>/dev/null || true
        ;;
    3)
        echo -e "\n${GREEN}🧪 Running benchmark...${NC}"
        python benchmark.py
        echo ""
        read -p "Start Jarvis now? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            python main.py
        fi
        ;;
    4)
        echo -e "\n${GREEN}🐛 Starting in debug mode...${NC}"
        python -u main.py 2>&1 | tee jarvis_debug.log
        ;;
    5)
        echo -e "${YELLOW}👋 Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Invalid option${NC}"
        exit 1
        ;;
esac