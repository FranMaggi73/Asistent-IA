#!/bin/bash
# start_all.sh - Inicia Jarvis completo en 3 terminales automáticamente
# Uso: bash start_all.sh

# Colores
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR=$(pwd)

echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}     🚀 JARVIS AUTO-LAUNCHER (3 Terminals)${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo ""

# Verificar entorno virtual
if [ ! -d "jarvis" ]; then
    echo -e "${YELLOW}⚠️  No se encontró entorno virtual 'jarvis'${NC}"
    exit 1
fi

# Función para detectar terminal emulator
detect_terminal() {
    if [ -n "$ALACRITTY_SOCKET" ]; then
        echo "alacritty"
    elif [ -n "$KITTY_WINDOW_ID" ]; then
        echo "kitty"
    elif command -v gnome-terminal &> /dev/null; then
        echo "gnome-terminal"
    elif command -v konsole &> /dev/null; then
        echo "konsole"
    elif command -v xterm &> /dev/null; then
        echo "xterm"
    elif command -v terminator &> /dev/null; then
        echo "terminator"
    elif [ "$OS" = "Windows_NT" ]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

TERMINAL=$(detect_terminal)

case $TERMINAL in
    gnome-terminal)
        echo -e "${GREEN}✓ Detectado: GNOME Terminal${NC}"
        
        # Tab 1: Rasa server
        gnome-terminal --tab --title="Rasa Server" -- bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            echo '🤖 Iniciando Rasa Server...' && \
            rasa run --enable-api --cors '*'; \
            exec bash"
        
        sleep 2
        
        # Tab 2: Action server
        gnome-terminal --tab --title="Actions" -- bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            echo '⚙️  Iniciando Action Server...' && \
            rasa run actions; \
            exec bash"
        
        sleep 3
        
        # Tab 3: Jarvis main
        gnome-terminal --tab --title="Jarvis" -- bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            echo '🎙️  Iniciando Jarvis...' && \
            sleep 5 && \
            python main.py; \
            exec bash"
        ;;
    
    konsole)
        echo -e "${GREEN}✓ Detectado: Konsole${NC}"
        
        konsole --new-tab -e bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            rasa run --enable-api --cors '*'; \
            exec bash" &
        
        sleep 2
        
        konsole --new-tab -e bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            rasa run actions; \
            exec bash" &
        
        sleep 3
        
        konsole --new-tab -e bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            sleep 5 && \
            python main.py; \
            exec bash" &
        ;;
    
    terminator)
        echo -e "${GREEN}✓ Detectado: Terminator${NC}"
        
        terminator -e "bash -c '
            cd \"$PROJECT_DIR\" && \
            source jarvis/bin/activate && \
            rasa run --enable-api --cors \"*\"; \
            exec bash'" &
        
        sleep 2
        
        terminator -e "bash -c '
            cd \"$PROJECT_DIR\" && \
            source jarvis/bin/activate && \
            rasa run actions; \
            exec bash'" &
        
        sleep 3
        
        terminator -e "bash -c '
            cd \"$PROJECT_DIR\" && \
            source jarvis/bin/activate && \
            sleep 5 && \
            python main.py; \
            exec bash'" &
        ;;
    
    xterm)
        echo -e "${GREEN}✓ Detectado: xterm${NC}"
        
        xterm -T "Rasa Server" -e bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            rasa run --enable-api --cors '*'; \
            exec bash" &
        
        sleep 2
        
        xterm -T "Actions" -e bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            rasa run actions; \
            exec bash" &
        
        sleep 3
        
        xterm -T "Jarvis" -e bash -c "
            cd '$PROJECT_DIR' && \
            source jarvis/bin/activate && \
            sleep 5 && \
            python main.py; \
            exec bash" &
        ;;
    
    windows)
        echo -e "${GREEN}✓ Detectado: Windows${NC}"
        
        # Rasa server
        start "Rasa Server" cmd /k "cd /d %cd% && jarvis\\Scripts\\activate && rasa run --enable-api --cors * && pause"
        
        timeout /t 2 /nobreak > nul
        
        # Actions
        start "Actions" cmd /k "cd /d %cd% && jarvis\\Scripts\\activate && rasa run actions && pause"
        
        timeout /t 3 /nobreak > nul
        
        # Jarvis
        start "Jarvis" cmd /k "cd /d %cd% && jarvis\\Scripts\\activate && timeout /t 5 /nobreak && python main.py && pause"
        ;;
    
    *)
        echo -e "${YELLOW}⚠️  Terminal no detectado automáticamente${NC}"
        echo ""
        echo "Por favor abre 3 terminales manualmente y ejecuta:"
        echo ""
        echo "Terminal 1:"
        echo "  cd $PROJECT_DIR"
        echo "  source jarvis/bin/activate"
        echo "  rasa run --enable-api --cors '*'"
        echo ""
        echo "Terminal 2:"
        echo "  cd $PROJECT_DIR"
        echo "  source jarvis/bin/activate"
        echo "  rasa run actions"
        echo ""
        echo "Terminal 3:"
        echo "  cd $PROJECT_DIR"
        echo "  source jarvis/bin/activate"
        echo "  python main.py"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Terminales lanzadas${NC}"
echo -e "${CYAN}💡 Espera ~10s a que Rasa inicie antes de usar Jarvis${NC}"
echo ""
echo -e "${YELLOW}Para detener todo: Ctrl+C en cada terminal${NC}"