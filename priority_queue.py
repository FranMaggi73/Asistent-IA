"""
Sistema de cola de prioridades para tareas de Jarvis
(Actualmente no usado, pero disponible para futuras optimizaciones)
"""

import asyncio
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Callable, Any
import time


class TaskPriority(IntEnum):
    """Niveles de prioridad (menor número = mayor prioridad)"""
    CRITICAL = 0   # Wake word detection, user input
    HIGH = 1       # Speech transcription, TTS
    MEDIUM = 2     # action execution
    LOW = 3        # Cache updates, logging
    BACKGROUND = 4 # Model preloading, cleanup


# Sistema de prioridades (placeholder para uso futuro)
# Puedes ignorar este archivo por ahora