"""
🔧 BYJY-Trader Core Module
Moteur de trading principal avec architecture modulaire
"""

__version__ = "0.1.0"
__author__ = "BYJY Team"
__description__ = "Bot de Trading Personnel Avancé avec IA"

# Core components exports
from .config import Settings, get_settings
from .database import DatabaseManager
from .logger import get_logger

__all__ = [
    "Settings",
    "get_settings", 
    "DatabaseManager",
    "get_logger"
]