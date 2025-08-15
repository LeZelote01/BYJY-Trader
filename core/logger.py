"""
📝 Logging System
Système de logging avancé avec support multi-niveaux et rotation
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import structlog
from rich.logging import RichHandler
from rich.console import Console

from .config import get_settings


class JSONFormatter(logging.Formatter):
    """Formatter JSON pour les logs structurés"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Ajouter les champs personnalisés
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # Ajouter l'exception si présente
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class TradingFilter(logging.Filter):
    """Filtre pour les logs de trading"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Filtrer les logs sensibles en production
        settings = get_settings()
        if settings.is_production():
            sensitive_fields = ['api_key', 'api key', 'secret', 'password', 'token']
            message = record.getMessage().lower()
            if any(field in message for field in sensitive_fields):
                return False
        return True


def setup_logging() -> None:
    """Configure le système de logging"""
    settings = get_settings()
    
    # Créer le répertoire logs
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration de base
    logging.basicConfig(level=logging.NOTSET, handlers=[])
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Console handler avec Rich
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=True,
        enable_link_path=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(TradingFilter())
    
    # File handler général avec rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logs_dir / "byjy_trader.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(TradingFilter())
    
    # Error handler séparé
    error_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logs_dir / "errors.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    
    # Trading handler pour les logs de trading
    trading_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logs_dir / "trading.log",
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=10,
        encoding="utf-8"
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(JSONFormatter())
    
    # Ajouter les handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Logger spécialisé pour le trading
    trading_logger = logging.getLogger("byjy.trading")
    trading_logger.addHandler(trading_handler)
    trading_logger.setLevel(logging.INFO)
    
    # Configurer structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Réduire les logs externes
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    
    # Log de démarrage
    logger = get_logger("byjy.startup")
    logger.info("Logging system initialized", extra={
        "log_level": settings.log_level,
        "logs_dir": str(settings.logs_dir),
        "environment": settings.environment
    })


def get_logger(name: str, extra_data: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Retourne un logger configuré avec des données extra optionnelles
    
    Args:
        name: Nom du logger (ex: "byjy.trading", "byjy.ai")
        extra_data: Données supplémentaires à inclure dans tous les logs
        
    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    
    if extra_data:
        # Créer un adaptateur pour inclure les données extra
        logger = logging.LoggerAdapter(logger, extra_data)
    
    return logger


def get_trading_logger(symbol: Optional[str] = None, strategy: Optional[str] = None) -> logging.Logger:
    """
    Retourne un logger spécialisé pour le trading
    
    Args:
        symbol: Symbole tradé (ex: "BTCUSDT")
        strategy: Nom de la stratégie (ex: "grid_trading")
        
    Returns:
        Logger trading configuré
    """
    extra_data = {}
    if symbol:
        extra_data["symbol"] = symbol
    if strategy:
        extra_data["strategy"] = strategy
    
    return get_logger("byjy.trading", extra_data)


def get_ai_logger(model: Optional[str] = None) -> logging.Logger:
    """
    Retourne un logger spécialisé pour l'IA
    
    Args:
        model: Nom du modèle IA (ex: "lstm_price_predictor")
        
    Returns:
        Logger IA configuré
    """
    extra_data = {}
    if model:
        extra_data["model"] = model
    
    return get_logger("byjy.ai", extra_data)


# Initialiser le logging au démarrage du module
try:
    setup_logging()
except Exception as e:
    # Fallback en cas d'erreur
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.error(f"Failed to setup advanced logging: {e}")