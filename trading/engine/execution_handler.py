"""
⚡ Execution Handler - Gestionnaire d'Exécution
Gestion de l'exécution des signaux de trading
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from core.logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    """Types de signaux de trading"""
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class TradingSignal:
    """Signal de trading généré par une stratégie"""
    strategy_id: str
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 à 1.0
    suggested_quantity: Optional[float] = None
    suggested_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "suggested_quantity": self.suggested_quantity,
            "suggested_price": self.suggested_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class ExecutionHandler:
    """
    Gestionnaire d'exécution des signaux de trading
    Transforme les signaux des stratégies en ordres/positions
    """
    
    def __init__(self):
        self.order_manager = None
        self.position_manager = None
        
        # Configuration d'exécution
        self.config = {
            "min_confidence_threshold": 0.6,  # Confiance minimum pour exécuter
            "max_position_size": 1000.0,      # Taille max position
            "enable_stop_loss": True,
            "enable_take_profit": True,
            "paper_trading": True
        }
        
        # Historique des signaux
        self.signal_history: List[TradingSignal] = []
        
        # Statistiques
        self.stats = {
            "signals_received": 0,
            "signals_executed": 0,
            "signals_ignored": 0,
            "execution_errors": 0
        }
        
        logger.info("ExecutionHandler initialisé")
    
    def set_managers(self, order_manager, position_manager):
        """Configure les gestionnaires d'ordres et positions"""
        self.order_manager = order_manager
        self.position_manager = position_manager
        logger.info("Gestionnaires configurés dans ExecutionHandler")
    
    def process_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Traite un signal de trading
        Retourne (succès, message)
        """
        try:
            # Enregistrement du signal
            self.signal_history.append(signal)
            self.stats["signals_received"] += 1
            
            logger.info(f"Signal reçu - {signal.symbol} {signal.signal_type.value} (conf: {signal.confidence})")
            
            # Validation du signal
            is_valid, validation_msg = self._validate_signal(signal)
            if not is_valid:
                self.stats["signals_ignored"] += 1
                logger.warning(f"Signal ignoré: {validation_msg}")
                return False, validation_msg
            
            # Exécution selon le type de signal
            success, message = self._execute_signal(signal)
            
            if success:
                self.stats["signals_executed"] += 1
                logger.info(f"Signal exécuté avec succès: {message}")
            else:
                self.stats["execution_errors"] += 1
                logger.error(f"Erreur exécution signal: {message}")
            
            return success, message
            
        except Exception as e:
            self.stats["execution_errors"] += 1
            error_msg = f"Erreur traitement signal: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Valide un signal de trading
        """
        # Vérification confiance minimum
        if signal.confidence < self.config["min_confidence_threshold"]:
            return False, f"Confiance trop faible: {signal.confidence} < {self.config['min_confidence_threshold']}"
        
        # Vérification gestionnaires
        if not self.order_manager or not self.position_manager:
            return False, "Gestionnaires non configurés"
        
        # Vérification symbole
        if not signal.symbol or len(signal.symbol) < 3:
            return False, "Symbole invalide"
        
        # Vérification quantité suggérée
        if signal.suggested_quantity and signal.suggested_quantity > self.config["max_position_size"]:
            return False, f"Quantité trop importante: {signal.suggested_quantity} > {self.config['max_position_size']}"
        
        return True, "Signal valide"
    
    def _execute_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Exécute un signal de trading
        """
        try:
            if signal.signal_type == SignalType.BUY:
                return self._execute_buy_signal(signal)
            elif signal.signal_type == SignalType.SELL:
                return self._execute_sell_signal(signal)
            elif signal.signal_type == SignalType.CLOSE_LONG:
                return self._execute_close_long_signal(signal)
            elif signal.signal_type == SignalType.CLOSE_SHORT:
                return self._execute_close_short_signal(signal)
            elif signal.signal_type == SignalType.HOLD:
                return True, "Signal HOLD - Aucune action requise"
            else:
                return False, f"Type de signal non supporté: {signal.signal_type}"
                
        except Exception as e:
            return False, f"Erreur exécution: {str(e)}"
    
    def _execute_buy_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Exécute un signal d'achat"""
        try:
            # Calcul de la quantité
            quantity = signal.suggested_quantity or self._calculate_position_size(signal)
            
            # Prix d'ordre
            price = signal.suggested_price  # None pour market order
            
            # Création de l'ordre d'achat
            order_id = self.order_manager.create_order(
                symbol=signal.symbol,
                side="buy",
                quantity=quantity,
                order_type="market" if price is None else "limit",
                price=price,
                strategy_id=signal.strategy_id
            )
            
            # Si ordre market exécuté immédiatement, créer la position
            if price is None:  # Market order
                position_id = self.position_manager.open_position(
                    symbol=signal.symbol,
                    side="long",
                    quantity=quantity,
                    entry_price=signal.suggested_price or 50000,  # Prix simulé
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_id=signal.strategy_id
                )
                return True, f"Position LONG ouverte - Order: {order_id}, Position: {position_id}"
            else:
                return True, f"Ordre d'achat créé - ID: {order_id}"
                
        except Exception as e:
            return False, f"Erreur signal BUY: {str(e)}"
    
    def _execute_sell_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Exécute un signal de vente"""
        try:
            # Calcul de la quantité
            quantity = signal.suggested_quantity or self._calculate_position_size(signal)
            
            # Prix d'ordre
            price = signal.suggested_price
            
            # Création de l'ordre de vente
            order_id = self.order_manager.create_order(
                symbol=signal.symbol,
                side="sell",
                quantity=quantity,
                order_type="market" if price is None else "limit",
                price=price,
                strategy_id=signal.strategy_id
            )
            
            # Si ordre market exécuté immédiatement, créer la position short
            if price is None:  # Market order
                position_id = self.position_manager.open_position(
                    symbol=signal.symbol,
                    side="short",
                    quantity=quantity,
                    entry_price=signal.suggested_price or 50000,  # Prix simulé
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_id=signal.strategy_id
                )
                return True, f"Position SHORT ouverte - Order: {order_id}, Position: {position_id}"
            else:
                return True, f"Ordre de vente créé - ID: {order_id}"
                
        except Exception as e:
            return False, f"Erreur signal SELL: {str(e)}"
    
    def _execute_close_long_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Ferme les positions longues"""
        try:
            # Trouver les positions longues pour ce symbole
            position = self.position_manager.get_position(signal.symbol)
            if not position:
                return False, f"Aucune position LONG trouvée pour {signal.symbol}"
            
            if position["side"] != "long":
                return False, f"Position {signal.symbol} n'est pas LONG"
            
            # Fermeture de la position
            success = self.position_manager.close_position(
                position["id"], 
                signal.suggested_price
            )
            
            if success:
                return True, f"Position LONG fermée - ID: {position['id']}"
            else:
                return False, "Échec fermeture position LONG"
                
        except Exception as e:
            return False, f"Erreur signal CLOSE_LONG: {str(e)}"
    
    def _execute_close_short_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Ferme les positions courtes"""
        try:
            # Trouver les positions courtes pour ce symbole
            position = self.position_manager.get_position(signal.symbol)
            if not position:
                return False, f"Aucune position SHORT trouvée pour {signal.symbol}"
            
            if position["side"] != "short":
                return False, f"Position {signal.symbol} n'est pas SHORT"
            
            # Fermeture de la position
            success = self.position_manager.close_position(
                position["id"], 
                signal.suggested_price
            )
            
            if success:
                return True, f"Position SHORT fermée - ID: {position['id']}"
            else:
                return False, "Échec fermeture position SHORT"
                
        except Exception as e:
            return False, f"Erreur signal CLOSE_SHORT: {str(e)}"
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """
        Calcule la taille de position basée sur le risk management
        """
        # Pour l'instant, taille fixe - sera amélioré avec le risk manager
        base_size = 100.0
        
        # Ajustement basé sur la confiance
        size = base_size * signal.confidence
        
        # Limitation selon la config
        return min(size, self.config["max_position_size"])
    
    def get_signal_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne l'historique des signaux"""
        try:
            # Trier par timestamp décroissant
            sorted_signals = sorted(
                self.signal_history,
                key=lambda x: x.timestamp,
                reverse=True
            )
            
            # Limiter les résultats
            limited_signals = sorted_signals[:limit]
            
            return [signal.to_dict() for signal in limited_signals]
            
        except Exception as e:
            logger.error(f"Erreur récupération historique signaux: {str(e)}")
            return []
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'exécution"""
        total_signals = self.stats["signals_received"]
        
        return {
            **self.stats,
            "execution_rate": (self.stats["signals_executed"] / max(1, total_signals)) * 100,
            "error_rate": (self.stats["execution_errors"] / max(1, total_signals)) * 100,
            "ignore_rate": (self.stats["signals_ignored"] / max(1, total_signals)) * 100,
            "config": self.config
        }