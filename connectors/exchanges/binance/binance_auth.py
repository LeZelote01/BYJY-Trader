"""
🔐 Binance Authentication - Phase 2.4

Gestion sécurisée de l'authentification Binance.
Support API key/secret avec signature HMAC-SHA256.

Features:
- Signature requêtes sécurisée
- Gestion timestamps et nonces
- Headers authentification
- Validation permissions API
"""

import hmac
import hashlib
import time
from urllib.parse import urlencode
from typing import Dict, Any, Optional
import aiohttp

from core.logger import get_logger

logger = get_logger(__name__)


class BinanceAuth:
    """
    Gestionnaire d'authentification Binance
    
    Implémente la logique de signature et d'authentification
    selon les spécifications API Binance.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialise l'authentification Binance
        
        Args:
            api_key: Clé API Binance
            api_secret: Secret API Binance
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.permissions: Optional[Dict[str, bool]] = None
        self.account_type: Optional[str] = None
        
        logger.info("🔐 BinanceAuth initialisé")
    
    def generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Générer signature HMAC-SHA256 pour requête
        
        Args:
            params: Paramètres de la requête
            
        Returns:
            str: Signature hexadécimale
        """
        try:
            # Créer query string
            query_string = urlencode(params, doseq=True)
            
            # Générer signature HMAC-SHA256
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"❌ Erreur génération signature: {e}")
            raise
    
    def get_headers(self, include_signature: bool = True) -> Dict[str, str]:
        """
        Générer headers d'authentification
        
        Args:
            include_signature: Inclure signature dans headers
            
        Returns:
            Dict avec headers nécessaires
        """
        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        return headers
    
    def prepare_signed_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Préparer requête signée
        
        Args:
            method: Méthode HTTP (GET, POST, etc.)
            endpoint: Endpoint API
            params: Paramètres de la requête
            
        Returns:
            Dict avec URL, headers, et données préparées
        """
        if params is None:
            params = {}
        
        # Ajouter timestamp si requis pour signature
        if self._requires_signature(endpoint):
            params['timestamp'] = int(time.time() * 1000)
            
            # Générer signature
            signature = self.generate_signature(params)
            params['signature'] = signature
        
        # Préparer headers
        headers = self.get_headers()
        
        # Construire URL et données selon méthode
        if method.upper() == "GET":
            query_string = urlencode(params) if params else ""
            url = f"{endpoint}?{query_string}" if query_string else endpoint
            data = None
        else:
            url = endpoint
            data = params
        
        return {
            "url": url,
            "headers": headers,
            "data": data,
            "method": method.upper()
        }
    
    async def test_connectivity(self, base_url: str) -> Dict[str, Any]:
        """
        Tester connectivité et permissions API
        
        Args:
            base_url: URL de base de l'API
            
        Returns:
            Dict avec résultats test
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: Ping serveur (pas d'auth requise)
                ping_url = f"{base_url}/api/v3/ping"
                start_time = time.time()
                
                async with session.get(ping_url) as response:
                    if response.status == 200:
                        latency = (time.time() - start_time) * 1000
                        logger.info(f"✅ Ping Binance réussi - Latence: {latency:.2f}ms")
                    else:
                        logger.error(f"❌ Échec ping Binance: {response.status}")
                        return {
                            "success": False,
                            "error": f"Ping failed: {response.status}"
                        }
                
                # Test 2: Account info (auth requise)
                account_request = self.prepare_signed_request(
                    "GET",
                    f"{base_url}/api/v3/account"
                )
                
                async with session.request(
                    account_request["method"],
                    account_request["url"],
                    headers=account_request["headers"],
                    json=account_request["data"]
                ) as response:
                    
                    if response.status == 200:
                        account_data = await response.json()
                        
                        # Extraire permissions
                        self.permissions = {
                            "spot": account_data.get("canTrade", False),
                            "margin": account_data.get("canMargin", False),
                            "futures": account_data.get("canFutures", False)
                        }
                        
                        self.account_type = account_data.get("accountType", "SPOT")
                        
                        logger.info(f"✅ Authentification Binance réussie - Type: {self.account_type}")
                        
                        return {
                            "success": True,
                            "latency_ms": latency,
                            "permissions": self.permissions,
                            "account_type": self.account_type,
                            "balances_count": len(account_data.get("balances", []))
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Échec authentification Binance: {response.status} - {error_text}")
                        
                        return {
                            "success": False,
                            "error": f"Auth failed: {response.status} - {error_text}"
                        }
        
        except Exception as e:
            logger.error(f"❌ Erreur test connectivité: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _requires_signature(self, endpoint: str) -> bool:
        """
        Vérifier si endpoint requiert signature
        
        Args:
            endpoint: Endpoint à vérifier
            
        Returns:
            bool: True si signature requise
        """
        # Endpoints publics (pas de signature)
        public_endpoints = [
            "/api/v3/ping",
            "/api/v3/time",
            "/api/v3/exchangeInfo",
            "/api/v3/depth",
            "/api/v3/trades",
            "/api/v3/historicalTrades",
            "/api/v3/aggTrades", 
            "/api/v3/klines",
            "/api/v3/ticker/24hr",
            "/api/v3/ticker/price",
            "/api/v3/ticker/bookTicker",
            "/api/v3/avgPrice"
        ]
        
        # Vérifier si endpoint est public
        for public_endpoint in public_endpoints:
            if public_endpoint in endpoint:
                return False
        
        # Par défaut, signature requise
        return True
    
    def validate_permissions(self, required_permission: str) -> bool:
        """
        Valider si permissions suffisantes
        
        Args:
            required_permission: Permission requise (spot, margin, futures)
            
        Returns:
            bool: True si permission accordée
        """
        if not self.permissions:
            logger.warning("⚠️ Permissions non encore chargées")
            return False
        
        return self.permissions.get(required_permission, False)
    
    def get_auth_info(self) -> Dict[str, Any]:
        """
        Récupérer informations d'authentification
        
        Returns:
            Dict avec infos auth (sans secrets)
        """
        return {
            "api_key_prefix": self.api_key[:8] + "..." if self.api_key else None,
            "permissions": self.permissions,
            "account_type": self.account_type,
            "authenticated": bool(self.permissions)
        }
    
    def __str__(self) -> str:
        return f"<BinanceAuth(authenticated={bool(self.permissions)}, type={self.account_type})>"
    
    def __repr__(self) -> str:
        return self.__str__()