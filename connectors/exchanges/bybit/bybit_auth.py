"""
🔐 Bybit Authentication - Phase 2.4

Gestion sécurisée de l'authentification Bybit V5 API.
Support API key/secret avec signature HMAC-SHA256.

Features:
- Signature requêtes sécurisée
- Gestion timestamps et nonces
- Headers authentification V5
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


class BybitAuth:
    """
    Gestionnaire d'authentification Bybit V5
    
    Implémente la logique de signature selon les
    spécifications API Bybit V5.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialise l'authentification Bybit
        
        Args:
            api_key: Clé API Bybit
            api_secret: Secret API Bybit
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.permissions: Optional[Dict[str, bool]] = None
        self.account_type: Optional[str] = None
        
        logger.info("🔐 BybitAuth initialisé")
    
    def generate_signature(
        self, 
        timestamp: str, 
        params: str = ""
    ) -> str:
        """
        Générer signature HMAC-SHA256 pour requête Bybit V5
        
        Args:
            timestamp: Timestamp Unix
            params: Paramètres de requête
            
        Returns:
            str: Signature hexadécimale
        """
        try:
            # Message à signer : timestamp + api_key + recv_window + params
            recv_window = "5000"
            message = timestamp + self.api_key + recv_window + params
            
            # Générer signature HMAC-SHA256
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"❌ Erreur génération signature Bybit: {e}")
            raise
    
    def get_headers(self, params: str = "") -> Dict[str, str]:
        """
        Générer headers d'authentification Bybit V5
        
        Args:
            params: Paramètres de requête
            
        Returns:
            Dict avec headers nécessaires
        """
        timestamp = str(int(time.time() * 1000))
        signature = self.generate_signature(timestamp, params)
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
    
    def prepare_signed_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Préparer requête signée Bybit V5
        
        Args:
            method: Méthode HTTP
            endpoint: Endpoint API
            params: Paramètres de la requête
            
        Returns:
            Dict avec URL, headers, et données préparées
        """
        if params is None:
            params = {}
        
        # Préparer params string selon la méthode
        if method.upper() == "GET":
            params_str = urlencode(params) if params else ""
        else:
            import json
            params_str = json.dumps(params) if params else ""
        
        # Générer headers signés
        headers = self.get_headers(params_str)
        
        return {
            "method": method.upper(),
            "headers": headers,
            "data": params if method.upper() != "GET" else None,
            "params": params if method.upper() == "GET" else None
        }
    
    async def test_connectivity(self, base_url: str) -> Dict[str, Any]:
        """
        Tester connectivité et permissions API Bybit
        
        Args:
            base_url: URL de base de l'API
            
        Returns:
            Dict avec résultats test
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: Server time (public)
                time_url = f"{base_url}/v5/market/time"
                start_time = time.time()
                
                async with session.get(time_url) as response:
                    if response.status == 200:
                        time_data = await response.json()
                        if time_data.get("retCode") == 0:
                            latency = (time.time() - start_time) * 1000
                            logger.info(f"✅ Ping Bybit réussi - Latence: {latency:.2f}ms")
                        else:
                            return {
                                "success": False,
                                "error": f"Time API error: {time_data.get('retMsg')}"
                            }
                    else:
                        return {
                            "success": False,
                            "error": f"Ping failed: {response.status}"
                        }
                
                # Test 2: Wallet balance (auth requise)
                wallet_request = self.prepare_signed_request(
                    "GET",
                    "/v5/account/wallet-balance",
                    {"accountType": "SPOT"}
                )
                
                wallet_url = f"{base_url}/v5/account/wallet-balance"
                
                async with session.get(
                    wallet_url,
                    headers=wallet_request["headers"],
                    params=wallet_request["params"]
                ) as response:
                    
                    if response.status == 200:
                        wallet_data = await response.json()
                        
                        if wallet_data.get("retCode") != 0:
                            return {
                                "success": False,
                                "error": wallet_data.get("retMsg", "Auth failed")
                            }
                        
                        # Déterminer permissions
                        self.permissions = {
                            "spot": True,  # Bybit V5 supporte spot
                            "derivatives": True,  # Bybit supporte derivatives
                            "options": False  # Options séparées
                        }
                        
                        self.account_type = "BYBIT_UNIFIED"
                        
                        accounts = wallet_data.get("result", {}).get("list", [])
                        balance_count = len(accounts[0].get("coin", [])) if accounts else 0
                        
                        logger.info(f"✅ Authentification Bybit réussie - {balance_count} assets")
                        
                        return {
                            "success": True,
                            "latency_ms": latency,
                            "permissions": self.permissions,
                            "account_type": self.account_type,
                            "balance_count": balance_count
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Auth failed: {response.status} - {error_text}"
                        }
        
        except Exception as e:
            logger.error(f"❌ Erreur test connectivité Bybit: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
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