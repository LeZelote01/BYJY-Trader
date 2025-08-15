"""
🔐 Kraken Authentication - Phase 2.4

Gestion sécurisée de l'authentification Kraken Pro API.
Support API key/secret avec signature HMAC-SHA512.

Features:
- Signature requêtes sécurisée
- Gestion timestamps et nonces  
- Headers authentification
- Validation permissions API
"""

import hmac
import hashlib
import time
import base64
from urllib.parse import urlencode
from typing import Dict, Any, Optional
import aiohttp

from core.logger import get_logger

logger = get_logger(__name__)


class KrakenAuth:
    """
    Gestionnaire d'authentification Kraken Pro
    
    Implémente la logique de signature selon les
    spécifications API Kraken.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialise l'authentification Kraken
        
        Args:
            api_key: Clé API Kraken
            api_secret: Secret API Kraken (base64)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.permissions: Optional[Dict[str, bool]] = None
        self.account_type: Optional[str] = None
        
        logger.info("🔐 KrakenAuth initialisé")
    
    def generate_signature(self, urlpath: str, data: Dict[str, Any], nonce: str) -> str:
        """
        Générer signature HMAC-SHA512 pour requête Kraken
        
        Args:
            urlpath: Path de l'API
            data: Données de la requête
            nonce: Nonce unique
            
        Returns:
            str: Signature base64
        """
        try:
            # Préparer message à signer
            postdata = urlencode(data)
            encoded = (nonce + postdata).encode()
            message = urlpath.encode() + hashlib.sha256(encoded).digest()
            
            # Décoder le secret base64
            secret_decoded = base64.b64decode(self.api_secret)
            
            # Générer signature HMAC-SHA512
            signature = hmac.new(secret_decoded, message, hashlib.sha512)
            
            # Encoder en base64
            return base64.b64encode(signature.digest()).decode()
            
        except Exception as e:
            logger.error(f"❌ Erreur génération signature Kraken: {e}")
            raise
    
    def prepare_signed_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Préparer requête signée Kraken
        
        Args:
            method: Méthode HTTP
            endpoint: Endpoint API
            params: Paramètres de la requête
            
        Returns:
            Dict avec URL, headers, et données préparées
        """
        if params is None:
            params = {}
        
        # Générer nonce
        nonce = str(int(time.time() * 1000000))
        params['nonce'] = nonce
        
        # Générer signature
        signature = self.generate_signature(endpoint, params, nonce)
        
        # Préparer headers
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        return {
            "method": method.upper(),
            "headers": headers,
            "data": params
        }
    
    async def test_connectivity(self, base_url: str) -> Dict[str, Any]:
        """
        Tester connectivité et permissions API Kraken
        
        Args:
            base_url: URL de base de l'API
            
        Returns:
            Dict avec résultats test
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: Server time (public)
                time_url = f"{base_url}/0/public/Time"
                start_time = time.time()
                
                async with session.get(time_url) as response:
                    if response.status == 200:
                        latency = (time.time() - start_time) * 1000
                        logger.info(f"✅ Ping Kraken réussi - Latence: {latency:.2f}ms")
                    else:
                        return {
                            "success": False,
                            "error": f"Ping failed: {response.status}"
                        }
                
                # Test 2: Account balance (auth requise)
                balance_request = self.prepare_signed_request(
                    "POST",
                    "/0/private/Balance"
                )
                
                balance_url = f"{base_url}/0/private/Balance"
                
                async with session.post(
                    balance_url,
                    headers=balance_request["headers"],
                    data=balance_request["data"]
                ) as response:
                    
                    if response.status == 200:
                        balance_data = await response.json()
                        
                        if balance_data.get("error"):
                            return {
                                "success": False,
                                "error": "; ".join(balance_data["error"])
                            }
                        
                        # Déterminer permissions
                        self.permissions = {
                            "spot": True,  # Kraken supporte trading spot
                            "margin": False,  # Non vérifié ici
                            "futures": False  # Futures séparés
                        }
                        
                        self.account_type = "KRAKEN_SPOT"
                        
                        balance_count = len(balance_data.get("result", {}))
                        logger.info(f"✅ Authentification Kraken réussie - {balance_count} assets")
                        
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
            logger.error(f"❌ Erreur test connectivité Kraken: {e}")
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