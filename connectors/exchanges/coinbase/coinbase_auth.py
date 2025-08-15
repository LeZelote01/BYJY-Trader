"""
🔐 Coinbase Advanced Trading Authentication - Phase 2.4

Gestion sécurisée de l'authentification Coinbase Advanced Trading.
Support OAuth2 avec signature HMAC-SHA256.

Features:
- Signature JWT pour requêtes
- Gestion timestamps et nonces
- Headers authentification Coinbase
- Validation permissions API
"""

import hmac
import hashlib
import time
import base64
import json
from urllib.parse import urlencode
from typing import Dict, Any, Optional
import aiohttp

from core.logger import get_logger

logger = get_logger(__name__)


class CoinbaseAuth:
    """
    Gestionnaire d'authentification Coinbase Advanced Trading
    
    Implémente la logique de signature JWT selon les
    spécifications API Coinbase Advanced Trading.
    """
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str = None):
        """
        Initialise l'authentification Coinbase
        
        Args:
            api_key: Clé API Coinbase
            api_secret: Secret API Coinbase (base64)
            passphrase: Passphrase API (pour Coinbase Pro legacy)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.permissions: Optional[Dict[str, bool]] = None
        self.account_id: Optional[str] = None
        
        logger.info("🔐 CoinbaseAuth initialisé")
    
    def generate_signature(
        self, 
        timestamp: str, 
        method: str, 
        path: str, 
        body: str = ""
    ) -> str:
        """
        Générer signature HMAC-SHA256 pour requête Coinbase
        
        Args:
            timestamp: Timestamp Unix
            method: Méthode HTTP
            path: Path de l'API
            body: Corps de la requête
            
        Returns:
            str: Signature base64
        """
        try:
            # Message à signer : timestamp + method + path + body
            message = timestamp + method.upper() + path + body
            
            # Décoder le secret base64
            secret_decoded = base64.b64decode(self.api_secret)
            
            # Générer signature HMAC-SHA256
            signature = hmac.new(
                secret_decoded,
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            # Encoder en base64
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            return signature_b64
            
        except Exception as e:
            logger.error(f"❌ Erreur génération signature Coinbase: {e}")
            raise
    
    def get_headers(
        self, 
        method: str, 
        path: str, 
        body: str = ""
    ) -> Dict[str, str]:
        """
        Générer headers d'authentification Coinbase
        
        Args:
            method: Méthode HTTP
            path: Path de l'API
            body: Corps de la requête
            
        Returns:
            Dict avec headers nécessaires
        """
        timestamp = str(int(time.time()))
        
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-SIGN": self.generate_signature(timestamp, method, path, body),
            "Content-Type": "application/json"
        }
        
        # Ajouter passphrase si fourni (legacy Coinbase Pro)
        if self.passphrase:
            headers["CB-ACCESS-PASSPHRASE"] = self.passphrase
        
        return headers
    
    def prepare_signed_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Préparer requête signée Coinbase
        
        Args:
            method: Méthode HTTP
            endpoint: Endpoint API complet
            params: Paramètres URL query
            body: Corps de la requête
            
        Returns:
            Dict avec URL, headers, et données préparées
        """
        # Extraire le path depuis l'endpoint complet
        if "coinbase.com" in endpoint:
            path = "/" + endpoint.split("coinbase.com/", 1)[1]
        else:
            path = endpoint.replace("https://api.exchange.coinbase.com", "")
        
        # Ajouter query params au path
        if params:
            query_string = urlencode(params)
            path += f"?{query_string}"
        
        # Préparer body
        body_str = ""
        if body and method.upper() != "GET":
            body_str = json.dumps(body)
        
        # Générer headers signés
        headers = self.get_headers(method, path, body_str)
        
        # Construire URL finale
        base_url = endpoint.split("?")[0] if "?" in endpoint else endpoint
        if params and method.upper() == "GET":
            url = f"{base_url}?{urlencode(params)}"
        else:
            url = base_url
        
        return {
            "url": url,
            "headers": headers,
            "data": json.loads(body_str) if body_str else None,
            "method": method.upper()
        }
    
    async def test_connectivity(self, base_url: str) -> Dict[str, Any]:
        """
        Tester connectivité et permissions API Coinbase
        
        Args:
            base_url: URL de base de l'API
            
        Returns:
            Dict avec résultats test
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: Server time (public endpoint)
                time_url = f"{base_url}/time"
                start_time = time.time()
                
                async with session.get(time_url) as response:
                    if response.status == 200:
                        latency = (time.time() - start_time) * 1000
                        logger.info(f"✅ Ping Coinbase réussi - Latence: {latency:.2f}ms")
                    else:
                        logger.error(f"❌ Échec ping Coinbase: {response.status}")
                        return {
                            "success": False,
                            "error": f"Ping failed: {response.status}"
                        }
                
                # Test 2: User accounts (auth requise)
                accounts_request = self.prepare_signed_request(
                    "GET",
                    f"{base_url}/accounts"
                )
                
                async with session.request(
                    accounts_request["method"],
                    accounts_request["url"],
                    headers=accounts_request["headers"],
                    json=accounts_request["data"]
                ) as response:
                    
                    if response.status == 200:
                        accounts_data = await response.json()
                        
                        # Extraire informations comptes
                        accounts = accounts_data.get("accounts", [])
                        
                        # Déterminer permissions
                        self.permissions = {
                            "spot": True,  # Coinbase Advanced supporte trading spot
                            "margin": False,  # Non supporté actuellement
                            "futures": False  # Non supporté actuellement
                        }
                        
                        if accounts:
                            self.account_id = accounts[0].get("uuid")
                        
                        logger.info(f"✅ Authentification Coinbase réussie - {len(accounts)} comptes")
                        
                        return {
                            "success": True,
                            "latency_ms": latency,
                            "permissions": self.permissions,
                            "account_type": "COINBASE_ADVANCED",
                            "accounts_count": len(accounts)
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Échec authentification Coinbase: {response.status} - {error_text}")
                        
                        return {
                            "success": False,
                            "error": f"Auth failed: {response.status} - {error_text}"
                        }
        
        except Exception as e:
            logger.error(f"❌ Erreur test connectivité Coinbase: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_permissions(self, required_permission: str) -> bool:
        """
        Valider si permissions suffisantes
        
        Args:
            required_permission: Permission requise (spot, margin, futures)
            
        Returns:
            bool: True si permission accordée
        """
        if not self.permissions:
            logger.warning("⚠️ Permissions Coinbase non encore chargées")
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
            "account_type": "COINBASE_ADVANCED",
            "account_id": self.account_id,
            "authenticated": bool(self.permissions)
        }
    
    def __str__(self) -> str:
        return f"<CoinbaseAuth(authenticated={bool(self.permissions)}, account_id={self.account_id})>"
    
    def __repr__(self) -> str:
        return self.__str__()