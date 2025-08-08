# 🔑 Guide Complet - Clés API Exchanges

*Dernière mise à jour : 2025-01-08*

## 📋 Vue d'Ensemble

Ce guide vous explique comment obtenir et configurer les clés API pour chaque exchange supporté par BYJY-Trader. **Utilisez toujours les modes sandbox/testnet pour les tests avant de passer en production.**

---

## 🟡 1. BINANCE

### 📖 Étapes d'obtention

1. **Créer un compte** sur [Binance](https://www.binance.com) ou [Binance Testnet](https://testnet.binance.vision/)

2. **Accéder aux paramètres API** :
   - Connexion → Sécurité → Gestion API
   - Ou directement : `https://www.binance.com/en/my/settings/api-management`

3. **Créer une nouvelle API** :
   - Nom : "BYJY-Trader"
   - Type : "System generated"
   - Activer les permissions :
     - ✅ Enable Spot & Margin Trading
     - ✅ Enable Reading
     - ❌ Enable Futures (optionnel)
     - ❌ Enable Withdrawals (non recommandé)

4. **Récupérer les clés** :
   - **API Key** : Copier la clé publique
   - **Secret Key** : Copier lors de la création (invisible après)

### 🔒 Configuration BYJY-Trader

```
Exchange: Binance
API Key: votre_binance_api_key
Secret Key: votre_binance_secret_key  
Mode: Sandbox (testnet.binance.vision)
Permissions: Spot Trading + Read
```

### 🧪 URLs de Test
- **Mainnet** : `https://api.binance.com`
- **Testnet** : `https://testnet.binance.vision`

**Documentation** : [Binance API Documentation](https://binance-docs.github.io/apidocs/spot/en/)

---

## 🔵 2. COINBASE ADVANCED

### 📖 Étapes d'obtention

1. **Créer un compte** sur [Coinbase Advanced Trade](https://advanced-trade.coinbase.com)

2. **Accéder aux paramètres API** :
   - Portfolio → API Settings
   - Ou : `https://cloud.coinbase.com/access/api`

3. **Créer une nouvelle API Key** :
   - Name : "BYJY-Trader"
   - Permissions :
     - ✅ wallet:accounts:read
     - ✅ wallet:buys:create  
     - ✅ wallet:sells:create
     - ✅ wallet:trades:read
     - ❌ wallet:withdrawals:create (non recommandé)

4. **Récupérer les clés** :
   - **API Key** : La clé publique générée
   - **API Secret** : Le secret (sauvegarder immédiatement)

### 🔒 Configuration BYJY-Trader

```
Exchange: Coinbase
API Key: votre_coinbase_api_key
Secret Key: votre_coinbase_secret_key
Mode: Sandbox (api-public.sandbox.exchange.coinbase.com)
Permissions: View + Trade
```

### 🧪 URLs de Test
- **Mainnet** : `https://api.coinbase.com`
- **Sandbox** : `https://api-public.sandbox.exchange.coinbase.com`

**Documentation** : [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-auth)

---

## ⚫ 3. KRAKEN

### 📖 Étapes d'obtention

1. **Créer un compte** sur [Kraken](https://www.kraken.com)

2. **Accéder aux paramètres API** :
   - Security → API Management
   - Ou : `https://www.kraken.com/u/security/api`

3. **Générer une nouvelle API Key** :
   - Description : "BYJY-Trader"
   - Key permissions :
     - ✅ Query Funds
     - ✅ Query Open Orders
     - ✅ Query Closed Orders  
     - ✅ Query Trades History
     - ✅ Create & Cancel Orders
     - ❌ Withdraw Funds (non recommandé)
     - ❌ Transfer funds (non recommandé)

4. **Récupérer les clés** :
   - **API Key** : La clé publique
   - **Private Key** : Le secret privé

### 🔒 Configuration BYJY-Trader

```
Exchange: Kraken
API Key: votre_kraken_api_key  
Secret Key: votre_kraken_private_key
Mode: Production uniquement (Kraken n'a pas de sandbox)
Permissions: Query + Trading orders seulement
```

### ⚠️ **Attention** : Kraken n'offre pas de mode sandbox. Testez avec de petits montants.

**Documentation** : [Kraken REST API](https://docs.kraken.com/rest/)

---

## 🟠 4. BYBIT

### 📖 Étapes d'obtention

1. **Créer un compte** sur [Bybit](https://www.bybit.com) ou [Bybit Testnet](https://testnet.bybit.com)

2. **Accéder aux paramètres API** :
   - Account → API Management
   - Ou : `https://www.bybit.com/app/user/api-management`

3. **Créer une nouvelle API** :
   - API key name : "BYJY-Trader"
   - Type : "System generated"
   - Permissions :
     - ✅ Contract (pour futures)
     - ✅ Spot (pour spot trading)
     - ✅ Wallet (lecture soldes)
     - ❌ Options (optionnel)
     - ❌ Derivatives (optionnel)

4. **Récupérer les clés** :
   - **API Key** : La clé publique
   - **Secret** : Le secret (copier immédiatement)

### 🔒 Configuration BYJY-Trader

```
Exchange: Bybit
API Key: votre_bybit_api_key
Secret Key: votre_bybit_secret_key  
Mode: Testnet (api-testnet.bybit.com)
Permissions: Spot + Contract + Wallet
```

### 🧪 URLs de Test
- **Mainnet** : `https://api.bybit.com`
- **Testnet** : `https://api-testnet.bybit.com`

**Documentation** : [Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/guide)

---

## 🛡️ Sécurité & Bonnes Pratiques

### ⚡ **Règles de Sécurité**

1. **Jamais partager les clés** - Ne les communiquez à personne
2. **Permissions minimales** - N'activez que ce qui est nécessaire  
3. **Pas de withdrawal** - N'autorisez jamais les retraits via API
4. **Rotation régulière** - Changez vos clés tous les 3-6 mois
5. **IP whitelisting** - Restreignez l'accès à vos IPs si possible

### 🧪 **Mode Sandbox d'Abord**

```bash
# Ordre de test recommandé :
1. Configuration en mode Sandbox/Testnet
2. Tests des fonctionnalités de base
3. Vérification des balances et ordres  
4. Tests de trading avec petites quantités
5. Migration progressive vers Production
```

### 📊 **Permissions Recommandées**

| Exchange | Lecture | Trading Spot | Trading Futures | Retraits |
|----------|---------|--------------|-----------------|----------|
| Binance  | ✅      | ✅           | ❌              | ❌       |
| Coinbase | ✅      | ✅           | N/A             | ❌       |
| Kraken   | ✅      | ✅           | ❌              | ❌       |
| Bybit    | ✅      | ✅           | ❌              | ❌       |

---

## 🔧 Configuration dans BYJY-Trader

### 1. **Interface Web**
- Aller dans **Configuration** → **Exchanges**
- Sélectionner l'exchange
- Mode **Sandbox** recommandé
- Saisir API Key et Secret
- **Test de connexion** avant sauvegarde

### 2. **Variables d'Environnement** (Alternative)
```bash
# Dans /app/backend/.env
BINANCE_API_KEY=votre_clé
BINANCE_API_SECRET=votre_secret

COINBASE_API_KEY=votre_clé  
COINBASE_API_SECRET=votre_secret
# etc...
```

---

## ❓ Dépannage Commun

### **Erreur : "Invalid API Key"**
- Vérifiez que la clé est copiée complètement
- Assurez-vous que l'API est activée  
- Mode sandbox/production correspond

### **Erreur : "Insufficient permissions"**  
- Activez les permissions Trading + Read
- Redémarrez l'API après changement de permissions

### **Erreur : "IP not whitelisted"**
- Ajoutez votre IP dans les paramètres API
- Ou désactivez la restriction IP temporairement

### **Connexion échoue en Testnet**
- URLs de testnet différentes selon l'exchange
- Certains exchanges nécessitent des comptes testnet séparés

---

## 📞 Support

- **Documentation BYJY-Trader** : README.md
- **Issues GitHub** : Problèmes techniques  
- **Support Exchange** : Contactez directement l'exchange pour les problèmes de compte

---

*⚠️ **Avertissement** : Le trading de cryptomonnaies comporte des risques. N'utilisez que des fonds que vous pouvez vous permettre de perdre. Testez toujours en mode sandbox avant d'utiliser de l'argent réel.*