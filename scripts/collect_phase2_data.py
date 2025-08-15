#!/usr/bin/env python3
"""
Script de collecte de données Phase 2 - Intelligence Artificielle
Collecte les données pour les symboles prioritaires selon le roadmap BYJY-Trader
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import time

# Configuration selon le roadmap Phase 2
SYMBOLS_PRIORITY = [
    "AAPL",   # Apple - Stock
    "MSFT",   # Microsoft - Stock  
    "TSLA",   # Tesla - Stock
    "BTC-USD", # Bitcoin - Crypto
    "ETH-USD"  # Ethereum - Crypto
]

API_BASE_URL = "http://localhost:8001"
START_DATE = "2024-01-01T00:00:00Z"  # 12+ mois de données

async def collect_symbol_data(session, symbol):
    """Collecte les données pour un symbole spécifique"""
    print(f"🔄 Démarrage collecte {symbol}...")
    
    collection_request = {
        "symbol": symbol,
        "sources": ["yahoo"],
        "intervals": ["1d"],
        "start_date": START_DATE,
        "generate_features": True
    }
    
    try:
        async with session.post(
            f"{API_BASE_URL}/api/data/collect",
            json=collection_request,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                result = await response.json()
                task_id = result.get("task_id")
                print(f"✅ {symbol}: Collecte démarrée - Task ID: {task_id}")
                return task_id
            else:
                error_text = await response.text()
                print(f"❌ {symbol}: Erreur {response.status} - {error_text}")
                return None
                
    except Exception as e:
        print(f"❌ {symbol}: Exception - {str(e)}")
        return None

async def check_task_status(session, task_id, symbol):
    """Vérifie le statut d'une tâche de collecte"""
    try:
        async with session.get(
            f"{API_BASE_URL}/api/data/tasks/{task_id}",
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status == 200:
                result = await response.json()
                status = result.get("status", "unknown")
                print(f"📊 {symbol}: Status = {status}")
                return result
            else:
                print(f"⚠️ {symbol}: Erreur status {response.status}")
                return None
    except Exception as e:
        print(f"⚠️ {symbol}: Erreur check status - {str(e)}")
        return None

async def main():
    """Fonction principale de collecte"""
    print("🚀 BYJY-Trader Phase 2 - Collecte Données IA")
    print("=" * 50)
    
    # Créer session HTTP
    async with aiohttp.ClientSession() as session:
        
        # Étape 1: Démarrer toutes les collectes
        print("📡 Étape 1: Démarrage collectes multiples...")
        tasks = {}
        
        for symbol in SYMBOLS_PRIORITY:
            task_id = await collect_symbol_data(session, symbol)
            if task_id:
                tasks[symbol] = task_id
            await asyncio.sleep(2)  # Rate limiting
        
        print(f"\n✅ {len(tasks)} collectes démarrées sur {len(SYMBOLS_PRIORITY)} symboles")
        
        # Étape 2: Monitoring des tâches
        print("\n📊 Étape 2: Monitoring des tâches...")
        completed = set()
        max_checks = 30  # Max 5 minutes de monitoring
        
        for check_round in range(max_checks):
            print(f"\n🔍 Vérification {check_round + 1}/{max_checks}")
            
            for symbol, task_id in tasks.items():
                if symbol not in completed:
                    status_result = await check_task_status(session, task_id, symbol)
                    if status_result and status_result.get("status") == "completed":
                        completed.add(symbol)
                        data_points = status_result.get("result", {}).get("data_points", 0)
                        print(f"🎉 {symbol}: TERMINÉ - {data_points} points collectés")
            
            # Si toutes les tâches sont terminées
            if len(completed) == len(tasks):
                break
                
            await asyncio.sleep(10)  # Attendre 10s entre les vérifications
        
        # Résumé final
        print("\n" + "=" * 50)
        print("📊 RÉSUMÉ COLLECTE PHASE 2")
        print("=" * 50)
        print(f"✅ Collectes terminées: {len(completed)}/{len(tasks)}")
        print(f"📈 Symboles collectés: {', '.join(sorted(completed))}")
        
        if len(completed) < len(tasks):
            pending = set(tasks.keys()) - completed
            print(f"⏳ En cours: {', '.join(sorted(pending))}")
        
        print("\n🎯 Prochaine étape: Entraînement modèles LSTM")

if __name__ == "__main__":
    asyncio.run(main())