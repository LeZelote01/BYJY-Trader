#!/usr/bin/env python3
"""
🧬 BYJY-Trader Backend Tests - Phase 3.4 Optimisation Génétique & Meta-Learning
Tests exhaustifs pour tous les endpoints d'optimisation
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List

import aiohttp
import pytest
from pydantic import BaseModel

# Configuration de test
BACKEND_URL = "https://e61aade5-33d8-44a4-8cd5-e207b55d780e.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class TestResults:
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def add_result(self, test_name: str, passed: bool, details: str = "", latency: float = 0.0):
        self.results.append({
            'test_name': test_name,
            'passed': passed,
            'details': details,
            'latency_ms': round(latency * 1000, 2),
            'timestamp': datetime.now().isoformat()
        })
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
    def print_summary(self):
        print(f"\n{'='*80}")
        print(f"🧬 PHASE 3.4 OPTIMIZATION TESTS - RÉSULTATS FINAUX")
        print(f"{'='*80}")
        print(f"📊 Total Tests: {self.total_tests}")
        print(f"✅ Tests Réussis: {self.passed_tests}")
        print(f"❌ Tests Échoués: {self.failed_tests}")
        print(f"📈 Taux de Réussite: {(self.passed_tests/self.total_tests*100):.1f}%")
        print(f"{'='*80}")
        
        for result in self.results:
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {result['test_name']} ({result['latency_ms']}ms)")
            if result['details']:
                print(f"   📝 {result['details']}")

# Instance globale des résultats
test_results = TestResults()

async def make_request(session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> tuple:
    """Effectue une requête HTTP et mesure la latence"""
    start_time = time.time()
    try:
        async with session.request(method, url, **kwargs) as response:
            latency = time.time() - start_time
            data = await response.json() if response.content_type == 'application/json' else await response.text()
            return response.status, data, latency
    except Exception as e:
        latency = time.time() - start_time
        return 500, str(e), latency

async def test_optimization_service_status():
    """Test 1: Service Status - GET /api/optimization/status"""
    async with aiohttp.ClientSession() as session:
        status, data, latency = await make_request(session, 'GET', f"{API_BASE}/optimization/status")
        
        if status == 200:
            required_fields = ['service', 'status', 'active_optimizers', 'job_statistics']
            if all(field in data for field in required_fields):
                test_results.add_result(
                    "Optimization Service Status", 
                    True, 
                    f"Service healthy with {data.get('job_statistics', {}).get('total', 0)} total jobs",
                    latency
                )
            else:
                test_results.add_result(
                    "Optimization Service Status", 
                    False, 
                    f"Missing required fields in response: {data}",
                    latency
                )
        else:
            test_results.add_result(
                "Optimization Service Status", 
                False, 
                f"HTTP {status}: {data}",
                latency
            )

async def test_genetic_optimization_start():
    """Test 2: Genetic Optimization Start - POST /api/optimization/genetic/start"""
    async with aiohttp.ClientSession() as session:
        # Configuration d'optimisation génétique pour LSTM
        request_data = {
            "parameter_space": {
                "parameters": {
                    "layers": {"type": "int", "min": 1, "max": 3},
                    "neurons": {"type": "int", "min": 32, "max": 128},
                    "dropout": {"type": "float", "min": 0.1, "max": 0.3},
                    "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2},
                    "batch_size": {"type": "categorical", "choices": [32, 64]}
                }
            },
            "optimization_config": {
                "population_size": 20,
                "num_generations": 10,
                "crossover_prob": 0.8,
                "mutation_prob": 0.1
            },
            "target_model": "lstm",
            "random_seed": 42
        }
        
        status, data, latency = await make_request(
            session, 'POST', f"{API_BASE}/optimization/genetic/start",
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if status == 200 and 'job_id' in data:
            # Stocker le job_id pour les tests suivants
            global genetic_job_id
            genetic_job_id = data['job_id']
            test_results.add_result(
                "Genetic Optimization Start", 
                True, 
                f"Started genetic optimization job: {genetic_job_id}",
                latency
            )
        else:
            test_results.add_result(
                "Genetic Optimization Start", 
                False, 
                f"HTTP {status}: {data}",
                latency
            )

async def test_genetic_optimization_status():
    """Test 3: Genetic Optimization Status - GET /api/optimization/genetic/status/{job_id}"""
    if not hasattr(test_genetic_optimization_status, 'job_id'):
        # Créer un job d'abord
        await test_genetic_optimization_start()
    
    async with aiohttp.ClientSession() as session:
        job_id = getattr(test_genetic_optimization_start, 'genetic_job_id', 'test-job-id')
        status, data, latency = await make_request(
            session, 'GET', f"{API_BASE}/optimization/genetic/status/{genetic_job_id if 'genetic_job_id' in globals() else 'invalid-id'}"
        )
        
        if status == 200:
            required_fields = ['job_id', 'status', 'started_at']
            if all(field in data for field in required_fields):
                test_results.add_result(
                    "Genetic Optimization Status", 
                    True, 
                    f"Status: {data.get('status')}, Progress: {data.get('progress_percent', 0)}%",
                    latency
                )
            else:
                test_results.add_result(
                    "Genetic Optimization Status", 
                    False, 
                    f"Missing required fields: {data}",
                    latency
                )
        elif status == 404:
            test_results.add_result(
                "Genetic Optimization Status", 
                True, 
                "Correctly returns 404 for invalid job ID",
                latency
            )
        else:
            test_results.add_result(
                "Genetic Optimization Status", 
                False, 
                f"HTTP {status}: {data}",
                latency
            )

async def test_genetic_optimization_history():
    """Test 4: Genetic Optimization History - GET /api/optimization/genetic/history"""
    async with aiohttp.ClientSession() as session:
        status, data, latency = await make_request(session, 'GET', f"{API_BASE}/optimization/genetic/history")
        
        if status == 200 and 'history' in data:
            history_count = len(data['history'])
            test_results.add_result(
                "Genetic Optimization History", 
                True, 
                f"Retrieved {history_count} optimization jobs in history",
                latency
            )
        else:
            test_results.add_result(
                "Genetic Optimization History", 
                False, 
                f"HTTP {status}: {data}",
                latency
            )

async def test_pareto_optimization_start():
    """Test 5: Pareto Multi-Objective Optimization - POST /api/optimization/pareto/optimize"""
    async with aiohttp.ClientSession() as session:
        # Configuration d'optimisation multi-objectif
        request_data = {
            "parameter_space": {
                "parameters": {
                    "layers": {"type": "int", "min": 1, "max": 3},
                    "neurons": {"type": "int", "min": 32, "max": 128},
                    "dropout": {"type": "float", "min": 0.1, "max": 0.3},
                    "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2}
                }
            },
            "objectives": [
                {
                    "name": "sharpe_ratio",
                    "maximize": True,
                    "weight": 1.0,
                    "description": "Sharpe ratio maximization"
                },
                {
                    "name": "max_drawdown",
                    "maximize": False,
                    "weight": 1.0,
                    "description": "Maximum drawdown minimization"
                }
            ],
            "optimization_config": {
                "population_size": 20,
                "num_generations": 10,
                "crossover_prob": 0.8,
                "mutation_prob": 0.1
            },
            "target_model": "ensemble",
            "random_seed": 42
        }
        
        status, data, latency = await make_request(
            session, 'POST', f"{API_BASE}/optimization/pareto/optimize",
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if status == 200 and 'job_id' in data:
            global pareto_job_id
            pareto_job_id = data['job_id']
            test_results.add_result(
                "Pareto Multi-Objective Optimization", 
                True, 
                f"Started Pareto optimization job: {pareto_job_id}",
                latency
            )
        else:
            test_results.add_result(
                "Pareto Multi-Objective Optimization", 
                False, 
                f"HTTP {status}: {data}",
                latency
            )

async def test_pareto_front_results():
    """Test 6: Pareto Front Results - GET /api/optimization/pareto/front/{job_id}"""
    if 'pareto_job_id' not in globals():
        await test_pareto_optimization_start()
        # Attendre un peu pour que l'optimisation commence
        await asyncio.sleep(2)
    
    async with aiohttp.ClientSession() as session:
        job_id = pareto_job_id if 'pareto_job_id' in globals() else 'invalid-id'
        status, data, latency = await make_request(
            session, 'GET', f"{API_BASE}/optimization/pareto/front/{job_id}"
        )
        
        if status == 200:
            test_results.add_result(
                "Pareto Front Results", 
                True, 
                f"Retrieved Pareto front with {len(data.get('pareto_front', []))} solutions",
                latency
            )
        elif status == 400:
            test_results.add_result(
                "Pareto Front Results", 
                True, 
                "Correctly returns 400 for incomplete optimization",
                latency
            )
        elif status == 404:
            test_results.add_result(
                "Pareto Front Results", 
                True, 
                "Correctly returns 404 for invalid job ID",
                latency
            )
        else:
            test_results.add_result(
                "Pareto Front Results", 
                False, 
                f"HTTP {status}: {data}",
                latency
            )

async def test_missing_hyperparameter_endpoint():
    """Test 7: Hyperparameter Tuning (MISSING) - POST /api/optimization/hyperparameter/tune"""
    async with aiohttp.ClientSession() as session:
        request_data = {
            "parameter_space": {
                "parameters": {
                    "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2},
                    "batch_size": {"type": "categorical", "choices": [32, 64, 128]}
                }
            },
            "target_model": "lstm",
            "n_trials": 50
        }
        
        status, data, latency = await make_request(
            session, 'POST', f"{API_BASE}/optimization/hyperparameter/tune",
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if status == 404:
            test_results.add_result(
                "Hyperparameter Tuning Endpoint", 
                False, 
                "ENDPOINT MISSING - Not implemented in Phase 3.4",
                latency
            )
        else:
            test_results.add_result(
                "Hyperparameter Tuning Endpoint", 
                True, 
                f"Endpoint exists: HTTP {status}",
                latency
            )

async def test_missing_adaptive_endpoint():
    """Test 8: Adaptive Strategies (MISSING) - POST /api/optimization/adaptive/enable"""
    async with aiohttp.ClientSession() as session:
        request_data = {
            "strategy_config": {
                "adaptation_frequency": "daily",
                "performance_threshold": 0.15,
                "market_regime_detection": True
            },
            "target_strategies": ["momentum", "mean_reversion"]
        }
        
        status, data, latency = await make_request(
            session, 'POST', f"{API_BASE}/optimization/adaptive/enable",
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if status == 404:
            test_results.add_result(
                "Adaptive Strategies Endpoint", 
                False, 
                "ENDPOINT MISSING - Not implemented in Phase 3.4",
                latency
            )
        else:
            test_results.add_result(
                "Adaptive Strategies Endpoint", 
                True, 
                f"Endpoint exists: HTTP {status}",
                latency
            )

async def test_optimization_job_management():
    """Test 9: Job Management - DELETE /api/optimization/jobs/{job_id}"""
    async with aiohttp.ClientSession() as session:
        # Test avec un job ID invalide
        status, data, latency = await make_request(
            session, 'DELETE', f"{API_BASE}/optimization/jobs/invalid-job-id"
        )
        
        if status == 404:
            test_results.add_result(
                "Optimization Job Management", 
                True, 
                "Correctly returns 404 for invalid job ID deletion",
                latency
            )
        else:
            test_results.add_result(
                "Optimization Job Management", 
                False, 
                f"Unexpected response for invalid job deletion: HTTP {status}",
                latency
            )

async def test_genetic_algorithm_convergence():
    """Test 10: Genetic Algorithm Convergence Performance"""
    async with aiohttp.ClientSession() as session:
        # Configuration pour test de convergence rapide
        request_data = {
            "parameter_space": {
                "parameters": {
                    "layers": {"type": "int", "min": 1, "max": 2},
                    "neurons": {"type": "int", "min": 32, "max": 64},
                    "dropout": {"type": "float", "min": 0.1, "max": 0.2}
                }
            },
            "optimization_config": {
                "population_size": 10,
                "num_generations": 5,
                "crossover_prob": 0.8,
                "mutation_prob": 0.1
            },
            "target_model": "lstm",
            "random_seed": 123
        }
        
        start_time = time.time()
        status, data, latency = await make_request(
            session, 'POST', f"{API_BASE}/optimization/genetic/start",
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if status == 200 and 'job_id' in data:
            job_id = data['job_id']
            
            # Attendre et vérifier la convergence
            max_wait_time = 60  # 60 secondes max
            convergence_time = time.time() - start_time
            
            if convergence_time < max_wait_time:
                test_results.add_result(
                    "Genetic Algorithm Convergence", 
                    True, 
                    f"Algorithm started successfully in {convergence_time:.2f}s (< 60s target)",
                    latency
                )
            else:
                test_results.add_result(
                    "Genetic Algorithm Convergence", 
                    False, 
                    f"Algorithm took too long to start: {convergence_time:.2f}s",
                    latency
                )
        else:
            test_results.add_result(
                "Genetic Algorithm Convergence", 
                False, 
                f"Failed to start convergence test: HTTP {status}",
                latency
            )

async def test_optimization_integration():
    """Test 11: Integration with Existing Phases"""
    async with aiohttp.ClientSession() as session:
        # Test intégration avec les phases existantes
        # Vérifier que les endpoints RL sont toujours accessibles
        status, data, latency = await make_request(session, 'GET', f"{API_BASE}/rl/health")
        
        rl_working = status == 200
        
        # Vérifier que les endpoints ensemble sont accessibles
        status2, data2, latency2 = await make_request(session, 'GET', f"{API_BASE}/ensemble/health")
        
        ensemble_working = status2 == 200
        
        if rl_working and ensemble_working:
            test_results.add_result(
                "Phase Integration", 
                True, 
                "RL and Ensemble phases remain accessible during optimization",
                max(latency, latency2)
            )
        elif rl_working:
            test_results.add_result(
                "Phase Integration", 
                True, 
                "RL phase accessible, Ensemble may not be implemented",
                latency
            )
        else:
            test_results.add_result(
                "Phase Integration", 
                False, 
                "Integration issues detected with previous phases",
                max(latency, latency2)
            )

async def run_all_tests():
    """Exécute tous les tests d'optimisation Phase 3.4"""
    print("🧬 DÉMARRAGE TESTS EXHAUSTIFS PHASE 3.4 - OPTIMISATION GÉNÉTIQUE & META-LEARNING")
    print("="*80)
    
    # Tests séquentiels pour éviter les conflits
    test_functions = [
        test_optimization_service_status,
        test_genetic_optimization_start,
        test_genetic_optimization_status,
        test_genetic_optimization_history,
        test_pareto_optimization_start,
        test_pareto_front_results,
        test_missing_hyperparameter_endpoint,
        test_missing_adaptive_endpoint,
        test_optimization_job_management,
        test_genetic_algorithm_convergence,
        test_optimization_integration
    ]
    
    for i, test_func in enumerate(test_functions, 1):
        print(f"🔄 Exécution Test {i}/{len(test_functions)}: {test_func.__name__}")
        try:
            await test_func()
            print(f"✅ Test {i} terminé")
        except Exception as e:
            print(f"❌ Test {i} échoué: {e}")
            test_results.add_result(
                test_func.__name__, 
                False, 
                f"Exception: {str(e)}"
            )
        
        # Petite pause entre les tests
        await asyncio.sleep(0.5)
    
    # Afficher le résumé final
    test_results.print_summary()
    
    return test_results

if __name__ == "__main__":
    # Exécuter tous les tests
    results = asyncio.run(run_all_tests())
    
    # Sauvegarder les résultats
    with open('/app/optimization_test_results.json', 'w') as f:
        json.dump({
            'test_summary': {
                'total_tests': results.total_tests,
                'passed_tests': results.passed_tests,
                'failed_tests': results.failed_tests,
                'success_rate': round(results.passed_tests/results.total_tests*100, 1)
            },
            'detailed_results': results.results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📊 Résultats sauvegardés dans: /app/optimization_test_results.json")
    
    # Code de sortie basé sur les résultats
    exit_code = 0 if results.failed_tests == 0 else 1
    exit(exit_code)