"""
🧬 Optimization API Routes
FastAPI routes for genetic algorithm and multi-objective optimization
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import uuid
import numpy as np
from datetime import datetime

from core.logger import get_logger
from ai.optimization import GeneticOptimizer, ParetoOptimizer

logger = get_logger(__name__)

router = APIRouter(prefix="/api/optimization")

# Global optimization instances (in production, use proper state management)
genetic_optimizers: Dict[str, GeneticOptimizer] = {}
pareto_optimizers: Dict[str, ParetoOptimizer] = {}
optimization_jobs: Dict[str, Dict[str, Any]] = {}


# Pydantic models for API
class OptimizationConfig(BaseModel):
    population_size: int = Field(default=100, ge=10, le=500)
    num_generations: int = Field(default=200, ge=10, le=1000)
    crossover_prob: float = Field(default=0.8, ge=0.0, le=1.0)
    mutation_prob: float = Field(default=0.1, ge=0.0, le=1.0)
    elitism_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    early_stopping_patience: int = Field(default=20, ge=5, le=100)


class ParameterSpace(BaseModel):
    parameters: Dict[str, Dict[str, Any]]


class ObjectiveConfig(BaseModel):
    name: str
    maximize: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    description: Optional[str] = None
    target_range: Optional[List[float]] = None


class GeneticOptimizationRequest(BaseModel):
    parameter_space: ParameterSpace
    optimization_config: OptimizationConfig = OptimizationConfig()
    fitness_weights: Optional[Dict[str, float]] = None
    target_model: str = Field(..., description="Target model to optimize (lstm, transformer, ensemble, trading)")
    random_seed: Optional[int] = None


class ParetoOptimizationRequest(BaseModel):
    parameter_space: ParameterSpace
    objectives: List[ObjectiveConfig]
    optimization_config: OptimizationConfig = OptimizationConfig()
    target_model: str = Field(..., description="Target model to optimize")
    random_seed: Optional[int] = None


class OptimizationStatus(BaseModel):
    job_id: str
    status: str
    progress_percent: float = 0.0
    current_generation: int = 0
    total_generations: int = 0
    best_fitness: Optional[float] = None
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None


# Genetic Algorithm Routes
@router.post("/genetic/start", response_model=Dict[str, str])
async def start_genetic_optimization(
    request: GeneticOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Start genetic algorithm optimization."""
    job_id = str(uuid.uuid4())
    
    try:
        # Create genetic optimizer
        optimizer = GeneticOptimizer(
            population_size=request.optimization_config.population_size,
            num_generations=request.optimization_config.num_generations,
            crossover_prob=request.optimization_config.crossover_prob,
            mutation_prob=request.optimization_config.mutation_prob,
            elitism_ratio=request.optimization_config.elitism_ratio,
            early_stopping_patience=request.optimization_config.early_stopping_patience,
            random_seed=request.random_seed
        )
        
        # Store optimizer
        genetic_optimizers[job_id] = optimizer
        
        # Initialize job tracking
        optimization_jobs[job_id] = {
            'type': 'genetic',
            'status': 'running',
            'started_at': datetime.now(),
            'target_model': request.target_model,
            'progress': 0.0,
            'results': None,
            'error': None
        }
        
        # Start optimization in background
        background_tasks.add_task(
            _run_genetic_optimization,
            job_id,
            optimizer,
            request.parameter_space.parameters,
            request.target_model,
            request.fitness_weights
        )
        
        logger.info(f"Started genetic optimization job {job_id} for {request.target_model}")
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Error starting genetic optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/genetic/status/{job_id}", response_model=OptimizationStatus)
async def get_genetic_optimization_status(job_id: str):
    """Get genetic optimization status."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    job_info = optimization_jobs[job_id]
    
    # Get detailed status from optimizer if running
    if job_id in genetic_optimizers and job_info['status'] == 'running':
        optimizer = genetic_optimizers[job_id]
        optimizer_status = optimizer.get_optimization_status()
        
        return OptimizationStatus(
            job_id=job_id,
            status=optimizer_status['status'],
            progress_percent=optimizer_status.get('progress_percent', 0.0),
            current_generation=optimizer_status.get('current_generation', 0),
            total_generations=optimizer_status.get('total_generations', 0),
            best_fitness=optimizer_status.get('best_overall_fitness'),
            started_at=job_info['started_at']
        )
    
    return OptimizationStatus(
        job_id=job_id,
        status=job_info['status'],
        progress_percent=job_info.get('progress', 0.0),
        started_at=job_info['started_at'],
        error_message=job_info.get('error')
    )


@router.get("/genetic/results/{job_id}")
async def get_genetic_optimization_results(job_id: str):
    """Get genetic optimization results."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    job_info = optimization_jobs[job_id]
    
    if job_info['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Optimization not completed. Status: {job_info['status']}")
    
    return job_info.get('results', {})


@router.post("/genetic/stop/{job_id}")
async def stop_genetic_optimization(job_id: str):
    """Stop genetic optimization."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    job_info = optimization_jobs[job_id]
    
    if job_info['status'] != 'running':
        return {"message": f"Optimization already {job_info['status']}"}
    
    # Stop optimizer
    if job_id in genetic_optimizers:
        genetic_optimizers[job_id].stop_optimization()
    
    job_info['status'] = 'stopped'
    logger.info(f"Stopped genetic optimization job {job_id}")
    
    return {"message": "Optimization stopped"}


@router.get("/genetic/history")
async def get_genetic_optimization_history():
    """Get history of genetic optimizations."""
    history = []
    
    for job_id, job_info in optimization_jobs.items():
        if job_info['type'] == 'genetic':
            history.append({
                'job_id': job_id,
                'status': job_info['status'],
                'target_model': job_info['target_model'],
                'started_at': job_info['started_at'],
                'progress': job_info.get('progress', 0.0)
            })
    
    return {"history": history}


# Pareto Optimization Routes
@router.post("/pareto/optimize", response_model=Dict[str, str])
async def start_pareto_optimization(
    request: ParetoOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Start Pareto multi-objective optimization."""
    job_id = str(uuid.uuid4())
    
    try:
        # Create Pareto optimizer
        optimizer = ParetoOptimizer(
            population_size=request.optimization_config.population_size,
            num_generations=request.optimization_config.num_generations,
            crossover_prob=request.optimization_config.crossover_prob,
            mutation_prob=request.optimization_config.mutation_prob,
            random_seed=request.random_seed
        )
        
        # Store optimizer
        pareto_optimizers[job_id] = optimizer
        
        # Initialize job tracking
        optimization_jobs[job_id] = {
            'type': 'pareto',
            'status': 'running',
            'started_at': datetime.now(),
            'target_model': request.target_model,
            'objectives': [obj.dict() for obj in request.objectives],
            'progress': 0.0,
            'results': None,
            'error': None
        }
        
        # Start optimization in background
        background_tasks.add_task(
            _run_pareto_optimization,
            job_id,
            optimizer,
            request.parameter_space.parameters,
            request.objectives,
            request.target_model
        )
        
        logger.info(f"Started Pareto optimization job {job_id} for {request.target_model}")
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Error starting Pareto optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pareto/front/{job_id}")
async def get_pareto_front(job_id: str):
    """Get Pareto front results."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    job_info = optimization_jobs[job_id]
    
    if job_info['type'] != 'pareto':
        raise HTTPException(status_code=400, detail="Job is not a Pareto optimization")
    
    if job_info['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Optimization not completed. Status: {job_info['status']}")
    
    results = job_info.get('results', {})
    return {
        'pareto_front': results.get('pareto_front', []),
        'compromise_solutions': results.get('compromise_solutions', []),
        'pareto_analysis': results.get('pareto_analysis', {})
    }


@router.get("/pareto/solutions/{job_id}")
async def get_pareto_solutions(job_id: str):
    """Get all Pareto solutions and analysis."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    job_info = optimization_jobs[job_id]
    
    if job_info['type'] != 'pareto':
        raise HTTPException(status_code=400, detail="Job is not a Pareto optimization")
    
    if job_info['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Optimization not completed. Status: {job_info['status']}")
    
    return job_info.get('results', {})


# Utility Routes
@router.get("/status")
async def get_optimization_service_status():
    """Get optimization service status."""
    running_jobs = len([job for job in optimization_jobs.values() if job['status'] == 'running'])
    completed_jobs = len([job for job in optimization_jobs.values() if job['status'] == 'completed'])
    failed_jobs = len([job for job in optimization_jobs.values() if job['status'] == 'failed'])
    
    return {
        'service': 'Optimization Service',
        'status': 'healthy',
        'active_optimizers': {
            'genetic': len(genetic_optimizers),
            'pareto': len(pareto_optimizers)
        },
        'job_statistics': {
            'running': running_jobs,
            'completed': completed_jobs,
            'failed': failed_jobs,
            'total': len(optimization_jobs)
        }
    }


@router.delete("/jobs/{job_id}")
async def delete_optimization_job(job_id: str):
    """Delete optimization job and cleanup resources."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    # Stop if running
    if optimization_jobs[job_id]['status'] == 'running':
        if job_id in genetic_optimizers:
            genetic_optimizers[job_id].stop_optimization()
            del genetic_optimizers[job_id]
        if job_id in pareto_optimizers:
            pareto_optimizers[job_id].stop_optimization()
            del pareto_optimizers[job_id]
    
    # Remove job
    del optimization_jobs[job_id]
    
    logger.info(f"Deleted optimization job {job_id}")
    return {"message": "Job deleted successfully"}


# Background task functions
async def _run_genetic_optimization(
    job_id: str,
    optimizer: GeneticOptimizer,
    parameter_space: Dict[str, Any],
    target_model: str,
    fitness_weights: Optional[Dict[str, float]] = None
):
    """Run genetic optimization in background."""
    try:
        # Mock fitness function for now (will be replaced with actual model evaluation)
        async def mock_fitness_function(parameters: Dict[str, Any]) -> Dict[str, float]:
            await asyncio.sleep(0.1)  # Simulate evaluation time
            
            # Mock metrics based on target model
            if target_model == 'lstm':
                return {
                    'sharpe_ratio': np.random.normal(1.5, 0.5),
                    'max_drawdown': np.random.uniform(0.05, 0.3),
                    'training_time': np.random.uniform(10, 300),
                    'accuracy': np.random.uniform(0.6, 0.9)
                }
            elif target_model == 'ensemble':
                return {
                    'sharpe_ratio': np.random.normal(1.8, 0.4),
                    'max_drawdown': np.random.uniform(0.03, 0.25),
                    'training_time': np.random.uniform(30, 600),
                    'accuracy': np.random.uniform(0.65, 0.95)
                }
            else:
                return {
                    'profit': np.random.uniform(0, 1),
                    'risk': np.random.uniform(0, 1),
                    'speed': np.random.uniform(0, 1)
                }
        
        # Run optimization
        results = await optimizer.optimize(
            parameter_space=parameter_space,
            fitness_function=mock_fitness_function,
            fitness_kwargs={'target_model': target_model}
        )
        
        # Store results
        optimization_jobs[job_id]['status'] = 'completed'
        optimization_jobs[job_id]['results'] = results
        optimization_jobs[job_id]['progress'] = 100.0
        
        logger.info(f"Genetic optimization job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Genetic optimization job {job_id} failed: {e}")
        optimization_jobs[job_id]['status'] = 'failed'
        optimization_jobs[job_id]['error'] = str(e)
    
    finally:
        # Cleanup optimizer
        if job_id in genetic_optimizers:
            del genetic_optimizers[job_id]


async def _run_pareto_optimization(
    job_id: str,
    optimizer: ParetoOptimizer,
    parameter_space: Dict[str, Any],
    objectives: List[ObjectiveConfig],
    target_model: str
):
    """Run Pareto optimization in background."""
    try:
        # Mock evaluation function
        async def mock_evaluation_function(parameters: Dict[str, Any]) -> Dict[str, float]:
            await asyncio.sleep(0.1)  # Simulate evaluation time
            
            # Generate mock metrics
            return {
                'sharpe_ratio': np.random.normal(1.5, 0.5),
                'max_drawdown': np.random.uniform(0.05, 0.3),
                'annual_return': np.random.normal(0.15, 0.1),
                'training_time': np.random.uniform(10, 300),
                'consistency_score': np.random.uniform(0.6, 0.95)
            }
        
        # Convert objectives to dict format
        objective_configs = [obj.dict() for obj in objectives]
        
        # Run optimization
        results = await optimizer.optimize(
            parameter_space=parameter_space,
            objective_configs=objective_configs,
            evaluation_function=mock_evaluation_function
        )
        
        # Store results
        optimization_jobs[job_id]['status'] = 'completed'
        optimization_jobs[job_id]['results'] = results
        optimization_jobs[job_id]['progress'] = 100.0
        
        logger.info(f"Pareto optimization job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Pareto optimization job {job_id} failed: {e}")
        optimization_jobs[job_id]['status'] = 'failed'
        optimization_jobs[job_id]['error'] = str(e)
    
    finally:
        # Cleanup optimizer
        if job_id in pareto_optimizers:
            del pareto_optimizers[job_id]