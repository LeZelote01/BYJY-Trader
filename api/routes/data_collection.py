# Data Collection API Routes for BYJY-Trader
# Phase 2.1 - Historical Data Collection System

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

from data.collectors.binance_collector import BinanceCollector
from data.collectors.yahoo_collector import YahooCollector
from data.collectors.coingecko_collector import CoinGeckoCollector
from data.processors.feature_engine import FeatureEngine
from data.storage.data_manager import DataManager
from core.logger import get_logger

logger = get_logger(__name__)

# Initialize router
router = APIRouter(tags=["Data Collection"])

# Initialize components (will be done in lifespan)
collectors = {}
feature_engine = None
data_manager = None

# Pydantic models
class CollectionRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)", min_length=1)
    sources: List[str] = Field(default=["yahoo"], description="Data sources (yahoo, binance, coingecko)")
    intervals: List[str] = Field(default=["1d"], description="Time intervals (1m, 5m, 15m, 1h, 1d)")
    start_date: datetime = Field(..., description="Start date for data collection")
    end_date: Optional[datetime] = Field(None, description="End date (optional)")
    generate_features: bool = Field(default=True, description="Generate technical features")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "sources": ["yahoo"],
                "intervals": ["1d"],
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-12-31T23:59:59Z",
                "generate_features": True
            }
        }

class CollectionStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

class DataQuery(BaseModel):
    symbol: str
    source: Optional[str] = None
    interval: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_features: bool = False
    limit: Optional[int] = None

# Background task tracking
active_tasks = {}

async def initialize_data_components():
    """Initialize data collection components."""
    global collectors, feature_engine, data_manager
    
    try:
        # Initialize collectors
        collectors = {
            'binance': BinanceCollector(rate_limit=0.1),  # 10 requests/second
            'yahoo': YahooCollector(rate_limit=2.0),      # 0.5 requests/second
            'coingecko': CoinGeckoCollector(rate_limit=1.5) # 0.67 requests/second
        }
        
        # Connect all collectors
        for name, collector in collectors.items():
            connected = await collector.connect()
            if connected:
                logger.info(f"Connected to {name} collector")
            else:
                logger.warning(f"Failed to connect to {name} collector")
        
        # Initialize feature engine and data manager
        feature_engine = FeatureEngine()
        data_manager = DataManager()
        await data_manager.initialize_tables()
        
        logger.info("Data collection components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing data components: {e}")
        raise

@router.get("/collectors/status")
async def get_collectors_status():
    """Get status of all data collectors."""
    try:
        status = {}
        for name, collector in collectors.items():
            status[name] = collector.get_status()
        
        return JSONResponse(content={
            "status": "success",
            "collectors": status,
            "total_collectors": len(collectors),
            "connected_collectors": sum(1 for c in collectors.values() if c.is_connected)
        })
        
    except Exception as e:
        logger.error(f"Error getting collectors status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collectors/{source}/symbols")
async def get_available_symbols(source: str):
    """Get available symbols for a specific data source."""
    try:
        if source not in collectors:
            raise HTTPException(status_code=400, detail=f"Unknown data source: {source}")
        
        collector = collectors[source]
        if not collector.is_connected:
            raise HTTPException(status_code=503, detail=f"{source} collector not connected")
        
        symbols = await collector.get_available_symbols()
        
        return JSONResponse(content={
            "status": "success",
            "source": source,
            "symbols": symbols,
            "total_symbols": len(symbols)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting symbols for {source}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collect")
async def start_data_collection(
    request: CollectionRequest,
    background_tasks: BackgroundTasks
):
    """Start historical data collection task."""
    try:
        # Validate request parameters
        if not request.symbol or len(request.symbol.strip()) == 0:
            raise HTTPException(
                status_code=422, 
                detail="Symbol cannot be empty"
            )
        
        if not request.sources:
            raise HTTPException(
                status_code=422, 
                detail="At least one source must be specified"
            )
        
        # Validate sources
        valid_sources = ["yahoo", "binance", "coingecko"]
        invalid_sources = [s for s in request.sources if s not in valid_sources]
        if invalid_sources:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid sources: {invalid_sources}. Valid sources: {valid_sources}"
            )
        
        # Generate unique task ID
        task_id = f"collect_{request.symbol}_{int(datetime.now().timestamp())}"
        
        # Initialize task status
        active_tasks[task_id] = CollectionStatus(
            task_id=task_id,
            status="started",
            progress=0.0,
            message="Initializing data collection",
            started_at=datetime.now()
        )
        
        # Start background collection task
        background_tasks.add_task(
            collect_data_background,
            task_id,
            request
        )
        
        return JSONResponse(content={
            "status": "success",
            "task_id": task_id,
            "message": "Data collection started",
            "estimated_duration": "2-10 minutes depending on data range"
        })
        
    except HTTPException:
        # Re-raise HTTPException with proper status codes
        raise
    except Exception as e:
        logger.error(f"Error starting data collection: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

async def collect_data_background(task_id: str, request: CollectionRequest):
    """Background task for data collection."""
    try:
        task = active_tasks[task_id]
        total_operations = len(request.sources) * len(request.intervals)
        completed_operations = 0
        
        collection_results = {
            'collected_data': {},
            'generated_features': {},
            'total_records': 0,
            'errors': []
        }
        
        # Collect data from each source and interval
        for source in request.sources:
            if source not in collectors:
                error_msg = f"Unknown data source: {source}"
                collection_results['errors'].append(error_msg)
                continue
            
            collector = collectors[source]
            if not collector.is_connected:
                error_msg = f"{source} collector not connected"
                collection_results['errors'].append(error_msg)
                continue
            
            for interval in request.intervals:
                try:
                    # Update task status
                    task.status = "collecting"
                    task.message = f"Collecting {request.symbol} from {source} ({interval})"
                    task.progress = (completed_operations / total_operations) * 0.7  # 70% for data collection
                    
                    logger.info(f"Collecting data: {request.symbol} {source} {interval}")
                    
                    # Collect historical data
                    df = await collector.collect_with_retry(
                        symbol=request.symbol,
                        interval=interval,
                        start_date=request.start_date,
                        end_date=request.end_date
                    )
                    
                    if not df.empty:
                        # Store raw data
                        stored_count = await data_manager.store_historical_data(
                            df, request.symbol, source, interval
                        )
                        
                        collection_results['collected_data'][f"{source}_{interval}"] = {
                            'records': len(df),
                            'stored': stored_count,
                            'date_range': {
                                'start': df['timestamp'].min().isoformat(),
                                'end': df['timestamp'].max().isoformat()
                            }
                        }
                        
                        collection_results['total_records'] += stored_count
                        
                        # Generate features if requested
                        if request.generate_features:
                            task.message = f"Generating features for {request.symbol} {source} {interval}"
                            task.progress = 0.7 + (completed_operations / total_operations) * 0.3  # 30% for features
                            
                            enhanced_df = feature_engine.generate_all_features(df)
                            
                            if enhanced_df is not None and not enhanced_df.empty:
                                feature_count = await data_manager.store_features(
                                    enhanced_df, request.symbol, interval
                                )
                                
                                feature_summary = feature_engine.get_feature_summary(enhanced_df)
                                
                                collection_results['generated_features'][f"{source}_{interval}"] = {
                                    'feature_records': feature_count,
                                    'feature_summary': feature_summary
                                }
                    else:
                        error_msg = f"No data returned for {request.symbol} from {source} {interval}"
                        collection_results['errors'].append(error_msg)
                    
                    completed_operations += 1
                    
                except Exception as e:
                    error_msg = f"Error collecting {source} {interval}: {str(e)}"
                    collection_results['errors'].append(error_msg)
                    logger.error(error_msg)
                    completed_operations += 1
        
        # Update final task status
        task.status = "completed"
        task.progress = 1.0
        task.completed_at = datetime.now()
        task.results = collection_results
        
        if collection_results['total_records'] > 0:
            task.message = f"Successfully collected {collection_results['total_records']} records"
        else:
            task.message = "No data collected"
            task.status = "failed"
        
        logger.info(f"Task {task_id} completed: {task.message}")
        
    except Exception as e:
        # Update task with error status
        task = active_tasks.get(task_id)
        if task:
            task.status = "failed"
            task.progress = 0.0
            task.message = f"Collection failed: {str(e)}"
            task.completed_at = datetime.now()
        
        logger.error(f"Background collection task {task_id} failed: {e}")

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a data collection task."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    task_dict = task.dict()
    
    # Convert datetime objects to ISO format strings
    if task_dict.get('started_at'):
        task_dict['started_at'] = task_dict['started_at'].isoformat()
    if task_dict.get('completed_at'):
        task_dict['completed_at'] = task_dict['completed_at'].isoformat()
    
    return JSONResponse(content=task_dict)

@router.get("/tasks")
async def get_all_tasks():
    """Get status of all data collection tasks."""
    tasks_dict = {}
    for task_id, task in active_tasks.items():
        task_data = task.dict()
        # Convert datetime objects to ISO format strings
        if task_data.get('started_at'):
            task_data['started_at'] = task_data['started_at'].isoformat()
        if task_data.get('completed_at'):
            task_data['completed_at'] = task_data['completed_at'].isoformat()
        tasks_dict[task_id] = task_data
    
    return JSONResponse(content={
        "active_tasks": len(active_tasks),
        "tasks": tasks_dict
    })

@router.post("/query")
async def query_data(query: DataQuery):
    """Query stored historical data."""
    try:
        # Get historical data
        df = await data_manager.get_historical_data(
            symbol=query.symbol,
            source=query.source,
            interval=query.interval,
            start_date=query.start_date,
            end_date=query.end_date,
            limit=query.limit
        )
        
        result = {
            "symbol": query.symbol,
            "total_records": len(df),
            "data": df.to_dict('records') if not df.empty else []
        }
        
        # Include features if requested
        if query.include_features and not df.empty:
            features_df = await data_manager.get_features(
                symbol=query.symbol,
                interval=query.interval,
                start_date=query.start_date,
                end_date=query.end_date
            )
            
            if not features_df.empty:
                result["features"] = features_df.to_dict('records')
                result["total_feature_records"] = len(features_df)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error querying data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_data_summary(symbol: Optional[str] = Query(None)):
    """Get summary of stored data."""
    try:
        summary = await data_manager.get_data_summary(symbol)
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache")
async def clear_data_cache():
    """Clear data manager cache."""
    try:
        await data_manager.clear_cache()
        return JSONResponse(content={"message": "Data cache cleared successfully"})
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_old_data(days_to_keep: int = Query(365, ge=1, le=3650)):
    """Clean up old data beyond specified days."""
    try:
        await data_manager.cleanup_old_data(days_to_keep)
        return JSONResponse(content={
            "message": f"Successfully cleaned up data older than {days_to_keep} days"
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def data_collection_health():
    """Health check for data collection system."""
    try:
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check collectors
        for name, collector in collectors.items():
            health_status["components"][name] = {
                "connected": collector.is_connected,
                "error_count": collector.error_count,
                "last_request": collector.last_request_time
            }
        
        # Check data manager
        try:
            summary = await data_manager.get_data_summary()
            health_status["components"]["data_manager"] = {
                "status": "healthy",
                "cache_items": summary.get("cache_stats", {}).get("cached_items", 0)
            }
        except Exception as e:
            health_status["components"]["data_manager"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check feature engine
        health_status["components"]["feature_engine"] = {
            "status": "healthy" if feature_engine else "not_initialized",
            "supported_features": len(feature_engine.supported_features) if feature_engine else 0
        }
        
        # Overall status
        all_healthy = all(
            comp.get("connected", False) if "connected" in comp 
            else comp.get("status") == "healthy"
            for comp in health_status["components"].values()
        )
        
        if not all_healthy:
            health_status["status"] = "degraded"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Error in data collection health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))