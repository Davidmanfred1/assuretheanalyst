"""
ETL (Extract, Transform, Load) Service
Handles data pipelines, transformations, and automated processing workflows
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from enum import Enum

from app.services.file_service import FileService
from app.services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TransformationType(str, Enum):
    CLEAN_MISSING = "clean_missing"
    REMOVE_DUPLICATES = "remove_duplicates"
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    ENCODE_CATEGORICAL = "encode_categorical"
    FEATURE_ENGINEERING = "feature_engineering"
    FILTER_ROWS = "filter_rows"
    SELECT_COLUMNS = "select_columns"
    AGGREGATE = "aggregate"
    MERGE_DATASETS = "merge_datasets"

class ETLPipeline:
    """Represents a data processing pipeline"""
    
    def __init__(self, pipeline_id: str, name: str, description: str = ""):
        self.pipeline_id = pipeline_id
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []
        self.status = PipelineStatus.PENDING
        self.created_at = datetime.now()
        self.last_run = None
        self.run_count = 0
        self.schedule = None
        self.input_datasets = []
        self.output_dataset = None
        
    def add_step(self, transformation_type: TransformationType, parameters: Dict[str, Any], order: int = None):
        """Add a transformation step to the pipeline"""
        step = {
            "step_id": str(uuid.uuid4()),
            "transformation_type": transformation_type,
            "parameters": parameters,
            "order": order if order is not None else len(self.steps),
            "created_at": datetime.now().isoformat()
        }
        self.steps.append(step)
        # Sort steps by order
        self.steps.sort(key=lambda x: x["order"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "schedule": self.schedule,
            "input_datasets": self.input_datasets,
            "output_dataset": self.output_dataset
        }

class ETLService:
    """Service for managing ETL pipelines and data transformations"""
    
    def __init__(self):
        self.file_service = FileService()
        self.analysis_service = AnalysisService()
        self.pipelines: Dict[str, ETLPipeline] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.scheduled_jobs: Dict[str, asyncio.Task] = {}
        self.job_history: List[Dict[str, Any]] = []
        
        # Scheduler will be started when needed
        self._scheduler_started = False
    
    async def create_pipeline(self, name: str, description: str = "") -> str:
        """Create a new ETL pipeline"""
        pipeline_id = str(uuid.uuid4())
        pipeline = ETLPipeline(pipeline_id, name, description)
        self.pipelines[pipeline_id] = pipeline
        
        logger.info(f"Created pipeline {pipeline_id}: {name}")
        return pipeline_id
    
    async def add_pipeline_step(self, pipeline_id: str, transformation_type: TransformationType, 
                              parameters: Dict[str, Any], order: int = None) -> bool:
        """Add a step to an existing pipeline"""
        if pipeline_id not in self.pipelines:
            return False
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.add_step(transformation_type, parameters, order)
        
        logger.info(f"Added step {transformation_type} to pipeline {pipeline_id}")
        return True
    
    async def run_pipeline(self, pipeline_id: str, input_datasets: List[str], 
                          output_name: str = None) -> Dict[str, Any]:
        """Execute a pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError("Pipeline not found")
        
        pipeline = self.pipelines[pipeline_id]
        
        if pipeline.status == PipelineStatus.RUNNING:
            raise ValueError("Pipeline is already running")
        
        # Create execution task
        task = asyncio.create_task(
            self._execute_pipeline(pipeline, input_datasets, output_name)
        )
        self.running_jobs[pipeline_id] = task
        
        try:
            result = await task
            return result
        finally:
            if pipeline_id in self.running_jobs:
                del self.running_jobs[pipeline_id]
    
    async def _execute_pipeline(self, pipeline: ETLPipeline, input_datasets: List[str], 
                               output_name: str = None) -> Dict[str, Any]:
        """Execute pipeline steps"""
        pipeline.status = PipelineStatus.RUNNING
        pipeline.last_run = datetime.now()
        pipeline.run_count += 1
        
        execution_log = {
            "pipeline_id": pipeline.pipeline_id,
            "execution_id": str(uuid.uuid4()),
            "start_time": datetime.now(),
            "input_datasets": input_datasets,
            "steps_executed": [],
            "errors": [],
            "status": "running"
        }
        
        try:
            # Load input datasets
            dataframes = {}
            for dataset_id in input_datasets:
                df = self.file_service.get_dataframe(dataset_id)
                if df is None:
                    raise ValueError(f"Dataset {dataset_id} not found")
                dataframes[dataset_id] = df
            
            # If multiple datasets, merge them (simple concatenation for now)
            if len(dataframes) == 1:
                current_df = list(dataframes.values())[0].copy()
            else:
                current_df = pd.concat(dataframes.values(), ignore_index=True)
            
            # Execute each step
            for step in pipeline.steps:
                step_start = datetime.now()
                try:
                    current_df = await self._execute_transformation(
                        current_df, step["transformation_type"], step["parameters"]
                    )
                    
                    execution_log["steps_executed"].append({
                        "step_id": step["step_id"],
                        "transformation_type": step["transformation_type"],
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed",
                        "rows_after": len(current_df),
                        "columns_after": len(current_df.columns)
                    })
                    
                except Exception as e:
                    error_msg = f"Step {step['step_id']} failed: {str(e)}"
                    execution_log["errors"].append(error_msg)
                    logger.error(error_msg)
                    raise
            
            # Save output dataset
            if output_name:
                # Create a temporary file and register it
                output_path = self.file_service.upload_dir / f"pipeline_output_{pipeline.pipeline_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                current_df.to_csv(output_path, index=False)
                
                # Register as new dataset (simplified)
                output_dataset_id = str(uuid.uuid4())
                self.file_service.datasets_registry[output_dataset_id] = {
                    "info": {
                        "name": output_name,
                        "file_type": "csv",
                        "size": output_path.stat().st_size,
                        "rows": len(current_df),
                        "columns": len(current_df.columns),
                        "upload_date": datetime.now(),
                        "file_path": str(output_path)
                    },
                    "dataframe": current_df
                }
                pipeline.output_dataset = output_dataset_id
            
            pipeline.status = PipelineStatus.COMPLETED
            execution_log["status"] = "completed"
            execution_log["end_time"] = datetime.now()
            execution_log["duration"] = (execution_log["end_time"] - execution_log["start_time"]).total_seconds()
            execution_log["output_dataset_id"] = pipeline.output_dataset
            
            self.job_history.append(execution_log)
            
            return {
                "success": True,
                "execution_id": execution_log["execution_id"],
                "output_dataset_id": pipeline.output_dataset,
                "rows_processed": len(current_df),
                "columns_processed": len(current_df.columns),
                "duration": execution_log["duration"],
                "steps_executed": len(execution_log["steps_executed"])
            }
            
        except Exception as e:
            pipeline.status = PipelineStatus.FAILED
            execution_log["status"] = "failed"
            execution_log["end_time"] = datetime.now()
            execution_log["error"] = str(e)
            
            self.job_history.append(execution_log)
            
            logger.error(f"Pipeline {pipeline.pipeline_id} failed: {str(e)}")
            raise
    
    async def _execute_transformation(self, df: pd.DataFrame, transformation_type: TransformationType, 
                                    parameters: Dict[str, Any]) -> pd.DataFrame:
        """Execute a single transformation step"""
        
        if transformation_type == TransformationType.CLEAN_MISSING:
            strategy = parameters.get("strategy", "drop")
            columns = parameters.get("columns", [])
            
            if strategy == "drop":
                if columns:
                    df = df.dropna(subset=columns)
                else:
                    df = df.dropna()
            elif strategy == "fill_mean":
                if columns:
                    for col in columns:
                        if col in df.columns and df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].mean(), inplace=True)
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == "fill_value":
                fill_value = parameters.get("fill_value", 0)
                if columns:
                    df[columns] = df[columns].fillna(fill_value)
                else:
                    df = df.fillna(fill_value)
        
        elif transformation_type == TransformationType.REMOVE_DUPLICATES:
            subset = parameters.get("subset", None)
            df = df.drop_duplicates(subset=subset)
        
        elif transformation_type == TransformationType.NORMALIZE:
            columns = parameters.get("columns", [])
            if not columns:
                columns = df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif transformation_type == TransformationType.STANDARDIZE:
            columns = parameters.get("columns", [])
            if not columns:
                columns = df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val != 0:
                        df[col] = (df[col] - mean_val) / std_val
        
        elif transformation_type == TransformationType.ENCODE_CATEGORICAL:
            columns = parameters.get("columns", [])
            method = parameters.get("method", "onehot")
            
            if method == "onehot":
                df = pd.get_dummies(df, columns=columns)
            elif method == "label":
                from sklearn.preprocessing import LabelEncoder
                for col in columns:
                    if col in df.columns:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
        
        elif transformation_type == TransformationType.FILTER_ROWS:
            condition = parameters.get("condition", "")
            if condition:
                # Simple filtering - can be enhanced
                df = df.query(condition)
        
        elif transformation_type == TransformationType.SELECT_COLUMNS:
            columns = parameters.get("columns", [])
            if columns:
                df = df[columns]
        
        elif transformation_type == TransformationType.AGGREGATE:
            group_by = parameters.get("group_by", [])
            agg_functions = parameters.get("agg_functions", {})
            
            if group_by and agg_functions:
                df = df.groupby(group_by).agg(agg_functions).reset_index()
        
        return df
    
    async def schedule_pipeline(self, pipeline_id: str, schedule_config: Dict[str, Any]) -> bool:
        """Schedule a pipeline to run automatically"""
        if pipeline_id not in self.pipelines:
            return False
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.schedule = schedule_config
        
        # Cancel existing scheduled job
        if pipeline_id in self.scheduled_jobs:
            self.scheduled_jobs[pipeline_id].cancel()
        
        # Create new scheduled job
        task = asyncio.create_task(
            self._schedule_job(pipeline_id, schedule_config)
        )
        self.scheduled_jobs[pipeline_id] = task
        
        logger.info(f"Scheduled pipeline {pipeline_id}")
        return True
    
    async def _schedule_job(self, pipeline_id: str, schedule_config: Dict[str, Any]):
        """Handle scheduled pipeline execution"""
        interval_minutes = schedule_config.get("interval_minutes", 60)
        input_datasets = schedule_config.get("input_datasets", [])
        output_name = schedule_config.get("output_name", f"scheduled_output_{pipeline_id}")
        
        while True:
            try:
                await asyncio.sleep(interval_minutes * 60)  # Convert to seconds
                
                if pipeline_id in self.pipelines:
                    await self.run_pipeline(pipeline_id, input_datasets, output_name)
                    logger.info(f"Scheduled execution completed for pipeline {pipeline_id}")
                else:
                    break  # Pipeline was deleted
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduled execution failed for pipeline {pipeline_id}: {str(e)}")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Clean up completed jobs
                completed_jobs = [
                    job_id for job_id, task in self.running_jobs.items()
                    if task.done()
                ]
                
                for job_id in completed_jobs:
                    del self.running_jobs[job_id]
                
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
    
    async def get_pipelines(self) -> List[Dict[str, Any]]:
        """Get all pipelines"""
        return [pipeline.to_dict() for pipeline in self.pipelines.values()]
    
    async def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific pipeline"""
        if pipeline_id in self.pipelines:
            return self.pipelines[pipeline_id].to_dict()
        return None
    
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline"""
        if pipeline_id not in self.pipelines:
            return False
        
        # Cancel running job
        if pipeline_id in self.running_jobs:
            self.running_jobs[pipeline_id].cancel()
            del self.running_jobs[pipeline_id]
        
        # Cancel scheduled job
        if pipeline_id in self.scheduled_jobs:
            self.scheduled_jobs[pipeline_id].cancel()
            del self.scheduled_jobs[pipeline_id]
        
        del self.pipelines[pipeline_id]
        logger.info(f"Deleted pipeline {pipeline_id}")
        return True
    
    async def get_job_history(self, pipeline_id: str = None) -> List[Dict[str, Any]]:
        """Get job execution history"""
        if pipeline_id:
            return [job for job in self.job_history if job["pipeline_id"] == pipeline_id]
        return self.job_history
    
    def get_transformation_types(self) -> Dict[str, Any]:
        """Get available transformation types and their parameters"""
        return {
            "clean_missing": {
                "name": "Clean Missing Values",
                "description": "Handle missing data with various strategies",
                "parameters": {
                    "strategy": ["drop", "fill_mean", "fill_value"],
                    "columns": "list of column names (optional)",
                    "fill_value": "value to fill with (for fill_value strategy)"
                }
            },
            "remove_duplicates": {
                "name": "Remove Duplicates",
                "description": "Remove duplicate rows",
                "parameters": {
                    "subset": "list of column names to check for duplicates (optional)"
                }
            },
            "normalize": {
                "name": "Normalize Data",
                "description": "Scale data to 0-1 range",
                "parameters": {
                    "columns": "list of column names (optional, defaults to all numeric)"
                }
            },
            "standardize": {
                "name": "Standardize Data",
                "description": "Scale data to mean=0, std=1",
                "parameters": {
                    "columns": "list of column names (optional, defaults to all numeric)"
                }
            },
            "encode_categorical": {
                "name": "Encode Categorical",
                "description": "Convert categorical variables to numeric",
                "parameters": {
                    "columns": "list of column names",
                    "method": ["onehot", "label"]
                }
            },
            "filter_rows": {
                "name": "Filter Rows",
                "description": "Filter rows based on conditions",
                "parameters": {
                    "condition": "pandas query string (e.g., 'age > 25')"
                }
            },
            "select_columns": {
                "name": "Select Columns",
                "description": "Select specific columns",
                "parameters": {
                    "columns": "list of column names to keep"
                }
            },
            "aggregate": {
                "name": "Aggregate Data",
                "description": "Group and aggregate data",
                "parameters": {
                    "group_by": "list of column names to group by",
                    "agg_functions": "dictionary of column: function pairs"
                }
            }
        }
