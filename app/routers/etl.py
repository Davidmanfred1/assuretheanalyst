"""
ETL Router
Handles ETL pipeline creation, execution, and management
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from pydantic import BaseModel

from app.services.etl_service import ETLService, TransformationType

router = APIRouter()

# Initialize ETL service
etl_service = ETLService()

class PipelineCreateRequest(BaseModel):
    name: str
    description: str = ""

class PipelineStepRequest(BaseModel):
    transformation_type: TransformationType
    parameters: Dict[str, Any]
    order: int = None

class PipelineExecuteRequest(BaseModel):
    input_datasets: List[str]
    output_name: str = None

class PipelineScheduleRequest(BaseModel):
    interval_minutes: int
    input_datasets: List[str]
    output_name: str = None

@router.post("/pipelines")
async def create_pipeline(request: PipelineCreateRequest):
    """Create a new ETL pipeline"""
    try:
        pipeline_id = await etl_service.create_pipeline(request.name, request.description)
        return {"success": True, "pipeline_id": pipeline_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")

@router.get("/pipelines")
async def get_pipelines():
    """Get all ETL pipelines"""
    try:
        pipelines = await etl_service.get_pipelines()
        return {"success": True, "pipelines": pipelines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipelines: {str(e)}")

@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get a specific pipeline"""
    try:
        pipeline = await etl_service.get_pipeline(pipeline_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return {"success": True, "pipeline": pipeline}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/steps")
async def add_pipeline_step(pipeline_id: str, request: PipelineStepRequest):
    """Add a step to a pipeline"""
    try:
        success = await etl_service.add_pipeline_step(
            pipeline_id, 
            request.transformation_type, 
            request.parameters, 
            request.order
        )
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return {"success": True, "message": "Step added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add step: {str(e)}")

@router.post("/pipelines/{pipeline_id}/execute")
async def execute_pipeline(pipeline_id: str, request: PipelineExecuteRequest):
    """Execute a pipeline"""
    try:
        result = await etl_service.run_pipeline(
            pipeline_id, 
            request.input_datasets, 
            request.output_name
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/schedule")
async def schedule_pipeline(pipeline_id: str, request: PipelineScheduleRequest):
    """Schedule a pipeline for automatic execution"""
    try:
        schedule_config = {
            "interval_minutes": request.interval_minutes,
            "input_datasets": request.input_datasets,
            "output_name": request.output_name
        }
        success = await etl_service.schedule_pipeline(pipeline_id, schedule_config)
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return {"success": True, "message": "Pipeline scheduled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule pipeline: {str(e)}")

@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline"""
    try:
        success = await etl_service.delete_pipeline(pipeline_id)
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return {"success": True, "message": "Pipeline deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {str(e)}")

@router.get("/pipelines/{pipeline_id}/history")
async def get_pipeline_history(pipeline_id: str):
    """Get execution history for a pipeline"""
    try:
        history = await etl_service.get_job_history(pipeline_id)
        return {"success": True, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.get("/history")
async def get_all_history():
    """Get all job execution history"""
    try:
        history = await etl_service.get_job_history()
        return {"success": True, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.get("/transformations")
async def get_transformation_types():
    """Get available transformation types"""
    try:
        transformations = etl_service.get_transformation_types()
        return {"success": True, "transformations": transformations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transformations: {str(e)}")

@router.post("/transform")
async def quick_transform(request: Dict[str, Any]):
    """Perform a quick transformation on a dataset"""
    try:
        dataset_id = request.get("dataset_id")
        transformation_type = request.get("transformation_type")
        parameters = request.get("parameters", {})
        output_name = request.get("output_name", f"transformed_{dataset_id}")
        
        if not dataset_id or not transformation_type:
            raise HTTPException(status_code=400, detail="dataset_id and transformation_type are required")
        
        # Create a temporary pipeline
        pipeline_id = await etl_service.create_pipeline(f"Quick Transform - {transformation_type}", "Temporary pipeline for quick transformation")
        
        # Add the transformation step
        await etl_service.add_pipeline_step(pipeline_id, TransformationType(transformation_type), parameters)
        
        # Execute the pipeline
        result = await etl_service.run_pipeline(pipeline_id, [dataset_id], output_name)
        
        # Clean up the temporary pipeline
        await etl_service.delete_pipeline(pipeline_id)
        
        return {"success": True, "result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform transformation: {str(e)}")

@router.get("/status")
async def get_etl_status():
    """Get ETL service status"""
    try:
        pipelines = await etl_service.get_pipelines()
        running_jobs = len(etl_service.running_jobs)
        scheduled_jobs = len(etl_service.scheduled_jobs)
        total_executions = len(etl_service.job_history)
        
        return {
            "success": True,
            "status": {
                "total_pipelines": len(pipelines),
                "running_jobs": running_jobs,
                "scheduled_jobs": scheduled_jobs,
                "total_executions": total_executions,
                "pipelines": pipelines
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")
