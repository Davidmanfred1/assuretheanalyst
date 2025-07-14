"""
File Upload Router
Handles file upload operations for various data formats
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
import os
import uuid
from pathlib import Path

from app.models.data_models import UploadResponse, ErrorResponse, DatasetInfo, FileType
from app.services.file_service import FileService
from app.services.data_quality_service import DataQualityService
from app.utils.validators import validate_file_type, validate_file_size

router = APIRouter()

# Services will be injected from main
file_service = None
quality_service = DataQualityService()

def set_file_service(service):
    global file_service
    file_service = service

@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a data file (CSV, Excel, JSON)
    """
    try:
        # Validate file type
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload CSV, Excel, or JSON files."
            )
        
        # Validate file size (max 50MB)
        if not validate_file_size(file.size):
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 50MB."
            )
        
        # Process the upload
        result = await file_service.process_upload(file)
        
        return UploadResponse(
            success=True,
            message="File uploaded successfully",
            dataset_id=result["dataset_id"],
            dataset_info=result["dataset_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/datasets")
async def list_datasets():
    """
    Get list of uploaded datasets
    """
    try:
        datasets = await file_service.get_datasets()
        return {"success": True, "datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve datasets: {str(e)}")

@router.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """
    Get information about a specific dataset
    """
    try:
        dataset_info = await file_service.get_dataset_info(dataset_id)
        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"success": True, "dataset": dataset_info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset info: {str(e)}")

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset
    """
    try:
        success = await file_service.delete_dataset(dataset_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"success": True, "message": "Dataset deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, rows: int = 10):
    """
    Get a preview of the dataset
    """
    try:
        preview_data = await file_service.get_dataset_preview(dataset_id, rows)
        if not preview_data:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"success": True, "preview": preview_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview dataset: {str(e)}")

@router.post("/quality-check/{dataset_id}")
async def assess_data_quality(dataset_id: str):
    """
    Perform comprehensive data quality assessment on a dataset
    """
    try:
        print(f"🔍 Quality check requested for dataset: {dataset_id}")

        if not file_service:
            print("❌ File service not initialized")
            raise HTTPException(status_code=500, detail="File service not initialized")

        print(f"✅ File service available")

        # Load the dataset
        df = file_service.load_dataset(dataset_id)
        if df is None:
            print(f"❌ Dataset {dataset_id} not found")
            raise HTTPException(status_code=404, detail="Dataset not found")

        print(f"✅ Dataset loaded: {df.shape}")

        # Check if quality service is available
        if not quality_service:
            print("❌ Quality service not initialized")
            raise HTTPException(status_code=500, detail="Quality service not initialized")

        print("✅ Quality service available")

        # Perform quality assessment
        print("🔍 Starting quality assessment...")
        quality_report = quality_service.assess_data_quality(df)
        print("✅ Quality assessment completed")

        return {
            "success": True,
            "quality_report": quality_report,
            "message": "Data quality assessment completed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Quality assessment exception: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")
