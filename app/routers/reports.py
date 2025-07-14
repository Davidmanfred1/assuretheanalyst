"""
Reports Router
Handles report generation and export functionality
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List

from app.models.analysis_models import ReportRequest, ReportResult
from app.services.report_service import ReportService

router = APIRouter()

# Initialize report service
report_service = ReportService()

@router.post("/generate", response_model=ReportResult)
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive data analysis report
    """
    try:
        result = await report_service.generate_report(
            dataset_id=request.dataset_id,
            report_type=request.report_type,
            sections=request.sections,
            format=request.format,
            title=request.title
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/download/{report_id}")
async def download_report(report_id: str):
    """
    Download a generated report
    """
    try:
        file_path = await report_service.get_report_file(report_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            path=file_path,
            filename=f"report_{report_id}.pdf",
            media_type="application/pdf"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")

@router.get("/list")
async def list_reports():
    """
    List all generated reports
    """
    try:
        reports = await report_service.list_reports()
        return {"success": True, "reports": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve reports: {str(e)}")

@router.delete("/{report_id}")
async def delete_report(report_id: str):
    """
    Delete a report
    """
    try:
        success = await report_service.delete_report(report_id)
        if not success:
            raise HTTPException(status_code=404, detail="Report not found")
        return {"success": True, "message": "Report deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")

@router.get("/templates")
async def get_report_templates():
    """
    Get available report templates
    """
    try:
        templates = report_service.get_report_templates()
        return {"success": True, "templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve templates: {str(e)}")

@router.post("/export/{dataset_id}")
async def export_dataset(dataset_id: str, format: str = "csv"):
    """
    Export dataset in specified format
    """
    try:
        file_path = await report_service.export_dataset(dataset_id, format)
        if not file_path:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return FileResponse(
            path=file_path,
            filename=f"dataset_{dataset_id}.{format}",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
