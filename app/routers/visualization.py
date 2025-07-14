"""
Visualization Router
Handles data visualization and chart generation
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from app.models.analysis_models import VisualizationRequest, VisualizationResult
from app.services.visualization_service import VisualizationService

router = APIRouter()

# Initialize visualization service
visualization_service = VisualizationService()

@router.post("/chart", response_model=VisualizationResult)
async def create_chart(request: VisualizationRequest):
    """
    Create a data visualization chart
    """
    try:
        result = await visualization_service.create_chart(
            dataset_id=request.dataset_id,
            chart_type=request.chart_type,
            x_column=request.x_column,
            y_column=request.y_column,
            color_column=request.color_column,
            title=request.title,
            parameters=request.parameters or {}
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart creation failed: {str(e)}")

@router.get("/chart/{chart_id}")
async def get_chart(chart_id: str):
    """
    Retrieve a previously created chart
    """
    try:
        chart = await visualization_service.get_chart(chart_id)
        if not chart:
            raise HTTPException(status_code=404, detail="Chart not found")
        return {"success": True, "chart": chart}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chart: {str(e)}")

@router.post("/dashboard")
async def create_dashboard(dataset_id: str, chart_configs: list):
    """
    Create a dashboard with multiple charts
    """
    try:
        dashboard = await visualization_service.create_dashboard(dataset_id, chart_configs)
        return {"success": True, "dashboard": dashboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard creation failed: {str(e)}")

@router.get("/datasets/{dataset_id}/charts")
async def list_charts(dataset_id: str):
    """
    List all charts for a dataset
    """
    try:
        charts = await visualization_service.get_dataset_charts(dataset_id)
        return {"success": True, "charts": charts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve charts: {str(e)}")

@router.delete("/chart/{chart_id}")
async def delete_chart(chart_id: str):
    """
    Delete a chart
    """
    try:
        success = await visualization_service.delete_chart(chart_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chart not found")
        return {"success": True, "message": "Chart deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chart: {str(e)}")

@router.get("/chart-types")
async def get_chart_types():
    """
    Get available chart types and their configurations
    """
    try:
        chart_types = visualization_service.get_available_chart_types()
        return {"success": True, "chart_types": chart_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chart types: {str(e)}")
