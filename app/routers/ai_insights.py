"""
AI Insights Router
Handles AI-powered insights, automated analysis, and natural language explanations
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.services.ai_insights_service import AIInsightsService

router = APIRouter()

# AI insights service will be initialized with shared file service
ai_insights_service = None

def set_file_service(service):
    global ai_insights_service
    ai_insights_service = AIInsightsService()
    ai_insights_service.file_service = service

@router.get("/automated/{dataset_id}")
async def get_automated_insights(dataset_id: str):
    """
    Generate automated insights for a dataset
    """
    try:
        insights = await ai_insights_service.generate_automated_insights(dataset_id)
        return {"success": True, "insights": insights}
    except ValueError as e:
        # Handle dataset not found gracefully
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@router.post("/explain")
async def explain_analysis_result(request: Dict[str, Any]):
    """
    Generate natural language explanation for analysis results
    """
    try:
        analysis_result = request.get("analysis_result")
        analysis_type = request.get("analysis_type")
        
        if not analysis_result or not analysis_type:
            raise HTTPException(status_code=400, detail="analysis_result and analysis_type are required")
        
        explanation = await ai_insights_service.explain_analysis_result(analysis_result, analysis_type)
        return {"success": True, "explanation": explanation}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")

@router.get("/recommendations/{dataset_id}")
async def get_recommendations(dataset_id: str):
    """
    Get AI-powered recommendations for a dataset
    """
    try:
        insights = await ai_insights_service.generate_automated_insights(dataset_id)
        recommendations = insights.get("recommendations", [])
        
        return {
            "success": True,
            "recommendations": recommendations,
            "data_quality_score": insights.get("data_quality_score", 0),
            "summary": insights.get("summary", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@router.get("/data-quality/{dataset_id}")
async def assess_data_quality(dataset_id: str):
    """
    Assess data quality and provide insights
    """
    try:
        insights = await ai_insights_service.generate_automated_insights(dataset_id)
        
        # Filter for data quality insights
        quality_insights = [
            insight for insight in insights.get("insights", [])
            if insight.get("type") == "data_quality"
        ]
        
        return {
            "success": True,
            "data_quality_score": insights.get("data_quality_score", 0),
            "quality_insights": quality_insights,
            "recommendations": insights.get("recommendations", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assess data quality: {str(e)}")

@router.get("/patterns/{dataset_id}")
async def detect_patterns(dataset_id: str):
    """
    Detect patterns in the dataset
    """
    try:
        insights = await ai_insights_service.generate_automated_insights(dataset_id)
        
        # Filter for pattern insights
        pattern_insights = [
            insight for insight in insights.get("insights", [])
            if insight.get("type") == "pattern"
        ]
        
        return {
            "success": True,
            "patterns": pattern_insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect patterns: {str(e)}")

@router.get("/relationships/{dataset_id}")
async def analyze_relationships(dataset_id: str):
    """
    Analyze relationships between variables
    """
    try:
        insights = await ai_insights_service.generate_automated_insights(dataset_id)
        
        # Filter for relationship insights
        relationship_insights = [
            insight for insight in insights.get("insights", [])
            if insight.get("type") == "relationship"
        ]
        
        return {
            "success": True,
            "relationships": relationship_insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze relationships: {str(e)}")
