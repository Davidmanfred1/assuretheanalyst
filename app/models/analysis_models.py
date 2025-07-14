"""
Analysis Models for AssureTheAnalyst
Models for statistical analysis and machine learning operations
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

class AnalysisType(str, Enum):
    DESCRIPTIVE = "descriptive"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ADVANCED_CLUSTERING = "advanced_clustering"
    FORECASTING = "forecasting"

class StatisticalTest(str, Enum):
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    NORMALITY = "normality"

class VisualizationType(str, Enum):
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"

class AnalysisRequest(BaseModel):
    """Model for analysis requests"""
    dataset_id: str = Field(..., description="Dataset identifier")
    analysis_type: AnalysisType = Field(..., description="Type of analysis")
    columns: List[str] = Field(..., description="Columns to analyze")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis parameters")

class StatisticalTestRequest(BaseModel):
    """Model for statistical test requests"""
    dataset_id: str = Field(..., description="Dataset identifier")
    test_type: StatisticalTest = Field(..., description="Type of statistical test")
    columns: List[str] = Field(..., description="Columns for the test")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Test parameters")

class VisualizationRequest(BaseModel):
    """Model for visualization requests"""
    dataset_id: str = Field(..., description="Dataset identifier")
    chart_type: VisualizationType = Field(..., description="Type of visualization")
    x_column: Optional[str] = Field(None, description="X-axis column")
    y_column: Optional[str] = Field(None, description="Y-axis column")
    color_column: Optional[str] = Field(None, description="Color grouping column")
    title: Optional[str] = Field(None, description="Chart title")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Chart parameters")

class AnalysisResult(BaseModel):
    """Model for analysis results"""
    analysis_type: AnalysisType
    results: Dict[str, Any] = Field(..., description="Analysis results")
    summary: str = Field(..., description="Human-readable summary")
    recommendations: Optional[List[str]] = Field(None, description="Analysis recommendations")
    insights: Optional[Dict[str, Any]] = Field(None, description="AI-generated insights")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Analysis timestamp")
    success: bool = Field(True, description="Whether analysis was successful")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")

class VisualizationResult(BaseModel):
    """Model for visualization results"""
    chart_type: VisualizationType
    chart_data: Dict[str, Any] = Field(..., description="Chart configuration and data")
    chart_html: str = Field(..., description="HTML representation of the chart")

class ReportRequest(BaseModel):
    """Model for report generation requests"""
    dataset_id: str = Field(..., description="Dataset identifier")
    report_type: str = Field(..., description="Type of report")
    sections: List[str] = Field(..., description="Report sections to include")
    format: str = Field(default="pdf", description="Output format (pdf, html, excel)")
    title: Optional[str] = Field(None, description="Report title")
    
class ReportResult(BaseModel):
    """Model for report generation results"""
    report_id: str = Field(..., description="Report identifier")
    file_path: str = Field(..., description="Path to generated report")
    format: str = Field(..., description="Report format")
    created_at: str = Field(..., description="Creation timestamp")
