"""
Data Models for AssureTheAnalyst
Pydantic models for data validation and serialization
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    SQL = "sql"

class DatasetInfo(BaseModel):
    """Model for dataset information"""
    name: str = Field(..., description="Dataset name")
    file_type: FileType = Field(..., description="File type")
    size: int = Field(..., description="File size in bytes")
    rows: Optional[int] = Field(None, description="Number of rows")
    columns: Optional[int] = Field(None, description="Number of columns")
    upload_date: datetime = Field(default_factory=datetime.now)
    file_path: str = Field(..., description="Path to uploaded file")

class ColumnInfo(BaseModel):
    """Model for column information"""
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Data type")
    null_count: int = Field(..., description="Number of null values")
    unique_count: int = Field(..., description="Number of unique values")
    min_value: Optional[Union[str, int, float]] = Field(None, description="Minimum value")
    max_value: Optional[Union[str, int, float]] = Field(None, description="Maximum value")
    mean_value: Optional[float] = Field(None, description="Mean value for numeric columns")

class DataSummary(BaseModel):
    """Model for data summary"""
    dataset_info: DatasetInfo
    columns: List[ColumnInfo]
    sample_data: List[Dict[str, Any]] = Field(..., description="Sample rows from dataset")
    
class UploadResponse(BaseModel):
    """Model for file upload response"""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Response message")
    dataset_id: Optional[str] = Field(None, description="Dataset identifier")
    dataset_info: Optional[DatasetInfo] = Field(None, description="Dataset information")

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: bool = Field(True, description="Error flag")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class SuccessResponse(BaseModel):
    """Model for success responses"""
    success: bool = Field(True, description="Success flag")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
