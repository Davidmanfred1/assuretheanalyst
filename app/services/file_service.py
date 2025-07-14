"""
File Service
Handles file upload, processing, and dataset management
"""

import os
import uuid
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.models.data_models import DatasetInfo, ColumnInfo, DataSummary, FileType
from app.utils.validators import sanitize_filename, get_file_type

class FileService:
    def __init__(self):
        self.upload_dir = Path("data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_registry = {}
    
    async def process_upload(self, file) -> Dict[str, Any]:
        """
        Process uploaded file and create dataset
        
        Args:
            file: Uploaded file object
            
        Returns:
            Dict containing dataset_id and dataset_info
        """
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        file_path = self.upload_dir / f"{dataset_id}_{safe_filename}"
        
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Determine file type
        file_type = get_file_type(file.filename)
        
        # Load and analyze data
        df = self._load_dataframe(file_path, file_type)
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=safe_filename,
            file_type=FileType(file_type),
            size=len(content),
            rows=len(df),
            columns=len(df.columns),
            upload_date=datetime.now(),
            file_path=str(file_path)
        )
        
        # Store in registry
        self.datasets_registry[dataset_id] = {
            "info": dataset_info,
            "dataframe": df
        }
        
        return {
            "dataset_id": dataset_id,
            "dataset_info": dataset_info
        }
    
    def _load_dataframe(self, file_path: Path, file_type: str) -> pd.DataFrame:
        """
        Load dataframe based on file type
        
        Args:
            file_path: Path to the file
            file_type: Type of file
            
        Returns:
            pandas DataFrame
        """
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'excel':
            return pd.read_excel(file_path)
        elif file_type == 'json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    async def get_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of all datasets
        
        Returns:
            List of dataset information
        """
        datasets = []
        for dataset_id, data in self.datasets_registry.items():
            datasets.append({
                "dataset_id": dataset_id,
                "info": data["info"].dict()
            })
        return datasets
    
    async def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific dataset
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dataset information or None if not found
        """
        if dataset_id not in self.datasets_registry:
            return None
        
        data = self.datasets_registry[dataset_id]
        df = data["dataframe"]
        
        # Generate column information
        columns_info = []
        for column in df.columns:
            col_data = df[column]
            columns_info.append(ColumnInfo(
                name=column,
                data_type=str(col_data.dtype),
                null_count=int(col_data.isnull().sum()),
                unique_count=int(col_data.nunique()),
                min_value=col_data.min() if pd.api.types.is_numeric_dtype(col_data) else None,
                max_value=col_data.max() if pd.api.types.is_numeric_dtype(col_data) else None,
                mean_value=col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else None
            ))
        
        return DataSummary(
            dataset_info=data["info"],
            columns=columns_info,
            sample_data=df.head(5).fillna("null").to_dict('records')
        ).dict()
    
    async def get_dataset_preview(self, dataset_id: str, rows: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get preview of dataset
        
        Args:
            dataset_id: Dataset identifier
            rows: Number of rows to preview
            
        Returns:
            Preview data or None if not found
        """
        if dataset_id not in self.datasets_registry:
            return None
        
        df = self.datasets_registry[dataset_id]["dataframe"]

        # Handle NaN values for JSON serialization
        preview_df = df.head(rows).fillna("null")

        return {
            "data": preview_df.to_dict('records'),
            "columns": list(df.columns),
            "total_rows": len(df)
        }
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            True if deleted successfully, False if not found
        """
        if dataset_id not in self.datasets_registry:
            return False
        
        # Remove file
        dataset_info = self.datasets_registry[dataset_id]["info"]
        file_path = Path(dataset_info.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Remove from registry
        del self.datasets_registry[dataset_id]
        return True
    
    def get_dataframe(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Get dataframe for a dataset

        Args:
            dataset_id: Dataset identifier

        Returns:
            pandas DataFrame or None if not found
        """
        if dataset_id not in self.datasets_registry:
            return None
        return self.datasets_registry[dataset_id]["dataframe"]

    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Load dataset (alias for get_dataframe for compatibility)

        Args:
            dataset_id: Dataset identifier

        Returns:
            pandas DataFrame or None if not found
        """
        return self.get_dataframe(dataset_id)
