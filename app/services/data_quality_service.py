"""
Data Quality Service
Comprehensive data validation and quality assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re


class DataQualityService:
    """
    Service for comprehensive data quality assessment and validation
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% non-null values
            'uniqueness': 0.95,    # 95% unique values for ID columns
            'validity': 0.90,      # 90% valid format values
            'consistency': 0.95,   # 95% consistent values
            'accuracy': 0.90       # 90% accurate values (domain-specific)
        }

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def assess_data_quality(self, df: pd.DataFrame, column_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment
        """
        quality_report = {
            'overall_score': 0.0,
            'dimensions': {},
            'column_quality': {},
            'issues': [],
            'recommendations': [],
            'summary': '',
            'timestamp': datetime.now().isoformat()
        }
        
        # Assess each quality dimension
        try:
            print("🔍 Assessing completeness...")
            quality_report['dimensions']['completeness'] = self._assess_completeness(df)
            print("✅ Completeness assessed")

            print("🔍 Assessing uniqueness...")
            quality_report['dimensions']['uniqueness'] = self._assess_uniqueness(df)
            print("✅ Uniqueness assessed")

            print("🔍 Assessing validity...")
            quality_report['dimensions']['validity'] = self._assess_validity(df, column_types)
            print("✅ Validity assessed")

            print("🔍 Assessing consistency...")
            quality_report['dimensions']['consistency'] = self._assess_consistency(df)
            print("✅ Consistency assessed")

            print("🔍 Assessing accuracy...")
            quality_report['dimensions']['accuracy'] = self._assess_accuracy(df, column_types)
            print("✅ Accuracy assessed")
        except Exception as e:
            print(f"❌ Error in dimension assessment: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            raise
        
        # Assess quality for each column
        for column in df.columns:
            quality_report['column_quality'][column] = self._assess_column_quality(df[column], column)
        
        # Calculate overall score
        dimension_scores = [dim['score'] for dim in quality_report['dimensions'].values()]
        quality_report['overall_score'] = np.mean(dimension_scores)
        
        # Generate issues and recommendations
        quality_report['issues'] = self._identify_issues(quality_report)
        quality_report['recommendations'] = self._generate_recommendations(quality_report)
        quality_report['summary'] = self._generate_summary(quality_report)
        
        # Convert numpy types to Python native types for JSON serialization
        return self._convert_numpy_types(quality_report)
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness (missing values)"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells)
        
        column_completeness = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            column_completeness[column] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_count / len(df) * 100),
                'completeness_ratio': float(1 - missing_count / len(df))
            }
        
        return {
            'score': float(completeness_ratio),
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells),
            'missing_percentage': float(missing_cells / total_cells * 100),
            'column_details': column_completeness,
            'threshold_met': completeness_ratio >= self.quality_thresholds['completeness']
        }
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness (duplicate detection)"""
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        uniqueness_ratio = 1 - (duplicate_rows / total_rows)
        
        column_uniqueness = {}
        for column in df.columns:
            unique_count = df[column].nunique()
            non_null_count = df[column].count()
            if non_null_count > 0:
                uniqueness_ratio_col = unique_count / non_null_count
            else:
                uniqueness_ratio_col = 0
            
            column_uniqueness[column] = {
                'unique_count': int(unique_count),
                'total_count': int(non_null_count),
                'uniqueness_ratio': float(uniqueness_ratio_col),
                'duplicate_count': int(non_null_count - unique_count)
            }
        
        return {
            'score': float(uniqueness_ratio),
            'duplicate_rows': int(duplicate_rows),
            'total_rows': int(total_rows),
            'duplicate_percentage': float(duplicate_rows / total_rows * 100),
            'column_details': column_uniqueness,
            'threshold_met': uniqueness_ratio >= self.quality_thresholds['uniqueness']
        }
    
    def _assess_validity(self, df: pd.DataFrame, column_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Assess data validity (format and type correctness)"""
        column_validity = {}
        validity_scores = []
        
        for column in df.columns:
            validity_info = self._validate_column_format(df[column], column, column_types)
            column_validity[column] = validity_info
            validity_scores.append(validity_info['validity_ratio'])
        
        overall_validity = np.mean(validity_scores) if validity_scores else 0
        
        return {
            'score': float(overall_validity),
            'column_details': column_validity,
            'threshold_met': overall_validity >= self.quality_thresholds['validity']
        }
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency (standardization and format consistency)"""
        column_consistency = {}
        consistency_scores = []
        
        for column in df.columns:
            consistency_info = self._check_column_consistency(df[column], column)
            column_consistency[column] = consistency_info
            consistency_scores.append(consistency_info['consistency_ratio'])
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0
        
        return {
            'score': float(overall_consistency),
            'column_details': column_consistency,
            'threshold_met': overall_consistency >= self.quality_thresholds['consistency']
        }
    
    def _assess_accuracy(self, df: pd.DataFrame, column_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Assess data accuracy (domain-specific validation)"""
        column_accuracy = {}
        accuracy_scores = []
        
        for column in df.columns:
            accuracy_info = self._validate_column_accuracy(df[column], column, column_types)
            column_accuracy[column] = accuracy_info
            accuracy_scores.append(accuracy_info['accuracy_ratio'])
        
        overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
        
        return {
            'score': float(overall_accuracy),
            'column_details': column_accuracy,
            'threshold_met': overall_accuracy >= self.quality_thresholds['accuracy']
        }
    
    def _validate_column_format(self, series: pd.Series, column_name: str, column_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Validate column format and data types"""
        non_null_count = series.count()
        if non_null_count == 0:
            return {'validity_ratio': 1.0, 'issues': [], 'valid_count': 0, 'invalid_count': 0}
        
        issues = []
        invalid_count = 0
        
        # Detect column type if not provided
        expected_type = column_types.get(column_name) if column_types else self._detect_column_type(series)
        
        if expected_type == 'email':
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = series.dropna()[~series.dropna().astype(str).str.match(email_pattern)]
            invalid_count += len(invalid_emails)
            if len(invalid_emails) > 0:
                issues.append(f"Invalid email format in {len(invalid_emails)} entries")
        
        elif expected_type == 'phone':
            phone_pattern = r'^[\+]?[1-9][\d]{0,15}$'
            invalid_phones = series.dropna()[~series.dropna().astype(str).str.match(phone_pattern)]
            invalid_count += len(invalid_phones)
            if len(invalid_phones) > 0:
                issues.append(f"Invalid phone format in {len(invalid_phones)} entries")
        
        elif expected_type == 'date':
            try:
                pd.to_datetime(series.dropna(), errors='coerce')
                invalid_dates = series.dropna()[pd.to_datetime(series.dropna(), errors='coerce').isnull()]
                invalid_count += len(invalid_dates)
                if len(invalid_dates) > 0:
                    issues.append(f"Invalid date format in {len(invalid_dates)} entries")
            except:
                invalid_count = non_null_count
                issues.append("Unable to parse dates")
        
        elif expected_type == 'numeric':
            try:
                pd.to_numeric(series.dropna(), errors='coerce')
                invalid_numbers = series.dropna()[pd.to_numeric(series.dropna(), errors='coerce').isnull()]
                invalid_count += len(invalid_numbers)
                if len(invalid_numbers) > 0:
                    issues.append(f"Invalid numeric format in {len(invalid_numbers)} entries")
            except:
                invalid_count = non_null_count
                issues.append("Unable to parse numbers")
        
        valid_count = non_null_count - invalid_count
        validity_ratio = valid_count / non_null_count if non_null_count > 0 else 1.0
        
        return {
            'validity_ratio': float(validity_ratio),
            'valid_count': int(valid_count),
            'invalid_count': int(invalid_count),
            'issues': issues,
            'expected_type': expected_type
        }
    
    def _check_column_consistency(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Check column consistency (standardization)"""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return {'consistency_ratio': 1.0, 'issues': []}
        
        issues = []
        inconsistent_count = 0
        
        # Check for mixed case in text columns
        if non_null_series.dtype == 'object':
            text_values = non_null_series.astype(str)
            
            # Check case consistency
            if len(text_values) > 1:
                all_upper = text_values.str.isupper().all()
                all_lower = text_values.str.islower().all()
                all_title = text_values.str.istitle().all()
                
                if not (all_upper or all_lower or all_title):
                    mixed_case_count = len(text_values) - (
                        text_values.str.isupper().sum() + 
                        text_values.str.islower().sum() + 
                        text_values.str.istitle().sum()
                    )
                    if mixed_case_count > 0:
                        inconsistent_count += mixed_case_count
                        issues.append(f"Mixed case formatting in {mixed_case_count} entries")
            
            # Check for leading/trailing whitespace
            whitespace_issues = text_values[text_values != text_values.str.strip()]
            if len(whitespace_issues) > 0:
                inconsistent_count += len(whitespace_issues)
                issues.append(f"Leading/trailing whitespace in {len(whitespace_issues)} entries")
        
        consistency_ratio = 1 - (inconsistent_count / len(non_null_series))
        
        return {
            'consistency_ratio': float(max(0, consistency_ratio)),
            'inconsistent_count': int(inconsistent_count),
            'issues': issues
        }
    
    def _validate_column_accuracy(self, series: pd.Series, column_name: str, column_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Validate column accuracy using domain-specific rules"""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return {'accuracy_ratio': 1.0, 'issues': []}
        
        issues = []
        inaccurate_count = 0
        
        # Domain-specific validation rules
        column_lower = column_name.lower()
        
        # Age validation
        if 'age' in column_lower:
            if pd.api.types.is_numeric_dtype(series):
                invalid_ages = non_null_series[(non_null_series < 0) | (non_null_series > 150)]
                inaccurate_count += len(invalid_ages)
                if len(invalid_ages) > 0:
                    issues.append(f"Unrealistic age values in {len(invalid_ages)} entries")
        
        # Percentage validation
        if 'percent' in column_lower or 'rate' in column_lower:
            if pd.api.types.is_numeric_dtype(series):
                invalid_percentages = non_null_series[(non_null_series < 0) | (non_null_series > 100)]
                inaccurate_count += len(invalid_percentages)
                if len(invalid_percentages) > 0:
                    issues.append(f"Invalid percentage values in {len(invalid_percentages)} entries")
        
        # Price/Amount validation
        if any(term in column_lower for term in ['price', 'amount', 'cost', 'salary']):
            if pd.api.types.is_numeric_dtype(series):
                negative_amounts = non_null_series[non_null_series < 0]
                inaccurate_count += len(negative_amounts)
                if len(negative_amounts) > 0:
                    issues.append(f"Negative amounts in {len(negative_amounts)} entries")
        
        accuracy_ratio = 1 - (inaccurate_count / len(non_null_series))
        
        return {
            'accuracy_ratio': float(max(0, accuracy_ratio)),
            'inaccurate_count': int(inaccurate_count),
            'issues': issues
        }
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect the likely data type of a column"""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return 'unknown'
        
        # Check if numeric
        try:
            pd.to_numeric(non_null_series)
            return 'numeric'
        except:
            pass
        
        # Check if date
        try:
            pd.to_datetime(non_null_series)
            return 'date'
        except:
            pass
        
        # Check if email
        if non_null_series.dtype == 'object':
            sample_values = non_null_series.astype(str).head(10)
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if sample_values.str.match(email_pattern).any():
                return 'email'
        
        return 'text'
    
    def _assess_column_quality(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Assess overall quality for a single column"""
        completeness = 1 - (series.isnull().sum() / len(series))
        uniqueness = series.nunique() / series.count() if series.count() > 0 else 0
        
        # Simple validity check
        validity = 1.0  # Default to valid
        if series.dtype == 'object':
            # Check for obviously invalid entries (very long strings, special characters)
            text_series = series.dropna().astype(str)
            if len(text_series) > 0:
                very_long = text_series.str.len() > 1000
                validity = 1 - (very_long.sum() / len(text_series))
        
        overall_score = np.mean([completeness, uniqueness, validity])
        
        return {
            'overall_score': float(overall_score),
            'completeness': float(completeness),
            'uniqueness': float(uniqueness),
            'validity': float(validity)
        }
    
    def _identify_issues(self, quality_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify data quality issues"""
        issues = []
        
        # Check dimension thresholds
        for dimension, data in quality_report['dimensions'].items():
            if not data.get('threshold_met', True):
                issues.append({
                    'type': 'dimension_threshold',
                    'severity': 'high',
                    'dimension': dimension,
                    'score': data['score'],
                    'threshold': self.quality_thresholds[dimension],
                    'message': f"{dimension.title()} score ({data['score']:.1%}) below threshold ({self.quality_thresholds[dimension]:.1%})"
                })
        
        # Check for columns with poor quality
        for column, quality in quality_report['column_quality'].items():
            if quality['overall_score'] < 0.7:
                issues.append({
                    'type': 'poor_column_quality',
                    'severity': 'medium',
                    'column': column,
                    'score': quality['overall_score'],
                    'message': f"Column '{column}' has poor quality score ({quality['overall_score']:.1%})"
                })
        
        return issues
    
    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Completeness recommendations
        completeness = quality_report['dimensions']['completeness']
        if completeness['score'] < 0.9:
            recommendations.append("Address missing values through imputation or data collection")
        
        # Uniqueness recommendations
        uniqueness = quality_report['dimensions']['uniqueness']
        if uniqueness['score'] < 0.9:
            recommendations.append("Remove or investigate duplicate records")
        
        # Validity recommendations
        validity = quality_report['dimensions']['validity']
        if validity['score'] < 0.9:
            recommendations.append("Standardize data formats and validate input constraints")
        
        # Consistency recommendations
        consistency = quality_report['dimensions']['consistency']
        if consistency['score'] < 0.9:
            recommendations.append("Implement data standardization rules for consistent formatting")
        
        # Overall recommendations
        if quality_report['overall_score'] < 0.8:
            recommendations.append("Consider implementing automated data quality monitoring")
            recommendations.append("Establish data governance policies and validation rules")
        
        return recommendations
    
    def _generate_summary(self, quality_report: Dict[str, Any]) -> str:
        """Generate a summary of the data quality assessment"""
        overall_score = quality_report['overall_score']
        
        if overall_score >= 0.9:
            quality_level = "Excellent"
        elif overall_score >= 0.8:
            quality_level = "Good"
        elif overall_score >= 0.7:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        issues_count = len(quality_report['issues'])
        recommendations_count = len(quality_report['recommendations'])
        
        summary = f"Data quality assessment complete. Overall quality: {quality_level} ({overall_score:.1%}). "
        summary += f"Identified {issues_count} issues with {recommendations_count} recommendations for improvement."
        
        return summary
