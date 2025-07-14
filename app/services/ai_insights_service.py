"""
AI-Powered Insights Service
Generates automated insights, natural language explanations, and AI-driven recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

from app.services.file_service import FileService
from app.services.analysis_service import AnalysisService

class AIInsightsService:
    """Service for generating AI-powered insights and recommendations"""
    
    def __init__(self):
        self.file_service = FileService()
        self.analysis_service = AnalysisService()
        
        # Insight templates for different scenarios
        self.insight_templates = {
            "high_correlation": "Strong correlation detected between {var1} and {var2} (r={correlation:.3f}). This suggests that {interpretation}.",
            "outliers_detected": "Found {count} potential outliers in {column}. These values are {description} and may indicate {possible_causes}.",
            "missing_data": "Column {column} has {percentage:.1f}% missing values. Consider {recommendations}.",
            "skewed_distribution": "{column} shows {skew_type} skewness (skew={skew:.2f}). This indicates {interpretation}.",
            "trend_detected": "Time series shows a {trend_direction} trend with R²={r_squared:.3f}. {interpretation}.",
            "cluster_insights": "Data naturally groups into {n_clusters} clusters. {interpretation}.",
            "anomaly_insights": "Detected {count} anomalies ({percentage:.1f}% of data). {interpretation}."
        }
    
    async def generate_automated_insights(self, dataset_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive automated insights for a dataset
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")
        
        insights = {
            "dataset_id": dataset_id,
            "generated_at": datetime.now().isoformat(),
            "insights": [],
            "recommendations": [],
            "data_quality_score": 0,
            "summary": ""
        }
        
        # Data quality assessment
        quality_insights = await self._assess_data_quality(df)
        insights["insights"].extend(quality_insights["insights"])
        insights["recommendations"].extend(quality_insights["recommendations"])
        insights["data_quality_score"] = quality_insights["score"]
        
        # Statistical insights
        statistical_insights = await self._generate_statistical_insights(df)
        insights["insights"].extend(statistical_insights)
        
        # Pattern detection
        pattern_insights = await self._detect_patterns(df)
        insights["insights"].extend(pattern_insights)
        
        # Relationship insights
        relationship_insights = await self._analyze_relationships(df)
        insights["insights"].extend(relationship_insights)
        
        # Generate overall summary
        insights["summary"] = await self._generate_summary(insights["insights"])
        
        return insights
    
    async def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and generate insights"""
        insights = []
        recommendations = []
        score_components = []
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_percentage = (missing_data.sum() / total_cells) * 100
        
        if missing_percentage > 0:
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    percentage = (missing_count / len(df)) * 100
                    if percentage > 20:
                        insights.append({
                            "type": "data_quality",
                            "severity": "high",
                            "message": self.insight_templates["missing_data"].format(
                                column=col, 
                                percentage=percentage,
                                recommendations="imputation, removal, or collection of additional data"
                            ),
                            "details": {"column": col, "missing_count": missing_count, "percentage": percentage}
                        })
                        recommendations.append(f"Address high missing data in {col} ({percentage:.1f}% missing)")
        
        # Data type consistency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Duplicate detection
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            insights.append({
                "type": "data_quality",
                "severity": "medium",
                "message": f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}% of data). Consider removing duplicates to improve analysis accuracy.",
                "details": {"duplicate_count": duplicate_count, "percentage": duplicate_percentage}
            })
            recommendations.append("Remove duplicate rows to improve data quality")
        
        # Calculate quality score
        missing_score = max(0, 100 - missing_percentage)
        duplicate_score = max(0, 100 - (duplicate_count / len(df)) * 100)
        consistency_score = 85  # Placeholder for more complex consistency checks
        
        overall_score = (missing_score + duplicate_score + consistency_score) / 3
        score_components = [missing_score, duplicate_score, consistency_score]
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "score": overall_score,
            "score_components": score_components
        }
    
    async def _generate_statistical_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights from statistical analysis"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return insights
        
        # Skewness analysis
        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                skew_type = "right" if skewness > 0 else "left"
                interpretation = self._interpret_skewness(skewness, col)
                
                insights.append({
                    "type": "statistical",
                    "severity": "medium" if abs(skewness) > 2 else "low",
                    "message": self.insight_templates["skewed_distribution"].format(
                        column=col,
                        skew_type=skew_type,
                        skew=skewness,
                        interpretation=interpretation
                    ),
                    "details": {"column": col, "skewness": skewness, "type": skew_type}
                })
        
        # Outlier detection using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(df)) * 100
                description = self._describe_outliers(outliers[col], df[col])
                possible_causes = self._suggest_outlier_causes(col, outlier_percentage)
                
                insights.append({
                    "type": "statistical",
                    "severity": "high" if outlier_percentage > 5 else "medium",
                    "message": self.insight_templates["outliers_detected"].format(
                        count=outlier_count,
                        column=col,
                        description=description,
                        possible_causes=possible_causes
                    ),
                    "details": {
                        "column": col,
                        "outlier_count": outlier_count,
                        "percentage": outlier_percentage,
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    }
                })
        
        return insights
    
    async def _detect_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect patterns in the data"""
        insights = []
        
        # Check for time series patterns
        date_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].head(10))
                    date_columns.append(col)
                except:
                    pass
        
        if date_columns:
            insights.append({
                "type": "pattern",
                "severity": "low",
                "message": f"Detected potential time series data in columns: {', '.join(date_columns)}. Consider time series analysis for temporal patterns.",
                "details": {"date_columns": date_columns}
            })
        
        # Check for categorical patterns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            
            if unique_ratio < 0.05:  # Low cardinality
                insights.append({
                    "type": "pattern",
                    "severity": "low",
                    "message": f"Column {col} has low cardinality ({df[col].nunique()} unique values). This suggests it's a good candidate for categorical analysis or grouping.",
                    "details": {"column": col, "unique_count": df[col].nunique(), "unique_ratio": unique_ratio}
                })
            elif unique_ratio > 0.95:  # High cardinality
                insights.append({
                    "type": "pattern",
                    "severity": "medium",
                    "message": f"Column {col} has very high cardinality ({df[col].nunique()} unique values). Consider if this is an identifier or needs preprocessing.",
                    "details": {"column": col, "unique_count": df[col].nunique(), "unique_ratio": unique_ratio}
                })
        
        return insights
    
    async def _analyze_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze relationships between variables"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return insights
        
        # Correlation analysis
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > 0.7:
                    var1, var2 = numeric_cols[i], numeric_cols[j]
                    interpretation = self._interpret_correlation(correlation, var1, var2)
                    
                    insights.append({
                        "type": "relationship",
                        "severity": "high" if abs(correlation) > 0.9 else "medium",
                        "message": self.insight_templates["high_correlation"].format(
                            var1=var1,
                            var2=var2,
                            correlation=correlation,
                            interpretation=interpretation
                        ),
                        "details": {
                            "variable1": var1,
                            "variable2": var2,
                            "correlation": correlation,
                            "strength": "very strong" if abs(correlation) > 0.9 else "strong"
                        }
                    })
        
        return insights
    
    def _interpret_skewness(self, skewness: float, column: str) -> str:
        """Interpret skewness values"""
        if skewness > 2:
            return f"most {column} values are concentrated at lower levels with few high values"
        elif skewness > 1:
            return f"{column} distribution has a longer tail towards higher values"
        elif skewness < -2:
            return f"most {column} values are concentrated at higher levels with few low values"
        elif skewness < -1:
            return f"{column} distribution has a longer tail towards lower values"
        else:
            return f"{column} distribution is approximately symmetric"
    
    def _describe_outliers(self, outlier_values: pd.Series, all_values: pd.Series) -> str:
        """Describe outlier characteristics"""
        outlier_mean = outlier_values.mean()
        data_mean = all_values.mean()
        
        if outlier_mean > data_mean:
            return "significantly higher than typical values"
        else:
            return "significantly lower than typical values"
    
    def _suggest_outlier_causes(self, column: str, percentage: float) -> str:
        """Suggest possible causes for outliers"""
        if percentage > 10:
            return "data entry errors, measurement issues, or genuine extreme cases"
        elif percentage > 5:
            return "measurement errors or rare but valid extreme cases"
        else:
            return "natural variation or special circumstances"
    
    def _interpret_correlation(self, correlation: float, var1: str, var2: str) -> str:
        """Interpret correlation between variables"""
        strength = "very strong" if abs(correlation) > 0.9 else "strong"
        direction = "positive" if correlation > 0 else "negative"
        
        if direction == "positive":
            return f"as {var1} increases, {var2} tends to increase as well"
        else:
            return f"as {var1} increases, {var2} tends to decrease"
    
    async def _generate_summary(self, insights: List[Dict[str, Any]]) -> str:
        """Generate an overall summary of insights"""
        if not insights:
            return "No significant insights detected in the dataset."
        
        high_severity = len([i for i in insights if i["severity"] == "high"])
        medium_severity = len([i for i in insights if i["severity"] == "medium"])
        low_severity = len([i for i in insights if i["severity"] == "low"])
        
        summary_parts = []
        
        if high_severity > 0:
            summary_parts.append(f"{high_severity} critical insight(s) requiring immediate attention")
        
        if medium_severity > 0:
            summary_parts.append(f"{medium_severity} important insight(s) for consideration")
        
        if low_severity > 0:
            summary_parts.append(f"{low_severity} informational insight(s)")
        
        summary = f"Analysis revealed {', '.join(summary_parts)}. "
        
        # Add specific recommendations based on insight types
        insight_types = [i["type"] for i in insights]
        if "data_quality" in insight_types:
            summary += "Data quality improvements are recommended. "
        if "relationship" in insight_types:
            summary += "Strong relationships between variables were detected. "
        if "statistical" in insight_types:
            summary += "Statistical anomalies require investigation. "
        
        return summary
    
    async def explain_analysis_result(self, analysis_result: Dict[str, Any], analysis_type: str) -> str:
        """
        Generate natural language explanation for analysis results
        """
        explanations = {
            "descriptive": self._explain_descriptive_analysis,
            "correlation": self._explain_correlation_analysis,
            "regression": self._explain_regression_analysis,
            "classification": self._explain_classification_analysis,
            "clustering": self._explain_clustering_analysis,
            "anomaly_detection": self._explain_anomaly_detection,
            "time_series": self._explain_time_series_analysis
        }
        
        if analysis_type in explanations:
            return explanations[analysis_type](analysis_result)
        else:
            return f"Analysis of type '{analysis_type}' completed successfully."
    
    def _explain_descriptive_analysis(self, result: Dict[str, Any]) -> str:
        """Explain descriptive analysis results"""
        stats = result.get("results", {}).get("descriptive_statistics", {})
        if not stats:
            return "Descriptive analysis completed but no statistics available."
        
        explanations = []
        for column, column_stats in stats.items():
            mean_val = column_stats.get("mean", 0)
            std_val = column_stats.get("std", 0)
            
            if std_val > 0:
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                if cv > 50:
                    explanations.append(f"{column} shows high variability (CV={cv:.1f}%)")
                elif cv < 10:
                    explanations.append(f"{column} shows low variability (CV={cv:.1f}%)")
        
        if explanations:
            return "Key findings: " + "; ".join(explanations) + "."
        else:
            return "Descriptive statistics calculated for all numeric variables."
    
    def _explain_correlation_analysis(self, result: Dict[str, Any]) -> str:
        """Explain correlation analysis results"""
        strong_corr = result.get("results", {}).get("strong_correlations", [])
        
        if not strong_corr:
            return "No strong correlations (>0.7) were found between variables, suggesting they are relatively independent."
        
        explanations = []
        for corr in strong_corr[:3]:  # Top 3
            var1, var2 = corr["variable1"], corr["variable2"]
            corr_val = corr["correlation"]
            direction = "positively" if corr_val > 0 else "negatively"
            strength = "very strongly" if abs(corr_val) > 0.9 else "strongly"
            
            explanations.append(f"{var1} and {var2} are {strength} {direction} correlated (r={corr_val:.3f})")
        
        return "Key correlations found: " + "; ".join(explanations) + "."
    
    def _explain_regression_analysis(self, result: Dict[str, Any]) -> str:
        """Explain regression analysis results"""
        results = result.get("results", {})
        best_model = None
        best_r2 = 0
        
        for model_name, model_results in results.items():
            r2 = model_results.get("r2_score", 0)
            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name
        
        if best_model:
            performance = "excellent" if best_r2 > 0.8 else "good" if best_r2 > 0.6 else "moderate" if best_r2 > 0.4 else "poor"
            return f"The {best_model} model shows {performance} performance with R²={best_r2:.3f}, explaining {best_r2*100:.1f}% of the variance in the target variable."
        
        return "Regression analysis completed but model performance information is not available."
    
    def _explain_classification_analysis(self, result: Dict[str, Any]) -> str:
        """Explain classification analysis results"""
        results = result.get("results", {})
        best_model = None
        best_accuracy = 0
        
        for model_name, model_results in results.items():
            accuracy = model_results.get("accuracy", 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        if best_model:
            performance = "excellent" if best_accuracy > 0.9 else "good" if best_accuracy > 0.8 else "moderate" if best_accuracy > 0.7 else "poor"
            return f"The {best_model} classifier achieves {performance} performance with {best_accuracy*100:.1f}% accuracy."
        
        return "Classification analysis completed but accuracy information is not available."
    
    def _explain_clustering_analysis(self, result: Dict[str, Any]) -> str:
        """Explain clustering analysis results"""
        results = result.get("results", {})
        n_clusters = results.get("n_clusters", 0)
        
        if n_clusters > 0:
            return f"The data naturally groups into {n_clusters} distinct clusters, suggesting {n_clusters} different patterns or segments in your dataset."
        
        return "Clustering analysis completed but cluster information is not available."
    
    def _explain_anomaly_detection(self, result: Dict[str, Any]) -> str:
        """Explain anomaly detection results"""
        results = result.get("results", {})
        anomaly_count = results.get("anomalies_detected", 0)
        total_points = results.get("total_points", 0)
        percentage = results.get("anomaly_percentage", 0)
        
        if anomaly_count > 0:
            severity = "high" if percentage > 10 else "moderate" if percentage > 5 else "low"
            return f"Detected {anomaly_count} anomalies ({percentage:.1f}% of data) with {severity} concentration. These outliers may represent data errors, rare events, or interesting patterns worth investigating."
        
        return "No significant anomalies detected. The data appears to be consistent and well-behaved."
    
    def _explain_time_series_analysis(self, result: Dict[str, Any]) -> str:
        """Explain time series analysis results"""
        results = result.get("results", {})
        trend = results.get("trend_analysis", {})
        trend_direction = trend.get("trend_direction", "stable")
        r_squared = trend.get("r_squared", 0)
        
        if trend_direction != "stable":
            strength = "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "weak"
            return f"The time series shows a {strength} {trend_direction} trend (R²={r_squared:.3f}), indicating a clear temporal pattern in the data."
        
        return "The time series appears stable with no significant trend detected."
