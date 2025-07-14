"""
AI-Powered Insights Engine
Generates intelligent insights and recommendations from analysis results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import json
from datetime import datetime


class InsightsEngine:
    """
    AI-powered insights generation engine that provides intelligent
    interpretations and recommendations based on analysis results.
    """
    
    def __init__(self):
        self.insight_templates = self._load_insight_templates()
        self.thresholds = self._load_thresholds()
    
    def generate_insights(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive insights from analysis results
        """
        analysis_type = analysis_result.get('analysis_type', '')
        results = analysis_result.get('results', {})
        
        insights = {
            'key_findings': [],
            'recommendations': [],
            'alerts': [],
            'confidence_score': 0.0,
            'narrative': '',
            'next_steps': []
        }
        
        if analysis_type == 'descriptive':
            insights = self._generate_descriptive_insights(results, insights)
        elif analysis_type == 'correlation':
            insights = self._generate_correlation_insights(results, insights)
        elif analysis_type == 'clustering':
            insights = self._generate_clustering_insights(results, insights)
        elif analysis_type == 'forecasting':
            insights = self._generate_forecasting_insights(results, insights)
        elif analysis_type == 'pca':
            insights = self._generate_pca_insights(results, insights)
        
        # Generate overall narrative
        insights['narrative'] = self._generate_narrative(analysis_type, insights)
        insights['confidence_score'] = self._calculate_confidence_score(results, insights)
        
        return insights
    
    def _generate_descriptive_insights(self, results: Dict, insights: Dict) -> Dict:
        """Generate insights for descriptive analysis"""
        stats = results.get('statistics', {})
        
        for column, column_stats in stats.items():
            if isinstance(column_stats, dict):
                mean_val = column_stats.get('mean', 0)
                std_val = column_stats.get('std', 0)
                skewness = column_stats.get('skewness', 0)
                outliers = column_stats.get('outliers', [])
                
                # Key findings
                if abs(skewness) > 1:
                    skew_direction = "right" if skewness > 0 else "left"
                    insights['key_findings'].append(
                        f"{column} shows significant {skew_direction}-skewed distribution (skewness: {skewness:.2f})"
                    )
                
                if len(outliers) > 0:
                    outlier_percentage = (len(outliers) / column_stats.get('count', 1)) * 100
                    insights['key_findings'].append(
                        f"{column} contains {len(outliers)} outliers ({outlier_percentage:.1f}% of data)"
                    )
                    
                    if outlier_percentage > 5:
                        insights['alerts'].append({
                            'type': 'warning',
                            'message': f"High outlier percentage in {column} may indicate data quality issues"
                        })
                
                # Coefficient of variation
                if mean_val != 0:
                    cv = (std_val / abs(mean_val)) * 100
                    if cv > 50:
                        insights['key_findings'].append(
                            f"{column} shows high variability (CV: {cv:.1f}%)"
                        )
                        insights['recommendations'].append(
                            f"Consider investigating the high variability in {column} - it may indicate different data segments"
                        )
        
        # General recommendations
        insights['recommendations'].extend([
            "Review outliers to ensure data quality",
            "Consider data transformation for skewed distributions",
            "Investigate high-variability columns for potential segmentation opportunities"
        ])
        
        insights['next_steps'].extend([
            "Perform correlation analysis to identify relationships",
            "Consider clustering analysis to find data segments",
            "Apply appropriate transformations for skewed data"
        ])
        
        return insights
    
    def _generate_correlation_insights(self, results: Dict, insights: Dict) -> Dict:
        """Generate insights for correlation analysis"""
        correlations = results.get('correlations', {})
        
        strong_correlations = []
        moderate_correlations = []
        
        for pair, corr_data in correlations.items():
            if isinstance(corr_data, dict):
                corr_value = corr_data.get('correlation', 0)
                p_value = corr_data.get('p_value', 1)
                
                if abs(corr_value) >= 0.7 and p_value < 0.05:
                    strong_correlations.append((pair, corr_value))
                    insights['key_findings'].append(
                        f"Strong {'positive' if corr_value > 0 else 'negative'} correlation between {pair}: {corr_value:.3f}"
                    )
                elif abs(corr_value) >= 0.3 and p_value < 0.05:
                    moderate_correlations.append((pair, corr_value))
        
        if strong_correlations:
            insights['recommendations'].append(
                "Strong correlations detected - consider these relationships in predictive modeling"
            )
            insights['alerts'].append({
                'type': 'info',
                'message': f"Found {len(strong_correlations)} strong correlations that may indicate causal relationships"
            })
        
        if len(strong_correlations) > 5:
            insights['alerts'].append({
                'type': 'warning',
                'message': "Many strong correlations detected - check for multicollinearity issues"
            })
        
        insights['next_steps'].extend([
            "Investigate causal relationships for strong correlations",
            "Consider feature selection for predictive modeling",
            "Perform regression analysis for significant correlations"
        ])
        
        return insights
    
    def _generate_clustering_insights(self, results: Dict, insights: Dict) -> Dict:
        """Generate insights for clustering analysis"""
        cluster_info = results.get('cluster_info', {})
        silhouette_score = results.get('silhouette_score', 0)
        
        if silhouette_score > 0.7:
            insights['key_findings'].append(
                f"Excellent cluster separation achieved (silhouette score: {silhouette_score:.3f})"
            )
            insights['recommendations'].append(
                "Clusters are well-defined - consider using them for targeted strategies"
            )
        elif silhouette_score > 0.5:
            insights['key_findings'].append(
                f"Good cluster separation (silhouette score: {silhouette_score:.3f})"
            )
        else:
            insights['alerts'].append({
                'type': 'warning',
                'message': f"Poor cluster separation (silhouette score: {silhouette_score:.3f}) - consider different parameters"
            })
        
        if cluster_info:
            num_clusters = len(cluster_info)
            insights['key_findings'].append(f"Data naturally segments into {num_clusters} distinct groups")
            
            # Analyze cluster sizes
            cluster_sizes = [info.get('size', 0) for info in cluster_info.values()]
            if max(cluster_sizes) / min(cluster_sizes) > 10:
                insights['alerts'].append({
                    'type': 'info',
                    'message': "Highly imbalanced cluster sizes detected - one group dominates the data"
                })
        
        insights['next_steps'].extend([
            "Analyze cluster characteristics to understand segments",
            "Develop targeted strategies for each cluster",
            "Consider cluster-specific analysis and modeling"
        ])
        
        return insights
    
    def _generate_forecasting_insights(self, results: Dict, insights: Dict) -> Dict:
        """Generate insights for forecasting analysis"""
        trend_info = results.get('trend_analysis', {})
        r_squared = trend_info.get('r_squared', 0)
        slope = trend_info.get('slope', 0)
        
        if r_squared > 0.8:
            insights['key_findings'].append(
                f"Strong trend pattern detected (R² = {r_squared:.3f})"
            )
            trend_direction = "increasing" if slope > 0 else "decreasing"
            insights['key_findings'].append(f"Data shows a clear {trend_direction} trend")
        elif r_squared > 0.5:
            insights['key_findings'].append(
                f"Moderate trend pattern (R² = {r_squared:.3f})"
            )
        else:
            insights['alerts'].append({
                'type': 'warning',
                'message': f"Weak trend pattern (R² = {r_squared:.3f}) - forecasting may be unreliable"
            })
        
        if abs(slope) > 0:
            rate_of_change = abs(slope)
            insights['key_findings'].append(
                f"Rate of change: {rate_of_change:.2f} units per time period"
            )
        
        insights['recommendations'].extend([
            "Use trend analysis for strategic planning",
            "Monitor for trend changes that might affect forecasts",
            "Consider seasonal patterns if working with time series data"
        ])
        
        return insights
    
    def _generate_pca_insights(self, results: Dict, insights: Dict) -> Dict:
        """Generate insights for PCA analysis"""
        explained_variance = results.get('explained_variance_ratio', [])
        
        if explained_variance:
            cumulative_variance = np.cumsum(explained_variance)
            
            # Find number of components for 80% variance
            components_80 = np.argmax(cumulative_variance >= 0.8) + 1
            
            insights['key_findings'].append(
                f"First {components_80} components explain 80% of data variance"
            )
            
            if components_80 <= 3:
                insights['key_findings'].append(
                    "Data can be effectively reduced to 2-3 dimensions"
                )
                insights['recommendations'].append(
                    "Consider using reduced dimensions for visualization and modeling"
                )
            
            first_component_variance = explained_variance[0] * 100
            insights['key_findings'].append(
                f"Primary component explains {first_component_variance:.1f}% of variance"
            )
        
        insights['next_steps'].extend([
            "Use principal components for dimensionality reduction",
            "Analyze component loadings to understand feature importance",
            "Consider PCA for noise reduction in modeling"
        ])
        
        return insights
    
    def _generate_narrative(self, analysis_type: str, insights: Dict) -> str:
        """Generate a natural language narrative of the insights"""
        key_findings = insights.get('key_findings', [])
        recommendations = insights.get('recommendations', [])
        alerts = insights.get('alerts', [])
        
        narrative_parts = []
        
        # Introduction
        analysis_names = {
            'descriptive': 'descriptive statistical analysis',
            'correlation': 'correlation analysis',
            'clustering': 'clustering analysis',
            'forecasting': 'trend and forecasting analysis',
            'pca': 'principal component analysis'
        }
        
        analysis_name = analysis_names.get(analysis_type, 'data analysis')
        narrative_parts.append(f"Based on the {analysis_name}, several important insights emerge.")
        
        # Key findings
        if key_findings:
            narrative_parts.append("Key findings include:")
            for finding in key_findings[:3]:  # Limit to top 3 findings
                narrative_parts.append(f"• {finding}")
        
        # Alerts
        if alerts:
            warning_alerts = [alert for alert in alerts if alert.get('type') == 'warning']
            if warning_alerts:
                narrative_parts.append("Important considerations:")
                for alert in warning_alerts[:2]:  # Limit to top 2 alerts
                    narrative_parts.append(f"• {alert.get('message', '')}")
        
        # Recommendations
        if recommendations:
            narrative_parts.append("Recommended actions:")
            for rec in recommendations[:2]:  # Limit to top 2 recommendations
                narrative_parts.append(f"• {rec}")
        
        return " ".join(narrative_parts)
    
    def _calculate_confidence_score(self, results: Dict, insights: Dict) -> float:
        """Calculate confidence score for the insights"""
        score = 0.5  # Base score
        
        # Increase confidence based on data quality indicators
        if results.get('sample_size', 0) > 100:
            score += 0.1
        if results.get('sample_size', 0) > 1000:
            score += 0.1
        
        # Adjust based on statistical significance
        p_values = []
        for key, value in results.items():
            if isinstance(value, dict) and 'p_value' in value:
                p_values.append(value['p_value'])
        
        if p_values:
            significant_tests = sum(1 for p in p_values if p < 0.05)
            score += (significant_tests / len(p_values)) * 0.2
        
        # Adjust based on alerts
        warning_alerts = len([alert for alert in insights.get('alerts', []) 
                            if alert.get('type') == 'warning'])
        score -= warning_alerts * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _load_insight_templates(self) -> Dict:
        """Load insight templates for different analysis types"""
        return {
            'descriptive': {
                'high_skewness': "The data shows significant skewness, indicating an asymmetric distribution.",
                'outliers_detected': "Outliers detected that may require investigation.",
                'high_variability': "High variability suggests diverse data patterns."
            },
            'correlation': {
                'strong_positive': "Strong positive correlation suggests variables increase together.",
                'strong_negative': "Strong negative correlation indicates inverse relationship.",
                'multicollinearity': "Multiple strong correlations may indicate redundant features."
            }
        }
    
    def _load_thresholds(self) -> Dict:
        """Load statistical thresholds for insight generation"""
        return {
            'correlation': {
                'strong': 0.7,
                'moderate': 0.3,
                'weak': 0.1
            },
            'skewness': {
                'high': 1.0,
                'moderate': 0.5
            },
            'outlier_percentage': {
                'high': 5.0,
                'moderate': 2.0
            }
        }
