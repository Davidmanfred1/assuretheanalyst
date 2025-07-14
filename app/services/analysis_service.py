"""
Analysis Service
Handles statistical analysis and machine learning operations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .insights_engine import InsightsEngine

from app.models.analysis_models import AnalysisResult, AnalysisType, StatisticalTest
from app.services.file_service import FileService

class AnalysisService:
    def __init__(self):
        self.file_service = FileService()
        self.insights_engine = InsightsEngine()

    def _clean_for_json(self, obj):
        """Clean data for JSON serialization by handling NaN values and numpy types"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return self._clean_for_json(obj.fillna("null").to_dict())
        elif pd.isna(obj) or (isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj))):
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):
            # Handle numpy dtypes and other numpy objects
            if hasattr(obj, 'item'):
                return obj.item()
            else:
                return str(obj)
        else:
            return obj
    
    async def descriptive_analysis(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform descriptive statistical analysis
        
        Args:
            dataset_id: Dataset identifier
            columns: Columns to analyze
            parameters: Analysis parameters
            
        Returns:
            AnalysisResult with descriptive statistics
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            # Try to get from global file service
            from app.main import file_service as global_file_service
            df = global_file_service.get_dataframe(dataset_id)
            if df is None:
                raise ValueError(f"Dataset {dataset_id} not found")
        
        # Select specified columns or all numeric columns
        if columns:
            analysis_df = df[columns]
        else:
            analysis_df = df.select_dtypes(include=[np.number])
        
        # Calculate descriptive statistics
        desc_stats = analysis_df.describe()
        
        # Enhanced statistics
        results = {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(analysis_df.columns),
                "analyzed_columns": list(analysis_df.columns)
            },
            "descriptive_statistics": self._clean_for_json(desc_stats.to_dict()),
            "data_types": {k: str(v) for k, v in analysis_df.dtypes.to_dict().items()},
            "missing_values": self._clean_for_json(analysis_df.isnull().sum().to_dict()),
            "missing_percentage": self._clean_for_json((analysis_df.isnull().sum() / len(analysis_df) * 100).to_dict()),
            "unique_values": self._clean_for_json(analysis_df.nunique().to_dict()),
            "duplicate_rows": int(df.duplicated().sum())
        }

        # Advanced statistical measures
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results["skewness"] = self._clean_for_json(analysis_df[numeric_cols].skew().to_dict())
            results["kurtosis"] = self._clean_for_json(analysis_df[numeric_cols].kurtosis().to_dict())

            # Variance and standard error
            results["variance"] = self._clean_for_json(analysis_df[numeric_cols].var().to_dict())
            results["standard_error"] = self._clean_for_json((analysis_df[numeric_cols].std() / np.sqrt(len(analysis_df))).to_dict())

            # Confidence intervals (95%)
            confidence_intervals = {}
            for col in numeric_cols:
                mean = analysis_df[col].mean()
                std_err = analysis_df[col].std() / np.sqrt(len(analysis_df[col].dropna()))
                margin = 1.96 * std_err  # 95% confidence interval
                confidence_intervals[col] = {
                    "lower": self._clean_for_json(mean - margin),
                    "upper": self._clean_for_json(mean + margin)
                }
            results["confidence_intervals_95"] = confidence_intervals

            # Outlier detection using IQR method
            outliers = {}
            for col in numeric_cols:
                Q1 = analysis_df[col].quantile(0.25)
                Q3 = analysis_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (analysis_df[col] < lower_bound) | (analysis_df[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                outliers[col] = {
                    "count": int(outlier_count),
                    "percentage": float(outlier_count / len(analysis_df[col].dropna()) * 100),
                    "lower_bound": self._clean_for_json(lower_bound),
                    "upper_bound": self._clean_for_json(upper_bound)
                }

            results["outliers"] = outliers

        # Generate visualization data
        results["visualizations"] = self._generate_descriptive_charts(analysis_df)
        
        summary = f"Enhanced descriptive analysis completed for {len(analysis_df.columns)} columns and {len(analysis_df)} rows with {len(results.get('outliers', {}))} outlier analyses."

        # Generate AI insights
        analysis_result_dict = {
            'analysis_type': 'descriptive',
            'results': results,
            'summary': summary
        }
        insights = self.insights_engine.generate_insights(analysis_result_dict)

        return AnalysisResult(
            analysis_type=AnalysisType.DESCRIPTIVE,
            results=results,
            summary=summary,
            recommendations=self._generate_descriptive_recommendations(results),
            insights=insights
        )

    def _generate_descriptive_charts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate chart data for descriptive analysis"""
        charts = {}

        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            # Histogram data for each numeric column
            histograms = {}
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    hist, bin_edges = np.histogram(data, bins=20)
                    histograms[col] = {
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                        "chart_data": {
                            "data": [{
                                "x": bin_edges[:-1].tolist(),
                                "y": hist.tolist(),
                                "type": "bar",
                                "name": f"{col} Distribution",
                                "marker": {"color": "#667eea"}
                            }],
                            "layout": {
                                "title": f"Distribution of {col}",
                                "xaxis": {"title": col},
                                "yaxis": {"title": "Frequency"},
                                "template": "plotly_white"
                            }
                        }
                    }

            charts["histograms"] = histograms

            # Box plot data
            if len(numeric_cols) > 0:
                box_data = []
                for col in numeric_cols:
                    data = df[col].dropna()
                    if len(data) > 0:
                        box_data.append({
                            "y": data.tolist(),
                            "type": "box",
                            "name": col,
                            "boxpoints": "outliers"
                        })

                charts["box_plot"] = {
                    "data": box_data,
                    "layout": {
                        "title": "Box Plot - All Numeric Columns",
                        "yaxis": {"title": "Values"},
                        "template": "plotly_white"
                    }
                }

            # Correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                charts["correlation_heatmap"] = {
                    "data": [{
                        "z": corr_matrix.values.tolist(),
                        "x": corr_matrix.columns.tolist(),
                        "y": corr_matrix.columns.tolist(),
                        "type": "heatmap",
                        "colorscale": "RdBu",
                        "zmid": 0
                    }],
                    "layout": {
                        "title": "Correlation Matrix",
                        "template": "plotly_white"
                    }
                }

            # Summary statistics bar chart
            if len(numeric_cols) > 0:
                means = df[numeric_cols].mean()
                stds = df[numeric_cols].std()

                charts["summary_stats"] = {
                    "data": [
                        {
                            "x": means.index.tolist(),
                            "y": means.values.tolist(),
                            "type": "bar",
                            "name": "Mean",
                            "marker": {"color": "#667eea"}
                        },
                        {
                            "x": stds.index.tolist(),
                            "y": stds.values.tolist(),
                            "type": "bar",
                            "name": "Standard Deviation",
                            "marker": {"color": "#f093fb"},
                            "yaxis": "y2"
                        }
                    ],
                    "layout": {
                        "title": "Summary Statistics",
                        "xaxis": {"title": "Columns"},
                        "yaxis": {"title": "Mean", "side": "left"},
                        "yaxis2": {"title": "Standard Deviation", "side": "right", "overlaying": "y"},
                        "template": "plotly_white"
                    }
                }

        except Exception as e:
            logger.warning(f"Error generating charts: {e}")
            charts["error"] = str(e)

        return charts
    
    async def correlation_analysis(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform correlation analysis
        
        Args:
            dataset_id: Dataset identifier
            columns: Columns to analyze
            parameters: Analysis parameters
            
        Returns:
            AnalysisResult with correlation matrix
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Select numeric columns
        if columns:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation analysis")
        
        # Calculate multiple correlation methods
        pearson_corr = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')
        kendall_corr = numeric_df.corr(method='kendall')

        # Find correlations by strength
        correlations_by_strength = {
            "strong": [],      # |r| > 0.7
            "moderate": [],    # 0.3 < |r| <= 0.7
            "weak": [],        # 0.1 < |r| <= 0.3
            "very_weak": []    # |r| <= 0.1
        }

        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_value = pearson_corr.iloc[i, j]
                abs_corr = abs(corr_value)

                correlation_info = {
                    "variable1": pearson_corr.columns[i],
                    "variable2": pearson_corr.columns[j],
                    "pearson": self._clean_for_json(corr_value),
                    "spearman": self._clean_for_json(spearman_corr.iloc[i, j]),
                    "kendall": self._clean_for_json(kendall_corr.iloc[i, j]),
                    "strength": ""
                }

                if abs_corr > 0.7:
                    correlation_info["strength"] = "strong"
                    correlations_by_strength["strong"].append(correlation_info)
                elif abs_corr > 0.3:
                    correlation_info["strength"] = "moderate"
                    correlations_by_strength["moderate"].append(correlation_info)
                elif abs_corr > 0.1:
                    correlation_info["strength"] = "weak"
                    correlations_by_strength["weak"].append(correlation_info)
                else:
                    correlation_info["strength"] = "very_weak"
                    correlations_by_strength["very_weak"].append(correlation_info)

        # Calculate p-values for significance testing
        p_values = self._calculate_correlation_pvalues(numeric_df)

        results = {
            "correlation_matrices": {
                "pearson": self._clean_for_json(pearson_corr.to_dict()),
                "spearman": self._clean_for_json(spearman_corr.to_dict()),
                "kendall": self._clean_for_json(kendall_corr.to_dict())
            },
            "correlations_by_strength": correlations_by_strength,
            "p_values": p_values,
            "correlation_summary": {
                "total_pairs": len(pearson_corr.columns) * (len(pearson_corr.columns) - 1) // 2,
                "strong_correlations": len(correlations_by_strength["strong"]),
                "moderate_correlations": len(correlations_by_strength["moderate"]),
                "weak_correlations": len(correlations_by_strength["weak"]),
                "max_correlation": self._clean_for_json(pearson_corr.max().max()),
                "min_correlation": self._clean_for_json(pearson_corr.min().min()),
                "mean_absolute_correlation": self._clean_for_json(pearson_corr.abs().mean().mean())
            },
            "visualizations": self._generate_correlation_charts(pearson_corr, spearman_corr, numeric_df)
        }
        
        summary = f"Enhanced correlation analysis completed for {len(numeric_df.columns)} variables. Found {len(correlations_by_strength['strong'])} strong, {len(correlations_by_strength['moderate'])} moderate correlations."

        # Generate AI insights
        analysis_result_dict = {
            'analysis_type': 'correlation',
            'results': results,
            'summary': summary
        }
        insights = self.insights_engine.generate_insights(analysis_result_dict)

        return AnalysisResult(
            analysis_type=AnalysisType.CORRELATION,
            results=results,
            summary=summary,
            timestamp=datetime.now(),
            insights=insights
        )

    def _calculate_correlation_pvalues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate p-values for correlation significance"""
        try:
            from scipy.stats import pearsonr

            p_values = {}
            columns = df.columns

            for i, col1 in enumerate(columns):
                p_values[col1] = {}
                for j, col2 in enumerate(columns):
                    if i != j:
                        # Remove NaN values for correlation calculation
                        data1 = df[col1].dropna()
                        data2 = df[col2].dropna()

                        # Find common indices
                        common_idx = data1.index.intersection(data2.index)
                        if len(common_idx) > 2:
                            corr, p_val = pearsonr(df.loc[common_idx, col1], df.loc[common_idx, col2])
                            p_values[col1][col2] = self._clean_for_json(p_val)
                        else:
                            p_values[col1][col2] = None
                    else:
                        p_values[col1][col2] = 0.0  # Perfect correlation with itself

            return p_values

        except ImportError:
            logger.warning("scipy not available for p-value calculation")
            return {}
        except Exception as e:
            logger.warning(f"Error calculating p-values: {e}")
            return {}

    def _generate_correlation_charts(self, pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation visualization charts"""
        charts = {}

        try:
            # Correlation heatmap (Pearson)
            charts["pearson_heatmap"] = {
                "data": [{
                    "z": pearson_corr.values.tolist(),
                    "x": pearson_corr.columns.tolist(),
                    "y": pearson_corr.columns.tolist(),
                    "type": "heatmap",
                    "colorscale": "RdBu",
                    "zmid": 0,
                    "text": pearson_corr.round(3).values.tolist(),
                    "texttemplate": "%{text}",
                    "textfont": {"size": 10}
                }],
                "layout": {
                    "title": "Pearson Correlation Matrix",
                    "template": "plotly_white",
                    "width": 600,
                    "height": 600
                }
            }

            # Spearman correlation heatmap
            charts["spearman_heatmap"] = {
                "data": [{
                    "z": spearman_corr.values.tolist(),
                    "x": spearman_corr.columns.tolist(),
                    "y": spearman_corr.columns.tolist(),
                    "type": "heatmap",
                    "colorscale": "RdBu",
                    "zmid": 0,
                    "text": spearman_corr.round(3).values.tolist(),
                    "texttemplate": "%{text}",
                    "textfont": {"size": 10}
                }],
                "layout": {
                    "title": "Spearman Correlation Matrix",
                    "template": "plotly_white",
                    "width": 600,
                    "height": 600
                }
            }

            # Scatter plots for strong correlations
            scatter_plots = {}
            for i in range(len(pearson_corr.columns)):
                for j in range(i+1, len(pearson_corr.columns)):
                    corr_value = pearson_corr.iloc[i, j]
                    if abs(corr_value) > 0.5:  # Show scatter for moderate+ correlations
                        col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]

                        scatter_plots[f"{col1}_vs_{col2}"] = {
                            "data": [{
                                "x": df[col1].tolist(),
                                "y": df[col2].tolist(),
                                "type": "scatter",
                                "mode": "markers",
                                "marker": {
                                    "color": "#667eea",
                                    "size": 6,
                                    "opacity": 0.7
                                },
                                "name": f"r = {corr_value:.3f}"
                            }],
                            "layout": {
                                "title": f"{col1} vs {col2} (r = {corr_value:.3f})",
                                "xaxis": {"title": col1},
                                "yaxis": {"title": col2},
                                "template": "plotly_white"
                            }
                        }

            charts["scatter_plots"] = scatter_plots

        except Exception as e:
            logger.warning(f"Error generating correlation charts: {e}")
            charts["error"] = str(e)

        return charts
    
    async def regression_analysis(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform regression analysis
        
        Args:
            dataset_id: Dataset identifier
            columns: Feature columns
            parameters: Analysis parameters including target_column
            
        Returns:
            AnalysisResult with regression results
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")
        
        target_column = parameters.get("target_column")
        if not target_column:
            raise ValueError("Target column must be specified for regression analysis")
        
        # Prepare features and target
        X = df[columns].select_dtypes(include=[np.number])
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[model_name] = {
                "mse": mean_squared_error(y_test, y_pred),
                "r2_score": r2_score(y_test, y_pred),
                "feature_importance": dict(zip(X.columns, getattr(model, 'feature_importances_', [0]*len(X.columns))))
            }
        
        summary = f"Regression analysis completed using {len(columns)} features to predict {target_column}."
        
        return AnalysisResult(
            analysis_type=AnalysisType.REGRESSION,
            results=results,
            summary=summary,
            recommendations=self._generate_regression_recommendations(results)
        )

    async def classification_analysis(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform classification analysis
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")

        target_column = parameters.get("target_column")
        if not target_column:
            raise ValueError("Target column must be specified for classification analysis")

        # Prepare features and target
        X = df[columns].select_dtypes(include=[np.number])
        y = df[target_column]

        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Handle missing values
        X = X.fillna(X.mean())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        models = {
            "logistic_regression": LogisticRegression(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "feature_importance": dict(zip(X.columns, getattr(model, 'feature_importances_', [0]*len(X.columns))))
            }

        summary = f"Classification analysis completed using {len(columns)} features to predict {target_column}."

        return AnalysisResult(
            analysis_type=AnalysisType.CLASSIFICATION,
            results=results,
            summary=summary,
            recommendations=self._generate_classification_recommendations(results)
        )

    async def clustering_analysis(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform clustering analysis
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")

        # Prepare data
        X = df[columns].select_dtypes(include=[np.number])
        X = X.fillna(X.mean())

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal number of clusters using elbow method
        n_clusters = parameters.get("n_clusters", 3)
        inertias = []
        K_range = range(1, min(11, len(X)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = X[cluster_labels == i]
            cluster_stats[f"cluster_{i}"] = {
                "size": len(cluster_data),
                "mean": cluster_data.mean().to_dict(),
                "std": cluster_data.std().to_dict()
            }

        results = {
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": kmeans.inertia_,
            "cluster_statistics": cluster_stats,
            "elbow_data": {"k_values": list(K_range), "inertias": inertias}
        }

        summary = f"Clustering analysis completed with {n_clusters} clusters using {len(columns)} features."

        return AnalysisResult(
            analysis_type=AnalysisType.CLUSTERING,
            results=results,
            summary=summary,
            recommendations=self._generate_clustering_recommendations(results)
        )

    async def statistical_test(self, dataset_id: str, test_type: StatisticalTest, columns: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical tests
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")

        if test_type == StatisticalTest.T_TEST:
            return self._perform_t_test(df, columns, parameters)
        elif test_type == StatisticalTest.CHI_SQUARE:
            return self._perform_chi_square_test(df, columns, parameters)
        elif test_type == StatisticalTest.ANOVA:
            return self._perform_anova_test(df, columns, parameters)
        elif test_type == StatisticalTest.NORMALITY:
            return self._perform_normality_test(df, columns, parameters)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

    async def get_dataset_summary(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get comprehensive dataset summary
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        summary = {
            "basic_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "memory_usage": df.memory_usage(deep=True).sum()
            },
            "missing_data": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict()
        }

        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()

        if len(categorical_cols) > 0:
            summary["categorical_summary"] = {}
            for col in categorical_cols:
                summary["categorical_summary"][col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }

        return summary

    # Helper methods for recommendations
    def _generate_descriptive_recommendations(self, results: Dict[str, Any]) -> List[str]:
        recommendations = []

        # Check for missing values
        missing_values = results.get("missing_values", {})
        high_missing = [col for col, count in missing_values.items() if count > 0]
        if high_missing:
            recommendations.append(f"Consider handling missing values in columns: {', '.join(high_missing)}")

        # Check for skewness
        skewness = results.get("skewness", {})
        highly_skewed = [col for col, skew in skewness.items() if abs(skew) > 2]
        if highly_skewed:
            recommendations.append(f"Consider transforming highly skewed columns: {', '.join(highly_skewed)}")

        return recommendations

    def _generate_correlation_recommendations(self, strong_correlations: List[Dict]) -> List[str]:
        recommendations = []

        if strong_correlations:
            recommendations.append("Strong correlations detected. Consider multicollinearity in modeling.")
            for corr in strong_correlations[:3]:  # Top 3
                recommendations.append(f"High correlation between {corr['variable1']} and {corr['variable2']} ({corr['correlation']:.3f})")
        else:
            recommendations.append("No strong correlations found. Variables appear to be independent.")

        return recommendations

    async def advanced_clustering_analysis(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform advanced clustering analysis with multiple algorithms
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Select numeric columns for clustering
        if columns:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found for clustering")

        # Remove missing values
        numeric_df = numeric_df.dropna()

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        results = {}

        # K-Means Clustering
        try:
            # Determine optimal number of clusters using elbow method
            inertias = []
            k_range = range(2, min(11, len(numeric_df)))

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)

            # Use elbow method to find optimal k
            optimal_k = self._find_elbow_point(list(k_range), inertias)

            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(scaled_data)

            results["kmeans"] = {
                "optimal_clusters": optimal_k,
                "cluster_labels": kmeans_labels.tolist(),
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "inertia": self._clean_for_json(kmeans.inertia_),
                "silhouette_score": self._clean_for_json(self._calculate_silhouette_score(scaled_data, kmeans_labels))
            }

        except Exception as e:
            logger.warning(f"K-Means clustering failed: {e}")
            results["kmeans"] = {"error": str(e)}

        # DBSCAN Clustering
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(scaled_data)

            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = list(dbscan_labels).count(-1)

            results["dbscan"] = {
                "n_clusters": n_clusters,
                "n_noise_points": n_noise,
                "cluster_labels": dbscan_labels.tolist(),
                "silhouette_score": self._clean_for_json(self._calculate_silhouette_score(scaled_data, dbscan_labels)) if n_clusters > 1 else None
            }

        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")
            results["dbscan"] = {"error": str(e)}

        # Generate visualizations
        try:
            results["visualizations"] = self._generate_clustering_charts(scaled_data, numeric_df.columns.tolist(), results)
        except Exception as e:
            logger.warning(f"Error generating clustering visualizations: {e}")
            results["visualizations"] = {"error": str(e)}

        summary = f"Advanced clustering analysis completed. K-Means found {results.get('kmeans', {}).get('optimal_clusters', 'N/A')} clusters, DBSCAN found {results.get('dbscan', {}).get('n_clusters', 'N/A')} clusters."

        return AnalysisResult(
            analysis_type=AnalysisType.ADVANCED_CLUSTERING,
            results=results,
            summary=summary,
            timestamp=datetime.now()
        )

    async def forecasting_analysis(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform time series forecasting analysis
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        # For now, implement a simple trend analysis
        # In a full implementation, you'd use libraries like statsmodels or prophet

        if not columns:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                raise ValueError("No numeric columns found for forecasting")
            columns = numeric_columns[:1]  # Use first numeric column

        results = {}

        for column in columns:
            if column not in df.columns:
                continue

            series = df[column].dropna()
            if len(series) < 10:
                results[column] = {"error": "Insufficient data points for forecasting"}
                continue

            # Simple trend analysis
            x = np.arange(len(series))
            y = series.values

            # Linear trend
            coeffs = np.polyfit(x, y, 1)
            trend_line = np.poly1d(coeffs)

            # Calculate trend statistics
            slope = coeffs[0]
            r_squared = np.corrcoef(x, y)[0, 1] ** 2

            # Simple forecast (extend trend)
            forecast_periods = parameters.get('forecast_periods', 10)
            future_x = np.arange(len(series), len(series) + forecast_periods)
            forecast = trend_line(future_x)

            results[column] = {
                "trend_slope": self._clean_for_json(slope),
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "r_squared": self._clean_for_json(r_squared),
                "forecast": forecast.tolist(),
                "forecast_periods": forecast_periods,
                "original_values": y.tolist()
            }

        # Generate visualizations
        try:
            results["visualizations"] = self._generate_forecasting_charts(df, columns, results)
        except Exception as e:
            logger.warning(f"Error generating forecasting visualizations: {e}")
            results["visualizations"] = {"error": str(e)}

        summary = f"Forecasting analysis completed for {len([c for c in columns if c in results and 'error' not in results[c]])} variables."

        return AnalysisResult(
            analysis_type=AnalysisType.FORECASTING,
            results=results,
            summary=summary,
            timestamp=datetime.now()
        )

    async def dimensionality_reduction_analysis(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform dimensionality reduction analysis using PCA
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Select numeric columns
        if columns:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found for dimensionality reduction")

        if numeric_df.shape[1] < 2:
            raise ValueError("At least 2 numeric columns required for dimensionality reduction")

        # Remove missing values
        numeric_df = numeric_df.dropna()

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        results = {}

        # PCA Analysis
        try:
            # Determine optimal number of components
            n_components = min(numeric_df.shape[1], numeric_df.shape[0] - 1)
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)

            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

            # Find number of components for 95% variance
            components_95 = np.argmax(cumulative_variance >= 0.95) + 1

            results["pca"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": cumulative_variance.tolist(),
                "components_for_95_variance": int(components_95),
                "total_components": int(n_components),
                "principal_components": pca_result.tolist(),
                "feature_loadings": pca.components_.tolist(),
                "original_features": numeric_df.columns.tolist()
            }

        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            results["pca"] = {"error": str(e)}

        # Generate visualizations
        try:
            results["visualizations"] = self._generate_pca_charts(pca_result, pca.explained_variance_ratio_, numeric_df.columns.tolist())
        except Exception as e:
            logger.warning(f"Error generating PCA visualizations: {e}")
            results["visualizations"] = {"error": str(e)}

        summary = f"Dimensionality reduction completed. {components_95} components explain 95% of variance from {n_components} original features."

        return AnalysisResult(
            analysis_type=AnalysisType.DIMENSIONALITY_REDUCTION,
            results=results,
            summary=summary,
            timestamp=datetime.now()
        )

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find the elbow point in K-means clustering"""
        if len(k_values) < 3:
            return k_values[0] if k_values else 2

        # Calculate the rate of change
        diffs = np.diff(inertias)
        diff_ratios = np.diff(diffs) / diffs[:-1]

        # Find the point with maximum change in rate
        elbow_idx = np.argmax(diff_ratios) + 1
        return k_values[min(elbow_idx, len(k_values) - 1)]

    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                return silhouette_score(data, labels)
            return 0.0
        except Exception:
            return 0.0

    def _generate_clustering_charts(self, scaled_data: np.ndarray, feature_names: List[str], results: Dict) -> Dict[str, Any]:
        """Generate clustering visualization charts"""
        charts = {}

        try:
            # If we have PCA results, use first 2 components for visualization
            if scaled_data.shape[1] > 2:
                pca = PCA(n_components=2)
                plot_data = pca.fit_transform(scaled_data)
                x_label, y_label = "PC1", "PC2"
            else:
                plot_data = scaled_data
                x_label, y_label = feature_names[0], feature_names[1] if len(feature_names) > 1 else "PC2"

            # K-Means visualization
            if "kmeans" in results and "cluster_labels" in results["kmeans"]:
                kmeans_labels = results["kmeans"]["cluster_labels"]

                charts["kmeans_scatter"] = {
                    "data": [{
                        "x": plot_data[:, 0].tolist(),
                        "y": plot_data[:, 1].tolist(),
                        "mode": "markers",
                        "type": "scatter",
                        "marker": {
                            "color": kmeans_labels,
                            "colorscale": "Viridis",
                            "size": 8
                        },
                        "text": [f"Cluster {label}" for label in kmeans_labels]
                    }],
                    "layout": {
                        "title": f"K-Means Clustering ({results['kmeans']['optimal_clusters']} clusters)",
                        "xaxis": {"title": x_label},
                        "yaxis": {"title": y_label},
                        "template": "plotly_white"
                    }
                }

            # DBSCAN visualization
            if "dbscan" in results and "cluster_labels" in results["dbscan"]:
                dbscan_labels = results["dbscan"]["cluster_labels"]

                charts["dbscan_scatter"] = {
                    "data": [{
                        "x": plot_data[:, 0].tolist(),
                        "y": plot_data[:, 1].tolist(),
                        "mode": "markers",
                        "type": "scatter",
                        "marker": {
                            "color": dbscan_labels,
                            "colorscale": "Plasma",
                            "size": 8
                        },
                        "text": [f"Cluster {label}" if label != -1 else "Noise" for label in dbscan_labels]
                    }],
                    "layout": {
                        "title": f"DBSCAN Clustering ({results['dbscan']['n_clusters']} clusters)",
                        "xaxis": {"title": x_label},
                        "yaxis": {"title": y_label},
                        "template": "plotly_white"
                    }
                }

        except Exception as e:
            charts["error"] = str(e)

        return charts

    def _generate_forecasting_charts(self, df: pd.DataFrame, columns: List[str], results: Dict) -> Dict[str, Any]:
        """Generate forecasting visualization charts"""
        charts = {}

        try:
            for column in columns:
                if column not in results or "error" in results[column]:
                    continue

                result = results[column]
                original = result["original_values"]
                forecast = result["forecast"]

                # Create time series plot
                original_x = list(range(len(original)))
                forecast_x = list(range(len(original), len(original) + len(forecast)))

                charts[f"{column}_forecast"] = {
                    "data": [
                        {
                            "x": original_x,
                            "y": original,
                            "type": "scatter",
                            "mode": "lines+markers",
                            "name": "Historical Data",
                            "line": {"color": "#1f77b4"}
                        },
                        {
                            "x": forecast_x,
                            "y": forecast,
                            "type": "scatter",
                            "mode": "lines+markers",
                            "name": "Forecast",
                            "line": {"color": "#ff7f0e", "dash": "dash"}
                        }
                    ],
                    "layout": {
                        "title": f"{column} - Time Series Forecast",
                        "xaxis": {"title": "Time Period"},
                        "yaxis": {"title": column},
                        "template": "plotly_white"
                    }
                }

        except Exception as e:
            charts["error"] = str(e)

        return charts

    def _generate_pca_charts(self, pca_result: np.ndarray, explained_variance: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Generate PCA visualization charts"""
        charts = {}

        try:
            # Scree plot
            components = list(range(1, len(explained_variance) + 1))

            charts["scree_plot"] = {
                "data": [{
                    "x": components,
                    "y": explained_variance.tolist(),
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Explained Variance",
                    "line": {"color": "#1f77b4"}
                }],
                "layout": {
                    "title": "PCA Scree Plot",
                    "xaxis": {"title": "Principal Component"},
                    "yaxis": {"title": "Explained Variance Ratio"},
                    "template": "plotly_white"
                }
            }

            # Cumulative variance plot
            cumulative_variance = np.cumsum(explained_variance)

            charts["cumulative_variance"] = {
                "data": [{
                    "x": components,
                    "y": cumulative_variance.tolist(),
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Cumulative Variance",
                    "line": {"color": "#ff7f0e"}
                }],
                "layout": {
                    "title": "Cumulative Explained Variance",
                    "xaxis": {"title": "Number of Components"},
                    "yaxis": {"title": "Cumulative Variance Ratio"},
                    "template": "plotly_white"
                }
            }

            # 2D PCA plot if we have at least 2 components
            if pca_result.shape[1] >= 2:
                charts["pca_2d"] = {
                    "data": [{
                        "x": pca_result[:, 0].tolist(),
                        "y": pca_result[:, 1].tolist(),
                        "type": "scatter",
                        "mode": "markers",
                        "marker": {
                            "color": "#667eea",
                            "size": 6,
                            "opacity": 0.7
                        }
                    }],
                    "layout": {
                        "title": "PCA - First Two Components",
                        "xaxis": {"title": f"PC1 ({explained_variance[0]:.1%} variance)"},
                        "yaxis": {"title": f"PC2 ({explained_variance[1]:.1%} variance)"},
                        "template": "plotly_white"
                    }
                }

        except Exception as e:
            charts["error"] = str(e)

        return charts

    def _generate_regression_recommendations(self, results: Dict[str, Any]) -> List[str]:
        recommendations = []

        best_model = max(results.keys(), key=lambda k: results[k]["r2_score"])
        best_r2 = results[best_model]["r2_score"]

        recommendations.append(f"Best performing model: {best_model} (R² = {best_r2:.3f})")

        if best_r2 < 0.5:
            recommendations.append("Low R² score suggests poor model fit. Consider feature engineering or different algorithms.")
        elif best_r2 > 0.8:
            recommendations.append("High R² score indicates good model performance.")

        return recommendations

    def _generate_classification_recommendations(self, results: Dict[str, Any]) -> List[str]:
        recommendations = []

        best_model = max(results.keys(), key=lambda k: results[k]["accuracy"])
        best_accuracy = results[best_model]["accuracy"]

        recommendations.append(f"Best performing model: {best_model} (Accuracy = {best_accuracy:.3f})")

        if best_accuracy < 0.7:
            recommendations.append("Low accuracy suggests need for feature engineering or different algorithms.")
        elif best_accuracy > 0.9:
            recommendations.append("High accuracy indicates excellent model performance.")

        return recommendations

    def _generate_clustering_recommendations(self, results: Dict[str, Any]) -> List[str]:
        recommendations = []

        n_clusters = results["n_clusters"]
        cluster_stats = results["cluster_statistics"]

        recommendations.append(f"Data segmented into {n_clusters} distinct clusters.")

        # Check cluster balance
        cluster_sizes = [stats["size"] for stats in cluster_stats.values()]
        if max(cluster_sizes) / min(cluster_sizes) > 5:
            recommendations.append("Unbalanced cluster sizes detected. Consider different clustering parameters.")

        return recommendations

    # Statistical test helper methods
    def _perform_t_test(self, df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform t-test"""
        if len(columns) != 2:
            raise ValueError("T-test requires exactly 2 columns")

        col1, col2 = columns
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()

        statistic, p_value = stats.ttest_ind(data1, data2)

        return {
            "test_type": "t_test",
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference"
        }

    def _perform_chi_square_test(self, df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform chi-square test"""
        if len(columns) != 2:
            raise ValueError("Chi-square test requires exactly 2 columns")

        col1, col2 = columns
        contingency_table = pd.crosstab(df[col1], df[col2])

        statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        return {
            "test_type": "chi_square",
            "statistic": statistic,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "significant": p_value < 0.05,
            "contingency_table": contingency_table.to_dict(),
            "interpretation": "Significant association" if p_value < 0.05 else "No significant association"
        }

    def _perform_anova_test(self, df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ANOVA test"""
        if len(columns) < 2:
            raise ValueError("ANOVA requires at least 2 columns")

        groups = [df[col].dropna() for col in columns]
        statistic, p_value = stats.f_oneway(*groups)

        return {
            "test_type": "anova",
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "interpretation": "Significant difference between groups" if p_value < 0.05 else "No significant difference between groups"
        }

    def _perform_normality_test(self, df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform normality test (Shapiro-Wilk)"""
        results = {}

        for col in columns:
            data = df[col].dropna()
            if len(data) > 3:  # Minimum sample size for Shapiro-Wilk
                statistic, p_value = stats.shapiro(data)
                results[col] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "is_normal": p_value > 0.05,
                    "interpretation": "Normal distribution" if p_value > 0.05 else "Non-normal distribution"
                }

        return {
            "test_type": "normality",
            "results": results
        }

    # Advanced Analytics Methods
    async def time_series_analysis(self, dataset_id: str, date_column: str, value_column: str, parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform time series analysis including trend, seasonality, and forecasting
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")

        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        # Set date as index
        ts_data = df.set_index(date_column)[value_column]

        # Basic time series statistics
        results = {
            "data_points": len(ts_data),
            "date_range": {
                "start": ts_data.index.min().isoformat(),
                "end": ts_data.index.max().isoformat()
            },
            "basic_stats": {
                "mean": float(ts_data.mean()),
                "std": float(ts_data.std()),
                "min": float(ts_data.min()),
                "max": float(ts_data.max())
            }
        }

        # Trend analysis using linear regression
        x_numeric = np.arange(len(ts_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, ts_data.values)

        results["trend_analysis"] = {
            "slope": slope,
            "r_squared": r_value**2,
            "p_value": p_value,
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        }

        # Simple moving averages
        window_size = parameters.get("moving_average_window", 7)
        ts_data_ma = ts_data.rolling(window=window_size).mean()

        results["moving_averages"] = {
            "window_size": window_size,
            "latest_ma": float(ts_data_ma.iloc[-1]) if not ts_data_ma.empty else None
        }

        # Seasonality detection (basic)
        if len(ts_data) >= 24:  # Need sufficient data
            # Check for weekly seasonality (if daily data)
            weekly_pattern = ts_data.groupby(ts_data.index.dayofweek).mean()
            results["seasonality"] = {
                "weekly_pattern": weekly_pattern.to_dict(),
                "weekly_variation": float(weekly_pattern.std())
            }

        summary = f"Time series analysis completed for {len(ts_data)} data points from {ts_data.index.min().date()} to {ts_data.index.max().date()}"

        recommendations = []
        if abs(slope) > ts_data.std() / len(ts_data):
            recommendations.append(f"Strong {'upward' if slope > 0 else 'downward'} trend detected")
        if len(ts_data) < 30:
            recommendations.append("More data points recommended for robust time series analysis")

        return AnalysisResult(
            analysis_type="time_series",
            results=results,
            summary=summary,
            recommendations=recommendations
        )

    async def anomaly_detection(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Detect anomalies in the dataset using Isolation Forest
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")

        # Select numeric columns
        if columns:
            analysis_df = df[columns].select_dtypes(include=[np.number])
        else:
            analysis_df = df.select_dtypes(include=[np.number])

        if analysis_df.empty:
            raise ValueError("No numeric columns found for anomaly detection")

        # Handle missing values
        analysis_df = analysis_df.fillna(analysis_df.mean())

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(analysis_df)

        # Isolation Forest for anomaly detection
        contamination = parameters.get("contamination", 0.1)  # Expected proportion of outliers
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_data)

        # Get anomaly scores
        anomaly_scores = iso_forest.decision_function(scaled_data)

        # Identify anomalies
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        normal_indices = np.where(anomaly_labels == 1)[0]

        results = {
            "total_points": len(analysis_df),
            "anomalies_detected": len(anomaly_indices),
            "anomaly_percentage": (len(anomaly_indices) / len(analysis_df)) * 100,
            "anomaly_indices": anomaly_indices.tolist(),
            "anomaly_scores": {
                "min": float(anomaly_scores.min()),
                "max": float(anomaly_scores.max()),
                "mean": float(anomaly_scores.mean())
            }
        }

        # Get top anomalies
        top_anomalies_idx = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])[:5]]
        results["top_anomalies"] = []

        for idx in top_anomalies_idx:
            anomaly_data = analysis_df.iloc[idx].to_dict()
            anomaly_data["anomaly_score"] = float(anomaly_scores[idx])
            anomaly_data["row_index"] = int(idx)
            results["top_anomalies"].append(anomaly_data)

        summary = f"Anomaly detection completed. Found {len(anomaly_indices)} anomalies ({(len(anomaly_indices) / len(analysis_df)) * 100:.1f}%) out of {len(analysis_df)} data points."

        recommendations = []
        if len(anomaly_indices) > len(analysis_df) * 0.2:
            recommendations.append("High number of anomalies detected. Consider reviewing data quality.")
        elif len(anomaly_indices) == 0:
            recommendations.append("No anomalies detected. Data appears to be consistent.")
        else:
            recommendations.append("Review detected anomalies for potential data quality issues or interesting patterns.")

        return AnalysisResult(
            analysis_type="anomaly_detection",
            results=results,
            summary=summary,
            recommendations=recommendations
        )

    async def dimensionality_reduction(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform dimensionality reduction using PCA and t-SNE
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")

        # Select numeric columns
        if columns:
            analysis_df = df[columns].select_dtypes(include=[np.number])
        else:
            analysis_df = df.select_dtypes(include=[np.number])

        if analysis_df.empty:
            raise ValueError("No numeric columns found for dimensionality reduction")

        # Handle missing values
        analysis_df = analysis_df.fillna(analysis_df.mean())

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(analysis_df)

        results = {}

        # PCA Analysis
        n_components = min(parameters.get("n_components", 3), analysis_df.shape[1])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)

        results["pca"] = {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "transformed_data": pca_result.tolist()
        }

        # t-SNE Analysis (for visualization)
        if len(analysis_df) <= 1000:  # t-SNE is computationally expensive
            tsne_components = min(2, analysis_df.shape[1])
            tsne = TSNE(n_components=tsne_components, random_state=42, perplexity=min(30, len(analysis_df)-1))
            tsne_result = tsne.fit_transform(scaled_data)

            results["tsne"] = {
                "n_components": tsne_components,
                "transformed_data": tsne_result.tolist()
            }

        summary = f"Dimensionality reduction completed. PCA with {n_components} components explains {sum(pca.explained_variance_ratio_):.1%} of variance."

        recommendations = []
        if sum(pca.explained_variance_ratio_[:2]) > 0.8:
            recommendations.append("First 2 principal components capture most variance - good for 2D visualization")
        if n_components < analysis_df.shape[1]:
            recommendations.append(f"Consider using {n_components} components instead of {analysis_df.shape[1]} for reduced complexity")

        return AnalysisResult(
            analysis_type="dimensionality_reduction",
            results=results,
            summary=summary,
            recommendations=recommendations
        )

    async def advanced_clustering(self, dataset_id: str, columns: List[str], parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform advanced clustering using multiple algorithms
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")

        # Select numeric columns
        if columns:
            analysis_df = df[columns].select_dtypes(include=[np.number])
        else:
            analysis_df = df.select_dtypes(include=[np.number])

        if analysis_df.empty:
            raise ValueError("No numeric columns found for clustering")

        # Handle missing values
        analysis_df = analysis_df.fillna(analysis_df.mean())

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(analysis_df)

        results = {}

        # K-Means Clustering
        n_clusters = parameters.get("n_clusters", 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(scaled_data)

        results["kmeans"] = {
            "n_clusters": n_clusters,
            "labels": kmeans_labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_)
        }

        # DBSCAN Clustering
        eps = parameters.get("eps", 0.5)
        min_samples = parameters.get("min_samples", 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(scaled_data)

        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)

        results["dbscan"] = {
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters_dbscan,
            "n_noise_points": n_noise,
            "labels": dbscan_labels.tolist()
        }

        # Cluster quality metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        if len(set(kmeans_labels)) > 1:
            results["quality_metrics"] = {
                "kmeans_silhouette": float(silhouette_score(scaled_data, kmeans_labels)),
                "kmeans_calinski_harabasz": float(calinski_harabasz_score(scaled_data, kmeans_labels))
            }

        if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
            results["quality_metrics"]["dbscan_silhouette"] = float(silhouette_score(scaled_data, dbscan_labels))

        summary = f"Advanced clustering completed. K-Means found {n_clusters} clusters, DBSCAN found {n_clusters_dbscan} clusters with {n_noise} noise points."

        recommendations = []
        if n_noise > len(analysis_df) * 0.1:
            recommendations.append("High number of noise points in DBSCAN. Consider adjusting eps parameter.")
        if "quality_metrics" in results:
            if results["quality_metrics"].get("kmeans_silhouette", 0) > 0.5:
                recommendations.append("K-Means clustering shows good separation between clusters.")

        return AnalysisResult(
            analysis_type="advanced_clustering",
            results=results,
            summary=summary,
            recommendations=recommendations
        )
