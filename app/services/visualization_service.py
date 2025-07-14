"""
Visualization Service
Handles data visualization and chart generation using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import uuid
from typing import Dict, List, Any, Optional

from app.models.analysis_models import VisualizationType, VisualizationResult
from app.services.file_service import FileService

class VisualizationService:
    def __init__(self):
        self.file_service = FileService()
        self.charts_registry = {}
    
    async def create_chart(self, dataset_id: str, chart_type: VisualizationType, 
                          x_column: Optional[str] = None, y_column: Optional[str] = None,
                          color_column: Optional[str] = None, title: Optional[str] = None,
                          parameters: Dict[str, Any] = None) -> VisualizationResult:
        """
        Create a data visualization chart
        
        Args:
            dataset_id: Dataset identifier
            chart_type: Type of chart to create
            x_column: X-axis column
            y_column: Y-axis column
            color_column: Color grouping column
            title: Chart title
            parameters: Additional chart parameters
            
        Returns:
            VisualizationResult with chart data and HTML
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")
        
        parameters = parameters or {}
        
        # Generate chart based on type
        if chart_type == VisualizationType.HISTOGRAM:
            fig = self._create_histogram(df, x_column, parameters)
        elif chart_type == VisualizationType.SCATTER:
            fig = self._create_scatter_plot(df, x_column, y_column, color_column, parameters)
        elif chart_type == VisualizationType.LINE:
            fig = self._create_line_chart(df, x_column, y_column, color_column, parameters)
        elif chart_type == VisualizationType.BAR:
            fig = self._create_bar_chart(df, x_column, y_column, color_column, parameters)
        elif chart_type == VisualizationType.BOX:
            fig = self._create_box_plot(df, x_column, y_column, color_column, parameters)
        elif chart_type == VisualizationType.HEATMAP:
            fig = self._create_heatmap(df, parameters)
        elif chart_type == VisualizationType.PIE:
            fig = self._create_pie_chart(df, x_column, parameters)
        elif chart_type == "3d_scatter":
            fig = self._create_3d_scatter(df, x_column, y_column, parameters.get("z_column"), color_column, parameters)
        elif chart_type == "surface":
            fig = self._create_surface_plot(df, x_column, y_column, parameters.get("z_column"), parameters)
        elif chart_type == "network":
            fig = self._create_network_graph(df, parameters)
        elif chart_type == "treemap":
            fig = self._create_treemap(df, x_column, y_column, parameters)
        elif chart_type == "sunburst":
            fig = self._create_sunburst(df, parameters)
        elif chart_type == "parallel_coordinates":
            fig = self._create_parallel_coordinates(df, columns, color_column, parameters)
        elif chart_type == "radar":
            fig = self._create_radar_chart(df, parameters)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Set title
        if title:
            fig.update_layout(title=title)
        
        # Generate chart ID and store
        chart_id = str(uuid.uuid4())
        chart_html = fig.to_html(include_plotlyjs='cdn')
        chart_data = fig.to_dict()
        
        self.charts_registry[chart_id] = {
            "chart_type": chart_type,
            "chart_data": chart_data,
            "chart_html": chart_html,
            "dataset_id": dataset_id
        }
        
        return VisualizationResult(
            chart_type=chart_type,
            chart_data=chart_data,
            chart_html=chart_html
        )
    
    def _create_histogram(self, df: pd.DataFrame, column: str, parameters: Dict[str, Any]) -> go.Figure:
        """Create histogram"""
        if not column:
            raise ValueError("Column must be specified for histogram")
        
        bins = parameters.get("bins", 30)
        
        fig = px.histogram(
            df, 
            x=column, 
            nbins=bins,
            title=f"Distribution of {column}"
        )
        
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_column: str, y_column: str, 
                           color_column: Optional[str], parameters: Dict[str, Any]) -> go.Figure:
        """Create scatter plot"""
        if not x_column or not y_column:
            raise ValueError("Both x and y columns must be specified for scatter plot")
        
        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=f"{y_column} vs {x_column}",
            trendline=parameters.get("trendline", None)
        )
        
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, x_column: str, y_column: str,
                          color_column: Optional[str], parameters: Dict[str, Any]) -> go.Figure:
        """Create line chart"""
        if not x_column or not y_column:
            raise ValueError("Both x and y columns must be specified for line chart")
        
        fig = px.line(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=f"{y_column} over {x_column}"
        )
        
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, x_column: str, y_column: Optional[str],
                         color_column: Optional[str], parameters: Dict[str, Any]) -> go.Figure:
        """Create bar chart"""
        if not x_column:
            raise ValueError("X column must be specified for bar chart")
        
        if y_column:
            # Grouped bar chart
            fig = px.bar(
                df,
                x=x_column,
                y=y_column,
                color=color_column,
                title=f"{y_column} by {x_column}"
            )
        else:
            # Count bar chart
            value_counts = df[x_column].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Count of {x_column}"
            )
        
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, x_column: Optional[str], y_column: str,
                        color_column: Optional[str], parameters: Dict[str, Any]) -> go.Figure:
        """Create box plot"""
        if not y_column:
            raise ValueError("Y column must be specified for box plot")
        
        fig = px.box(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=f"Distribution of {y_column}" + (f" by {x_column}" if x_column else "")
        )
        
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> go.Figure:
        """Create correlation heatmap"""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for heatmap")
        
        correlation_matrix = numeric_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, column: str, parameters: Dict[str, Any]) -> go.Figure:
        """Create pie chart"""
        if not column:
            raise ValueError("Column must be specified for pie chart")
        
        value_counts = df[column].value_counts()
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribution of {column}"
        )
        
        return fig
    
    async def get_chart(self, chart_id: str) -> Optional[Dict[str, Any]]:
        """Get a previously created chart"""
        return self.charts_registry.get(chart_id)
    
    async def create_dashboard(self, dataset_id: str, chart_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a dashboard with multiple charts"""
        charts = []
        
        for config in chart_configs:
            chart_result = await self.create_chart(
                dataset_id=dataset_id,
                chart_type=VisualizationType(config["chart_type"]),
                x_column=config.get("x_column"),
                y_column=config.get("y_column"),
                color_column=config.get("color_column"),
                title=config.get("title"),
                parameters=config.get("parameters", {})
            )
            charts.append(chart_result.dict())
        
        dashboard_id = str(uuid.uuid4())
        dashboard = {
            "dashboard_id": dashboard_id,
            "dataset_id": dataset_id,
            "charts": charts,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        return dashboard
    
    async def get_dataset_charts(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get all charts for a dataset"""
        charts = []
        for chart_id, chart_data in self.charts_registry.items():
            if chart_data["dataset_id"] == dataset_id:
                charts.append({
                    "chart_id": chart_id,
                    "chart_type": chart_data["chart_type"],
                    "dataset_id": dataset_id
                })
        return charts
    
    async def delete_chart(self, chart_id: str) -> bool:
        """Delete a chart"""
        if chart_id in self.charts_registry:
            del self.charts_registry[chart_id]
            return True
        return False
    
    def get_available_chart_types(self) -> Dict[str, Any]:
        """Get available chart types and their configurations"""
        return {
            "histogram": {
                "name": "Histogram",
                "description": "Distribution of a single variable",
                "required_columns": ["x"],
                "optional_parameters": ["bins"]
            },
            "scatter": {
                "name": "Scatter Plot",
                "description": "Relationship between two variables",
                "required_columns": ["x", "y"],
                "optional_columns": ["color"],
                "optional_parameters": ["trendline"]
            },
            "line": {
                "name": "Line Chart",
                "description": "Trend over time or ordered variable",
                "required_columns": ["x", "y"],
                "optional_columns": ["color"]
            },
            "bar": {
                "name": "Bar Chart",
                "description": "Categorical data comparison",
                "required_columns": ["x"],
                "optional_columns": ["y", "color"]
            },
            "box": {
                "name": "Box Plot",
                "description": "Distribution and outliers",
                "required_columns": ["y"],
                "optional_columns": ["x", "color"]
            },
            "heatmap": {
                "name": "Heatmap",
                "description": "Correlation matrix visualization",
                "required_columns": [],
                "note": "Uses all numeric columns"
            },
            "pie": {
                "name": "Pie Chart",
                "description": "Proportional data visualization",
                "required_columns": ["x"]
            },
            "3d_scatter": {
                "name": "3D Scatter Plot",
                "description": "Three-dimensional scatter visualization",
                "required_columns": ["x", "y", "z"],
                "optional_columns": ["color", "size"]
            },
            "surface": {
                "name": "Surface Plot",
                "description": "3D surface visualization",
                "required_columns": ["x", "y", "z"]
            },
            "network": {
                "name": "Network Graph",
                "description": "Network relationship visualization",
                "required_parameters": ["source_column", "target_column"],
                "optional_parameters": ["weight_column"]
            },
            "treemap": {
                "name": "Treemap",
                "description": "Hierarchical data visualization",
                "required_columns": ["category", "value"]
            },
            "parallel_coordinates": {
                "name": "Parallel Coordinates",
                "description": "Multi-dimensional data visualization",
                "required_columns": ["multiple_numeric"],
                "optional_columns": ["color"]
            },
            "radar": {
                "name": "Radar Chart",
                "description": "Multi-variate data comparison",
                "required_parameters": ["categories", "values_column"]
            }
        }

    def _create_3d_scatter(self, df: pd.DataFrame, x_column: str, y_column: str,
                          z_column: str, color_column: Optional[str], parameters: Dict[str, Any]) -> go.Figure:
        """Create 3D scatter plot"""
        if not x_column or not y_column or not z_column:
            raise ValueError("X, Y, and Z columns must be specified for 3D scatter plot")

        fig = px.scatter_3d(
            df,
            x=x_column,
            y=y_column,
            z=z_column,
            color=color_column,
            title=f"3D Scatter: {z_column} vs {y_column} vs {x_column}",
            size=parameters.get("size_column"),
            hover_data=parameters.get("hover_columns", [])
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=x_column,
                yaxis_title=y_column,
                zaxis_title=z_column
            )
        )

        return fig

    def _create_surface_plot(self, df: pd.DataFrame, x_column: str, y_column: str,
                           z_column: str, parameters: Dict[str, Any]) -> go.Figure:
        """Create 3D surface plot"""
        if not x_column or not y_column or not z_column:
            raise ValueError("X, Y, and Z columns must be specified for surface plot")

        # Create a pivot table for the surface
        try:
            pivot_df = df.pivot_table(values=z_column, index=y_column, columns=x_column, aggfunc='mean')
        except Exception:
            # If pivot fails, create a simple surface from the data
            pivot_df = df.set_index([y_column, x_column])[z_column].unstack(fill_value=0)

        fig = go.Figure(data=[go.Surface(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis'
        )])

        fig.update_layout(
            title=f"Surface Plot: {z_column} over {x_column} and {y_column}",
            scene=dict(
                xaxis_title=x_column,
                yaxis_title=y_column,
                zaxis_title=z_column
            )
        )

        return fig

    def _create_network_graph(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> go.Figure:
        """Create network graph visualization"""
        source_col = parameters.get("source_column")
        target_col = parameters.get("target_column")
        weight_col = parameters.get("weight_column")

        if not source_col or not target_col:
            raise ValueError("Source and target columns must be specified for network graph")

        # Create a simple network visualization without networkx dependency
        # Get unique nodes
        nodes = list(set(df[source_col].tolist() + df[target_col].tolist()))
        node_positions = {node: (i % 10, i // 10) for i, node in enumerate(nodes)}

        # Create edges
        edge_x = []
        edge_y = []

        for _, row in df.iterrows():
            source_pos = node_positions[row[source_col]]
            target_pos = node_positions[row[target_col]]
            edge_x.extend([source_pos[0], target_pos[0], None])
            edge_y.extend([source_pos[1], target_pos[1], None])

        # Create the plot
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        ))

        # Add nodes
        node_x = [node_positions[node][0] for node in nodes]
        node_y = [node_positions[node][1] for node in nodes]

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=nodes,
            textposition="middle center",
            marker=dict(
                size=15,
                color='lightblue',
                line=dict(width=2, color='black')
            ),
            name='Nodes'
        ))

        fig.update_layout(
            title="Network Graph",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        return fig

    def _create_treemap(self, df: pd.DataFrame, category_column: str, value_column: str, parameters: Dict[str, Any]) -> go.Figure:
        """Create treemap visualization"""
        if not category_column or not value_column:
            raise ValueError("Category and value columns must be specified for treemap")

        # Aggregate data by category
        agg_df = df.groupby(category_column)[value_column].sum().reset_index()

        fig = px.treemap(
            agg_df,
            path=[category_column],
            values=value_column,
            title=f"Treemap: {value_column} by {category_column}"
        )

        return fig
