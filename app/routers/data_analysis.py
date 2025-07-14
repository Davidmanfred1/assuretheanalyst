"""
Data Analysis Router
Handles statistical analysis and data processing operations
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import io
import json

from app.models.analysis_models import (
    AnalysisRequest, AnalysisResult, StatisticalTestRequest, AnalysisType
)
from app.services.analysis_service import AnalysisService

router = APIRouter()

# Analysis service will be initialized with shared file service
analysis_service = None

def set_file_service(service):
    global analysis_service
    analysis_service = AnalysisService()
    analysis_service.file_service = service

@router.post("/run")
async def run_analysis(request: AnalysisRequest):
    """
    Run analysis based on analysis type
    """
    try:
        analysis_type = request.analysis_type.lower()

        if analysis_type == "descriptive":
            result = await analysis_service.descriptive_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "correlation":
            result = await analysis_service.correlation_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "regression":
            result = await analysis_service.regression_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "classification":
            result = await analysis_service.classification_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "clustering":
            result = await analysis_service.clustering_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "time-series":
            result = await analysis_service.time_series_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "anomaly-detection":
            result = await analysis_service.anomaly_detection(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "dimensionality-reduction":
            result = await analysis_service.dimensionality_reduction_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "advanced-clustering":
            result = await analysis_service.advanced_clustering_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        elif analysis_type == "forecasting":
            result = await analysis_service.forecasting_analysis(
                request.dataset_id, request.columns, request.parameters
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}")

        return {"success": True, "result": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/descriptive", response_model=AnalysisResult)
async def descriptive_analysis(request: AnalysisRequest):
    """
    Perform descriptive statistical analysis
    """
    try:
        if request.analysis_type != AnalysisType.DESCRIPTIVE:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")
        
        result = await analysis_service.descriptive_analysis(
            request.dataset_id, 
            request.columns, 
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/correlation", response_model=AnalysisResult)
async def correlation_analysis(request: AnalysisRequest):
    """
    Perform correlation analysis
    """
    try:
        if request.analysis_type != AnalysisType.CORRELATION:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")
        
        result = await analysis_service.correlation_analysis(
            request.dataset_id, 
            request.columns, 
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")

@router.post("/regression", response_model=AnalysisResult)
async def regression_analysis(request: AnalysisRequest):
    """
    Perform regression analysis
    """
    try:
        if request.analysis_type != AnalysisType.REGRESSION:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")
        
        result = await analysis_service.regression_analysis(
            request.dataset_id, 
            request.columns, 
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regression analysis failed: {str(e)}")

@router.post("/classification", response_model=AnalysisResult)
async def classification_analysis(request: AnalysisRequest):
    """
    Perform classification analysis
    """
    try:
        if request.analysis_type != AnalysisType.CLASSIFICATION:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")
        
        result = await analysis_service.classification_analysis(
            request.dataset_id, 
            request.columns, 
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification analysis failed: {str(e)}")

@router.post("/clustering", response_model=AnalysisResult)
async def clustering_analysis(request: AnalysisRequest):
    """
    Perform clustering analysis
    """
    try:
        if request.analysis_type != AnalysisType.CLUSTERING:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")
        
        result = await analysis_service.clustering_analysis(
            request.dataset_id, 
            request.columns, 
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@router.post("/statistical-test")
async def statistical_test(request: StatisticalTestRequest):
    """
    Perform statistical tests
    """
    try:
        result = await analysis_service.statistical_test(
            request.dataset_id,
            request.test_type,
            request.columns,
            request.parameters or {}
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical test failed: {str(e)}")

@router.post("/export/{format}")
async def export_analysis_results(format: str, analysis_result: dict):
    """
    Export analysis results to various formats
    """
    try:
        if format.lower() == "pdf":
            return await export_to_pdf(analysis_result)
        elif format.lower() == "excel":
            return await export_to_excel(analysis_result)
        elif format.lower() == "csv":
            return await export_to_csv(analysis_result)
        elif format.lower() == "json":
            return await export_to_json(analysis_result)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

async def export_to_pdf(analysis_result: dict) -> StreamingResponse:
    """Export analysis results to PDF"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # Analysis Type and Summary
        story.append(Paragraph(f"<b>Analysis Type:</b> {analysis_result.get('analysis_type', 'N/A').replace('_', ' ').title()}", styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Summary:</b> {analysis_result.get('summary', 'No summary available')}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Results
        results = analysis_result.get('results', {})
        if results:
            story.append(Paragraph("<b>Results:</b>", styles['Heading2']))
            story.append(Spacer(1, 6))

            # Convert results to readable format
            for key, value in results.items():
                if key == 'visualizations':
                    continue  # Skip visualizations for PDF

                story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b>", styles['Heading3']))

                if isinstance(value, dict):
                    # Create table for dictionary data
                    data = [['Metric', 'Value']]
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            data.append([k.replace('_', ' ').title(), f"{v:.4f}" if isinstance(v, float) else str(v)])
                        else:
                            data.append([k.replace('_', ' ').title(), str(v)[:50] + "..." if len(str(v)) > 50 else str(v)])

                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)
                else:
                    story.append(Paragraph(str(value), styles['Normal']))

                story.append(Spacer(1, 12))

        # Timestamp
        timestamp = analysis_result.get('timestamp', 'N/A')
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"<i>Generated on: {timestamp}</i>", styles['Normal']))

        doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=analysis_report.pdf"}
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="PDF export requires reportlab package")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")

async def export_to_excel(analysis_result: dict) -> StreamingResponse:
    """Export analysis results to Excel"""
    try:
        import pandas as pd

        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Analysis Type': [analysis_result.get('analysis_type', 'N/A').replace('_', ' ').title()],
                'Summary': [analysis_result.get('summary', 'No summary available')],
                'Timestamp': [analysis_result.get('timestamp', 'N/A')]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Results sheets
            results = analysis_result.get('results', {})
            for key, value in results.items():
                if key == 'visualizations':
                    continue  # Skip visualizations for Excel

                sheet_name = key.replace('_', ' ').title()[:31]  # Excel sheet name limit

                if isinstance(value, dict):
                    # Convert dictionary to DataFrame
                    if all(isinstance(v, (list, tuple)) for v in value.values()):
                        # If all values are lists, create columns
                        df = pd.DataFrame(value)
                    else:
                        # Otherwise, create key-value pairs
                        df = pd.DataFrame(list(value.items()), columns=['Metric', 'Value'])
                elif isinstance(value, list):
                    df = pd.DataFrame(value)
                else:
                    df = pd.DataFrame([value], columns=[key])

                df.to_excel(writer, sheet_name=sheet_name, index=False)

        buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=analysis_report.xlsx"}
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="Excel export requires openpyxl package")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")

async def export_to_csv(analysis_result: dict) -> StreamingResponse:
    """Export analysis results to CSV"""
    try:
        import pandas as pd

        buffer = io.StringIO()

        # Write summary
        buffer.write("Analysis Report\n")
        buffer.write("=" * 50 + "\n")
        buffer.write(f"Analysis Type: {analysis_result.get('analysis_type', 'N/A').replace('_', ' ').title()}\n")
        buffer.write(f"Summary: {analysis_result.get('summary', 'No summary available')}\n")
        buffer.write(f"Timestamp: {analysis_result.get('timestamp', 'N/A')}\n")
        buffer.write("\n")

        # Write results
        results = analysis_result.get('results', {})
        for key, value in results.items():
            if key == 'visualizations':
                continue

            buffer.write(f"\n{key.replace('_', ' ').title()}\n")
            buffer.write("-" * 30 + "\n")

            if isinstance(value, dict):
                for k, v in value.items():
                    buffer.write(f"{k.replace('_', ' ').title()},{v}\n")
            elif isinstance(value, list):
                for item in value:
                    buffer.write(f"{item}\n")
            else:
                buffer.write(f"{value}\n")

            buffer.write("\n")

        buffer.seek(0)

        return StreamingResponse(
            io.StringIO(buffer.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=analysis_report.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")

async def export_to_json(analysis_result: dict) -> StreamingResponse:
    """Export analysis results to JSON"""
    try:
        buffer = io.StringIO()
        json.dump(analysis_result, buffer, indent=2, default=str)
        buffer.seek(0)

        return StreamingResponse(
            io.StringIO(buffer.getvalue()),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=analysis_report.json"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON export failed: {str(e)}")

@router.get("/datasets/{dataset_id}/summary")
async def get_dataset_summary(dataset_id: str):
    """
    Get comprehensive dataset summary
    """
    try:
        summary = await analysis_service.get_dataset_summary(dataset_id)
        return {"success": True, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@router.post("/time-series", response_model=AnalysisResult)
async def time_series_analysis(request: AnalysisRequest):
    """
    Perform time series analysis
    """
    try:
        if request.analysis_type != AnalysisType.TIME_SERIES:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")

        # Extract date and value columns from parameters
        date_column = request.parameters.get("date_column")
        value_column = request.parameters.get("value_column")

        if not date_column or not value_column:
            raise HTTPException(status_code=400, detail="date_column and value_column must be specified in parameters")

        result = await analysis_service.time_series_analysis(
            request.dataset_id,
            date_column,
            value_column,
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time series analysis failed: {str(e)}")

@router.post("/anomaly-detection", response_model=AnalysisResult)
async def anomaly_detection(request: AnalysisRequest):
    """
    Perform anomaly detection
    """
    try:
        if request.analysis_type != AnalysisType.ANOMALY_DETECTION:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")

        result = await analysis_service.anomaly_detection(
            request.dataset_id,
            request.columns,
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@router.post("/dimensionality-reduction", response_model=AnalysisResult)
async def dimensionality_reduction(request: AnalysisRequest):
    """
    Perform dimensionality reduction
    """
    try:
        if request.analysis_type != AnalysisType.DIMENSIONALITY_REDUCTION:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")

        result = await analysis_service.dimensionality_reduction(
            request.dataset_id,
            request.columns,
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dimensionality reduction failed: {str(e)}")

@router.post("/advanced-clustering", response_model=AnalysisResult)
async def advanced_clustering(request: AnalysisRequest):
    """
    Perform advanced clustering analysis
    """
    try:
        if request.analysis_type != AnalysisType.ADVANCED_CLUSTERING:
            raise HTTPException(status_code=400, detail="Invalid analysis type for this endpoint")

        result = await analysis_service.advanced_clustering(
            request.dataset_id,
            request.columns,
            request.parameters or {}
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced clustering failed: {str(e)}")
