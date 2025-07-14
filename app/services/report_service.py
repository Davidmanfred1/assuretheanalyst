"""
Report Service
Handles report generation and export functionality
"""

import os
import uuid
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# For PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

from app.models.analysis_models import ReportResult
from app.services.file_service import FileService
from app.services.analysis_service import AnalysisService

class ReportService:
    def __init__(self):
        self.file_service = FileService()
        self.analysis_service = AnalysisService()
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.reports_registry = {}
    
    async def generate_report(self, dataset_id: str, report_type: str, sections: List[str], 
                            format: str = "pdf", title: Optional[str] = None) -> ReportResult:
        """
        Generate a comprehensive data analysis report
        
        Args:
            dataset_id: Dataset identifier
            report_type: Type of report (comprehensive, summary, custom)
            sections: Report sections to include
            format: Output format (pdf, html, excel)
            title: Report title
            
        Returns:
            ReportResult with report information
        """
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            raise ValueError("Dataset not found")
        
        report_id = str(uuid.uuid4())
        report_title = title or f"Data Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Generate report content
        report_content = await self._generate_report_content(dataset_id, sections)
        
        # Create report file based on format
        if format.lower() == "pdf":
            file_path = await self._generate_pdf_report(report_id, report_title, report_content)
        elif format.lower() == "html":
            file_path = await self._generate_html_report(report_id, report_title, report_content)
        elif format.lower() == "excel":
            file_path = await self._generate_excel_report(report_id, report_title, report_content, df)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Store report information
        report_info = {
            "report_id": report_id,
            "title": report_title,
            "dataset_id": dataset_id,
            "format": format,
            "file_path": str(file_path),
            "created_at": datetime.now().isoformat(),
            "sections": sections
        }
        
        self.reports_registry[report_id] = report_info
        
        return ReportResult(
            report_id=report_id,
            file_path=str(file_path),
            format=format,
            created_at=datetime.now().isoformat()
        )
    
    async def _generate_report_content(self, dataset_id: str, sections: List[str]) -> Dict[str, Any]:
        """Generate report content based on requested sections"""
        content = {}
        
        if "summary" in sections:
            content["summary"] = await self.analysis_service.get_dataset_summary(dataset_id)
        
        if "descriptive" in sections:
            from app.models.analysis_models import AnalysisRequest, AnalysisType
            request = AnalysisRequest(
                dataset_id=dataset_id,
                analysis_type=AnalysisType.DESCRIPTIVE,
                columns=[]
            )
            content["descriptive"] = await self.analysis_service.descriptive_analysis(
                dataset_id, [], {}
            )
        
        if "correlation" in sections:
            from app.models.analysis_models import AnalysisRequest, AnalysisType
            request = AnalysisRequest(
                dataset_id=dataset_id,
                analysis_type=AnalysisType.CORRELATION,
                columns=[]
            )
            content["correlation"] = await self.analysis_service.correlation_analysis(
                dataset_id, [], {}
            )
        
        return content
    
    async def _generate_pdf_report(self, report_id: str, title: str, content: Dict[str, Any]) -> Path:
        """Generate PDF report"""
        file_path = self.reports_dir / f"report_{report_id}.pdf"
        
        doc = SimpleDocTemplate(str(file_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph("Generated by: AssureTheAnalyst - Manfred Incorporations", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Content sections
        for section_name, section_data in content.items():
            # Section header
            story.append(Paragraph(section_name.title(), styles['Heading2']))
            story.append(Spacer(1, 12))
            
            if section_name == "summary":
                self._add_summary_to_pdf(story, section_data, styles)
            elif section_name == "descriptive":
                self._add_descriptive_to_pdf(story, section_data, styles)
            elif section_name == "correlation":
                self._add_correlation_to_pdf(story, section_data, styles)
            
            story.append(Spacer(1, 20))
        
        doc.build(story)
        return file_path
    
    def _add_summary_to_pdf(self, story: List, data: Dict[str, Any], styles):
        """Add summary section to PDF"""
        basic_info = data.get("basic_info", {})
        
        # Basic information table
        table_data = [
            ["Metric", "Value"],
            ["Total Rows", f"{basic_info.get('rows', 'N/A'):,}"],
            ["Total Columns", f"{basic_info.get('columns', 'N/A'):,}"],
            ["Numeric Columns", f"{basic_info.get('numeric_columns', 'N/A'):,}"],
            ["Categorical Columns", f"{basic_info.get('categorical_columns', 'N/A'):,}"],
        ]
        
        table = Table(table_data)
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
    
    def _add_descriptive_to_pdf(self, story: List, data: Dict[str, Any], styles):
        """Add descriptive statistics to PDF"""
        story.append(Paragraph("Descriptive Statistics", styles['Heading3']))
        story.append(Paragraph(data.summary, styles['Normal']))
    
    def _add_correlation_to_pdf(self, story: List, data: Dict[str, Any], styles):
        """Add correlation analysis to PDF"""
        story.append(Paragraph("Correlation Analysis", styles['Heading3']))
        story.append(Paragraph(data.summary, styles['Normal']))
        
        # Strong correlations
        strong_corr = data.results.get("strong_correlations", [])
        if strong_corr:
            story.append(Paragraph("Strong Correlations Found:", styles['Heading4']))
            for corr in strong_corr[:5]:  # Top 5
                text = f"• {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f}"
                story.append(Paragraph(text, styles['Normal']))
    
    async def _generate_html_report(self, report_id: str, title: str, content: Dict[str, Any]) -> Path:
        """Generate HTML report"""
        file_path = self.reports_dir / f"report_{report_id}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metadata {{ color: #888; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p class="metadata">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="metadata">Generated by: AssureTheAnalyst - Manfred Incorporations</p>
        """
        
        for section_name, section_data in content.items():
            html_content += f"<h2>{section_name.title()}</h2>"
            html_content += f"<p>{getattr(section_data, 'summary', 'No summary available')}</p>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return file_path
    
    async def _generate_excel_report(self, report_id: str, title: str, content: Dict[str, Any], df: pd.DataFrame) -> Path:
        """Generate Excel report"""
        file_path = self.reports_dir / f"report_{report_id}.xlsx"
        
        with pd.ExcelWriter(str(file_path), engine='openpyxl') as writer:
            # Raw data
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Summary sheet
            if "summary" in content:
                summary_data = content["summary"]
                basic_info = summary_data.get("basic_info", {})
                
                summary_df = pd.DataFrame([
                    ["Total Rows", basic_info.get('rows', 'N/A')],
                    ["Total Columns", basic_info.get('columns', 'N/A')],
                    ["Numeric Columns", basic_info.get('numeric_columns', 'N/A')],
                    ["Categorical Columns", basic_info.get('categorical_columns', 'N/A')],
                ], columns=['Metric', 'Value'])
                
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return file_path
    
    async def get_report_file(self, report_id: str) -> Optional[str]:
        """Get report file path"""
        if report_id in self.reports_registry:
            return self.reports_registry[report_id]["file_path"]
        return None
    
    async def list_reports(self) -> List[Dict[str, Any]]:
        """List all generated reports"""
        return list(self.reports_registry.values())
    
    async def delete_report(self, report_id: str) -> bool:
        """Delete a report"""
        if report_id not in self.reports_registry:
            return False
        
        # Remove file
        file_path = Path(self.reports_registry[report_id]["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Remove from registry
        del self.reports_registry[report_id]
        return True
    
    def get_report_templates(self) -> Dict[str, Any]:
        """Get available report templates"""
        return {
            "comprehensive": {
                "name": "Comprehensive Analysis Report",
                "description": "Complete analysis including summary, descriptive statistics, and correlations",
                "sections": ["summary", "descriptive", "correlation"]
            },
            "summary": {
                "name": "Summary Report",
                "description": "Basic dataset overview and summary statistics",
                "sections": ["summary"]
            },
            "statistical": {
                "name": "Statistical Analysis Report",
                "description": "Detailed statistical analysis and insights",
                "sections": ["descriptive", "correlation"]
            }
        }
    
    async def export_dataset(self, dataset_id: str, format: str) -> Optional[str]:
        """Export dataset in specified format"""
        df = self.file_service.get_dataframe(dataset_id)
        if df is None:
            return None
        
        export_path = self.reports_dir / f"export_{dataset_id}.{format}"
        
        if format.lower() == "csv":
            df.to_csv(export_path, index=False)
        elif format.lower() == "excel":
            df.to_excel(export_path, index=False)
        elif format.lower() == "json":
            df.to_json(export_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return str(export_path)
