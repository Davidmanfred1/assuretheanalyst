#!/usr/bin/env python3
"""
🔍 DEBUG ANALYSIS & DATA QUALITY ERRORS
Identify and fix issues with analysis and data quality checks
"""

import requests
import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime

def debug_analysis_errors():
    print("🔍 DEBUGGING ANALYSIS & DATA QUALITY ERRORS")
    print("=" * 60)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    errors_found = []
    
    def log_error(category, error, details="", traceback_info=""):
        errors_found.append({
            'category': category,
            'error': error,
            'details': details,
            'traceback': traceback_info,
            'timestamp': datetime.now().isoformat()
        })
        print(f"❌ {category}: {error}")
        if details:
            print(f"   Details: {details}")
        if traceback_info:
            print(f"   Traceback: {traceback_info}")
    
    # 1. CREATE TEST DATASET
    print("📊 1. CREATING TEST DATASET")
    print("-" * 40)
    
    try:
        # Create a comprehensive test dataset with various data types and issues
        np.random.seed(42)
        test_data = pd.DataFrame({
            'id': range(1, 101),
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 15000, 100),
            'spending': np.random.normal(2000, 500, 100),
            'satisfaction': np.random.uniform(1, 10, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'date_column': pd.date_range('2023-01-01', periods=100, freq='D'),
            'boolean_col': np.random.choice([True, False], 100),
            'text_col': [f'text_{i}' for i in range(100)]
        })
        
        # Add some correlations
        test_data['spending'] = test_data['income'] * 0.04 + np.random.normal(0, 200, 100)
        test_data['satisfaction'] = 10 - (test_data['age'] - 50) * 0.05 + np.random.normal(0, 1, 100)
        
        # Add some data quality issues
        test_data.loc[5:10, 'income'] = None  # Missing values
        test_data.loc[15, 'age'] = 200  # Outlier
        test_data.loc[20:22, :] = test_data.loc[18:20, :].values  # Duplicates
        test_data.loc[25, 'satisfaction'] = 15  # Invalid range
        
        test_data.to_csv('debug_analysis_test.csv', index=False)
        print("✅ Test dataset created successfully")
        
    except Exception as e:
        log_error("Dataset Creation", "Failed to create test dataset", str(e), traceback.format_exc())
        return
    
    # 2. UPLOAD DATASET
    print("\n📤 2. UPLOADING DATASET")
    print("-" * 40)
    
    dataset_id = None
    try:
        with open('debug_analysis_test.csv', 'rb') as f:
            files = {'file': ('debug_analysis_test.csv', f, 'text/csv')}
            response = requests.post('http://localhost:8000/api/upload/', files=files, timeout=20)
        
        print(f"Upload response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Upload result: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                dataset_id = result.get('dataset_id')
                print(f"✅ Dataset uploaded successfully: {dataset_id}")
            else:
                log_error("Upload", "Upload failed", result.get('message', 'Unknown error'))
                return
        else:
            log_error("Upload", f"Upload request failed: {response.status_code}", response.text)
            return
            
    except Exception as e:
        log_error("Upload", "Upload exception", str(e), traceback.format_exc())
        return
    
    # 3. TEST DATASET PREVIEW
    print("\n👀 3. TESTING DATASET PREVIEW")
    print("-" * 40)
    
    try:
        response = requests.get(f'http://localhost:8000/api/upload/datasets/{dataset_id}/preview', timeout=15)
        print(f"Preview response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Dataset preview works")
            print(f"   Columns: {len(result.get('columns', []))}")
            print(f"   Rows: {result.get('rows', 0)}")
        else:
            log_error("Preview", f"Preview failed: {response.status_code}", response.text)
            
    except Exception as e:
        log_error("Preview", "Preview exception", str(e), traceback.format_exc())
    
    # 4. TEST ANALYSIS FUNCTIONALITY
    print("\n📊 4. TESTING ANALYSIS FUNCTIONALITY")
    print("-" * 40)
    
    analysis_types = [
        ('descriptive', ['age', 'income', 'spending']),
        ('correlation', ['age', 'income', 'spending', 'satisfaction']),
        ('clustering', ['age', 'income', 'spending']),
        ('dimensionality_reduction', ['age', 'income', 'spending', 'satisfaction']),
        ('forecasting', ['age', 'income'])
    ]
    
    for analysis_type, columns in analysis_types:
        print(f"\n🔍 Testing {analysis_type.upper()} analysis...")
        
        try:
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': analysis_type,
                'columns': columns,
                'parameters': {}
            }
            
            # Add specific parameters for certain analysis types
            if analysis_type == 'clustering':
                analysis_data['parameters']['n_clusters'] = 3
            elif analysis_type == 'dimensionality_reduction':
                analysis_data['parameters']['n_components'] = 2
            elif analysis_type == 'forecasting':
                analysis_data['parameters']['forecast_periods'] = 10
            
            print(f"   Request data: {json.dumps(analysis_data, indent=2)}")
            
            response = requests.post('http://localhost:8000/api/analysis/run', 
                                   json=analysis_data, timeout=30)
            
            print(f"   Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Response success: {result.get('success')}")
                
                if result.get('success'):
                    print(f"✅ {analysis_type.upper()} analysis successful")
                    
                    # Check for key components
                    if 'results' in result:
                        print(f"   📊 Results: Available")
                    if 'summary' in result:
                        print(f"   📝 Summary: {len(result['summary'])} characters")
                    if 'insights' in result and result['insights']:
                        insights = result['insights']
                        print(f"   🧠 AI Insights: {len(insights.get('key_findings', []))} findings")
                    if 'visualizations' in result:
                        print(f"   📈 Visualizations: {len(result['visualizations'])} charts")
                else:
                    log_error("Analysis", f"{analysis_type} analysis failed", 
                             result.get('message', 'Unknown error'), 
                             result.get('detail', ''))
            else:
                log_error("Analysis", f"{analysis_type} analysis HTTP error: {response.status_code}", 
                         response.text[:500])
                
        except Exception as e:
            log_error("Analysis", f"{analysis_type} analysis exception", str(e), traceback.format_exc())
    
    # 5. TEST DATA QUALITY CHECK
    print("\n🛡️ 5. TESTING DATA QUALITY CHECK")
    print("-" * 40)
    
    try:
        print(f"   Testing quality check for dataset: {dataset_id}")
        
        response = requests.post(f'http://localhost:8000/api/upload/quality-check/{dataset_id}', timeout=30)
        
        print(f"   Quality check response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Quality check success: {result.get('success')}")
            
            if result.get('success'):
                quality_report = result.get('quality_report', {})
                print("✅ Data quality check successful")
                print(f"   📊 Overall score: {quality_report.get('overall_score', 'N/A')}")
                print(f"   📋 Dimensions checked: {len(quality_report.get('dimensions', {}))}")
                print(f"   ⚠️ Issues found: {len(quality_report.get('issues', []))}")
                print(f"   💡 Recommendations: {len(quality_report.get('recommendations', []))}")
            else:
                log_error("Quality Check", "Quality check failed", 
                         result.get('message', 'Unknown error'),
                         result.get('detail', ''))
        else:
            log_error("Quality Check", f"Quality check HTTP error: {response.status_code}", 
                     response.text[:500])
            
    except Exception as e:
        log_error("Quality Check", "Quality check exception", str(e), traceback.format_exc())
    
    # 6. TEST EXPORT FUNCTIONALITY
    print("\n📤 6. TESTING EXPORT FUNCTIONALITY")
    print("-" * 40)
    
    try:
        # First run a simple analysis to get results for export
        analysis_data = {
            'dataset_id': dataset_id,
            'analysis_type': 'descriptive',
            'columns': ['age', 'income'],
            'parameters': {}
        }
        
        response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data, timeout=20)
        if response.status_code == 200:
            analysis_result = response.json()
            
            if analysis_result.get('success'):
                # Test export formats
                export_formats = ['json', 'csv']
                
                for format_type in export_formats:
                    try:
                        export_response = requests.post(
                            f'http://localhost:8000/api/analysis/export/{format_type}',
                            json=analysis_result,
                            timeout=20
                        )
                        
                        if export_response.status_code == 200:
                            print(f"✅ {format_type.upper()} export successful")
                        else:
                            log_error("Export", f"{format_type} export failed: {export_response.status_code}",
                                     export_response.text[:300])
                    except Exception as e:
                        log_error("Export", f"{format_type} export exception", str(e))
            else:
                log_error("Export Setup", "Analysis for export failed", analysis_result.get('message', ''))
        else:
            log_error("Export Setup", f"Analysis for export HTTP error: {response.status_code}")
            
    except Exception as e:
        log_error("Export", "Export test exception", str(e), traceback.format_exc())
    
    # SUMMARY
    print("\n" + "=" * 60)
    print("📋 ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    
    if errors_found:
        print(f"❌ Found {len(errors_found)} errors:")
        for i, error in enumerate(errors_found, 1):
            print(f"\n{i}. {error['category']}: {error['error']}")
            if error['details']:
                print(f"   Details: {error['details']}")
            if error['traceback']:
                print(f"   Traceback: {error['traceback'][:200]}...")
    else:
        print("✅ No errors found! All functionality working correctly.")
    
    print(f"\n🕒 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return errors_found

if __name__ == "__main__":
    errors = debug_analysis_errors()
    if errors:
        print(f"\n🔧 {len(errors)} issues need to be fixed.")
    else:
        print("\n🎉 All analysis and data quality functionality working perfectly!")
