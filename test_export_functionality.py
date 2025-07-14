#!/usr/bin/env python3
"""
Test Export Functionality
"""

import requests
import pandas as pd
import numpy as np

def test_export_functionality():
    print('📤 Testing Export Functionality...')
    
    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'sales': np.random.normal(1000, 200, 50),
        'marketing_spend': np.random.normal(500, 100, 50),
        'revenue': np.random.normal(5000, 1000, 50)
    })
    
    test_data.to_csv('export_test_data.csv', index=False)
    
    # Upload the dataset
    with open('export_test_data.csv', 'rb') as f:
        files = {'file': ('export_test_data.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/api/upload/', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            dataset_id = result.get('dataset_id')
            print(f'✅ Dataset uploaded: {dataset_id}')
            
            # Run descriptive analysis
            print('📊 Running descriptive analysis...')
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'descriptive',
                'columns': ['sales', 'marketing_spend', 'revenue'],
                'parameters': {}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
            if response.status_code == 200:
                analysis_result = response.json()
                if analysis_result.get('success'):
                    print('✅ Analysis completed successfully!')
                    
                    # Test different export formats
                    export_formats = ['pdf', 'excel', 'csv', 'json']
                    
                    for format_type in export_formats:
                        print(f'\n📤 Testing {format_type.upper()} export...')
                        
                        try:
                            export_response = requests.post(
                                f'http://localhost:8000/api/analysis/export/{format_type}',
                                json=analysis_result,
                                headers={'Content-Type': 'application/json'}
                            )
                            
                            if export_response.status_code == 200:
                                # Save the exported file
                                filename = f'test_export.{format_type if format_type != "excel" else "xlsx"}'
                                with open(filename, 'wb') as f:
                                    f.write(export_response.content)
                                print(f'✅ {format_type.upper()} export successful! Saved as {filename}')
                            else:
                                print(f'❌ {format_type.upper()} export failed: {export_response.status_code}')
                                print(f'   Error: {export_response.text}')
                        
                        except Exception as e:
                            print(f'❌ {format_type.upper()} export error: {e}')
                    
                else:
                    print(f'❌ Analysis failed: {analysis_result.get("message", "Unknown error")}')
            else:
                print(f'❌ Analysis request failed: {response.status_code}')
        else:
            print(f'❌ Upload failed: {result}')
    else:
        print(f'❌ Upload request failed: {response.status_code}')
    
    print('\n🎉 Export Functionality Testing Complete!')
    print('💡 Export features available:')
    print('   📄 PDF - Professional formatted reports')
    print('   📊 Excel - Structured data in spreadsheets')
    print('   📋 CSV - Simple comma-separated values')
    print('   💾 JSON - Raw data for developers')

if __name__ == "__main__":
    test_export_functionality()
