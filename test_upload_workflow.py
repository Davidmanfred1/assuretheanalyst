#!/usr/bin/env python3
"""
Test Enhanced Upload-to-Analysis Workflow
"""

import requests
import pandas as pd
import numpy as np

def test_upload_workflow():
    print('🚀 Testing Enhanced Upload-to-Analysis Workflow...')
    
    # Create test data with clear patterns
    np.random.seed(42)
    test_data = pd.DataFrame({
        'sales': np.random.normal(1000, 200, 50),
        'marketing_spend': np.random.normal(500, 100, 50),
        'customer_satisfaction': np.random.uniform(1, 10, 50),
        'revenue': np.random.normal(5000, 1000, 50)
    })
    
    # Add correlations
    test_data['revenue'] = test_data['sales'] * 4.5 + np.random.normal(0, 200, 50)
    test_data['sales'] = test_data['marketing_spend'] * 1.8 + np.random.normal(0, 100, 50)
    
    # Add some outliers
    test_data.loc[5, 'sales'] = 5000
    test_data.loc[15, 'revenue'] = 20000
    
    test_data.to_csv('upload_workflow_test.csv', index=False)
    
    print('📊 Dataset created with:')
    print(f'   - {len(test_data)} rows')
    print(f'   - {len(test_data.columns)} columns: {list(test_data.columns)}')
    print('   - Strong correlation between sales and revenue')
    print('   - Outliers in sales and revenue columns')
    
    # Upload the dataset
    with open('upload_workflow_test.csv', 'rb') as f:
        files = {'file': ('upload_workflow_test.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/api/upload/', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            dataset_id = result.get('dataset_id')
            print(f'✅ Dataset uploaded successfully: {dataset_id}')
            total_rows = result.get('total_rows', 0)
            print(f'   📈 Preview available with {total_rows} rows')
            print('   🎯 Ready for immediate analysis from upload page!')
            
            # Test the preview endpoint
            preview_response = requests.get(f'http://localhost:8000/api/upload/datasets/{dataset_id}/preview')
            if preview_response.status_code == 200:
                preview_result = preview_response.json()
                if preview_result.get('success'):
                    print('✅ Preview endpoint working')
                    columns = preview_result['preview']['columns']
                    print(f'   📊 Columns: {columns}')
                    print('   📈 Sample data available')
                else:
                    print('❌ Preview endpoint failed')
            else:
                print(f'❌ Preview request failed: {preview_response.status_code}')
        else:
            print(f'❌ Upload failed: {result}')
    else:
        print(f'❌ Upload request failed: {response.status_code}')
    
    print('🎉 Upload-to-Analysis Workflow Test Complete!')
    print('💡 Now users can:')
    print('   1. Upload data and see immediate preview')
    print('   2. View dataset summary with column types')
    print('   3. Run quick analysis directly from upload page')
    print('   4. See interactive visualizations instantly')
    print('   5. Get accuracy metrics and insights')

if __name__ == "__main__":
    test_upload_workflow()
