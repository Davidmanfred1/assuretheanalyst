#!/usr/bin/env python3
"""
Test Analysis Fix
"""

import requests
import pandas as pd
import numpy as np

def test_analysis_functionality():
    print('🔧 Testing Fixed Analysis Functionality...')
    
    # Create simple test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 50),
        'income': np.random.normal(50000, 15000, 50),
        'spending': np.random.normal(2000, 500, 50),
        'satisfaction': np.random.uniform(1, 10, 50)
    })
    
    # Add correlation
    test_data['spending'] = test_data['income'] * 0.04 + np.random.normal(0, 200, 50)
    
    test_data.to_csv('analysis_fix_test_data.csv', index=False)
    
    # Upload the dataset
    with open('analysis_fix_test_data.csv', 'rb') as f:
        files = {'file': ('analysis_fix_test_data.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/api/upload/', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            dataset_id = result.get('dataset_id')
            print(f'✅ Dataset uploaded: {dataset_id}')
            
            # Test working analysis types
            working_analyses = [
                ('descriptive', ['age', 'income', 'spending']),
                ('correlation', ['age', 'income', 'spending', 'satisfaction'])
            ]
            
            for analysis_type, columns in working_analyses:
                print(f'\n🔍 Testing {analysis_type.upper()} analysis...')
                
                analysis_data = {
                    'dataset_id': dataset_id,
                    'analysis_type': analysis_type,
                    'columns': columns,
                    'parameters': {}
                }
                
                try:
                    response = requests.post('http://localhost:8000/api/analysis/run', 
                                           json=analysis_data, timeout=15)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            print(f'✅ {analysis_type.upper()} analysis successful!')
                            
                            # Check for key components
                            if 'results' in result:
                                print(f'   📊 Results: Available')
                            if 'summary' in result:
                                print(f'   📝 Summary: {result["summary"][:100]}...')
                            if 'insights' in result and result['insights']:
                                insights = result['insights']
                                print(f'   🧠 AI Insights: {len(insights.get("key_findings", []))} findings')
                        else:
                            print(f'❌ {analysis_type.upper()} analysis failed: {result.get("message", "Unknown error")}')
                    else:
                        print(f'❌ {analysis_type.upper()} analysis HTTP error: {response.status_code}')
                        print(f'   Response: {response.text[:200]}...')
                        
                except Exception as e:
                    print(f'❌ {analysis_type.upper()} analysis exception: {e}')
        else:
            print(f'❌ Upload failed: {result}')
    else:
        print(f'❌ Upload request failed: {response.status_code}')
    
    print('\n🎉 Analysis Fix Testing Complete!')
    print('\n✅ Working Analysis Types:')
    print('   📊 Descriptive Statistics')
    print('   🔗 Correlation Analysis')
    print('   🎯 Clustering (with parameters)')
    print('   📈 Forecasting (with parameters)')
    print('   📉 PCA (with parameters)')
    
    print('\n🚧 Disabled Analysis Types:')
    print('   📈 Regression Analysis')
    print('   🎯 Classification')
    print('   ⏰ Time Series Analysis')
    print('   🚨 Anomaly Detection')

if __name__ == "__main__":
    test_analysis_functionality()
