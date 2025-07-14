#!/usr/bin/env python3
"""
Test Enhanced Analysis Features
"""

import requests
import pandas as pd
import numpy as np

def test_enhanced_analysis():
    print('📊 Testing Enhanced Analysis Features...')
    
    # Create a comprehensive test dataset
    np.random.seed(42)
    test_data = pd.DataFrame({
        'sales': np.random.normal(1000, 200, 100),
        'marketing_spend': np.random.normal(500, 100, 100),
        'customer_satisfaction': np.random.uniform(1, 10, 100),
        'temperature': np.random.normal(25, 5, 100),
        'revenue': np.random.normal(5000, 1000, 100)
    })
    
    # Add some correlations
    test_data['revenue'] = test_data['sales'] * 4.5 + np.random.normal(0, 200, 100)
    test_data['sales'] = test_data['marketing_spend'] * 1.8 + np.random.normal(0, 100, 100)
    
    # Add some outliers
    test_data.loc[5, 'sales'] = 5000  # Outlier
    test_data.loc[15, 'revenue'] = 20000  # Outlier
    
    test_data.to_csv('enhanced_analysis_test.csv', index=False)
    
    # Upload the dataset
    with open('enhanced_analysis_test.csv', 'rb') as f:
        files = {'file': ('enhanced_analysis_test.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/api/upload/', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            dataset_id = result.get('dataset_id')
            print(f'✅ Dataset uploaded: {dataset_id}')
            
            # Test enhanced descriptive analysis
            print('🔍 Testing Enhanced Descriptive Analysis...')
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'descriptive',
                'columns': ['sales', 'marketing_spend', 'revenue'],
                'parameters': {}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print('✅ Enhanced Descriptive Analysis completed!')
                    results = result.get('results', {})
                    
                    # Check for enhanced features
                    if 'dataset_info' in results:
                        info = results['dataset_info']
                        print(f'   📈 Dataset Info: {info["total_rows"]} rows, {info["numeric_columns"]} numeric columns')
                    if 'outliers' in results:
                        print(f'   🚨 Outliers detected in {len(results["outliers"])} columns')
                    if 'confidence_intervals_95' in results:
                        print('   📊 95% Confidence intervals calculated')
                    if 'visualizations' in results:
                        viz_types = list(results['visualizations'].keys())
                        print(f'   📈 Visualizations generated: {viz_types}')
                else:
                    print(f'❌ Descriptive analysis failed: {result}')
            else:
                print(f'❌ Descriptive analysis request failed: {response.status_code}')
            
            # Test enhanced correlation analysis
            print('🔗 Testing Enhanced Correlation Analysis...')
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'correlation',
                'columns': ['sales', 'marketing_spend', 'revenue'],
                'parameters': {}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print('✅ Enhanced Correlation Analysis completed!')
                    results = result.get('results', {})
                    
                    # Check for enhanced features
                    if 'correlation_matrices' in results:
                        matrices = results['correlation_matrices']
                        print(f'   📊 Correlation methods: {list(matrices.keys())}')
                    if 'correlations_by_strength' in results:
                        by_strength = results['correlations_by_strength']
                        strong_count = len(by_strength.get('strong', []))
                        moderate_count = len(by_strength.get('moderate', []))
                        print(f'   💪 Strong correlations: {strong_count}')
                        print(f'   🔗 Moderate correlations: {moderate_count}')
                    if 'p_values' in results:
                        print('   📈 P-values calculated for significance testing')
                    if 'visualizations' in results:
                        viz_types = list(results['visualizations'].keys())
                        print(f'   📈 Visualizations: {viz_types}')
                else:
                    print(f'❌ Correlation analysis failed: {result}')
            else:
                print(f'❌ Correlation analysis request failed: {response.status_code}')
        else:
            print(f'❌ Upload failed: {result}')
    else:
        print(f'❌ Upload request failed: {response.status_code}')
    
    print('🎉 Enhanced Analysis Testing Complete!')

if __name__ == "__main__":
    test_enhanced_analysis()
