#!/usr/bin/env python3
"""
Debug Correlation Analysis
"""

import requests
import pandas as pd
import numpy as np

def debug_correlation():
    print('🔍 Debugging Correlation Analysis...')
    
    # Create simple test data
    test_data = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],
        'c': [1, 3, 5, 7, 9]
    })
    test_data.to_csv('debug_corr_test.csv', index=False)
    
    # Upload
    with open('debug_corr_test.csv', 'rb') as f:
        files = {'file': ('debug_corr_test.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/api/upload/', files=files)
    
    if response.status_code == 200:
        result = response.json()
        dataset_id = result.get('dataset_id')
        print(f'Dataset uploaded: {dataset_id}')
        
        # Test correlation analysis with detailed error info
        analysis_data = {
            'dataset_id': dataset_id,
            'analysis_type': 'correlation',
            'columns': ['a', 'b', 'c'],
            'parameters': {}
        }
        
        response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
        print(f'Response status: {response.status_code}')
        if response.status_code != 200:
            print(f'Error response: {response.text}')
        else:
            result = response.json()
            success = result.get('success')
            print(f'Success: {success}')
            if not success:
                message = result.get('message', 'Unknown error')
                print(f'Error message: {message}')
            else:
                print('Correlation analysis successful!')
                print(f'Summary: {result.get("summary", "")}')
    else:
        print(f'Upload failed: {response.status_code}')

if __name__ == "__main__":
    debug_correlation()
