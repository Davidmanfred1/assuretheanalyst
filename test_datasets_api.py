#!/usr/bin/env python3
"""
Test Datasets API Response
"""

import requests
import json

def test_datasets_api():
    print('🔍 Testing Datasets API...')
    
    try:
        response = requests.get('http://localhost:8000/api/upload/datasets')
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print(f'Response: {json.dumps(result, indent=2)}')
            
            if result.get('success'):
                datasets = result.get('datasets', [])
                print(f'\n📊 Found {len(datasets)} datasets:')
                
                for i, dataset in enumerate(datasets):
                    print(f'  {i+1}. Dataset ID: {dataset.get("dataset_id")}')
                    info = dataset.get('info', {})
                    print(f'     Name: {info.get("name")}')
                    print(f'     Rows: {info.get("rows")}')
                    print(f'     Columns: {info.get("columns")}')
                    print(f'     Type: {info.get("file_type")}')
                    print()
            else:
                print('❌ API returned success=false')
        else:
            print(f'❌ API request failed: {response.text}')
            
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    test_datasets_api()
