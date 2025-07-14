#!/usr/bin/env python3
"""
Test Dataset Dropdown Issue
"""

import requests
import pandas as pd
import numpy as np
import time

def test_dataset_dropdown():
    print('🔍 Testing Dataset Dropdown Issue...')
    
    # Create a simple test dataset
    np.random.seed(42)
    test_data = pd.DataFrame({
        'test_column_1': np.random.randint(1, 100, 20),
        'test_column_2': np.random.normal(50, 10, 20),
        'test_column_3': np.random.choice(['A', 'B', 'C'], 20)
    })
    
    test_data.to_csv('dropdown_test_data.csv', index=False)
    print('✅ Test dataset created')
    
    # Upload the dataset
    print('\n📤 Uploading dataset...')
    with open('dropdown_test_data.csv', 'rb') as f:
        files = {'file': ('dropdown_test_data.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/api/upload/', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            dataset_id = result.get('dataset_id')
            print(f'✅ Dataset uploaded successfully: {dataset_id}')
            
            # Wait a moment for the upload to complete
            time.sleep(1)
            
            # Test the datasets API
            print('\n🔍 Testing datasets API...')
            response = requests.get('http://localhost:8000/api/upload/datasets')
            
            if response.status_code == 200:
                result = response.json()
                print(f'✅ Datasets API response: {result.get("success")}')
                
                if result.get('success') and result.get('datasets'):
                    datasets = result['datasets']
                    print(f'📊 Found {len(datasets)} datasets:')
                    
                    for i, dataset in enumerate(datasets):
                        print(f'  {i+1}. {dataset["info"]["name"]} (ID: {dataset["dataset_id"][:8]}...)')
                        
                    # Check if our new dataset is in the list
                    our_dataset = next((d for d in datasets if d['dataset_id'] == dataset_id), None)
                    if our_dataset:
                        print(f'✅ Our test dataset found in the list!')
                        print(f'   Name: {our_dataset["info"]["name"]}')
                        print(f'   Rows: {our_dataset["info"]["rows"]}')
                        print(f'   Columns: {our_dataset["info"]["columns"]}')
                    else:
                        print('❌ Our test dataset NOT found in the list!')
                        
                else:
                    print('❌ No datasets returned or API error')
            else:
                print(f'❌ Datasets API failed: {response.status_code}')
                
            # Test the specific dataset endpoint
            print(f'\n🔍 Testing specific dataset endpoint...')
            response = requests.get(f'http://localhost:8000/api/upload/datasets/{dataset_id}')
            
            if response.status_code == 200:
                result = response.json()
                print(f'✅ Specific dataset API works: {result.get("success")}')
            else:
                print(f'❌ Specific dataset API failed: {response.status_code}')
                
        else:
            print(f'❌ Upload failed: {result.get("message", "Unknown error")}')
    else:
        print(f'❌ Upload request failed: {response.status_code}')
        print(f'Response: {response.text}')
    
    print('\n🎯 SUMMARY:')
    print('If the datasets API is working but the dropdown is empty, the issue is likely:')
    print('1. JavaScript errors in the browser console')
    print('2. CORS issues (unlikely since same origin)')
    print('3. JavaScript execution timing issues')
    print('4. Missing AssureTheAnalyst object (now fixed)')
    print('\n💡 NEXT STEPS:')
    print('1. Open browser developer tools (F12)')
    print('2. Go to http://localhost:8000/analysis')
    print('3. Check the Console tab for JavaScript errors')
    print('4. Check the Network tab to see if the API call is made')
    print('5. Look for any error messages in the console')

if __name__ == "__main__":
    test_dataset_dropdown()
