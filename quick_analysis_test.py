import requests
import pandas as pd

# Create test data
test_data = pd.DataFrame({
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 70000, 55000, 65000]
})
test_data.to_csv('quick_test.csv', index=False)

# Upload
with open('quick_test.csv', 'rb') as f:
    files = {'file': ('quick_test.csv', f, 'text/csv')}
    response = requests.post('http://localhost:8000/api/upload/', files=files)

if response.status_code == 200:
    data = response.json()
    if data.get('success'):
        dataset_id = data.get('dataset_id')
        print(f'Upload successful: {dataset_id}')
        
        # Test analysis
        analysis_data = {
            'dataset_id': dataset_id,
            'analysis_type': 'descriptive',
            'columns': ['age', 'salary'],
            'parameters': {}
        }
        
        response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
        print(f'Analysis status: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print('Analysis successful!')
            print(f'Success: {result.get("success")}')
        else:
            print(f'Analysis failed: {response.text}')
    else:
        print(f'Upload failed: {data}')
else:
    print(f'Upload request failed: {response.status_code}')
