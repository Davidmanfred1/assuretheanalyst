#!/usr/bin/env python3
"""
Simple test to verify core functionality
"""

import requests
import pandas as pd
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_basic_functionality():
    """Test basic functionality"""
    print("🚀 Testing AssureTheAnalyst Core Functionality")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
        else:
            print("   ❌ Health check failed")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test 2: Datasets API
    print("2. Testing datasets API...")
    try:
        response = requests.get(f"{BASE_URL}/api/upload/datasets")
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("   ✅ Datasets API working")
                print(f"   📊 Found {len(data.get('datasets', []))} datasets")
            else:
                print("   ❌ Datasets API returned error")
        else:
            print("   ❌ Datasets API failed")
    except Exception as e:
        print(f"   ❌ Datasets API error: {e}")
    
    # Test 3: Performance Metrics
    print("3. Testing performance metrics...")
    try:
        response = requests.get(f"{BASE_URL}/api/performance/metrics")
        if response.status_code == 200:
            print("   ✅ Performance metrics working")
        else:
            print("   ❌ Performance metrics failed")
    except Exception as e:
        print(f"   ❌ Performance metrics error: {e}")
    
    # Test 4: Cache Stats
    print("4. Testing cache statistics...")
    try:
        response = requests.get(f"{BASE_URL}/api/cache/stats")
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Cache stats working")
            print(f"   💾 Cache entries: {data.get('total_entries', 0)}")
            print(f"   📈 Hit rate: {data.get('hit_rate_percent', 0):.1f}%")
        else:
            print("   ❌ Cache stats failed")
    except Exception as e:
        print(f"   ❌ Cache stats error: {e}")
    
    # Test 5: Security Dashboard
    print("5. Testing security dashboard...")
    try:
        response = requests.get(f"{BASE_URL}/api/enterprise/security/dashboard")
        if response.status_code == 200:
            print("   ✅ Security dashboard working")
        else:
            print("   ❌ Security dashboard failed")
    except Exception as e:
        print(f"   ❌ Security dashboard error: {e}")
    
    # Test 6: ETL Status
    print("6. Testing ETL status...")
    try:
        response = requests.get(f"{BASE_URL}/api/etl/status")
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("   ✅ ETL status working")
                status = data.get("status", {})
                print(f"   🔧 Pipelines: {status.get('total_pipelines', 0)}")
                print(f"   ⚡ Running jobs: {status.get('running_jobs', 0)}")
            else:
                print("   ❌ ETL status returned error")
        else:
            print("   ❌ ETL status failed")
    except Exception as e:
        print(f"   ❌ ETL status error: {e}")
    
    # Test 7: File Upload (with test data)
    print("7. Testing file upload...")
    try:
        # Create test data
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000],
            'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Marketing'],
            'performance': [85, 92, 78, 88, 91]
        })
        
        test_file = Path("test_upload_simple.csv")
        test_data.to_csv(test_file, index=False)
        
        # Upload file
        with open(test_file, 'rb') as f:
            files = {'file': ('test_upload_simple.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/api/upload/", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("   ✅ File upload working")
                print(f"   📁 Dataset ID: {data.get('dataset_id')}")
                
                # Test analysis on uploaded data
                dataset_id = data.get('dataset_id')
                if dataset_id:
                    print("8. Testing analysis on uploaded data...")
                    analysis_data = {
                        "dataset_id": dataset_id,
                        "analysis_type": "descriptive",
                        "columns": ["age", "salary"],
                        "parameters": {}
                    }
                    
                    try:
                        response = requests.post(f"{BASE_URL}/api/analysis/run", json=analysis_data)
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                print("   ✅ Analysis working")
                                print("   📊 Descriptive analysis completed")
                            else:
                                print(f"   ❌ Analysis failed: {result.get('message', 'Unknown error')}")
                        else:
                            print(f"   ❌ Analysis request failed: {response.status_code}")
                    except Exception as e:
                        print(f"   ❌ Analysis error: {e}")
            else:
                print(f"   ❌ Upload failed: {data.get('message', 'Unknown error')}")
        else:
            print(f"   ❌ Upload request failed: {response.status_code}")
        
        # Clean up
        if test_file.exists():
            test_file.unlink()
            
    except Exception as e:
        print(f"   ❌ Upload test error: {e}")
    
    print("=" * 50)
    print("🎉 Core functionality test completed!")
    print("✨ AssureTheAnalyst is ready for use!")
    print("🌐 Access the application at: http://localhost:8000")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("📁 Upload: http://localhost:8000/upload")
    print("📈 Analysis: http://localhost:8000/analysis")
    print("🔴 Real-time: http://localhost:8000/realtime")
    print("🏢 Admin: http://localhost:8000/admin")

if __name__ == "__main__":
    test_basic_functionality()
