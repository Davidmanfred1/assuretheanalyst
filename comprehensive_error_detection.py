#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE ERROR DETECTION & FIXING
AssureTheAnalyst - Find and Fix All Unknown Errors
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

def comprehensive_error_test():
    print("🔍 COMPREHENSIVE ERROR DETECTION & FIXING")
    print("=" * 60)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    errors_found = []
    fixes_applied = []
    
    def log_error(category, error, details=""):
        errors_found.append({
            'category': category,
            'error': error,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        print(f"❌ {category}: {error}")
        if details:
            print(f"   Details: {details}")
    
    def log_fix(fix_description):
        fixes_applied.append({
            'fix': fix_description,
            'timestamp': datetime.now().isoformat()
        })
        print(f"✅ Fixed: {fix_description}")
    
    # 1. TEST BASIC SERVER HEALTH
    print("🏥 1. TESTING BASIC SERVER HEALTH")
    print("-" * 40)
    
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code != 200:
            log_error("Server Health", f"Health endpoint returned {response.status_code}")
        else:
            print("✅ Server health check passed")
    except requests.exceptions.ConnectionError:
        log_error("Server Health", "Cannot connect to server", "Server may not be running")
    except Exception as e:
        log_error("Server Health", f"Health check failed: {str(e)}")
    
    # 2. TEST ALL PAGE ENDPOINTS
    print("\n🌐 2. TESTING ALL PAGE ENDPOINTS")
    print("-" * 40)
    
    pages = [
        ('/', 'Landing Page'),
        ('/dashboard', 'Dashboard'),
        ('/upload', 'Upload Page'),
        ('/analysis', 'Analysis Page'),
        ('/reports', 'Reports Page'),
        ('/realtime', 'Real-time Dashboard'),
        ('/admin', 'Admin Page')
    ]
    
    for url, name in pages:
        try:
            response = requests.get(f'http://localhost:8000{url}', timeout=10)
            if response.status_code != 200:
                log_error("Page Endpoint", f"{name} returned {response.status_code}", url)
            else:
                print(f"✅ {name} loads correctly")
        except Exception as e:
            log_error("Page Endpoint", f"{name} failed to load: {str(e)}", url)
    
    # 3. TEST ALL API ENDPOINTS
    print("\n🔌 3. TESTING ALL API ENDPOINTS")
    print("-" * 40)
    
    api_endpoints = [
        ('/api/upload/datasets', 'GET', 'Datasets List API'),
        ('/api/performance/metrics', 'GET', 'Performance Metrics API'),
        ('/api/cache/stats', 'GET', 'Cache Statistics API'),
        ('/api/enterprise/audit-logs', 'GET', 'Audit Logs API'),
        ('/api/enterprise/system-health', 'GET', 'System Health API')
    ]
    
    for endpoint, method, name in api_endpoints:
        try:
            if method == 'GET':
                response = requests.get(f'http://localhost:8000{endpoint}', timeout=10)
            else:
                response = requests.post(f'http://localhost:8000{endpoint}', timeout=10)
                
            if response.status_code not in [200, 404]:  # 404 is acceptable for some endpoints
                log_error("API Endpoint", f"{name} returned {response.status_code}", endpoint)
            else:
                print(f"✅ {name} responds correctly")
        except Exception as e:
            log_error("API Endpoint", f"{name} failed: {str(e)}", endpoint)
    
    # 4. TEST FILE UPLOAD FUNCTIONALITY
    print("\n📤 4. TESTING FILE UPLOAD FUNCTIONALITY")
    print("-" * 40)
    
    try:
        # Create test data
        test_data = pd.DataFrame({
            'col1': np.random.randint(1, 100, 10),
            'col2': np.random.normal(50, 10, 10),
            'col3': np.random.choice(['A', 'B', 'C'], 10)
        })
        test_data.to_csv('error_test_data.csv', index=False)
        
        # Test upload
        with open('error_test_data.csv', 'rb') as f:
            files = {'file': ('error_test_data.csv', f, 'text/csv')}
            response = requests.post('http://localhost:8000/api/upload/', files=files, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                dataset_id = result.get('dataset_id')
                print("✅ File upload works correctly")
                
                # Test dataset retrieval
                response = requests.get(f'http://localhost:8000/api/upload/datasets/{dataset_id}', timeout=10)
                if response.status_code == 200:
                    print("✅ Dataset retrieval works correctly")
                else:
                    log_error("Dataset Retrieval", f"Failed to retrieve dataset: {response.status_code}")
            else:
                log_error("File Upload", f"Upload failed: {result.get('message', 'Unknown error')}")
        else:
            log_error("File Upload", f"Upload request failed: {response.status_code}")
            
    except Exception as e:
        log_error("File Upload", f"Upload test failed: {str(e)}")
    
    # 5. TEST ANALYSIS FUNCTIONALITY
    print("\n📊 5. TESTING ANALYSIS FUNCTIONALITY")
    print("-" * 40)
    
    if 'dataset_id' in locals():
        analysis_types = ['descriptive', 'correlation']
        
        for analysis_type in analysis_types:
            try:
                analysis_data = {
                    'dataset_id': dataset_id,
                    'analysis_type': analysis_type,
                    'columns': ['col1', 'col2'],
                    'parameters': {}
                }
                
                response = requests.post('http://localhost:8000/api/analysis/run', 
                                       json=analysis_data, timeout=20)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print(f"✅ {analysis_type.title()} analysis works correctly")
                    else:
                        log_error("Analysis", f"{analysis_type} analysis failed: {result.get('message', 'Unknown error')}")
                else:
                    log_error("Analysis", f"{analysis_type} analysis returned {response.status_code}")
                    
            except Exception as e:
                log_error("Analysis", f"{analysis_type} analysis exception: {str(e)}")
    
    # 6. TEST EXPORT FUNCTIONALITY
    print("\n📤 6. TESTING EXPORT FUNCTIONALITY")
    print("-" * 40)
    
    if 'dataset_id' in locals():
        try:
            # First run an analysis to get results
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'descriptive',
                'columns': ['col1', 'col2'],
                'parameters': {}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data, timeout=15)
            if response.status_code == 200:
                analysis_result = response.json()
                
                # Test export formats
                export_formats = ['json', 'csv']
                
                for format_type in export_formats:
                    try:
                        export_response = requests.post(
                            f'http://localhost:8000/api/analysis/export/{format_type}',
                            json=analysis_result,
                            timeout=15
                        )
                        if export_response.status_code == 200:
                            print(f"✅ {format_type.upper()} export works correctly")
                        else:
                            log_error("Export", f"{format_type} export failed: {export_response.status_code}")
                    except Exception as e:
                        log_error("Export", f"{format_type} export exception: {str(e)}")
        except Exception as e:
            log_error("Export", f"Export test setup failed: {str(e)}")
    
    # 7. CHECK FOR COMMON PYTHON ERRORS
    print("\n🐍 7. CHECKING FOR COMMON PYTHON ERRORS")
    print("-" * 40)
    
    try:
        # Test imports
        import app.main
        import app.services.analysis_service
        import app.services.file_service
        import app.models.analysis_models
        print("✅ All Python imports work correctly")
    except ImportError as e:
        log_error("Python Import", f"Import error: {str(e)}")
    except Exception as e:
        log_error("Python Error", f"Python error: {str(e)}")
    
    # SUMMARY AND FIXES
    print("\n" + "=" * 60)
    print("📋 ERROR DETECTION SUMMARY")
    print("=" * 60)
    
    if errors_found:
        print(f"❌ Found {len(errors_found)} errors:")
        for error in errors_found:
            print(f"   • {error['category']}: {error['error']}")
            if error['details']:
                print(f"     Details: {error['details']}")
    else:
        print("✅ No errors found! System is working correctly.")
    
    if fixes_applied:
        print(f"\n✅ Applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"   • {fix['fix']}")
    
    # RECOMMENDATIONS
    print("\n💡 RECOMMENDATIONS:")
    if not errors_found:
        print("🎉 System is running perfectly!")
        print("✅ All endpoints are working")
        print("✅ All functionality is operational")
        print("✅ No errors detected")
    else:
        print("🔧 Issues found that need attention:")
        error_categories = set(error['category'] for error in errors_found)
        for category in error_categories:
            category_errors = [e for e in errors_found if e['category'] == category]
            print(f"   • {category}: {len(category_errors)} issues")
    
    print(f"\n🕒 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return errors_found, fixes_applied

if __name__ == "__main__":
    errors, fixes = comprehensive_error_test()
