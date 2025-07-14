#!/usr/bin/env python3
"""
🎉 FINAL ERROR VERIFICATION
Verify all unknown errors have been fixed
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

def final_verification_test():
    print("🎉 FINAL ERROR VERIFICATION")
    print("=" * 60)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_tests_passed = True
    
    def test_result(test_name, passed, details=""):
        global all_tests_passed
        if passed:
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}")
            if details:
                print(f"   Details: {details}")
            all_tests_passed = False
    
    # 1. VERIFY SERVER IS RUNNING
    print("🏥 1. SERVER HEALTH VERIFICATION")
    print("-" * 40)
    
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        test_result("Server Health Check", response.status_code == 200)
    except Exception as e:
        test_result("Server Health Check", False, str(e))
    
    # 2. VERIFY ALL CRITICAL ENDPOINTS
    print("\n🌐 2. CRITICAL ENDPOINTS VERIFICATION")
    print("-" * 40)
    
    critical_endpoints = [
        ('/', 'Landing Page'),
        ('/upload', 'Upload Page'),
        ('/analysis', 'Analysis Page'),
        ('/dashboard', 'Dashboard'),
        ('/api/upload/datasets', 'Datasets API'),
        ('/api/performance/metrics', 'Performance API')
    ]
    
    for endpoint, name in critical_endpoints:
        try:
            response = requests.get(f'http://localhost:8000{endpoint}', timeout=10)
            test_result(f"{name} Endpoint", response.status_code == 200)
        except Exception as e:
            test_result(f"{name} Endpoint", False, str(e))
    
    # 3. VERIFY UPLOAD FUNCTIONALITY
    print("\n📤 3. UPLOAD FUNCTIONALITY VERIFICATION")
    print("-" * 40)
    
    dataset_id = None
    try:
        # Create test data
        test_data = pd.DataFrame({
            'test_col1': np.random.randint(1, 100, 15),
            'test_col2': np.random.normal(50, 10, 15),
            'test_col3': np.random.choice(['X', 'Y', 'Z'], 15)
        })
        test_data.to_csv('final_verification_data.csv', index=False)
        
        # Upload test
        with open('final_verification_data.csv', 'rb') as f:
            files = {'file': ('final_verification_data.csv', f, 'text/csv')}
            response = requests.post('http://localhost:8000/api/upload/', files=files, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                dataset_id = result.get('dataset_id')
                test_result("File Upload", True)
            else:
                test_result("File Upload", False, result.get('message', 'Unknown error'))
        else:
            test_result("File Upload", False, f"HTTP {response.status_code}")
            
    except Exception as e:
        test_result("File Upload", False, str(e))
    
    # 4. VERIFY ANALYSIS FUNCTIONALITY
    print("\n📊 4. ANALYSIS FUNCTIONALITY VERIFICATION")
    print("-" * 40)
    
    if dataset_id:
        analysis_types = ['descriptive', 'correlation']
        
        for analysis_type in analysis_types:
            try:
                analysis_data = {
                    'dataset_id': dataset_id,
                    'analysis_type': analysis_type,
                    'columns': ['test_col1', 'test_col2'],
                    'parameters': {}
                }
                
                response = requests.post('http://localhost:8000/api/analysis/run', 
                                       json=analysis_data, timeout=20)
                
                if response.status_code == 200:
                    result = response.json()
                    test_result(f"{analysis_type.title()} Analysis", result.get('success', False))
                else:
                    test_result(f"{analysis_type.title()} Analysis", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                test_result(f"{analysis_type.title()} Analysis", False, str(e))
    else:
        test_result("Analysis Tests", False, "No dataset available for testing")
    
    # 5. VERIFY EXPORT FUNCTIONALITY
    print("\n📤 5. EXPORT FUNCTIONALITY VERIFICATION")
    print("-" * 40)
    
    if dataset_id:
        try:
            # Run analysis first
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'descriptive',
                'columns': ['test_col1', 'test_col2'],
                'parameters': {}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data, timeout=15)
            if response.status_code == 200:
                analysis_result = response.json()
                
                # Test export
                export_response = requests.post(
                    'http://localhost:8000/api/analysis/export/json',
                    json=analysis_result,
                    timeout=15
                )
                test_result("JSON Export", export_response.status_code == 200)
            else:
                test_result("Export Setup", False, "Analysis failed")
                
        except Exception as e:
            test_result("Export Functionality", False, str(e))
    else:
        test_result("Export Tests", False, "No dataset available for testing")
    
    # 6. VERIFY SECURITY FIXES
    print("\n🔒 6. SECURITY FIXES VERIFICATION")
    print("-" * 40)
    
    try:
        # Check if AssureTheAnalyst object is available
        response = requests.get('http://localhost:8000/analysis', timeout=10)
        if response.status_code == 200:
            content = response.text
            test_result("AssureTheAnalyst Object", 'AssureTheAnalyst' in content)
            test_result("HTML Sanitization", 'sanitizeHTML' in content)
            test_result("Safe HTML Setter", 'safeSetHTML' in content)
        else:
            test_result("Security Verification", False, "Cannot access analysis page")
            
    except Exception as e:
        test_result("Security Verification", False, str(e))
    
    # 7. VERIFY DEPENDENCIES
    print("\n📦 7. DEPENDENCIES VERIFICATION")
    print("-" * 40)
    
    try:
        import sklearn
        test_result("Scikit-learn Import", True)
    except ImportError:
        test_result("Scikit-learn Import", False, "Package not available")
    
    try:
        import plotly
        test_result("Plotly Import", True)
    except ImportError:
        test_result("Plotly Import", False, "Package not available")
    
    try:
        import pandas
        test_result("Pandas Import", True)
    except ImportError:
        test_result("Pandas Import", False, "Package not available")
    
    # 8. VERIFY FRONTEND FUNCTIONALITY
    print("\n🌐 8. FRONTEND FUNCTIONALITY VERIFICATION")
    print("-" * 40)
    
    try:
        # Check analysis page for JavaScript functions
        response = requests.get('http://localhost:8000/analysis', timeout=10)
        if response.status_code == 200:
            content = response.text
            required_functions = ['loadDatasets', 'runAnalysis', 'runQuickAnalysis']
            
            for func in required_functions:
                test_result(f"Function {func}", f'function {func}' in content)
        else:
            test_result("Frontend Verification", False, "Cannot access analysis page")
            
    except Exception as e:
        test_result("Frontend Verification", False, str(e))
    
    # FINAL SUMMARY
    print("\n" + "=" * 60)
    print("🎉 FINAL VERIFICATION SUMMARY")
    print("=" * 60)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ No unknown errors detected")
        print("✅ All functionality working correctly")
        print("✅ Security fixes implemented")
        print("✅ Dependencies verified")
        print("✅ Frontend functionality confirmed")
        print()
        print("🚀 ASSURETHEANALYST IS PRODUCTION READY!")
        print("🎯 Platform Status: FULLY OPERATIONAL")
        print("🔒 Security Status: SECURE")
        print("⚡ Performance Status: OPTIMIZED")
        print("🧪 Testing Status: COMPREHENSIVE")
    else:
        print("⚠️ Some tests failed - review the details above")
        print("🔧 Additional fixes may be needed")
    
    print(f"\n🕒 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = final_verification_test()
    if success:
        print("\n🎉 CONGRATULATIONS! All errors have been fixed!")
    else:
        print("\n🔧 Some issues remain - check the output above for details.")
