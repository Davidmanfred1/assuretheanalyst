#!/usr/bin/env python3
"""
🎉 FINAL COMPREHENSIVE TEST SUITE
AssureTheAnalyst - Complete Platform Testing
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

def run_comprehensive_test():
    print("🎉 ASSURETHEANALYST - FINAL COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"🕒 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test Results Tracking
    test_results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'test_details': []
    }
    
    def log_test(test_name, status, details=""):
        test_results['total_tests'] += 1
        if status:
            test_results['passed_tests'] += 1
            print(f"✅ {test_name}")
        else:
            test_results['failed_tests'] += 1
            print(f"❌ {test_name} - {details}")
        test_results['test_details'].append({
            'test': test_name,
            'status': 'PASS' if status else 'FAIL',
            'details': details
        })
    
    # 1. SYSTEM HEALTH CHECK
    print("🏥 SYSTEM HEALTH CHECK")
    print("-" * 30)
    
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        log_test("Health Endpoint", response.status_code == 200)
    except Exception as e:
        log_test("Health Endpoint", False, str(e))
    
    try:
        response = requests.get('http://localhost:8000/', timeout=5)
        log_test("Landing Page", response.status_code == 200)
    except Exception as e:
        log_test("Landing Page", False, str(e))
    
    # 2. DATA UPLOAD & PROCESSING
    print("\n📤 DATA UPLOAD & PROCESSING")
    print("-" * 30)
    
    # Create comprehensive test dataset
    np.random.seed(42)
    test_data = pd.DataFrame({
        'customer_id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'spending': np.random.normal(2000, 500, 100),
        'satisfaction': np.random.uniform(1, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some correlations and patterns
    test_data['spending'] = test_data['income'] * 0.04 + np.random.normal(0, 200, 100)
    test_data['satisfaction'] = 10 - (test_data['age'] - 50) * 0.05 + np.random.normal(0, 1, 100)
    
    # Add some quality issues for testing
    test_data.loc[5:10, 'income'] = None  # Missing values
    test_data.loc[15, 'age'] = 200  # Outlier
    test_data.loc[20, 'satisfaction'] = 15  # Invalid range
    
    test_data.to_csv('comprehensive_test_data.csv', index=False)
    
    dataset_id = None
    try:
        with open('comprehensive_test_data.csv', 'rb') as f:
            files = {'file': ('comprehensive_test_data.csv', f, 'text/csv')}
            response = requests.post('http://localhost:8000/api/upload/', files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                dataset_id = result.get('dataset_id')
                log_test("File Upload", True)
            else:
                log_test("File Upload", False, result.get('message', 'Unknown error'))
        else:
            log_test("File Upload", False, f"HTTP {response.status_code}")
    except Exception as e:
        log_test("File Upload", False, str(e))
    
    if dataset_id:
        # Test dataset preview
        try:
            response = requests.get(f'http://localhost:8000/api/upload/datasets/{dataset_id}/preview', timeout=5)
            log_test("Dataset Preview", response.status_code == 200)
        except Exception as e:
            log_test("Dataset Preview", False, str(e))
    
    # 3. ANALYSIS ENGINE TESTING
    print("\n📊 ANALYSIS ENGINE TESTING")
    print("-" * 30)
    
    if dataset_id:
        analysis_types = [
            ('descriptive', ['age', 'income', 'spending', 'satisfaction']),
            ('correlation', ['age', 'income', 'spending', 'satisfaction'])
        ]
        
        for analysis_type, columns in analysis_types:
            try:
                analysis_data = {
                    'dataset_id': dataset_id,
                    'analysis_type': analysis_type,
                    'columns': columns,
                    'parameters': {}
                }
                
                response = requests.post('http://localhost:8000/api/analysis/run', 
                                       json=analysis_data, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        # Check for AI insights
                        has_insights = 'insights' in result and result['insights'] is not None
                        log_test(f"{analysis_type.title()} Analysis", True)
                        log_test(f"{analysis_type.title()} AI Insights", has_insights)
                    else:
                        log_test(f"{analysis_type.title()} Analysis", False, result.get('message', 'Unknown error'))
                else:
                    log_test(f"{analysis_type.title()} Analysis", False, f"HTTP {response.status_code}")
            except Exception as e:
                log_test(f"{analysis_type.title()} Analysis", False, str(e))
    
    # 4. EXPORT FUNCTIONALITY
    print("\n📤 EXPORT FUNCTIONALITY")
    print("-" * 30)
    
    if dataset_id:
        # First run an analysis to get results for export
        try:
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'descriptive',
                'columns': ['age', 'income'],
                'parameters': {}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data, timeout=10)
            if response.status_code == 200:
                analysis_result = response.json()
                
                # Test export formats
                export_formats = ['json', 'csv']  # Test lighter formats first
                
                for format_type in export_formats:
                    try:
                        export_response = requests.post(
                            f'http://localhost:8000/api/analysis/export/{format_type}',
                            json=analysis_result,
                            timeout=10
                        )
                        log_test(f"{format_type.upper()} Export", export_response.status_code == 200)
                    except Exception as e:
                        log_test(f"{format_type.upper()} Export", False, str(e))
        except Exception as e:
            log_test("Export Setup", False, str(e))
    
    # 5. DATA QUALITY ASSESSMENT
    print("\n🛡️ DATA QUALITY ASSESSMENT")
    print("-" * 30)
    
    if dataset_id:
        try:
            response = requests.post(f'http://localhost:8000/api/upload/quality-check/{dataset_id}', timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    quality_report = result.get('quality_report', {})
                    overall_score = quality_report.get('overall_score', 0)
                    log_test("Data Quality Check", True)
                    log_test("Quality Score Generated", overall_score > 0)
                else:
                    log_test("Data Quality Check", False, result.get('message', 'Unknown error'))
            else:
                log_test("Data Quality Check", False, f"HTTP {response.status_code}")
        except Exception as e:
            log_test("Data Quality Check", False, str(e))
    
    # 6. UI PAGES ACCESSIBILITY
    print("\n🌐 UI PAGES ACCESSIBILITY")
    print("-" * 30)
    
    pages = [
        ('/', 'Landing Page'),
        ('/dashboard', 'Dashboard'),
        ('/upload', 'Upload Page'),
        ('/analysis', 'Analysis Page'),
        ('/reports', 'Reports Page')
    ]
    
    for url, page_name in pages:
        try:
            response = requests.get(f'http://localhost:8000{url}', timeout=5)
            log_test(f"{page_name} Accessibility", response.status_code == 200)
        except Exception as e:
            log_test(f"{page_name} Accessibility", False, str(e))
    
    # 7. API ENDPOINTS
    print("\n🔌 API ENDPOINTS")
    print("-" * 30)
    
    api_endpoints = [
        ('/api/upload/datasets', 'Datasets API'),
        ('/api/performance/metrics', 'Performance API'),
        ('/api/cache/stats', 'Cache API')
    ]
    
    for endpoint, name in api_endpoints:
        try:
            response = requests.get(f'http://localhost:8000{endpoint}', timeout=5)
            log_test(f"{name}", response.status_code == 200)
        except Exception as e:
            log_test(f"{name}", False, str(e))
    
    # FINAL RESULTS
    print("\n" + "=" * 60)
    print("🎉 FINAL TEST RESULTS")
    print("=" * 60)
    
    total = test_results['total_tests']
    passed = test_results['passed_tests']
    failed = test_results['failed_tests']
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print()
    
    if success_rate >= 90:
        print("🎉 EXCELLENT! Platform is production-ready!")
        print("🚀 All critical systems are functioning properly.")
    elif success_rate >= 80:
        print("✅ GOOD! Platform is mostly functional with minor issues.")
        print("🔧 Some non-critical features may need attention.")
    elif success_rate >= 70:
        print("⚠️ FAIR! Platform has some issues that should be addressed.")
        print("🛠️ Several features need debugging.")
    else:
        print("❌ POOR! Platform needs significant work before deployment.")
        print("🚨 Critical issues detected.")
    
    print()
    print("📋 DETAILED RESULTS:")
    for test in test_results['test_details']:
        status_icon = "✅" if test['status'] == 'PASS' else "❌"
        print(f"   {status_icon} {test['test']}")
        if test['details'] and test['status'] == 'FAIL':
            print(f"      └─ {test['details']}")
    
    print()
    print(f"🕒 Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 AssureTheAnalyst Comprehensive Testing Complete!")

if __name__ == "__main__":
    run_comprehensive_test()
