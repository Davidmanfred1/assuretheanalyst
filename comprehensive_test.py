#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for AssureTheAnalyst
Tests every feature and functionality to ensure excellence
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys

BASE_URL = "http://localhost:8000"

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def success(self, test_name):
        self.passed += 1
        print(f"   ✅ {test_name}")
    
    def failure(self, test_name, error=""):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"   ❌ {test_name} - {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} tests passed")
        if self.failed > 0:
            print(f"FAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'='*60}")
        return self.failed == 0

def test_basic_endpoints(results):
    """Test all basic API endpoints"""
    print("🔍 Testing Basic API Endpoints...")
    
    endpoints = [
        ("/health", "Health Check"),
        ("/api/upload/datasets", "Datasets API"),
        ("/api/cache/stats", "Cache Statistics"),
        ("/api/performance/metrics", "Performance Metrics"),
        ("/api/enterprise/security/dashboard", "Security Dashboard"),
        ("/api/etl/status", "ETL Status"),
        ("/api/etl/transformations", "ETL Transformations"),
        ("/api/ai-insights/automated/test", "AI Insights (expected to fail gracefully)"),
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if endpoint == "/api/ai-insights/automated/test":
                    # This should fail gracefully
                    if "error" in response.text.lower() or not data.get("success", True):
                        results.success(f"{name} (graceful failure)")
                    else:
                        results.failure(name, "Should fail gracefully for non-existent dataset")
                else:
                    results.success(name)
            elif response.status_code == 404 and "ai-insights" in endpoint:
                results.success(f"{name} (graceful 404)")
            else:
                results.failure(name, f"Status {response.status_code}")
        except Exception as e:
            results.failure(name, str(e))

def test_web_pages(results):
    """Test all web pages are accessible"""
    print("\n🌐 Testing Web Pages...")
    
    pages = [
        ("/", "Home Page"),
        ("/upload", "Upload Page"),
        ("/analysis", "Analysis Page"),
        ("/reports", "Reports Page"),
        ("/realtime", "Real-time Page"),
        ("/admin", "Admin Page"),
        ("/dashboard", "Dashboard Page"),
    ]
    
    for page, name in pages:
        try:
            response = requests.get(f"{BASE_URL}{page}", timeout=10)
            if response.status_code == 200 and "html" in response.headers.get("content-type", "").lower():
                results.success(name)
            else:
                results.failure(name, f"Status {response.status_code}")
        except Exception as e:
            results.failure(name, str(e))

def test_file_upload_and_analysis(results):
    """Test complete file upload and analysis workflow"""
    print("\n📁 Testing File Upload & Analysis Workflow...")
    
    # Create comprehensive test dataset
    np.random.seed(42)
    test_data = pd.DataFrame({
        'employee_id': range(1, 101),
        'name': [f'Employee_{i}' for i in range(1, 101)],
        'age': np.random.randint(22, 65, 100),
        'salary': np.random.normal(60000, 15000, 100).astype(int),
        'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], 100),
        'experience_years': np.random.randint(0, 20, 100),
        'performance_score': np.random.uniform(60, 100, 100),
        'satisfaction_rating': np.random.randint(1, 6, 100),
        'remote_work': np.random.choice([True, False], 100),
        'hire_date': pd.date_range('2020-01-01', '2023-12-31', periods=100)
    })
    
    test_file = Path("comprehensive_test_data.csv")
    test_data.to_csv(test_file, index=False)
    
    try:
        # Test file upload
        with open(test_file, 'rb') as f:
            files = {'file': ('comprehensive_test_data.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/api/upload/", files=files, timeout=30)
        
        if response.status_code == 200:
            upload_result = response.json()
            if upload_result.get("success"):
                dataset_id = upload_result.get("dataset_id")
                results.success("File Upload")
                
                # Test dataset preview
                try:
                    response = requests.get(f"{BASE_URL}/api/upload/datasets/{dataset_id}/preview", timeout=10)
                    if response.status_code == 200:
                        preview_data = response.json()
                        if preview_data.get("success"):
                            results.success("Dataset Preview")
                        else:
                            results.failure("Dataset Preview", "API returned success=False")
                    else:
                        results.failure("Dataset Preview", f"Status {response.status_code}")
                except Exception as e:
                    results.failure("Dataset Preview", str(e))
                
                # Test various analysis types
                analysis_types = [
                    ("descriptive", ["age", "salary", "performance_score"]),
                    ("correlation", ["age", "salary", "experience_years", "performance_score"]),
                ]
                
                for analysis_type, columns in analysis_types:
                    try:
                        analysis_data = {
                            "dataset_id": dataset_id,
                            "analysis_type": analysis_type,
                            "columns": columns,
                            "parameters": {}
                        }
                        
                        response = requests.post(f"{BASE_URL}/api/analysis/run", json=analysis_data, timeout=30)
                        if response.status_code == 200:
                            analysis_result = response.json()
                            if analysis_result.get("success"):
                                results.success(f"{analysis_type.title()} Analysis")
                            else:
                                results.failure(f"{analysis_type.title()} Analysis", "API returned success=False")
                        else:
                            results.failure(f"{analysis_type.title()} Analysis", f"Status {response.status_code}")
                    except Exception as e:
                        results.failure(f"{analysis_type.title()} Analysis", str(e))
                
                # Test AI Insights
                try:
                    response = requests.get(f"{BASE_URL}/api/ai-insights/automated/{dataset_id}", timeout=30)
                    if response.status_code == 200:
                        insights_result = response.json()
                        if insights_result.get("success"):
                            results.success("AI Insights Generation")
                        else:
                            results.failure("AI Insights Generation", "API returned success=False")
                    else:
                        results.failure("AI Insights Generation", f"Status {response.status_code}")
                except Exception as e:
                    results.failure("AI Insights Generation", str(e))
                
            else:
                results.failure("File Upload", upload_result.get("message", "Unknown error"))
        else:
            results.failure("File Upload", f"Status {response.status_code}")
    
    except Exception as e:
        results.failure("File Upload", str(e))
    
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()

def test_etl_functionality(results):
    """Test ETL pipeline functionality"""
    print("\n🔧 Testing ETL Functionality...")
    
    try:
        # Test getting transformation types
        response = requests.get(f"{BASE_URL}/api/etl/transformations", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and "transformations" in data:
                results.success("ETL Transformations List")
            else:
                results.failure("ETL Transformations List", "Invalid response format")
        else:
            results.failure("ETL Transformations List", f"Status {response.status_code}")
        
        # Test creating a pipeline
        pipeline_data = {
            "name": "Test Pipeline",
            "description": "Automated test pipeline"
        }
        
        response = requests.post(f"{BASE_URL}/api/etl/pipelines", json=pipeline_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                pipeline_id = data.get("pipeline_id")
                results.success("ETL Pipeline Creation")
                
                # Test adding a step to the pipeline
                step_data = {
                    "transformation_type": "clean_missing",
                    "parameters": {"strategy": "drop"},
                    "order": 1
                }
                
                response = requests.post(f"{BASE_URL}/api/etl/pipelines/{pipeline_id}/steps", json=step_data, timeout=10)
                if response.status_code == 200:
                    step_result = response.json()
                    if step_result.get("success"):
                        results.success("ETL Pipeline Step Addition")
                    else:
                        results.failure("ETL Pipeline Step Addition", "API returned success=False")
                else:
                    results.failure("ETL Pipeline Step Addition", f"Status {response.status_code}")
            else:
                results.failure("ETL Pipeline Creation", "API returned success=False")
        else:
            results.failure("ETL Pipeline Creation", f"Status {response.status_code}")
    
    except Exception as e:
        results.failure("ETL Functionality", str(e))

def test_cache_operations(results):
    """Test cache operations"""
    print("\n💾 Testing Cache Operations...")
    
    try:
        # Test cache stats
        response = requests.get(f"{BASE_URL}/api/cache/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "total_entries" in data and "hit_rate_percent" in data:
                results.success("Cache Statistics")
            else:
                results.failure("Cache Statistics", "Missing required fields")
        else:
            results.failure("Cache Statistics", f"Status {response.status_code}")
        
        # Test cache optimization
        response = requests.post(f"{BASE_URL}/api/cache/optimize", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                results.success("Cache Optimization")
            else:
                results.failure("Cache Optimization", "API returned success=False")
        else:
            results.failure("Cache Optimization", f"Status {response.status_code}")
    
    except Exception as e:
        results.failure("Cache Operations", str(e))

def test_real_time_features(results):
    """Test real-time features"""
    print("\n🔴 Testing Real-time Features...")
    
    try:
        # Test real-time stats
        response = requests.get(f"{BASE_URL}/api/realtime/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "total_connections" in data:
                results.success("Real-time Statistics")
            else:
                results.failure("Real-time Statistics", "Missing required fields")
        else:
            results.failure("Real-time Statistics", f"Status {response.status_code}")
    
    except Exception as e:
        results.failure("Real-time Features", str(e))

def test_security_features(results):
    """Test security features"""
    print("\n🔒 Testing Security Features...")
    
    try:
        # Test security dashboard
        response = requests.get(f"{BASE_URL}/api/enterprise/security/dashboard", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "total_events_24h" in data and "compliance_status" in data:
                results.success("Security Dashboard")
            else:
                results.failure("Security Dashboard", "Missing required fields")
        else:
            results.failure("Security Dashboard", f"Status {response.status_code}")
        
        # Test security alerts
        response = requests.get(f"{BASE_URL}/api/enterprise/alerts", timeout=10)
        if response.status_code == 200:
            # Should return a list (empty is fine)
            data = response.json()
            if isinstance(data, list):
                results.success("Security Alerts")
            else:
                results.failure("Security Alerts", "Should return a list")
        else:
            results.failure("Security Alerts", f"Status {response.status_code}")
    
    except Exception as e:
        results.failure("Security Features", str(e))

def test_performance_monitoring(results):
    """Test performance monitoring"""
    print("\n⚡ Testing Performance Monitoring...")
    
    try:
        # Test performance metrics
        response = requests.get(f"{BASE_URL}/api/performance/metrics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "cpu" in data and "memory" in data:
                results.success("Performance Metrics")
            else:
                results.failure("Performance Metrics", "Missing CPU or memory data")
        else:
            results.failure("Performance Metrics", f"Status {response.status_code}")
    
    except Exception as e:
        results.failure("Performance Monitoring", str(e))

def test_ui_responsiveness(results):
    """Test UI responsiveness and JavaScript functionality"""
    print("\n🎨 Testing UI Responsiveness...")
    
    # Test that pages load with proper content-type and contain expected elements
    pages_to_test = [
        ("/", ["AssureTheAnalyst", "Welcome", "Get Started"]),
        ("/upload", ["Upload Dataset", "Select File", "form"]),
        ("/analysis", ["Data Analysis", "Run Analysis", "dataset"]),
        ("/dashboard", ["Dashboard", "Statistics", "Performance"]),
        ("/admin", ["Admin Dashboard", "System", "Security"]),
    ]
    
    for page, expected_content in pages_to_test:
        try:
            response = requests.get(f"{BASE_URL}{page}", timeout=10)
            if response.status_code == 200:
                content = response.text.lower()
                if all(item.lower() in content for item in expected_content):
                    results.success(f"UI Content - {page}")
                else:
                    missing = [item for item in expected_content if item.lower() not in content]
                    results.failure(f"UI Content - {page}", f"Missing: {missing}")
            else:
                results.failure(f"UI Content - {page}", f"Status {response.status_code}")
        except Exception as e:
            results.failure(f"UI Content - {page}", str(e))

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 COMPREHENSIVE EXCELLENCE TEST FOR ASSURETHEANALYST")
    print("=" * 60)
    print("Testing every feature to ensure perfect functionality...")
    print("=" * 60)
    
    results = TestResults()
    
    # Run all test suites
    test_basic_endpoints(results)
    test_web_pages(results)
    test_file_upload_and_analysis(results)
    test_etl_functionality(results)
    test_cache_operations(results)
    test_real_time_features(results)
    test_security_features(results)
    test_performance_monitoring(results)
    test_ui_responsiveness(results)
    
    # Print final results
    success = results.summary()
    
    if success:
        print("\n🎉 EXCELLENCE ACHIEVED! 🎉")
        print("✨ All features are working perfectly!")
        print("🚀 AssureTheAnalyst is ready for production!")
        print("\n🌐 Access your application:")
        print(f"   • Main App: {BASE_URL}")
        print(f"   • Dashboard: {BASE_URL}/dashboard")
        print(f"   • Upload: {BASE_URL}/upload")
        print(f"   • Analysis: {BASE_URL}/analysis")
        print(f"   • Real-time: {BASE_URL}/realtime")
        print(f"   • Admin: {BASE_URL}/admin")
    else:
        print("\n⚠️  Some issues found - see details above")
        print("🔧 Please address the failed tests")
    
    return success

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
