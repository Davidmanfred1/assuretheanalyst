#!/usr/bin/env python3
"""
🔍 FRONTEND ERROR DETECTION
Check for JavaScript errors, missing functions, and UI issues
"""

import requests
import re

def check_frontend_errors():
    print("🔍 FRONTEND ERROR DETECTION")
    print("=" * 50)
    
    errors_found = []
    fixes_needed = []
    
    def log_error(category, error, details=""):
        errors_found.append({'category': category, 'error': error, 'details': details})
        print(f"❌ {category}: {error}")
        if details:
            print(f"   Details: {details}")
    
    def log_fix_needed(fix):
        fixes_needed.append(fix)
        print(f"🔧 Fix needed: {fix}")
    
    # 1. CHECK HTML TEMPLATES FOR COMMON ISSUES
    print("\n📄 1. CHECKING HTML TEMPLATES")
    print("-" * 30)
    
    templates = [
        'templates/index.html',
        'templates/analysis.html',
        'templates/upload.html',
        'templates/dashboard.html',
        'templates/base.html'
    ]
    
    for template in templates:
        try:
            with open(template, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for common HTML issues
            if 'onclick=' in content and 'function' not in content:
                log_error("HTML", f"onclick handlers found but functions may be missing in {template}")
            
            # Check for missing closing tags
            open_divs = content.count('<div')
            close_divs = content.count('</div>')
            if open_divs != close_divs:
                log_error("HTML", f"Mismatched div tags in {template}", f"Open: {open_divs}, Close: {close_divs}")
            
            # Check for missing script tags
            if 'AssureTheAnalyst' in content and '<script>' not in content and template != 'templates/base.html':
                log_error("JavaScript", f"AssureTheAnalyst used but no script tag in {template}")
            
            print(f"✅ {template} structure looks good")
            
        except FileNotFoundError:
            log_error("File", f"Template not found: {template}")
        except Exception as e:
            log_error("File", f"Error reading {template}: {str(e)}")
    
    # 2. CHECK JAVASCRIPT FUNCTION DEFINITIONS
    print("\n🔧 2. CHECKING JAVASCRIPT FUNCTIONS")
    print("-" * 30)
    
    try:
        with open('templates/analysis.html', 'r', encoding='utf-8') as f:
            analysis_content = f.read()
        
        # Check for function definitions vs calls
        function_calls = re.findall(r'(\w+)\s*\(', analysis_content)
        function_definitions = re.findall(r'function\s+(\w+)\s*\(', analysis_content)
        
        # Common functions that should be defined
        required_functions = [
            'loadDatasets', 'onDatasetChange', 'runAnalysis', 
            'runQuickAnalysis', 'exportResults', 'getAIInsights'
        ]
        
        for func in required_functions:
            if func not in function_definitions:
                log_error("JavaScript", f"Required function '{func}' not defined in analysis.html")
            else:
                print(f"✅ Function '{func}' is defined")
        
        # Check for undefined function calls
        undefined_calls = []
        for call in set(function_calls):
            if call not in function_definitions and call not in ['console', 'fetch', 'document', 'Array', 'JSON', 'parseInt', 'parseFloat', 'setTimeout', 'addEventListener']:
                undefined_calls.append(call)
        
        if undefined_calls:
            log_error("JavaScript", f"Potentially undefined function calls: {', '.join(undefined_calls)}")
        
    except Exception as e:
        log_error("JavaScript", f"Error checking JavaScript functions: {str(e)}")
    
    # 3. CHECK CSS AND STYLING ISSUES
    print("\n🎨 3. CHECKING CSS AND STYLING")
    print("-" * 30)
    
    try:
        with open('templates/base.html', 'r', encoding='utf-8') as f:
            base_content = f.read()
        
        # Check for CSS framework loading
        if 'bootstrap' not in base_content.lower():
            log_error("CSS", "Bootstrap CSS not found in base template")
        else:
            print("✅ Bootstrap CSS is loaded")
        
        # Check for Font Awesome
        if 'font-awesome' not in base_content.lower() and 'fontawesome' not in base_content.lower():
            log_error("CSS", "Font Awesome not found in base template")
        else:
            print("✅ Font Awesome is loaded")
        
        # Check for theme system
        if 'themeManager' not in base_content:
            log_error("CSS", "Theme manager not found in base template")
        else:
            print("✅ Theme system is implemented")
            
    except Exception as e:
        log_error("CSS", f"Error checking CSS: {str(e)}")
    
    # 4. CHECK FOR MISSING DEPENDENCIES
    print("\n📦 4. CHECKING DEPENDENCIES")
    print("-" * 30)
    
    try:
        # Check if all required Python packages are available
        required_packages = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn',
            'plotly', 'jinja2', 'python-multipart', 'openpyxl', 'reportlab'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} is available")
            except ImportError:
                log_error("Dependencies", f"Missing package: {package}")
                log_fix_needed(f"Install {package}: pip install {package}")
                
    except Exception as e:
        log_error("Dependencies", f"Error checking dependencies: {str(e)}")
    
    # 5. CHECK API RESPONSE FORMATS
    print("\n🔌 5. CHECKING API RESPONSE FORMATS")
    print("-" * 30)
    
    try:
        # Test datasets API response format
        response = requests.get('http://localhost:8000/api/upload/datasets', timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields
            if 'success' not in data:
                log_error("API", "Datasets API missing 'success' field")
            elif 'datasets' not in data:
                log_error("API", "Datasets API missing 'datasets' field")
            else:
                print("✅ Datasets API response format is correct")
                
                # Check dataset structure
                if data['datasets']:
                    dataset = data['datasets'][0]
                    required_fields = ['dataset_id', 'info']
                    for field in required_fields:
                        if field not in dataset:
                            log_error("API", f"Dataset missing required field: {field}")
                        else:
                            print(f"✅ Dataset has required field: {field}")
        else:
            log_error("API", f"Datasets API returned {response.status_code}")
            
    except Exception as e:
        log_error("API", f"Error checking API responses: {str(e)}")
    
    # 6. CHECK FOR COMMON SECURITY ISSUES
    print("\n🔒 6. CHECKING SECURITY ISSUES")
    print("-" * 30)
    
    try:
        # Check for potential XSS vulnerabilities
        templates_to_check = ['templates/analysis.html', 'templates/upload.html']
        
        for template in templates_to_check:
            try:
                with open(template, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for innerHTML usage without sanitization
                if '.innerHTML' in content and 'sanitize' not in content.lower():
                    log_error("Security", f"Potential XSS risk in {template}: innerHTML usage without sanitization")
                
                # Check for eval usage
                if 'eval(' in content:
                    log_error("Security", f"Security risk in {template}: eval() usage found")
                
                print(f"✅ {template} security check passed")
                
            except FileNotFoundError:
                continue
                
    except Exception as e:
        log_error("Security", f"Error checking security: {str(e)}")
    
    # SUMMARY
    print("\n" + "=" * 50)
    print("📋 FRONTEND ERROR SUMMARY")
    print("=" * 50)
    
    if errors_found:
        print(f"❌ Found {len(errors_found)} frontend issues:")
        for error in errors_found:
            print(f"   • {error['category']}: {error['error']}")
    else:
        print("✅ No frontend errors found!")
    
    if fixes_needed:
        print(f"\n🔧 Fixes needed ({len(fixes_needed)}):")
        for fix in fixes_needed:
            print(f"   • {fix}")
    else:
        print("\n✅ No fixes needed!")
    
    return errors_found, fixes_needed

if __name__ == "__main__":
    errors, fixes = check_frontend_errors()
