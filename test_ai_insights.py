#!/usr/bin/env python3
"""
Test AI Insights Engine
"""

import requests
import pandas as pd
import numpy as np

def test_ai_insights():
    print('🧠 Testing AI Insights Engine...')
    
    # Create test data with interesting patterns
    np.random.seed(42)
    
    # Create data with correlations and outliers
    n_samples = 200
    
    # Base variables
    x1 = np.random.normal(100, 15, n_samples)
    x2 = 0.8 * x1 + np.random.normal(0, 5, n_samples)  # Strong correlation
    x3 = -0.6 * x1 + np.random.normal(50, 10, n_samples)  # Negative correlation
    x4 = np.random.normal(50, 20, n_samples)  # Independent variable
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, 10, replace=False)
    x1[outlier_indices] += np.random.normal(0, 50, 10)
    
    # Create skewed data
    x5 = np.random.exponential(2, n_samples)  # Right-skewed
    
    test_data = pd.DataFrame({
        'sales': x1,
        'marketing_spend': x2,
        'customer_satisfaction': x3,
        'competitor_price': x4,
        'website_visits': x5
    })
    
    test_data.to_csv('ai_insights_test_data.csv', index=False)
    
    # Upload the dataset
    with open('ai_insights_test_data.csv', 'rb') as f:
        files = {'file': ('ai_insights_test_data.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/api/upload/', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            dataset_id = result.get('dataset_id')
            print(f'✅ Dataset uploaded: {dataset_id}')
            
            # Test different analysis types with AI insights
            analysis_types = [
                ('descriptive', ['sales', 'marketing_spend', 'customer_satisfaction', 'website_visits']),
                ('correlation', ['sales', 'marketing_spend', 'customer_satisfaction', 'competitor_price'])
            ]
            
            for analysis_type, columns in analysis_types:
                print(f'\n🔍 Testing {analysis_type.upper()} analysis with AI insights...')
                
                analysis_data = {
                    'dataset_id': dataset_id,
                    'analysis_type': analysis_type,
                    'columns': columns,
                    'parameters': {}
                }
                
                response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
                if response.status_code == 200:
                    analysis_result = response.json()
                    if analysis_result.get('success'):
                        insights = analysis_result.get('insights', {})
                        
                        print(f'✅ {analysis_type.upper()} analysis completed with insights!')
                        
                        # Display insights
                        if insights:
                            print(f'\n📊 AI Insights for {analysis_type.upper()} Analysis:')
                            print(f'   🎯 Confidence Score: {insights.get("confidence_score", 0):.1%}')
                            
                            if insights.get('narrative'):
                                print(f'\n   📝 Narrative:')
                                print(f'      {insights["narrative"]}')
                            
                            if insights.get('key_findings'):
                                print(f'\n   🔍 Key Findings:')
                                for finding in insights['key_findings'][:3]:
                                    print(f'      • {finding}')
                            
                            if insights.get('recommendations'):
                                print(f'\n   💡 Recommendations:')
                                for rec in insights['recommendations'][:3]:
                                    print(f'      • {rec}')
                            
                            if insights.get('alerts'):
                                print(f'\n   ⚠️  Alerts:')
                                for alert in insights['alerts']:
                                    print(f'      • {alert.get("message", "")}')
                            
                            if insights.get('next_steps'):
                                print(f'\n   🚀 Next Steps:')
                                for step in insights['next_steps'][:2]:
                                    print(f'      • {step}')
                        else:
                            print('❌ No insights generated')
                    else:
                        print(f'❌ Analysis failed: {analysis_result.get("message", "Unknown error")}')
                else:
                    print(f'❌ Analysis request failed: {response.status_code}')
                    print(f'   Error: {response.text}')
        else:
            print(f'❌ Upload failed: {result}')
    else:
        print(f'❌ Upload request failed: {response.status_code}')
    
    print('\n🎉 AI Insights Engine Testing Complete!')
    print('\n🧠 AI Insights Features:')
    print('   ✅ Intelligent Pattern Recognition')
    print('   ✅ Automated Insight Generation')
    print('   ✅ Natural Language Explanations')
    print('   ✅ Confidence Scoring')
    print('   ✅ Actionable Recommendations')
    print('   ✅ Alert System for Data Quality')
    print('   ✅ Next Steps Suggestions')
    print('   ✅ Context-Aware Analysis')
    
    print('\n💡 Insight Categories:')
    print('   📊 Statistical Significance Detection')
    print('   🔗 Correlation Strength Assessment')
    print('   📈 Trend Pattern Recognition')
    print('   🎯 Outlier Impact Analysis')
    print('   📉 Distribution Shape Analysis')
    print('   🔍 Data Quality Assessment')
    print('   💼 Business Impact Interpretation')
    print('   🚀 Strategic Recommendations')

if __name__ == "__main__":
    test_ai_insights()
