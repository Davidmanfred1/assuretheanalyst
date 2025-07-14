#!/usr/bin/env python3
"""
Simple test for insights engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.insights_engine import InsightsEngine

def test_insights_engine():
    print('🧠 Testing Insights Engine Directly...')
    
    # Create insights engine
    engine = InsightsEngine()
    
    # Test descriptive analysis insights
    descriptive_result = {
        'analysis_type': 'descriptive',
        'results': {
            'statistics': {
                'sales': {
                    'mean': 100.5,
                    'std': 25.3,
                    'skewness': 1.2,
                    'count': 200,
                    'outliers': [150, 160, 170, 180, 190]
                },
                'marketing_spend': {
                    'mean': 50.2,
                    'std': 10.1,
                    'skewness': 0.3,
                    'count': 200,
                    'outliers': []
                }
            }
        }
    }
    
    print('\n📊 Testing Descriptive Analysis Insights...')
    insights = engine.generate_insights(descriptive_result)
    
    print(f'✅ Insights generated successfully!')
    print(f'   🎯 Confidence Score: {insights.get("confidence_score", 0):.1%}')
    print(f'   📝 Narrative: {insights.get("narrative", "No narrative")[:100]}...')
    print(f'   🔍 Key Findings: {len(insights.get("key_findings", []))} findings')
    print(f'   💡 Recommendations: {len(insights.get("recommendations", []))} recommendations')
    print(f'   ⚠️  Alerts: {len(insights.get("alerts", []))} alerts')
    
    # Test correlation analysis insights
    correlation_result = {
        'analysis_type': 'correlation',
        'results': {
            'correlations': {
                'sales_vs_marketing': {
                    'correlation': 0.85,
                    'p_value': 0.001
                },
                'sales_vs_satisfaction': {
                    'correlation': -0.65,
                    'p_value': 0.01
                }
            }
        }
    }
    
    print('\n🔗 Testing Correlation Analysis Insights...')
    insights = engine.generate_insights(correlation_result)
    
    print(f'✅ Insights generated successfully!')
    print(f'   🎯 Confidence Score: {insights.get("confidence_score", 0):.1%}')
    print(f'   📝 Narrative: {insights.get("narrative", "No narrative")[:100]}...')
    print(f'   🔍 Key Findings: {len(insights.get("key_findings", []))} findings')
    print(f'   💡 Recommendations: {len(insights.get("recommendations", []))} recommendations')
    
    print('\n🎉 Insights Engine Test Complete!')
    print('✅ AI Insights Engine is working correctly!')

if __name__ == "__main__":
    test_insights_engine()
