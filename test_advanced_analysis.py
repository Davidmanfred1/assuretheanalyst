#!/usr/bin/env python3
"""
Test Advanced Analysis Features
"""

import requests
import pandas as pd
import numpy as np

def test_advanced_analysis():
    print('🚀 Testing Advanced Analysis Features...')
    
    # Create test data suitable for advanced analysis
    np.random.seed(42)
    n_samples = 100
    
    # Create correlated features for clustering and PCA
    base_data = np.random.randn(n_samples, 2)
    
    test_data = pd.DataFrame({
        'feature1': base_data[:, 0],
        'feature2': base_data[:, 1],
        'feature3': base_data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2,  # Correlated with feature1
        'feature4': base_data[:, 1] * 0.6 + np.random.randn(n_samples) * 0.3,  # Correlated with feature2
        'feature5': np.random.randn(n_samples),  # Independent
        'time_series': np.cumsum(np.random.randn(n_samples) * 0.1) + np.arange(n_samples) * 0.05  # Trending time series
    })
    
    test_data.to_csv('advanced_analysis_test.csv', index=False)
    
    print('📊 Dataset created with:')
    print(f'   - {len(test_data)} rows')
    print(f'   - {len(test_data.columns)} features with correlations and trends')
    
    # Upload the dataset
    with open('advanced_analysis_test.csv', 'rb') as f:
        files = {'file': ('advanced_analysis_test.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/api/upload/', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            dataset_id = result.get('dataset_id')
            print(f'✅ Dataset uploaded: {dataset_id}')
            
            # Test Advanced Clustering
            print('\n🌟 Testing Advanced Clustering...')
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'advanced-clustering',
                'columns': ['feature1', 'feature2', 'feature3', 'feature4'],
                'parameters': {}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print('✅ Advanced Clustering Analysis completed!')
                    results = result.get('results', {})
                    if 'kmeans' in results:
                        kmeans = results['kmeans']
                        print(f'   🔵 K-Means: {kmeans.get("optimal_clusters", "N/A")} clusters')
                        print(f'   📊 Silhouette Score: {kmeans.get("silhouette_score", "N/A"):.3f}')
                    if 'dbscan' in results:
                        dbscan = results['dbscan']
                        print(f'   🔴 DBSCAN: {dbscan.get("n_clusters", "N/A")} clusters, {dbscan.get("n_noise_points", "N/A")} noise points')
                else:
                    print(f'❌ Advanced Clustering failed: {result.get("message", "Unknown error")}')
            else:
                print(f'❌ Advanced Clustering request failed: {response.status_code}')
            
            # Test Forecasting
            print('\n🔮 Testing Forecasting...')
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'forecasting',
                'columns': ['time_series'],
                'parameters': {'forecast_periods': 10}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print('✅ Forecasting Analysis completed!')
                    results = result.get('results', {})
                    if 'time_series' in results:
                        ts_result = results['time_series']
                        print(f'   📈 Trend Direction: {ts_result.get("trend_direction", "N/A")}')
                        print(f'   📊 R-squared: {ts_result.get("r_squared", "N/A"):.3f}')
                        print(f'   🔮 Forecast Periods: {ts_result.get("forecast_periods", "N/A")}')
                else:
                    print(f'❌ Forecasting failed: {result.get("message", "Unknown error")}')
            else:
                print(f'❌ Forecasting request failed: {response.status_code}')
            
            # Test Dimensionality Reduction
            print('\n📉 Testing Dimensionality Reduction...')
            analysis_data = {
                'dataset_id': dataset_id,
                'analysis_type': 'dimensionality-reduction',
                'columns': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
                'parameters': {}
            }
            
            response = requests.post('http://localhost:8000/api/analysis/run', json=analysis_data)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print('✅ Dimensionality Reduction completed!')
                    results = result.get('results', {})
                    if 'pca' in results:
                        pca_result = results['pca']
                        print(f'   📊 Components for 95% variance: {pca_result.get("components_for_95_variance", "N/A")}')
                        print(f'   📈 Total components: {pca_result.get("total_components", "N/A")}')
                        if 'explained_variance_ratio' in pca_result:
                            var_ratios = pca_result['explained_variance_ratio'][:3]  # First 3
                            print(f'   🔍 First 3 components explain: {[f"{v:.1%}" for v in var_ratios]}')
                else:
                    print(f'❌ Dimensionality Reduction failed: {result.get("message", "Unknown error")}')
            else:
                print(f'❌ Dimensionality Reduction request failed: {response.status_code}')
                
        else:
            print(f'❌ Upload failed: {result}')
    else:
        print(f'❌ Upload request failed: {response.status_code}')
    
    print('\n🎉 Advanced Analysis Testing Complete!')
    print('💡 New capabilities available:')
    print('   🌟 Advanced Clustering with K-Means and DBSCAN')
    print('   🔮 Time Series Forecasting with trend analysis')
    print('   📉 PCA Dimensionality Reduction with variance analysis')
    print('   📊 Interactive visualizations for all advanced analyses')

if __name__ == "__main__":
    test_advanced_analysis()
