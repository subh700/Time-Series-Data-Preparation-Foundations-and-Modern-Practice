import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt

class BenchmarkDatasets:
    """Guide to essential time series forecasting benchmark datasets."""
    
    def __init__(self):
        self.datasets = self._catalog_datasets()
        self.download_links = self._setup_download_links()
    
    def _catalog_datasets(self) -> Dict[str, Any]:
        """Catalog essential benchmark datasets with characteristics."""
        
        return {
            "electricity_transformer_temperature": {
                "name": "ETT (ETTh1, ETTh2, ETTm1, ETTm2)",
                "description": "Electricity transformer temperature data from China",
                "variables": 7,
                "frequency": ["15min (ETTm)", "1hour (ETTh)"],
                "timespan": "July 2016 - July 2018 (2 years)",
                "samples": {"ETTh1": 17420, "ETTh2": 17420, "ETTm1": 69680, "ETTm2": 69680},
                "domain": "Energy/Electricity",
                "characteristics": {
                    "seasonality": "Strong daily and weekly patterns",
                    "trend": "Long-term trends present",
                    "missing_values": "None",
                    "complexity": "Medium"
                },
                "typical_horizons": [96, 192, 336, 720],
                "split_ratio": "6:2:2 (train:validation:test)",
                "best_for": "Long-term forecasting, multivariate analysis",
                "download_source": "https://github.com/zhouhaoyi/ETDataset"
            },
            
            "electricity_consumption": {
                "name": "Electricity",
                "description": "Hourly electricity consumption of 321 clients",
                "variables": 321,
                "frequency": "1 hour",
                "timespan": "2012-2014 (3 years)",
                "samples": 26304,
                "domain": "Energy/Electricity",
                "characteristics": {
                    "seasonality": "Very strong daily and weekly patterns",
                    "trend": "Moderate trends",
                    "missing_values": "Minimal",
                    "complexity": "High (many variables)"
                },
                "typical_horizons": [24, 48, 168, 720],
                "split_ratio": "7:1:2",
                "best_for": "High-dimensional forecasting, correlation analysis",
                "download_source": "UCI ML Repository"
            },
            
            "traffic_occupancy": {
                "name": "Traffic",
                "description": "Road occupancy rates on San Francisco Bay area freeways",
                "variables": 862,
                "frequency": "1 hour", 
                "timespan": "2015-2016 (2 years)",
                "samples": 17544,
                "domain": "Transportation",
                "characteristics": {
                    "seasonality": "Strong daily and weekly patterns",
                    "trend": "Minimal long-term trends",
                    "missing_values": "Some",
                    "complexity": "High (spatial correlations)"
                },
                "typical_horizons": [12, 24, 48, 168],
                "split_ratio": "7:1:2",
                "best_for": "Spatial-temporal forecasting, real-time applications",
                "download_source": "California PeMS"
            },
            
            "weather_dataset": {
                "name": "Weather",
                "description": "Meteorological data from Max Planck Institute",
                "variables": 21,
                "frequency": "10 minutes",
                "timespan": "2009-2016 (7 years)",
                "samples": 52696,
                "domain": "Meteorology",
                "characteristics": {
                    "seasonality": "Strong seasonal patterns",
                    "trend": "Climate trends",
                    "missing_values": "Minimal",
                    "complexity": "Medium"
                },
                "typical_horizons": [144, 288, 576, 1152],
                "split_ratio": "7:1:2",
                "best_for": "Environmental forecasting, climate analysis",
                "download_source": "BGC Jena Research Centre"
            },
            
            "exchange_rates": {
                "name": "Exchange",
                "description": "Daily exchange rates of 8 countries",
                "variables": 8,
                "frequency": "1 day",
                "timespan": "1990-2016 (26 years)",
                "samples": 7588,
                "domain": "Finance",
                "characteristics": {
                    "seasonality": "Weekly patterns",
                    "trend": "Long-term economic trends",
                    "missing_values": "Weekends/holidays",
                    "complexity": "Medium"
                },
                "typical_horizons": [30, 60, 90, 180],
                "split_ratio": "7:1:2",
                "best_for": "Financial forecasting, economic analysis",
                "download_source": "Federal Reserve Economic Data"
            },
            
            "solar_power": {
                "name": "Solar",
                "description": "Solar power generation from 137 PV plants",
                "variables": 137,
                "frequency": "10 minutes",
                "timespan": "2006 (1 year)",
                "samples": 52560,
                "domain": "Energy/Renewable",
                "characteristics": {
                    "seasonality": "Strong daily patterns, weather dependent",
                    "trend": "Seasonal trends",
                    "missing_values": "Weather-related gaps",
                    "complexity": "High (weather correlation)"
                },
                "typical_horizons": [144, 288, 576, 1152],
                "split_ratio": "6:2:2",
                "best_for": "Renewable energy forecasting, weather impact analysis",
                "download_source": "NREL Solar Integration National Dataset"
            },
            
            "illness_surveillance": {
                "name": "ILI (Influenza-Like Illness)",
                "description": "Weekly influenza-like illness data from CDC",
                "variables": 7,
                "frequency": "1 week",
                "timespan": "2002-2021 (19 years)",
                "samples": 966,
                "domain": "Healthcare",
                "characteristics": {
                    "seasonality": "Strong seasonal flu patterns",
                    "trend": "Long-term health trends",
                    "missing_values": "Rare",
                    "complexity": "Medium"
                },
                "typical_horizons": [4, 8, 12, 26],
                "split_ratio": "7:1:2",
                "best_for": "Healthcare forecasting, epidemic modeling",
                "download_source": "CDC FluView"
            }
        }
    
    def _setup_download_links(self) -> Dict[str, str]:
        """Setup direct download links for datasets."""
        
        return {
            "ett_datasets": "https://github.com/zhouhaoyi/ETDataset",
            "electricity": "https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014",
            "traffic": "http://pems.dot.ca.gov",
            "weather": "https://www.bgc-jena.mpg.de/wetter/",
            "exchange": "https://fred.stlouisfed.org/",
            "solar": "https://www.nrel.gov/grid/solar-power-data.html",
            "ili": "https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html",
            "kaggle_collection": "https://www.kaggle.com/datasets/giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets",
            "unified_benchmark": "https://github.com/juyongjiang/TimeSeriesDatasets"
        }
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """Get summary table of all benchmark datasets."""
        
        summary_data = []
        for key, dataset in self.datasets.items():
            summary_data.append({
                'Dataset': dataset['name'],
                'Domain': dataset['domain'],
                'Variables': dataset['variables'],
                'Samples': dataset['samples'] if isinstance(dataset['samples'], int) else 'Multiple',
                'Frequency': dataset['frequency'][0] if isinstance(dataset['frequency'], list) else dataset['frequency'],
                'Complexity': dataset['characteristics']['complexity'],
                'Best For': dataset['best_for'][:50] + '...' if len(dataset['best_for']) > 50 else dataset['best_for']
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_dataset_characteristics(self):
        """Visualize dataset characteristics."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Domain distribution
        domains = [ds['domain'] for ds in self.datasets.values()]
        domain_counts = pd.Series(domains).value_counts()
        axes[0, 0].pie(domain_counts.values, labels=domain_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Datasets by Domain')
        
        # Variable count distribution
        var_counts = [ds['variables'] for ds in self.datasets.values()]
        axes[0, 1].hist(var_counts, bins=10, alpha=0.7, color='skyblue')
        axes[0, 1].set_xlabel('Number of Variables')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Variable Counts')
        
        # Complexity levels
        complexities = [ds['characteristics']['complexity'] for ds in self.datasets.values()]
        complexity_counts = pd.Series(complexities).value_counts()
        axes[1, 0].bar(complexity_counts.index, complexity_counts.values, color='lightcoral')
        axes[1, 0].set_xlabel('Complexity Level')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Dataset Complexity Distribution')
        
        # Sample size ranges
        sample_sizes = []
        for ds in self.datasets.values():
            if isinstance(ds['samples'], dict):
                sample_sizes.extend(ds['samples'].values())
            else:
                sample_sizes.append(ds['samples'])
        
        axes[1, 1].hist(sample_sizes, bins=10, alpha=0.7, color='lightgreen')
        axes[1, 1].set_xlabel('Number of Samples')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Sample Sizes')
        
        plt.tight_layout()
        return fig

# Demonstrate dataset catalog
datasets = BenchmarkDatasets()

print("📊 TIME SERIES FORECASTING BENCHMARK DATASETS")
print("=" * 60)

# Show summary table
summary_df = datasets.get_dataset_summary()
print("\n📋 DATASET SUMMARY:")
print(summary_df.to_string(index=False))

# Show detailed information for key datasets
print(f"\n🔍 DETAILED DATASET INFORMATION:")

key_datasets = ['electricity_transformer_temperature', 'traffic_occupancy', 'weather_dataset']
for key in key_datasets:
    dataset = datasets.datasets[key]
    print(f"\n• {dataset['name']}:")
    print(f"  Description: {dataset['description']}")
    print(f"  Variables: {dataset['variables']}")
    print(f"  Frequency: {dataset['frequency']}")
    print(f"  Best for: {dataset['best_for']}")
    print(f"  Download: {dataset['download_source']}")

print(f"\n📥 QUICK ACCESS LINKS:")
print(f"  • Unified Collection: {datasets.download_links['unified_benchmark']}")
print(f"  • Kaggle Collection: {datasets.download_links['kaggle_collection']}")

# Visualize characteristics
datasets.plot_dataset_characteristics()
plt.savefig('dataset_characteristics.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n💡 DATASET SELECTION GUIDE:")
print("• For beginners: Start with ETT datasets (medium complexity, well-documented)")
print("• For multivariate analysis: Use Electricity or Traffic datasets")
print("• For real-time applications: Weather or Solar datasets")
print("• For financial modeling: Exchange rates dataset")
print("• For healthcare applications: ILI dataset")
