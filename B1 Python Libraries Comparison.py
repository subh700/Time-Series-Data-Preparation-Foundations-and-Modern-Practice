class LibraryComparison:
    """Comprehensive comparison of time series forecasting libraries."""
    
    def __init__(self):
        self.libraries = self._catalog_libraries()
        self.comparison_matrix = self._create_comparison_matrix()
    
    def _catalog_libraries(self) -> Dict[str, Any]:
        """Catalog major time series forecasting libraries."""
        
        return {
            "statsmodels": {
                "name": "Statsmodels",
                "github_stars": "7200+",
                "year_released": 2010,
                "primary_focus": "Statistical models",
                "strengths": [
                    "Comprehensive statistical tests",
                    "ARIMA family models",
                    "Seasonal decomposition",
                    "Well-documented theory"
                ],
                "weaknesses": [
                    "Limited ML integration",
                    "No automatic model selection",
                    "Slower for large datasets"
                ],
                "best_for": "Statistical analysis, hypothesis testing, classical methods",
                "model_types": ["ARIMA", "SARIMA", "VAR", "State Space Models"],
                "installation": "pip install statsmodels",
                "learning_curve": "Medium",
                "documentation_quality": "Excellent"
            },
            
            "sktime": {
                "name": "sktime",
                "github_stars": "5000+",
                "year_released": 2019,
                "primary_focus": "Scikit-learn compatible time series ML",
                "strengths": [
                    "Scikit-learn API compatibility",
                    "Extensive transformer library",
                    "Model composition and pipelines",
                    "Multiple forecasting tasks"
                ],
                "weaknesses": [
                    "Steep learning curve",
                    "Heavy dependency on scikit-learn",
                    "Limited deep learning support"
                ],
                "best_for": "ML practitioners, pipeline building, research",
                "model_types": ["Classical", "ML", "Ensemble", "Reduction methods"],
                "installation": "pip install sktime",
                "learning_curve": "High",
                "documentation_quality": "Good"
            },
            
            "darts": {
                "name": "Darts",
                "github_stars": "3800+",
                "year_released": 2021,
                "primary_focus": "Modern forecasting with deep learning",
                "strengths": [
                    "Modern API design",
                    "Deep learning models",
                    "Backtesting framework",
                    "Probabilistic forecasting"
                ],
                "weaknesses": [
                    "Newer library (less mature)",
                    "Limited classical methods",
                    "Resource intensive"
                ],
                "best_for": "Deep learning, modern workflows, practitioners",
                "model_types": ["ARIMA", "Deep Learning", "ML", "Ensembles"],
                "installation": "pip install darts",
                "learning_curve": "Medium",
                "documentation_quality": "Very Good"
            },
            
            "prophet": {
                "name": "Prophet (Meta)",
                "github_stars": "14000+",
                "year_released": 2017,
                "primary_focus": "Business forecasting",
                "strengths": [
                    "No parameter tuning required", 
                    "Handles holidays and seasonality",
                    "Robust to missing data",
                    "Intuitive for business users"
                ],
                "weaknesses": [
                    "Limited to single method",
                    "Not suitable for short series",
                    "Slower training"
                ],
                "best_for": "Business forecasting, non-experts, seasonal data",
                "model_types": ["Additive regression"],
                "installation": "pip install prophet",
                "learning_curve": "Low",
                "documentation_quality": "Excellent"
            },
            
            "nixtla_ecosystem": {
                "name": "Nixtla (StatsForecast, MLForecast, NeuralForecast)",
                "github_stars": "2000+ (combined)",
                "year_released": 2022,
                "primary_focus": "High-performance forecasting",
                "strengths": [
                    "Extremely fast execution",
                    "Scalable to millions of series",
                    "Comprehensive model coverage",
                    "GPU acceleration"
                ],
                "weaknesses": [
                    "Newer ecosystem",
                    "Learning multiple packages",
                    "Limited customization"
                ],
                "best_for": "Large-scale forecasting, production systems",
                "model_types": ["Statistical", "ML", "Deep Learning"],
                "installation": "pip install statsforecast mlforecast neuralforecast",
                "learning_curve": "Medium",
                "documentation_quality": "Good"
            },
            
            "autots": {
                "name": "AutoTS",
                "github_stars": "450+",
                "year_released": 2020,
                "primary_focus": "Automated time series forecasting",
                "strengths": [
                    "Full automation",
                    "Multiple models tested",
                    "Easy to use",
                    "Good for quick prototyping"
                ],
                "weaknesses": [
                    "Limited control",
                    "Black box approach",
                    "Resource intensive"
                ],
                "best_for": "Quick experiments, non-experts, baseline models",
                "model_types": ["Multiple automated"],
                "installation": "pip install autots",
                "learning_curve": "Low",
                "documentation_quality": "Fair"
            },
            
            "tensorflow_keras": {
                "name": "TensorFlow/Keras",
                "github_stars": "164000+",
                "year_released": 2015,
                "primary_focus": "Deep learning framework",
                "strengths": [
                    "Complete deep learning framework",
                    "Extensive community",
                    "Production-ready",
                    "Advanced architectures"
                ],
                "weaknesses": [
                    "Requires deep learning expertise",
                    "No built-in time series utilities",
                    "Complex for simple tasks"
                ],
                "best_for": "Custom deep learning models, research, complex problems",
                "model_types": ["LSTM", "CNN", "Transformers", "Custom architectures"],
                "installation": "pip install tensorflow",
                "learning_curve": "High",
                "documentation_quality": "Excellent"
            }
        }
    
    def _create_comparison_matrix(self) -> pd.DataFrame:
        """Create comparison matrix for easy selection."""
        
        criteria = [
            "Classical Models", "Machine Learning", "Deep Learning", 
            "Ease of Use", "Performance", "Documentation", "Community"
        ]
        
        # Ratings: 1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent
        ratings = {
            "statsmodels": [5, 2, 1, 3, 3, 5, 4],
            "sktime": [4, 5, 2, 2, 4, 4, 3],
            "darts": [3, 4, 5, 4, 4, 4, 3],
            "prophet": [1, 1, 1, 5, 3, 5, 4],
            "nixtla_ecosystem": [5, 4, 4, 3, 5, 3, 2],
            "autots": [4, 4, 2, 5, 3, 2, 2],
            "tensorflow_keras": [1, 3, 5, 2, 5, 5, 5]
        }
        
        df = pd.DataFrame(ratings, index=criteria).T
        return df
    
    def get_recommendation(self, use_case: str) -> Dict[str, str]:
        """Get library recommendation based on use case."""
        
        recommendations = {
            "beginner": {
                "primary": "prophet",
                "secondary": "autots",
                "reason": "Easy to use, requires minimal setup, good documentation"
            },
            "statistical_analysis": {
                "primary": "statsmodels", 
                "secondary": "sktime",
                "reason": "Comprehensive statistical methods and tests"
            },
            "machine_learning": {
                "primary": "sktime",
                "secondary": "darts", 
                "reason": "Excellent ML integration and pipelines"
            },
            "deep_learning": {
                "primary": "darts",
                "secondary": "tensorflow_keras",
                "reason": "Modern DL architectures with time series focus"
            },
            "production_scale": {
                "primary": "nixtla_ecosystem",
                "secondary": "darts",
                "reason": "High performance and scalability"
            },
            "research": {
                "primary": "sktime",
                "secondary": "tensorflow_keras",
                "reason": "Flexibility and extensibility for new methods"
            },
            "business_forecasting": {
                "primary": "prophet",
                "secondary": "darts",
                "reason": "Business-friendly features and interpretability"
            }
        }
        
        return recommendations.get(use_case, {
            "primary": "darts",
            "secondary": "sktime", 
            "reason": "Good balance of features and usability"
        })

# Demonstrate library comparison
lib_comp = LibraryComparison()

print("🛠️ TIME SERIES FORECASTING LIBRARIES COMPARISON")
print("=" * 60)

# Show comparison matrix
comparison_df = lib_comp.comparison_matrix
print("\n📊 LIBRARY COMPARISON MATRIX (1=Poor, 5=Excellent):")
print(comparison_df.to_string())

# Show recommendations for different use cases
print(f"\n🎯 RECOMMENDATIONS BY USE CASE:")
use_cases = ["beginner", "machine_learning", "deep_learning", "production_scale"]

for use_case in use_cases:
    rec = lib_comp.get_recommendation(use_case)
    print(f"\n• {use_case.replace('_', ' ').title()}:")
    print(f"  Primary: {rec['primary']}")
    print(f"  Secondary: {rec['secondary']}")
    print(f"  Reason: {rec['reason']}")

# Show installation commands
print(f"\n💻 QUICK INSTALLATION COMMANDS:")
key_libs = ['darts', 'sktime', 'prophet', 'statsmodels']
for lib_key in key_libs:
    if lib_key in lib_comp.libraries:
        lib_info = lib_comp.libraries[lib_key]
        print(f"  • {lib_info['name']}: {lib_info['installation']}")

print(f"\n⭐ GITHUB POPULARITY (Stars):")
for lib_key, lib_info in lib_comp.libraries.items():
    print(f"  • {lib_info['name']}: {lib_info['github_stars']}")
