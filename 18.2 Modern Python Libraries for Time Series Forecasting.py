import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesLibraryComparison:
    """
    Comprehensive comparison of Python libraries for time series forecasting.
    Updated for 2024-2025 ecosystem.
    """
    
    def __init__(self):
        self.libraries = self._initialize_library_info()
        self.comparison_results = {}
    
    def _initialize_library_info(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive library information."""
        
        return {
            # Statistical Libraries
            "statsmodels": {
                "category": "Statistical",
                "description": "Classical statistical models (ARIMA, SARIMAX, VAR)",
                "strengths": [
                    "Comprehensive statistical tests",
                    "Well-documented theory",
                    "Extensive model diagnostics",
                    "Mature and stable"
                ],
                "weaknesses": [
                    "Limited deep learning support",
                    "Manual parameter tuning",
                    "Slower for large datasets"
                ],
                "use_cases": [
                    "Classical time series analysis",
                    "Statistical inference",
                    "Academic research",
                    "Model interpretability"
                ],
                "installation": "pip install statsmodels",
                "learning_curve": "Medium",
                "community_support": "High",
                "last_updated": "2024"
            },
            
            # Modern Forecasting Libraries  
            "sktime": {
                "category": "Modern ML",
                "description": "Unified framework for ML time series tasks",
                "strengths": [
                    "Scikit-learn compatible API",
                    "Wide range of algorithms",
                    "Good pipeline support",
                    "Active development"
                ],
                "weaknesses": [
                    "Complex for beginners",
                    "Documentation gaps",
                    "Limited deep learning"
                ],
                "use_cases": [
                    "ML-based forecasting",
                    "Time series classification",
                    "Feature extraction",
                    "Pipeline development"
                ],
                "installation": "pip install sktime",
                "learning_curve": "High",
                "community_support": "Medium",
                "last_updated": "2024"
            },
            
            "darts": {
                "category": "Modern ML",
                "description": "Easy-to-use forecasting library with ML/DL support",
                "strengths": [
                    "User-friendly API",
                    "Support for multivariate series",
                    "Built-in backtesting",
                    "Deep learning integration"
                ],
                "weaknesses": [
                    "Limited statistical models",
                    "Newer library (less mature)",
                    "Memory intensive"
                ],
                "use_cases": [
                    "Deep learning forecasting",
                    "Multivariate forecasting",
                    "Rapid prototyping",
                    "Production systems"
                ],
                "installation": "pip install darts",
                "learning_curve": "Low",
                "community_support": "Medium",
                "last_updated": "2024"
            },
            
            "nixtla": {
                "category": "Modern ML",
                "description": "Fast statistical and ML forecasting tools",
                "strengths": [
                    "Very fast performance",
                    "Automatic model selection",
                    "Good for large datasets",
                    "Cloud-ready"
                ],
                "weaknesses": [
                    "Limited customization",
                    "Newer ecosystem",
                    "Less documentation"
                ],
                "use_cases": [
                    "Large-scale forecasting",
                    "Automatic model selection",
                    "Cloud deployments",
                    "Real-time predictions"
                ],
                "installation": "pip install statsforecast mlforecast",
                "learning_curve": "Low",
                "community_support": "Growing",
                "last_updated": "2024"
            },
            
            # Prophet-based Libraries
            "prophet": {
                "category": "Specialized",
                "description": "Facebook's decomposable time series model",
                "strengths": [
                    "Handles seasonality well",
                    "Robust to missing data",
                    "Easy to use",
                    "Good for business data"
                ],
                "weaknesses": [
                    "No longer actively maintained",
                    "Limited to one algorithm",
                    "Not suitable for all data types"
                ],
                "use_cases": [
                    "Business forecasting",
                    "Daily/weekly data",
                    "Strong seasonal patterns",
                    "Non-experts"
                ],
                "installation": "pip install prophet",
                "learning_curve": "Low",
                "community_support": "Low (deprecated)",
                "last_updated": "2023 (deprecated)"
            },
            
            "neuralprophet": {
                "category": "Deep Learning",
                "description": "Neural network version of Prophet",
                "strengths": [
                    "Improved Prophet with deep learning",
                    "Better accuracy",
                    "Active development",
                    "Similar API to Prophet"
                ],
                "weaknesses": [
                    "Requires more data",
                    "Longer training time",
                    "Less interpretable"
                ],
                "use_cases": [
                    "Large datasets",
                    "Complex patterns",
                    "Prophet users seeking upgrades",
                    "Seasonal forecasting"
                ],
                "installation": "pip install neuralprophet",
                "learning_curve": "Medium",
                "community_support": "Medium",
                "last_updated": "2024"
            },
            
            # Deep Learning Libraries
            "pytorch_forecasting": {
                "category": "Deep Learning",
                "description": "PyTorch-based deep learning for forecasting",
                "strengths": [
                    "State-of-the-art models",
                    "GPU acceleration",
                    "Flexible architecture",
                    "Research-oriented"
                ],
                "weaknesses": [
                    "Steep learning curve",
                    "Complex setup",
                    "Requires deep learning expertise"
                ],
                "use_cases": [
                    "Research projects",
                    "Complex multivariate forecasting",
                    "Large datasets",
                    "Custom architectures"
                ],
                "installation": "pip install pytorch-forecasting",
                "learning_curve": "High",
                "community_support": "Medium",
                "last_updated": "2024"
            },
            
            "tensorflow": {
                "category": "Deep Learning",
                "description": "Google's deep learning framework",
                "strengths": [
                    "Extensive ecosystem",
                    "Production-ready",
                    "TensorBoard integration",
                    "Mobile deployment"
                ],
                "weaknesses": [
                    "Complex for simple tasks",
                    "Steep learning curve",
                    "Verbose code"
                ],
                "use_cases": [
                    "Large-scale production",
                    "Complex neural networks",
                    "Multi-platform deployment",
                    "Research and development"
                ],
                "installation": "pip install tensorflow",
                "learning_curve": "High",
                "community_support": "High",
                "last_updated": "2024"
            },
            
            # Specialized Libraries
            "gluonts": {
                "category": "Deep Learning",
                "description": "Amazon's probabilistic time series toolkit",
                "strengths": [
                    "Probabilistic forecasting",
                    "Pre-trained models",
                    "AWS integration",
                    "Research-backed"
                ],
                "weaknesses": [
                    "Complex setup",
                    "Limited documentation",
                    "AWS-centric"
                ],
                "use_cases": [
                    "Probabilistic forecasting",
                    "AWS deployments",
                    "Research projects",
                    "Uncertainty quantification"
                ],
                "installation": "pip install gluonts",
                "learning_curve": "High",
                "community_support": "Medium",
                "last_updated": "2024"
            },
            
            "pyflux": {
                "category": "Statistical",
                "description": "Modern Bayesian time series library",
                "strengths": [
                    "Bayesian inference",
                    "Modern statistical models",
                    "Uncertainty quantification",
                    "Clean API"
                ],
                "weaknesses": [
                    "Limited development",
                    "Smaller community",
                    "Documentation issues"
                ],
                "use_cases": [
                    "Bayesian modeling",
                    "Uncertainty quantification",
                    "Academic research",
                    "Small to medium datasets"
                ],
                "installation": "pip install pyflux",
                "learning_curve": "Medium",
                "community_support": "Low",
                "last_updated": "2022"
            }
        }
    
    def generate_comparison_matrix(self) -> pd.DataFrame:
        """Generate comprehensive comparison matrix."""
        
        comparison_data = []
        
        for lib_name, lib_info in self.libraries.items():
            comparison_data.append({
                'Library': lib_name,
                'Category': lib_info['category'],
                'Learning Curve': lib_info['learning_curve'],
                'Community Support': lib_info['community_support'],
                'Last Updated': lib_info['last_updated'],
                'Main Strengths': ', '.join(lib_info['strengths'][:2]),
                'Best For': ', '.join(lib_info['use_cases'][:2])
            })
        
        return pd.DataFrame(comparison_data)
    
    def recommend_library(self, 
                         use_case: str,
                         experience_level: str,
                         dataset_size: str,
                         complexity: str) -> Dict[str, Any]:
        """
        Recommend best library based on requirements.
        
        Args:
            use_case: Type of forecasting task
            experience_level: beginner, intermediate, advanced
            dataset_size: small, medium, large
            complexity: simple, moderate, complex
        """
        
        recommendations = {
            'primary': None,
            'alternatives': [],
            'reasoning': "",
            'setup_guide': ""
        }
        
        # Decision logic
        if experience_level == "beginner":
            if use_case in ["business_forecasting", "seasonal_data"]:
                if self.libraries["prophet"]["last_updated"] == "2023 (deprecated)":
                    recommendations['primary'] = "neuralprophet"
                    recommendations['reasoning'] = "NeuralProphet is the successor to Prophet with better accuracy"
                else:
                    recommendations['primary'] = "prophet"
                recommendations['alternatives'] = ["darts", "nixtla"]
            else:
                recommendations['primary'] = "darts"
                recommendations['alternatives'] = ["nixtla", "statsmodels"]
                recommendations['reasoning'] = "Darts offers user-friendly API with good performance"
        
        elif experience_level == "intermediate":
            if complexity == "complex" or dataset_size == "large":
                recommendations['primary'] = "sktime"
                recommendations['alternatives'] = ["darts", "nixtla"]
                recommendations['reasoning'] = "sktime provides comprehensive ML toolkit with sklearn compatibility"
            else:
                recommendations['primary'] = "darts"
                recommendations['alternatives'] = ["statsmodels", "sktime"]
        
        else:  # advanced
            if use_case == "research" or complexity == "complex":
                recommendations['primary'] = "pytorch_forecasting"
                recommendations['alternatives'] = ["gluonts", "tensorflow"]
                recommendations['reasoning'] = "PyTorch Forecasting offers state-of-the-art models for research"
            elif dataset_size == "large":
                recommendations['primary'] = "nixtla"
                recommendations['alternatives'] = ["darts", "pytorch_forecasting"]
                recommendations['reasoning'] = "Nixtla provides fast performance for large-scale forecasting"
            else:
                recommendations['primary'] = "sktime"
                recommendations['alternatives'] = ["statsmodels", "darts"]
        
        # Add setup guide for primary recommendation
        if recommendations['primary']:
            lib_info = self.libraries[recommendations['primary']]
            recommendations['setup_guide'] = f"""
Setup Guide for {recommendations['primary']}:

1. Installation:
   {lib_info['installation']}

2. Key Strengths:
   {', '.join(lib_info['strengths'])}

3. Best Use Cases:
   {', '.join(lib_info['use_cases'])}

4. Learning Resources:
   - Official documentation
   - Community tutorials
   - Example notebooks
            """
        
        return recommendations
    
    def create_selection_flowchart(self):
        """Create visual library selection flowchart."""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create flowchart logic
        flowchart_text = """
Library Selection Flowchart

Start
  │
  ├─ Experience Level?
  │   │
  │   ├─ Beginner → Darts or NeuralProphet
  │   │
  │   ├─ Intermediate → sktime or Darts
  │   │
  │   └─ Advanced → PyTorch Forecasting or Nixtla
  │
  ├─ Dataset Size?
  │   │
  │   ├─ Small (<10k) → statsmodels or Prophet
  │   │
  │   ├─ Medium (10k-1M) → Darts or sktime
  │   │
  │   └─ Large (>1M) → Nixtla or PyTorch Forecasting
  │
  └─ Use Case?
      │
      ├─ Research → PyTorch Forecasting or GluonTS
      │
      ├─ Production → Darts or Nixtla
      │
      └─ Education → statsmodels or Prophet
        """
        
        ax.text(0.05, 0.95, flowchart_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Time Series Library Selection Guide', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('reports/library_selection_guide.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def benchmark_libraries(self, sample_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different libraries on sample data.
        
        Args:
            sample_data: Sample time series data
            
        Returns:
            Benchmark results
        """
        
        results = {}
        
        # This would implement actual benchmarking
        # For demonstration, we'll return mock results
        mock_results = {
            'statsmodels': {'accuracy': 0.85, 'speed': 0.7, 'memory': 0.9},
            'darts': {'accuracy': 0.88, 'speed': 0.8, 'memory': 0.7},
            'sktime': {'accuracy': 0.87, 'speed': 0.6, 'memory': 0.8},
            'nixtla': {'accuracy': 0.86, 'speed': 0.95, 'memory': 0.9},
            'neuralprophet': {'accuracy': 0.89, 'speed': 0.5, 'memory': 0.6}
        }
        
        return mock_results
    
    def generate_setup_scripts(self, libraries: List[str]) -> Dict[str, str]:
        """Generate setup scripts for selected libraries."""
        
        setup_scripts = {}
        
        for lib in libraries:
            if lib in self.libraries:
                lib_info = self.libraries[lib]
                
                script = f"""#!/bin/bash
# Setup script for {lib}

echo "Setting up {lib} for time series forecasting..."

# Create virtual environment
python -m venv {lib}_env
source {lib}_env/bin/activate  # On Windows: {lib}_env\\Scripts\\activate

# Install {lib}
{lib_info['installation']}

# Install common dependencies
pip install pandas numpy matplotlib seaborn jupyter

# Verify installation
python -c "import {lib}; print('{lib} installed successfully')"

echo "{lib} setup completed!"
"""
                setup_scripts[lib] = script
        
        return setup_scripts


# Tool Selection Advisor
class ToolSelectionAdvisor:
    """Interactive tool selection advisor."""
    
    def __init__(self):
        self.comparison = TimeSeriesLibraryComparison()
    
    def interactive_recommendation(self):
        """Interactive library recommendation system."""
        
        print("🔍 Time Series Library Recommendation System")
        print("=" * 50)
        
        # Collect requirements
        print("\nPlease answer the following questions:")
        
        experience = input("Experience level (beginner/intermediate/advanced): ").strip().lower()
        if experience not in ['beginner', 'intermediate', 'advanced']:
            experience = 'intermediate'
        
        use_case = input("Primary use case (research/production/education/business): ").strip().lower()
        
        dataset_size = input("Dataset size (small/medium/large): ").strip().lower()
        if dataset_size not in ['small', 'medium', 'large']:
            dataset_size = 'medium'
        
        complexity = input("Problem complexity (simple/moderate/complex): ").strip().lower()
        if complexity not in ['simple', 'moderate', 'complex']:
            complexity = 'moderate'
        
        # Get recommendation
        recommendation = self.comparison.recommend_library(
            use_case, experience, dataset_size, complexity
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("📋 RECOMMENDATION RESULTS")
        print("=" * 50)
        
        print(f"\n🎯 Primary Recommendation: {recommendation['primary']}")
        print(f"🔄 Alternatives: {', '.join(recommendation['alternatives'])}")
        print(f"\n💡 Reasoning: {recommendation['reasoning']}")
        
        print(f"\n{recommendation['setup_guide']}")
        
        return recommendation
    
    def create_comparison_report(self):
        """Create comprehensive comparison report."""
        
        comparison_df = self.comparison.generate_comparison_matrix()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Category distribution
        category_counts = comparison_df['Category'].value_counts()
        axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Library Categories')
        
        # Learning curve distribution
        learning_curve_counts = comparison_df['Learning Curve'].value_counts()
        axes[0, 1].bar(learning_curve_counts.index, learning_curve_counts.values)
        axes[0, 1].set_title('Learning Curve Distribution')
        axes[0, 1].set_xlabel('Learning Curve')
        axes[0, 1].set_ylabel('Number of Libraries')
        
        # Community support
        support_counts = comparison_df['Community Support'].value_counts()
        axes[1, 0].bar(support_counts.index, support_counts.values)
        axes[1, 0].set_title('Community Support Levels')
        axes[1, 0].set_xlabel('Support Level')
        axes[1, 0].set_ylabel('Number of Libraries')
        
        # Timeline
        update_counts = comparison_df['Last Updated'].value_counts()
        axes[1, 1].bar(update_counts.index, update_counts.values)
        axes[1, 1].set_title('Last Update Distribution')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Number of Libraries')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/library_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save comparison table
        comparison_df.to_csv('reports/library_comparison.csv', index=False)
        
        print("📊 Comparison report saved to 'reports/' directory")
        
        return comparison_df


# Example usage
def demonstrate_tool_selection():
    """Demonstrate tool selection process."""
    
    print("Demonstrating Time Series Library Selection")
    print("=" * 50)
    
    # Initialize advisor
    advisor = ToolSelectionAdvisor()
    
    # Create comparison report
    comparison_df = advisor.create_comparison_report()
    print(f"\nComparison completed for {len(comparison_df)} libraries")
    
    # Example recommendations for different scenarios
    scenarios = [
        {
            'name': 'Beginner Business Forecasting',
            'use_case': 'business',
            'experience': 'beginner',
            'dataset_size': 'small',
            'complexity': 'simple'
        },
        {
            'name': 'Advanced Research Project',
            'use_case': 'research',
            'experience': 'advanced',
            'dataset_size': 'large',
            'complexity': 'complex'
        },
        {
            'name': 'Production System',
            'use_case': 'production',
            'experience': 'intermediate',
            'dataset_size': 'medium',
            'complexity': 'moderate'
        }
    ]
    
    print("\n" + "=" * 50)
    print("SCENARIO-BASED RECOMMENDATIONS")
    print("=" * 50)
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        
        recommendation = advisor.comparison.recommend_library(
            scenario['use_case'],
            scenario['experience'],
            scenario['dataset_size'],
            scenario['complexity']
        )
        
        print(f"   🎯 Recommended: {recommendation['primary']}")
        print(f"   🔄 Alternatives: {', '.join(recommendation['alternatives'])}")
        print(f"   💡 Reason: {recommendation['reasoning']}")
    
    # Generate selection flowchart
    advisor.comparison.create_selection_flowchart()
    
    return advisor


if __name__ == "__main__":
    # Run tool selection demonstration
    advisor = demonstrate_tool_selection()
    
    # Optional: Run interactive recommendation
    print("\n" + "=" * 50)
    response = input("Would you like an interactive recommendation? (y/n): ")
    if response.lower() == 'y':
        advisor.interactive_recommendation()
