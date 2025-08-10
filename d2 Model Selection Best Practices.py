class ModelSelectionFramework:
    """Framework for systematic model selection in time series forecasting."""
    
    def __init__(self):
        self.selection_criteria = self._define_selection_criteria()
        self.validation_strategies = self._define_validation_strategies()
    
    def _define_selection_criteria(self) -> Dict[str, Any]:
        """Define comprehensive model selection criteria."""
        
        return {
            "performance_metrics": {
                "accuracy": ["MAE", "RMSE", "MAPE", "MASE"],
                "probabilistic": ["CRPS", "Quantile Loss", "Coverage"],
                "business_specific": ["Directional Accuracy", "Custom Loss Functions"]
            },
            
            "practical_considerations": {
                "interpretability": {
                    "high": ["Linear models", "ARIMA", "Exponential Smoothing"],
                    "medium": ["Tree-based models", "Prophet"],
                    "low": ["Neural networks", "Complex ensembles"]
                },
                
                "computational_requirements": {
                    "training_time": "Consider for real-time applications",
                    "inference_time": "Critical for high-frequency forecasting",
                    "memory_usage": "Important for resource-constrained environments",
                    "scalability": "Essential for large-scale deployments"
                },
                
                "maintenance_complexity": {
                    "low": ["Simple statistical models", "Prophet"],
                    "medium": ["ML models with automated pipelines"],
                    "high": ["Custom deep learning architectures"]
                }
            },
            
            "robustness_factors": {
                "stability": "Performance consistency across different periods",
                "generalization": "Performance on out-of-sample data",
                "resilience": "Handling of outliers and anomalies",
                "adaptability": "Ability to handle distribution shifts"
            }
        }
    
    def _define_validation_strategies(self) -> Dict[str, Any]:
        """Define time series specific validation strategies."""
        
        return {
            "temporal_split": {
                "description": "Standard train/validation/test split respecting temporal order",
                "advantages": ["Simple", "Mimics real deployment"],
                "disadvantages": ["Single point in time", "May not be representative"],
                "when_to_use": "Stable time series with consistent patterns",
                "implementation": """
# Temporal split example
def temporal_split(data, train_ratio=0.6, val_ratio=0.2):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return data[:train_end], data[train_end:val_end], data[val_end:]
                """
            },
            
            "rolling_window": {
                "description": "Rolling window cross-validation with expanding or fixed window",
                "advantages": ["Multiple evaluation points", "Robust performance estimate"],
                "disadvantages": ["Computationally expensive", "Overlapping training sets"],
                "when_to_use": "When you need robust performance estimates",
                "implementation": """
# Rolling window validation
def rolling_window_validation(data, initial_window, horizon, step_size=1):
    results = []
    
    for i in range(initial_window, len(data) - horizon + 1, step_size):
        train = data[:i]
        test = data[i:i+horizon]
        
        # Fit model on train, evaluate on test
        # results.append(evaluate_model(train, test))
    
    return results
                """
            },
            
            "blocked_cross_validation": {
                "description": "Block-based CV to reduce temporal dependency",
                "advantages": ["Reduces overfitting", "Better for dependent data"],
                "disadvantages": ["Complex to implement", "May not reflect real usage"],
                "when_to_use": "Highly autocorrelated time series",
                "implementation": """
# Blocked CV with gaps
def blocked_cv(data, n_splits=5, test_size=0.2, gap=0.1):
    n = len(data)
    block_size = n // n_splits
    
    for i in range(n_splits):
        test_start = i * block_size
        test_end = test_start + int(block_size * test_size)
        gap_end = test_end + int(block_size * gap)
        
        train = data[:test_start] + data[gap_end:]
        test = data[test_start:test_end]
        
        yield train, test
                """
            }
        }
    
    def create_selection_matrix(self, models: List[str], criteria: List[str]) -> pd.DataFrame:
        """Create model selection matrix for systematic comparison."""
        
        # Sample scoring (in practice, would be based on actual evaluation)
        np.random.seed(42)
        scores = np.random.rand(len(models), len(criteria)) * 5
        
        df = pd.DataFrame(scores, index=models, columns=criteria)
        
        # Add weighted score (example weights)
        weights = {
            'Accuracy': 0.3,
            'Interpretability': 0.2,
            'Speed': 0.2,
            'Robustness': 0.15,
            'Maintainability': 0.15
        }
        
        if all(criterion in weights for criterion in criteria):
            df['Weighted Score'] = sum(df[criterion] * weights[criterion] for criterion in criteria)
        
        return df.round(3)
    
    def generate_selection_report(self, model_results: Dict[str, Any]) -> str:
        """Generate comprehensive model selection report."""
        
        report = """
# MODEL SELECTION REPORT

## Executive Summary
Based on comprehensive evaluation across multiple criteria, the following model selection is recommended:

**Recommended Model**: {best_model}
**Primary Reason**: {primary_reason}
**Secondary Considerations**: {secondary_considerations}

## Evaluation Criteria

### Performance Metrics
- Accuracy: {accuracy_score}
- Robustness: {robustness_score}
- Interpretability: {interpretability_score}

### Practical Considerations
- Training Time: {training_time}
- Inference Speed: {inference_speed}
- Memory Requirements: {memory_requirements}
- Maintenance Complexity: {maintenance_complexity}

## Risk Assessment
- Model Complexity Risk: {complexity_risk}
- Overfitting Risk: {overfitting_risk}
- Deployment Risk: {deployment_risk}

## Recommendations
1. **Immediate Action**: Deploy recommended model with monitoring
2. **Short-term**: Establish baseline performance metrics
3. **Medium-term**: Implement A/B testing framework
4. **Long-term**: Consider ensemble approaches

## Monitoring Plan
- Key metrics to track: {monitoring_metrics}
- Alert thresholds: {alert_thresholds}
- Review frequency: {review_frequency}
        """.format(**model_results)
        
        return report

# Demonstrate model selection framework
selection_framework = ModelSelectionFramework()

print("🎯 MODEL SELECTION FRAMEWORK")
print("=" * 50)

# Show selection criteria
criteria = selection_framework.selection_criteria
print("\n📊 SELECTION CRITERIA:")
print(f"Performance Metrics: {', '.join(criteria['performance_metrics']['accuracy'])}")
print(f"Practical Considerations: {len(criteria['practical_considerations'])} categories")
print(f"Robustness Factors: {len(criteria['robustness_factors'])} factors")

# Show validation strategies
print(f"\n🔄 VALIDATION STRATEGIES:")
for strategy, details in selection_framework.validation_strategies.items():
    print(f"\n• {strategy.replace('_', ' ').title()}")
    print(f"  Description: {details['description']}")
    print(f"  When to use: {details['when_to_use']}")

# Create example selection matrix
models = ['ARIMA', 'Prophet', 'XGBoost', 'LSTM', 'Ensemble']
criteria_names = ['Accuracy', 'Interpretability', 'Speed', 'Robustness', 'Maintainability']

selection_matrix = selection_framework.create_selection_matrix(models, criteria_names)
print(f"\n📋 MODEL SELECTION MATRIX (1-5 scale, higher is better):")
print(selection_matrix.to_string())

# Show top performer
if 'Weighted Score' in selection_matrix.columns:
    best_model = selection_matrix['Weighted Score'].idxmax()
    best_score = selection_matrix.loc[best_model, 'Weighted Score']
    print(f"\n🏆 TOP PERFORMER: {best_model} (Score: {best_score:.3f})")

print(f"\n💡 SELECTION BEST PRACTICES:")
print("1. Define success criteria before model comparison")
print("2. Use appropriate validation strategy for your data")
print("3. Consider practical constraints alongside accuracy")
print("4. Document decision rationale for future reference")
print("5. Plan for model monitoring and updates")
