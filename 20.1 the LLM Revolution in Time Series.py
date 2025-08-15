import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch
import torch.nn as nn

class LLMTimeSeriesFramework:
    """Framework demonstrating LLM-based time series forecasting approaches."""
    
    def __init__(self):
        self.llm_approaches = self._catalog_llm_approaches()
        self.integration_strategies = self._define_integration_strategies()
    
    def _catalog_llm_approaches(self) -> Dict[str, Any]:
        """Catalog different LLM approaches for time series."""
        
        return {
            "direct_prompting": {
                "name": "Direct Prompting Approaches",
                "description": "Convert time series to text and use LLM prompts",
                "examples": ["LLMTime", "GPT-4 for forecasting"],
                "advantages": [
                    "Zero-shot capability",
                    "Leverages pre-trained knowledge", 
                    "Natural language explanations",
                    "No additional training required"
                ],
                "limitations": [
                    "Limited numerical precision",
                    "Context length constraints",
                    "Tokenization artifacts",
                    "High computational cost"
                ],
                "best_for": "Quick prototyping, interpretable forecasts",
                "example_prompt": """
You are a time series forecasting expert. Given the following temperature data:
[20.1, 20.3, 19.8, 19.5, 18.9, 18.2, 17.8, 17.5]
This represents hourly temperature readings. Based on the declining trend, 
predict the next 4 values. Consider seasonal patterns and provide reasoning.
                """
            },
            
            "fine_tuned_llms": {
                "name": "Fine-Tuned LLM Approaches", 
                "description": "Adapt pre-trained LLMs specifically for time series tasks",
                "examples": ["Time-LLM", "AutoTimes", "Logo-LLM"],
                "advantages": [
                    "Better numerical handling",
                    "Task-specific optimization",
                    "Maintains linguistic capabilities",
                    "Transfer learning benefits"
                ],
                "limitations": [
                    "Requires training data",
                    "Computational overhead",
                    "Domain adaptation challenges",
                    "Potential overfitting"
                ],
                "best_for": "Production systems, domain-specific applications",
                "architecture_components": [
                    "Temporal embedding layers",
                    "Cross-modal alignment",
                    "Specialized attention mechanisms",
                    "Numerical precision handling"
                ]
            },
            
            "hybrid_llm_architectures": {
                "name": "Hybrid LLM-Time Series Architectures",
                "description": "Combine LLMs with traditional time series models",
                "examples": ["Logo-LLM", "LLM-PS", "TimeXL"],
                "advantages": [
                    "Best of both worlds",
                    "Complementary strengths",
                    "Flexible architecture",
                    "Interpretable outputs"
                ],
                "limitations": [
                    "Architectural complexity",
                    "Training coordination",
                    "Computational requirements",
                    "Integration challenges"
                ],
                "best_for": "High-stakes applications, research exploration",
                "key_innovations": [
                    "Multi-scale feature extraction",
                    "Local-global pattern modeling",
                    "Semantic-temporal fusion",
                    "Hierarchical representations"
                ]
            }
        }
    
    def _define_integration_strategies(self) -> Dict[str, Any]:
        """Define strategies for integrating LLMs with time series."""
        
        return {
            "tokenization_strategies": {
                "numerical_tokenization": {
                    "description": "Convert numbers to discrete tokens",
                    "methods": ["Binning", "Quantization", "Scientific notation"],
                    "pros": ["Direct LLM compatibility", "Preserves sequence structure"],
                    "cons": ["Precision loss", "Vocabulary explosion"]
                },
                
                "patch_embedding": {
                    "description": "Group time points into patches for embedding",
                    "methods": ["Fixed-size patches", "Adaptive patching", "Overlapping windows"],
                    "pros": ["Efficient processing", "Captures local patterns"],
                    "cons": ["Patch boundary effects", "Resolution trade-offs"]
                },
                
                "statistical_encoding": {
                    "description": "Convert time series to statistical descriptions",
                    "methods": ["Descriptive statistics", "Trend descriptions", "Pattern summaries"],
                    "pros": ["Rich contextual information", "Natural language format"],
                    "cons": ["Information compression", "Subjective descriptions"]
                }
            },
            
            "alignment_mechanisms": {
                "cross_modal_attention": {
                    "description": "Align time series and text representations",
                    "implementation": "Multi-head attention between modalities",
                    "benefits": ["Flexible alignment", "Learnable relationships"]
                },
                
                "projection_layers": {
                    "description": "Project time series to LLM embedding space",
                    "implementation": "Linear/MLP transformation layers",
                    "benefits": ["Simple integration", "Maintains LLM structure"]
                },
                
                "adapter_modules": {
                    "description": "Lightweight adaptation layers",
                    "implementation": "Small networks inserted into LLM",
                    "benefits": ["Parameter efficient", "Preserves pre-training"]
                }
            }
        }
    
    def demonstrate_llm_forecasting_pipeline(self, time_series: np.ndarray, 
                                           context: str = "") -> Dict[str, Any]:
        """Demonstrate a complete LLM-based forecasting pipeline."""
        
        pipeline_steps = {
            "step_1_preprocessing": {
                "description": "Convert time series to LLM-compatible format",
                "input": f"Raw time series with {len(time_series)} points",
                "processing": self._convert_to_text_representation(time_series),
                "output": "Text representation of time series"
            },
            
            "step_2_contextualization": {
                "description": "Add domain context and instructions",
                "input": "Text time series + domain context",
                "processing": self._create_forecasting_prompt(time_series, context),
                "output": "Structured prompt for LLM"
            },
            
            "step_3_llm_processing": {
                "description": "Process through LLM (simulated)",
                "input": "Structured prompt",
                "processing": "LLM inference (would use actual model in practice)",
                "output": "LLM forecast response"
            },
            
            "step_4_postprocessing": {
                "description": "Extract and validate numerical forecasts",
                "input": "LLM response text",
                "processing": self._extract_numerical_forecasts(time_series),
                "output": "Validated numerical predictions"
            }
        }
        
        return pipeline_steps
    
    def _convert_to_text_representation(self, ts: np.ndarray) -> str:
        """Convert time series to text representation."""
        
        # Statistical summary
        stats = {
            'mean': np.mean(ts),
            'std': np.std(ts),
            'trend': 'increasing' if ts[-1] > ts[0] else 'decreasing',
            'volatility': 'high' if np.std(ts) > np.mean(ts) * 0.1 else 'low'
        }
        
        # Recent values
        recent_values = ', '.join([f"{val:.2f}" for val in ts[-10:]])
        
        text_repr = f"""
Time Series Summary:
- Recent 10 values: [{recent_values}]
- Mean: {stats['mean']:.2f}
- Standard deviation: {stats['std']:.2f} 
- Overall trend: {stats['trend']}
- Volatility: {stats['volatility']}
        """
        
        return text_repr.strip()
    
    def _create_forecasting_prompt(self, ts: np.ndarray, context: str) -> str:
        """Create structured prompt for forecasting."""
        
        text_repr = self._convert_to_text_representation(ts)
        
        prompt = f"""
You are an expert time series forecaster. 

{context}

{text_repr}

Based on this data, please:
1. Analyze the patterns you observe
2. Predict the next 5 values
3. Provide confidence levels for your predictions
4. Explain your reasoning

Format your response as:
Predictions: [val1, val2, val3, val4, val5]
Confidence: [conf1, conf2, conf3, conf4, conf5] (0-1 scale)
Reasoning: [Your analysis here]
        """
        
        return prompt
    
    def _extract_numerical_forecasts(self, ts: np.ndarray) -> Dict[str, Any]:
        """Extract numerical forecasts (simulated for demonstration)."""
        
        # Simulate LLM output processing
        trend = (ts[-1] - ts[-5]) / 5  # Simple trend estimation
        noise = np.random.normal(0, np.std(ts) * 0.1, 5)
        
        forecasts = [ts[-1] + trend * (i+1) + noise[i] for i in range(5)]
        confidence = [max(0.5, 1.0 - 0.1 * i) for i in range(5)]  # Decreasing confidence
        
        return {
            'predictions': forecasts,
            'confidence': confidence,
            'reasoning': f"Observed {('increasing' if trend > 0 else 'decreasing')} trend of {trend:.3f} per period"
        }
    
    def compare_llm_approaches(self) -> pd.DataFrame:
        """Compare different LLM approaches for time series."""
        
        comparison_data = []
        
        for approach_name, approach_info in self.llm_approaches.items():
            comparison_data.append({
                'Approach': approach_info['name'],
                'Training Required': 'No' if approach_name == 'direct_prompting' else 'Yes',
                'Zero-Shot Capable': 'Yes' if approach_name == 'direct_prompting' else 'Limited',
                'Numerical Precision': 'Low' if approach_name == 'direct_prompting' else 'High',
                'Interpretability': 'High',
                'Computational Cost': 'High' if approach_name == 'direct_prompting' else 'Very High',
                'Best Use Case': approach_info['best_for']
            })
        
        return pd.DataFrame(comparison_data)

# Demonstrate LLM time series framework
llm_framework = LLMTimeSeriesFramework()

print("🤖 LLM-POWERED TIME SERIES FORECASTING")
print("=" * 60)

# Show approach comparison
comparison_df = llm_framework.compare_llm_approaches()
print(f"\n📊 LLM APPROACH COMPARISON:")
print(comparison_df.to_string(index=False))

# Demonstrate pipeline
sample_ts = np.array([20.1, 20.3, 19.8, 19.5, 18.9, 18.2, 17.8, 17.5, 17.2, 16.8])
context = "This is temperature data from a weather station during autumn."

pipeline = llm_framework.demonstrate_llm_forecasting_pipeline(sample_ts, context)

print(f"\n🔄 LLM FORECASTING PIPELINE:")
for step_name, step_info in pipeline.items():
    print(f"\n{step_name.upper()}:")
    print(f"  Description: {step_info['description']}")
    print(f"  Input: {step_info['input']}")
    print(f"  Output: {step_info['output']}")

# Show key innovations
print(f"\n💡 KEY INNOVATIONS IN LLM TIME SERIES:")
hybrid_innovations = llm_framework.llm_approaches['hybrid_llm_architectures']['key_innovations']
for i, innovation in enumerate(hybrid_innovations, 1):
    print(f"{i}. {innovation}")

print(f"\n🎯 CURRENT LIMITATIONS & SOLUTIONS:")
print("• Tokenization artifacts → Advanced numerical encoding schemes")
print("• Context length limits → Hierarchical attention mechanisms") 
print("• Computational cost → Efficient fine-tuning techniques (LoRA, adapters)")
print("• Domain adaptation → Multi-domain pre-training strategies")
