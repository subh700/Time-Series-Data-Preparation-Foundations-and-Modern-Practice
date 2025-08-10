class BeginnerTutorials:
    """Progressive tutorials for time series forecasting beginners."""
    
    def tutorial_1_data_exploration(self):
        """Tutorial 1: Understanding and exploring time series data."""
        
        print("🎓 TUTORIAL 1: TIME SERIES DATA EXPLORATION")
        print("=" * 50)
        
        tutorial_content = {
            "objectives": [
                "Load and inspect time series data",
                "Identify temporal patterns",
                "Create basic visualizations",
                "Understand data characteristics"
            ],
            
            "exercises": [
                {
                    "exercise": "Load ETT dataset and create time plots",
                    "code": """
# Exercise 1: Load and visualize ETT data
import pandas as pd
import matplotlib.pyplot as plt

# Load data
url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'
df = pd.read_csv(url)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Create basic plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
df['HUFL'].plot(ax=axes[0,0], title='High UseFul Load')
df['HULL'].plot(ax=axes[0,1], title='High UseLess Load')
df['MUFL'].plot(ax=axes[1,0], title='Middle UseFul Load')
df['OT'].plot(ax=axes[1,1], title='Oil Temperature')
plt.tight_layout()
plt.show()

# Basic statistics
print("Dataset shape:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())
print("Missing values:", df.isnull().sum())
                    """,
                    "expected_outcome": "Understanding data structure and temporal patterns"
                },
                
                {
                    "exercise": "Seasonal decomposition analysis",
                    "code": """
# Exercise 2: Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Focus on one variable
series = df['OT'].dropna()

# Perform decomposition
decomposition = seasonal_decompose(series, model='additive', period=24)

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(15, 12))
series.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# Seasonal strength
seasonal_strength = decomposition.seasonal.var() / (decomposition.seasonal.var() + decomposition.resid.var())
print(f"Seasonal strength: {seasonal_strength:.3f}")
                    """,
                    "expected_outcome": "Understanding trend, seasonality, and residuals"
                }
            ],
            
            "homework": [
                "Apply the same analysis to different variables in the dataset",
                "Try different decomposition periods (12, 24, 168 hours)",
                "Compare seasonal patterns across different variables"
            ]
        }
        
        return tutorial_content
    
    def tutorial_2_basic_forecasting(self):
        """Tutorial 2: Basic forecasting methods."""
        
        tutorial_content = {
            "objectives": [
                "Implement naive forecasting methods",
                "Calculate basic error metrics",
                "Compare different simple methods",
                "Understand forecasting evaluation"
            ],
            
            "exercises": [
                {
                    "exercise": "Implement and compare naive methods",
                    "code": """
# Exercise 1: Naive forecasting methods
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Prepare data
series = df['OT'].dropna()
train_size = int(0.8 * len(series))
train, test = series[:train_size], series[train_size:]

# Method 1: Naive (last value)
naive_forecast = [train.iloc[-1]] * len(test)

# Method 2: Seasonal naive (last seasonal value)
seasonal_naive = []
for i in range(len(test)):
    seasonal_naive.append(train.iloc[-(24 - i % 24)])

# Method 3: Moving average
window = 24
ma_forecast = []
for i in range(len(test)):
    if i == 0:
        ma_forecast.append(train.iloc[-window:].mean())
    else:
        # Update with actual values as they become available
        recent_values = list(train.iloc[-window+i:]) + list(test.iloc[:i])
        ma_forecast.append(np.mean(recent_values[-window:]))

# Calculate errors
naive_mae = mean_absolute_error(test, naive_forecast)
seasonal_naive_mae = mean_absolute_error(test, seasonal_naive)
ma_mae = mean_absolute_error(test, ma_forecast)

print(f"Naive MAE: {naive_mae:.4f}")
print(f"Seasonal Naive MAE: {seasonal_naive_mae:.4f}")
print(f"Moving Average MAE: {ma_mae:.4f}")

# Plot results
plt.figure(figsize=(15, 6))
plt.plot(test.index, test.values, label='Actual', linewidth=2)
plt.plot(test.index, naive_forecast, label=f'Naive (MAE: {naive_mae:.3f})', linestyle='--')
plt.plot(test.index, seasonal_naive, label=f'Seasonal Naive (MAE: {seasonal_naive_mae:.3f})', linestyle=':')
plt.plot(test.index, ma_forecast, label=f'Moving Average (MAE: {ma_mae:.3f})', linestyle='-.')
plt.legend()
plt.title('Comparison of Basic Forecasting Methods')
plt.show()
                    """,
                    "expected_outcome": "Understanding different naive methods and their performance"
                }
            ]
        }
        
        return tutorial_content
    
    def tutorial_3_exponential_smoothing(self):
        """Tutorial 3: Exponential smoothing methods."""
        
        tutorial_content = {
            "objectives": [
                "Implement simple exponential smoothing",
                "Apply Holt's method for trend",
                "Use Holt-Winters for seasonality",
                "Parameter optimization"
            ],
            
            "exercises": [
                {
                    "exercise": "Progressive exponential smoothing implementation",
                    "code": """
# Exercise 1: Exponential Smoothing Suite
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Simple Exponential Smoothing
simple_es = ExponentialSmoothing(train, trend=None, seasonal=None)
simple_es_fit = simple_es.fit()
simple_es_forecast = simple_es_fit.forecast(len(test))

# Holt's Linear Trend
holt = ExponentialSmoothing(train, trend='add', seasonal=None)
holt_fit = holt.fit()
holt_forecast = holt_fit.forecast(len(test))

# Holt-Winters Seasonal
hw_add = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=24)
hw_add_fit = hw_add.fit()
hw_add_forecast = hw_add_fit.forecast(len(test))

hw_mult = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=24)
hw_mult_fit = hw_mult.fit()
hw_mult_forecast = hw_mult_fit.forecast(len(test))

# Calculate MAE for each method
methods = {
    'Simple ES': simple_es_forecast,
    'Holt Linear': holt_forecast,
    'HW Additive': hw_add_forecast,
    'HW Multiplicative': hw_mult_forecast
}

results = {}
for name, forecast in methods.items():
    mae = mean_absolute_error(test, forecast)
    results[name] = mae
    print(f"{name} MAE: {mae:.4f}")

# Plot comparison
plt.figure(figsize=(15, 8))
plt.plot(test.index, test.values, label='Actual', linewidth=2, color='black')

colors = ['blue', 'red', 'green', 'orange']
for i, (name, forecast) in enumerate(methods.items()):
    plt.plot(test.index, forecast, label=f'{name} (MAE: {results[name]:.3f})', 
             color=colors[i], linestyle='--', alpha=0.8)

plt.legend()
plt.title('Exponential Smoothing Methods Comparison')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Display model parameters
print(f"\\nHolt-Winters Additive Parameters:")
print(f"Alpha (level): {hw_add_fit.params['smoothing_level']:.4f}")
print(f"Beta (trend): {hw_add_fit.params['smoothing_trend']:.4f}")
print(f"Gamma (seasonal): {hw_add_fit.params['smoothing_seasonal']:.4f}")
                    """,
                    "expected_outcome": "Understanding exponential smoothing hierarchy and parameter interpretation"
                }
            ]
        }
        
        return tutorial_content

# Create tutorial instance
tutorials = BeginnerTutorials()

print("📚 BEGINNER TUTORIALS OVERVIEW")
print("=" * 50)

# Show tutorial 1 structure
tutorial_1 = tutorials.tutorial_1_data_exploration()
print(f"\n🎯 TUTORIAL 1: DATA EXPLORATION")
print("Objectives:")
for obj in tutorial_1['objectives']:
    print(f"  • {obj}")

print(f"\nExercises: {len(tutorial_1['exercises'])}")
print(f"Homework tasks: {len(tutorial_1['homework'])}")

# Show all tutorial topics
tutorial_topics = [
    "1. Time Series Data Exploration",
    "2. Basic Forecasting Methods", 
    "3. Exponential Smoothing",
    "4. ARIMA Modeling",
    "5. Model Evaluation and Selection",
    "6. Seasonal ARIMA (SARIMA)",
    "7. Multiple Time Series",
    "8. Introduction to Machine Learning Methods"
]

print(f"\n📋 COMPLETE BEGINNER TUTORIAL SEQUENCE:")
for topic in tutorial_topics:
    print(f"  {topic}")

print(f"\n⏱️ ESTIMATED TIME COMMITMENT:")
print("  • 2-3 hours per tutorial")
print("  • 1-2 tutorials per week")
print("  • Total: 4-6 weeks for completion")
