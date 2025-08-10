class IntermediateTutorials:
    """Intermediate level tutorials focusing on advanced methods."""
    
    def tutorial_5_arima_deep_dive(self):
        """Advanced ARIMA modeling with diagnostics."""
        
        tutorial_content = {
            "objectives": [
                "Manual ARIMA parameter selection",
                "Model diagnostics and validation",
                "Handling non-stationarity",
                "Advanced ARIMA variants"
            ],
            
            "key_exercise": """
# Advanced ARIMA Tutorial
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Stationarity testing
def check_stationarity(series):
    # ADF Test
    adf_result = adfuller(series)
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    
    # KPSS Test  
    kpss_result = kpss(series)
    print(f"KPSS Statistic: {kpss_result[0]:.6f}")
    print(f"p-value: {kpss_result[1]:.6f}")
    
    return adf_result[1] < 0.05, kpss_result[1] > 0.05

# Step 2: Differencing if needed
series = df['OT'].dropna()
is_stationary_adf, is_stationary_kpss = check_stationarity(series)

if not (is_stationary_adf and is_stationary_kpss):
    diff_series = series.diff().dropna()
    print("\\nAfter first differencing:")
    check_stationarity(diff_series)

# Step 3: ACF/PACF analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plot_acf(series, ax=axes[0,0], title='ACF - Original')
plot_pacf(series, ax=axes[0,1], title='PACF - Original')
plot_acf(diff_series, ax=axes[1,0], title='ACF - Differenced')
plot_pacf(diff_series, ax=axes[1,1], title='PACF - Differenced')
plt.tight_layout()
plt.show()

# Step 4: Model fitting and diagnostics
model = ARIMA(train, order=(1,1,1))
fitted_model = model.fit()

print("\\nModel Summary:")
print(fitted_model.summary())

# Step 5: Residual diagnostics
residuals = fitted_model.resid
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residual plot
axes[0,0].plot(residuals)
axes[0,0].set_title('Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot')

# ACF of residuals
plot_acf(residuals, ax=axes[1,0], title='ACF of Residuals')

# Histogram
axes[1,1].hist(residuals, bins=30, alpha=0.7)
axes[1,1].set_title('Residual Distribution')

plt.tight_layout()
plt.show()

# Ljung-Box test for residual autocorrelation
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
print(f"\\nLjung-Box Test p-values:")
print(lb_test)
            """,
            
            "expected_outcome": "Master ARIMA modeling process and diagnostics"
        }
        
        return tutorial_content
    
    def tutorial_6_ml_methods(self):
        """Machine learning approaches to time series."""
        
        tutorial_content = {
            "objectives": [
                "Feature engineering for time series",
                "Applying ML algorithms to forecasting",
                "Cross-validation for time series",
                "Ensemble methods"
            ],
            
            "key_exercise": """
# ML Methods for Time Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Feature Engineering
def create_features(data, window_sizes=[1, 3, 7, 14]):
    df = data.copy()
    
    # Lag features
    for lag in range(1, 8):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # Rolling statistics
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
    
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df.dropna()

# Prepare ML dataset
ml_data = pd.DataFrame({'value': series})
feature_data = create_features(ml_data)

# Time series split
def time_series_split(data, n_splits=5):
    n = len(data)
    step = n // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = step * (i + 2)
        test_start = train_end
        test_end = min(test_start + step, n)
        
        yield (data.iloc[:train_end], data.iloc[test_start:test_end])

# Cross-validation
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

cv_results = {}
for name, model in models.items():
    scores = []
    
    for train_data, test_data in time_series_split(feature_data):
        X_train = train_data.drop('value', axis=1)
        y_train = train_data['value']
        X_test = test_data.drop('value', axis=1)
        y_test = test_data['value']
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        scores.append(mae)
    
    cv_results[name] = {
        'mean_mae': np.mean(scores),
        'std_mae': np.std(scores)
    }

print("Cross-Validation Results:")
for name, results in cv_results.items():
    print(f"{name}: MAE = {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
            """,
            
            "expected_outcome": "Apply ML techniques effectively to time series problems"
        }
        
        return tutorial_content

# Show intermediate tutorials
intermediate = IntermediateTutorials()

print(f"\n📈 INTERMEDIATE TUTORIALS (WEEKS 3-6)")
print("=" * 50)

intermediate_topics = [
    "5. Advanced ARIMA Modeling and Diagnostics",
    "6. Machine Learning Methods for Time Series", 
    "7. Deep Learning with LSTM and GRU",
    "8. Multivariate Time Series Analysis",
    "9. Handling Missing Data and Outliers",
    "10. Model Selection and Hyperparameter Tuning",
    "11. Probabilistic Forecasting",
    "12. Real-time Forecasting Systems"
]

for topic in intermediate_topics:
    print(f"  {topic}")

print(f"\n🎯 LEARNING OUTCOMES:")
print("  • Master statistical and ML approaches")
print("  • Build production-ready forecasting systems")
print("  • Understand model diagnostics and validation")
print("  • Handle complex real-world scenarios")
