# Template 1: Quick Prophet Forecast
def quick_prophet_forecast(df, target_col, periods=30):
    """Quick forecasting with Prophet."""
    from prophet import Prophet
    
    # Prepare data
    prophet_df = df.reset_index()
    prophet_df = prophet_df.rename(columns={'timestamp': 'ds', target_col: 'y'})
    
    # Fit and predict
    model = Prophet()
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return model, forecast

# Template 2: ARIMA with auto parameter selection
def auto_arima_forecast(series, forecast_periods=30):
    """Auto ARIMA forecasting."""
    from pmdarima import auto_arima
    
    model = auto_arima(series, seasonal=True, stepwise=True)
    forecast = model.predict(n_periods=forecast_periods)
    
    return model, forecast

# Template 3: Simple LSTM with Keras
def simple_lstm_forecast(data, window_size=60, forecast_periods=30):
    """Simple LSTM forecasting template."""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    return model, scaler

# Template 4: Darts comprehensive workflow
def darts_workflow(series, model_type='ARIMA'):
    """Complete Darts workflow."""
    from darts import TimeSeries
    from darts.models import ARIMA, ExponentialSmoothing, Prophet
    from darts.metrics import mape
    
    # Convert to Darts TimeSeries
    ts = TimeSeries.from_pandas(series)
    
    # Split data
    train, val = ts.split_before(0.8)
    
    # Select model
    if model_type == 'ARIMA':
        model = ARIMA()
    elif model_type == 'ETS':
        model = ExponentialSmoothing()
    elif model_type == 'Prophet':
        model = Prophet()
    
    # Fit and predict
    model.fit(train)
    forecast = model.predict(len(val))
    
    # Evaluate
    error = mape(val, forecast)
    
    return model, forecast, error
