def mlp_imputation(series, gap_start, gap_end, window_size=24):
    """Use MLP to impute long gaps in time series."""
    # Prepare training data from available observations
    X_train, y_train = create_supervised_dataset(series, window_size)
    
    # Optimize MLP architecture automatically
    model = optimize_mlp_architecture(X_train, y_train)
    
    # Generate predictions for gap period
    imputed_values = []
    current_window = get_pre_gap_window(series, gap_start, window_size)
    
    for t in range(gap_start, gap_end):
        prediction = model.predict(current_window.reshape(1, -1))[0]
        imputed_values.append(prediction)
        # Update window with prediction for next step
        current_window = np.roll(current_window, -1)
        current_window[-1] = prediction
    
    return imputed_values
