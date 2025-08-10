class AdaptiveLagFeatureEngineer:
    """
    Intelligent lag feature creation with automatic selection based on
    autocorrelation analysis and business domain knowledge.
    """
    
    def __init__(self, max_lags=50, significance_threshold=0.05, domain_lags=None):
        self.max_lags = max_lags
        self.significance_threshold = significance_threshold
        self.domain_lags = domain_lags or []
        self.selected_lags = []
        
    def analyze_autocorrelation_structure(self, series, plot=False):
        """
        Analyze autocorrelation structure to identify meaningful lags.
        """
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        # Calculate ACF and PACF
        acf_values, acf_confint = acf(
            series.dropna(), 
            nlags=self.max_lags,
            alpha=self.significance_threshold,
            missing='drop'
        )
        
        pacf_values, pacf_confint = pacf(
            series.dropna(),
            nlags=self.max_lags,
            alpha=self.significance_threshold
        )
        
        # Identify significant lags
        significant_acf_lags = []
        significant_pacf_lags = []
        
        for lag in range(1, len(acf_values)):
            # Check if ACF value is outside confidence interval
            if (acf_values[lag] < acf_confint[lag, 0] or 
                acf_values[lag] > acf_confint[lag, 1]):
                significant_acf_lags.append(lag)
                
            # Check if PACF value is outside confidence interval  
            if lag < len(pacf_values):
                if (pacf_values[lag] < pacf_confint[lag, 0] or 
                    pacf_values[lag] > pacf_confint[lag, 1]):
                    significant_pacf_lags.append(lag)
        
        # Ljung-Box test for residual autocorrelation
        ljung_box_result = acorr_ljungbox(series.dropna(), lags=min(20, len(series)//5))
        
        analysis_results = {
            'acf_values': acf_values,
            'pacf_values': pacf_values,
            'significant_acf_lags': significant_acf_lags,
            'significant_pacf_lags': significant_pacf_lags,
            'ljung_box_pvalues': ljung_box_result['lb_pvalue'],
            'recommended_lags': self._recommend_lags(significant_acf_lags, significant_pacf_lags)
        }
        
        if plot:
            self._plot_autocorrelation_analysis(analysis_results)
        
        return analysis_results
    
    def _recommend_lags(self, acf_lags, pacf_lags, importance_weights=None):
        """
        Recommend lags based on statistical significance and domain knowledge.
        """
        if importance_weights is None:
            importance_weights = {'acf': 0.6, 'pacf': 0.3, 'domain': 0.1}
        
        # Combine significant lags from ACF and PACF
        all_lags = set(acf_lags + pacf_lags + self.domain_lags)
        
        # Score each lag
        lag_scores = {}
        
        for lag in all_lags:
            score = 0
            
            # ACF significance
            if lag in acf_lags:
                score += importance_weights['acf']
            
            # PACF significance
            if lag in pacf_lags:
                score += importance_weights['pacf']
            
            # Domain knowledge
            if lag in self.domain_lags:
                score += importance_weights['domain']
            
            # Penalty for very high lags (prefer parsimony)
            if lag > 30:
                score *= 0.8
            elif lag > 50:
                score *= 0.6
            
            lag_scores[lag] = score
        
        # Select top lags
        sorted_lags = sorted(lag_scores.keys(), key=lambda x: lag_scores[x], reverse=True)
        
        # Return top lags with minimum score threshold
        min_score = 0.3
        recommended_lags = [lag for lag in sorted_lags if lag_scores[lag] >= min_score]
        
        return recommended_lags[:15]  # Limit to top 15 lags
    
    def create_lag_features(self, data, target_col, timestamp_col=None, 
                          custom_lags=None, include_rolling_stats=True):
        """
        Create comprehensive lag features including rolling statistics.
        """
        
        if custom_lags is None:
            # Analyze autocorrelation to determine lags
            analysis = self.analyze_autocorrelation_structure(data[target_col])
            self.selected_lags = analysis['recommended_lags']
        else:
            self.selected_lags = custom_lags
        
        feature_df = data.copy()
        
        # Create lag features
        for lag in self.selected_lags:
            feature_df[f'{target_col}_lag_{lag}'] = feature_df[target_col].shift(lag)
        
        # Create rolling statistics if requested
        if include_rolling_stats:
            # Short-term rolling features
            for window in [3, 7, 14]:
                if window <= max(self.selected_lags):
                    feature_df[f'{target_col}_rolling_mean_{window}'] = (
                        feature_df[target_col].rolling(window=window).mean()
                    )
                    feature_df[f'{target_col}_rolling_std_{window}'] = (
                        feature_df[target_col].rolling(window=window).std()
                    )
                    feature_df[f'{target_col}_rolling_min_{window}'] = (
                        feature_df[target_col].rolling(window=window).min()
                    )
                    feature_df[f'{target_col}_rolling_max_{window}'] = (
                        feature_df[target_col].rolling(window=window).max()
                    )
        
        return {
            'features': feature_df,
            'selected_lags': self.selected_lags,
            'feature_importance': self._calculate_feature_importance(feature_df, target_col)
        }
    
    def _calculate_feature_importance(self, data, target_col):
        """Calculate importance scores for created lag features."""
        from scipy.stats import pearsonr
        
        lag_features = [col for col in data.columns if 'lag_' in col]
        importance_scores = {}
        
        for feature in lag_features:
            valid_data = data[[target_col, feature]].dropna()
            if len(valid_data) > 10:
                correlation, p_value = pearsonr(
                    valid_data[target_col], 
                    valid_data[feature]
                )
                importance_scores[feature] = {
                    'correlation': abs(correlation),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return importance_scores

