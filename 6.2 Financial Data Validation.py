class FinancialDataValidator:
    """Comprehensive validation for financial time series data."""
    
    def __init__(self, instrument_type='stock'):
        self.instrument_type = instrument_type
        self.validation_rules = self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize validation rules based on instrument type."""
        base_rules = {
            'price_positive': True,
            'volume_non_negative': True,
            'reasonable_price_changes': True,
            'market_hours_only': False  # Can be enabled for specific markets
        }
        
        if self.instrument_type == 'stock':
            base_rules.update({
                'max_daily_change': 0.20,  # 20% maximum daily change
                'min_price': 0.01,  # Minimum price in dollars
                'max_price': 10000,  # Maximum reasonable stock price
            })
        elif self.instrument_type == 'forex':
            base_rules.update({
                'max_daily_change': 0.10,  # 10% maximum daily change
                'min_price': 0.0001,  # Minimum forex rate
                'max_price': 1000,  # Maximum reasonable forex rate
            })
        
        return base_rules
    
    def validate_data(self, data, price_col='close', volume_col='volume'):
        """Perform comprehensive validation of financial data."""
        violations = {
            'negative_prices': [],
            'negative_volumes': [],
            'excessive_price_changes': [],
            'impossible_values': [],
            'market_hours_violations': []
        }
        
        # Price validation
        negative_price_mask = data[price_col] <= 0
        violations['negative_prices'] = data[negative_price_mask].index.tolist()
        
        # Volume validation (if available)
        if volume_col in data.columns:
            negative_volume_mask = data[volume_col] < 0
            violations['negative_volumes'] = data[negative_volume_mask].index.tolist()
        
        # Price change validation
        price_changes = data[price_col].pct_change()
        excessive_change_mask = np.abs(price_changes) > self.validation_rules['max_daily_change']
        violations['excessive_price_changes'] = data[excessive_change_mask].index.tolist()
        
        # Impossible value ranges
        impossible_mask = (
            (data[price_col] < self.validation_rules['min_price']) |
            (data[price_col] > self.validation_rules['max_price'])
        )
        violations['impossible_values'] = data[impossible_mask].index.tolist()
        
        return violations
    
    def clean_violations(self, data, violations, strategy='conservative'):
        """Clean identified violations based on strategy."""
        cleaned_data = data.copy()
        cleaning_log = []
        
        if strategy == 'conservative':
            # Remove rows with negative prices or volumes
            invalid_rows = set()
            invalid_rows.update(violations['negative_prices'])
            invalid_rows.update(violations['negative_volumes'])
            invalid_rows.update(violations['impossible_values'])
            
            if invalid_rows:
                cleaned_data = cleaned_data.drop(index=list(invalid_rows))
                cleaning_log.append(f"Removed {len(invalid_rows)} rows with invalid values")
            
            # Cap excessive price changes
            if violations['excessive_price_changes']:
                for idx in violations['excessive_price_changes']:
                    if idx in cleaned_data.index:
                        # Replace with interpolated value
                        interpolated_value = self._interpolate_price(cleaned_data, idx)
                        cleaned_data.loc[idx, 'close'] = interpolated_value
                
                cleaning_log.append(f"Capped {len(violations['excessive_price_changes'])} excessive price changes")
        
        return cleaned_data, cleaning_log
    
    def _interpolate_price(self, data, problem_index, window=5):
        """Interpolate price for problematic observations."""
        # Get surrounding valid observations
        start_idx = max(0, data.index.get_loc(problem_index) - window)
        end_idx = min(len(data), data.index.get_loc(problem_index) + window + 1)
        
        surrounding_data = data.iloc[start_idx:end_idx]
        valid_data = surrounding_data[surrounding_data.index != problem_index]
        
        if len(valid_data) >= 2:
            # Linear interpolation based on time
            interp_value = valid_data['close'].interpolate(method='time').loc[problem_index]
            return interp_value
        else:
            # Fallback to previous valid value
            return data['close'].loc[:problem_index].dropna().iloc[-1]
